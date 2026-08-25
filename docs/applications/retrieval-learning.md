# 检索表示学习：从 InfoNCE 到 ColBERT 与 SPLADE

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：希望训练、诊断或评测检索器的算法与 RAG 工程师。
- **先修**：[机器学习基础](../foundations/ml-dl.md)、[RAG 检索](rag-retrieval.md)。
- **首次阅读**：打分架构 → InfoNCE → negatives → DPR/ColBERT/SPLADE → 分层评测。
- **完成信号**：能推导目标梯度，并分开诊断 model、ANN 与 reranker recall。
- **卡住时**：先运行本章可手算的小例子，观察候选 mask 如何改变梯度。

</div>

RAG 不只是在向量库里调用 `search()`。Dense retriever 的相似度来自训练目标、候选集合、正负标签、pooling 和索引近似的共同作用。本章沿着一条可验证主线回答：**query 与 document 为什么会在向量空间里靠近，失败又应归因到哪一层？**

学完后应能：

- 区分 bi-encoder、cross-encoder 与 late interaction 的信息交互和成本；
- 写出单正例与多正例 InfoNCE，并解释梯度、temperature 和 candidate set；
- 判断 in-batch negative、hard negative 与 false negative 分别改变了什么；
- 说明 DPR、ColBERT 与 SPLADE 复用了哪些机制、牺牲了什么；
- 分开测量表示模型、exact search、ANN 和 reranker 的误差；
- 运行一个 NumPy 对照示例，并区分公式检查与真实 retriever 质量评测。

## 先认识贯穿本页的四个候选

沿用 [RAG 请求 A](rag-request-lifecycle.md)：“为什么要先做 ACL 权限过滤？”把候选文档抽象成四类：

| 候选 | 内容与标签 | 训练时希望发生什么 |
|---|---|---|
| (d^+) | 直接说明“先授权，再检索和重排”；已标正例 | 分数升高 |
| (d_{easy}) | 讨论 SFT 数据，与问题无关 | 分数降低，通常很容易区分 |
| (d_{hard}) | 同样频繁出现 RAG、ACL 和重排，但没有回答顺序 | 学会区分“词很像”和“真正回答问题” |
| (d_{para}) | 用另一种说法给出同一正确答案，却漏掉正例标签 | 本应升高，单正例训练却会错误压低 |

本页的 NumPy 实验用二维向量表示这四种关系。它没有把真实文本送入 encoder，但能让你直接观察：候选集合和标签
怎样改变 loss 与梯度。读每个方法时，都问一句：它会怎样给这四个候选打分？

## 1. 先把检索拆成四层

一次 dense retrieval 至少包含四个不同对象：

1. **表示模型**：把 query 和 document 变成向量或 token/vector 集合；
2. **打分函数**：点积、cosine、MaxSim 或 cross-encoder score；
3. **候选域**：哪些文档参与训练分母，线上又有哪些文档经过授权后可见；
4. **搜索算法**：exact matrix search，或 HNSW、IVF、PQ 等 ANN 近似。

因此，Recall 下降有多种可能原因：目标文档没有摄取，ACL 把它挡住，查询/文档编码器发生漂移，ANN 没有找回
精确 top-k，重排器截断了候选，或者 qrels 本身不完整。

只有先确定损失发生在哪一层，才能判断是不是 embedding 模型的问题。

## 2. Bi-encoder 与 Cross-encoder

### 2.1 Bi-encoder：先编码，再相似度检索

[Bi-encoder](../reference/glossary.md#term-bi-encoder) 分别编码 query 和 document：

\[
q=g_\theta(x),\qquad d=h_\phi(y),\qquad s(x,y)=q^\top d.
\]

若使用 cosine，相当于先做 L2 normalization 再点积。document vector 可离线计算并建立 ANN index，所以一次 query 不必重新运行全部文档编码器。这是大规模第一阶段召回的关键计算性质。

“分别编码”不等于一定使用两套参数。Query 与 document encoder 可以共享参数，也可以各用一套；这项选择会改变
模型的归纳偏置、存储和更新方式。

保存模型身份时，至少记录两侧 checkpoint、tokenizer、pooling、normalization 和最大长度。

### 2.2 Cross-encoder：让 token 在打分前交互

[Cross-encoder](../reference/glossary.md#term-cross-encoder) 把 query 与一个 document 联合输入模型，再输出相关性分数：

\[
s(x,y)=f_\psi([x; y]).
\]

这种结构让 query token 与 document token 在多层 Attention 中直接交互，通常比单个向量点积表达力更强。

代价是每一对 query 和 document 都要单独运行模型，最终分数也不能预先保存成通用文档向量。因此，常见系统先用
bi-encoder 召回数百条，再让 cross-encoder 重排其中已经授权的几十条。

两者不是“旧模型与新模型”，而是不同计算预算下的打分分解。把 cross-encoder 放到百万文档全量扫描，或只用 bi-encoder 处理极细粒度否定和数字差异，都可能选错层。

## 3. 一个向量从哪里来

Transformer 输出 token hidden states，检索器还需把它们变为定长表示。常见 [pooling](../reference/glossary.md#term-pooling) 包括：

- special/CLS token pooling；
- 对未被 attention mask 的 token 做 mean pooling；
- 使用最后一个有效 token；
- 学习 attention pooling。

Pooling 是训练契约的一部分，不能在上线时随意替换。做平均 pooling 时，补齐位置不应进入平均值；做末 token
pooling 时，要根据左/右 padding 找到真正的最后一个有效 token。

训练时若使用点积，上线时临时改成 cosine，文档的排序也可能改变。

L2 normalization 使点积只比较方向，但会移除 norm 携带的信息。它是否正确取决于训练目标，不能因为“向量库默认 cosine”就临时添加。

## 4. Contrastive learning 与 InfoNCE

### 4.1 单正例目标

给 query \(q_i\)、一个正文档 \(d_i^+\) 和候选集合 \(C_i\)，常见 [InfoNCE](../reference/glossary.md#term-infonce) 形式是：

\[
\mathcal L_i
=-
\log
\frac{\exp(s(q_i,d_i^+)/\tau)}
{\sum_{d_j\in C_i}\exp(s(q_i,d_j)/\tau)}.
\]

它就是在当前候选集合上的 softmax cross-entropy。设 \(z_{ij}=s(q_i,d_j)/\tau\)，单正例 one-hot target 为 \(y_{ij}\)，则：

\[
\frac{\partial \mathcal L_i}{\partial z_{ij}}=p_{ij}-y_{ij}.
\]

正例得到负梯度，梯度下降会提高其 logit；每个被放进分母却未标正的文档都得到正梯度，梯度下降会降低其 logit。这正是为什么“负样本只是 DataLoader 细节”是错误说法。

这里的 \(p_{ij}\) 是**训练候选集内的归一化概率**，不是线上“文档相关概率”。换 batch、去重规则或候选池，分母和数值都会变；除非另做校准，不应拿它直接决定业务置信度。

### 4.2 多正例目标

同一 query 可能有多个正确段落。若正例集合为 \(P_i\)，一种多正例目标是：

\[
\mathcal L_i
=-
\log
\frac{\sum_{d_p\in P_i}\exp(s(q_i,d_p)/\tau)}
{\sum_{d_j\in C_i}\exp(s(q_i,d_j)/\tau)}.
\]

它优化“正例集合获得多少概率质量”，不强迫某一个正例独占全部质量。其 logit 梯度为候选 softmax 减去“只在正例集合内归一化”的 soft target。多正例目标能缓解已知同义文档互相排斥，但不能自动发现漏标正例。

### 4.3 Temperature 不只是采样温度

\(\tau>0\) 控制训练 softmax 的尖锐程度。减小 \(\tau\) 会放大分数差，也会放大 loss 对原始分数的梯度尺度。
当正例已经排第一时，当前 loss 往往下降，但梯度也可能过度集中或出现数值问题。

这里的 temperature 用于检索器训练。生成模型的 sampling temperature 使用相似的缩放形式，但作用对象和数据分布
完全不同。

比较 temperature 时，还要固定向量是否归一化、batch 与候选集大小、loss 聚合方式和优化器学习率。否则，结果差异
不能只归因于 \(\tau\)。

## 5. Negative 决定模型学会区分什么

### 5.1 Random negative

从全库随机抽取的负例便宜，但常只需主题词就能排除。Loss 很快下降不代表模型学会了细粒度相关性。

### 5.2 In-batch negatives

[In-batch negatives](../reference/glossary.md#term-in-batch-negatives) 把同一批次中其他 query 的正例，当作当前 query
的负例。一个批次有 \(B\) 对样本时，不额外编码就能形成最多 \(B\times B\) 的分数矩阵。

代价是批次组成会直接改变训练目标。同源问句、重复文档和多语种比例都会改变负例分布；跨设备收集向量后，候选域
还会进一步扩大。

分布式训练必须另外说明：远端 document embedding 是否保留梯度，全局正例下标怎样重映射，以及 padding 和重复项
怎样从候选分母中 mask 掉。

### 5.3 Hard negative

[Hard negative](../reference/glossary.md#term-hard-negative) 与 query 表面或语义相近，却按任务定义不相关。它能提供更强判别信号，例如：同一产品的旧版本、数字相近但单位不同、包含关键词却不回答问题的段落。

过难不一定更好。若 hard negative 来自不完整 qrels，它可能是未标注正例；若全部由当前模型挖掘，训练会偏向该模型已有的盲点。应混合 random、lexical、dense 与规则生成负例，并记录来源。

### 5.4 False negative

[False negative](../reference/glossary.md#term-false-negative) 实际相关却未被标正。在单正例 InfoNCE 中，它仍进入分母，梯度会主动压低其 score。这不是简单的标签噪声均值化问题：热门主题和近重复文档更容易被系统性误伤。

常见缓解包括：

- 按 stable document/answer identity 去重；
- 用 qrels、教师模型或规则形成 multi-positive mask；
- 对不确定近邻 ignore，而不是强行当负例；
- 抽样人工审查最难 negatives；
- 在真实语料版本上重新 mining，不复用已污染候选。

### 5.5 Negative mining 与泄漏

[Negative mining](../reference/glossary.md#term-negative-mining) 使用 BM25、旧版 dense retriever 或 reranker，从大库中
寻找训练负例。挖掘过程只能使用训练 split 可以获得的信息。

以下做法都会造成模型选择泄漏：在完整 corpus/query 集上调挖掘阈值，使用测试集 qrels 排除 false negative，或者根据
测试指标反复选择 miner。

正确顺序是先按独立单位切分数据，再只在训练集内挖掘负例。验证集用于选择配置，测试集只运行最终冻结评测。
实验还要保存 miner 身份、索引版本、top-k、过滤规则和每条负例的来源。

## 6. DPR-style 训练管道

[DPR](../reference/glossary.md#term-dpr) 是理解稠密段落检索的一个清晰样板。查询编码器与段落编码器分别产生向量，
再用点积和对比损失训练。候选通常包含正段落、同批次其他查询的正例，以及 BM25 等方法挖掘的负例。

把方法名还原为数据流：

```text
question -> query encoder -> q -------------------+
                                                    -> score matrix -> InfoNCE
positive / mined passages -> passage encoder -> d -+
```

“使用 DPR”这个名称不足以复现实验。还要记录：

- 正例怎样选择，passage 怎样切分；
- 负例来自哪一版索引；
- 两个 encoder 是否共享参数；
- 长度截断、pooling 与 normalization；
- Temperature、batch size 和 loss 聚合方式。

## 7. Late interaction 与 ColBERT

单向量 bi-encoder 在编码后丢失了 token 级匹配结构，cross-encoder 对每个候选做联合前向又太昂贵。
[Late interaction](../reference/glossary.md#term-late-interaction) 取中间路线：查询和文档各自保留多个 token 向量，
到检索时再执行轻量交互。

[ColBERT](../reference/glossary.md#term-colbert) 的代表性 MaxSim score 为：

\[
s(q,d)=\sum_i\max_j q_i^\top d_j.
\]

每个 query token 先找分数最高的 document token，再把这些最高分相加。这样既保留了词项级匹配，也仍能预先计算
文档 token 向量。

代价是索引更大，候选生成与压缩更复杂。MaxSim 还允许多个 query token 重复匹配同一个 document token，解释分数时
要注意这一点。

“Late”指交互发生在独立编码之后，不表示文档直到最后才读取。实际 ColBERT 系统还涉及特殊标记、归一化、残差压缩和多阶段索引；本仓库 toy 只验证 MaxSim 公式。

## 8. Learned sparse retrieval 与 SPLADE

Dense retrieval 把语义压进低维连续向量，传统 BM25 则使用词项稀疏向量。
[Learned sparse retrieval](../reference/glossary.md#term-learned-sparse-retrieval) 在词表维度上学习非负稀疏权重，
试图同时保留倒排索引的可部署性和神经模型的语义扩展能力。

[SPLADE](../reference/glossary.md#term-splade) 的常见 max-pooling 形式可写成：

\[
w_j(x)=\max_i \log(1+\operatorname{ReLU}(z_{ij})),
\]

其中 \(z_{ij}\) 是第 \(i\) 个 token 对词表项 \(j\) 的 MLM head logit。输入中没有出现的词也可能得到权重，
这就是学习到的词项扩展。

训练时还要对 query 和 document 表示施加稀疏度或 FLOPS 正则。只有上面的 pooling 公式，并不会自动得到高效索引。

它与 BM25 的关键区别不是“都能倒排”：BM25 权重由词频、文档频率和长度规则确定；SPLADE 权重由模型与训练数据学习。与 dense vector 的区别也不是“一个可解释、一个不可解释”这么绝对，而是表示空间、索引执行和训练约束不同。

## 9. Exact search、ANN 与 Reranker 的误差分层

固定 query/document embeddings 后，先用矩阵乘法得到全库 exact ranking。再比较 ANN：

- **HNSW**：图搜索；`ef_search` 增大通常提高 recall 也增加延迟；
- **IVF**：先选 centroid/inverted lists；`nprobe` 决定扫描范围；
- **PQ**：对子空间做 codebook quantization，降低内存和带宽但引入 score 近似。

至少报告三个不同指标：

1. `model Recall@k`：exact ranking 相对 qrels 的召回；
2. `ANN recall@k`：ANN 结果相对 exact top-k 的重合或覆盖；
3. `end-to-end Recall@k`：线上授权、过滤、ANN 和 rerank 后相对 qrels 的召回。

三个指标对应三种排查方向：

- Model Recall 差：检查表示模型、标签和训练数据；
- Model Recall 好而 ANN recall 差：检查索引参数和近似算法；
- 大候选集有答案，重排后却消失：检查 cross-encoder、截断和授权后的候选身份。

它们都叫 recall，但测量对象不同。

## 10. Reranker、Distillation 与训练漏斗

Cross-encoder 可以直接用作线上 reranker，也可以充当教师，为大量 query-document 对生成软分数，再通过
[distillation](../reference/glossary.md#term-distillation) 训练 bi-encoder。教师分数不是人工真值；它可能包含位置、
长度、语言和训练域偏差。

训练时应分开比较：

- 只用人工/行为 qrels；
- 加教师 soft labels；
- 加 mined hard negatives；
- 同预算下的组合。

线上 reranker 只能读取已经授权的 candidate content。先把越权文档送入 cross-encoder、最后再过滤，即使最终结果没有返回，也已经越过信息边界。

## 11. Qrels、Pooling 与不完整标注

[Qrels](../reference/glossary.md#term-qrels) 是 query-document relevance judgments。这里还有一个易混淆点：

- **embedding pooling** 把 token representations 聚合成向量；
- **qrels pooling** 汇总多个检索系统的高排名结果，交给人标注以扩大 judging pool。

Qrels pooling 会产生依赖既有系统的漏标：没有进入候选池的文档通常只是“未判断”，不应自动算作“不相关”。
如果新模型与组成候选池的旧系统差异很大，它找出的未标注文档会更多，离线指标可能因此对它不公平。

评测协议应固定：

- query 的独立单位以及 train/validation/test split；
- corpus snapshot、chunk identity、权限视图；
- graded relevance、多个正确 passage 和 no-answer 定义；
- unjudged item 如何处理；
- Recall@k、MRR、MAP、nDCG 等指标的分母和 tie-breaking；
- 每个 slice 的 query count 与置信区间。

同一原始文档切成十个近重复 chunk 时，命中十条不等于发现十份独立证据。按任务需要同时报告 chunk-level 与 source/document-level coverage。

## 12. 单卡训练时最容易漏掉的契约

一个最小训练 record 不应只有三段字符串，至少绑定：

```text
query_id, query_text, split, locale
positive_document_ids, positive_relevance
candidate_document_ids, candidate_source
corpus_version, chunker_version, qrels_version
```

训练循环还应明确：

- tokenizer 与 query/document prefix；
- pooling、normalization、score 和 temperature；
- per-device/global batch 以及 cross-device negatives；
- duplicate/false-negative mask；
- loss 是按 query、pair 还是 token reduction；
- miner/index/checkpoint identity；
- validation 使用 exact 还是 ANN search。

单张消费级 GPU 可以从小 batch 和梯度累积开始，但梯度累积不会自动增加同一次前向中的 in-batch negative 数。
每个 micro-batch 若独立计算 loss，候选分母仍然只包含当前 micro-batch。

要扩大负例域，需要显式使用 embedding cache、跨 batch memory 或其他算法，同时处理过期 embedding 和梯度语义。

## 13. 运行一个可手算的 NumPy 例子 { #exact-control }

运行：

~~~powershell
python projects/rag-foundations/retriever_learning_toy.py
python -m pytest tests/test_retriever_learning.py -q
~~~

这个例子不下载模型，而是在 CPU 上把几组二维数组当作 encoder 输出。这样可以逐项手算并检查：

- 多正例 InfoNCE 的 loss 及 query/document analytic gradients；
- analytic gradients 与 finite difference 一致；
- easy negative 的 loss 约为 `0.1269`，hard negative 增至 `0.6444`；
- 同一个 hard negative 把 temperature 从 `1` 降到 `0.25` 时，本例 loss 为 `0.5130`；
- 漏标的第二个相关文档在单正例目标中收到“降低 logit”的正梯度；改成多正例后 loss 从 `0.8620` 降为 `0.1688`，其梯度方向反转；
- ColBERT-style masked MaxSim 得到 `[2.0, 1.6]`；
- SPLADE-style max pooling 忽略 masked token，并只产生非负词表权重。

实现位于 `src/about_llm/rag/retriever_learning.py`。Mask、shape、finite value、temperature 或 positive set
不符合要求时，程序会报错并停止计算。

### 这个例子说明什么，还不能说明什么

本仓库准备的二维 embedding 和 qrels 可以检查公式实现，也能观察负例、temperature 和正例集合怎样改变梯度方向。

实验范围止于 CPU 上的数组计算。以下工作都需要另外执行：

- 加载真实 Transformer 或 DPR、ColBERT、SPLADE checkpoint；
- 构建 ANN 索引并在 GPU 上运行；
- 训练 encoder 参数并评估收敛；
- 在目标数据上测量 false negative、索引性能、模型质量和生产安全。

## 14. 常见错误结论

**“Batch 越大，InfoNCE 一定越好。”** 更多 negatives 也可能带来更多 false negatives，并改变数据分布和优化尺度。

**“Loss 下降说明 Recall 会提高。”** Loss 只在训练候选分布上计分；线上 corpus、ANN、权限和 qrels 可能不同。

**“Hard negative 越难越有价值。”** 未标注正例、过时文档或任务边界模糊时，难例可能提供错误梯度。

**“Cross-encoder 比 bi-encoder 准，所以直接替换即可。”** 它们的计算复杂度和预计算能力不同，通常处于检索漏斗的不同层。

**“ANN recall 低就是 embedding 不好。”** ANN recall 衡量近似搜索相对精确排序丢失了多少候选。表示质量应由
model recall 单独衡量。

**“SPLADE 是可学习的 BM25。”** 两者都可使用 inverted index，但权重生成、扩展、正则和训练目标不同。

## 15. 方法选择

| 约束 | 优先基线 | 下一步 |
|---|---|---|
| 语料小、延迟宽松 | BM25 + cross-encoder | 验证是否真的需要 learned first stage |
| 百万级语料、语义改写多 | bi-encoder + exact 小样本评测 | 再选 ANN 并单测近似损失 |
| 精确词项和语义扩展都重要 | BM25/dense hybrid | 比较 SPLADE 或 learned fusion |
| 单向量漏掉细粒度 token 匹配 | bi-encoder baseline | 在同预算下比较 ColBERT/late interaction |
| qrels 少且漏标严重 | lexical/dense 多系统 pooling | 人审 hard negatives，建立 multi-positive labels |
| 第一阶段召回够、排序不够 | authorization-first cross-encoder | distill 到 bi-encoder 或优化候选漏斗 |

## 16. 面试时怎样回答

面对“怎样训练一个 dense retriever”，不要只说“用 sentence-transformers 微调”。可以按以下顺序：

1. 定义 relevance、query 独立单位、corpus 与 qrels；
2. 选择 bi-encoder、pooling、normalization 与 score；
3. 写出 InfoNCE 分母，说明 positive/negative 来源和 false-negative mask；
4. train 内 mining，validation 选配置，test 冻结；
5. exact search 测表示质量，再测 ANN approximation recall；
6. 对授权后的候选做 cross-encoder rerank；
7. 保存 checkpoint、tokenizer、index、qrels 和 metric 分母；
8. 用 slice、消融与真实失败案例决定是否发布。

## 17. 自测与实践

1. 为什么另一个 query 的 positive 不一定是当前 query 的 negative？
2. 推导多正例 loss 对 logit 的梯度，并解释正例之间如何分配 target mass。
3. 为什么 gradient accumulation 不等价于更大的 in-batch negative 集合？
4. 设计一个同时区分 model recall、ANN recall 和 reranker recall 的实验表。
5. 给一个 false negative 例子，并说明怎样用 identity、qrels 或 ignore mask 缓解。
6. 比较单向量 dot product、ColBERT MaxSim 与 cross-encoder 的预计算边界。
7. 运行 toy，先预测 hard negative 与多正例的梯度方向，再核对输出。

## 一手资料

- Karpukhin et al., [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- Xiong et al., [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/abs/2007.00808)
- Khattab and Zaharia, [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- Formal et al., [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086)
- Thakur et al., [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)

下一步回到[召回、混合检索与重排](rag-retrieval.md)，把表示学习结果接入授权、ANN、融合和 reranker 漏斗。
