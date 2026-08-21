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

## 1. 先把检索拆成四层

一次 dense retrieval 至少包含四个不同对象：

1. **表示模型**：把 query 和 document 变成向量或 token/vector 集合；
2. **打分函数**：点积、cosine、MaxSim 或 cross-encoder score；
3. **候选域**：哪些文档参与训练分母，线上又有哪些文档经过授权后可见；
4. **搜索算法**：exact matrix search，或 HNSW、IVF、PQ 等 ANN 近似。

因此“Recall 下降”不能直接推出“embedding 变差”。它可能来自：目标文档未摄取、ACL 过滤、query/document encoder 漂移、ANN 没找回 exact top-k、reranker 截断或 qrels 不完整。先固定层次，才有可解释实验。

## 2. Bi-encoder 与 Cross-encoder

### 2.1 Bi-encoder：先编码，再相似度检索

[Bi-encoder](../reference/glossary.md#term-bi-encoder) 分别编码 query 和 document：

\[
q=g_\theta(x),\qquad d=h_\phi(y),\qquad s(x,y)=q^\top d.
\]

若使用 cosine，相当于先做 L2 normalization 再点积。document vector 可离线计算并建立 ANN index，所以一次 query 不必重新运行全部文档编码器。这是大规模第一阶段召回的关键计算性质。

“两个 encoder”不表示一定有两套不同参数。可以共享参数，也可以用不同 query/document encoder；选择改变的是归纳偏置、存储和更新契约。模型 identity 至少应绑定两侧 checkpoint、tokenizer、pooling、normalization 和最大长度。

### 2.2 Cross-encoder：让 token 在打分前交互

[Cross-encoder](../reference/glossary.md#term-cross-encoder) 把 query 与一个 document 联合输入模型，再输出相关性分数：

\[
s(x,y)=f_\psi([x; y]).
\]

它允许 query token 与 document token 在多层 attention 中交互，通常比单个向量点积表达力强；代价是每个 query-document pair 都要前向，无法把最终 pair score 预计算成一个通用 document vector。因此常见漏斗是：bi-encoder 召回数百条，cross-encoder 只重排授权后的几十条。

两者不是“旧模型与新模型”，而是不同计算预算下的打分分解。把 cross-encoder 放到百万文档全量扫描，或只用 bi-encoder 处理极细粒度否定和数字差异，都可能选错层。

## 3. 一个向量从哪里来

Transformer 输出 token hidden states，检索器还需把它们变为定长表示。常见 [pooling](../reference/glossary.md#term-pooling) 包括：

- special/CLS token pooling；
- 对未被 attention mask 的 token 做 mean pooling；
- 使用最后一个有效 token；
- 学习 attention pooling。

Pooling 是模型契约，不是可随意替换的后处理。Mean pooling 若把 padding 算入分母，会让长度改变向量尺度；last-token pooling 若取到左/右 padding，会直接读错位置。若训练时点积、上线时改成 cosine，score ordering 也可能改变。

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

\(\tau>0\) 控制训练 softmax 的尖锐程度。较小 \(\tau\) 放大 score gap 和对 score 的梯度尺度；当正例已经最高时常降低当前 loss，但也可能造成饱和、梯度集中或数值问题。它与生成时 sampling temperature 使用相似数学缩放，却属于不同协议和数据分布。

调 temperature 时必须同时固定：是否 normalize、batch/candidate size、loss reduction 和 optimizer learning rate。否则不能把差异单独归因于 \(\tau\)。

## 5. Negative 决定模型学会区分什么

### 5.1 Random negative

从全库随机抽取的负例便宜，但常只需主题词就能排除。Loss 很快下降不代表模型学会了细粒度相关性。

### 5.2 In-batch negatives

[In-batch negatives](../reference/glossary.md#term-in-batch-negatives) 把同一 batch 中其他 query 的正例当作当前 query 的负例。一个 batch 有 \(B\) 对样本时，不额外编码即可形成最多 \(B\times B\) score matrix。

它的代价是 batch composition 成为目标的一部分：同源问句、重复文档、多语种比例和跨设备 all-gather 都会改变负例分布。Distributed training 中还必须说明远端 document embeddings 是否保留梯度、global positive index 如何重映射，以及 padding/重复项如何 mask。

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

[Negative mining](../reference/glossary.md#term-negative-mining) 用 BM25、旧 dense retriever 或 reranker 从大库寻找训练负例。Mining 输入必须只使用训练 split 可以获得的信息。

若先在完整 corpus/query 集上调 mining threshold、用 test qrels 排除 false negatives，或根据 test metric 反复选择 miner，就发生了 model-selection leakage。
正确流程是：先按独立单位 split，再在 train 内 mining；validation 只选配置；test 只做最终冻结评测。保存 miner identity、index version、top-k、过滤规则和每条负例来源。

## 6. DPR-style 训练管道

[DPR](../reference/glossary.md#term-dpr) 是理解 dense passage retrieval 的一个清晰样板：query encoder 与 passage encoder 产生向量，使用点积和对比损失，以正 passage、batch 内其他 positives 以及 BM25 等来源的 negatives 训练。

把方法名还原为数据流：

```text
question -> query encoder -> q -------------------+
                                                    -> score matrix -> InfoNCE
positive / mined passages -> passage encoder -> d -+
```

真正影响复现的通常不是“是否叫 DPR”，而是：positive 如何选、passage 如何切、negative 来自哪一版 index、encoder 是否共享、长度截断、pooling、normalization、temperature、batch size 和 reduction。只报模型名称而不保存这些字段，无法复现训练目标。

## 7. Late interaction 与 ColBERT

单向量 bi-encoder 在编码后丢失了 token 级匹配结构；cross-encoder 又太昂贵。[Late interaction](../reference/glossary.md#term-late-interaction) 取中间路线：query/document 分别编码成多个 token vectors，检索时才进行轻量交互。

[ColBERT](../reference/glossary.md#term-colbert) 的代表性 MaxSim score 为：

\[
s(q,d)=\sum_i\max_j q_i^\top d_j.
\]

每个 query token 找最匹配的 document token，再跨 query token 求和。它保留比单向量更多的词项级证据，同时仍能预计算 document token vectors；代价是索引更大、候选生成与压缩更复杂，MaxSim 也可能让多个 query token 重复匹配同一 document token。

“Late”指交互发生在独立编码之后，不表示文档直到最后才读取。实际 ColBERT 系统还涉及特殊标记、归一化、残差压缩和多阶段索引；本仓库 toy 只验证 MaxSim 公式。

## 8. Learned sparse retrieval 与 SPLADE

Dense retrieval 把语义压进低维连续向量；传统 BM25 使用可解释的词项稀疏向量。[Learned sparse retrieval](../reference/glossary.md#term-learned-sparse-retrieval) 学习词表维度上的非负稀疏权重，尝试同时获得 lexical index 的可部署性与神经语义扩展。

[SPLADE](../reference/glossary.md#term-splade) 的常见 max-pooling 形式可写成：

\[
w_j(x)=\max_i \log(1+\operatorname{ReLU}(z_{ij})),
\]

其中 \(z_{ij}\) 是 token \(i\) 对词表项 \(j\) 的 MLM-head logit。输入没出现的词项也可能获得权重，形成 learned expansion。训练还需对 query/document 表示施加稀疏度或 FLOPS 正则；只做上式并不会自动得到高效稀疏索引。

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

若第 1 项差，优先看表示与训练数据；第 1 项好而第 2 项差，调索引；候选好但 rerank 后变差，检查 cross-encoder、截断与授权后候选 identity。三个数共用“recall”一词，但 estimand 不同。

## 10. Reranker、Distillation 与训练漏斗

Cross-encoder 可作为线上 reranker，也可作为教师给大量 query-document pairs 打 soft score，再 [distillation](../reference/glossary.md#term-distillation) 到 bi-encoder。教师分数不是 gold：它可能有位置、长度、语言和训练域偏差。

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

后者会产生 system-dependent missing labels：未进入 pool 的文档通常是 unjudged，不应自动等同于 non-relevant。比较一个与 pooling systems 非常不同的新模型时，未标注文档更多，指标可能对它不公平。

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

单张消费级 GPU 可从小 batch + gradient accumulation 开始，但 accumulation 不会自动增加同一次 forward 的 in-batch negative 数。
若每个 micro-batch 独立算 loss，候选分母仍只是 micro-batch；要扩大负例域，需要显式 embedding cache、跨 batch memory 或其他算法，并处理 stale embeddings 与梯度语义。

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

本仓库准备的 embeddings 和 qrels 足以检查当前公式实现，以及改变 negative、temperature 或正例集合时的
梯度方向。它没有运行 Transformer、真实 DPR/ColBERT/SPLADE checkpoint、ANN index 或 GPU，也没有训练
encoder 参数。模型质量、训练收敛、真实 false-negative 比例、索引性能和生产安全仍需在目标数据与系统上验证。

## 14. 常见错误结论

**“Batch 越大，InfoNCE 一定越好。”** 更多 negatives 也可能带来更多 false negatives，并改变数据分布和优化尺度。

**“Loss 下降说明 Recall 会提高。”** Loss 只在训练候选分布上计分；线上 corpus、ANN、权限和 qrels 可能不同。

**“Hard negative 越难越有价值。”** 未标注正例、过时文档或任务边界模糊时，难例可能提供错误梯度。

**“Cross-encoder 比 bi-encoder 准，所以直接替换即可。”** 它们的计算复杂度和预计算能力不同，通常处于检索漏斗的不同层。

**“ANN recall 低就是 embedding 不好。”** ANN recall 是相对 exact ranking 的近似误差；先分开测 model recall 与 ANN recall。

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
