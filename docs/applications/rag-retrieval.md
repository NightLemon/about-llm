# RAG 召回、混合检索与重排

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要设计 RAG 召回、排序和离线评测的开发者与算法工程师。
- **先修**：[一次 RAG 请求的生命周期](rag-request-lifecycle.md)、向量与排序的基本直觉。
- **首次阅读**：相关性 → BM25/dense → hybrid → reranker → 指标与失败归因。
- **完成信号**：能用固定 qrels 判断失败来自 corpus、retriever、ANN、reranker 还是 packing。
- **卡住时**：先只保留 BM25 与 10 条人工 query，确认 answer-bearing evidence 能否进 top-k。

</div>

检索不是寻找“最像问题的一段话”，而是在有限候选预算内覆盖回答所需证据。

在[请求 A](rag-request-lifecycle.md)中，BM25 找到了三段相关资料：
一段直接回答 ACL 顺序，两段只提供背景。Reranker 的工作是把有限上下文留给更可用的证据。

## 先把“相关”拆成三层

对 query 与 chunk，至少区分：

| 层次 | 判断问题 | 请求 A 的例子 |
|---|---|---|
| Topical | 是否在谈同一主题？ | “RAG 还需要评测拒答” |
| Answer-bearing | 是否含回答所需事实？ | “ACL 必须先于排序” |
| Usable | 权限、版本、时间和来源是否允许使用？ | `tenant-a / engineering` 可见的当前版本 |

Embedding 很容易优化 topical similarity；RAG 真正需要后两层。

一个问题也可能需要多份证据。标注时应保存 evidence set，不能强迫每个 query 只有唯一正确文档。

## 候选漏斗

生产检索通常不是单个模型，而是一条漏斗：

```mermaid
flowchart LR
  Q["Query + caller context"] --> A["授权与 metadata filter"]
  A --> S["Sparse candidates"]
  A --> D["Dense candidates"]
  S --> F["Fusion"]
  D --> F
  F --> R["Reranker"]
  R --> P["Packing"]
```

每一层用更高成本处理更少候选：

```text
百万级 corpus
-> 数百个 sparse/dense 候选
-> 数十个 rerank 候选
-> 数个 context chunk
```

如果答案在第一步就没进入候选，后面的 reranker 和 LLM 无法把它创造出来。

## 授权先于查询期统计与打分

可信索引进程可以按自己的服务权限建立索引；在线请求必须按 caller 再确定可见集合。

本仓库 BM25 的查询路径是：

1. 先选出 tenant 与 ACL 可见的 document indices。
2. 只在这些 indices 上计算 IDF 与平均长度。
3. 只对这些 documents 计算 query score。
4. 返回的 candidate 再由 reranker 与 packer 二次授权。

若只在最后删除越权结果，隐藏文档仍可能改变可见 score、候选深度、缓存与时序。

授权上下文必须进入 cache key。Policy revision 改变后，旧 cache 也要失效。

## Sparse retrieval：先学会 BM25

BM25 的常见形式为：

\[
\operatorname{score}(q,d)=
\sum_{t\in q}\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1(1-b+b|d|/\overline{|d|})}.
\]

直觉可以拆成三件事：

1. Query 词在文档出现得越多，贡献越高，但收益会饱和。
2. 在 corpus 中越罕见的词，IDF 越高。
3. 长文档更容易偶然命中，所以用平均长度归一化。

BM25 特别适合：

- 错误码、产品型号、API 字段和代码符号；
- 人名、组织名和罕见缩写；
- 用户已经知道精确关键词的查询；
- 透明、便宜且可解释的回归基线。

中文必须明确 tokenizer。按单字、词、字符 n-gram 或领域词典切分会得到不同统计，
不能只写“使用 BM25”而不版本化分析器。

### 字段化检索

标题、正文、标签和代码符号不应自动等权。BM25F 或多字段查询可以提高标题和精确字段权重。

权重必须在固定评测集上调。导航标题和模板文字会重复出现，盲目提升 title boost 可能把模板噪声排到前面。

## Dense retrieval：语义相近不等于可回答

Bi-encoder 分别编码 query 和 document：

\[
e_q=f_q(q),\qquad e_d=f_d(d),
\]

再用 inner product 或 cosine 打分：

\[
s(q,d)=\frac{e_q^\top e_d}{\lVert e_q\rVert\lVert e_d\rVert}.
\]

归一化向量上，cosine 与 inner product 的排序等价。

Dense retrieval 能找出同义改写，例如“访问控制应在哪一步执行”与“ACL 必须先于排序”。
它也可能忽略产品号、否定、数字和时效，因此通常不应无评测地替换 sparse baseline。

### Embedding contract

接入一个 Embedding 模型时，要固定：

- model 与 revision；
- query/document 是否使用不同 instruction prefix；
- tokenizer、最大长度和 truncation；
- pooling 与 normalization；
- 输出维度、dtype 和相似度函数；
- corpus snapshot 与 index revision。

漏掉模型要求的 query prefix，影响可能大于更换 ANN 参数。
更换 Embedding、pooling 或归一化通常意味着重建文档索引。

### 表示是怎样学出来的

Bi-encoder 常用 InfoNCE 或多正例对比目标。Negative 决定模型被要求区分什么：

- Random negative 容易，但训练信号可能太弱。
- In-batch negative 便宜，却可能包含漏标相关文档。
- Hard negative 信号强，也最容易放大 false negative。
- Mining 必须与 held-out split 隔离，避免把评测信息带回训练。

完整公式、梯度和 ColBERT/SPLADE 路线见[检索表示学习](retrieval-learning.md)。

## ANN：把模型误差与索引误差分开

Exact vector search 随文档量线性增长。ANN 用近似换吞吐：

| 方法 | 核心直觉 | 主要代价 |
|---|---|---|
| HNSW | 在多层邻接图中导航 | 内存较高，`ef_search` 影响质量与延迟 |
| IVF | 先选 coarse clusters，再扫描若干 list | `nprobe` 太小会漏候选 |
| PQ | 对向量子空间量化 | 压缩更强，距离误差更大 |
| DiskANN 类 | 让大部分索引驻留 SSD | I/O、预取和尾延迟更复杂 |

调 ANN 前，先用同一批 embedding 的 exact top-k 作为 oracle：

\[
\operatorname{ANNRecall@k}=
\frac{|R_{\text{ann}}^k\cap R_{\text{exact}}^k|}{k}.
\]

它衡量 ANN 是否复现 exact neighbor，不衡量这些 neighbor 是否真的 answer-bearing。

过滤会改变图或分区的可达性。ANN 参数必须在真实 tenant、ACL 和 metadata 分布下评测。

## Hybrid retrieval：融合排名而不是硬加分数

BM25 score 与 cosine score 不在同一尺度，直接相加很脆弱。
Reciprocal Rank Fusion（RRF）只使用名次：

\[
\operatorname{RRF}(d)=
\sum_r\frac{1}{k+\operatorname{rank}_r(d)}.
\]

它不要求各检索器校准到相同分布，是很稳的 hybrid baseline。

Weighted RRF 可以偏向某一路，但权重、各路候选深度和缺失 rank 都是协议的一部分。
学习融合器能利用 score 与 metadata，却更容易过拟合和漂移。

Query router 可以让错误码优先 sparse、自然语言解释优先 dense。
在有充分 trace 前，先并行召回并记录各路贡献，避免硬路由把某类 query 永久送错通道。

## Metadata filter 不是普通相关性特征

时间、产品、区域、语言、文档状态和 ACL 应尽量进入候选生成。

对“当前报销政策”而言，已失效版本不是分数略低，而是不可用于当前答案。
对历史问题，旧版本又可能是唯一正确证据，因此需要 `effective_from/effective_to` 与 query time。

LLM 可以从 query 抽取结构化 filter，但输出要经过 schema、枚举与授权校验。
严格过滤为零时应请求澄清，不能静默扩大到无权或失效来源。

## Reranker：用更贵的交互换更细的判断

Bi-encoder 在编码 query 与 document 时不交互，适合大规模召回。
Cross-encoder 把 query–document 拼在一起，让 token 在打分前交互，通常更准确但更慢。

常见漏斗是：

```text
100–1000 ANN/hybrid candidates
-> 20–100 cross-encoder candidates
-> 5–15 context units
```

Reranker 输入要保留标题和必要 parent context。若最大长度把答案段截掉，
有限且漂亮的 score 也没有意义。至少记录 truncation 比例、batch、dtype、额外 p95 和 GPU 成本。

生成式 reranker 可以处理复杂约束，但也会引入不稳定输出、文档 Prompt injection 与更高成本。
它必须只读取授权候选，输出固定 ID/schema，并与 cross-encoder baseline 比较。

### 请求 A 的教学 control

请求 A 的 recorded scores 把 ACL 顺序段排第一、引用边界段排第二、一般评测段排第三。

这条 fixture 验证授权、query/chunk 绑定、score shape、有限数与稳定排序；
它没有运行 learned model，也没有 gold qrels 质量对比。精确边界见
[RAG 证据页](../evidence/rag-answer-controls.md)。

## Query rewrite 与多跳

“那它的限制呢？”需要结合对话历史恢复实体。可以生成 standalone query，
但必须保存原 query、改写 query 和引用的 history turn，避免改写偷偷改变意图。

多跳问题可以使用：

- Decomposition：拆成若干子问题；
- Iterative retrieval：用上一跳实体扩展下一跳；
- Multi-query：生成多个改写后融合；
- HyDE：先生成假想答案或文档再检索；
- Graph retrieval：沿实体关系与文本证据共同搜索。

每种方法都会增加调用数、错误传播和注入面。需要最大深度、无进展终止、来源追踪与预算。

## 检索指标：每个数字都要写清分母

设 gold evidence 集合为 (G_q)，前 (k) 个结果为 (R_q^k)：

\[
\operatorname{Recall@k}(q)=
\frac{|G_q\cap R_q^k|}{|G_q|}.
\]

Recall 关注是否找全。MRR 关注第一个相关结果的位置。nDCG 支持分级相关性。

本仓库的 source-level Precision 使用实际返回且被检查的槽位作分母：

\[
\operatorname{Precision@k}(q)=
\frac{|G_q\cap R_q^k|}{|R_q^k|}.
\]

若系统只返回两个结果，分母是 2，不固定写成 k。其他实现可能采用固定 k，比较前必须统一。

多跳任务还要看 all-evidence recall：

\[
\operatorname{AllEvidence@k}(q)=
\mathbf{1}[H_q\subseteq R_q^k],
\]

其中 (H_q) 是缺一不可的 required evidence。它不能替代普通 Recall，因为失败 case 不告诉你究竟缺几个证据。

### No-answer 与不完整 qrels

No-answer query 的 gold 集合为空，不能塞进普通 Recall 分母。
应显式标记 `answerable=false`，并分开评价 retrieval signal 与最终拒答行为。

Gold 往往不完备。未标注的新 top result 应先叫 unjudged，不应自动当作 false positive。
可通过 pooling 补标，并同时报告 judged@k coverage。

## 用 trace 定位第一个错误

| 观察 | 归因 |
|---|---|
| Gold source 不在 tenant corpus | 摄取或语料缺口 |
| Gold source 被 ACL 挡住 | Case security context 或 policy |
| 可见 gold 未进大候选集 | Sparse/dense/ANN |
| 大候选中有、rerank 后消失 | Reranker 或截断 |
| Rerank 后有、Prompt 中没有 | Packing |
| Prompt 有证据、答案仍错 | Generation 或 citation |

只有保存每层候选，才能区分“召回漏了”和“重排丢了”。

## 选型顺序

| 当前问题 | 优先动作 |
|---|---|
| 没有可复现基线 | BM25 + 人工 qrels |
| 精确实体经常漏 | 保留 sparse，检查 tokenizer 与字段 |
| 语义改写经常漏 | 加 dense，与 sparse 做 hybrid |
| Exact vector 好、ANN 差 | 调 HNSW/IVF/PQ 与过滤策略 |
| 大候选有答案、前几名没有 | 加 reranker，检查 truncation |
| Top results 重复 | 去重、source quota 与 MMR |
| No-answer 时仍有相似结果 | 建 evidence sufficiency 与拒答评测 |

不要在没有 qrels 时用最终答案的主观观感选择 retriever。

## 可运行入口

先完成[实验 5](../practice/labs/lab-5-rag-request.md)，再运行项目中的 retrieval evaluation：

~~~powershell
python -m about_llm.rag.cli evaluate `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --top-k 3
~~~

检索表示学习的 NumPy exact control 位于：

~~~powershell
python projects/rag-foundations/retriever_learning_toy.py
python -m pytest tests/test_retriever_learning.py -q
~~~

## 面试追问

**为什么有向量库还要 BM25？**
精确实体、型号和错误码经常更适合 sparse；两路错误模式互补，hybrid 也提供可解释基线。

**怎样判断该优化 Embedding 还是 ANN？**
先用相同 embedding 做 exact search。Exact evidence recall 已低，问题在表示或语料；
exact 高而 ANN 低，才优先调索引。

**为什么 reranker 前还要再授权？**
上游候选不是永久可信，reranker 会读取正文并可能写日志或缓存。二次门禁限制错误传播范围。

## 自测

1. Topical、answer-bearing 与 usable relevance 分别是什么？
2. 为什么 ANN recall 高不代表 evidence recall 高？
3. RRF 为什么通常比直接相加 BM25 与 cosine score 更稳？
4. Reranker score 是有限实数，为什么仍需记录 truncation？
5. No-answer query 为什么不能进入普通 Recall@k 分母？
