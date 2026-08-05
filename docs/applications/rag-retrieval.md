# RAG 召回、混合检索与重排

检索的任务不是找“看起来相似的文字”，而是在有限候选预算内覆盖回答所需证据。一个 query 可能同时需要精确产品号、同义概念、时间过滤和两段跨文档证据，因此生产 RAG 通常是多路召回、过滤、融合、重排和去重的组合。

## 先定义相关性

对 query \(q\) 和 chunk \(d\)，相关性至少有三层：

- topical：主题相似；
- answer-bearing：包含回答所需事实；
- usable：在时间、权限、版本和可信度约束下可用于回答。

通用 embedding 容易优化第一层，RAG 真正需要后两层。标注时应记录证据集合而不只是一条文档，因为很多问题有多个等价来源或必须组合多个 chunk。

## Sparse retrieval：BM25

BM25 对精确术语、错误码、型号和人名很强。一个常见形式是：

\[
\operatorname{score}(q,d)=\sum_{t\in q}\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b|d|/\overline{|d|})}
\]

直觉上，词在当前文档出现越多越相关，但收益饱和；长文档得到归一化；全库罕见词权重大。中文需要明确分词或字符/ngram 策略。仓库的透明 BM25 用于理解和回归基线，生产可替换为 Elasticsearch/OpenSearch，但 tokenizer、字段 boost 和过滤必须版本化。

### 字段化检索

标题、正文、标签、代码符号不应等权。BM25F 或多字段查询可对标题和精确字段加权。权重必须在固定评测集上调；把所有命中标题的结果硬推到最前会被导航标题和模板污染。

## Dense retrieval

双塔模型把 query 与 document 分别编码为向量，常用 cosine 或 inner product：

\[
s(q,d)=\frac{e_q^\top e_d}{\|e_q\|\|e_d\|}

归一化后 cosine 与 inner product 排序等价。必须确认模型是否要求 query/document 不同前缀、最大长度、pooling 与归一化；漏掉 instruction prefix 可能比换索引算法影响更大。

### 选择 embedding

不要只看公开榜单。至少在目标数据比较：

- 中文、英文与混合语言；
- 短 query、长问题和关键字 query；
- 精确实体、语义改写和否定；
- 最大 token、维度、吞吐、许可和本地/云部署；
- 新旧模型切换时的索引重建成本。

向量维度高不自动更好。它增加存储、内存带宽和 ANN 成本，质量要看目标 evidence recall。

## Approximate Nearest Neighbor

精确搜索复杂度随文档量线性增长。ANN 用少量召回损失换吞吐：

- HNSW：图搜索，查询快、召回高，内存较大；`ef_search` 控制查询质量/延迟。
- IVF：先找若干 centroid，再扫描对应 inverted lists；`nprobe` 控制候选范围。
- PQ：对向量子空间量化，显著压缩但降低精度；常配 IVF。
- DiskANN 类：把大部分索引放 SSD，适合超内存规模。

调 ANN 时先用同一 embedding 的 exact top-k 做 oracle，测 ANN recall@k 和延迟；否则无法区分 embedding 质量与索引近似损失。过滤会改变图/分区可达性，必须在真实 tenant/ACL 分布下测试。

## Hybrid retrieval

Sparse 和 dense 分数量纲不同，直接相加脆弱。Reciprocal Rank Fusion 只使用名次：

\[
\operatorname{RRF}(d)=\sum_r \frac{1}{k+\operatorname{rank}_r(d)}
\]

它对分数校准不敏感，是可靠基线。Weighted RRF 可偏向某一路，但权重与候选深度需要评测。另一方案是归一化分数后学习融合器，能利用更多信号，也更容易过拟合和发生分布漂移。

query routing 可以根据错误码/实体模式优先 sparse，根据自然语言解释优先 dense，但不应过早硬路由。先并行召回并记录各路贡献，再判断路由能否在不损质量的情况下降成本。

## Metadata filtering

时间、产品、区域、文档状态和 ACL 应尽量进入 candidate generation。LLM 从 query 抽过滤条件时，要有 schema、枚举和容错；抽错过滤比不加过滤更危险。可同时运行宽松检索，若严格过滤零结果则返回澄清，而不是静默扩大到无权或过期来源。

对“最新政策”这类问题，时间不是普通相关性特征：先排除失效版本，再排序。来源权威性也可作为约束或显式特征，不能只靠语义相似度。

## Reranking

bi-encoder 独立编码，适合大规模召回；cross-encoder 联合读取 query-document，交互更细但成本随候选数增长。常见漏斗是 100–1000 个 ANN 候选 → 20–100 个 cross-encoder 候选 → 5–15 个上下文单元。

重排输入要保留标题和必要 parent context，但不能超过 reranker 最大长度后悄悄截掉答案段。记录截断比例。批量大小、长度排序和半精度影响吞吐；评估同时报告 reranker 增益、额外 p95 和 GPU 成本。

生成式 reranker/LLM 选择器适合复杂约束，但输出不稳定、昂贵且易受文档提示注入。它应在授权、清洗后的候选上运行，输出固定 ID/schema，并与 cross-encoder 基线比较。

## 多跳与查询改写

单次相似检索很难回答“项目 A 的负责人所在部门采用什么值班规则”。可用：

1. decomposition：拆成子问题，逐步检索；
2. iterative retrieval：用第一跳实体扩展下一跳；
3. graph retrieval：将实体/关系与文本索引结合；
4. multi-query：生成若干语义改写后融合；
5. HyDE：先生成假想答案，再用其向量检索。

这些方法会放大调用数、错误传播和提示注入面。每一步要有最大深度、去重、无进展终止、来源追踪和预算。不能只展示一个成功 demo；用标注的单跳/多跳切片分别测增益。

## Context selection

top-k 不是最终上下文。候选还要去重、合并相邻 chunk、控制每来源配额、保留必要 parent，并在 token budget 内选择。可把它看成带预算的集合覆盖：既覆盖多个必要事实，又避免十段重复内容。

常用策略：

- 对同 source 的相邻 chunk 合并，但重新检查长度；
- 用 maximal marginal relevance 在相关性与多样性间平衡；
- 每来源设置软上限，避免一个长文档垄断；
- 为高优先级政策或精确匹配预留槽位；
- 最终按文档原始顺序排列相关片段，减少阅读跳跃。

“lost in the middle”意味着长上下文中间证据可能被忽略。可把关键证据放在首尾、压缩冗余、分阶段综合，但必须通过任务评测验证，不能只依据位置经验法则。

## 离线指标

设 gold evidence 集合为 \(G_q\)，前 \(k\) 个结果为 \(R_q^k\)：

\[
\operatorname{Recall@k}=\frac{|G_q\cap R_q^k|}{|G_q|}
\]

MRR 关注第一个相关结果，nDCG 支持分级相关性，Precision@k 反映上下文污染。多跳任务还要测 all-evidence recall：所有必要证据是否同时出现。报告总体均值之外的 query 类型、语言、权限、时间和长尾实体切片。

若 gold 不完备，未标注文档不等于不相关。可对系统新增 top 结果做 pooling 后补标，或使用 judged@k 覆盖率提示指标可信程度。

## 在线与诊断

在线记录 query、过滤器、各路候选/分数/版本、融合贡献、重排、选入上下文和最终引用。敏感正文按政策脱敏。核心指标包括零结果率、引用点击/展开、答案接受、人工升级、检索 p50/p95、缓存命中和每问费用。

错误 taxonomy 应至少区分：语料缺失、解析失败、过滤错误、召回失败、rerank 失败、context packing 丢失、生成未使用证据。只有分类后才知道该调 embedding 还是 prompt。

## 可运行实验

Notebook `03_rag_retrieval_and_evaluation.ipynb` 用可控向量比较 BM25、dense 和 RRF。进一步实验：

1. 构造包含型号、同义改写、否定和跨租户的 30 条 query。
2. 分别测 sparse、dense、RRF 在 k=1/3/10 的 recall 与 MRR。
3. 注入近重复 chunk，观察 precision 与上下文 token。
4. 用 exact dense 作为 oracle，再模拟 ANN 漏召回。
5. 固定候选，替换 reranker，测质量增益/额外延迟。

## 面试追问

**为什么 dense 相似度高仍可能不能回答？** 它可能只主题相关、版本过期、缺关键数字或没有授权；answer-bearing 与 usable relevance 需要数据、过滤和 rerank 共同保证。

**如何判断该优化召回还是重排？** 检查 gold evidence 在大候选集是否出现。若 recall@100 已丢失，重排无能为力；若在 100 内却进不了最终上下文，才是 rerank/packing 问题。

**为什么不能只看最终答案准确率调检索？** 生成模型可能靠参数记忆答对，也可能有证据仍生成错。最终指标掩盖组件原因，必须同时测 evidence recall、context precision、引用和答案质量。
