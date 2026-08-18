# 生产 RAG：从一次回答到长期运行

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：要把 RAG baseline 做成多用户服务的平台、后端与算法工程师。
- **先修**：[请求生命周期](rag-request-lifecycle.md)、[摄取](rag-ingestion.md)与[引用/拒答](rag-generation.md)。
- **首次阅读**：边界 → 更新 → 服务 → 可观测性 → 评测发布 → 故障恢复。
- **完成信号**：能为身份、索引版本、缓存、终态、删除和回滚画出一条完整证据链。
- **卡住时**：先用单进程、单 tenant、SQLite/BM25 做正确性 control，再替换分布式组件。

</div>

离线 Demo 只需要回答一次问题。生产 RAG 必须在文档变化、权限变化、流量波动、
模型失败和依赖故障时，持续给出可解释且不越权的终态。

生产化的核心不是换成更大的向量库，而是让每个边界都有 identity、状态、门禁和恢复方式。

## 先把系统分成三个平面

```mermaid
flowchart TB
  subgraph CP["控制面"]
    SRC["Source registry"] --> ING["Parse / chunk / embed"]
    ING --> IDX["Versioned indexes"]
    IDX --> REL["Validation / release / rollback"]
  end
  subgraph DP["在线数据面"]
    API["Gateway + trusted identity"] --> RET["Authorized retrieval"]
    RET --> RR["Rerank + packing"]
    RR --> GEN["Generator"]
    GEN --> PUB["Publication policy"]
  end
  subgraph EP["证据面"]
    TRACE["Trace / metrics / evaluation"]
    AUDIT["Security audit / incident evidence"]
  end
  REL --> RET
  DP --> TRACE
  CP --> TRACE
  TRACE --> AUDIT
```

- **控制面**把原始资料变成可发布、可回滚的索引版本。
- **数据面**处理每个用户请求，不能继承控制面的宽权限。
- **证据面**保存诊断与发布依据，也必须独立授权和脱敏。

把三者混在一个进程和一套数据库中，初期可以工作，但安全与恢复边界仍要在设计中明确。

## 请求路径：可信身份必须先到

一次请求至少需要：

```text
request_id
authenticated subject
tenant_id
principals / entitlements
authorization policy revision
query + conversation state
deadline and budget
```

Query、top-k 和生成参数可以来自请求；tenant 与 principals 不能来自未验证 body。

请求 A 在生产环境中的顺序仍然是：

```text
authenticate
-> authorize corpus scope
-> retrieve
-> rerank
-> pack
-> generate
-> validate citations/claims
-> publish or refuse
```

框架可以封装调用，不能改变这条安全顺序。

## 摄取路径：先构建，再发布

不要边写当前索引边服务流量。推荐 immutable version + alias：

1. 为 source snapshot 分配版本。
2. 解析、切分、提取 metadata 与 ACL。
3. 生成 lexical/vector index 到新 namespace。
4. 校验数量、失败率、抽样解析、ACL 与 retrieval regression。
5. 发布 manifest，原子切换 read alias。
6. 保留上一版，直到观察窗口结束。
7. 异步清理旧版，但遵守审计和删除策略。

Manifest 至少绑定：

```text
corpus snapshot
parser/chunker revision
embedding model/revision
tokenizer/pooling/normalization
index type and parameters
metadata/ACL schema
build time and validation report
```

只有 alias 切换成功，不代表新索引质量合格；发布前必须跑固定 qrels 和安全负例。

## 更新、删除与权限变化

文档生命周期不只是 upsert：

```text
create -> update -> supersede -> revoke -> delete -> purge evidence
```

需要分别定义：

- 新版本何时生效；
- 旧版本是否仍可回答历史问题；
- Source 删除怎样传播到 chunks、vector index、cache 和备份；
- ACL 收紧后旧 cache、trace 和 rendered context 怎样失效；
- Partial failure 如何重试而不产生双版本；
- 法务保留与用户删除请求冲突时谁决策。

空抓取不能自动解释为“删除全部”。网络失败与来源真的为空必须是不同状态。

## 索引一致性不是一个布尔值

一个 source 可能同时存在于 object store、metadata DB、sparse index、vector index 和 cache。

跨存储通常没有单个 ACID 事务。常见选择是：

- Outbox / change log 驱动各索引更新；
- 每份派生数据带 source version；
- Query 只读取同一已发布 snapshot；
- 后台 reconciliation 找 missing/extra/stale records；
- 失败时回退到上一完整版本，而不是混合新旧。

“最终一致”必须写出允许的 stale window、用户可见行为和删除 SLA。

## 多租户与缓存

多租户系统至少防四类混淆：

1. Corpus 与 index namespace 混淆。
2. Rerank 或 response cache key 缺少 caller/policy context。
3. Trace、错误日志和 APM 收集越权正文。
4. Shared embedding/cache 让隐藏文档影响可见 score 或时序。

Cache key 至少考虑：

```text
tenant / subject or entitlement set
policy revision
query and rewrite
corpus/index revision
retriever/reranker/model revision
generation and publication policy
```

高基数 entitlement 不能简单拼成长字符串。可以使用 canonical identity 或授权结果版本，
但 hash 只绑定内容，不替代授权。

ACL 收紧时，TTL 等待通常不够。需要主动失效或每次命中后重新授权。

## API 契约与终态

一个生产 endpoint 不应只返回 `answer: string`。至少区分：

```text
answer
abstain
reject
error
timeout
cancelled
```

每个终态有唯一 reason code。HTTP status、业务 action 与模型 finish reason 是不同层，不能混成一个字段。

Public response 可以包含：

```json
{
  "request_id": "server-generated",
  "action": "answer",
  "answer": "... [S1]",
  "citations": [{"id": "S1", "title": "..."}],
  "missing_information": []
}
```

不要返回内部 ACL、原始越权候选、被 reject 的 raw output 或完整审计对象。

## Deadline、取消与并发

客户端超时或断连，不等于后台工作已经停止。

对每一层分别问：

- HTTP handler 是否停止等待？
- Retriever/reranker/model 调用是否支持 cooperative cancellation？
- 线程、GPU sequence 或远端请求是否仍在运行？
- 并发 permit 何时释放？
- Usage 与费用最终是否已知？

如果同步数据库或模型工作在线程中执行，取消 coroutine 不能杀死线程。
Permit 应在后台工作真正结束后释放，否则系统会报告虚假的可用容量。

并发控制要覆盖 retrieval、rerank 和 generation 各自瓶颈，不能只限制 HTTP request 数。

## Trace：保存证据链，而不是全文堆积

一次请求的最小 trace 可包含：

```text
request/caller/policy identity
query and rewrite hash
corpus/index revision
authorized candidate count
retrieval/rerank IDs, scores and truncation
packing decisions and prompt token ledger
model/runtime/generation identity
raw-output hash and restricted location
claim/citation findings
final action, reason and timestamps
```

需要正文时，优先保存受控 snapshot reference 与 content hash；直接复制全文会扩大敏感数据面。

Unsigned SHA-256 可以发现 bytes 漂移，不能认证谁产生了工件，也不能阻止攻击者协同改写全部文件。

## 可观测性：先定义 attempt 分母

每个用户请求可能触发 rewrite、multiple retrieval、rerank、生成 retry 或 judge。
业务 request 与 provider attempt 要分别计数。

至少监控：

| 维度 | 指标 |
|---|---|
| 入口 | QPS、queue、认证/授权失败、deadline |
| Retrieval | candidate count、zero results、Recall 切片、filter rate |
| Rerank | candidate depth、truncation、额外 p95、GPU 利用 |
| Packing | token 使用、budget drop、source concentration |
| Generation | TTFT、E2E、token、provider error、cost |
| Answer | answer/abstain/reject/error、citation 与 accepted risk |
| Security | 跨权限 attempt、blocked candidate、注入与数据外传告警 |
| Ingestion | freshness、parse failure、stale/delete backlog |

平均值会隐藏 tenant、语言、文档类型和长尾 query。Dashboard 必须支持这些切片。

## SLO 与质量门禁分开

一个低延迟错误答案不是成功。上线门禁至少有四类：

1. **质量**：evidence recall、claim support、completeness、false refusal。
2. **安全**：tenant/ACL、注入、source exposure、public projection。
3. **可靠性**：success rate、timeout、error、恢复与删除传播。
4. **性能成本**：queue、p95/p99、token、GPU/CPU、每成功任务成本。

发布规则应写成联合条件，例如：

```text
security regressions = 0
and accepted-answer risk <= threshold
and false-refusal increase <= threshold
and p95/cost within budget
```

不能用平均正确率抵消一次跨租户泄漏。

## 评测集与线上反馈

离线集至少包含：

- Answerable、no-answer、partial-answer；
- 冲突版本、过期来源与时间查询；
- 跨 tenant、ACL blocked 与 policy change；
- 数字、否定、单位、条件和多跳；
- 文档 Prompt injection 与伪造 citation；
- Timeout、parser error、provider error 和取消。

线上 thumbs-up/down 很稀疏且有选择偏差，不能直接当 gold。
将用户反馈用于 error discovery，再经过隐私、标注与切分流程进入评测集。

Judge 模型也要固定 revision、rubric 和 calibration。Judge 通过不替代人工高风险抽查。

## 容量与成本账本

每个业务请求的成本可能包括：

```text
query rewrite
embedding
sparse/vector retrieval
reranker
generator input/output
claim judge
storage and egress
retry and failed attempts
```

报告每成功任务成本，而不只报告每次模型调用成本：

\[
\text{cost per successful task}=
\frac{\text{all attempt cost}}{\text{verified successful tasks}}.
\]

缓存可以省成本，也可能造成权限和新鲜度事故。先证明 cache key 与 invalidation，再讨论命中率。

## 备份、恢复与回滚

需要恢复的不只是 metadata DB：

- Source snapshot 与解析工件；
- Sparse/vector index 与 alias；
- Embedding/reranker/model manifest；
- Policy、Prompt 与 schema；
- Evaluation reports 与 release decision；
- 必要的审计证据。

恢复演练应在新位置执行，验证 schema、row/chunk identity、索引可读性和固定 query。

单个 SQLite backup 成功，只证明该 fixture 的本地快照路径。
它不证明远端 vector store、object store、cache 和流量切换的 RPO/RTO。

## 常见事故与第一检查点

| 事故 | 第一检查点 |
|---|---|
| 更新后答案突然变差 | corpus/index alias、parser/chunker 与 qrels regression |
| 某 tenant 看到别人的标题 | trusted identity、namespace、cache key、public projection |
| Recall 高但答案漏条件 | packing、truncation、claim completeness |
| No-answer 问题大量硬答 | evidence sufficiency 与 publication action |
| 延迟上升但模型没变 | queue、candidate depth、reranker、index compaction |
| 删除后仍能引用旧内容 | cache、旧 snapshot、trace retention、备份策略 |
| 客户端取消后容量不恢复 | 后台线程/远端调用与 permit 生命周期 |

事故复盘要保存第一个错误环节，不只贴最终坏答案。

## 渐进式生产路线

1. 单机 BM25 + extractive answer，建立权限与分母。
2. 加 SQLite/source version，演练更新、删除与恢复。
3. 加 dense/ANN 与 reranker，分开测表示、索引和排序误差。
4. 接目标 tokenizer 与 LLM，保留原始失败和 citation/refusal gate。
5. 加 API 认证、deadline、并发、trace 与 public projection。
6. 在 staging 使用真实 IAM、目标 corpus 和 shadow traffic。
7. 小流量发布，联合观察质量、安全、可靠性与成本。

每一步都应保留上一层透明 control，而不是被更复杂框架覆盖。

## 可运行入口

[RAG Foundations](../practice/projects/rag-foundations.md) 提供单机 reference：

- Markdown split 与 stable chunk；
- Authorization-first BM25、rerank 与 packing；
- Exact-span answer、citation 与拒答；
- SQLite upsert/delete/backup/restore；
- Persistent extractive ASGI service；
- 固定 Qwen 原始失败、policy replay 与 guarded control。

先运行：

~~~powershell
python projects/rag-foundations/rag_request_walkthrough.py
python projects/rag-foundations/rag_service_control.py
~~~

它们是本地教学 control，不是生产部署模板。精确边界见
[RAG 证据页](../evidence/rag-answer-controls.md)。

## 系统设计面试回答顺序

1. 先问 corpus、用户、权限、freshness、流量与质量目标。
2. 画控制面、数据面和证据面。
3. 沿请求讲授权、召回、重排、packing、生成与发布。
4. 沿更新讲 version、alias、delete、reconciliation 与 rollback。
5. 给出质量、安全、SLO、成本的联合门禁。
6. 最后讨论缓存、分布式索引、多区域与灾备。

不要从“选择哪家向量数据库”开始。数据库是组件，不是系统边界。

## 自测

1. 控制面有权读全部 corpus，为什么数据面仍要按请求重新授权？
2. ACL 收紧后，只等待 response cache TTL 有什么风险？
3. 为什么 `client cancelled` 不能自动记为后端工作已停止？
4. Index alias 切换成功为什么不能作为质量发布依据？
5. 每成功任务成本为什么要把失败 attempt 计入分子？
