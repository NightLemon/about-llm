# RAG 生产架构、缓存与运维

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：RAG 平台、API、SRE 和发布负责人。
- **先修**：RAG 摄取、检索、生成链路和基本服务 SLO。
- **首次阅读**：参考架构 → API → 延迟预算 → 版本一致性 → 降级/灾备。
- **完成信号**：能定义版本、SLO、缓存身份、降级和恢复演练。
- **卡住时**：先完成[RAG 权限与超时验收](../practice/project-index.md#arag)。

</div>

生产 RAG 是一组独立扩缩、独立失败的服务：摄取、索引、检索、重排、生成、评测和反馈。把它们封装成一个同步 chain 会让故障边界、权限和成本不可见。本章给出从单机原型到多租户服务的演进路线。

## 参考架构

```mermaid
flowchart TB
    subgraph Offline["Offline / asynchronous"]
      S["Sources"] --> I["Ingestion workers"]
      I --> M["Manifest + lineage"]
      I --> X["Sparse / vector indexes"]
      X --> V["Offline evaluation"]
    end
    subgraph Online["Online request path"]
      U["Authenticated request"] --> Q["Query service"]
      Q --> X
      X --> R["Fusion + reranker"]
      R --> C["Authorized context builder"]
      C --> G["Generator"]
      G --> A["Citation / policy checks"]
      A --> U
    end
    Q --> T["Trace + metrics"]
    R --> T
    G --> T
```

离线数据面允许重试和批处理；在线路径有严格延迟预算。索引服务不应依赖生成模型可用，生成服务也不负责判断 ACL。

## API 契约

请求包含 tenant、认证主体、query、对话引用、locale、时间上下文和可选过滤；权限从可信身份层派生，不允许客户端直接传“我可访问 admin”。响应包含 answer、结构化 citations、insufficient 状态、request id、index version 和可公开的 usage。

内部检索响应要保留每阶段 rank/score/source，而外部响应只暴露经过授权的元数据。schema 版本必须显式；新增字段向后兼容，语义变化需要新版本。

仓库已有一个故意受限但真实运行的 reference boundary：`PersistentExtractiveRAGService` 从现有 SQLite schema-v1 文件启动，FastAPI body 只接受 `query_id/query/top_k/budget_units`，tenant/principals 必须由注入的 `AuthResolver` 提供。每个查询新开 SQLite connection，先 `visible_chunks`，再构建 BM25，并返回 server-generated request id、citations、ordered document IDs 与完整 extractive artifact。缺 credential 为 401，body 自报 tenant 因 extra field 为 422，数据库消失时 readiness 为 503；内部异常只返回稳定 code，不回显路径、token 或 exception。

该服务的 queue/execution deadline 还固定一个常被忽略的语义：`wait_for(asyncio.to_thread(...))` 超时不能终止同步 thread。若 504 后立即释放 semaphore，真实并发可能超过配置上限。reference 会 shield 后台 task，并在它真正结束前继续占用 permit；这保持单进程 capacity ledger，但卡死 thread 也会长期占位。多 Uvicorn worker/replica 的 semaphore 彼此独立，不能冒充全局 admission。项目 control 使用 HTTPX ASGI transport，不执行真实 TCP/TLS/reverse proxy/remote authentication，因此不构成线上可用性或安全证明。

## 延迟预算

总延迟大致为：

\[
T=T_{auth}+T_{rewrite}+\max(T_{sparse},T_{dense})+T_{rerank}+T_{prompt}+T_{generation}+T_{checks}
\]

并行 sparse/dense 可以降低 wall time；rerank 和生成通常占大头。流式输出能改善 TTFT，但引用/安全检查若只能在完整答案后执行，就不能未经策略直接把未验证 token 发给高风险用户。

为各阶段设置 timeout 与降级：dense 超时可退到 BM25，reranker 超时可用融合排名；生成超时返回可恢复错误。任何降级都带 `degraded_reason` 并进入指标，不能把降级答案混入正常质量统计。

## 缓存层次

1. parse/cache：key 为 source content hash + parser version。
2. embedding cache：normalized text hash + model revision + prefix/pooling。
3. retrieval cache：tenant/ACL version + query/filter + index/config version。
4. rerank cache：query + ordered candidate content hashes + reranker revision。
5. response cache：完整授权上下文、生成配置和 policy version。

越靠后越难安全命中。response cache 可能包含用户历史和敏感答案，默认不要跨主体共享。缓存内容应加密、TTL 明确，并支持 ACL/删除事件失效。cache hit 也要记录使用的版本。

### Semantic cache

用 query embedding 找相似历史问题可以节省生成成本，但“相似”不等于同一意图，数字、时间和否定尤其危险。只在低风险、稳定知识、相同权限与过滤条件下使用，并设置严格阈值、答案 freshness 和人工可关闭开关。

## 一致性与版本

一次请求应固定 `index_snapshot`。embedding 模型、向量和距离配置构成不可分割版本；不能用新 query encoder 搜旧 document vectors。reranker、prompt、generator 和 policy 也进入 system version。

索引切换采用 build → validate → shadow → alias switch → monitor → retire。验证包括文档/chunk 数、向量维度、随机抽样可回链、ACL 完整、gold query 指标和性能。切换后保留旧 alias 以秒级回滚。

## 可靠性

### 重试与幂等

摄取 job 使用 source/version 作为幂等键。embedding 批次可重试，但 upsert 与 manifest 提交要有事务或可恢复状态。在线生成重试可能产生两次费用和不同答案；只对明确的连接前失败自动重试，并保留 provider request id。

仓库的 `SQLiteChunkStore` 是这条原则的单机 reference：每次 source 更新用 `BEGIN IMMEDIATE` 读取当前 version，要求 caller 提供 expected-current-version；stale chunks、source manifest 和新 chunks 同事务提交。fingerprint 绑定 source bytes/ACL/metadata 以及 chunker revision/`max_chars`，所以同一 version 下改变内容或切分配置都会失败，删除必须走显式 API。SQLite trigger 注入失败测试证明单库 rollback，不证明远端向量库、object store 与 source DB 的分布式原子性。

`about-llm-rag store-upsert/store-delete/store-retrieve` 把该契约暴露为可运行 JSON CLI。创建要求 `--expect-absent`，更新和删除要求 `--expected-current-version`；upsert 从 JSONL 里按 tenant/source 精确选一条，拒绝 duplicate key、`NaN/Infinity` 与溢出为无穷的 number。成功 JSON 中的 `committed: true` 只表示该 SQLite 事务已提交；`remote_vector_index_updated: false` 和 `cross_store_atomicity_proved: false` 防止把本地成功误读为完整生产索引已发布。冲突/输入/文件/SQLite 错误使用 exit code 2，调用方必须把非零退出作为失败，而不是解析一份假成功 stdout。

授权读取先用 tenant 限定行，再过滤 principal ACL，之后才允许构造 scorer。该实现把 ACL JSON 解码后在进程内过滤，适合小型 reference；大规模共享 collection 应把强制 tenant/ACL predicate 下推到受信查询层，并验证 ANN prefilter/postfilter 对召回和泄露面的影响。

### Backpressure

索引重建、批量上传和在线 query 争用 embedding/GPU。分离队列与容量池，为在线流量保留并发；队列长度和最老任务年龄触发限流。不要让无限重试淹没下游。

### 灾难恢复

保存 source manifest、解析产物或可重建原始引用、索引配置和版本；向量索引可以重建，但时间可能很长。定义 RPO/RTO，定期演练从备份恢复 metadata、重建索引、切换流量，而不是只确认备份文件存在。

仓库的 `store-backup` 使用 SQLite online backup API 创建不覆盖旧文件的一致快照，并生成 strict manifest：物理 size/SHA-256 之外，还检查 `quick_check`、foreign keys、schema-v1 精确对象集合，以及有序 source/chunk row fingerprint。逻辑检查会拒绝 chunk content hash/stable ID、ordinal、source version、ACL/metadata 一致性漂移，也拒绝在不升级 schema version 时偷偷加入 trigger/table/index。`store-verify-backup` 可在恢复前独立复查，`store-restore` 只写入不存在的新路径并再次验证恢复后逻辑 identity；原数据库在快照后继续变化不会改写 backup。

这项证据只覆盖单机 SQLite schema-v1。manifest 的 canonical SHA-256 没有密钥，不认证操作者、主机、备份时间或来源；本工具不加密文件，也没有证明单文件 `fsync` 在断电后的目录项 durability。测试恢复 tiny fixture 不等于达到目标 RPO/RTO，更没有备份远端 ANN/vector collection、object store、cache、generation trace 或删除 tombstone。生产演练必须把这些依赖列入同一恢复 runbook，保存耗时与抽样 query 结果，并通过新 alias/canary 切流，不能因为 `quick_check=ok` 就直接替换线上索引。

## 多租户隔离

共享索引成本低但过滤错误影响面大；独立 collection 隔离强但运维和小租户资源浪费大。可按风险/规模分层：高敏租户独立，普通租户共享物理集群但逻辑分区和强制过滤。

限额包括文档/向量数、摄取速率、query QPS、rerank 候选、生成 token 和费用。noisy neighbor 要在队列、CPU、内存、GPU 和 provider 配额各层控制。trace 和离线评测数据也必须继承 tenant 边界。

## 安全与隐私

- 来源接入前验证许可、数据分类和 retention；
- 凭据使用 secret manager，连接器最小权限；
- 文档和 query 在静态/传输中加密；
- 日志默认不存完整敏感正文，debug 采样有审批与过期；
- 防止 URL fetcher SSRF、压缩炸弹、恶意文件和解析器漏洞；
- 模型供应商的数据保留、区域和训练政策进入选型；
- 删除事件传播到索引、缓存、评测与备份策略。

prompt injection、数据外泄和越权检索要有专门红队集。安全通过率是 release guardrail，不被平均答案分抵消。

## 可观测性

每个 request trace 包含：

- authentication/tenant/policy version 的不可敏感标识；
- rewrite 结果与过滤器；
- 各路检索耗时、候选 ID、分数和 index version；
- rerank/packing 决策、证据 token；
- provider/model、TTFT、输出 token、结束原因和费用；
- 引用审计、拒答/降级和用户反馈。

高基数 ID 不宜全部做 metrics label，应放 trace/log。指标按 endpoint、tenant tier、模型版本和结果状态聚合。建立 SLO：可用性、p95、freshness、权限违规为零、答案/引用质量抽样。

离线 gold 诊断可先比较 source manifest、case tenant/principals 与 required evidence：来源不在当前租户快照是 corpus/ingestion miss；来源存在但 caller 无权是 case/ACL context mismatch；来源可见却没进大候选才是 retrieval miss。评测工具可以输出这些状态，但不能在面向终端用户的响应或跨租户日志中暴露“其他租户存在同名秘密来源”。

录制答案评测不能只保存 source id 列表。生产 trace 还应保存或可回链当次 source version/content hash、实际 packed text、短引用 ID 到稳定 ID 的授权映射、packer/prompt/generator/policy revision，以及后续 claim judgment 的独立 provenance。否则索引更新后，同一个 ID 可能对应不同正文，无法复现 judge 当时看到的证据。正文快照若含敏感数据，应使用受控 artifact store、加密、retention 和访问审计，而不是塞进普通 metrics label。

仓库的 `audit-traces` 提供离线 reference gate：对 query hash、tenant/principals、逐 chunk identity/bytes、canonical context 和 recorded-answer fingerprint 做 exact join，并把 prompt/raw-output identity 纳入 canonical trace/manifest fingerprint。这是“提供的文件彼此一致且能由当前 corpus 重建”的证据，不是在线调用证明。生产实现还需让 gateway/runtime 直接签发 trace，把 decoding parameters、policy/index/packer/runtime revision 和 provider request id 纳入签名或 append-only ledger，并对历史 corpus snapshot 做受控保留。否则攻击者可一起改写 trace、answer、corpus 和无密钥 hash，仍得到一个自洽结果。

非 LLM 的 `answer-extractive` 使用独立 artifact schema，而不伪造 tokenizer token IDs 或 chat-template identity。它绑定授权 packed context、source bytes/version、span offsets、lexical gate 和 recorded answer，适合定位“检索到了但证据 gate 拒答”与“span 被 budget 丢掉”等控制路径；接入真实 tokenizer/生成器后应切换到 generation trace，不能拿 extractive artifact 声称模型执行或 token-window 合规。

## 成本模型

单问成本可拆为 embedding/query、vector search、rerank、input tokens、output tokens 和基础设施摊销。长上下文会同时增加模型延迟和费用；盲目增大 top-k 不是免费保险。

容量估算从流量分布而不是平均值出发：峰值 QPS、并发、query/文档长度、候选数、cache hit、模型 token/s。对每个优化报告质量变化和单位请求成本，避免只降低费用却提高人工升级率。

## 发布流程

1. 离线固定集：召回、答案、引用、权限、延迟和成本。
2. 组件消融：确认变化来自哪个阶段。
3. shadow：新系统接收真实请求但不返回，比较分布和错误。
4. canary：小流量、明确自动回滚阈值。
5. 扩量：观察关键切片与长尾，不只看总体。
6. 复盘：保存配置、结果和决策，不覆盖旧报告。

新 embedding 往往需要全量重建，应与 prompt/generator 变化分开发布，否则无法定位回归。

## 故障演练

至少演练：向量库超时、部分分片丢失、reranker OOM、provider 429/5xx、索引版本不匹配、ACL 服务不可用、缓存污染、删除事件卡住和恶意文档。对安全依赖应 fail closed；对非安全质量组件可显式降级。

## 从原型到生产

- L1：内存 BM25/dense，固定小语料和离线指标。
- L2：可重放 ingestion、单机事务持久 store、真实 embedding/reranker、版本化评测集。
- L3：API、持久索引、trace、ACL、缓存、故障测试。
- L4：shadow/canary、SLO、容量/成本、删除与灾备、红队门禁。

LangChain/LlamaIndex 可编排调用，但领域对象、ACL、评测和版本应保持框架无关。仓库的 adapter 项目不再只测对象构造：同一 canonical BM25 会被绑定到 LangChain `BaseRetriever.invoke()` 与 LlamaIndex `BaseRetriever.retrieve()`，并把框架结果与同次 canonical retrieval 逐字段对账。固定 control 中 engineering 主体得到两个文档、匿名主体得到一个，跨租户与错误 principal 文档均在评分前排除；两边的 Prompt SHA-256、deterministic extractive answer artifact、Recall@4 与 nDCG@4 对齐。

这仍不证明框架默认提供 ACL。保护字段进入 LlamaIndex node 供审计时，会从默认 embed/LLM metadata content 排除；自定义 formatter 仍可主动读取它们。Round-trip validator 只能检测相对于 supplied canonical results 的本地漂移，不能认证 supplied results 的来源，也不覆盖 learned embedding、vector index、reranker、LLM query engine、网络或生产性能。只有把这些组件逐一固定并保存 trace 后，才能比较完整框架 RAG。

Reference ASGI service 已补 API、持久 store、ACL、trace response、readiness 和 queue/timeout fault tests，但仍没有 production authentication、cache、真实网络压测、全局 admission、learned retrieval/LLM 或部署演练，所以当前项目不能仅因“有 FastAPI endpoint”就提升为完整 L3。

## 系统设计回答框架

面试中先问规模、freshness、权限、答案风险和延迟；给出数据面/在线面；再讲索引选择、版本与一致性、质量指标、缓存键、安全、降级和成本。不要一上来只说“向量数据库 + GPT”。优秀答案会指出：没有标注集无法声称优化，没有 lineage 无法审计，没有目标硬件压测无法承诺容量。
