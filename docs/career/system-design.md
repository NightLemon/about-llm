# LLM 系统设计题

## 通用回答框架

1. 澄清用户、流量、数据、风险和成功指标。
2. 给最小基线，不默认一定需要 Agent 或微调。
3. 画在线链路、离线数据链路与信任边界。
4. 定义接口、数据模型、权限和版本。
5. 做容量估算：QPS、token、并发、显存、延迟和成本。
6. 定义离线/在线评测、观测和发布门禁。
7. 分析故障、降级、恢复、安全和回滚。
8. 最后才讨论高级优化和未来扩展。

## 先把需求变成可计算的约束

系统设计不是组件背诵。开场先写一张约束表，并标出哪些是业务给定、哪些是假设、哪些必须靠压测获得：

| 类别 | 至少澄清 | 不应偷换的概念 |
|---|---|---|
| 流量 | 平均/峰值请求率、并发、突发窗口、租户分布 | 平均 QPS 不等于峰值并发 |
| 输入 | P50/P95 token、检索文档数、附件大小、语言 | 字符数不等于 token 数 |
| 输出 | P50/P95 token、流式、停止条件、结构约束 | 最大输出上限不等于实际输出均值 |
| SLO | availability、offered/dispatch TTFT、TPOT、端到端 p95/p99 | 模型执行时间不等于用户延迟；semaphore 后计时会漏 client queue |
| 质量 | 任务成功、忠实度、引用、拒答、安全 slice | 单个自动分数不等于产品质量 |
| 风险 | 数据级别、权限、副作用、地域、保留期 | Prompt 约束不等于权限边界 |
| 成本 | token、GPU、检索、重排、工具、存储、人工 | provider token 费用不等于总拥有成本 |

常用一阶公式只负责发现数量级错误：

\[
L = \lambda W
\]

Little's Law 中，稳定系统的平均在途请求数 \(L\) 等于平均到达率 \(\lambda\) 乘平均停留时间 \(W\)。它要求口径与时间窗口一致，也不能代替尾延迟和突发压测。

若每请求平均报告 \(t_{in}\) 个输入 token、\(t_{out}\) 个输出 token，粗略 API usage volume 为：

\[
R_{token}=QPS\,(t_{in}+t_{out})
\]

它适合估算 usage/费用或描述流量，不是 GPU forward work。最简单的无 cache/speculation/beam causal generation 中，每个成功请求的 forward positions 是 \(t_{in}+t_{out}-1\)，而真实 padded slots、kernel FLOPs 与显存流量还受 batch、prefix cache、调度和实现影响。prefill 与 decode 的资源曲线也不同，应分别记录输入/输出分布，不能用任一总 token rate 唯一决定容量。最终实例数来自目标模型、runtime、量化、硬件和真实长度分布下的压测，并包含故障余量。

## 用故障树组织答案

面对“为什么请求失败或超时”，不要只列监控项。先按可观测因果链拆解：

```text
用户失败
├─ 接入：认证、配额、payload、断连
├─ 排队：突发、优先级饥饿、长请求阻塞
├─ 上下文：检索空、ACL 过滤、超长、模板错误
├─ 模型：OOM、超时、坏输出、拒答、版本回归
├─ 工具：429、超时、部分成功、重复副作用
└─ 响应：schema、引用、流式协议、日志/计费遗漏
```

每个叶子至少回答四件事：检测信号、用户可见行为、自动恢复或降级、需要人工介入的阈值。降级必须保留安全不变量；例如可以降低输出预算，但不能跳过 ACL、审批或出站策略。

## 题一：企业知识库问答

### 需求澄清

文档类型与规模、更新频率、用户/租户、ACL、语言、是否需引用、允许延迟、无答案行为和数据驻留。成功指标至少包含检索覆盖、答案忠实度、引用、权限、p95 延迟与每成功回答成本。

### 设计

离线：对象存储 → 解析/OCR → 结构切分 → 内容哈希/去重 → metadata/ACL → sparse+dense index → 版本发布。

在线：身份 → query 改写 → ACL filter → BM25+dense 召回 → RRF/rerank → 去重与 token budget → 带 source id 生成 → claim/citation 验证 → 响应。

### 关键取舍

- 小 chunk 检索、父 chunk 返回；
- BM25 保留型号/错误码；
- 索引更新用蓝绿版本，cache key 包含 index version 与 ACL；
- 无证据就 abstain，不用参数记忆补齐；
- 权限在检索前执行，生成器永远看不到无权文档。
- Context packer 对完整 prospective prompt 使用目标 tokenizer/revision 计数，先预留输出，再记录每个候选因 duplicate、source quota 或 budget 被丢弃；byte/字符预算不能代替 token gate。

### 容量估算

以 QPS、平均/峰值输入 token、输出 token、检索/重排/模型延迟拆分。Little's Law 粗估并发等于到达率乘平均服务时间；GPU 容量再用目标 workload 压测，不从参数量直接猜。

窗口容量以最终 chat template 渲染结果为准。System、history、query、evidence 与 reserved output 分别做规划，但不要假设分段 token 数严格可加；上线前保存最终 input token ids/count 和 packing decision ledger，避免“检索到了但被静默截断”无法归因。

举例：若峰值 20 QPS、端到端平均停留 2.5 秒，一阶平均在途量是 50，而不是“需要 50 张卡”。先拆出检索、排队与模型阶段，再用 P50/P95 长度混合回放测试单实例可持续吞吐。若单实例在满足 TTFT/TPOT SLO 时只能稳定承载 4 QPS，理论最少 5 个实例；随后再加入滚动发布、单实例故障和突发余量。这个数字仍是容量计划，不是生产验证。

容量压测要说明 arrival process。Closed-loop completion-driven worker 会在系统变慢时降低 offered QPS；constant/Poisson open-loop 更容易暴露排队，但还要保存 scheduled 与 dispatch 时间、监控 generator lag。只生成一个有限 seeded schedule 证明配置可复现，不证明负载机跟得上、生产流量服从该分布或 client queue 就是 server queue。

### 故障

解析错、索引延迟、召回为空、证据冲突、模型超时、引用错误和跨租户缓存。每类有指标、降级和回滚；删除请求传播到原文、索引、cache 与日志。

降级顺序示例：重排器不可用时回到经离线门禁验证的 hybrid baseline；生成模型超时时返回检索证据或明确失败；索引 freshness 超标时标示版本或阻止高风险回答。不要在召回为空时让模型用参数记忆静默作答。

验收证据要分层：离线固定集证明检索/引用逻辑，故障注入证明降级与删除传播，目标负载压测证明容量假设，只有真实流量观测或受控线上实验才能支持生产 SLO 声明。

## 题二：能发邮件和建工单的 Agent

### 最小方案

优先固定 workflow：理解请求 → 收集必填字段 → 生成草稿 → 用户确认 → 执行 → 查询结果。模型不直接持有邮件/工单凭证。

### 执行契约

ToolCall 包含 call_id、工具、结构化参数；执行层使用认证 context 和 server-resolved resource 校验 ACL/policy，不能信模型自报 tenant。proposal hash 标识 tool + arguments；ledger/审批使用同时绑定 subject、资源/tool/policy revision 的 execution identity。每次 cache replay 先重新授权；policy 不确定默认拒绝。副作用审批绑定 execution identity 和过期时间。幂等 ledger 与业务事务/Outbox 防止重放。

Outbox 的准确表述是：同一数据库事务原子写业务状态和 `pending` effect；worker 用 lease claim，携带稳定 `effect_id` 调 provider，成功后保存 receipt 并转 `delivered`。provider 成功但 ack 前 crash 会在 lease expiry 后重投，所以这是 at-least-once delivery，不是 exactly-once external effect。只有 provider honor idempotency key 才能去重；receipt 需再由 audit/业务状态验证。retry 记录脱敏 machine error code，terminal failure 进入 dead letter 与 operator runbook。面试中还应主动说明 SQLite reference 不证明 broker、跨库事务、跨区域恢复或真实 provider 语义。

### 安全

邮件、网页和工单正文是不可信数据；不把其中指令提升权限。秘密在 credential broker；工具最小权限；附件扫描；出站域名和收件人策略；高风险动作二次确认。

### 恢复

每步状态持久化。超时后先按 call_id 查询外部系统，不盲目重试。达到步数、费用、时间或无进展阈值转人工。审计记录 proposed、approved、executed 和 result。

`finish` 必须经过独立 completion verifier；审批暂停要原子保存 call/execution identity、事件版本、planner state、handler counter 和已经消耗的 step/token/cost，恢复时重新授权并执行原 pending decision，而不是从空状态重跑或再次收费。checkpoint hash 不提供认证，仍需签名/MAC、加密、存储 ACL、lease、防回滚和一次性 grant。active monotonic budget 若排除审批等待，还必须另设绝对 task deadline。循环检测至少区分连续同 fingerprint、短周期和连续同类错误，但有限窗口启发式不能替代任务状态进展 verifier。

### 评测

任务成功、字段正确、步骤数、恢复、重复副作用、越权、注入抵抗、延迟和成本。真实副作用在隔离模拟环境回归；handler attempt 不等于 effect applied，后者必须由 outbox/环境状态 verifier 提供。

发布表不压成单个平均分：task success 分母是有 final-state verifier judgment 的 case；blocked-unsafe 分母是 policy-denied proposal；unapproved-attempt 分母是实际进入 handler 的副作用步骤；duplicate-effect 分母是被 verifier 确认 applied 的 effect。任何 executed policy violation、未审批 attempt、重复 effect 或 unresolved pending 都是独立 guardrail，不能由任务成功率抵消。trace 固定 environment、policy、tool 与 verifier version，并由控制面生成而不是让模型自报。

至少做以下故障注入：工具在执行前超时、执行成功但响应丢失、返回 429、返回 schema-valid 但业务失败、审批后参数被篡改、进程在写 ledger 前后崩溃。`schema-valid` 只表示形状满足契约，不表示收件人、金额或业务状态正确。

## 题三：单卡多租户 LLM 服务

### 需求

模型/量化、显存、上下文、平均/峰值 QPS、交互或批处理、TTFT/TPOT SLO、租户优先级和数据隔离。

### 服务

API gateway 做认证、配额、最大 token 与取消；scheduler 做 continuous batching、长度感知和公平；engine 管理权重、Paged KV 和 prefix cache；observer 记录队列、TTFT、TPOT、tokens/s、KV usage、preemption 与错误。

### 容量

显存 = 权重 + KV + workspace。KV 按层数、长度、KV 头、head dim、dtype 和并发估算，再保留碎片/峰值余量。分别扫描 prefill-heavy 和 decode-heavy workload。

对未做 prefix sharing 的 decoder-only 服务，KV 元素数量的一阶口径为：

\[
2\,B\,T\,L\,H_{kv}\,D_h
\]

其中 2 对应 K/V，\(B\) 是并发序列，\(T\) 是已缓存 token，\(L\) 是层数，\(H_{kv}\) 是 KV 头数，\(D_h\) 是 head dimension；再乘每元素字节数。实际 runtime 的 block、量化、prefix sharing、滑动窗口、碎片和临时 workspace 会改变测量值。

实现层再维护 per-sequence logical block table 与 physical block refcount。Prefix fork 共享 block；对 shared partial tail 的 append 先预留 COW replacement 和其余新块，再复制/换表/写入，不能先填一个 slot 后才发现无块。容量指标把 logical tokens（共享前缀按序列重复）、physical materialized positions、allocated slots、tail fragmentation 和 sharing saved blocks 分开；释放/取消必须递减引用并观测 disconnect-to-release。仓库 CPU allocator 只证明 metadata 状态机，不代表真实 GPU KV、vLLM preemption 或吞吐。

### 隔离

Prefix/response cache 的 identity 由 gateway 的可信认证状态构造，不能信任请求体自报 tenant。KV reuse 绑定 tenant/visibility domain、authorization/policy revision、model/tokenizer/template/adapter、RoPE/position config、KV dtype 与 exact token prefix；fingerprint 后仍做 full comparison。使用中的 entry 以 lease/refcount pin 住，LRU 不得淘汰；容量全被 lease 占用时 fail closed。敏感 prompt 不跨安全域共享，并处理 TTL/删除、加密和 hit/miss timing side channel。Adapter 动态加载设显存与切换限制。

### 过载

有限队列、最大排队时间、按 token 估工作量。超载时快速 429、路由小模型或降低非必要生成预算；不悄悄跳过安全检查。

公平性不能只按请求数：一个 32k 输入不应与一个 64-token 请求记同等工作量。可用 estimated prefill/decode token、deadline 与租户配额做 admission/scheduling；估算错误时仍需硬性上下文和输出上限。

### 发布

固定模型 revision、runtime 与 kernel；shadow/canary；质量/延迟联合门禁；保留上一权重、template、generation config 与 engine image 回滚。

## 面试中的证据等级

回答最后主动声明证据边界：

1. **公式估算**：发现数量级问题，不证明实现性能。
2. **离线单元/回放**：证明指定输入与故障下的逻辑，不证明真实供应商或线上分布。
3. **目标硬件压测**：证明某个模型、runtime、配置和 workload 下的容量，不可无条件外推。
4. **shadow/canary**：观察真实分布与回归，但 shadow 不证明副作用路径正确。
5. **生产 SLO/业务结果**：需要明确时间窗、样本量、流量占比、告警排除与回滚记录。

如果作品集只有 CPU 离线实验，就直接这样写；不要把离线 L2 的重放结果描述成生产 availability、GPU throughput 或线上 A/B 提升。

## 面试官常见追问

- 为什么不用微调或为什么不用 RAG？
- 数据更新和删除如何传播？
- p99 超时、GPU OOM、provider 429 怎么降级？
- 如何证明没有跨租户泄漏？
- 指标变好是否统计显著、是否过拟合测试集？
- 当模型升级但 Prompt/索引没变，为什么仍可能回归？
- 成本估算包含哪些被忽略的重试、工具和人工费用？
