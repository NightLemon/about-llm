# 云模型 API 可靠性：重试、不确定结果与预算

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已能解析 typed response，准备处理重试、流式故障、费用和生产治理的工程师。
- **先修**：[云 API 契约基础](cloud-api-contracts.md)、deadline、幂等与基本数据库事务。
- **首次阅读**：三问重试法 → timeout/cancel → per-attempt 预算 → 真实 smoke。
- **完成信号**：能为每个 attempt 判断是否重放，并解释 outcome uncertain 如何结算和对账。
- **卡住时**：画出一次 logical call 的 attempt timeline，不要先写自动重试循环。

</div>

**学习入口**：[契约基础](cloud-api-contracts.md) · [实验 0C](../practice/labs/lab-0c-cloud-budget.md) · [生产检查表](../practice/production-checklist.md) · [证据台账](../evidence/cloud-api-controls.md)
{ .doc-nav }

云模型调用最棘手的失败，不是明确的 400 或成功的 200，而是“客户端不知道远端做了什么”。

请求可能已经被接收并生成、计费或触发工具，只是响应在返回途中丢失。可靠系统必须把这种不确定性保留下来，而不是用一个 retry 按钮把它覆盖。

## 先区分 logical call 与 attempt

用户的一次任务是 logical call；每次真正发送到 provider 的网络请求是 attempt。

~~~text
logical call
├── attempt 1: timeout after write → outcome uncertain
├── decision: can we replay?
└── attempt 2: completed → usage settled
~~~

两个 attempt 都可能生成和计费。Task 最终成功，也不能删除第一次 attempt 的未知 usage。

日志和预算 identity 至少使用：

~~~text
logical-call-id
attempt number
canonical request fingerprint
provider/model/API revision
reservation id
provider request id if observed
terminal classification
~~~

## 重试前回答三个独立问题

### 1. Retryable？

当前 error/status 在这个 provider、endpoint 和 API version 下，是否被确认是瞬时失败？

网络库异常名或 status 属于 5xx，不足以自动回答。Retry allowlist、Retry-After 和配额语义都要绑定版本。

### 2. Replay safe？

再次发送是否会造成不可接受的重复生成、费用或业务副作用？

只读 prompt 没有数据库写入，也可能产生第二份不同输出和第二笔费用。若工具 loop 已经执行写操作，必须依赖业务幂等与 effect ledger，而不是模型 call ID。

### 3. Outcome known？

客户端能否证明上一次未被 provider 接收，或已经得到明确 terminal？

Connect 前失败有机会被分类为 proven not sent；write/read timeout 通常无法证明。收到 HTTP response 也不自动回答是否产生了 usage。

只有 policy 允许 retry、replay safe 且前一次 outcome 足够明确时，才进入自动重放。否则停止并 reconciliation。

## 用失败矩阵替代一个 if

| 场景 | Outcome | 默认方向 |
|---|---|---|
| 本地 schema/preflight 失败 | known not sent | 修配置，不重试 |
| 连接前确定失败 | known not sent | 按 policy 有界重试 |
| 收到明确非 2xx response | response known，usage 仍按契约判断 | 记录后决策 |
| write/read/attempt timeout | uncertain | 停止或人工对账 |
| 2xx stream 中途截断 | partial + uncertain | 不透明重放 |
| tool effect 后 ack 丢失 | effect uncertain | 查询 effect ledger |
| caller cancellation | local wait stopped | 不假设 server cancelled |

矩阵应由 typed failure stage 驱动，而不是解析任意异常字符串。原始远端 error body 也不应直接进入普通日志。

## Deadline 是一条共同时间线

至少区分：

- pool acquire timeout；
- connect timeout；
- write timeout；
- read/idle timeout；
- 单 attempt timeout；
- logical call overall deadline；
- caller cancellation。

使用 monotonic clock，让 attempt、backoff 和 Retry-After 共享同一 overall deadline。

~~~text
t0 reserve
→ connect/write/read
→ failure classification
→ retry decision
→ backoff
→ next reserve
→ next attempt
→ overall deadline
~~~

每一步都要检查剩余时间。不能让三个 30 秒 attempt 加两个 20 秒 sleep，突破声明的 60 秒业务 deadline。

### Retry-After 怎样处理

若 provider contract 允许，解析非负 delta-seconds 或 HTTP-date：

- absent：使用本地 backoff；
- valid：按服务端时间等待，并受 policy/deadline 约束；
- malformed：记录后回退本地 policy；
- 超过 deadline：停止，不为赶时间提前发送。

指数退避与 jitter 是 caller policy，不是 provider 事实。测试要注入 clock 和 random source，避免依赖真实 sleep。

## Cancellation 不证明远端停止

取消本地 coroutine、关闭 response 或断开 socket，只证明客户端停止等待。

它不能单独证明：

- provider 没有继续生成；
- server 已释放资源；
- usage 为零；
- 工具没有执行；
- billing 已取消。

取消后要 terminalize 本地 reservation，并根据 provider receipt、usage 或 billing export 对账。没有证据时保留 uncertain。

## 2xx stream 开始后默认不重放

一旦向用户发布了 partial output，透明重试可能导致：

- 文本重复或分叉；
- tool proposal 重复；
- 两次 usage；
- 用户已经消费但本地没有 commit；
- 新一次随机生成不再延续旧文本。

可选策略是：

1. 以 incomplete terminal 结束；
2. 发起新的 logical call，并明确新 identity；
3. 仅使用 provider 正式支持且已验证的 resume contract。

不要让通用 SSE decoder 自行 reconnect。Framing 层不知道业务、usage 和工具 effect。

## HTTP target 先于费用 reservation

发送前完成：

- exact origin allowlist；
- HTTPS-only；
- 禁止 userinfo、fragment 和非预期 query；
- redirect 默认关闭；
- 显式 proxy、certificate、DNS 与 egress policy；
- 认证信息来自 secret manager 或受控环境注入；
- request body 和 headers 通过 strict serialization。

若 URL/preflight 已失败，不应先占用预算，也不能把 secret 放进 URL 或错误消息。

成功 response 仍要限制 Content-Type、body bytes、duplicate JSON keys、NaN/Infinity、顶层类型和 schema。完整 body 缓冲后才检查大小，并不能防止下载过程占用过多内存。

## 为什么必须发送前 reserve

只在响应后累加 usage 会产生并发超支。若十个请求同时看到剩余预算，每个都可能各自发送到上限。

对 attempt \(i\)，发送前可保守预留：

\[
R_i=\widehat T_{\mathrm{in}}^{(i)}
+T_{\mathrm{out,max}}^{(i)}.
\]

费用估计为：

\[
\widehat C_i
=C_{\mathrm{in}}(\widehat T_{\mathrm{in}}^{(i)})
+C_{\mathrm{out}}(T_{\mathrm{out,max}}^{(i)})
+C_{\mathrm{other,max}}^{(i)}.
\]

Input token 可能只是目标 tokenizer/template estimate；cache、reasoning、tool、tier、currency 和 tax 要按带日期 pricing contract 单独处理。

Reservation 是风险上限，不是实际 usage 或发票。

## 每个 reservation 只有一个终态

~~~mermaid
stateDiagram-v2
    [*] --> reserved
    reserved --> cancelled: proven never sent
    reserved --> settled: complete trusted usage
    reserved --> uncertain: sent / usage unknown
    settled --> [*]
    cancelled --> [*]
    uncertain --> [*]
~~~

- **Cancelled**：有结构化证据证明 attempt 未发送。
- **Settled**：严格解析到可信 usage。
- **Uncertain**：可能已发送，但 outcome 或 usage 不足。

若 actual usage 超过 reservation，必须先记录已发生的真实值，再触发 post-call breach。为了让 cap 看起来没超而截断 ledger，会让审计失真。

### Retry 要重新 reserve

每个 attempt 都建立独立 reservation：

~~~text
call-42:attempt:1 → uncertain
call-42:attempt:2 → settled
logical call total  → sum of both terminal amounts
~~~

第一次的费用若无法证明为零，就不能被第二次成功覆盖。Hard limit 也必须在第二次真正发送前重新判断。

## Local ledger 不是远程 exactly-once

SQLite transaction 可以串行化同一文件的 writers，却无法与 provider generation/billing 原子提交。

Worker crash 后，active reservation 继续占额度是更保守的选择。按 TTL 自动释放会把“本地进程消失”误写成“provider 未收到请求”。

跨机器或跨区域需要共享 durable quota；仍要用 provider request ID、usage/billing export 和人工 reconciliation 处理不确定窗口。

同理，业务 outbox 和幂等键能降低重复 tool effect，却不能单独证明 exactly-once。

## 成本指标使用 task 分母

只报平均 request cost 会奖励廉价失败。更有意义的是：

\[
\text{cost per verified task}
=\frac{\sum_i C_i}
{\sum_i \mathbb 1[\text{task}_i\text{ verified success}]}.
\]

同时报告：

- all-attempt cost；
- success-conditional cost；
- retry amplification；
- uncertain reservation ratio；
- cache/reasoning/tool usage；
- pricing snapshot、currency、tax、tier 和 credits；
- billing export reconciliation gap。

没有账单导出对账时称为估算，不称为发票成本。

## 生产 adapter 的依赖方向

~~~text
canonical types
↑
provider request/response adapters
↑
strict JSON and provider event state machines
↑
HTTP/SSE transport
↑
retry orchestration + per-attempt budget
↑
business policy / tool runtime
~~~

Canonical types 不依赖供应商 SDK。业务 runtime 也不直接读取 SDK response class。

这种方向允许离线回放 parser，固定 SDK/API 升级差异，并把真实 network smoke 放到显式 opt-in、带费用上限的测试中。

## 安全和发布工件

普通日志只保存稳定 error category、status、相对时间、脱敏 request ID、decision 和 artifact identity。

不要默认保存：

- API key 或认证 headers；
- raw prompt/response；
- secret、PII 与完整 tool result；
- provider opaque reasoning/state；
- 任意远端异常文本。

内部 trajectory 与公开 artifact 分开。公开输出使用 closed-schema allowlist，再独立做 secret/PII、版权、consent 和用途审查。

## 从离线 control 到真实 smoke

真实测试必须显式 opt-in，并限制：

1. Exact allowed origin。
2. Model 与 API revision。
3. Request count、concurrency 与 output cap。
4. Per-attempt 和 total cost。
5. Overall、attempt 与 idle timeout。
6. 禁止真实副作用 tools。
7. Artifact 脱敏与保存期限。
8. Provider usage/billing reconciliation。

分层记录：

~~~text
DNS/TLS/HTTP connected
→ authentication accepted
→ non-stream typed response
→ stream terminal and usage
→ cancellation observation
→ error/rate-limit behavior
→ billing export reconciliation
~~~

一层通过不能替代下一层。一次成功 smoke 只证明该时间、账号、model 和输入上的运行，不证明生产 SLO。

## 故障定位顺序

### 解析错误

先保存受控 raw artifact identity，再检查 Content-Type、strict JSON、schema revision 和 unknown fields。不要转成字符串后继续业务。

### 流式重复或缺字

依次检查 byte framing、event identity、item/block index、delta/done reconciliation、terminal 和 reconnect policy。

### 重试风暴

检查 retry allowlist、Retry-After、logical deadline、outcome-uncertain gate 和全局并发。不要只增大 backoff。

### 预算与账单不一致

沿 logical call → attempt → reservation → provider request ID → usage/billing export 追踪，区分 estimation、隐藏 usage、失败 attempt 和 crash window。

### 工具重复执行

查询业务 effect ledger，核对 subject、resource、tool、normalized arguments、policy revision、idempotency key 和 receipt。

## 一个可运行的故障实验

用 MockTransport 或本地 fake provider 构造：

1. Connect 前失败。
2. 写入后 read timeout。
3. 非 2xx + Retry-After。
4. 2xx stream 发布部分文本后截断。
5. 第一次 uncertain、第二次成功。
6. Tool effect 成功但 ack 丢失。

对每个 case，先预测 retryable、replay-safe、outcome-known、reservation terminal 和用户可见状态，再运行对账。

仓库中的 strict fixtures、逐 attempt ledger、命令与当前边界见[云 API 证据台账](../evidence/cloud-api-controls.md)。

## 常见错误

- 对所有 429/5xx 或网络异常自动重试。
- 把 read timeout 当作 proven not sent。
- Logical call 只 reserve 一次，却内部发送多个 attempts。
- Client cancel 后立即释放费用或工具 effect。
- 2xx stream 截断后透明 reconnect。
- 用 SQLite transaction 声称远程 exactly-once billing。
- 只报成功请求平均成本，不统计失败和 uncertain attempts。
- 用离线 MockTransport 结果声称真实 provider 计费和取消已验证。

## 面试时怎样回答

面对“怎样设计 LLM API 重试”，先回答三问：错误是否 retryable、请求是否 replay safe、上次 outcome 是否 known。

然后说明每个 attempt 独立 reserve 和 terminalize，partial stream 不透明重放，工具副作用进入 outbox/effect reconciliation，最后与 provider billing 对账。

这个回答比“指数退避 + jitter”更完整，因为它覆盖了真正的重复生成、费用和副作用风险。

## 自测

1. 为什么同一个 logical call 的第二次发送需要新的 reservation？
2. 哪些 failure stage 可能证明请求未发送？哪些通常不能？
3. Client cancellation 后，为什么 active reservation 不能自动释放？
4. 为什么 2xx partial stream 默认不应透明重放？
5. Cost per verified task 比平均 request cost 多揭示了什么？

## 继续学习

- [实验 0C](../practice/labs/lab-0c-cloud-budget.md)：逐 attempt 预算实验。
- [Cloud API 项目](../practice/projects/cloud-api-contracts.md)：strict adapters、SSE 与 retry controls。
- [Agent Runtime](../applications/agent-runtime.md)：tool effect、outbox 和 reconciliation。
- [生产检查表](../practice/production-checklist.md)：发布、观测和回滚。
- [云 API 证据台账](../evidence/cloud-api-controls.md)：精确策略、fixtures 和未验证范围。
