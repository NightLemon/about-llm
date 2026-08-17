# Agent 工具协议、幂等与故障恢复

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：Agent runtime、安全、授权和恢复工程师。
- **先修**：[Agent 架构](agent-architecture.md)、JSON Schema、身份与幂等基础。
- **首次阅读**：tool contract → 验证顺序 → policy/approval → effect identity → 恢复。
- **完成信号**：能通过未授权、重复执行、超时未知和崩溃恢复测试。
- **卡住时**：回到[系统安全](../quality/safety.md)先画主体、资源和信任边界。

</div>

工具调用把自然语言的不确定性连接到真实副作用，是 Agent 最危险也最值得工程化的边界。本章从 schema、授权、幂等、并发、超时和 reconciliation 建立一个可审计 runtime。

## Tool contract

工具定义至少包含：名称、用途、参数 JSON Schema、返回 schema、副作用等级、所需权限、幂等语义、timeout、费用/速率限制和错误码。描述要区分相似工具，并写明不适用场景。

参数使用窄类型和枚举，不接受含糊 `options: object`。资源标识由服务端解析并检查归属；路径、URL、SQL、shell 命令等高风险字符串需要专用校验器。

返回值应小而结构化：

~~~json
{
  "status": "ok",
  "data": {"ticket_id": "T-17"},
  "provenance": {"system": "support", "version": "v3"},
  "retryable": false
}
~~~

大量正文写入 artifact store，返回 handle、hash 和摘要。异常转换为稳定错误类别，模型不应依赖 provider 原始堆栈。

执行层还必须在 handler 返回后重新建立结果边界。本仓库先把返回值编码为严格 canonical JSON，再 JSON round-trip 脱离 handler 持有的对象，最后递归冻结 mapping/list；调用方随后修改原对象不会改变 outcome、cache 或 SQLite value。`NaN`、非字符串 key、set/自定义对象等非 JSON 返回被记为 `failed`，claim 保持 pending，旧 call id 不会因为序列化失败而再次进入 handler。JSON-valid 只证明可移植的值域和稳定快照，不证明业务语义、来源或远端 effect 正确。

## 模型 Planner 输入输出边界

不要让 provider SDK 的自由文本直接构造 handler 参数。一个可审计的 planner adapter 至少分四层：

1. 用 canonical request 绑定 system prompt/prompt revision、task state、剩余预算、tool catalog、输出 cap 和预期 model revision；
2. transport 返回 raw text 时同时保存 provider request id、实际 model revision、input/output usage、cost 与 finish reason，缺失时 fail closed；
3. strict parser 只接受一个 closed-schema JSON object，再转换为 typed proposal；
4. runtime 重新做工具 schema、资源归属、policy、approval、预算和幂等检查，绝不把“模型成功解析”当成授权。

严格 JSON 不等于普通 `json.loads` 的默认行为。边界应拒绝 duplicate object key、`NaN/Infinity`、overflow float、Markdown code fence、trailing prose、未知字段和未知工具；evidence id 要求非空且不重复。还应检查 provider 报告的 output tokens 没有超过发出的 cap，并把 request、完整 normalized response 和最终 typed action 共同绑定为 decision identity。Input usage 往往只有响应后才知道，因此整次 input+output 仍可能使总 token budget 超限；上层 loop 要记账并阻止该 action 执行，不能倒填成“未调用”。

最近 tool observation 必须明确标成不可信数据。System prompt 的这句话只能降低风险，无法建立安全边界：恶意网页仍和指令出现在同一模型上下文。可信 subject/tenant/capability 不进入模型可编辑参数；resource resolver、policy、approval 与 verifier 必须在模型外。Prompt-facing JSON Schema 也只是候选工具说明；只有 runtime 对同一 schema/revision 执行受支持的 validator，才能声称该结构契约已执行。

仓库 `JSONSchemaToolContract` 提供这个同源路径：一份冻结 schema 同时派生 `PlannerToolContract` 和 runtime `Tool`，request 还绑定 validator revision。Profile 使用 `jsonschema` 的 Draft 2020-12 validator，不自行发明 schema 语义；要求 explicit draft、closed root object，local `$ref/$dynamicRef`，拒绝 `$id`/external reference，并限制 schema/instance bytes。`format` 默认只是 annotation；显式打开才执行当前库已知 checker，validator identity 包含精确库版本与 mode。失败信息保留 keyword/instance-schema JSON Pointer，但不回显参数值。

这仍不是业务 validator：JSON Schema 不 coercion、不应用 `default`，也不能验证 authenticated subject、tenant ownership、数据库当前 version、审批或副作用。复杂跨资源条件应在 server-owned resolver/semantic validator 中完成，并保持 schema → resolver → policy 的顺序。手写 Planner contract 与 callback 仍可能漂移；只有从共享 contract factory 派生的工具具有这里的同源保证。External reference 被拒绝意味着当前实现不适合需要远程 schema registry 的系统；生产 registry 必须固定内容 hash、缓存/availability、授权和迁移策略，不能让 validator 临时访问模型提供的 URL。

仓库 `model_planner_control.py` 用 exact recorded request replay 运行两步正例，并运行 request drift、fenced JSON、runtime schema rejection 和 missing capability 四条负例。无密钥 fingerprint 只证明 canonical bytes 相等，不认证真实 provider；recorded response metadata 也不是账单或线上模型证据。生产 trace 若保留 raw response，必须把 prompt injection、secret/PII、访问控制、加密和 retention 纳入数据治理。

Provider 返回的 opaque reasoning/thinking/signature block 还多一层风险：它可能是客户端代管、后续会被模型再次解释的状态工件。不能因为 ciphertext 或 signature 未被修改，就把它当作当前 subject/session/model 已授权的历史。恢复前需要验证 authenticated subject、tenant、session/branch、predecessor、model audience、expiry、key status 与 replay identity；外部下载或公开 trajectory 中的 block 默认不恢复执行。协议与离线反例见 [Opaque Reasoning 工件与轨迹安全](../quality/reasoning-artifact-security.md)。

Recorded fixture 应由 allowlist projection 从原始响应生成。公开 trajectory 默认移除 reasoning/signature/未知 opaque block；只对 visible text 做 secret scrub 不足，因为使用者既无法检查 opaque 内容，也无法判断它是否包含隐藏 instruction。需要长期暂停/恢复的企业 workflow 应使用受控存储、身份绑定和版本化迁移，而不是把 raw provider transcript 当作可移植 checkpoint。

## 验证顺序

推荐顺序：

1. 找到工具并验证 schema；
2. 规范化参数并计算 proposal fingerprint；
3. 用可信主体解析资源 owner/version，执行 ACL 和 policy；policy 不确定时 fail closed；
4. 计算包含主体、资源、工具和 policy 版本的 execution fingerprint；
5. 每次 proposal（包括 cache replay）先重新授权，再判断是否需要审批并验证 grant；
6. 检查 task/tool/token/time/cost budget；
7. 原子 claim execution fingerprint；
8. 执行、记录结果并更新状态。

参数先验证再让用户审批，避免用户批准一个执行时会被重新解释的模糊动作。授权必须在每次调用时检查，不能因为 Agent 之前访问过资源就继承权限；cache hit 也不能绕过撤权后的重新授权。

## 主体、资源与 policy

`ExecutionContext(task, subject, tenant, capabilities)` 必须来自认证网关/任务控制面，不能从模型参数里的 `user_id` 或 `tenant_id` 构造。工具的 `resolve_resource` 将业务标识解析为 server-owned `ResourceRef(tenant, type, id, version)`；真实实现应查数据库/目录服务，并避免用错误差异或时序泄露资源是否存在。resolver 本身必须只读、限时、可审计，不能偷偷执行目标副作用。

仓库 `CapabilityPolicy` 是刻意简单的 reference：仅允许同 tenant 且持有完全相等 capability 的请求，没有 wildcard、role inheritance、条件表达式、deny override 或集中吊销。未传 policy 时使用 `DefaultDenyPolicy`；policy backend 返回 indeterminate 也不得进入 handler。它证明 fail-closed 控制流和 cache re-authorization，不等于生产 IAM/RBAC/ABAC 已实现。

## 副作用分级

- read-only：只读且无敏感泄露；仍有费用和 SSRF 风险。
- reversible：可撤销写入，如创建草稿；需要权限和审计。
- irreversible/high-impact：付款、发送、删除、发布、权限变更；需要强审批或双人复核。

“可逆”不等于低风险：发送错误内部消息即使可删除也可能已经被阅读。分级要看业务影响、传播速度和补偿可靠性。

## Fingerprint 与 call id

`call_id` 标识一次逻辑动作。proposal fingerprint 由 tool name 与规范化参数决定：

\[
f=\texttt{sha256:}\,\|\,\operatorname{SHA256}
\left(\operatorname{canonical\_json}(\{\text{tool name},\text{arguments}\})\right)
\]

它回答“模型提议的 tool + arguments 是否逐字节相同”。真正写入 ledger 的 execution fingerprint 还绑定 task/subject/tenant、tool version/side-effect/capability、server-resolved resource owner/id/version，以及 policy version/effect/reason：

\[
e=\operatorname{SHA256}(f,\text{context},\text{tool contract},\text{resource revision},\text{policy decision})
\]

同 call id + 同 execution fingerprint 才返回缓存结果；同 call id + 不同 execution identity 是冲突。这防止换主体、资源版本、工具实现或 policy 后静默复用旧结果；同时每次 replay 仍先重新授权，所以 capability 被撤销时返回 `policy_denied`，而不是缓存 payload。

本仓库 reference runtime 只接受有限 JSON 值：拒绝 `NaN`/`Infinity`、非字符串 object key 和任意 Python 对象；Agent CLI artifact 还拒绝重复 object key 和未知字段，避免 parser 静默覆盖审批值或拼写错误。按 UTF-8、排序 key、固定分隔符编码，并把脱离调用方的快照递归冻结。若业务需要时间、金额或 Unicode normalization，必须先在业务 schema 中显式规范化，通用 JSON 序列化不会替你定义这些语义。

SHA-256 指纹避免把完整参数直接写进 ledger，但它既不是加密保险箱，也不证明两个调用语义等价、已获授权或输入不含秘密。低熵敏感参数仍可能被枚举猜测；生产系统应最小化参数、隔离加密 payload，并对 hash 和审计记录实施访问控制。当前 execution identity 与早期保存 canonical JSON 明文、或只保存 proposal hash 的实验 ledger 都不兼容，升级后应新建 ledger 或显式迁移，不能混用。

幂等键不能只用“当前时间 + 随机数”，否则重试无法复用。它应来自持久 task/step/call identity，并传递给支持 idempotency key 的外部 provider。

## Exactly-once 幻觉

本地数据库 claim 与远端副作用无法一般性地组成原子事务。执行序列：

1. ledger 写 `pending`；
2. 调远端成功；
3. 进程在写 `completed` 前崩溃。

重放可能重复付款，不重放又可能丢结果。SQLite/Redis 锁只能防并发，不能解决这个不确定窗口。正确方案按能力排序：

- 使用远端幂等键并查询结果；
- transactional outbox/inbox，在同业务数据库内原子提交；
- prepare/confirm 两阶段业务协议；
- saga + 可验证补偿；
- 人工 reconciliation。

不要宣称通用 exactly-once；通常能做到 at-least-once delivery + idempotent effect，或 at-most-once attempt + reconciliation。

## Transactional outbox 状态机

Transactional outbox 精确解决的是一个较窄的问题：把**本地业务状态**和“未来需要投递的 effect 记录”放进同一个数据库事务。它不把远端 provider 纳入该事务，也不直接保证外部 effect exactly once。一个可审计状态机可写成：

~~~text
pending --claim(lease)--> claimed --ack(receipt)--> delivered
   ^                         |
   |                         +--retryable error--> pending(next_attempt_at)
   +---- lease expired ------+
                             +--terminal error--> dead_letter
~~~

关键不变量：

- `task_id`、`effect_id` 和 effect fingerprint 有唯一约束；本地 task state 与 `pending` outbox row 同事务写入，任一插入失败则全部回滚；
- worker 以短 lease 领取 due row；lease 只表示本地并发所有权，不代表远端没有收到其他请求；
- 每次投递都把稳定 `effect_id` 作为 provider idempotency key，重试不能生成新 key；
- provider 成功后、worker 写本地 receipt 前崩溃时，lease 到期后必然**可能重投**；只有 provider 真正 honor 同一 idempotency key，两个 request 才可能折叠为一个 effect；
- receipt 是 provider supplied artifact，需绑定 request/effect identity 并在可能时用 provider audit 或业务状态复核；它本身不证明真实 effect 已发生；
- retry 只记录脱敏、稳定的 machine `error_code`，不把原始 provider response、token 或 PII 塞入 outbox；terminal failure 进入 dead letter，由有权限的 operator 按 runbook 调查、修正、重放为新事件或补偿，不能无限自动重试。

仓库的 `SQLiteTransactionalOutbox` 实现 `pending / claimed / delivered / dead_letter`、claim/renewal、退避调度、过期 lease 重投和 event timeline。下面的无网络实验故意模拟“provider 已成功、worker 在 ack 前崩溃”：

~~~powershell
python projects/safe-agent/outbox_demo.py `
  --database artifacts/agent/outbox-demo-001.db
~~~

预期是两个 provider call、同一个 idempotency key、一个模拟 effect、最终 `delivered`。这是 local SQLite + in-memory idempotent provider 的确定性证据，只证明 reference 状态机的 at-least-once delivery；它不证明真实 provider 支持幂等、receipt 可信、跨区域容灾、broker redelivery、数据库断电 durability 或 exactly-once external effect。

## Ledger 状态

仓库 runtime 使用 `pending / completed` 主状态，并通过 reconciliation 记录 `externally_confirmed / abandoned / compensated`。关键不变量：

- claim 是唯一约束保护的原子操作；
- handler 失败后 pending 不自动重放；
- 外部确认成功可补写 completed value；
- completed/reconciliation value 进入持久 ledger 前也按严格 JSON 验证，不能依赖 `json.dumps` 对 NaN 或非字符串 key 的宽松改写；
- 放弃/补偿保留原 claim 与审计事件；
- 重试业务意图必须生成新 call id 并重新审批；
- late completion 不能覆盖已 reconciliation 的状态。

这避免“超时 = 失败”的错误假设。超时只说明客户端不知道结果。仓库 `max_tool_calls` 当前准确语义是 handler-attempt budget：未获审批、缓存 replay 和预算前拒绝不计入；完整 Agent loop 仍必须另外限制 proposal/step、模型 token、时间和费用。

进程恢复不能把 handler-attempt budget 清零或扩大上限。本仓库 runtime 支持从可信 checkpoint 恢复 `initial_executed_tool_calls`，并要求它位于 `[0, max_tool_calls]`；checkpoint 同时绑定原 `max_tool_calls`，loop resume 要求 counter 与 cap 都完全一致。这个 counter 只适用于单 task/reference runtime，生产共享 worker 应由持久化 task budget service 原子扣减，不能让客户端自报已用次数。

## Concurrency control

同一 task 可并行只读调用，但写入共享资源要声明冲突域，例如 `account:42`。可使用 optimistic version、数据库 row lock 或队列单写。模型生成的步骤顺序不提供并发安全。

工具返回资源 version/ETag，后续更新带 `if_match`；冲突时重新观察并 replan，而不是覆盖别人更新。审批绑定旧 fingerprint 时，replan 后需要新审批。

## Timeout、取消与重试

分清连接 timeout、读取 timeout、业务 deadline 和用户取消。取消本地协程不代表远端动作被取消。工具 contract 标明取消语义，并记录 provider request id。

自动重试只用于明确 transient 且副作用幂等的错误，使用指数退避、jitter、最大次数和总 deadline。schema/permission/insufficient funds 不重试。429 尊重 `Retry-After`；重试预算独立，防止 outage 时 retry storm。

## 审批协议

仓库核心不再接受 `approved=True`。副作用第一次执行返回 `needs_approval` 和 execution fingerprint；可信审批服务签发 `ApprovalGrant`，至少绑定 authorized subject、task、call id、execution fingerprint 与 expiry。执行时任何参数、主体、资源 version、tool version 或 policy version 漂移都会得到 `approval_rejected`。

approval 与 checkpoint 是两件工件：checkpoint 保存“暂停在哪一步和已消耗多少”，grant 保存“谁在何时授权哪个 execution identity”。恢复不能把 grant 写进 planner 输入后当布尔值，也不能因为 checkpoint hash 相同就跳过当前 policy。过期 grant 的 resume 返回 typed `approval_rejected` 并保留原 pending checkpoint，operator 可在重新核验后签发新 grant；handler 未被调用。

审批内容示例：

~~~text
subject=user-7
task=task-9
call_id=call-3
execution_fingerprint=sha256:...
scope=send_email:customer-17
expires_at=...
~~~

UI 显示收件人、标题、正文摘要/全文、附件、发送后果。令牌由可信服务签名并一次性消费。仓库 `ApprovalGrant` 只是进程内 typed contract：它不验签、不检查 approver 是否真的有授权权力，也不提供跨服务 bearer-token 格式。Safe Agent CLI 为离线实验构造并标注 `simulated_unsigned_fixture`，不能复制到生产认证边界。

## Sandbox 与秘密

代码/shell/browser 工具在最小权限 sandbox 运行：限制文件根、网络域、进程、CPU/内存/时间和输出大小。防止符号链接/path traversal、内网 SSRF、云 metadata endpoint 和命令拼接。

秘密由工具代理注入请求，不进入 prompt/observation。工具输出做 secret/PII 扫描，但不能依赖扫描作为唯一边界。高敏工具使用独立 worker 和凭据，避免一个被注入的浏览任务获得生产数据库权限。

## Audit 与 trace

每步记录 task/call、主体、模型/prompt/policy/tool version、原始 proposal、规范化参数 hash、授权决定、approval、claim、外部 request id、结果 hash、耗时、费用和状态转换。敏感 payload 分离加密并按 retention 删除。

审计日志防篡改、按需检索且不允许 Agent 自己删除。在线指标不把 call id 做 label；trace 中保存高基数细节。

## 可运行代码

~~~python
preview = runtime.execute(call, context=context)
grant = approval_service.approve(
    subject=context.subject_id,
    task_id=context.task_id,
    call_id=call.call_id,
    execution_fingerprint=preview.execution_fingerprint,
)
outcome = runtime.execute(call, context=context, approval=grant)
if outcome.status == ExecutionStatus.FAILED and "pending" in outcome.message:
    # 先查 provider audit log，不要重复 execute(call)
    stale = ledger.list_stale_pending(older_than_seconds=300)
    ledger.resolve_external_completion(
        call.call_id,
        {"remote_id": "..."},
        note="verified by provider request id",
    )
~~~

如果确认动作未发生，可标记 abandoned；若发生后完成逆向操作，标记 compensated。两者都不允许旧 call id 复用。

## 测试矩阵

- 同 ID 同参数只执行一次；同 ID 不同参数冲突；
- caller 深层修改不改变快照，NaN/Infinity/非字符串 key 被拒绝；
- 无审批写操作不执行；审批过期/参数变化失败；
- policy 未配置/不确定、capability 缺失和跨 tenant 全部 fail closed；
- cache replay 重新授权；撤权后不返回旧 payload；
- subject/resource/tool/policy version 漂移改变 execution identity 并使旧审批失效；
- 两进程同时 claim 只有一个获胜；
- task state 与 outbox insert 原子提交，注入 outbox insert 失败时两者都不存在；
- provider 成功后 ack 前崩溃，lease 到期用同一 effect id 重投；stale worker 不能 ack；
- retry schedule、lease renewal、dead letter 与脱敏 error code；
- handler 前失败、执行中超时、远端成功本地写失败；
- handler 返回后修改原对象、嵌套结果写入、NaN/不透明结果；
- abandoned/compensated 后 late completion 被拒绝；
- 429/5xx 重试上限和 non-retryable 错误；
- path/URL/SQL 注入、跨 tenant 资源、恶意工具输出；
- crash/restart 后状态与 artifact 可恢复。

## 面试追问

**数据库有唯一 call_id，为什么还会重复付款？** 唯一约束只保证一个本地执行者，不能原子覆盖远端付款与本地 completed 写入之间的崩溃窗口；需要远端幂等或 reconciliation。

**补偿是否等于回滚？** 不是。退款是新业务动作，可能失败、产生费用或无法撤回已传播信息。Saga 记录每步和补偿结果，不能假设恢复到完全未发生状态。

**为什么旧 call id 不能在 abandoned 后重试？** 旧 ID 的审计含义是“那一次不确定动作”。复用会混淆两次授权和远端结果；新尝试必须有新身份与审批，同时关联原事件。
