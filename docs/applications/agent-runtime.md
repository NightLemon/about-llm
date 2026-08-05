# Agent 工具协议、幂等与故障恢复

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

## 验证顺序

推荐顺序：

1. 找到工具并验证 schema；
2. 规范化参数并计算 fingerprint；
3. 认证主体、资源 ACL 和 policy；
4. 判断是否需要审批并验证 approval token；
5. 检查 task/tool/token/time/cost budget；
6. 原子 claim 幂等键；
7. 执行、记录结果并更新状态。

参数先验证再让用户审批，避免用户批准一个执行时会被重新解释的模糊动作。授权必须在每次调用时检查，不能因为 Agent 之前访问过资源就继承权限。

## 副作用分级

- read-only：只读且无敏感泄露；仍有费用和 SSRF 风险。
- reversible：可撤销写入，如创建草稿；需要权限和审计。
- irreversible/high-impact：付款、发送、删除、发布、权限变更；需要强审批或双人复核。

“可逆”不等于低风险：发送错误内部消息即使可删除也可能已经被阅读。分级要看业务影响、传播速度和补偿可靠性。

## Fingerprint 与 call id

`call_id` 标识一次逻辑动作，fingerprint 由 tool name 与规范化参数决定：

\[
f=\operatorname{canonical\_json}(tool, arguments)
\]

同 call id + 同 fingerprint 返回缓存结果；同 call id + 不同 fingerprint 是冲突。规范化要固定 key 排序、数值/时区/Unicode 表示，并在签名后禁止模型修改参数。

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

## Ledger 状态

仓库 runtime 使用 `pending / completed` 主状态，并通过 reconciliation 记录 `externally_confirmed / abandoned / compensated`。关键不变量：

- claim 是唯一约束保护的原子操作；
- handler 失败后 pending 不自动重放；
- 外部确认成功可补写 completed value；
- 放弃/补偿保留原 claim 与审计事件；
- 重试业务意图必须生成新 call id 并重新审批；
- late completion 不能覆盖已 reconciliation 的状态。

这避免“超时 = 失败”的错误假设。超时只说明客户端不知道结果。

## Concurrency control

同一 task 可并行只读调用，但写入共享资源要声明冲突域，例如 `account:42`。可使用 optimistic version、数据库 row lock 或队列单写。模型生成的步骤顺序不提供并发安全。

工具返回资源 version/ETag，后续更新带 `if_match`；冲突时重新观察并 replan，而不是覆盖别人更新。审批绑定旧 fingerprint 时，replan 后需要新审批。

## Timeout、取消与重试

分清连接 timeout、读取 timeout、业务 deadline 和用户取消。取消本地协程不代表远端动作被取消。工具 contract 标明取消语义，并记录 provider request id。

自动重试只用于明确 transient 且副作用幂等的错误，使用指数退避、jitter、最大次数和总 deadline。schema/permission/insufficient funds 不重试。429 尊重 `Retry-After`；重试预算独立，防止 outage 时 retry storm。

## 审批协议

审批内容示例：

~~~text
subject=user-7
task=task-9
call_id=call-3
fingerprint=sha256:...
scope=send_email:customer-17
expires_at=...
~~~

UI 显示收件人、标题、正文摘要/全文、附件、发送后果。令牌由可信服务签名并一次性消费。模型不能自己生成 `approved=true`；布尔值只适合教学 API，不适合跨服务生产授权。

## Sandbox 与秘密

代码/shell/browser 工具在最小权限 sandbox 运行：限制文件根、网络域、进程、CPU/内存/时间和输出大小。防止符号链接/path traversal、内网 SSRF、云 metadata endpoint 和命令拼接。

秘密由工具代理注入请求，不进入 prompt/observation。工具输出做 secret/PII 扫描，但不能依赖扫描作为唯一边界。高敏工具使用独立 worker 和凭据，避免一个被注入的浏览任务获得生产数据库权限。

## Audit 与 trace

每步记录 task/call、主体、模型/prompt/policy/tool version、原始 proposal、规范化参数 hash、授权决定、approval、claim、外部 request id、结果 hash、耗时、费用和状态转换。敏感 payload 分离加密并按 retention 删除。

审计日志防篡改、按需检索且不允许 Agent 自己删除。在线指标不把 call id 做 label；trace 中保存高基数细节。

## 可运行代码

~~~python
outcome = runtime.execute(call, approved=True)
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
- 无审批写操作不执行；审批过期/参数变化失败；
- 两进程同时 claim 只有一个获胜；
- handler 前失败、执行中超时、远端成功本地写失败；
- abandoned/compensated 后 late completion 被拒绝；
- 429/5xx 重试上限和 non-retryable 错误；
- path/URL/SQL 注入、跨 tenant 资源、恶意工具输出；
- crash/restart 后状态与 artifact 可恢复。

## 面试追问

**数据库有唯一 call_id，为什么还会重复付款？** 唯一约束只保证一个本地执行者，不能原子覆盖远端付款与本地 completed 写入之间的崩溃窗口；需要远端幂等或 reconciliation。

**补偿是否等于回滚？** 不是。退款是新业务动作，可能失败、产生费用或无法撤回已传播信息。Saga 记录每步和补偿结果，不能假设恢复到完全未发生状态。

**为什么旧 call id 不能在 abandoned 后重试？** 旧 ID 的审计含义是“那一次不确定动作”。复用会混淆两次授权和远端结果；新尝试必须有新身份与审批，同时关联原事件。
