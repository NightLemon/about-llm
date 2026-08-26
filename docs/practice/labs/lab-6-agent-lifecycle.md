# 实验 6：追踪一次 Agent 退款

我们跟随一笔 300 元退款，观察它从“读到订单”走到“核对退款结果”的全过程：

1. 把 observation 变成模型给出的 proposal；
2. 用 schema、ACL 和 approval 判断这项提议能否执行；
3. 在 execution 之后，用 idempotency、verifier 和 recovery 收口副作用。

实验使用固定输入，不调用真实模型或支付服务。重点是解释：远端已经受理请求但本地超时时，为什么不能直接重试。

**相关教材**：[Agent 总览](../../applications/agents.md) ·
[一次 Agent 退款任务](../../applications/agent-task-lifecycle.md) ·
[Safe Agent 项目](../projects/safe-agent.md)
{ .doc-nav }

## 完成标准

完成后，你应该能不看输出回答：

1. 哪些字段来自用户或 Planner，哪些字段只能来自可信控制面。
2. Schema 通过后，为什么仍可能被 ACL 拒绝。
3. 为什么审批必须绑定 execution fingerprint，而不是一句“确认退款”。
4. 远端已受理、本地超时时，ledger 为什么必须保持 `pending`。
5. 为什么重放没有再次调用 provider，最终却能恢复成 `cached`。
6. 这个实验为什么不能证明 exactly-once 或生产安全。

预计时间为 45–90 分钟。

## 准备环境

从仓库根目录安装 Agent 依赖：

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
~~~

本实验使用：

```text
projects/safe-agent/refund_lifecycle.py
src/about_llm/agents/runtime.py
src/about_llm/agents/sqlite_ledger.py
tests/test_agent_refund_lifecycle.py
```

脚本每次创建临时 SQLite 数据库，不会调用网络、写入真实订单或留下待清理的退款。

## 第一步：先预测九个阶段

先不要运行脚本。已知条件如下：

```text
authenticated subject = user-42
tenant = tenant-shop-a
capability = refund:request
order-1001 owner = user-42
order-1001 paid = 30000 cents
requested refund = 30000 cents
```

退款服务的固定行为是：先创建 refund receipt，再让第一次响应超时。

填写预测：

| 阶段 | 你预测的状态 | Handler 会运行吗 | 能否报告完成 |
|---|---|---:|---:|
| Schema |  |  |  |
| ACL |  |  |  |
| Approval 前 |  |  |  |
| 第一次 execution |  |  |  |
| 同 call ID replay |  |  |  |
| Provider verifier |  |  |  |
| Reconciliation 后 replay |  |  |  |

不要把“函数抛异常”和“外部 effect 没发生”写成同一个判断。

## 第二步：运行完整 walkthrough

~~~powershell
python projects/safe-agent/refund_lifecycle.py
~~~

输出按 `stages` 排列。先只记录状态，不急着读 fingerprint：

```text
proposal.tool_name
schema.valid_proposal_accepted
acl.authorized_proposal.status
approval.call_id
execution.status
execution.local_ledger_state
idempotency.handler_attempted_on_replay
verifier.status
recovery.replay_after_reconciliation.status
```

预期主线为：

```text
proposal -> schema accepted -> ACL allow -> needs_approval
-> handler attempted -> timeout/pending -> replay fenced
-> provider query passed -> externally_confirmed -> cached
```

`failed -> pending -> passed -> cached` 不是互相矛盾。四个状态分别属于本地调用、ledger、外部事实验证和恢复后重放。

## 第三步：画出数据来源

把 `observation` 与 `proposal` 中的字段放入下表：

| 字段 | 用户/模型可提议 | 可信服务注入 | 说明 |
|---|---:|---:|---|
| `reason` | 是 | 否 | 仍需 schema 和业务规则 |
| `amount_cents` | 是 | 否 | 不能超过服务端订单余额 |
| `subject_id` | 否 | 是 | 来自认证 session/token |
| `tenant_id` | 否 | 是 | 不能由请求 body 自报 |
| `capabilities` | 否 | 是 | 来自授权系统 |
| `call_id` | 否 | 是 | 由 orchestrator 分配逻辑 identity |
| order owner/version | 否 | 是 | 从订单 source of truth 解析 |
| approval | 否 | 是 | 由可信确认流程签发 |

脚本的 `proposal.model_did_not_supply` 应与这张表一致。

## 第四步：观察 closed schema 的边界

正常 proposal 只有：

```json
{
  "order_id": "order-1001",
  "amount_cents": 30000,
  "reason": "item_damaged"
}
```

内置负例额外加入 `tenant_id=tenant-shop-b`。找到：

```text
schema.closed_schema_negative_control.rejected = true
schema.closed_schema_negative_control.keyword = additionalProperties
```

回答两个问题：

1. 为什么不能简单允许未知字段，然后在 Handler 中忽略？
2. 为什么 `amount_cents` 类型和范围都合法，仍不能说明该订单可退 300 元？

Schema 负责结构契约；存在性、归属、余额、状态和频率是资源与业务 policy。

## 第五步：观察 ACL 发生在副作用之前

正常订单通过 tenant、owner 和 exact capability 检查，然后停在：

```text
acl.authorized_proposal.status = needs_approval
```

内置负例把 `order_id` 换成 `tenant-shop-b` 的 `order-9001`。预期：

```text
status = policy_denied
policy_reason = tenant_mismatch
provider_attempts_after_acl = 0
```

最后一个断言最重要。若系统先调用 provider、再过滤响应，即使用户没看见结果，副作用也已经越权发生。

## 第六步：解释 approval 绑定

比较：

```text
acl.authorized_proposal.execution_fingerprint
approval.execution_fingerprint
```

两者必须相同。然后思考三个漂移：

- Planner 把金额从 300 元改成 299 元；
- 订单版本从 `order@7` 变成 `order@8`；
- Policy 从 `refund-acl@v1` 升级并改变判定。

这些变化都不应继续使用旧 approval。用户确认的是一个规范化、已授权的具体动作，不是一张永久“允许退款”通行证。

脚本已经内置第一个负例。检查：

```text
approval.drifted_amount_negative_control.status = approval_rejected
approval.drifted_amount_negative_control.message = ... approval_execution_mismatch
approval.provider_attempts_after_drift = 0
```

注意：这个固定样例里的 grant 没有数字签名。字段匹配只能检查绑定关系，不能认证工件是谁签发的。

## 第七步：区分 attempt、effect 与 completion

第一次执行的观察是：

```text
execution.handler_attempted = true
execution.status = failed
execution.local_ledger_state = pending
execution.provider_request_attempts = 1
execution.provider_effect_count = 1
```

按时间顺序解释：

1. Runtime 先 claim 本地 call；
2. Handler 调用退款服务；
3. Provider 保存 accepted receipt；
4. 响应丢失；
5. 本地没有足够证据 complete ledger。

此时正确用户答复应是“退款状态正在核对”，而不是成功或失败。真实系统还应给出可追踪 task ID 与后续处理方式。

## 第八步：确认重放没有制造第二笔退款

找到 `idempotency`：

```text
status = failed
handler_attempted_on_replay = false
provider_request_attempts = 1
provider_effect_count = 1
```

Runtime 看到 pending claim 后停止。它没有因为“上次函数报错”就重新执行 Handler。

这个结果只说明当前 SQLite 固定故障样例中的 replay fence 按设计工作。它没有覆盖：

- provider 一定实现 idempotency；
- 任意多进程/多区域竞争都只产生一次 effect；
- 本地 ledger 与 provider 共享事务；
- 所有 crash window 都已经覆盖。

## 第九步：让 verifier 和 recovery 收尾

Verifier 不读取 Planner 的自述，而是按 trusted idempotency key 查询 provider，并逐项核对：

```text
order_id
amount_cents
reason
provider_status
idempotency_key
```

通过后，reconciliation 记录：

```text
resolution = externally_confirmed
note = provider audit query confirmed the accepted refund
```

再查看两个负例：

```text
verifier.mismatched_receipt_negative_control.status = failed
recovery.revoked_replay_negative_control.status = policy_denied
```

前者说明 `accepted` 字样不能覆盖金额不匹配；后者说明 reconciliation 后的 cache replay 仍会重新授权。

恢复后的 replay 为 `cached`，provider effect count 仍为 1。现在才有依据组织最终用户答复。

回答：如果 provider 只返回 `{"status": "accepted"}`，为什么 verifier 仍不应通过？

## 第十步：运行高风险回归

~~~powershell
python -m pytest tests/test_agent_refund_lifecycle.py -q
~~~

这两个测试只锁定本章最关键的判定条件：

- closed schema 和跨 tenant 请求在 Handler 前被拒绝；
- pending replay 不重复调用 provider；
- 独立查询通过后才 reconciliation；
- 恢复后结果为 cached，effect count 仍为 1；
- 报告明确否认真实模型、真实 provider 与 exactly-once 证据。

它们没有把所有 JSON 组合和实现细节都写成测试，这样教材仍可以重构，而关键教学结论有独立断言。

## 实验记录模板

```text
任务：subject、tenant、order、amount、capability
运行前预测：九个阶段的状态和 handler 是否运行
信任边界：模型提出 / 可信上下文提供 / 服务端解析
ACL 负例：资源、reason code、provider attempts
故障时间线：claim、effect、timeout、pending
重放证据：handler attempted、request attempts、effect count
Verifier：查询来源、匹配字段、verdict
Recovery：resolution、cached result、最终用户投影
证据边界：本实验未证明的模型、协议和生产结论
```

下一步再进入 [Agent Runtime](../../applications/agent-runtime.md)，理解 execution fingerprint、outbox、checkpoint
和并发控制；不要先用 LangChain、LlamaIndex 或 MCP 把这九个状态重新藏进一个 `invoke()`。
