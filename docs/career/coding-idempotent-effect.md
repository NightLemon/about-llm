# 编码题 6：幂等的工具执行

**练习导航**：[返回编码轮索引](coding-round.md) · [Agent 任务生命周期](../applications/agent-task-lifecycle.md) · [Safe Agent](../practice/projects/safe-agent.md)
{ .doc-nav }

本页是一道可以独立计时、实现和验收的练习。开始前先写清输入输出、错误契约和至少两个边界例。

**题面**：Agent 要调用外部 API 执行退款。同一笔退款可能因为重试、崩溃恢复或用户重复点击
被触发多次。设计执行层，使本地记录可恢复，并让支持稳定幂等键的外部 API 将重复投递合并为一次业务效果。

**必须先问清楚的三件事**：

- 幂等键由谁生成？可以由可信服务为逻辑动作生成一次 UUID 并持久化，也可以由稳定业务 ID 派生；重试必须复用它，不能每个 attempt 新建 UUID。
- 外部 API 是否支持幂等键？如果不支持，能否查询"这笔退款是否已完成"？
- 请求超时后，远端结果是**未知**而不是失败——业务上允许悬挂多久？

**可运行的最小闭环**：这题不能只靠一段没有 storage、exception 和并发语义的伪代码练习。
仓库的 `projects/safe-agent/refund_lifecycle.py` 用临时 SQLite 数据库运行完整的固定退款案例：
先原子记录待处理 action；provider 在第一次投递后故意丢失响应；重启后的 runtime 用持久化 pending
记录围栏住重投；最后用同一个稳定 key 查询并核对回执。直接运行：

```powershell
.venv\Scripts\python.exe projects/safe-agent/refund_lifecycle.py
```

该入口需要已安装项目的 `agents` optional dependency；本仓库的 `.venv` 已提供该环境。

输出中的 `stages.execution` 应显示 `provider_request_attempts: 1`、`provider_effect_count: 1` 和
`local_ledger_state: "pending"`；`stages.idempotency.handler_attempted_on_replay` 应为 `false`；
对账后的 `stages.recovery` 仍保持一次 provider effect。这个脚本实际执行 SQLite claim/reconciliation、
稳定 `CALL_ID` 作为 provider idempotency key，以及“远端成功后 timeout”的路径；它只使用模拟 provider，
不证明真实网络或支付服务的 exactly-once。

第一次接受一个逻辑动作时，要把键与这次执行的参数、主体和资源等信息绑定并保存。重试复用原来的键；
若同一个键带来不同的执行内容，应拒绝冲突，不能直接重放旧结果。实现前可先读
[先 claim，再调用支付服务](../practice/projects/safe-agent.md#claim)，把本地原子登记与外部调用分开画出来。

你自己的实现也应至少表达三态：`succeeded`、`failed` 和 **`in_flight`（结果未知）**。
把超时当成失败直接重试，是这类系统最常见的重复扣款来源。`in_flight` 必须由对账流程
（查询远端状态或人工介入）推进，不能由重试自动清除。

本地状态机只控制何时投递，无法跨越“远端已成功、本地还没写入 `succeeded`”的崩溃窗口。
因此每次投递都要携带同一个稳定的 provider idempotency key；若 provider 不支持，就要以可查询的业务回执
对账。事务发件箱保证本地业务状态与待投递记录原子提交，投递语义仍是 at-least-once。
它处理的崩溃窗口与 pending 对账不同，具体时间线见 [Outbox 说明](../practice/projects/safe-agent.md#outbox)。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 崩溃后的重复投递 | 发送可能发生两次，业务效果被重复执行 | 模拟远端成功后本地崩溃；断言两次发送使用同一幂等键，provider 只保留一笔退款 |
| 超时后重试 | 把 unknown 当 failed，重复扣款 | 模拟远端成功后 timeout，断言重启后不再投递，先查询并对账 |
| 写账本后崩溃 | 恢复时状态丢失 | 在 claim 与 provider 回执之间注入崩溃，断言恢复后仍进入对账 |
| 审批参数被改 | 用旧审批执行了新参数 | 改变参数后断言指纹不匹配，拒绝执行 |
| 并发同键 | 两个 worker 同时执行 | 断言只有一个成功获取执行权 |

**面试官的下一个追问**：

- *事务发件箱能保证 exactly-once 吗？* 不能。它只保证本地状态与待发送记录原子提交，
  投递仍是 at-least-once；远端成功但本地确认丢失时仍会重发。
  见[核心题第 17 题](interview-questions.md#effect-idempotency)。
- *参数做成 SHA-256 指纹后是否就安全了？* 无密钥哈希只能发现漂移，
  不认证执行者、时间或来源。

**仓库参考实现**：`projects/safe-agent/refund_lifecycle.py` 展示可运行的 SQLite pending/reconciliation
闭环；`src/about_llm/agents/outbox.py` 的 `SQLiteTransactionalOutbox` 展示多 worker 的租约、过期回收和
attempt 计数。测试见 `tests/test_agent_outbox.py`。

```bash
.venv\Scripts\python.exe -m pytest tests/test_agent_outbox.py -q
```

## 完成信号

在不查看参考实现的情况下重新写一遍，并能解释复杂度、失败行为、一个反例以及测试为什么能推翻错误实现。
