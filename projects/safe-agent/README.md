# Safe Agent：从模型提案到可恢复的副作用

这个项目用一笔 300 元退款说明 Agent 为什么不能“让模型直接调用工具”。任务会从观察和动作提案开始，经过参数校验、
权限与审批，再进入执行、幂等保护、结果核验和故障恢复。任何一步失败，都必须留下明确终态。

第一次学习请从[项目教学页](../../docs/practice/projects/safe-agent.md)开始。那里按一笔真实任务解释完整链路；本页只保留
快速运行、脚本索引和排错信息。

## 第一次只跑这笔退款 { #refund-lifecycle }

```powershell
python projects/safe-agent/refund_lifecycle.py
```

运行前先预测：第一次 provider 已受理退款但响应丢失后，runtime 应该重试、标记失败，还是进入 pending？再次处理时，
怎样确认不会多退一次？

输出会把同一任务分成以下阶段：

```text
observation
→ typed proposal
→ closed schema
→ resource ACL
→ approval
→ execution claim
→ provider attempt
→ pending / reconciliation
→ receipt verifier
→ completed
```

重点看调用 ID、提案指纹、执行指纹、provider 幂等键和可信回执怎样贯穿前后。
完整逐步解释见[第一次只跑这笔退款](../../docs/practice/projects/safe-agent.md#refund-lifecycle)。

## 这条主线必须保持什么

- 模型只能提出动作，不能决定自己的权限、审批状态或执行结果。
- 工具参数先通过封闭 schema，再由服务端解析真实资源并执行 ACL。
- 有副作用的调用在执行前必须拿到与当前动作绑定的审批。
- 相同 `call_id` 只能对应同一执行身份；不同身份不能复用旧结果。
- 超时或断连后，如果外部状态未知，就进入 pending，而不是盲目重试。
- 最终 assistant 文本不能证明任务成功；verifier 必须读取可信 receipt 或业务状态。

这些不变量对应的实现与反例见[Agent 任务生命周期](../../docs/applications/agent-task-lifecycle.md)。

## 运行故障与恢复路径

`scenario.example.jsonl` 使用三个内置离线工具，不发送邮件、不访问网络，也不产生真实退款。为每次实验选择一个新的
SQLite 路径，以便保留旧记录：

```powershell
python -m about_llm.agents.cli run `
  --scenario projects/safe-agent/scenario.example.jsonl `
  --ledger artifacts/agent/scenario-001.db `
  --max-tool-calls 5

python -m about_llm.agents.cli pending `
  --ledger artifacts/agent/scenario-001.db `
  --older-than-seconds 0
```

样例中的 `uncertain-1` 会保持 pending。真实系统必须先查询 provider audit、业务数据库或 outbox，再选择处理方式。
下面的 `abandoned` 只适用于这个没有外部副作用的离线样例：

```powershell
python -m about_llm.agents.cli resolve `
  --ledger artifacts/agent/scenario-001.db `
  --call-id uncertain-1 `
  --resolution abandoned `
  --note "offline example verified: no external operation exists"

python -m about_llm.agents.cli inspect `
  --ledger artifacts/agent/scenario-001.db `
  --call-id uncertain-1
```

如果外部动作已经完成，使用 `external` 并提供可信结果；如果已经执行逆向操作，使用 `compensated`。三种处理都会保留
原 call ID，旧调用不会重新执行。

## Planner loop 与审批暂停

下面的离线 planner cases 展示正常完成、重复动作、短周期循环、连续策略错误和等待审批：

```powershell
python -m about_llm.agents.cli loop `
  --cases projects/safe-agent/loop.example.jsonl
```

要观察审批前暂停和跨进程恢复，使用全新 ledger 与 checkpoint 路径：

```powershell
python -m about_llm.agents.cli pause-loop `
  --cases projects/safe-agent/loop.example.jsonl `
  --case-id approval-pause `
  --ledger artifacts/agent/loop-001.db `
  --checkpoint artifacts/agent/loop-001.checkpoint.json

python -m about_llm.agents.cli resume-loop `
  --cases projects/safe-agent/loop.example.jsonl `
  --case-id approval-pause `
  --ledger artifacts/agent/loop-001.db `
  --checkpoint artifacts/agent/loop-001.checkpoint.json
```

恢复时会重新授权原 decision，并保留此前的预算和事件；不会再次向 planner 计入相同 token/cost。这里的审批和工具仍是
离线模拟，生产系统还需要验证 approver 身份、签名、一次性消费、过期时间和会话绑定。

## 根据当前问题选择脚本

先安装 Agent 依赖：

```powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
```

| 你想理解什么 | 入口 |
|---|---|
| 一笔副作用任务怎样完成并恢复 | `refund_lifecycle.py` |
| Schema、ACL、审批、幂等与 pending | `python -m about_llm.agents.cli run/pending/resolve` |
| Planner 预算、循环检测和 checkpoint | `python -m about_llm.agents.cli loop/pause-loop/resume-loop` |
| 模型文本如何变成 typed proposal | `model_planner_control.py` |
| LangChain/LlamaIndex 工具怎样接入同一 runtime | `framework_tool_adapter_control.py` |
| 两个框架的 model→tool→model loop | `framework_agent_loop_control.py` |
| MCP 官方 SDK 的内存、stdio 与 HTTP transport | `mcp_sdk_*_control.py` |
| 手写协议 parser 与官方 SDK 有什么边界 | `mcp_stdio_control.py`、`mcp_streamable_http_control.py` |
| A2A Agent Card、message 与 task round trip | `a2a_loopback_control.py` |
| Outbox 在 ack-before-crash 后怎样安全重投 | `outbox_demo.py` |
| Observation 的信息价值与 hard constraint | `decision_theory_toy.py` |
| Recorded trajectory 怎样进入发布 gate | `python -m about_llm.agents.cli evaluate ...` |

框架、MCP 和 A2A 的学习顺序见[项目路线 D–G](../../docs/practice/projects/safe-agent.md#framework-tool-adapters)。
精确输入、结果和适用范围保存在[项目实验台账](../../docs/evidence/project-controls.md)，避免把 transport round trip
误写成业务权限、安全或生产互操作已经完成。

## 主要输入与输出

| 文件或目录 | 用途 |
|---|---|
| `scenario.example.jsonl` | ACL、审批、cache、跨租户与 pending 的离线场景 |
| `loop.example.jsonl` | Planner 完成、循环、预算和审批暂停场景 |
| `trajectory.example.jsonl` | Task success、policy、effect 与 pending 分开记录的轨迹样例 |
| `artifacts/agent/*.db` | SQLite 调用账本；保留 claim、attempt、pending 和 resolution |
| `artifacts/agent/*.checkpoint.json` | Planner 暂停时的预算、事件和 pending decision |

这些样例中的认证上下文、审批 grant、provider receipt 和 usage 都由仓库离线准备。接入真实系统时，必须替换为可信
控制面输入，并为敏感 trace 设置访问控制、加密和保留期限。

## 常见故障

| 现象 | 先检查 |
|---|---|
| 模型输出合法 JSON，工具仍被拒绝 | Runtime schema、服务端资源解析、ACL 和审批是独立检查 |
| 同一动作执行了两次 | Call ID、execution fingerprint、claim 唯一约束和 provider idempotency key |
| 请求超时后不知道是否成功 | 不要重放；进入 pending，并查询 provider 或业务状态 |
| Approval 通过后参数发生变化 | Grant 是否绑定 proposal/execution fingerprint，变化后应重新审批 |
| Cache 命中绕过了新权限 | Replay 前是否重新检查当前 tenant、ACL 和 policy revision |
| Agent 文本说“完成”，评测却失败 | Verifier 应读取 receipt/effect，而不是相信 assistant 文本 |
| Loop 一直消耗 token | Step/token/cost/time 预算，以及重复 action/error 和 cycle 检测 |
| Resume 后重复向 planner 计费 | Checkpoint 是否保存 pending decision、累计 usage 和 planner history |
| MCP/A2A 调通但业务仍不安全 | Transport schema 不负责认证、授权、审批、幂等和 effect verification |
| Outbox 重投产生重复副作用 | Provider 是否真正支持并遵守同一个幂等键 |

## 运行检查

核心 runtime、策略、恢复与评测：

```powershell
python -m pytest `
  tests/test_agent_refund_lifecycle.py `
  tests/test_agent_runtime.py `
  tests/test_agent_policy.py `
  tests/test_agent_schema.py `
  tests/test_agent_loop.py `
  tests/test_sqlite_agent_ledger.py `
  tests/test_agent_evaluation.py -q
```

协议与框架适配：

```powershell
python -m pytest `
  tests/test_agent_framework_tool_adapters.py `
  tests/test_mcp_sdk_memory.py `
  tests/test_mcp_sdk_stdio.py `
  tests/test_mcp_sdk_streamable_http.py `
  tests/test_a2a_loopback.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

默认项目不会调用真实模型、远程工具或外部副作用。要把它升级为生产 Agent，还需要接入真实身份系统、审批服务、
持久化密钥/令牌、provider 状态查询、并发 lease、监控告警和人工恢复 runbook。
