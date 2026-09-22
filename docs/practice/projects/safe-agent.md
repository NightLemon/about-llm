# Safe Agent：把提案接成可恢复的副作用

**项目导航**：[项目索引](../project-index.md) · [退款生命周期](../../applications/agent-task-lifecycle.md) ·
[Agent Runtime](../../applications/agent-runtime.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/safe-agent/README.md) ·
[项目证据](../../evidence/project-controls.md)
{ .doc-nav }

生命周期页负责解释一笔退款，Runtime 页负责可复用原语。本项目只回答怎样把 proposal、policy、approval、ledger、
handler、verifier 和 recovery 组装成可交付系统。

## 第一次只跑这笔退款 {#refund-lifecycle}

```powershell
python projects/safe-agent/refund_lifecycle.py
```

先预测九个阶段的状态，再对照输出。固定模拟器应出现：跨租户请求在 handler 前被拒绝；合法退款在远端受理、响应丢失后
保持 `pending`；同 call ID 重放不再次执行 handler；独立查询与字段核对后，恢复重放读取已确认结果。

完整安装、脚本和测试命令见[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/safe-agent/README.md)。

## 项目组装顺序

| 阶段 | 要接入的组件 | 升级条件 |
|---|---|---|
| 1. Proposal | Typed observation、closed schema、稳定 tool/version | 模型字段不能伪造 subject、tenant 或 capability |
| 2. Authorization | 服务端解析资源、ACL、policy revision | 拒绝路径中 handler 调用数为 0 |
| 3. Approval | 绑定 subject/task/resource/tool/args/policy 的 execution fingerprint | 任一字段漂移都会让旧批准失效 |
| 4. Claim {#claim} | 持久 ledger 在调用远端前领取执行权 | 并发与重放不能越过 pending fence |
| 5. Effect | Provider idempotency key 与原始 attempt 记录 | 本地 timeout 不被误写为远端失败 |
| 6. Verification | 独立查询并核对资源、金额、reason 与 receipt | 不能仅凭 `accepted` 字样完成 |
| 7. Recovery | Reconcile 后写入终态；返回 cached 前重新授权 | 撤权主体不能读取旧结果 |

每层只使用前一层的有类型产物。模型给出正确工具名也不能跳过授权；SQLite 原子 claim 也不能证明远端 exactly-once。

### 先 claim，再调用远端 {#claim}

运行时必须先持久化调用身份并取得执行权，再进入 handler。若 claim 已是 `pending`，恢复流程先查询远端事实，不能把
同一个动作当作一次全新调用。

## 手动处理一条 pending 记录 {#run}

把 `pending` 看成需要操作的业务状态，而不是“失败”的别名。操作人员应按 call/execution/idempotency identity 查询远端，
保存原始回执，由 verifier 比对预期效果，再选择 reconcile、继续等待或升级人工。不要因本地进程重启或 TTL 到期自动重放。

本地状态、provider 请求和最终用户投影是三份不同记录。真实接入前还要验证支付服务身份、幂等范围、签名、账务和
多节点竞争。

## Outbox 处理另一段崩溃窗口 {#outbox}

事务发件箱把业务状态与“待发送消息”写入同一本地事务，再由 worker 至少一次投递。它缩小数据库提交与消息发布之间的
窗口，却仍可能在远端成功、ack 写回前重复发送。Provider 是否按幂等键去重、以及效果怎样对账，仍需独立证据。

## 框架只是接线层 {#framework-tool-adapters}

LangChain、LlamaIndex 或其他框架 adapter 只负责把框架对象映射到同一个可信 Runtime。验收时检查：原始 proposal、
closed schema、trusted context、policy/approval decision、handler attempt 和 verifier receipt 是否完整保留；框架默认值
不能绕过控制面。

### 决策理论只选择允许动作 {#decision-theory-control}

Belief update、信息价值和停止条件可以帮助规划器选择下一步，但 hard policy 先形成允许动作集合。固定二状态实验见
[Agent 决策理论](../../applications/agent-decision-theory.md)，不要把更高 expected utility 当成授权。

### MCP 与 A2A 只改变跨系统接线 {#mcp-a2a}

协议验证负责生命周期、schema 和 transport；业务 Runtime 继续负责主体、租户、审批、副作用和验证。
完整互操作路线见[MCP 与 A2A](../../applications/agent-interoperability.md)。

## 评测时分开任务与副作用 {#evaluation}

至少分别记录 task outcome、policy decision、approval、handler attempted、effect observed、verification 与 recovery。
“回答正确”不能覆盖越权动作；“没有返回成功”也不能推出 effect 未发生。评测定义见
[Agent 评测](../../quality/agent-evaluation.md)。

## 交付检查

- 负例在 handler 前停止，并保存明确 reason code。
- Approval 与完整 execution identity 绑定，漂移会失效。
- Claim 在 handler 前持久化，pending 重放不会再次进入 handler。
- Verifier 使用独立业务查询，字段不完整或不匹配时不完成。
- Cache replay 重新授权；撤权后不返回历史 payload。
- Outbox、framework、MCP/A2A 和决策实验分别标明证据层级。
- 报告明确固定模拟器、SQLite 和 loopback 不能证明真实 IAM、provider 或 exactly-once。

完成后应交付状态图、失败时间线、运行配置、原始结果和恢复说明，而不是重复生命周期教材或整份 README。
