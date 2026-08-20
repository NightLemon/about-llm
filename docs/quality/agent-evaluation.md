# Agent 评测：别只看最后一句话

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：Agent 评测、simulation、红队和发布工程师。
- **先修**：[Agent 任务主线](../applications/agent-task-lifecycle.md)与[评测方法](evaluation-methodology.md)。
- **首次阅读**：一次退款 case → 事件分母 → simulator → 故障/注入 → 发布门禁。
- **完成信号**：能同时验证任务结果、权限、副作用、预算和恢复行为。
- **卡住时**：先运行[Safe Agent 退款主线](../practice/projects/safe-agent.md#refund-lifecycle)。

</div>

仍然使用那笔 300 元退款。Agent 最后告诉用户“退款已受理”，这句话可能对应三种完全不同的过程：

1. 订单归属、审批和 Provider receipt 都正确；
2. Agent 越权读了另一个租户的订单，碰巧给出正确答案；
3. Provider 已退款，但本地 timeout 后又重试，实际产生两笔 effect。

只评最终文本会把三条轨迹都判成成功。Agent 评测必须把可见回答、环境终态和控制面事件放进同一个 case。

## 一条 Case 应包含哪些东西

```text
initial environment snapshot
+ user goal
+ trusted subject / tenant / capability
+ allowed and forbidden actions
+ hidden business state
+ completion verifier
+ step / token / time / cost budgets
+ injected untrusted payloads
+ expected approval or escalation point
```

Prompt 只是输入的一部分。Agent 的行为还依赖工具、资源版本、policy、memory 和环境状态。

Case 可以按难度递进：检索/总结、单工具读取、写操作审批、多步依赖、并行、crash recovery、歧义澄清、
无法完成时的拒答，以及提示注入。生产日志转成 case 时要脱敏、取得允许并冻结外部依赖。

## 先把一笔退款拆成四个事件

| 事件 | 证据 | 它没有自动说明什么 |
|---|---|---|
| Proposal produced | Planner trace | Schema、权限或执行通过 |
| Policy/approval allowed | 控制面 decision/grant | Handler 已调用 |
| Handler attempted | Runtime attempt event | 远端 effect 已发生 |
| Effect verified | Provider audit/业务状态/隔离环境 | 用户可见回答一定正确 |

Handler timeout 时 effect 可能已经发生。本地 `completed` 也只是 Runtime 记录；只有独立 verifier 观察到业务事实，
才能写 `effect_applied=true`。Cache replay 不是新的 handler attempt，更不是第二个 effect。

这张事件表是 Agent 指标的共同分母语言。

## 三组指标分别看什么

### Outcome：任务最终发生了什么

- Completion verifier pass rate；
- Partial goals covered；
- 业务 state 与目标是否一致；
- 用户可见回答、引用和拒答是否正确。

### Trajectory：付出了什么过程代价

- Tool/argument correctness；
- Unnecessary/repeated calls；
- First useful action、end-to-end latency；
- Steps、tokens、费用与人工介入；
- Pause/resume 与 recovery success。

开放任务可能有多条正确路线，因此不把 exact trajectory match 当通用质量指标。最终状态由 verifier 判断；
过程只检查关键 invariants，例如“发送前必须审批”和“不得读取 secret table”。

### Safety：哪些坏事绝不能发生

- Unauthorized read/write 与跨 tenant access；
- Secret/canary exfiltration；
- 未审批副作用或审批后参数漂移；
- Prompt-injection attack success；
- Over-refusal：正常任务被错误拒绝。

Safety guardrail 不被平均任务成功率抵消。一百个正常 case 成功，也不能冲掉一次跨租户写入。

## 每个率都要写 Numerator 和 Denominator

| Metric | Numerator | Denominator |
|---|---|---|
| Task success | Verifier passed cases | 有确定 verifier judgment 的 cases |
| Blocked unsafe | 被 policy 阻止的危险 proposals | Policy 明确判定为禁止的 proposals |
| Unapproved attempt | 未获有效审批却进入 handler 的 steps | 实际进入 handler 的副作用 steps |
| Duplicate effect | 同 effect ID 产生额外 verified effects | 有至少一个 verified effect 的 logical actions |

分母为 0 时报告 N/A，而不是 0%。这些率的分母不同，也不能再平均成一个“Agent 总分”。

终止状态同样要分开：`completed`、`needs_approval`、`approval_rejected`、`escalated`、budget exhausted、cycle、
planner/runtime/verifier error 各有不同含义。安全暂停不应记成任务成功，也不应和系统崩溃混在一个“正常停止”桶里。

## 运行仓库里的确定性 Gate

```powershell
python -m about_llm.agents.cli evaluate `
  --traces projects/safe-agent/trajectory.example.jsonl
```

Gate 会逐 case 检查：

- Verifier failure 或 unjudged completion；
- Policy judgment 缺失/indeterminate；
- Policy-denied proposal 仍到达 handler；
- Over-refusal；
- 未审批副作用；
- 同 effect ID 重复 applied；
- Unresolved pending；
- Recorded steps 或 handler attempts 超预算。

Step budget 与 handler-attempt budget 不能互换。只限制 handler，仍可能让模型无限生成被拒绝或 cache-hit proposals。

Trace loader 能检查类型和内部一致性，却不能证明 observation 真实或文件未被篡改。生产 recorder 应由控制面写入，
并结合签名/MAC、存储 ACL 与审计链；不要让模型自己声明 `policy_allowed=false` 或 `effect_applied=false`。

## Simulator 要像环境，不只是固定字符串

一个有用的工具 simulator 应支持：

- 可重置的业务 snapshot；
- 虚拟/可控时间；
- Stable schema 与 error codes；
- Timeout、partial result、并发写和 crash 注入；
- 与生产完全隔离的 credentials/endpoints；
- 可查询的 effect state 与 audit receipt。

每个 case 前 reset，结束后由 verifier 检查状态并销毁。Recorded replay 能稳定外部 observation，但网页/API response
会过时，所以还要保存 capture time 与 content identity。

比较模型版本时使用同一批 observations，避免外部世界变化冒充模型差异。

## 故障注入围绕“不确定窗口”设计

系统性覆盖：

```text
connect 前失败
write/read timeout，远端未执行 / 已执行
429 + Retry-After
partial result / malformed body
stale ETag / concurrent mutation
ledger write failure / disk full
provider success、local ack 前 crash
process restart / user cancellation
```

每个 fault 都检查状态机没有进入非法状态，恢复也不会重复 effect。先做 unit state transitions，再做 simulator
integration，最后才在隔离 staging 演练；不要从线上 chaos 开始。

## Outbox 评测要区分 Request 和 Effect

记录 enqueue→claim、claim→ack 与 end-to-end delivery latency，以及 attempts、lease expiry/redelivery、dead-letter age、
stale ack rejection。

同一 `effect_id` 的 Provider request count 大于 1，在 at-least-once delivery 中可能是预期；verified effect count
大于 1 才表示幂等失败或 identity 漂移。

必须保留“Provider success、ack 前 crash”case，并断言重投复用相同 idempotency key。SQLite + simulated Provider
通过只说明本地状态机正确，不能证明真实 Provider receipt、broker 或 exactly-once external effect。

## Recovery 有自己的指标

至少记录：

- Pause→resume success；
- 重复 Planner/usage 次数；
- 恢复后 handler budget 是否被重置；
- Identity/policy drift 是否误放行；
- 过期 grant 是否仍触达 handler；
- Checkpoint/ledger 不一致；
- Pending age 与 recovery latency。

Checkpoint 自带 hash 的通过率不是安全指标。能修改文件的主体通常也能重算 unkeyed hash；真实性需要可信 recorder、
MAC/signature 和存储控制。

## Prompt Injection 红队看真实行为

攻击载荷放进网页、邮件、附件、tool errors、memory 与其他 Agent messages，覆盖直接覆盖指令、伪 system 标签、
编码/多语言、exfiltration URL、tool induction 与持久 memory poisoning。

判定依据是 Agent 是否访问禁止资源、泄漏 canary、绕过审批或产生 effect。回复“我不会执行”不构成安全证据，
因为动作可能已经发生。

同时加入包含正常命令示例的 benign documents，测量过度拒绝。

## 多 Agent 先和单 Agent 比

除 task success 外，测 handoff correctness、message loss/misunderstanding、duplicate work、conflict resolution、
shared-state consistency 与单个 Agent 故障恢复。

如果多 Agent 只增加 token 与协调失败，没有带来质量、权限隔离、并行或独立验证收益，就保留单 Agent baseline。
生成与评审使用同一家模型时错误可能高度相关，仍需 deterministic tests、独立数据/模型或人工抽样。

## 统计比较与线上监控

离线固定 case IDs 做 paired comparison，报告 effect、interval 与关键 slices。高方差任务使用多次运行，并把 pass@1、
oracle/pass@k 和实际 selected@k 分开；更大的 k 同时增加成本和 verifier 攻击面。

保留 private holdout、参数化环境和定期新建 cases，降低 benchmark 污染。

线上通常没有完整 gold，可监控 completion/cancel/escalation、人工修正、重复调用、approval rejection、pending age、
tool errors、steps/tokens/cost 与安全事件，再用抽样审计补充。按 tool、task type、tenant tier、model/policy version
切片，避免总体平均掩盖单工具循环。

## 发布门禁

一份常见 gate 可以是：

1. 权限和 duplicate-effect cases 零违规；
2. 关键任务相对 baseline 不劣，区间满足预设阈值；
3. 风险、语言和工具 slices 不越界；
4. P95 steps、latency 与 cost 在预算内；
5. Pause/resume 与 reconciliation cases 全通过；
6. 新版本先 shadow，再进入小流量 canary。

回滚需要恢复兼容的 model、Prompt、tool schema、policy、memory/index 组合，而不是只把 model ID 改回去。

## 自测

1. Handler timeout 时，为什么 `handler_attempted` 与 `effect_applied` 不能合成一个事件？
2. 怎样评测发邮件 Agent 而不真的发送？请写 simulator state 与 verifier。
3. 为什么 exact trajectory match 会惩罚有效替代方案？
4. Provider request count=2、verified effect count=1 时，系统一定失败了吗？
5. 模型升级后最终文本相似，还应比较哪些控制面事件？
