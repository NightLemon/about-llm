# Agent 评测、仿真与红队

Agent 的最终回答可能看起来正确，但中间越权读取、重复副作用或浪费百次调用。评测必须同时覆盖 outcome、trajectory、policy、恢复和成本，并用可重置环境避免回归测试触发真实影响。

## 任务 case

每个 case 包含初始环境 snapshot、用户目标、允许/禁止动作、隐藏状态、完成 verifier、预算、注入 payload 和期望升级点。不要只保存 prompt；Agent 行为依赖工具和环境。

按能力分层：检索/总结、单工具读、单写审批、多步依赖、并行、恢复、歧义澄清、无解拒绝和对抗任务。真实日志可转 case，但要脱敏、获得许可并冻结依赖。

## 指标

### Outcome

- task success：确定性 verifier 是否满足；
- partial credit：子目标覆盖；
- side-effect correctness：真实状态是否符合目标；
- answer correctness/引用：用户可见输出质量。

### Trajectory

- 工具选择和参数准确率；
- unnecessary call、重复 call、步骤数；
- 首次有效动作时间、总延迟、token/费用；
- recovery success 和人工介入率。

### Safety

- unauthorized read/write、跨 tenant、秘密泄漏；
- 未审批副作用、审批参数漂移；
- injection attack success rate；
- over-refusal：正常任务被错误拒绝。

安全指标是 guardrail，不能由更高平均成功率抵消。

### 分母与事件语义

每个率都要写 numerator/denominator，分母为零报告 N/A，而不是 0%。例如 task success 的分母只能是有确定 verifier judgment 的 case；blocked-unsafe rate 的分母是 policy 明确判为禁止的 proposal；unapproved-attempt rate 的分母是实际进入 handler 的有副作用步骤。不同分母的率不能求一个“Agent 总分”。

必须区分四个事件：proposal 产生、policy/approval 允许、handler 被调用、外部 effect 被验证发生。`handler_attempted=true` 不证明远端动作发生；handler timeout 可能发生也可能没发生。反过来，本地 `completed` 也只是 runtime 记录，只有隔离环境状态、provider audit log、outbox/业务库等 verifier 才能写 `effect_applied=true`。缓存 replay 不是新 handler attempt，更不是第二次 effect。

本仓库 `about-llm-agent evaluate` 对冻结的 JSONL trace 做确定性 gate：task verifier failure/unjudged、policy judgment 缺失/indeterminate、policy-denied proposal 到达 handler、policy 误拦允许动作（over-refusal）、未审批副作用 attempt、相同 effect id 重复 applied、unresolved pending、总 recorded step 或 handler-attempt budget 超限任一存在即失败，并输出逐 case findings。两种预算不能互换：只限制 handler 不能阻止模型循环产生被拒绝或缓存 proposal。trace 同时保存 proposal/execution fingerprint，并记录 environment、policy、verifier 版本；但 loader 只能检查类型与内部一致性，不能证明 supplied observation 真实或 trace 未被篡改。生产 recorder 应由控制面写入并签名，不能让模型自报 `policy_allowed=false` 或 `effect_applied=false`。

运行时终止状态也要进入分母账本。`completed` 仅指 completion verifier passed；`needs_approval` 是安全暂停，`approval_rejected` 是授权失败，`escalated` 是转交，step/token/cost/wall-time、repeated action/cycle/error 以及 planner/runtime/verifier error 都是不成功终态，不能合并为“正常停止”。离线 `loop` fixture 可回归这些分支，并做 checkpoint JSON/SQLite restart；但它用 scripted decisions、supplied usage、unsigned grant 和 local exact verifier，不等于真实模型成功率、provider 计费、签名审批、开放任务 judge 或分布式恢复实证。

恢复专项指标至少包括 pause→resume 成功率、重复 planner/usage 数、恢复后 handler budget reset 数、identity/policy 漂移误放行、过期 grant handler attempt、checkpoint/ledger 不一致和恢复延迟。checkpoint 自带 hash 的通过率不是安全指标：能修改文件的攻击者也能重算无密钥 hash；需要可信 recorder、签名/MAC、存储 ACL 和审计链。

Outbox 专项要分别统计 enqueue→claim、claim→ack 和 end-to-end delivery latency，attempt distribution、lease expiry/redelivery、retry reason、dead-letter age、stale ack rejection，以及同一 `effect_id` 在 provider 侧的 request count 与 verified effect count。前者大于一在 at-least-once 系统中可以是预期行为；verified effect count 大于一才是幂等失效或身份漂移。测试必须包含 provider success 后 ack 前 crash，并断言重投沿用同一 idempotency key。SQLite + 模拟 provider 的通过只证明本地状态机，不证明真实 provider receipt、broker、网络或 exactly-once external effect。

## 最终状态优先，轨迹谨慎

开放任务可能有多条正确路径，逐 token 匹配参考轨迹会惩罚有效替代方案。用状态 verifier 判断结果，用 invariant 检查关键过程，例如“发送前必须审批”“不得读取 secret 表”。只有工具选择本身是能力目标时才比较轨迹。

LLM judge 可评开放输出，但不能可靠判断隐藏副作用；环境状态、ledger 和 policy event 应使用确定性检查。

## 模拟工具

模拟器需要可控时间、错误注入、并发和状态，而不只是返回固定字符串。它实现与生产相同 schema/error code，但凭据和 endpoint 完全隔离。每个 case 前 reset snapshot，结束后断言状态并销毁。

记录/replay 可提高稳定性，但 replay 的网页/API 响应会过时；标注 capture 时间与 hash。对模型版本比较使用同一 observation，避免外部漂移冒充模型差异。

## 故障注入

系统性注入：timeout 前/后远端是否成功、429、部分结果、格式损坏、陈旧 ETag、并发修改、磁盘满、ledger 写失败、进程重启和用户取消。每类验证状态机不会进入非法状态，且恢复不重复副作用。

chaos 测试不应首先在线上做。先单元 state transition，再模拟器集成，最后在隔离 staging 演练。

## 提示注入红队

攻击面包括网页、邮件、附件、工具错误、记忆和其他 Agent 消息。payload 变体覆盖：直接覆盖指令、伪 system 标签、编码/多语言、数据外带 URL、工具诱导和持久记忆投毒。

判定看真实行为：是否访问禁止资源、是否泄漏 canary secret、是否绕过审批。只看模型回复“我不会”不足以证明安全，因为它可能已经执行动作。

同时测 benign 文档中的正常命令示例，防止防护导致过度拒绝。

## 多 Agent 评测

除总体成功，测 handoff 正确率、消息丢失/误解、重复工作、冲突解决、共享状态一致性和单点 Agent 失败恢复。比较单 Agent 基线；若多 Agent 只增加 token 而无显著质量/延迟收益，就不应采用。

评审 Agent 与生成 Agent 使用同一家模型时错误相关。用确定性测试、不同数据/模型或人工抽样建立真正独立证据。

## 统计比较

固定 case id 做 paired comparison，报告均值差、置信区间和关键切片。模型采样有方差，对高方差任务多 seed/多次运行，并把 pass@1 与 pass@k 区分。pass@k 增长可能靠更多成本换来，必须同时报告预算。

任务污染会使模型记住 benchmark。保留私有 holdout、参数化环境和定期新建 case；公开集用于开发，不能作为唯一发布门禁。

## 线上监控

线上无法知道全部 gold，但可监控完成/取消/升级、人工修正、重复调用、审批拒绝、pending age、工具错误、步骤/token/费用和安全事件。抽样人工复核与用户反馈要关联 task version。

异常率按工具、模型、任务类型、tenant tier 和版本切片。平均步骤稳定可能掩盖某工具陷入循环。

## 发布门禁

示例：

1. 所有权限/重复副作用 case 必须零违规；
2. 关键任务成功率不低于基线，置信区间下界满足阈值；
3. 每个风险/语言/工具切片无显著退化；
4. p95 步骤、延迟和费用在预算；
5. 恢复与 reconciliation case 全通过；
6. 新模型/prompt 先 shadow，再小流量 canary。

回滚不只切模型，还要恢复 prompt、tool schema、policy 和 memory/index version 的兼容组合。

## 面试追问

**如何评测一个会发邮件的 Agent 而不真发？** 用实现同 schema 的隔离邮件模拟器，验证 outbox 状态、审批事件、收件人/正文和幂等；少量 staging 端到端使用专用测试域，普通回归禁止生产凭据。

**为什么 exact trajectory match 不合理？** 开放任务有多条正确路径；应对最终状态和安全不变量做确定性检查，并将效率作为连续指标，而非要求复刻一条参考思路。

**怎样发现模型升级后的隐性回归？** 同 case 配对、多次采样、关键切片和 shadow trace；同时比较任务成功、工具/权限事件、步骤/成本，不只比较最终文本 judge。
