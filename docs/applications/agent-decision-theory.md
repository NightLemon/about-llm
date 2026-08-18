# Agent 决策理论：从 POMDP、Belief State 到安全终止

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：希望从形式化决策理解 Agent，而不只会套 ReAct/Planner 模板的开发者与算法工程师。
- **先修**：[Agent 架构](agent-architecture.md)、[LLM 强化学习](../training/reinforcement-learning.md)。
- **首次阅读**：POMDP → belief update → expected utility → information value → constraints → safety/liveness。
- **完成信号**：能判断应执行、先观察、升级还是停止，并写出对应假设与反例。
- **卡住时**：先运行本章二状态 exact control，再把数组名称替换成自己的任务对象。

</div>

Agent 不只是“语言模型循环调用工具”。它是在真实状态不可完全观察、动作有成本和副作用、观察可能不可靠、任务有终止条件的环境中连续决策。本章把这些工程事实还原为决策理论对象，再解释形式化模型能证明什么、不能替代什么。

## 1. 为什么普通 Workflow 不需要 POMDP

若所有状态、转移和异常分支都可枚举，程序直接读取可信数据库状态并执行确定性规则，那么有限状态机或 workflow 已经足够。加入概率模型不会自动提高可靠性。

Agent 更接近部分可观测决策问题，因为：

- 用户意图、网页真实性、远端副作用和工具进度不完全可见；
- 同一工具结果可能有噪声、过时、缺字段或恶意内容；
- 一次 action 会改变环境以及下一步可见信息；
- 系统要在质量、费用、延迟、风险和权限之间选择；
- “继续搜索”“先确认”“执行”“拒绝”都可能是不同动作。

形式化的目的不是让每个 Agent 都部署一个 POMDP solver，而是强迫设计者分清：什么是真实状态、什么只是 observation、系统相信什么、动作改变什么、用什么标准停止。

## 2. 从 MDP 到 POMDP

### 2.1 MDP 假设状态可见

[MDP](../reference/glossary.md#term-mdp) 常写成 \((S,A,T,R,\gamma)\)：

- \(S\)：环境 state；
- \(A\)：action；
- \(T(s'\mid s,a)\)：transition model；
- \(R(s,a,s')\)：reward 或效用信号；
- \(\gamma\)：discount factor。

Markov 假设不是“世界没有历史”，而是当前 state 已包含预测未来所需的信息。若 state 只保存最近一句对话，却漏掉 pending side effect、授权版本和截止时间，它通常不满足这个假设。

### 2.2 POMDP 把 observation 与 state 分开

[POMDP](../reference/glossary.md#term-pomdp) 再加入 observation space \(\Omega\) 和 observation model：

\[
Z(o\mid s',a)=P(O_{t+1}=o\mid S_{t+1}=s',A_t=a).
\]

Agent 看见的是 \(o\)，不是隐藏的真实 \(s'\)。工具返回 `completed` 是 observation；远端订单是否只创建了一次、数据是否最新、网页内容是否可信，属于需要独立验证的 state 命题。

POMDP policy 通常根据 belief state 选择动作：

\[
\pi(a\mid b),\qquad b(s)=P(S=s\mid \text{history}).
\]

Belief 在已知模型假设下是 history 的充分统计量。真实 Agent 的 transition/observation model 往往不完整且会漂移，所以 belief 只是带版本和证据边界的近似，不是事实本身。

## 3. State、Observation、Context 与 Memory

这些词在 Agent 文档中经常被混用：

| 对象 | 回答的问题 | 例子 |
|---|---|---|
| [State](../reference/glossary.md#term-state) | 环境此刻实际上是什么 | 订单已创建但本地尚未 ack |
| [Observation](../reference/glossary.md#term-observation) | 系统收到了什么信号 | provider timeout、receipt、网页文本 |
| [Belief state](../reference/glossary.md#term-belief-state) | 在当前证据和模型下相信什么 | 已创建概率 0.7，未创建概率 0.3 |
| Context | 本轮送给模型的有限输入 | policy 摘要、最近事件、工具 schema |
| Memory | 跨步骤或跨任务保存的记录 | event log、用户确认、artifact reference |

Context 不是完整 state：它可能被截断、摘要或污染。Memory 也不是 belief：数据库中保存一条 observation，不表示系统已经评估其可信度。模型 hidden state 更不能直接当可审计 belief，因为外部控制面无法读取、校准或稳定重放它。

工程上应分别保存：原始 observation、provenance、验证结果、用于更新 belief 的模型版本，以及最后作出的 decision。不要只保存自然语言摘要。

## 4. Belief Update

给定先前 belief \(b_t(s)\)、已执行动作 \(a_t\)、transition \(T\) 和新 observation \(o_{t+1}\)，先预测：

\[
\bar b_{t+1}(s')=\sum_s T(s'\mid s,a_t)b_t(s),
\]

再用 Bayes rule 更新：

\[
b_{t+1}(s')=
\frac{Z(o_{t+1}\mid s',a_t)\bar b_{t+1}(s')}
{\sum_x Z(o_{t+1}\mid x,a_t)\bar b_{t+1}(x)}.
\]

分母是收到该 observation 的边际概率。如果为零，说明 observation 与当前模型矛盾；正确处理不是除零或硬塞进 prompt，而是进入模型失配、数据损坏或异常来源分支。

二状态例子中，先验 \(P(A)=0.6\)、\(P(B)=0.4\)，诊断信号在真实状态下有 0.85 正确率。观察到 `A` 的概率是：

\[
0.6\times0.85+0.4\times0.15=0.57,
\]

后验为：

\[
P(A\mid o_A)=\frac{0.51}{0.57}\approx0.8947.
\]

这个数只在先验、signal likelihood 和二状态假设下成立。把 0.85 换成模型自述 confidence，或把网页文本当独立传感器，都会改变甚至破坏结论。

## 5. Expected Utility：先定义代价，再优化

[Expected utility](../reference/glossary.md#term-expected-utility) 在 belief \(b\) 下给 action 计分：

\[
EU(a\mid b)=\sum_s b(s)U(a,s).
\]

[Utility](../reference/glossary.md#term-utility) 描述决策者对结果的偏好尺度。它与以下对象不同：

- reward 是环境或 evaluator 给出的单步/轨迹训练信号；
- metric 是评测协议中的测量值；
- model score 可能只是排序 logit；
- 金额、延迟和风险是 utility 的输入，不天然可直接相加。

若“修对”效用为 `+10`，“修错”为 `-14`，先验为 `[0.6, 0.4]`，直接选择 `repair_a` 的期望效用是：

\[
0.6\times10+0.4\times(-14)=0.4.
\]

期望值默认风险中性。少量灾难性损失可能被大量小收益平均掉；高风险系统还需 worst-case、chance constraint、CVaR、分级审批或直接禁用某动作。不能因为平均 utility 为正就越过权限。

### Hard constraint 先于 utility

假设一个未授权 shortcut 在所有状态效用都写成 `100`，它仍不能进入 argmax。正确顺序是：

1. schema、主体、资源、capability、policy 和 approval 形成 allowed action set；
2. 只在 allowed set 内比较 utility；
3. handler 后用 effect verifier 判断真实结果。

这是“优化目标”与“可行动作集合”的区别。把越权罚成 `-1000` 仍不是安全边界：模型或 optimizer 可能在更大收益、错误概率或尺度漂移下接受这项惩罚。

## 6. Value of Information：何时先观察

Agent 经常在“立即执行”和“先调用诊断工具”之间选择。[Value of information](../reference/glossary.md#term-value-of-information) 把 observation 对后续决策的改善显式化。

对不会改变 hidden state 的一次 noisy observation，expected value of sample information（EVSI）为：

\[
\operatorname{EVSI}
=\sum_o P(o)\max_a E[U(a,S)\mid o]
-\max_a E[U(a,S)].
\]

若 observation cost 为 \(c\)，净信息价值为 `EVSI - c`。EVSI 在精确模型下不小于零，因为收到 signal 后仍可忽略它、执行原 action；扣除成本后可以为负。

在前述二状态例子中，0.85 准确率的 signal 会让两个 observation 分支分别选择 `repair_a` 和 `repair_b`。观察前最佳效用为 `0.4`，观察后但未扣成本的期望效用为 `6.4`，所以：

```text
EVSI = 6.4 - 0.4 = 6.0
net VOI = 6.0 - observation_cost(1.0) = 5.0
```

若 signal 准确率只有 0.51，两个分支都不会改变原决策，EVSI 为零；付出 1.0 成本后净值为 `-1.0`。更多 observation 不等于更理性，关键是它是否可能改变 action 或降低风险。

### Information action 与 effect action

读取状态、运行 dry-run、查询审计日志通常以获取信息为主；“尝试创建一次再看结果”会改变世界，不是纯 observation。后者需要 transition、幂等、pending 和 reconciliation，不能直接套静态 VOI 公式。

Observation 还必须经过权限和可信度检查。信息价值高不表示可以读取无权数据；恶意网页提供“确定答案”也不意味着 likelihood 可靠。

## 7. Horizon 与闭环规划

[Horizon](../reference/glossary.md#term-horizon) 是规划考虑的未来步数或时间范围。短 horizon 便宜但可能贪心，长 horizon 能看到延迟收益，也会放大 transition model 误差和搜索成本。

区分三种计划：

- **Open-loop plan**：先生成固定动作序列，不根据新 observation 改变；
- **Closed-loop policy**：每次 observation 后重新选择 action；
- **Receding-horizon / model-predictive control**：规划未来若干步，只执行第一步，观察后再规划。

LLM 的 plan-and-execute 若每步检查前置条件并允许 replan，接近第三种工程模式。Plan 本身不是 policy：`[search, edit, deploy]` 没说明搜索为空、测试失败或审批过期时怎么办。

Tree search、MCTS 或 best-of-N 可以扩展候选路径，但需要 transition simulator、value/verifier 和预算。共享同一模型的 generator 与 evaluator 可能相关地犯错；搜索找到高分路径不证明 world model 或 reward 正确。

## 8. Planning under Uncertainty

[Planning under uncertainty](../reference/glossary.md#term-planning-under-uncertainty) 不只比较动作序列，还要比较 observation 分支：

```text
inspect
├─ observation=A -> repair_a
└─ observation=B -> repair_b
```

这棵 policy tree 与“先 inspect，再永远 repair_a”的 linear plan 不同。实际系统可以不显式求解整棵 POMDP，而用以下近似：

- 把高风险不确定点变成 clarification/approval；
- 只对少量关键状态保存 calibrated belief；
- 用 deterministic workflow 包围一个局部模型 decision；
- 对每个 action 写 precondition、observation contract 和 fallback；
- 设置 horizon、branch、token、tool、time 与 cost budget。

近似不是问题，隐藏近似才是问题。报告应说明忽略了哪些 latent state、相关 observation 和非平稳变化。

## 9. Constrained MDP 与权限边界

[Constrained MDP](../reference/glossary.md#term-constrained-mdp) 常把一个或多个累计 cost 加到优化约束：

\[
\max_\pi E_\pi\left[\sum_t\gamma^t R_t\right]
\quad\text{s.t.}\quad
E_\pi\left[\sum_t\gamma^t C_{k,t}\right]\le d_k.
\]

它适合表达平均费用、延迟、人工升级率或资源预算。但 expected constraint 仍可能允许少量严重违规，只要平均值不超阈值。

权限、租户隔离、不可逆副作用审批和禁止读取 secrets 通常应实现为 hard transition guard 或 [safety property](../reference/glossary.md#term-safety-property)，而不是可交易的 soft cost。CMDP 可以帮助选策略，不能替代 IAM、schema、sandbox 和 effect verifier。

## 10. Safety、Liveness 与 Termination

形式化验证常区分：

- **Safety property**：“坏事永远不发生”，例如未审批动作从不进入 handler；
- **Liveness property**：“好事最终发生”，例如每个已接收任务最终完成、失败或升级；
- **Termination**：运行最终进入某个 terminal state；
- **Bounded liveness**：在 N 步或 deadline 内进入允许终态。

一个 terminal state 可达，只证明“存在一条结束路径”，不证明所有执行都会结束。若还有可达的非终态 cycle，scheduler/模型可能永远循环；若有非终态 dead end，任务会卡住而不是完成。

对有限 nondeterministic transition graph，在“不假设公平调度、进入 terminal 即停止”的语义下，可用以下充分检查：

1. 从 initial states 计算全部 reachable states；
2. reachable forbidden 为空，才满足当前 safety property；
3. 至少一个 terminal reachable，只说明 may terminate；
4. reachable nonterminal subgraph 无 dead end 且无 cycle，才保证所有路径终止。

真实 Agent 的状态空间通常不有限，工具和模型也不是完整 transition graph。本章检查只能验证 authored abstraction；生产仍需 deadline、loop detector、idempotency、reconciliation 和人工接管。

## 11. Belief 与 Agent Memory

Working/episodic/semantic memory 向下一轮提供 observation 和先验线索，但不自动形成正确 belief：

- 一条旧 memory 可能在新版本下失效；
- 多条 memory 可能来自同一个相关错误源，不能当独立证据相乘；
- “用户喜欢 A”与“用户在任务 X 选过 A”scope 不同；
- 删除或撤权后，belief 更新也必须移除相应证据。

保存 posterior 数字却不保存 prior、likelihood、evidence identity 和模型版本，无法审计。很多开放 Agent 不需要伪精确概率；`known / unknown / conflicting / needs verification` 的 typed belief 也比模型自由文本更可靠。

## 12. 多 Agent 不是独立传感器

多个 Agent 使用同一基础模型、相似 prompt 和相同网页时，错误高度相关。三票一致不能直接按独立 Bernoulli 证据更新 belief。

多 Agent 真正增加信息的条件包括：不同数据源、工具权限、模型族、独立 verifier 或明确 adversarial review。否则它主要改变搜索预算和上下文分工，不自动增加 epistemic independence。

通信消息仍是 observation；接收方需验证 schema、身份、artifact 和授权。A2A `completed` 是远端任务状态，不是本地业务成功证明。

## 13. 映射到现有 Agent Runtime

| 决策理论对象 | 本仓库工程对象 | 仍需外部建立 |
|---|---|---|
| state | task、ledger、provider effect | 真实世界是否与记录一致 |
| observation | typed tool outcome、receipt、artifact | provenance、可信度、完整性 |
| belief | pending/known/uncertain 与人工结论 | 校准概率或领域状态模型 |
| action set | Tool registry + policy + approval | 集中 IAM 与真实 capability |
| utility/cost | token、费用、延迟、任务结果 | 用户偏好、风险尺度、业务价值 |
| transition | handler/outbox/reconciliation | provider 语义与故障分布 |
| terminal | completed/failed/escalated | 独立 completion/effect verifier |

Runtime 的顺序仍然是 schema → trusted resource resolution → policy → approval → handler → verifier。决策理论帮助选择 allowed actions 中的下一步，不会把 model proposal 变成授权。

## 14. 可运行 finite exact control { #exact-control }

运行：

~~~powershell
python projects/safe-agent/decision_theory_toy.py
python -m pytest tests/test_agent_decision_theory.py -q
~~~

固定 CPU control 枚举两个 hidden states、四个 actions 和两个 observations，验证：

- belief update 得到 observation probability `0.57` 与 posterior `[0.8947, 0.1053]`；
- 未授权 action 的 unconstrained utility 为 `100`，仍被 hard allow-mask 排除；
- 强 signal 的 observation probabilities 为 `[0.57, 0.43]`，EVSI 为 `6.0`，扣除成本后 net VOI 为 `5.0`；
- 弱 signal 不改变 action，EVSI 为 `0`，扣除成本后为 `-1.0`；
- terminal reachable 的图仍可因 nonterminal cycle 而不保证 termination；
- reachable forbidden state 与 termination 是两个独立结论。

实现位于 `src/about_llm/agents/decision_theory.py`。Probability normalization、shape、finite value、hard action mask 和 transition graph schema 都会 fail closed。

### 这个实验没有证明什么

Control 没有调用 LLM、真实工具、provider、网络或审批服务，也没有从数据学习 transition/observation model。Utility、prior 和 likelihood 都由作者指定；有限二状态/四状态图不代表开放环境。因此它只验证公式、分支和术语边界，不证明真实 Agent 的校准、任务成功、安全、终止、性能或最优策略。

## 15. 常见错误结论

**“模型看到了完整对话，所以 state 可见。”** 对话是有限且可能污染的 observation/context，不包含所有外部状态。

**“Confidence 0.9 就是 belief 0.9。”** 除非事件、标注和校准协议固定，模型自述 confidence 不是 posterior probability。

**“多调用一次工具总能减少不确定性。”** Signal 可能不改变 action、成本过高、相关重复，或工具本身改变环境。

**“Expected utility 最大就应该执行。”** 先做 hard authorization 和 safety gate；utility 只在 allowed set 内比较。

**“存在 completed 路径就会完成。”** 另一个 cycle/dead end 足以破坏 guaranteed termination。

**“加 step limit 就证明 liveness。”** 它证明 bounded stop policy，不证明任务最终成功；预算耗尽可能进入 failed/escalated。

**“多个 Agent 一致就是独立证据。”** 同模型、同数据和同 prompt 产生相关错误，不能按独立投票计算置信度。

## 16. 面试时怎样回答

面对“Agent 怎样决定下一步”，可以按以下顺序：

1. 区分 hidden state、observation、persisted task state 和 model context；
2. 给不确定状态维护 typed belief 或至少 `known/unknown/conflicting`；
3. 在 policy/approval 形成的 allowed action set 内比较 expected utility；
4. 用 value of information 判断先查询、澄清还是执行；
5. 每个 effect action 后更新 state，并对 uncertain outcome 做 reconciliation；
6. 用 safety invariant、verifier、cycle/dead-end detection 和 deadline 定义停止；
7. 保存 prior、observation、decision、模型/策略版本和证据边界。

## 17. 自测与实践

1. 为什么 tool output 是 observation，而不是自动等于 state？
2. 用 Bayes rule 推导本章 `0.8947` posterior。
3. 构造一个 signal 很准确但 observation cost 更高、因此不值得查询的例子。
4. 为什么 EVSI 在不扣成本前非负？这个结论依赖什么选择能力？
5. 给一个 expected cost constraint 通过但 safety property 失败的策略。
6. 画一个 terminal reachable、却存在 nonterminal cycle 的状态图。
7. 把自己的 Agent trace 映射为 state/observation/action/terminal，并列出没有建模的 latent state。
8. 运行 toy，先预测强/弱 signal 和 forbidden action 的选择，再核对 JSON。

## 一手资料

- Kaelbling, Littman and Cassandra, [Planning and Acting in Partially Observable Stochastic Domains](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/kaelbling96a.html)
- Sutton and Barto, [Reinforcement Learning: An Introduction, second edition](http://incompleteideas.net/book/the-book-2nd.html)
- Howard, [Information Value Theory](https://doi.org/10.1109/TSSC.1966.300074)
- Altman, [Constrained Markov Decision Processes](https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf)
- Alpern and Schneider, [Defining Liveness](https://doi.org/10.1016/0022-0000(85)90056-0)

下一步阅读[Agent 架构](agent-architecture.md)与[工具协议和故障恢复](agent-runtime.md)，把 belief、constraints 和终止性质落实到 typed state machine。
