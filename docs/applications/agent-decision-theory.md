# Agent 怎样在不确定中决定下一步

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：希望理解 Agent 为什么要观察、行动、升级或停止的开发者与算法工程师。
- **先修**：[Agent 架构](agent-architecture.md)；了解概率与条件概率即可，强化学习不是必需前置。
- **首次阅读**：先运行二状态例子，再按“看到什么 → 相信什么 → 选什么 → 何时停止”的顺序阅读。
- **完成信号**：能为一个不确定任务写出隐藏状态、可见信号、允许动作和终态，并手算一次决策。
- **卡住时**：把 `fault_a` 读成“配置错误”，把 `fault_b` 读成“依赖故障”，暂时忽略 POMDP 缩写。

</div>

凌晨告警显示服务错误率升高。现在有两种主要可能：

- **故障 A**：配置刚刚漂移，应该回滚配置；
- **故障 B**：下游依赖异常，应该切换依赖。

Agent 尚不知道真实原因。它只能看到告警、日志和诊断工具返回的信号。立即修复有机会快速恢复，也可能修错；
先诊断会增加延迟，却可能改变下一步选择；绕过审批直接改生产最快，但当前身份没有这项权限。

这一个例子足以提出本章的四个问题：

1. 工具返回的是事实，还是关于事实的带噪信号？
2. Agent 应立即修复，还是先花成本诊断？
3. 一个收益很高但未授权的动作，能否进入比较？
4. “存在一条恢复路径”是否表示任务一定会结束？

POMDP、belief state、expected utility 和 value of information 都是在精确回答这些问题。先跟完这次故障，
再看术语会容易得多。

## 先运行这个二状态故障 { #exact-control }

脚本把故障和动作写成了通用名称：

| 脚本名称 | 在本章中的含义 |
|---|---|
| `fault_a` | 配置错误 |
| `fault_b` | 依赖故障 |
| `repair_a` | 回滚配置 |
| `repair_b` | 切换依赖 |
| `escalate` | 不自动修改，升级给人工 |
| `forbidden_shortcut` | 绕过审批直接改生产 |

运行前先预测三件事：先验更偏向故障 A 时，哪个允许动作会胜出；准确率 0.85 的诊断是否值得花 1.0 成本；
以及终态可达时，流程是否仍可能无限循环。

~~~powershell
python projects/safe-agent/decision_theory_toy.py
python -m pytest tests/test_agent_decision_theory.py -q
~~~

第一次不要从整份 JSON 的第一行读到最后一行。按下面的顺序观察：

~~~text
state_labels / action_labels
hard_constraint
belief_update_after_observation_a
strong_signal
weak_signal
transition_systems
scope
~~~

你会得到这条决策链：

~~~text
先验：故障 A 0.6，故障 B 0.4
→ 不诊断时，最佳允许动作是 repair_a，期望效用 0.4
→ 强诊断返回 A 后，belief 更新为 [0.8947, 0.1053]
→ 强诊断的 EVSI 是 6.0，扣除诊断成本后净值是 5.0
→ 弱诊断不会改变动作，扣除成本后的净值是 -1.0
→ forbidden_shortcut 即使得分 100，也不进入允许动作的比较
~~~

最后三组状态图分别展示：安全且保证结束、坏状态可达、终态可达但仍可能循环。后文会逐步推导这些结果。

## 第一步：分清真实状态和可见信号

故障原因在环境中客观存在，但 Agent 不能直接读取它。决策理论把两者分开：

| 对象 | 它回答什么 | 故障例子 |
|---|---|---|
| [State](../reference/glossary.md#term-state) | 环境实际上处于什么状态 | 真正原因是配置错误 A |
| [Observation](../reference/glossary.md#term-observation) | 系统收到了什么信号 | 诊断工具报告 A |
| [Belief state](../reference/glossary.md#term-belief-state) | 根据当前证据，对各状态相信多少 | A 为 0.6，B 为 0.4 |
| Context | 本轮送给模型的有限输入 | 告警摘要、工具 schema、最近日志 |
| Memory | 跨步骤保存的记录 | 历史事件、人工确认、诊断工件 |

工具报告 A 不等于真实原因一定是 A。Provider 返回 `completed` 也不等于外部副作用已经独立验证。
Observation 可能带噪、过时、缺字段，甚至来自恶意内容。

Context 只是模型本轮能看到的输入，可能被截断或摘要。Memory 只是保存下来的记录，也可能已经失效。
只有在明确的先验和信号模型下，系统才可以把 observation 更新成可审计的 belief。

工程记录至少应保留：

~~~text
原始 observation
来源与时间
验证结果
更新 belief 所用的模型版本
更新前后的 belief
最终 decision
~~~

只保存“模型认为大概率是 A”这句话，无法重放这次判断。

## 第二步：用 Bayes rule 更新 belief

脚本开始时采用先验：

\[
P(A)=0.6,\qquad P(B)=0.4.
\]

强诊断工具的准确率为 0.85。它的信号模型是：

| 诊断结果 | 真实 A | 真实 B |
|---|---:|---:|
| 报告 A | 0.85 | 0.15 |
| 报告 B | 0.15 | 0.85 |

本例假设读取诊断不会改变故障原因，所以状态转移是单位矩阵。收到“报告 A”之前，系统先根据动作预测下一状态：

\[
\bar b_{t+1}(s')=\sum_s T(s'\mid s,a_t)b_t(s).
\]

再用 observation likelihood 更新：

\[
b_{t+1}(s')=
\frac{Z(o_{t+1}\mid s',a_t)\bar b_{t+1}(s')}
{\sum_x Z(o_{t+1}\mid x,a_t)\bar b_{t+1}(x)}.
\]

收到“报告 A”的总概率是：

\[
P(o_A)=0.6\times0.85+0.4\times0.15=0.57.
\]

因此：

\[
P(A\mid o_A)=\frac{0.6\times0.85}{0.57}
\approx 0.8947.
\]

新的 belief 是 `[0.8947, 0.1053]`。这不是诊断工具直接吐出的 confidence，而是由先验、两个 likelihood
和二状态假设共同算出的后验。

如果分母为零，说明当前模型认为这个 observation 不可能出现。程序会报错；真实系统应进入数据损坏、
来源异常或模型失配分支，不能偷偷把不一致信号塞回 Prompt。

## 第三步：先排除禁止动作，再比较期望效用

[Expected utility](../reference/glossary.md#term-expected-utility) 用 belief 对每个动作在不同状态下的结果加权：

\[
EU(a\mid b)=\sum_s b(s)U(a,s).
\]

本例使用以下人为设定的相对效用：

| 动作 | 故障 A | 故障 B | 是否允许 |
|---|---:|---:|---|
| 回滚配置 `repair_a` | +10 | -14 | 是 |
| 切换依赖 `repair_b` | -14 | +10 | 是 |
| 升级人工 `escalate` | 0 | 0 | 是 |
| 绕过审批 `forbidden_shortcut` | +100 | +100 | 否 |

这些数字是为了手算而指定的偏好尺度，不是从事故数据估计出的金额或风险。它们表达一件事：修对有收益，
修错的损失更大，升级人工作为相对基线。

在先验 `[0.6, 0.4]` 下，直接回滚配置的期望效用是：

\[
0.6\times10+0.4\times(-14)=0.4.
\]

切换依赖的期望效用是 `-4.4`，升级人工是 `0`。因此，不做诊断时，三个允许动作中 `repair_a` 最优。

绕过审批被人为写成 `100`，它仍不能进入最大值选择。执行顺序应是：

1. Schema、主体、资源、capability、policy 和 approval 形成允许动作集合；
2. 只在这个集合内比较期望效用；
3. Handler 返回后，再由 effect verifier 检查真实结果。

把越权动作罚成 `-1000` 仍不等于权限边界。收益尺度、概率或优化器一旦变化，软惩罚仍可能被抵消。
未授权动作应该从可执行集合中移除。

### Utility、reward 和 metric 不是同一个数

- **Utility** 表示特定决策主体对结果的偏好；
- **Reward** 是环境或评估器提供的训练信号；
- **Metric** 是评测协议测得的量；
- 模型 score 可能只是排序 logit。

延迟、金额和风险可以成为 utility 的输入，但不能因为它们都是数字就直接相加。少量灾难性损失也可能被平均收益掩盖；
高风险系统还需要最坏情况分析、chance constraint、CVaR、审批或直接禁用动作。

## 第四步：判断一次诊断是否值得

现在比较两种策略：

~~~text
立即行动：按先验直接 repair_a

先诊断：
├─ 报告 A → 更新 belief → repair_a
└─ 报告 B → 更新 belief → repair_b
~~~

[Value of information](../reference/glossary.md#term-value-of-information) 衡量新信号能否改善后续决策。
对于不会改变隐藏状态的一次诊断，样本信息的期望价值（EVSI）是：

\[
\operatorname{EVSI}
=\sum_o P(o)\max_a E[U(a,S)\mid o]
-\max_a E[U(a,S)].
\]

在模型和动作集合固定、收到信号后仍可沿用原动作的前提下，未扣成本的 EVSI 不小于零。因为信号无用时，
决策者可以忽略它。诊断成本、延迟和副作用另行扣除后，净值可以为负。

强诊断让两个分支选择不同修复。观察前最佳效用是 `0.4`；观察后的加权效用是 `6.4`：

~~~text
EVSI = 6.4 - 0.4 = 6.0
净信息价值 = 6.0 - 诊断成本 1.0 = 5.0
~~~

所以在这组假设下，先诊断值得。

把诊断准确率降到 `0.51` 后，无论它报告 A 还是 B，后验仍会选择 `repair_a`。信号没有改变动作，
EVSI 为 `0`；扣除成本后净值为 `-1.0`。

这解释了为什么“再调用一个工具”不一定更理性。信号可能太弱、与已有证据重复，或成本高于它带来的决策改善。

### 诊断动作与副作用动作要分开

读取日志、执行 dry-run 和查询审计记录通常以获取信息为主。“先尝试修改一次再看结果”会改变环境，
不是纯诊断。后者需要状态转移、审批、幂等、pending 和 reconciliation，不能直接套上面的静态公式。

信息有价值也不代表 Agent 有权读取它。恶意网页给出一个很确定的答案，也不表示这条来源具有可信的 likelihood。

## 什么时候需要 MDP 或 POMDP

如果所有状态、异常和转移都能列举，程序又能读取可信数据库中的完整状态，那么有限状态机或普通 workflow
通常已经足够。加入概率模型不会自动提高可靠性。

[MDP](../reference/glossary.md#term-mdp) 常写成 \((S,A,T,R,\gamma)\)：

- \(S\) 是状态集合；
- \(A\) 是动作集合；
- \(T(s'\mid s,a)\) 描述状态怎样转移；
- \(R(s,a,s')\) 是 reward 或效用信号；
- \(\gamma\) 控制未来结果的折扣。

Markov 假设的含义不是“世界没有历史”，而是当前 state 已包含预测未来所需的信息。如果 state 漏掉了
待确认的副作用、权限版本或截止时间，这个假设通常不成立。

[POMDP](../reference/glossary.md#term-pomdp) 再把 observation 与 state 分开：

\[
Z(o\mid s',a)=P(O_{t+1}=o\mid S_{t+1}=s',A_t=a),
\]

并让 policy 根据 belief 选择动作：

\[
\pi(a\mid b),\qquad b(s)=P(S=s\mid \text{history}).
\]

在模型假设完全已知时，belief 可以概括与未来决策有关的历史。真实 Agent 的状态转移和信号模型通常不完整，
也会随系统版本漂移。因此，工程中的 belief 是带来源、版本和适用范围的近似，不是新的事实来源。

采用这些概念不表示必须部署完整的 POMDP solver。它们首先帮助团队把隐藏状态、可见信号、动作后果和停止规则
说清楚。

## 从一次选择走向闭环规划

一次修复后，Agent 还要观察服务是否恢复，再决定完成、继续诊断或升级。此时需要区分 plan 与 policy。

| 方式 | 它怎样处理新 observation |
|---|---|
| Open-loop plan | 预先生成固定动作序列，中间结果变化也不改计划 |
| Closed-loop policy | 每次得到新信号后重新选择动作 |
| Receding-horizon planning | 规划未来若干步，只执行第一步，观察后再规划；也常称 model-predictive control |

`[诊断, 回滚, 完成]` 只是一个动作序列。它没有说明诊断指向 B、回滚失败或审批过期时怎么办。
更完整的 policy 应包含分支：

~~~text
diagnose
├─ signal=A → request_approval(repair_a)
├─ signal=B → request_approval(repair_b)
└─ conflicting / unavailable → escalate
~~~

[Horizon](../reference/glossary.md#term-horizon) 是规划显式考虑的未来步数或时间范围。短 horizon 便宜，
但可能只追求眼前恢复；长 horizon 能看到验证和回滚的延迟收益，也会放大状态模型误差与搜索成本。

Tree search、MCTS 或 best-of-N 能扩展候选路径，但仍需要状态转移模拟、价值判断和预算。若生成器与评估器
来自同一个模型，它们可能相关地犯错。搜索找到高分路径，不证明 world model 或评分标准正确。

### 在不确定状态下规划什么

[Planning under uncertainty](../reference/glossary.md#term-planning-under-uncertainty) 规划的是 observation 分支，
而不只是更长的动作列表。开放系统通常不会显式求解整棵 policy tree，可以采用这些近似：

- 把高风险不确定点变成澄清或审批；
- 只为少量关键状态维护校准后的 belief；
- 用确定性 workflow 包围局部模型决策；
- 为每个动作写前置条件、返回契约和 fallback；
- 限制分支数、工具调用、时间、token 与费用。

近似本身没有问题。需要明确写出哪些隐藏状态、相关信号和非平稳变化尚未建模。

## 平均成本约束不能代替硬权限

[Constrained MDP](../reference/glossary.md#term-constrained-mdp) 可以把累计成本放进策略优化：

\[
\max_\pi E_\pi\left[\sum_t\gamma^t R_t\right]
\quad\text{s.t.}\quad
E_\pi\left[\sum_t\gamma^t C_{k,t}\right]\le d_k.
\]

这适合限制平均费用、延迟、人工升级率或资源消耗。但期望约束可能容许少量严重违规，只要它们被其他样本平均掉。

权限、租户隔离、不可逆副作用审批和禁止读取 secrets 通常应实现为硬的状态转移条件或
[safety property](../reference/glossary.md#term-safety-property)。CMDP 可以帮助选择允许策略，不能替代 IAM、
Schema、sandbox 和副作用验证。

## 终态可达，不表示一定会结束

故障修复流程至少要区分三个问题：

- **Safety**：未审批动作是否永远不会进入执行器；
- **Liveness**：已接收任务是否最终会完成、失败或升级；
- **Termination**：运行是否最终进入某个终态。
- **Bounded liveness**：是否能在指定步数或 deadline 内进入允许终态。

考虑下面的流程：

~~~text
triage
├─ repair_a → terminal
└─ repair_b ─┬→ terminal
             └→ repair_b
~~~

`terminal` 明明可达，`repair_b` 的自循环却允许系统永远重试。只能证明“可能结束”，不能证明所有路径都会结束。
如果某个非终态没有任何出口，它会卡住，同样不满足保证终止。

脚本对有限、非确定性状态图执行四项检查：

1. 从初始状态找出所有可达状态；
2. 检查是否有可达状态被标为 forbidden；
3. 检查是否至少有一个终态可达；
4. 检查可达的非终态子图是否存在死路或环。

在“不假设公平调度、进入终态即停止”的语义下，只有终态可达且非终态没有死路和环，才能保证所有路径有限结束。

这项结论只适用于给定的有限图。真实 Agent 的状态空间、工具和外部系统更加开放，仍需要 deadline、循环检测、
幂等、reconciliation 和人工接管。步数上限可以保证“到点停止”，但不能保证任务成功。

## Memory 怎样影响 belief

Working、episodic 和 semantic memory 会为下一轮提供 observation 或先验线索，但不会自动生成正确 belief：

- 旧记录可能在新版本下失效；
- 多条记录可能来自同一个错误源，不能当作独立证据相乘；
- “用户喜欢 A”与“用户在任务 X 选过 A”的适用范围不同；
- 数据被删除或撤权后，后续 belief 也要移除相应证据。

若系统保存 posterior 数字，还应保存 prior、likelihood、证据身份和模型版本。很多开放任务不需要伪精确概率；
`known / unknown / conflicting / needs_verification` 这类有类型的状态，通常比模型自由文本更容易审计。

## 多 Agent 不等于多个独立传感器

三个 Agent 使用同一基础模型、相似 Prompt 和同一网页时，错误往往高度相关。三票一致不能直接按三个独立证据
更新 belief。

真正增加信息需要更独立的来源，例如不同数据源、工具权限、模型族、确定性验证程序或明确的对抗审查。
否则，多 Agent 主要增加搜索预算和上下文分工，不会自动增加证据独立性。

Agent 之间的消息仍然只是 observation。接收方要验证消息格式、身份、工件和授权。远端任务返回 `completed`，
也不等于本地业务结果已经验证。

## 把术语映射回 Agent Runtime

| 决策理论对象 | 本仓库中的工程对象 | 仍要向外部系统确认 |
|---|---|---|
| State | task、ledger、Provider effect | 真实世界是否与记录一致 |
| Observation | 有类型的工具结果、receipt、artifact | 来源、可信度与完整性 |
| Belief | pending、known、uncertain 与人工结论 | 校准概率或领域状态模型 |
| Action set | Tool registry、policy 与 approval | 集中 IAM 和真实 capability |
| Utility / cost | token、费用、延迟与任务结果 | 用户偏好、风险尺度和业务价值 |
| Transition | handler、outbox 与 reconciliation | Provider 语义和故障分布 |
| Terminal | completed、failed、escalated | 独立的完成和副作用验证 |

Runtime 的执行顺序仍然是：

~~~text
schema
→ trusted resource resolution
→ policy
→ approval
→ handler
→ verifier
~~~

决策理论只帮助 Agent 在允许动作中选择下一步。它不会把模型提出的 proposal 自动变成授权。

## 这个小程序说明了什么

脚本真实执行 NumPy 计算并穷举有限状态图。它验证：

- Bayes 更新得到 observation probability `0.57` 与后验 `[0.8947, 0.1053]`；
- 允许动作集合会排除高分的禁止动作；
- 强、弱两种诊断分别得到净信息价值 `5.0` 和 `-1.0`；
- “坏状态可达”和“所有路径终止”是两个独立结论；
- 概率未归一化、shape 不匹配、非法数值或错误状态图会被拒绝。

| 本例没有建立的证据 | 需要怎样补充 |
|---|---|
| 先验与诊断准确率是否符合真实事故 | 用目标环境历史数据估计并做校准 |
| 效用数值是否代表业务偏好 | 由业务、风险和运维负责人共同定义 |
| LLM 与工具在开放环境中的行为 | 运行真实任务集、故障注入与人工审计 |
| 生产 Agent 是否安全并稳定结束 | 验证 Runtime、权限、恢复机制和长期运行 |

因此，这个例子适合检查公式和建立直觉，不代表真实 Agent 已经完成校准或安全验证。

## 常见误解

**“模型看到了完整对话，所以 state 可见。”** 对话只是有限且可能污染的 context。

**“模型 confidence 为 0.9，所以 belief 是 0.9。”** Belief 需要明确事件、先验和 likelihood。

**“多调用一次工具总会减少不确定性。”** 弱信号可能不改变动作，净信息价值还会被成本抵消。

**“期望效用最高的动作就应该执行。”** 权限和审批先形成允许动作集合，效用只在集合内比较。

**“有一条 completed 路径，所以任务会完成。”** 另一个可达环或死路就能破坏保证终止。

**“加入 step limit 就证明 liveness。”** 它只定义预算耗尽时的停止策略，任务可能以失败或升级结束。

**“多个 Agent 投票就是独立证据。”** 同模型、同数据和同 Prompt 会产生相关错误。

## 面试时怎样回答

面对“Agent 怎样决定下一步”，可以沿本章故障例子回答：

1. 先区分隐藏 state、可见 observation、持久化任务状态和模型 context；
2. 用带来源的信号更新 belief，或至少维护 `known / unknown / conflicting`；
3. 由 policy 和 approval 形成允许动作集合；
4. 在允许集合内比较期望效用；
5. 用信息价值判断先诊断、澄清还是执行；
6. 每次副作用后重新观察，并对不确定结果做 reconciliation；
7. 用安全不变量、verifier、环/死路检测和 deadline 定义停止。

关键不是声称系统求解了完整 POMDP，而是清楚说明哪些状态不可见、概率从哪里来、哪些动作根本不能执行，
以及系统怎样从循环中退出。

## 自测

1. 为什么诊断工具报告 A 只是 observation，而不是 state？
2. 手算 `0.57` 和 `0.8947` 分别代表什么。
3. 为什么准确率 0.51 的诊断在本例中没有决策价值？
4. 构造一个信号很准确、但因成本过高而不值得调用的例子。
5. 为什么 `forbidden_shortcut` 得分 100 仍不能被选择？
6. 画一个终态可达、但存在非终态环的流程图。
7. 给一个平均成本约束通过、硬安全属性却失败的策略。
8. 把自己的 Agent trace 映射成 state、observation、action 和 terminal，并列出未建模状态。

## 一手资料

- Kaelbling, Littman and Cassandra, [Planning and Acting in Partially Observable Stochastic Domains](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/kaelbling96a.html)
- Sutton and Barto, [Reinforcement Learning: An Introduction, second edition](http://incompleteideas.net/book/the-book-2nd.html)
- Howard, [Information Value Theory](https://doi.org/10.1109/TSSC.1966.300074)
- Altman, [Constrained Markov Decision Processes](https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf)
- Alpern and Schneider, [Defining Liveness](https://doi.org/10.1016/0022-0000(85)90056-0)

下一步阅读 [Agent 架构](agent-architecture.md) 与[工具协议和故障恢复](agent-runtime.md)，把 belief、约束和终止性质
落实到有类型的状态机。
