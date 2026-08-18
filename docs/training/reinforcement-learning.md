# LLM 强化学习：从 Policy Gradient 到 GRPO 与 RLVR

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要理解或评审 LLM post-training、online RL 和 verifiable reward 的开发者与算法工程师。
- **先修**：[数学基础](../foundations/math.md)中的概率与梯度、[生成](../core/generation.md)和[对齐入门](alignment-basics.md)。
- **首次阅读**：contextual bandit → policy gradient → MDP/advantage → PPO → GRPO/RLVR → 方法选择。
- **完成信号**：能从采样分布推导 REINFORCE，区分 old/current/reference policy，并为 reward loophole 写出独立评测。
- **卡住时**：先运行本章的三动作 bandit，再回到[对齐完整章节](alignment.md)看 SFT、RM 和 DPO 数据流。

</div>

强化学习（Reinforcement Learning, RL）不是“给模型一个分数再反向传播”这么简单。它处理的是：动作由当前 policy 采样、reward 可能延迟、动作改变后续状态，而训练数据分布又随 policy 改变。

对 LLM 来说，一个 response 可以视为 token actions 的 trajectory。数学题答案、代码测试或工具执行结果可以提供 reward，但 verifier 只是任务代理；把它优化得更高不自动意味着帮助性、事实性或安全性提高。

## 1. 先从 contextual bandit 开始

Bandit 只有一次决策，没有状态转移。给定上下文 \(x\)，policy 在动作集合中采样 \(a\)，环境返回 reward \(r(x,a)\)：

\[
J(\theta)=\mathbb E_{a\sim\pi_\theta(\cdot\mid x)}[r(x,a)].
\]

若 policy 是 logits \(z\) 上的 categorical softmax，动作概率为 \(p_i\)，且每个动作 reward \(r_i\) 已知，则可以枚举：

\[
J=\sum_i p_i r_i,
\qquad
\frac{\partial J}{\partial z_j}=p_j(r_j-J).
\]

这个有限问题很重要：它给出 policy gradient 的 exact oracle。真实 LLM 无法枚举所有 response，但采样估计器应在这个小问题上对账。

## 2. Score-function / REINFORCE 梯度

利用恒等式 \(\nabla_\theta \pi=\pi\nabla_\theta\log\pi\)：

\[
\nabla_\theta J
=\mathbb E_{a\sim\pi_\theta}
\left[r(x,a)\nabla_\theta\log\pi_\theta(a\mid x)\right].
\]

用采样动作近似期望就是 score-function estimator，经典序列版本常称 REINFORCE。Reward 本身不需要可微；梯度通过 sampled action 的 log probability 进入 policy。

这不表示离散采样被“直接求导”。对动作取样仍不可微，恒等式把目标梯度改写成可用样本估计的期望。

### 2.1 Baseline 不改变期望的条件

可减去不依赖当前 sampled action 的 baseline \(b(x)\)：

\[
\mathbb E[(r-b)\nabla\log\pi]
=\mathbb E[r\nabla\log\pi]
-b\nabla\sum_a\pi(a)
=\nabla J.
\]

Baseline 可以依赖 state/context，但不能偷看本次 action 后再任意变化。Action-dependent baseline 若没有对应修正，通常会引入 bias。

常用 \(b=V(s)\) 近似预期 return。它往往降低方差，但 expected reward baseline 不一定是整个梯度向量总方差下的最优常数；最优值还与各动作 score-gradient norm 有关。

### 2.2 精确可运行对照

运行：

~~~powershell
python projects/single-gpu-finetuning/policy_gradient_toy.py
python -m pytest tests/test_policy_gradient.py -q
~~~

脚本枚举三动作 policy，而不是依赖 Monte Carlo 恰好接近答案。固定 logits [-0.4, 0.1, 0.3]、rewards [0, 1, 4] 时：

- exact expected reward 约为 2.081241；
- exact gradient 约为 [-0.446381, -0.382343, 0.828724]；
- baseline 为 0、expected reward 或最小方差常数时，估计器期望相同；
- 三者 total variance 约为 2.609983、0.884520、0.784108。

测试还用 central finite difference 独立核对 exact gradient。这个 control 没有执行环境、模型或 Monte Carlo sampling，不能证明某个 RL 训练稳定或 reward 有效。

## 3. 从 bandit 到 MDP

Markov Decision Process（MDP）包含：

- state \(s_t\)：当前决策所需的信息；
- action \(a_t\sim\pi_\theta(\cdot\mid s_t)\)；
- transition \(P(s_{t+1}\mid s_t,a_t)\)；
- reward \(r_t\)；
- discount \(\gamma\) 与 return \(G_t=\sum_{k\ge0}\gamma^k r_{t+k}\)。

在一个纯文本 response 中，可把 prompt 与已生成 prefix 视为 state，下一个 token 视为 action。调用工具后，state 还包括可信观察、权限状态、预算和外部环境结果，已经不只是 token prefix。

“Markov”表示当前 state 对未来足够，不表示现实世界天然可观测。若模型只能看到部分环境信息，更接近 POMDP；此时 history、belief 或外部 memory 是 state estimator 的一部分。

### 3.1 Terminated 与 truncated

**terminated** 表示环境进入真正终态，后续 value 为 0。**truncated** 表示因长度、超时或采集预算停止，是否 bootstrap next value 取决于任务语义。

两者都不应让 GAE recursion 穿过 trajectory 边界。把长度截断当成功终止会系统性扭曲长回答、工具任务和超时行为的 value target。

## 4. Credit assignment

终局 reward 要回答“哪些动作贡献了结果”。核心函数是：

\[
V^\pi(s)=\mathbb E[G_t\mid s_t=s],
\quad
Q^\pi(s,a)=\mathbb E[G_t\mid s_t=s,a_t=a],
\quad
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]

Advantage 比较一个动作相对当前 state 平均行为好多少。它既是 credit signal，也充当 state-dependent baseline。

LLM 常只有 response-level reward。把同一个终局 reward 复制给所有 token 是合法但高方差的 Monte Carlo 估计；引入 token-level KL、value model 或 process reward 会改变 credit 分配，也会增加新的模型误差和攻击面。

## 5. Actor–critic、TD 与 GAE

Actor 是 policy，critic 估计 value。一步 TD residual 为：

\[
\delta_t=r_t+\gamma b_tV(s_{t+1})-V(s_t),
\]

其中 \(b_t\) 明确控制是否 bootstrap。Generalized Advantage Estimation（GAE）递推为：

\[
A_t=\delta_t+\gamma\lambda c_t A_{t+1},
\]

\(c_t\) 在 episode boundary 和 padding 处为 0。较小 \(\lambda\) 更依赖 value、方差低而 bias 可能更大；较大 \(\lambda\) 更接近 Monte Carlo return。

运行现有 NumPy control：

~~~powershell
python projects/single-gpu-finetuning/ppo_objective_toy.py
python -m pytest tests/test_ppo_objectives.py -q
~~~

它分别验证 terminal、truncated bootstrap、padding gap、GAE recursion 和 PPO clipping，不运行 rollout engine 或语言模型。

## 6. On-policy、off-policy 与数据身份

On-policy 方法要求数据由当前 policy 或可控的旧版本生成；policy 改变后，旧数据的分布身份也改变。PPO 保存 old log probability，在有限轮更新内使用 probability ratio 修正。

Off-policy 方法允许复用其他 behavior policy 的数据，但需要重要性权重、value learning 或其他校正假设。没有保存 behavior policy、sampling config、prompt/template 和 mask，就不能事后可靠恢复 ratio。

SFT 数据和固定 preference pairs 是 offline 数据；PPO rollouts 是 online-policy 数据。把旧 response 文本重新 tokenize 后假设它来自当前 policy，不等于拥有原始 behavior log probabilities。

## 7. PPO 中的三个 policy

LLM PPO 常同时出现：

1. **current policy** \(\pi_\theta\)：正在更新；
2. **old policy** \(\pi_{old}\)：生成本轮 rollout，用于 ratio；
3. **reference policy** \(\pi_{ref}\)：通常冻结，用于 KL regularization。

三者不能用一个 base model 名称代替。Old policy 会周期性刷新；reference 的用途是限制相对初始行为漂移。

Clipped surrogate 为：

\[
L^{clip}=\mathbb E_t\left[
\min(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)
\right],
\qquad
\rho_t=\exp(\log\pi_\theta-\log\pi_{old}).
\]

正 advantage 在 ratio 过大时截上界；负 advantage 在 ratio 过小时截下界。Clip 不是 full-distribution KL 的硬约束：未采样动作的概率可以剧烈变化，而 sampled ratio 仍等于 1。

## 8. GRPO-style group-relative advantage

Group Relative Policy Optimization 类方法对同一个 prompt 采样一组 responses，计算 reward 后在组内构造相对 advantage。一个常见教学形式是：

\[
A_i=\frac{r_i-\bar r}{\sqrt{\operatorname{Var}(r)}+\epsilon}.
\]

这样可以不训练独立 value model，但没有消除 baseline、方差或 credit assignment 问题：

- 同组 rewards 全相等时 advantage 为 0，当前 group 没有相对学习信号；
- group 太小会让 mean/std 噪声很大；
- response 相关、重复或由同一错误模式产生时，有效样本量更低；
- reward scale、standard-deviation convention、token reduction 和 KL 处理会改变目标；
- 只有 response reward 时，同一 advantage 仍会作用于一串 token log probabilities。

policy gradient toy 对 [0,1,4,4] 得到零均值、单位标准差 advantages；对 [2,2,2,2] 明确返回 degenerate zero vector，而不是用 epsilon 制造伪信号。

“GRPO”在论文和库中并不唯一决定 clipping、reference KL、长度归一化、group std、old policy 或 multi-epoch 更新。复现时必须写出实际公式、mask 和代码版本。

## 9. RLVR、ORM 与 PRM

Reinforcement Learning with Verifiable Rewards（RLVR）使用可程序核验的结果信号，例如数学答案、代码测试、schema 或模拟器状态。它降低部分人工标注成本，但只优化 verifier 可见的目标。

- Outcome Reward Model（ORM）或 outcome verifier 评价最终答案；信号便宜但稀疏。
- Process Reward Model（PRM）评价中间 step/state；信号更密，但必须处理多条合法路径和局部标签可靠性。

“Verifiable”不等于“不可利用”。常见 loophole 包括答案 parser 容错、测试覆盖不足、超时被算作通过、环境状态没有清理，以及模型把 hidden test 信息带入上下文。

必须把 verifier score 与独立 task success 分开。Best-of-N 中 oracle success 上升、proxy score 上升而 verifier-selected success 下降是完全可能的。

## 10. Preference optimization 方法谱系

先按数据和分布分类，再看算法名：

| 方法族 | 数据 | 是否在线采样 | 核心问题 |
|---|---|---:|---|
| SFT | 单个理想 response | 否 | 拟合 demonstration likelihood |
| RM + PPO | rollout + reward/value | 是 | 在受约束策略更新中最大化 reward |
| DPO/IPO 类 | chosen/rejected pairs + reference 语义 | 否 | 学 reference-relative preference gap |
| ORPO 类 | SFT target + preference pair | 否 | 联合 likelihood 与 pairwise preference term |
| KTO 类 | desirable/undesirable unary feedback | 否 | 从非成对反馈构造 reference-relative utility |
| SimPO 类 | chosen/rejected pairs | 否 | 用长度归一化 policy score 和 margin，避免显式 reference forward |
| GRPO/RLVR | 同 prompt 的 sampled group + reward | 是 | 用组内相对信号更新生成 policy |

这些名称不能替代目标公式。是否有 reference、loss 是 sum 还是 mean、chosen/rejected 是否同源、长度怎样归一化、KL 在 reward 还是 loss 中，都必须从固定实现核对。

## 11. Reward hacking 与 optimizer's curse

Policy 会寻找 reward 中最容易提高的方向，而不是理解设计者真正意图。这是 Goodhart-type failure：代理指标成为优化目标后，原有相关性可能失效。

增加采样或搜索还会产生 optimizer's curse。候选越多，越容易找到 verifier 高估的异常项；因此同时报告：

- oracle@N 与 verifier-selected@N；
- proxy score 与真实 success；
- verifier calibration、OOD 和 adversarial slices；
- calls、tokens、费用、wall-clock 和 tail latency。

训练 reward curve 只能用于诊断训练过程，不能替代 held-out outcome evaluation。

## 12. LLM RL 的 mask 与分母

一个 batch 中至少有 prompt tokens、response tokens、padding、EOS 和可能的 tool/environment tokens。必须明确：

- 哪些 token 计入 policy log probability；
- 哪些 token 拥有 advantage；
- value loss 和 entropy 的分母；
- per-token、per-sequence 或 per-prompt group 的权重；
- response 长度增加是否获得更多 loss weight；
- truncated、invalid 和 verifier error 是否留在 attempted 分母。

先对一条 trajectory 打印 token IDs、mask、old/current/ref log probabilities、reward、advantage 和 return，再扩大 batch。

## 13. 方法选择

| 现有证据 | 推荐起点 |
|---|---|
| 有可靠 demonstrations，基础行为还不稳定 | SFT |
| 有固定 pairwise preferences，不需要在线探索 | DPO 类 offline objective |
| 有可复用 preference scorer | 先训练并审计 RM，再决定是否上线 RL |
| 环境反馈依赖 policy 行为 | PPO/其他 online RL |
| 有高精度程序 verifier 和足够多可解题目 | RLVR/GRPO-style 路线，同时保留人工与安全切片 |
| 问题是权限、实时事实或副作用可靠性 | RAG、工具和 runtime policy，不先靠 RL |

复杂算法应回答一个具体 baseline failure。若 SFT 已达到上限的原因是数据错误或 evaluator 泄漏，增加 online RL 只会更快优化错误目标。

## 14. 证据阶梯

1. **公式 control**：有限 action 枚举、finite difference、mask 和边界测试；
2. **toy optimization**：真实更新小 policy，确认 objective 与行为同向；
3. **small model**：固定 tokenizer/checkpoint，记录 rollout 与完整训练状态；
4. **held-out evaluation**：任务、安全、长度、reward shortcut 和 verifier OOD；
5. **system evaluation**：吞吐、失败、取消、费用、回滚与事件响应；
6. **目标环境**：真实用户 outcome 和独立安全审阅。

本仓库已覆盖前两层中的 categorical gradient、GAE/PPO objective，以及若干 tiny text/Transformer policy 更新；没有完成目标大模型 RLVR、真实 PRM、分布式 rollout 或线上用户实验。

## 15. 常见错误结论

- “Reward 不可微，所以不能训练”：score-function 对 policy log probability 求梯度，不要求 reward 可微。
- “减 baseline 会改变最优 policy”：action-independent baseline 不改变 estimator expectation，但会改变方差。
- “PPO clip 等于 KL trust region”：clip 只约束 sampled surrogate 的局部形状。
- “GRPO 不用 critic，所以没有 value/baseline 问题”：组内均值就是一种随机 baseline，组构造决定信号质量。
- “测试通过就是 verifiable truth”：测试只证明实现覆盖的条件，parser 和环境也可能被利用。
- “更多采样一定更好”：selection proxy 失校准时，Best-of-N 可以越选越差。
- “RL 后 reward 上升就是能力提升”：还需独立任务、安全和分布外评测。

## 自测与实践

1. 从 softmax Jacobian 推导 \(\nabla_{z_j}J=p_j(r_j-J)\)。
2. 为什么 baseline 可以依赖 prompt，却不能未经修正依赖 sampled response？
3. 给定 terminal 与 truncated trajectory，分别写出 bootstrap 和 continuation mask。
4. 解释 old policy 与 reference policy 在 PPO 中为何不是同一个角色。
5. 组内四个 reward 全相同时，GRPO-style advantage 为什么应该为 0？
6. 为代码 RLVR 写出三个可能被利用的 test/verifier loopholes。
7. 设计一个同时报告 reward、真实 success、长度、KL 和成本的发布表。

## 一手资料

- Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*, 1992。
- Sutton & Barto, *Reinforcement Learning: An Introduction*, second edition。
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017。
- Ouyang et al., *Training language models to follow instructions with human feedback*, 2022。
- Rafailov et al., *Direct Preference Optimization*, 2023。
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024。
