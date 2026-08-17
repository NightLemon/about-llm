# 对齐进阶：从偏好数据到 DPO 与 PPO

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已理解 SFT，想进一步掌握奖励模型、DPO、PPO 和对齐评测的工程师。
- **先修**：[对齐入门](alignment-basics.md)、条件概率、梯度优化与基础策略学习。
- **首次阅读**：偏好数据 → Reward Model → KL → DPO；PPO 放在第二遍。
- **完成信号**：能从数据、目标、状态和失败模式四方面比较 SFT、RM、DPO 与 PPO。
- **卡住时**：先运行偏好数据审计，再用一对 chosen/rejected 手算 DPO log-ratio。

</div>

**学习入口**：[对齐入门](alignment-basics.md) · [微调总览](finetuning.md) · [单卡微调项目](../practice/projects/single-gpu-finetuning.md) · [对齐证据台账](../evidence/alignment-controls.md)
{ .doc-nav }

对齐（alignment）不是把一个 loss 降下来，而是让模型行为在明确的人群、规则和场景中更符合目标，并用系统约束处理训练无法保证的权限与副作用。

本章沿着一条数据流学习：示范建立行为先验，偏好比较表达 trade-off，Reward Model 把比较压成代理分数，DPO 或 PPO 再改变 policy。

每一步都可能继承上一步的偏差，因此最终判断必须回到独立任务、安全和系统评测。

## 先写清“对齐给谁”

同一个回答可能帮助终端用户，却违反系统所有者的数据政策；也可能符合多数标注者偏好，却伤害某一语言或群体。

项目开始前先固定：

- instruction hierarchy 与不可覆盖约束；
- 目标用户、语言、地区和专业水平；
- helpfulness、truthfulness、harmlessness 等维度怎样权衡；
- 必须拒绝、可以安全替代和应正常帮助的边界；
- 哪些决定必须交给人或 policy engine；
- 申诉、纠错和 incident response。

“人类偏好”不是客观无噪声标量。它是特定标注者、rubric、界面、候选模型和时间共同产生的观测。

## 先看完整数据流

~~~mermaid
flowchart LR
    A["任务与约束"] --> B["SFT 示范"]
    B --> C["SFT policy"]
    C --> D["采样候选"]
    D --> E["偏好 judgments"]
    E --> F{"训练路线"}
    F --> G["DPO / offline preference"]
    F --> H["Reward Model"]
    H --> I["PPO / online RL"]
    G --> J["candidate policy"]
    I --> J
    J --> K["独立评测与发布"]
~~~

SFT、DPO 和 PPO 的输入并不相同：

| 方法 | 主要数据 | 直接优化什么 |
|---|---|---|
| SFT | prompt + target response | target token likelihood |
| Reward Model | prompt + pairwise judgment | preferred 与 rejected 的分数差 |
| DPO | prompt + chosen/rejected + reference | reference-relative pair classification |
| PPO/RL | policy rollouts + reward/value | 受约束的期望回报 |

如果基础格式、知识和指令跟随还没有稳定，直接增加 preference objective 不会自动补齐这些能力。

## SFT 建立行为先验

Response-only SFT 的目标是：

\[
\mathcal L_{\mathrm{SFT}}
=-\sum_{t\in \mathrm{response}}
\log \pi_\theta(y_t\mid x,y_{<t}).
\]

高质量示范可以教会格式、语气、任务流程和拒答样例。但一个 target 会把多解任务压成单一示范，也会复制教师的冗长、风格和错误。

训练前必须打印 token IDs、role boundaries 和 labels，确认：

- prompt token 是否正确 mask；
- assistant 区域是否真的有监督；
- truncation 是否切掉关键 response；
- chat template 是否与部署一致；
- tool result 与 assistant action 是否被正确区分。

Loss 下降只说明模型更接近训练 targets，不说明目标任务、安全或事实性已经改善。

## 偏好数据首先是一份测量记录

一条 pairwise example 不应只剩 chosen 和 rejected。至少保存：

~~~json
{
  "prompt_id": "p-1042",
  "prompt": "...",
  "candidate_a": "...",
  "candidate_b": "...",
  "presentation_order": ["b", "a"],
  "label": "a",
  "strength": "slight",
  "rubric_revision": "help-safe-grounded-v3",
  "annotator_pool": "domain-experts-cn",
  "generator_revisions": ["model-x@rev", "model-y@rev"]
}
~~~

还要记录语言、风险、group、split、生成参数和 policy version。Tie、invalid 和 disagreement 是数据，不应为了适配 trainer 偷改成 winner。

### 为什么要保留原始 judgment

最终多数标签无法回答：

- 标注者是否一致；
- A/B 展示顺序是否影响选择；
- 某一人群或语言是否系统性不同；
- 一个强偏好由多少条独立 judgment 支持；
- adjudication 是否覆盖了分歧。

关键切片应保留逐标注者结果、展示顺序、rubric、耗时、盲化与 adjudication。Agreement 和 Fleiss’ κ 描述一致性，但一致也可能是共享偏差。

### 常见标注捷径

- **Position bias**：偏好先显示或左侧候选。
- **Verbosity bias**：更长回答显得更充分。
- **Style bias**：标题和自信措辞掩盖错误。
- **Authority bias**：未经核验的引用看起来可信。
- **Identity leakage**：标注者猜出候选模型。
- **Criterion collapse**：多个维度被压成含糊的“更好”。

通过随机交换 A/B、隐藏身份、分维度标注、允许 tie，并在关键切片上双人标注，降低但不能消除这些偏差。

## Reward Model 学的是相对排序

给定 prompt \(x\)、preferred response \(y_w\) 和 rejected response \(y_l\)，Bradley–Terry 模型写成：

\[
P(y_w\succ y_l\mid x)
=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
\]

对应 loss 为：

\[
\mathcal L_{\mathrm{RM}}
=-\log \sigma(r_w-r_l)
=\operatorname{softplus}(-(r_w-r_l)).
\]

只有 reward difference 影响 pair probability。给同一 prompt 的两个 reward 同时加常数，预测不会变化，所以绝对 reward 不是可跨模型解释的“用户价值单位”。

### 从线性 RM 看懂 shortcut

先设 response 特征为 \(f(x,y)\)，线性 scorer 为：

\[
r_w(x,y)=w^\top f(x,y).
\]

假设训练集中 preferred 回答通常更长，模型可能给 length feature 很大的正权重。Pair accuracy 会提高，但它学到的是数据捷径，而不是事实性或帮助性。

因此 RM 评测除了 held-out pair accuracy，还应加入：

- 长度匹配 pairs；
- 风格改写但事实不变的 pairs；
- 事实错误却措辞自信的 hard negatives；
- 语言、风险、领域和 generator 切片；
- score distribution、margin 与 calibration。

Transformer RM 只是把特征提取换成模型 hidden states 和 scalar head，并没有消除 shortcut。

## KL 约束回答“不要走太远”

若直接最大化 learned reward，policy 会寻找 RM 的漏洞。常见目标加入相对 reference policy 的 KL penalty：

\[
\max_\theta\;
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[
R(x,y)-\beta
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right].
\]

更完整地写，就是 reward 减去 \(\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\) 的期望。

\(\beta\) 大时更保守，小则允许 policy 为追求 reward 移动更远。它不是安全系数，也不能阻止 reference 本身已有的错误。

工程上要同时记录 reward、KL、response length、entropy 和真实任务指标。只看总 objective，可能掩盖 reward 上升完全来自更长输出或分布漂移。

## DPO：直接学习 reference-relative 偏好

DPO 不单独训练一个在线使用的 RM。它把偏好模型与 KL-regularized policy 的关系代入一个 pairwise classification objective。

先定义 policy 对 pair 的 log-probability gap：

\[
\Delta_\theta
=\log\pi_\theta(y_w\mid x)
-\log\pi_\theta(y_l\mid x),
\]

reference gap 为：

\[
\Delta_{\mathrm{ref}}
=\log\pi_{\mathrm{ref}}(y_w\mid x)
-\log\pi_{\mathrm{ref}}(y_l\mid x).
\]

DPO loss 可写为：

\[
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(
\beta(\Delta_\theta-\Delta_{\mathrm{ref}})
\right).
\]

直觉是：candidate policy 应比 reference 更偏向 chosen，而不是只让 chosen 的绝对概率变大。

### 一对样本怎样流过 DPO { #target-qwen-dpo-control }

对同一个 prompt，分别渲染 chosen 和 rejected：

1. 用同一 tokenizer/template 得到 token IDs。
2. 只在 response tokens 上累加 log probability。
3. 分别由 policy 和 frozen reference 计算 sequence log-probability。
4. 得到两个 gap，再计算 logistic loss。
5. 梯度只更新 policy 或 adapter。

Prompt masking、EOS、truncation 或 template 任一错位，都会让目标变成另一件事。

### Beta、长度和 reduction

DPO 的 \(\beta\) 控制 reference-relative preference signal 的尺度。它不是简单学习率，改变它会改变 loss 对 log-ratio 的敏感度。

Sequence log-probability 是 token log-probability 的和时，较长 response 拥有更多项；若改成平均，又得到不同目标。训练、验证和不同实现必须明确 sum/mean、mask 与 length normalization。

DPO 仍依赖可靠 pairs、固定 reference、support overlap 和独立评测。它省掉显式 RM + online rollout，不等于消除了对齐假设。

## PPO：在 policy 自己的输出上学习

PPO 适合需要在线采样、环境反馈或可验证 reward 的场景。它比 DPO 多出完整 rollout 状态：

~~~text
prompt
→ old policy samples response
→ reward / verifier scores trajectory
→ value estimates returns
→ advantages
→ clipped policy and value updates
~~~

Token 级 probability ratio 为：

\[
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}.
\]

Clipped surrogate 的最大化形式是：

\[
\mathbb E_t
\left[
\min\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
\]

Clip 限制单轮更新幅度，但不是训练稳定或安全的保证。RLHF 还常有 reference KL、value loss、entropy、reward normalization 和 response masks。

### Reward 怎样分配给 token

很多任务只在序列结束时得到一个 score。Value model 和 Generalized Advantage Estimation（GAE）用于估计每一步相对预期的好坏。

若 terminal reward、KL penalty、padding mask 或 bootstrap terminal 处理错误，代码仍可能反向传播，却优化错误目标。

PPO 验收至少观察：

- old/new/ref log probabilities；
- reward、KL、advantages 与 returns；
- clip fraction、entropy 与 value error；
- response length 和 invalid rate；
- held-out task、安全与 shortcut slices。

### 为什么 PPO 更容易产生证据错觉

Reward curve 上升可能来自 RM shortcut、输出变长、采样分布改变或 verifier loophole。Tiny CPU rollout 通过只能证明局部数学和状态转移，不证明目标 checkpoint 已完成 RLHF。

PPO 只有在 reward 可审计、在线探索确有价值、计算预算允许且回滚机制成熟时，才优先于更简单的 SFT/DPO 路线。

## DPO、RM+PPO 怎样选择

| 条件 | 更适合的起点 |
|---|---|
| 有高质量示范，基础行为未稳定 | SFT |
| 有固定、可靠的 offline pairs | DPO |
| 需要复用可解释的 preference scorer | Reward Model |
| 需要 policy 在线探索或环境反馈 | PPO / Online RL |
| 有确定判据的任务 | Verifier + sampling/RL，可同时保留人工切片 |
| 问题本质是权限或实时事实 | 系统 policy、工具或 RAG，不先靠训练 |

复杂方法只有在能回答“它解决了哪一个 baseline 失败”时才值得加入。

## RLAIF、原则与 Verifier

RLAIF 用模型根据 rubric 或 principles 提供反馈，可以扩大标注规模，但不会消除人类治理。人仍要制定原则、校准 evaluator、抽检分歧并定义发布门槛。

Outcome reward 只检查最终结果；process supervision 还检查中间步骤。后者可能提供更密信号，但也需要可靠的 step annotations 或 verifier。

程序测试、数学答案和 schema validator 提供可验证信号，却仍可能有测试不完整、解析漏洞和 reward hacking。Verifier pass 不是普遍正确性证明。

训练信号应组合：

- 可验证的 task outcome；
- 人工审阅的过程/安全切片；
- 对 evaluator shortcut 的 adversarial cases；
- 与真实用户结果的独立评测。

## 拒答是一个三分类问题

只测 harmful prompt refusal，会把“全部拒绝”误判成安全。至少划分：

| Case | 正确行为 |
|---|---|
| 明确允许且安全 | 正常帮助 |
| 可通过降风险处理 | 安全替代或澄清 |
| 明确禁止或越权 | 拒绝并给出合适边界 |

同时报告 under-refusal 和 over-refusal，并按语言、措辞和风险切片。训练得到的拒答行为不能替代工具 ACL 和数据权限。

## 对齐评测回到真实结果

Offline evaluation 使用固定 prompts，对 baseline/candidate 做 paired comparison：

- task success 与 format correctness；
- factuality、citation 与 abstention；
- preference win rate 与 ties；
- safety、under/over-refusal；
- length、style 和 generator slices；
- KL、reward 与能力回归；
- latency、token 和成本。

所有指标保留 attempted 分母，不能只统计成功解析的样本。Model-as-judge 需要与人工标注校准，并检查位置、长度、self-preference 和 contamination。

上线再观察业务 outcome、申诉、事故、fallback 和用户切片。在线提升不能反向证明训练算法是唯一原因；prompt、routing 和流量组成也可能变化。

## 系统层仍要强制约束

模型对齐不能保证：

- 用户身份与资源所有权；
- 工具 schema、参数和金额合法；
- 写操作幂等、审批和副作用完成；
- 敏感数据不进入日志或外部 provider；
- 高风险决策具备人工复核。

运行时需要认证、授权、budget、sandbox、effect receipt、audit 和 rollback。模型输出是 proposal，不是 capability token。

## 一个渐进式学习实验

选择一个资源可承受的 Instruct checkpoint：

1. **SFT baseline**：验证 template、labels 和 held-out behavior。
2. **Preference audit**：保留 A/B、tie、order、generator、group 和 split。
3. **DPO control**：固定 policy/reference，手算一对样本的四个 sequence log-probabilities。
4. **训练运行**：从单 batch overfit 到小规模 adapter，保存 before/after artifact。
5. **独立评测**：比较任务、安全、长度、KL 和通用能力。
6. **RM/PPO 扩展**：只有 DPO 无法解决在线探索需求时再增加。

实现、命令、Qwen target DPO control 和各种 negative cases 见[对齐证据台账](../evidence/alignment-controls.md)。

## 常见错误

- 把“人类偏好”当作跨人群通用的客观标量。
- 丢弃 tie、invalid、展示顺序和逐标注者 disagreement。
- RM pair accuracy 高就认为没有长度或风格 shortcut。
- DPO 不固定 reference、template、mask、reduction 和 beta。
- PPO reward 上升就声称真实任务与安全提升。
- 用 tiny fixture 拼成目标 checkpoint 已完成 RLHF 的结论。
- 只测 harmful refusal，不测安全请求的 over-refusal。
- 用模型行为替代工具权限、幂等和人工审批。

## 面试时怎样回答

面对“解释 RLHF/DPO”，按数据流回答：

1. SFT 用 demonstrations 建立 policy prior。
2. Pairwise judgments 可以训练 RM，或直接进入 DPO。
3. DPO 优化 policy 相对 reference 的 chosen/rejected log-ratio。
4. PPO 在 old policy rollouts 上，用 reward、value、advantage 和 clip 更新。
5. 两条路线都必须回到独立任务、安全、shortcut 和系统评测。

继续追问时，应能解释 reward shift 不可辨识、DPO sequence reduction、PPO old/ref policy 的不同角色，以及 reward hacking 为什么不是简单调小学习率能解决。

## 自测

1. 为什么最终 chosen/rejected 标签不能替代原始 judgments？
2. Bradley–Terry RM 的绝对分数为什么不能解释成用户价值单位？
3. DPO 中 policy gap 为什么还要减去 reference gap？
4. PPO clip 限制了什么，又没有保证什么？
5. 一次 DPO loss 下降后，还缺哪些证据才能发布？

## 继续学习

- [单卡微调项目](../practice/projects/single-gpu-finetuning.md)：SFT、LoRA 与 DPO 的渐进路线。
- [SFT 数据闭环](sft-data-pipeline.md)：模板、labels、数据治理和切分。
- [Agent Runtime](../applications/agent-runtime.md)：权限、副作用和回放。
- [对齐证据台账](../evidence/alignment-controls.md)：严格数据契约、公式 controls 与命令。
