# 评测统计：从“分数更高”到可信结论

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要比较 Prompt、模型、RAG、微调或服务版本的开发者和算法工程师。
- **先修**：[数学基础](math.md)中的概率、均值、方差与基本 Python。
- **首次阅读**：Estimand → 配对差值 → 区间 → 切片与多重比较 → 发布门禁。
- **完成信号**：能写出一次模型比较的采样单位、分母、效应量和不确定性。
- **卡住时**：先为四个 cases 手算 baseline/candidate 差值，不要先调用统计库。

</div>

**学习入口**：[评测测量学](../quality/evaluation-measurement.md) · [评测方法](../quality/evaluation-methodology.md) · [Evaluation Gate](../practice/projects/evaluation-gate.md)
{ .doc-nav }

“模型 A 得分 82，模型 B 得分 80”还不是结论。你需要知道测量的对象是什么、case 从哪里来、两分是否配对、失败是否进入分母，以及观察差异有多不确定。
若还没有验证 construct、rubric、标注 reliability 与 criterion validity，先读[评测测量学](../quality/evaluation-measurement.md)；统计精度不能修复错误测量。

本章不把统计学变成公式目录，而是回答一次真实发布决策中的五个问题。

## 第一个问题：你究竟想估计什么

Estimand 是你希望从数据推断的目标量。它至少包含：

~~~text
目标用户与任务分布
+ 独立采样单位
+ baseline / candidate 系统身份
+ 输入、工具与生成预算
+ 指标与聚合方式
+ 时间窗口
~~~

例如：

> 在固定季度中文客服目标流量分布上，使用相同知识库、工具和输出预算时，candidate 相对 baseline 的 case-level verified task success 平均差。

这比“哪个模型更强”多了关键边界。若一个实验按人工题库均匀采样，另一个按线上流量加权，它们估计的不是同一个量。

### 系统身份不只包括 model ID

比较对象通常是：

~~~text
model + prompt/template + sampling
+ RAG index/retriever
+ tool schemas/policy
+ parser/verifier
+ runtime
~~~

只换模型也要固定其他组件；同时改 Prompt、RAG 和 verifier，则结论属于整个系统 bundle，不能只归因模型。

## 第二个问题：采样单位和分母是什么

一个 case 可以产生多个 attempts、candidates、tool steps 和 judge records：

~~~text
case
→ attempts
→ candidate outputs
→ parsed result
→ policy decision
→ verified outcome
~~~

最终指标不能悄悄只保留 parsed-success。Timeout、refusal、invalid JSON、OOM 和 provider error 都属于 attempted 分母，除非 estimand 明确排除。

至少同时报告：

- total cases；
- attempted/completed/parsed/verified-success；
- failure taxonomy；
- overall 和关键 slices；
- missing/abstain 的处理规则。

### 聚类会制造伪样本量

同一用户的 100 个对话、同一文档的 100 个切片或同一 prompt 的 20 个采样，并不一定是 220 个独立观察。

若变化主要发生在用户/文档层，应把 user/document 当 bootstrap cluster。把相关记录当独立 case 会低估 standard error，让区间过窄。

## 第三个问题：为什么优先做配对比较

对同一 case \(i\)，记录 baseline \(b_i\) 和 candidate \(c_i\)，分析差值：

\[
d_i=c_i-b_i.
\]

平均提升：

\[
\bar d=\frac1n\sum_{i=1}^{n}d_i.
\]

配对设计让每个 case 自己充当难度对照，通常比比较两组独立均值方差更小。

### 一个四 case 例子

| Case | Baseline | Candidate | Difference |
|---|---:|---:|---:|
| A | 0 | 1 | +1 |
| B | 1 | 1 | 0 |
| C | 0 | 0 | 0 |
| D | 1 | 1 | 0 |

Candidate 的 success rate 是 0.75，baseline 是 0.50，paired mean difference 是 +0.25。

但样本只有四个，结论非常不稳定。正确表述是“这四个固定 cases 上改善 1 个”，而不是“真实提升 25%”。

### 配对要求相同 case identity

Baseline 与 candidate 必须使用同一输入、gold、工具状态和评测规则。若一边 timeout 后被删除，或两边使用了不同 case revision，配对关系已经破坏。

随机模型输出可以使用固定 seed policy，但不要误以为相同 seed 保证不同模型产生可逐 token 对齐的随机轨迹。

## 第四个问题：差异有多不确定

样本均值：

\[
\bar x=\frac1n\sum_i x_i.
\]

样本方差：

\[
s^2=\frac1{n-1}\sum_i(x_i-\bar x)^2.
\]

独立同分布近似下，mean 的 standard error 约为 \(s/\sqrt n\)。真实 LLM cases 常不满足简单独立同分布，因此要让 resampling unit 与数据生成过程一致。

### Paired bootstrap 在做什么

对 case IDs 有放回采样，每次同时取该 case 的 baseline 和 candidate，重新计算 \(\bar d\)：

~~~text
original pairs: A B C D
bootstrap draw: B B D A
recompute mean difference
repeat many times
~~~

Bootstrap distribution 可以形成区间或 improvement probability。它不能修复：

- 题库污染；
- case 选择偏差；
- judge 错误；
- 重复或泄漏；
- 错误的独立采样单位；
- 样本覆盖不足。

### Confidence interval 不表示什么

频率学 95% confidence interval 的严格含义是：若反复执行同一采样与区间构造过程，长期约 95% 的区间覆盖真实参数。

对一次已经算出的 interval，不能简单说“真实值有 95% 概率在这里”，除非使用明确的 Bayesian model 和 prior。

小样本、极端离散指标、heavy tails 或强依赖数据会让普通 bootstrap CI 不稳定。报告 case 数、分布、cluster 选择和敏感性，不只给上下界。

## 效应量先于显著性

统计显著只表示在某些假设下，零差异与数据不太相容。它不回答差异是否足以抵消费用、延迟或安全风险。

发布前定义 minimum practical effect：

~~~text
quality mean difference ≥ δ
and lower confidence bound ≥ required floor
and safety regression ≤ tolerance
and latency/cost within cap
~~~

一个极小提升可以因样本巨大而显著；一个重要提升也可能因样本不足而区间宽。决策应同时看 point estimate、interval、业务阈值和风险。

## 多重比较会制造“最好结果”

尝试 30 个 prompts、10 个模型和 20 个 slices，只报告最好的一个，会产生 winner's curse。

降低风险的方法：

- 预注册 primary metric 与 release rule；
- 把探索集与 final test 分开；
- 披露总共尝试了多少比较；
- 必要时做 multiplicity correction；
- 将事后发现明确标为 exploratory；
- 新发现用独立数据确认。

反复查看 final test 并继续调 Prompt，会把 test 变成训练反馈。

## Overall 可能掩盖切片退化

Simpson's paradox 的直觉是：流量配比变化可能让 overall 提升，即使每个关键组都退化；反过来也可能发生。

至少按任务相关维度报告：

- 语言和地区；
- 风险等级；
- 输入/输出长度；
- tool 类型；
- 来源/领域；
- 用户群或租户；
- provider/runtime failure。

关键安全 slice 使用 guardrail，不让平均质量提升抵消越权或泄露。

切片太多又会回到多重比较问题。预先指定核心 slices，其他作为诊断。

## LLM-as-judge 是一个测量仪器

Judge 不是真值来源。它也可能有：

- position bias；
- verbosity/style bias；
- self-preference；
- 语言和领域偏差；
- rubric misunderstanding；
- parser/format failure；
- 与被评模型共享的盲区。

固定 judge model/revision、prompt、rubric、sampling 和 parser；随机交换 A/B 顺序，并保留 tie/invalid。

用人工标注集校准：

| 问题 | 可能指标 |
|---|---|
| Winner 是否一致 | agreement / accuracy |
| 是否漏判关键错误 | recall by failure type |
| 是否偏好更长答案 | win rate by matched length |
| 不同语言是否可靠 | slice agreement |
| Judge 是否稳定 | repeated-decision consistency |

高相关性也不能证明 judge 在关键安全错误上有足够 recall。保存 disagreement examples，比只报一个 correlation 更有学习价值。

## 一个发布门禁怎样组合指标

一次 release decision 可以按顺序做：

1. **Artifact gate**：case、baseline、candidate、scorer identity 完整。
2. **Protocol gate**：全部 attempts 有 typed terminal，分母无丢失。
3. **Quality gate**：primary effect 与 interval 达标。
4. **Safety gate**：关键风险无超过容忍度的退化。
5. **System gate**：latency、cost、capacity 与 failure rate 达标。
6. **Human review**：审阅新增失败和关键 disagreements。

任何一项 fail，都不应该被其他平均分补偿。门禁规则要在看 candidate 结果前固定。

## 可运行配对实验

从四个手算 cases 开始，再运行仓库 toy：

~~~powershell
python projects/evaluation-gate/paired_randomization_toy.py
python projects/evaluation-gate/clustered_bootstrap_toy.py
python -m pytest tests/test_evaluation_statistics.py -q
~~~

实验前预测：

1. Case-level 与 cluster-level resampling 的区间谁更宽？
2. 加入大量同文档近重复 cases 后，naive sample size 怎样变化？
3. 删除 zero-difference cases 会怎样偏置 estimate？

运行后保存逐 case differences、cluster IDs、seed、resample 次数和完整命令。

Toy 输出证明统计实现对固定输入的行为，不证明你的 case set 有代表性或发布结论有效。

## 常见错误

- 只说“模型更强”，没有 estimand。
- 只统计解析成功样本，删除 timeout/refusal/error。
- Baseline/candidate 使用不同 case revision，却仍做 paired test。
- 把同一用户/文档的重复切片当独立样本。
- 把一次 95% CI 解释成参数有 95% 概率位于其中。
- 只看 p-value，不定义最小业务效应。
- 尝试很多 Prompt/指标后只报告最佳结果。
- 用 overall 掩盖关键语言或安全 slice 退化。
- Judge 分数很高就当 ground truth，不做人工校准。
- 两个 authored cases 的 bootstrap 通过就声称生产提升。

## 面试时怎样回答

面对“如何比较两个 LLM 系统”，按五步回答：

1. 定义目标流量、采样单位、系统 bundle、预算和 primary metric。
2. 对同一 cases 做 paired baseline/candidate 运行，保留全部 failures。
3. 计算 per-case difference，并按真实独立单位 bootstrap/cluster。
4. 同时报告 effect、interval、关键 slices、安全和成本。
5. 预先固定 release gate，并审阅 judge disagreements 和新增失败。

继续追问时，应能说明 paired design 为什么降方差、confidence interval 的频率学含义，以及为什么统计显著不等于值得发布。

## 自测

1. 为什么同一文档的 100 个切片不一定等于 100 个独立 cases？
2. 配对比较要求 baseline 与 candidate 固定哪些 identity？
3. Bootstrap 能处理抽样不确定性，却不能修复哪些问题？
4. Overall 上升时，你会检查哪些 slices 和流量权重？
5. 怎样验证一个 LLM judge 足以用于你的 release gate？

## 继续学习

- [评测方法](../quality/evaluation-methodology.md)：指标、错误分类与实验设计。
- [Evaluation Gate](../practice/projects/evaluation-gate.md)：可运行统计与发布 artifact。
- [Agent 评测](../quality/agent-evaluation.md)：task success、安全与副作用。
- [RAG 生产实践](../applications/rag-production.md)：retrieval、citation 与端到端分层。
- [数学基础](math.md)：概率、数值稳定和优化。
