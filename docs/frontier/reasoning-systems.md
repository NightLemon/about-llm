# 推理系统：从多采样到 Verifier 与工具

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解 reasoning post-training、test-time compute 和推理评测的工程师。
- **先修**：[生成](../core/generation.md)、条件概率与[评测统计](../foundations/evaluation-statistics.md)。
- **首次阅读**：定义任务 → single-sample baseline → self-consistency → best-of-N → tools。
- **完成信号**：能分开 candidate oracle quality、selection quality 和最终系统成功。
- **卡住时**：先在一个可验证二分类任务上比较 1 次与 5 次采样。

</div>

**专题导航**：[前沿总览](reasoning-long-context-moe.md) · [对齐](../training/alignment.md) · [推理服务](../practice/projects/inference-serving.md) · [证据台账](../evidence/frontier-controls.md)
{ .doc-nav }

“让模型想得更久”不是一个完整算法。推理系统通常组合训练得到的行为、test-time sampling/search、verifier 和外部 tools。

学习时始终追踪三个对象：

~~~text
candidate generator
→ candidate set
→ selector / verifier
→ final task outcome
~~~

候选集合中存在正确答案，不表示系统能选出来；verifier 分数上升，也不表示目标任务更正确。

## 先给 reasoning 一个可操作定义

可以测量的 reasoning 任务通常要求：

- 多步组合或约束满足；
- 算法、程序或形式规则执行；
- 跨多个证据的推导；
- 规划并调用工具；
- 在 counterfactual 或新模板上迁移。

可见 chain-of-thought 更长不等于内部推理更强。答案正确也可能来自记忆、数据泄漏或 shortcut。

任务集应包含 held-out templates、难度切片、counterfactuals 和可执行 verifier。Rationale 只是一类可观察输出，不是内部计算的完整真值。

## 训练阶段改变行为先验

### Reasoning SFT

使用问题—轨迹—答案示范训练。轨迹可来自专家、程序或强模型。

主要风险：

- 错误步骤被模仿；
- verbosity/style 被当成推理质量；
- 模板泄漏；
- 只学会 rationale 格式；
- 可见解释与真实计算不忠实。

保留 rejected/error trajectories，可以训练纠错或 verifier；但训练 loss 下降仍要回到独立任务验证。

### Outcome 与 process reward

Outcome reward 只检查最终答案，适合数学答案、代码测试或 schema 等有明确判据的任务。它便宜，但信号稀疏，也容易利用 parser/test loophole。

Process reward 给中间 step/state 打分，能提供更密监督，却需要定义局部正确、第一处错误和多条合法路径。

标注步骤正确，不证明模型内部真的按相同步骤计算。Verifier 也只覆盖 specification 中编码的性质。

训练方法的更多数学见[对齐进阶](../training/alignment.md)。

## Test-time baseline 先用单样本

在增加 sampling/search 前，固定：

- prompt 与 template；
- temperature/top-p；
- max output tokens；
- tool/verifier availability；
- timeout 与成本；
- task verifier；
- case set。

记录 single-sample pass@1、length、latency、cost 和 failure taxonomy。

没有这个 baseline，就无法判断多采样的收益来自更大预算、不同 prompt 还是 selector。

## Self-consistency：多数票依赖错误结构

Self-consistency 从同一问题采样多个候选，将最终答案 canonicalize 后投票。

若每次正确概率为 \(p>0.5\)，候选真正独立，且只有正确/错误两个标签，奇数 \(N\) 的多数票成功率为：

\[
P(\text{majority})
=\sum_{k=(N+1)/2}^{N}
{N\choose k}p^k(1-p)^{N-k}.
\]

真实候选通常不独立。同一难题上的多次采样可能稳定地产生相同错误；共享 prompt、模型和知识盲区会造成相关性。

### 为什么平均 p 大于 0.5 仍可能退化

想象一半是 easy cases，单次正确率 0.9；一半是 hard cases，单次正确率 0.3。总体单样本平均是 0.6。

增加采样会让 easy cases 更稳定地正确，也让 hard cases 更稳定地投错。总体 majority 不一定像 i.i.d. \(p=0.6\) 公式那样上升。

因此要保存 per-case candidates 和 votes，并按 case cluster 估计收益。同一个问题产生的多条采样彼此相关，
统计时应把它们视为同一组，而不是独立 test case。

开放文本还需要 canonicalization。数学等价式、格式差异或多个不同错误答案会让 plurality 与 binary majority 完全不同。

## Best-of-N：候选质量与选择质量分开

Best-of-N 生成 \(N\) 个候选，由 verifier \(v\) 选最大分数：

\[
\hat y=\arg\max_{y_i\sim\pi}v(x,y_i).
\]

至少报告两个指标：

- **Oracle@N**：候选集合中是否存在正确答案；
- **Selected@N**：verifier 最终选择的答案是否正确。

Oracle@N 常随 N 上升，selected@N 可能先升后降。原因是更多采样也增加了遇到 verifier exploit 的机会。

### 一个最小反例

假设采样可能得到：

| Candidate | Probability | Verifier score | Target correct |
|---|---:|---:|---:|
| 普通错误 | 0.5 | 20 | 否 |
| 正确答案 | 0.4 | 80 | 是 |
| Verifier hack | 0.1 | 99 | 否 |

随着 N 增加，至少出现正确答案的概率上升；但至少出现 hack 的概率也快速上升，而且 selector 总会偏向 99 分的错误。

所以 verifier 必须在 adversarial、length-matched 和 distribution-shift cases 上校准。不要只优化 verifier 自己的分数。

## Search 需要显式状态

Tree/graph search 至少定义：

~~~text
state
actions / expansion
value or verifier
backup rule
branching and dedup
stop condition
budget
~~~

自然语言 state 难以判等，分支巨大，模型 value 也可能自信错误。

记录 expanded nodes、model/verifier calls、tokens、wall time、memory 和最终选择。只报准确率会隐藏几十倍搜索成本。

Beam、MCTS、A* 等名称不能代替 state/value/backup 的具体定义。

## Reflection 只有获得新信息才更可信

让模型“再检查一次”可能只产生新措辞或更自信的同一错误。

Reflection 更有价值的情况包括：

- unit tests 返回失败；
- compiler/solver 给出结构化错误；
- retrieval 带来新证据；
- verifier 指出可定位的约束违背；
- 人工反馈给出具体修正。

保存每轮 input、delta、feedback、stop reason 和 budget，防止无限循环。

## Tool-augmented reasoning

Calculator、Python、SQL、retrieval 和 proof assistant 可以把可验证子任务外包。

模型 proposal 仍需：

~~~text
complete arguments
→ schema/domain validation
→ authorization
→ budget/approval
→ execution
→ result validation
→ next reasoning step
~~~

工具结果可能错误、过期或含提示注入。Tool use 提升的是系统可访问的操作，不自动证明模型内部推理更强。

## Test-time scaling 画完整曲线

对预算 \(B\) 同时画：

- verified quality vs sampled tokens；
- quality vs model/verifier calls；
- quality vs wall-clock/cost；
- p95 latency 与 failure rate；
- oracle@N vs selected@N；
- 不同难度和领域 slices。

收益可能先升后降。低质量轨迹会污染 context，search 可能走偏，verifier exploit 随候选增多更常出现，超时也会进入分母。

停止策略应根据 marginal utility、confidence 或硬预算，而不是假定越长越好。

## 推理评测的常见泄漏

- 公开题模板进入预训练或合成数据；
- Exact parser 对格式敏感；
- Pass@k 与 pass@1 被混用；
- 同题多采样被当独立 test cases；
- Judge 偏好更长 rationale；
- 反复用 benchmark 选择 prompt；
- Final 正确但过程错误，或相反。

使用 case-level paired evaluation，并保留独立 final holdout。可验证任务优先执行 verifier，但仍审计 specification 和 loopholes。

## 可见 reasoning 与隐私

Reasoning text 可能包含 system instructions、用户敏感信息、工具 secrets、错误假设或未核验内容。

不要默认把内部 trajectory 全量：

- 返回给用户；
- 写入普通日志；
- 用作训练数据；
- 跨用户/会话重放；
- 当作事实或授权证据。

Public answer、audit trace 和 provider opaque state 使用不同 projection 与访问控制。

## 一个可运行实验

选择一个答案可执行验证的小 case set：

1. 运行 single sample baseline。
2. 对每题采样 \(N=3,5,9\)。
3. 计算 oracle@N。
4. 分别使用 majority 与弱 verifier 选择。
5. 保存 per-case candidates、normalized answer 和 selector score。
6. 报告 selected success、tokens、calls、latency 与 failures。

先预测：相关错误和 verifier hack 会在哪些 cases 上使 N 增大反而变差。

仓库的精确反例：

~~~powershell
python projects/inference-serving/self_consistency_correlation_toy.py
python projects/inference-serving/verifier_best_of_n_toy.py
~~~

这些小实验只检查几组预设概率分布上的数学关系。目标模型是否改善或退化，需要在它实际生成的候选上测量。
具体计算结果见[证据台账](../evidence/frontier-controls.md)。

## 常见错误

- 用 rationale 长度代替推理正确性。
- 用跨题平均单样本正确率套 i.i.d. majority 公式。
- 只报 oracle@N，不报 selected@N。
- 用 verifier score 上升代替 target success。
- Search 不记录 calls、tokens、memory 和 timeout。
- Reflection 没有新证据，却假定第二次一定更好。
- Tool proposal 绕过权限和 result validation。
- 把同题多个 candidates 当独立评测样本。

## 面试时怎样回答

面对“如何提升模型推理”，先分训练与 test time：

1. SFT/RL 改变 candidate generator。
2. Sampling/search 扩大 candidate set。
3. Verifier/majority 决定 selection quality。
4. Tools 提供外部可验证计算。
5. 发布比较固定 token/call/time budget 和 case-level 分母。

继续追问时，应能解释 oracle@N 与 selected@N 的差别，以及候选错误相关性为什么会破坏 self-consistency 的简单直觉。

## 自测

1. 可见 chain-of-thought 更长，为什么不能直接证明 reasoning 更强？
2. Self-consistency 的 i.i.d. 假设在真实任务中怎样失效？
3. Best-of-N 增加时，为什么 verifier hack 风险会增加？
4. Reflection 在什么条件下获得了真正新信息？
5. 怎样公平比较 single sample 和 search system？
