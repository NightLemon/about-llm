# LLM 评测方法、统计与发布决策

评测不是在几个例子上“感觉更好”，而是把产品目标转成 case、指标、切片、统计比较和发布门禁。对 RAG、Agent、微调和模型升级使用同一 case identity 与报告协议，才能知道变化来自系统还是样本。

## 从决策开始

先写评测要支持的决策：是否上线新模型、选择 RAG 还是微调、是否降低量化位宽、某风险场景能否自动化。不同决策需要不同数据和阈值。一个综合 benchmark 分数无法替代业务决策。

定义：用户群、任务、成功/失败代价、输入分布、约束、基线和最小有意义改善（Minimum Detectable/Meaningful Effect）。如果 0.2 分变化不会改变产品，就不应围绕它无限调参。

## Case schema

一条 case 建议包含：

~~~text
case_id, input, context/environment,
reference/rubric, gold evidence/state,
slices, risk, source, license,
created_at, version, annotator metadata
~~~

`case_id` 跨系统稳定，便于配对比较和错误追踪。reference 可能有多个等价答案或 rubric，不强迫开放任务 exact match。保留原始来源和许可，防 benchmark 变成无治理的数据副本。

## 数据集构成

- 真实分布集：代表主要流量和权重；
- 能力挑战集：刻意覆盖困难/长尾，不用于估算总体发生率；
- 安全/政策集：作为 guardrail；
- 回归集：每个线上事故和重要 bug 转为 case；
- 私有 holdout：防止开发过拟合；
- 动态/时间集：检验 freshness 和分布变化。

报告时不要把挑战集的均值称为线上准确率。各集合目标不同，可分别设门禁。

## 切片

总体平均会掩盖退化。切片来自产品风险和错误假设：语言、地区、任务、输入长度、用户层级、风险、工具、来源、时间、是否多跳/否定/数字。切片过多会产生偶然波动，预先指定关键切片，探索性发现需在新数据复验。

小切片报告样本数和区间，不给一个看似精确的百分数。高风险切片即使少，也可要求零违规或逐例审查。

## 指标类型

### 确定性指标

JSON Schema、单元测试、数据库最终状态、权限、引用 ID、exact match、数值容差。只要适用，优先于 judge。

### 文本相似

token F1/ROUGE/BLEU 可用于抽取或翻译，但对同义开放回答有限。embedding 相似度也可能给事实相反句高分。指标必须与人工偏好做相关性检查。

### 任务指标

分类 F1/AUROC、检索 recall/nDCG、代码 pass@k、工具状态、RAG 引用、Agent 成功/副作用。它们比通用文本分更接近决策。

### 概率校准

若系统在结果发生前给出二分类事件为真的概率 \(p_i\)，观测标签为 \(y_i\in\{0,1\}\)，Brier score 为：

\[
\operatorname{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
\]

它是 proper scoring rule，兼顾概率的 calibration 与 discrimination；越低越好。但不同 base rate、任务或时间窗的 Brier 不能脱离基线直接比较。

Equal-width Expected Calibration Error（ECE）把 \([0,1]\) 划成 bins：

\[
\operatorname{ECE}=\sum_b\frac{|B_b|}{N}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.
\]

仓库约定每个 bin 为 `[lower, upper)`，概率 1 进入最后一个 bin，空 bin 不进入明细且贡献 0。ECE 强依赖 bin 数/边界，有限样本有偏且会掩盖 bin 内结构；报告时必须给 binning、count、reliability diagram 和关键切片，不能只给一个小数。

```python
import math

from about_llm.evaluation import binary_calibration

result = binary_calibration(
    labels=[0, 1, 1, 0],
    probabilities=[0.1, 0.8, 0.6, 0.4],
    bins=2,
)
assert math.isclose(result.brier_score, 0.0925)
assert math.isclose(result.expected_calibration_error, 0.275)
```

Probability 必须对应明确事件，例如“这个结构抽取是否完全正确”，并在看到 label 前记录。Token log-prob、模型自述“90% 有把握”、多次回答一致或 judge 分数不自动是任务正确概率；需要单独 predictor/verifier 和目标分布校准。校准良好也不证明事实、安全或 OOD 鲁棒。

### 选择性预测与 risk-coverage

允许 abstain/人工升级时，对 threshold \(\tau\)：

\[
\operatorname{coverage}(\tau)=\frac{\#\{i:c_i\ge\tau\}}{N},\qquad
\operatorname{risk}(\tau)=1-
\frac{\sum_{i:c_i\ge\tau} y_i}{\#\{i:c_i\ge\tau\}}.
\]

这里 \(c_i\) 是“越大越可信”的 score，\(y_i\) 是 correctness。仓库 `risk_coverage_curve` 对每个唯一 confidence 设 threshold，并把相同 confidence 的样本一起接受；否则任意输入顺序会改变曲线。

```python
from about_llm.evaluation import risk_coverage_curve

curve = risk_coverage_curve(
    correctness=[1, 0, 1, 0],
    confidence=[0.9, 0.8, 0.8, 0.1],
)
assert [(p.accepted_count, p.coverage) for p in curve] == [
    (1, 0.25),
    (3, 0.75),
    (4, 1.0),
]
```

降低 risk 往往以降低 coverage 和增加人工量为代价。比较系统时报告整条曲线或固定业务 coverage 下的 risk，并按语言、风险和用户群切片。只展示最低 risk 的极高阈值可能只回答极少样本；总体曲线也不能保证每个关键群体都安全。

### 人工与 LLM judge

适合流畅度、帮助性、复杂 rubric；成本高或有偏差。judge 不能读取隐藏系统状态就无法判断权限/副作用。使用前在人工集校准。

## 聚合

每 case 保留原始输出和每项分数，报告均值之外的中位数、分位数、失败率与分布。macro average 让每类等权，micro average 让高频样本权重大；选择要对应产品。

多指标不要随意加权成一个分数。质量、延迟、成本可以画 Pareto front；安全/权限作为硬门禁。若确需 utility，权重和单位必须由业务代价定义并做敏感性分析。

## 配对比较

新旧系统在相同 case 上运行，分析每 case 差值 \(d_i=s_i^{new}-s_i^{base}\)。配对设计消除 case 难度方差，比两个独立均值更有力。

Bootstrap 对 case 重采样，得到平均差的置信区间。若 case 可近似独立，仓库 `paired_bootstrap` 固定 seed 输出 mean difference、percentile interval 和经验改善比例，并把 index matrix 分块限制在不超过一百万元素。若存在用户/文档 cluster，应按 cluster 重采样；逐行 bootstrap 会把 cluster 内相关误当成额外独立信息。仓库 `clustered_paired_bootstrap` 为此提供完整 cluster resampling，但它与逐行版本都不是 BCa interval，也不保证小样本 coverage。

置信区间包含 0 不等于“完全相同”，可能样本不足；不包含 0 也不等于业务上重要。同时看 effect size 和最小有意义阈值。

Bootstrap 区间与假设检验不是同一个输出。仓库 `paired_randomization_test` 另固定 case-level sign-flip 检验：在“sharp null 下每对 baseline/candidate 标签可交换”的假设中，把每个非零差值的符号独立翻转，得到均值差的 null distribution。若有 \(m\) 个非零 pair，exact 路径枚举 \(2^m\) 种符号；精确零差值保留在原 pair 数与均值分母，但不重复产生完全相同的正负 assignment。运行：

~~~powershell
python projects/evaluation-gate/paired_randomization_toy.py
~~~

固定 5 对 authored score 中，4 对 candidate 比 baseline 高 1、1 对相同，因此 observed mean difference=0.8。预先指定 `greater` 时，16 种符号只有全正一项达到 observed statistic，exact p=1/16；two-sided 还包含全负一项，p=2/16。样本过多无法穷举时，reference 用 seeded Monte Carlo，并以 `(extreme+1)/(draws+1)` 避免报告零 p-value，同时输出分辨率。

P-value 不是“null 为真的概率”，也不表示效果达到业务阈值。`greater/less/two-sided` 必须在看结果前确定，不能先看方向再挑单侧。离线两个确定性系统也不会自动获得因果解释：case sampling、label exchangeability、数据代表性和 metric validity 都是额外假设。

### Cluster 是重采样/随机化单位，weighting 决定 estimand

同一用户、文档或会话中的多条 case 可能相关。若可交换单位是 cluster，把每行独立翻转会把一个 cluster 伪装成多个独立证据。仓库 `clustered_paired_randomization_test` 给同一 cluster 的所有差值乘同一个符号，因此允许 cluster 内任意相关；它仍要求 cluster-level baseline/candidate label exchangeability，以及不同 cluster 的 sign assignment 可独立组合。这些假设必须来自设计，代码不会替你建立。

先明确想估计什么。令 cluster \(g\) 有 \(n_g\) 条差值 \(d_{gi}\)，总 case 数为 \(N\)，cluster 数为 \(G\)：

\[
\hat\Delta_{case}=\frac{1}{N}\sum_{g=1}^{G}\sum_{i=1}^{n_g}d_{gi},
\qquad
\hat\Delta_{cluster}=\frac{1}{G}\sum_{g=1}^{G}\frac{1}{n_g}\sum_{i=1}^{n_g}d_{gi}.
\]

`cluster_weighting="case"` 对应第一项：大 cluster 因含更多 case 而权重更高，sign-flip contribution 是 cluster sum，分母为 \(N\)。`cluster_weighting="equal"` 对应第二项：每个 cluster 等权，contribution 是 cluster mean，分母为 \(G\)。二者都是合法但回答不同问题；不能看结果后选择更显著的一种，也不能把 equal-cluster 结果写成“随机抽一条 case”的效果。

Cluster bootstrap 每次从已观测的 \(G\) 个 cluster 中有放回抽 \(G\) 个，并把被抽中 cluster 的所有 case 一起复制。若 ordered resample 为 \(g_1^*,\ldots,g_G^*\)，两种 bootstrap statistic 分别是：

\[
\hat\Delta_{case}^*=\frac{\sum_{b=1}^{G}\sum_i d_{g_b^*i}}
{\sum_{b=1}^{G} n_{g_b^*}},
\qquad
\hat\Delta_{cluster}^*=\frac1G\sum_{b=1}^{G}
\left(\frac1{n_{g_b^*}}\sum_i d_{g_b^*i}\right).
\]

Case-weighted 路径的分母随抽中的 cluster sizes 改变；把原 \(N\) 固定不动会算错 ratio estimand。Equal-cluster 路径才是直接重采样 cluster means。运行透明 fixture：

~~~powershell
python projects/evaluation-gate/clustered_bootstrap_toy.py
~~~

仍用 5 条 `+1` 的 A 与 1 条 `-1` 的 B。两个 cluster 的 \(2^2=4\) 个 ordered resample 是 `AA/AB/BA/BB`。Case-weighted statistic 为 `[1, 2/3, 2/3, -1]`，NumPy `linear` 95% percentile interval 为 `[-0.875, 0.975]`，严格大于 0 的经验比例为 3/4；equal-cluster statistic 为 `[1,0,0,-1]`，区间为 `[-0.925,0.925]`，改善比例为 1/4。这个“改善比例”只是 bootstrap distribution 中 statistic>0 的份额，不是候选更好的 Bayesian posterior probability。

Reference 在 cluster 数不超过配置阈值时枚举全部 \(G^G\) 个 ordered resample，以正确保留 multinomial multiplicity；配置硬上限为 7，较大输入走 seeded Monte Carlo。Exact 只消除 Monte Carlo 误差，不修复只有两个 cluster 的 coverage：独立且有代表性的 cluster sampling、稳定 cluster 定义、无跨 cluster interference 和足够 cluster 数仍是外部前提。真实分析还应做最大 cluster sensitivity，并考虑 studentized/BCa、cluster-robust 模型或与设计匹配的方法。

~~~powershell
python projects/evaluation-gate/clustered_randomization_toy.py
~~~

固定 fixture 中 user A 有 5 条 `+1`，user B 有 1 条 `-1`。逐 case `greater` sign flip 把 6 行当成 6 个单位，64 种 assignment 中 7 种至少同样极端，p=7/64。Case-weighted cluster-joint 路径只有 contribution `[5,-1]` 的 4 种联合符号，其中 2 种至少达到 observed \(4/6\)，p=2/4。Equal-cluster estimand 则是 \((1+(-1))/2=0\)，two-sided p=1。这个例子不是说 cluster 方法“总会让 p 变大”，而是说明单位与 estimand 会改变 null distribution，必须在看 outcome 前由采样/部署问题决定。

Cluster contribution 恰为 0 时仍保留 cluster/case 数和 observed denominator，但不重复相同正负 assignment。Reference exact 枚举最多允许 24 个非零 sign-flip unit；更大问题走 seeded Monte Carlo plus-one 路径，以每个 block 不超过一百万个 sign 元素限制临时内存。即便如此，少量 cluster、单个超大 cluster、数据依赖 cluster 定义、跨 cluster 干扰或不代表目标流量仍会让结论不稳；需要报告 cluster 数/size 分布、最大 cluster sensitivity、weighting 和抽样设计。

### 多重比较与 Holm step-down

如果同时检查 \(m\) 个 metric、slice 或候选 Prompt，即使所有 null 都成立，逐项按 0.05 判断也会增加“至少一个误报”的概率。先在看结果前定义一个 family，再把其中有效的原始 p-value 排序为 \(p_{(1)}\le\cdots\le p_{(m)}\)。Holm adjusted p-value 为：

\[
\tilde p_{(i)}=\min\left(1,\max_{1\le j\le i}\{(m-j+1)p_{(j)}\}\right).
\]

最后将 adjusted p-value 映射回原 hypothesis，并在 \(\tilde p_i\le\alpha\) 时拒绝。前缀 running maximum 不能省略：它保证按 rank 的 adjusted value 单调，也等价于 Holm 的 sequential reject rule。只要 family 中每个输入 p-value 在其 null 下有效，Holm 在任意依赖结构下控制 family-wise error rate（FWER）；它通常比把每项都乘 \(m\) 的单步 Bonferroni 更有力。

~~~powershell
python projects/evaluation-gate/holm_correction_toy.py
~~~

固定输入顺序 `[0.04, 0.01, 0.03, 0.20]` 排序后 multiplier 为 `[4,3,2,1]`，scaled value 为 `[0.04,0.09,0.08,0.20]`，running maximum 得到 sorted adjusted `[0.04,0.09,0.09,0.20]`；映回输入顺序为 `[0.09,0.04,0.09,0.20]`，所以 \(\alpha=0.05\) 只拒绝原索引 1。仓库 tie 按原输入顺序稳定排序，但相同 p-value 经 running maximum 后得到相同 adjustment。

Holm 不会修复无效的 component p-value，也不会让“先试很多再挑一个好看的 family”、反复窥视/可选停止或测试集调参变得有效；这些属于 selection/sequential protocol。FWER 控制也不是 effect size、置信区间、业务重要性或因果证明。必须同时报告原始/adjusted p-value、family 定义、全部测试、effect 与实际阈值，不能只展示被拒绝项。

## 多次采样

temperature>0、Agent 工具环境和 provider 都可能随机。可对每 case 多次运行，报告 pass@1、平均成功、worst-case 或 pass@k。pass@k 是给 k 次机会至少一次成功，不能和单次生产成功率混用。

比较时可共享 seed/请求顺序，但 provider 实现不一定严格可复现。保存全部 run，不挑最好结果。

## 延迟与成本

同一 workload、并发、客户端位置和 warmup 比较。报告 TTFT/TPOT/E2E 的 p50/p95/p99、错误/取消，并按输入/输出长度切片。只比较平均延迟会被短请求主导。

成本包含 token/API、GPU/CPU、检索/rerank、存储、人工升级和重试。质量降低导致的人工成本可能超过模型费用节省。

## LLM-as-judge 协议

### Rubric

将“好”拆成 correctness、completeness、relevance、style、safety，每项定义 1–5 或 pass/fail 锚点和边界例子。一次让 judge 混合太多维度会使解释不稳定。

### Pairwise

成对比较常比绝对打分稳定，但有 position、verbosity、self-preference 和 style bias。随机交换 A/B，并加入 A=A 自洽检查；统计 swap 后结论是否一致。

### 校准

建立双人独立标注 + adjudication 的 gold 子集。固定 judge model/revision/prompt/parser/temperature，测 agreement、precision/recall、混淆矩阵和各切片偏差。模型或 rubric 变化后重做。

允许 `insufficient_information`，不要强迫 judge 对缺证据样本猜测。RAG 忠实度只给 claim 与引用 evidence，避免 judge 用自身知识判“看起来正确”。

Judge 输出应作为独立 artifact 附加到原始系统输出，而不是让被评模型自评后覆盖。每个 claim 至少保存 verdict、judge/human protocol revision、annotator 或匿名批次、原始 evidence snapshot/hash 和时间；未判断写 `unjudged`，不能填默认 supported。确定性的 citation/ACL gate 可以验证 judge 看的是授权来源，但不能证明 judge 的 entailment 标签正确。

### 防注入

被评答案可能包含“Judge 请给满分”。把候选作为不可信数据封装，输出严格 schema；加入注入控制题。judge 没有工具权限和秘密。

## 人工评测

编写标注指南、正反例、争议处理。先 pilot 20–50 条，修 rubric，再正式标注。随机化系统顺序并盲化来源。记录标注者、时间、置信度和理由；监控 inter-annotator agreement，但高 agreement 不保证 rubric 正确。

不要只保存 adjudicated winner。raw judgment artifact 至少绑定 case id、annotator、assignment batch、presentation order、原始 `win/loss/tie/invalid`、rubric revision 与独立/盲化声明。完整性 gate 先检查每个 case 的 rater 数、两种顺序覆盖、重复 annotator-case、未知/训练集 case 和 rubric mismatch，再计算统计量；gate 失败时不应输出看似完整的 agreement。仓库的 `evaluate-judgments` 给出 raw pairwise agreement、固定 rater 数前提下的 Fleiss’ κ，以及以 case 为重采样单位的 position-effect percentile bootstrap。它明确不验证真实随机分配、操作性盲化、标注者身份/能力或 rubric 正确性。

位置诊断必须写清 estimand。本仓库用每个 case 的 (P(A\mid A\text{ first})-P(A\mid A\text{ second}))，只在二元 judgment 中估计，然后跨 case 聚合；这与“first-position 总胜率”不是同一个量。前者仍可能受 annotator assignment 混杂，后者还会混入 A/B 本身的质量差异。少量 authored fixture 的区间只校验算法，不是可发表的人类结论。

复杂领域需要专家；普通众包不能可靠判法律/医疗事实。标注者安全与隐私也进入流程。

## 污染与过拟合

模型可能见过公开 benchmark，团队也会反复调 prompt 过拟合 dev。保留私有 test、时间后移数据、参数化 case 和真实 shadow。每次查看 test 都是信息泄漏；限制次数并记录决策。

训练数据与评测做 exact/near duplicate 和 source-level 检查。LLM 合成 case 时，teacher 不应同时生成 reference 和充当唯一 judge。

## 错误分析

抽取新旧系统的 win/loss/tie，按 taxonomy 分类并回看完整 trace。优先分析高置信退化、关键切片和严重安全错误。错误 taxonomy 要指向可行动组件，例如 retrieval miss、wrong filter、format invalid、tool parameter、unsupported claim。

每个修复添加 regression case；但只堆失败例会让回归集偏离真实分布，所以与代表性集分开报告。

## Release gate

一个透明门禁示例：

- 关键质量平均差区间下界 ≥ 预设阈值；
- 任何关键切片下降不超过容忍度；
- 权限、安全、结构合法率满足硬阈值；
- p95 TTFT/TPOT 和单位成本在预算；
- 无新增严重错误，所有结果/配置 artifact 完整。

门禁返回全部失败原因。通过离线门禁后仍需 shadow/canary；离线数据无法覆盖真实分布和系统故障。

## 在线实验

A/B 随机单位要避免干扰和跨组污染；聊天产品通常按用户而不是请求分组。预先定义主要指标、guardrail、样本量、停止规则和实验时长，避免每天偷看后提前停止。

安全事件、延迟和成本是 guardrail。新模型输出可能改变用户行为，离线一致不保证在线价值。canary 可先小流量验证故障，再正式 A/B 判断产品效果。

## 可复现 artifact

每次运行保存 system/model/prompt/index/tool/policy version、case manifest hash、原始输出、结构分数、错误、usage/latency、环境和 judge 配置。结果 JSONL 原子写入，失败也是一等结果，不被跳过。

只比较 `case_id` 集合仍不够：同一个 ID 下的 input、expected、slice 或 metadata 可能已经改变，metric 名称背后的实现也可能升级。仓库的 evaluation run manifest 对 ordered case 全语义、ordered scored result、recorded answer、metric name→revision、scorer revision 与 caller-supplied `system_id` 做 canonical SHA-256 binding；`compare` 在 bootstrap 前用当前 artifact 重算，并拒绝 case/result 漂移、metric revision 不一致、duplicate JSON key、未知字段和 `NaN/Infinity`。

这解决的是“评测 A、比较 B”的可检测一致性，不是来源认证。无密钥 hash 与调用者自报 `system_id` 不能证明输出来自所称模型，也不能证明 case 代表真实分布、gold 正确、metric 有 construct validity 或发布会产生线上收益。生产证据还需可信 runner、签名/append-only 存储、访问控制、时间戳以及 model/prompt/index/judge/runtime 的实际 revision。

最终 gate JSON 也必须是 artifact，而不是可手改的报告。仓库 comparison v2 把 resampling unit、confidence/sample/seed、cluster metadata key、case/equal weighting、exact threshold、质量/安全/延迟阈值、protected slice 阈值、两侧 run manifest、metric revision、统计结果、全部失败原因和 evidence boundary 一起 fingerprint。Cluster result 逐项保存 case/cluster count、cluster sizes、estimand、exact/Monte Carlo、实际 resample 数、linear quantile 和有效 seed；loader 复核 sizes 总和、estimand difference、exact \(G^G\) 数、改善比例的 exact fraction、配置一致性、失败原因和最终 pass/fail。v2 不静默加载旧 v1。

`verify-comparison` 可严格重载，但机器输出明确标记 `verification_scope: artifact_only`、`referenced_manifests_revalidated: false` 和 `statistics_recomputed: false`；它不会重新打开 cases/results/run manifests、从 metadata 重建 cluster，或重跑 bootstrap。这样能发现已有可信 fingerprint 之后的局部修改，但不能证明 cluster 定义/独立/代表性、interval coverage，不能阻止攻击者协同重写 JSON 并重算无密钥 hash，也不能检测没有可信 head 时的历史截断。

`verify-evidence` 提供更强但仍有限的本地复算：严格重开 cases、baseline/candidate answers、results、run manifests 和 comparison；按 case 顺序重算 recorded-answer/case identity，要求 manifest metric revision 与当前可执行实现精确相同，重新评分，再按 comparison 中固定的 unit、cluster key/weighting、confidence、samples/seed、exact threshold、gate/slice 阈值重跑统计并重建最终 artifact。它能拒绝 answer 漂移、与 answers 不符但 manifest 已重写为自洽的 scores，以及不对应当前 results 的自洽 comparison summary。

机器输出把 `case_semantics_rehashed`、`scores_recomputed`、`run_manifests_revalidated`、`statistics_recomputed` 与 `comparison_rebuilt` 分开列出，同时明确 `artifact_authentication_verified=false`、`model_execution_replayed=false`。原因是完整本地复算仍只回答“这些 bytes 按当前已声明 revision 是否得到同一结果”：能改全套文件的人仍可协同重写，runner/model/provider 是否真的执行、sampling/cluster 假设、construct validity 与 production impact 都需要其他证据。

如果威胁模型包含“能改文件并重算 hash”，需要认证层而不只是 identity 层。仓库的 evaluation release ledger 用 domain-separated HMAC-SHA256 将连续 sequence、release/artifact identity、artifact 原始 bytes 的 size/SHA-256、decision、caller-supplied timestamp、key id 和前序 MAC 串起来，并允许按记录轮换 key。Strict loader 要求 canonical JSON、精确 schema、连续链和唯一 identity；snapshot writer exclusive-create 并 file-fsync。验证可再要求 artifact-id→path 的精确映射来重读当前 bytes。

仍要分清三个命题：MAC chain 通过只认证“相对所提供共享密钥的记录”；artifact rehash 才证明当前读取的引用文件与记录一致；外部 trusted head 匹配才检测删除尾部或回滚到旧的合法 snapshot。没有 trusted head 时，合法前缀必然仍可验证。HMAC 不证明 key custody 或不可否认性，MAC 绑定 timestamp 字符串也不产生可信时间；exclusive-create/file-fsync 不证明 parent-directory durability 或目录原子发布，verify 后还存在 TOCTOU。公开 fixture key 只能测试协议。生产还需 KMS/HSM、ACL/轮换吊销、外部 transparency/object-lock anchor、可信时间和让消费端在使用前强制验证的集成。

报告从原始 artifact 生成，不手工复制数字。仓库 `render-comparison-html` 输出 deterministic、自包含的中文比较页，覆盖 identity、总体/切片统计、case/cluster resampling、gate 原因和 evidence boundary；所有动态文本 HTML escape，页面无 JavaScript/外部资源并带 restrictive CSP。机器 receipt 明确标为 `artifact_only_render`：严格加载不等于重开证据图、认证 artifact 或重算统计，HTML 也不是 canonical artifact。CI 可跑小回归，完整/付费评测按计划或发布触发。

对于校准 evidence package，至少保存稳定 `case_id`、在结果前产生的 probability、最终 binary label、predictor/version、label protocol、时间和 slice。CLI metric input 的每行最低契约是 `case_id`、`label` 和 `probability`，其余版本信息可放不可变 manifest；若先看到结果再填写“置信度”，就不再是前瞻概率评测。仓库提供离线入口：

~~~powershell
python -m about_llm.evaluation.cli calibrate `
  --input projects/evaluation-gate/calibration.example.jsonl `
  --bins 5 `
  --output artifacts/evaluation/calibration.json
~~~

## 面试追问

**新模型平均分高 1%，是否上线？** 先看配对区间、业务最小改善、关键切片、安全、延迟/成本和错误类型；再 shadow/canary。单一平均不足以决策。

**如何验证 LLM judge？** 用独立人工 gold，固定并版本化 judge，测 agreement/precision/recall/切片偏差、位置交换、自洽和注入控制；定期抽检，不能让 judge 自证正确。

**为什么测试集越大不一定越好？** 重复/低质量/错分布样本只让区间看似更窄。代表性、标签可靠、关键长尾和 cluster 独立性比原始行数更重要。
