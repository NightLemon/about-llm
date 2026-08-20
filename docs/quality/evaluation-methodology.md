# LLM 评测方法：从 24 比 22 到发布决定

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要比较系统、解释不确定性并维护发布门禁的评测与算法工程师。
- **先修**：[评测总览](evaluation.md)、均值、比例和置信区间的基本直觉。
- **首次阅读**：先跟一次新旧系统比较，再按需深入 cluster、multiple testing 和 judge。
- **完成信号**：能在看结果前写清 estimand、比较单位、业务阈值和停止规则。
- **卡住时**：先只保留逐 case 差值，不要急着计算 p-value。

</div>

客服助手的 baseline 在 30 条 case 中答对 22 条，candidate 答对 24 条。能不能发布？

`24 > 22` 只告诉我们这个样本上的计数。发布还取决于两条新增正确答案来自哪里，
原先正确的 case 是否退化，高风险切片有没有失败，30 条是否来自 30 个独立用户，
以及 candidate 的延迟和费用怎样变化。

本章从这个决策向后展开评测方法。精确实现、固定 fixture、artifact schema 和 HMAC 证据链集中在
[评测门禁项目](../practice/projects/evaluation-gate.md)与[准确性台账](../evidence/accuracy-ledger.md)，
避免教学主线变成字段说明书。

## 第一步：先写发布问题

评测计划的第一行不是指标名称，而是它支持的决策：

```text
Decision:
  是否用 candidate 替换当前客服 baseline？

Population:
  中文售后会话中的首轮事实问答，不含人工已接管会话。

Primary estimand:
  在目标请求分布上，candidate − baseline 的任务成功率差。

Meaningful effect:
  至少提升 2 percentage points，且关键切片不退化。

Hard guardrails:
  越权、未经支持的高风险答案、schema 失败和严重安全错误。
```

Population、任务、失败代价、基线和最小有意义改善要在运行前固定。
若 0.2 分变化不会改变产品决定，就没有必要围绕它无限调 Prompt。

这里还要区分 Minimum Meaningful Effect 和统计分析能检测到的 Minimum Detectable Effect。
前者来自业务价值，后者受样本量、方差、显著性水平和方法影响。统计功效不足时应增加合适样本，
不能把业务阈值改小来迎合现有结果。

## 一条 case 要能回到真实失败

最小 case schema 可以包含：

```text
case_id
input + context / environment
reference / rubric / gold state
gold evidence
slices + risk
source + license
created_at + version
annotator metadata
```

`case_id` 在系统版本之间保持稳定，便于配对和回归。Input、reference 或 slice 变化时，
case semantic identity 应变化；不能只保留同一个名字，让比较器误以为它还是原 case。

开放任务允许多个等价答案或 rubric。结构化抽取则可以保存唯一 parsed value、字段容差和业务状态。
Case 还要保留来源与许可，避免 benchmark 变成失去治理的数据副本。

## 不同 case 集合回答不同问题

客服评测通常需要几组数据：

| 集合 | 用途 | 报告时怎样解释 |
|---|---|---|
| 代表性集 | 估计主要目标流量表现 | 按采样或目标流量权重聚合 |
| 能力挑战集 | 刻意覆盖困难和长尾 | 观察能力边界，不称为线上发生率 |
| 安全 / policy 集 | 检查不可接受行为 | 独立 hard gate |
| 回归集 | 保存事故和重要 bug | 防止已知失败再次出现 |
| 私有 holdout | 限制开发过拟合 | 减少查看次数，保留最终判断 |
| 时间后移集 | 检查 freshness 与漂移 | 与旧 snapshot 分开报告 |

总体平均很容易掩盖退化。切片应来自产品风险和失败假设，例如语言、地区、长度、用户层级、来源、
高风险意图、多跳、否定与数字。关键切片事前指定；探索性发现要在新数据上复验。

小切片同时展示样本数、原始分子分母和区间。高风险切片即使很小，也可以逐例审查，
而不是用一个小样本百分数制造精确感。

## 指标先声明比较对象

“Exact match” 不是单一含义。至少有四层：

| 层 | 比较对象 | 适合什么 |
|---|---|---|
| Byte identity | 固定 encoding / serialization 后的 bytes | 文件、wire artifact 和完整性 |
| Literal string exact | Decoded code-point sequence | 大小写、空格、标点都重要的文本协议 |
| Normalized exact | 经过明确 Unicode、大小写和空白规则的文本 | 允许表面变体的短答案 |
| Token F1 | 固定 tokenizer 后的 token overlap | 抽取覆盖和有多个局部匹配的答案 |

每次 normalization 都可能吞掉真实错误。仓库的 normalized exact 使用 NFKC、`casefold()`、`strip()`
和 whitespace collapse；token F1 在同样归一化后抽取 ASCII word 或单个中日韩统一表意文字。

例如：

```text
LLM-2026 -> llm-2026
literal / normalized / token-F1 = 0 / 1 / 1

{"answer":42} -> {"answer": 42}
literal / normalized / token-F1 = 0 / 0 / 1
```

第一个例子说明 case-folding 不适合大小写复制任务；第二个说明 token F1 忽略 JSON 标点和结构。
指标由任务契约选择，不能在看完分数后换成最有利的口径。

### JSON 要把 syntax、schema、value 和业务规则分开

结构化评测可以分四步：

1. Strict parse：拒绝 duplicate key、`NaN` 和 `Infinity`；
2. Schema：检查类型、required、额外字段和本地 references；
3. Value：比较 canonical parsed value，object key order 与 whitespace 可忽略；
4. Domain policy：验证账户、库存、金额、权限和当前业务状态。

`{"amount": 100, "currency": "USD"}` 可以 schema-valid，也可以与唯一 gold value 相等，
但仍可能引用错误账户或已失效汇率。开放任务有多个等价对象时，单一 gold value exact 也不合适。

仓库的 JSON Schema metric 把无效 expected schema 当作 case 配置错误，把 malformed model output 记为失败；
`format` 目前是 annotation，不做 coercion，也不应用 `default`。这些行为都属于 metric revision。

### Citation 也有 identity 与语义两层

Source ID 合法、quote 与指定 span 逐字一致，可以证明模型指向了某段 source bytes。
它还没有回答 claim 是否被该段证据支持。

固定反例中，claim “The moon is cheese.” 精确引用 `Earth is round.` 里的 `Earth`。
Span identity 可以通过，而 semantic verdict 应为 unsupported。完整 citation 评测需要分别保存：

```text
source / ACL / snapshot provenance
claim segmentation
source ID + exact span
supported / contradicted / insufficient verdict
answer completeness
final publication decision
```

### 任务指标比通用文本相似更接近决策

分类使用 F1 或 AUROC，检索使用 Recall / nDCG，代码使用测试或 pass@k，
RAG 检查检索、claim 和 citation，Agent 检查业务状态与副作用。

ROUGE、BLEU、token F1 和 embedding similarity 可以补充文本覆盖，
但语义相反的句子也可能拥有很高 overlap 或 embedding similarity。使用前要检查它们与人工或业务目标的关系。

## 当系统给出置信度时，先定义事件

概率 \(p_i\) 必须在观察 label 前产生，并对应明确事件，例如“这次结构化抽取是否完全正确”。
最终 label 为 \(y_i\in\{0,1\}\) 时，Brier score 是：

\[
\operatorname{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
\]

它越低越好，同时反映 calibration 与 discrimination。不同 base rate、任务和时间窗之间，
需要与各自基线一起比较。

Equal-width Expected Calibration Error（ECE）把概率区间分桶：

\[
\operatorname{ECE}=\sum_b\frac{|B_b|}{N}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.
\]

ECE 对 bin 数和边界敏感，有限样本下也可能有偏。报告应包含 binning、count、reliability diagram 和关键切片，
而不只是一位小数。

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

Token log-prob、模型自述“90% 有把握”、多次回答一致和 judge 分数都不会自然成为任务成功概率。
它们可以作为 predictor 的输入，再在目标分布上校准。

### Risk–coverage 回答“少答一些是否更可靠”

系统可以根据 confidence \(c_i\) abstain 或升级人工。Threshold \(\tau\) 下：

\[
\operatorname{coverage}(\tau)=\frac{\#\{i:c_i\ge\tau\}}{N},
\qquad
\operatorname{risk}(\tau)=1-
\frac{\sum_{i:c_i\ge\tau}y_i}{\#\{i:c_i\ge\tau}}.
\]

降低 risk 通常会降低 coverage，并增加人工处理。比较系统时展示整条曲线，
或固定业务 coverage 后比较 risk；总体曲线还要按语言、风险和用户群切片。

仓库实现把相同 confidence 的样本一起接受，避免输入顺序改变曲线：

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

## 先看逐 case 差值，再看聚合

回到 30 条客服 case。先生成四组：

```text
both correct
baseline only correct
candidate only correct
both wrong
```

`candidate only` 里的两条新增成功，可能被 `baseline only` 里的高风险退化抵消。
逐 case 结果能直接定位这种交换；只看 24 和 22 看不到。

每条 case 保存 raw output、各指标、终态和错误分类。聚合时根据决策选择 mean、median、quantile、failure rate 和分布。
Macro average 让类别等权，micro average 让高频 case 权重更高，两者对应不同用户问题。

质量、延迟和成本更适合用 Pareto front 展示。权限和严重安全错误通常是 hard gate，
不应与帮助性加权成一个“综合 87.3 分”。若业务确实定义 utility，权重、单位和敏感性分析也要公开。

## 新旧系统为什么要配对

让 baseline 和 candidate 回答同一 case，定义：

\[
d_i=s_i^{candidate}-s_i^{baseline}.
\]

配对差值消除了“这条 case 本来就更难”的一部分方差，也让每个 win/loss 可以回看原始输出。

### Bootstrap 区间回答效果的不确定性

若 case 可近似视为独立抽样单位，可以有放回重采样 case，并对每次样本计算平均差。
Percentile interval 描述当前设计和重采样假设下 effect estimate 的不确定性。

区间包含 0 可能是效果小，也可能是样本不足；区间不含 0 仍要与最小有意义 effect 比较。
仓库 `paired_bootstrap` 固定 seed，并限制临时 index matrix 大小。它不是 BCa interval，
也没有替真实数据建立独立性、代表性或小样本 coverage。

### Randomization test 回答 sharp null 下有多极端

在配对标签可交换的 sharp null 下，将每个非零 \(d_i\) 的符号翻转，形成 null distribution。
若有 \(m\) 个非零 pair，exact 路径枚举 \(2^m\) 个 sign assignments。

~~~powershell
python projects/evaluation-gate/paired_randomization_toy.py
~~~

固定 5 对 score 中，4 个差值为 `+1`，1 个为 `0`，observed mean difference 为 0.8。
事前指定 `greater` 时 exact p-value 为 `1/16`；two-sided 为 `2/16`。

P-value 描述当前 null 与检验设计下，至少同样极端结果的概率。它不是 null 为真的 posterior probability，
也不表示效果达到了业务阈值。`greater`、`less` 和 `two-sided` 要在看方向前确定。

## 独立单位是用户时，不能把每行当一个人

假设 30 条 case 只来自 8 个用户，同一用户的提问方式和难度高度相关。
逐行 bootstrap 或 sign flip 会把相关记录当成额外独立信息。

令 cluster \(g\) 有 \(n_g\) 条差值 \(d_{gi}\)，总 case 数为 \(N\)，cluster 数为 \(G\)。
两个常见 estimands 是：

\[
\hat\Delta_{case}=\frac{1}{N}\sum_{g=1}^{G}\sum_{i=1}^{n_g}d_{gi},
\qquad
\hat\Delta_{cluster}=\frac{1}{G}\sum_{g=1}^{G}
\frac{1}{n_g}\sum_{i=1}^{n_g}d_{gi}.
\]

- Case-weighted 回答“随机请求平均改善多少”，大用户因贡献更多请求而权重更高。
- Equal-cluster 回答“随机用户平均改善多少”，每个用户等权。

它们是两个产品问题，不是看完结果后任选的统计技巧。

Cluster bootstrap 每次有放回抽 \(G\) 个完整 cluster。Case-weighted statistic 的分母随抽中 cluster sizes 改变；
equal-cluster 则平均抽中的 cluster means。Cluster randomization test 也要给同一 cluster 的全部差值共享一个 sign。

运行透明反例：

~~~powershell
python projects/evaluation-gate/clustered_bootstrap_toy.py
python projects/evaluation-gate/clustered_randomization_toy.py
~~~

Fixture 中 user A 有 5 条 `+1`，user B 有 1 条 `-1`：

| 分析问题 | Observed effect / p-value |
|---|---|
| 逐 case sign flip | mean `4/6`，greater `p=7/64` |
| Case-weighted cluster sign flip | mean `4/6`，greater `p=2/4` |
| Equal-cluster sign flip | mean `0`，two-sided `p=1` |

结果不同，因为 independent unit 和 estimand 变了。Cluster 方法不会自动让 p-value 变大或变小。
它仍要求 cluster 定义稳定、cluster 间的组合假设合理，并需要足够多且有代表性的 clusters。

少量 cluster 时可以枚举全部 ordered resamples，消除 Monte Carlo 误差；
它不会制造更多独立用户，也不会修复 percentile interval 的小样本 coverage。
报告还应包含 cluster count、size 分布、最大 cluster sensitivity 与抽样设计。

## 同时试很多指标时，要记录完整 family

若同时检查 \(m\) 个 metric、slice 或 Prompt，即使所有 null 都成立，逐项按 0.05 判断也更容易出现至少一个误报。
Holm step-down 先将 p-values 排序为 \(p_{(1)}\le\cdots\le p_{(m)}\)：

\[
\tilde p_{(i)}=min\left(
1,
\max_{1\le j\le i}\{(m-j+1)p_{(j)}\}
\right).
\]

Running maximum 保证 adjusted values 随 rank 单调。只要 component p-values 在各自 null 下有效，
Holm 在任意依赖结构下控制 family-wise error rate。

~~~powershell
python projects/evaluation-gate/holm_correction_toy.py
~~~

固定输入 `[0.04, 0.01, 0.03, 0.20]` 得到 input-order adjusted
`[0.09, 0.04, 0.09, 0.20]`，所以 \(\alpha=0.05\) 时只拒绝原索引 1。

Holm 不能修复无效 p-value、事后挑 family、测试集调参或可选停止。
报告应保留 family 定义、全部原始/adjusted p-values、effect 和业务阈值，不能只展示显著项。

## 多次偷看同一实验，是另一类问题

Multiple testing 处理同一时点的多个 hypotheses；sequential testing 处理同一 hypothesis 被反复查看。
每周用固定样本检验一次，一旦 `p <= 0.05` 就停止，会提高总体假阳性率。

~~~powershell
python projects/evaluation-gate/sequential_peeking_toy.py
~~~

固定 looks 为 `[10,20,30,40,50]`。在 fixture 的 i.i.d. fair-sign null 下，
每次用 0.05 并在首次拒绝时停止，exact familywise rejection probability 约为 `0.1010`；
只在最终 \(n=50\) 查看一次约为 `0.03284`。

若事前确认最多五次 look，并用 Bonferroni 把 0.05 分成每次 0.01，
同一离散 fixture 的 familywise error 约为 `0.01522`。正式实验也可按设计使用 group-sequential boundary、
alpha spending、always-valid p-value / e-process 或 confidence sequence。

这个 toy 只处理无 tie、i.i.d. fair signs。线上 A/B 还要固定随机化单位、最大样本/时长、
主要指标、guardrails、流量 ramp、异常停止和完整 look ledger。

## 随机系统要保存每一次运行

Temperature、Agent 工具环境和 provider 都会引入随机性。每条 case 可以重复运行，
根据产品决策报告 pass@1、平均成功、worst-case 或 pass@k。

Pass@k 表示给 \(k\) 次机会至少一次成功，不能和单次生产成功率混用。
比较系统时可以共享 seed 和请求顺序，但远程 provider 未必严格可复现；原始 runs 全部保存，不能只挑最好一次。

延迟比较固定 workload、并发、客户端位置和 warmup，报告 TTFT、TPOT、E2E 的 p50/p95/p99，
并同时展示错误、取消和输入/输出长度。成本包含 token/API、GPU/CPU、检索、工具、存储、人工和失败重试。

## Judge 是测量仪器，不是答案来源

LLM-as-judge 适合帮助性、风格和复杂 rubric，但它看不到隐藏数据库状态，
无法凭文本判断权限或副作用是否真的发生。可执行 oracle 存在时，优先使用确定性检查。

一套 judge protocol 至少包括：

1. 把 correctness、completeness、relevance、style 和 safety 分开；
2. 为 1–5 或 pass/fail 写锚点和边界样例；
3. 固定 judge model、revision、Prompt、parser、temperature；
4. 在双人独立标注和 adjudication 的 gold subset 上校准；
5. 检查 precision/recall、confusion、position、verbosity 与关键切片；
6. 允许 `insufficient_information`，并加入 Prompt injection controls。

Pairwise judgment 常比绝对分数稳定，但仍有 position、verbosity、self-preference 和 style bias。
交换 A/B 顺序并加入 A=A case，可以发现一部分不一致。

Judge 输出作为独立 artifact 附在系统输出旁边。每个 claim 保存 verdict、protocol revision、annotator/batch、
evidence snapshot 和时间；未判断写 `unjudged`。被评答案是不可信输入，judge 没有工具权限或 secret。

## 人工评测也需要完整原始判断

先用 20–50 条 pilot 修订 rubric，再开始正式标注。随机化呈现顺序并尽可能盲化来源，
记录 annotator、assignment batch、时间、confidence、reason 和原始 `win/loss/tie/invalid`。

Adjudicated winner 不能替代原始 disagreement。完整性 gate 先检查每 case 的 rater 数、两种顺序覆盖、
重复 annotator-case、未知 case 和 rubric mismatch，再计算 agreement 或位置诊断。

仓库 `evaluate-judgments` 提供 raw pairwise agreement、固定 rater 数前提下的 Fleiss' \(\kappa\)，
以及按 case 重采样的 position-effect interval。Fixture 只校验计算口径，
不会证明真实分配随机、盲化有效、annotator 胜任或 rubric 正确。

高 agreement 说明标注者按当前 rubric 一致，不等于 rubric 测到了目标 construct。
法律、医疗和其他复杂领域还需要合格专家，并为标注者设计隐私与安全流程。

## 从错误分析走到 release gate

先对新旧系统的 win/loss/tie 建立可行动 taxonomy：

```text
retrieval miss
wrong ACL / filter
context dropped by budget
unsupported claim
schema invalid
tool parameter / business state error
timeout / outcome unknown
```

每个重要修复加入 regression case，但代表性集与回归集分开聚合，
避免事故样本不断积累后让“平均分”失去线上解释。

一个透明门禁可以要求：

- Primary effect 的区间下界达到事前阈值；
- 关键切片下降不超过容忍度；
- 权限、安全、schema 和高风险回答满足 hard gates；
- p95 TTFT / TPOT 与每成功任务成本在预算；
- 没有新增严重错误，cases、outputs、scores 和配置齐全。

Gate 返回全部失败原因，而不是只给一个红绿灯。离线通过后再进入 shadow 和 canary。
正式 A/B 按用户等合适单位随机，预先固定主要指标、guardrails、样本量、时长和停止规则。

公开 benchmark 可能进入训练，团队也会反复调 dev。使用私有 holdout、时间后移数据、参数化 case 和 source-level
near-duplicate 检查；每次查看 test 都会带来信息，次数与决策应进入实验记录。

## Artifact 让“评测的是 A，比较的也是 A”

每次 run 保存：

```text
system / model / prompt / index / tool / policy revisions
ordered case semantics
raw outputs + terminal failures
metric name + metric/scorer revisions
scores + slices
usage / latency / environment
judge / human protocol
```

Run manifest 将 ordered cases、answers、results 和 metric revisions 绑定起来。
Comparison artifact 再保存 paired unit、cluster key/weighting、bootstrap/randomization config、
质量/安全/延迟阈值、统计结果与所有失败原因。

这里有四层不同保证：

1. **Strict load**：schema、duplicate key 和 non-finite number 合法；
2. **Identity check**：当前 bytes 与 manifest / comparison 中记录一致；
3. **Recomputation**：重新评分并按固定配置重建统计结果；
4. **Authentication / history**：HMAC 或其他认证链，加上外部 trusted head 检测回滚与截断。

无密钥 SHA-256 只能发现已记录内容漂移。它不认证 `system_id`，也不证明模型/provider 真实执行。
HMAC chain 认证的是相对共享密钥的记录；artifact rehash 才检查引用文件，外部 trusted head 才能发现合法前缀回滚。

仓库提供 `verify-comparison` 的 artifact-only 检查和 `verify-evidence` 的本地复算，
并让机器输出明确区分是否重开 cases、重算 scores/statistics、认证 artifact 或重放 model execution。
精确 schema、固定 fixture、HMAC 字段和 HTML renderer 边界见[准确性台账](../evidence/accuracy-ledger.md)。

校准数据还要保存 probability 产生时间、predictor version、最终 label、label protocol 和 slice。
先看到结果再填写“置信度”，已经不再是前瞻概率评测。

~~~powershell
python -m about_llm.evaluation.cli calibrate `
  --input projects/evaluation-gate/calibration.example.jsonl `
  --bins 5 `
  --output artifacts/evaluation/calibration.json
~~~

## 回到 24 比 22

现在可以回答开头的问题：

1. 先列出 30 条 case 的 paired win/loss，而不只比较两个总数；
2. 核对 case 是否代表目标流量，多个 case 是否来自同一用户或文档；
3. 比较 primary effect 与 2 percentage-point 业务阈值，并报告区间；
4. 单独检查安全、权限和关键切片；
5. 联合比较延迟、成本与所有 terminal outcomes；
6. 验证 run/comparison artifact 完整后，再进入 shadow 或 canary。

`24/30` 可能支持发布，也可能只是某个小样本上的暂时领先。
评测方法的作用不是把它包装成一个更复杂的小数，而是让每个发布判断都能回到 case、假设和失败代价。

## 面试追问

**新模型平均高 1%，是否上线？** 看 paired effect、业务阈值、区间、关键切片、安全、延迟和成本，
再通过 shadow/canary 验证真实分布。

**如何验证 LLM judge？** 使用独立人工 gold，固定 judge 协议，检查 confusion、切片偏差、顺序交换、自洽与注入控制，
并保留 raw judgment。

**为什么测试集越大不一定越好？** 重复、相关、错标签和错分布只会让数字看起来更稳定。
独立抽样单位、代表性、标签质量与高风险覆盖比原始行数更重要。
