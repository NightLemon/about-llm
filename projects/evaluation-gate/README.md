# 统一评测与发布门禁

目标：让 RAG、Agent、微调和模型升级使用同一组 case id、切片、统计与发布决策，而不是各自展示几个成功示例。

## 当前实现

- Recall@k、MRR、graded nDCG、实际返回分母 Precision 与 all-evidence recall 共享原语；
- 逐 case 或完整 cluster 配对 bootstrap 的 estimand、置信区间和改善比例；
- 同时约束质量、安全和延迟的透明 ReleaseGate；
- JSONL runner、原子结果写入、exact match 与 token F1；
- 严格 JSON（拒绝重复 key、未知字段和 `NaN/Infinity`）与 versioned run manifest；
- case 全语义、ordered result、recorded answer、metric revision 和 caller-supplied system id 的 canonical binding；
- comparison v2 artifact：绑定 case/cluster resampling unit、estimand、bootstrap 配置、全部 gate 阈值、protected slices、run manifests、统计结果与失败原因；
- 完整本地 evidence graph verifier：重开 answers/results/manifests，重新评分并重建 comparison；
- deterministic HTML comparison report：自包含、无脚本/外部资源、全动态文本转义并显示证据范围；
- HMAC-SHA256 release ledger：key rotation、artifact byte rehash、外部 trusted-head 截断检测与 exclusive-create snapshot；
- JSON Schema 输出合规和已授权来源的引用语法/覆盖率；
- overall 与语言、风险、用户等切片汇总及 Markdown 表格；
- 固定 seed 的可复现测试。
- Binary Brier score、equal-width ECE 和 tie-aware risk-coverage 曲线。
- 固定 look schedule 下 exact doubled-tail sign test 的首次拒绝动态规划，以及 naive peeking 与预设 Bonferroni alpha split 对照。

## 固定 Qwen 真实权重的小型行为评测

通用 Evaluation Gate 默认只处理已记录输出，不会自行调用模型。`run_qwen_target_behavior_evaluation.py` 是一条明确分开的 target runner：它复用 Transformers Basics 中固定的 Qwen2.5-0.5B-Instruct revision、7-file/999,586,347-byte snapshot 和 checkpoint manifest，加载前重哈希，然后以 CPU FP32/eager、batch 1、greedy、`max_new_tokens=12` 对七条 authored case 逐条调用真实 `GenerationMixin.generate()`。

~~~powershell
# 真实重哈希、加载权重并运行七条 case
python projects/evaluation-gate/run_qwen_target_behavior_evaluation.py `
  --local-files-only

# 普通 CI：只复算 strict report，不加载约 1 GB snapshot
python projects/evaluation-gate/run_qwen_target_behavior_evaluation.py `
  --verify projects/evaluation-gate/target-qwen-behavior.recorded-report.json
python -m pytest tests/test_target_qwen_evaluation_control.py -q
~~~

Suite `sha256:27ada9b1…6201` 固定中英文算术、北京/Paris 事实、空证据拒答、大小写复制和 JSON 三类输出约束。Report `sha256:dd30a278…5c43` 保存每条 prompt token identity、continuation IDs、raw decoded output、EOS/cap terminal，并从原始文本重算三种不同指标：

| Case | Raw output | literal exact | normalized exact | token F1 |
|---|---|---:|---:|---:|
| 中文算术 | `42` | 1 | 1 | 1 |
| 英文算术 | `112` | 0 | 0 | 0 |
| 中文/英文事实 | `北京` / `Paris` | 2/2 | 2/2 | 2/2 |
| 空证据拒答 | `无法回答` | 1 | 1 | 1 |
| 大小写复制 | `llm-2026` | 0 | 1 | 1 |
| JSON | `{"answer": 42}` | 0 | 0 | 1 |

总体 literal exact 为 `4/7`，normalized exact 为 `5/7`，token F1 为 `6/7`。这不是三个互相竞争的“模型准确率”，而是三个不同判定函数：case-folding 让大小写错误通过 normalized exact；token F1 又忽略 JSON 标点/空格结构。结构化输出应另做 parse/schema/semantic gate，大小写敏感复制应使用 literal decoded-string exact；若要声明原始响应 byte identity，还必须另行保存并比较 bytes，不能把两者混写，也不能挑最高的 6/7 写进简历。

七条 case 是作者构造且不是外部预注册、独立抽样或 held-out benchmark；没有置信区间、baseline/candidate 比较、judge/人工标注、真实用户分布、长上下文、安全红队、GPU/vLLM、工具/RAG/训练或性能测量。报告中的 CPU 执行只证明固定 snapshot 和固定输入确实运行，不建立总体中文/英文能力、质量、泛化、校准、生产安全、许可或发布者身份。

## 可运行 CLI

CLI 处理**已记录输出**，不擅自调用付费 API。`answers.*.example.jsonl` 模拟两个系统在相同 case 上的输出与延迟；真实流水线应先保存 provider/model/prompt/tool/schema/workload 版本，再把脱敏输出交给 scorer。

分别给 baseline 与 candidate 打分并生成切片 Markdown 报告：

~~~powershell
python -m about_llm.evaluation.cli score `
  --cases projects/evaluation-gate/cases.example.jsonl `
  --answers projects/evaluation-gate/answers.baseline.example.jsonl `
  --results artifacts/evaluation/baseline.results.jsonl `
  --report artifacts/evaluation/baseline.report.md `
  --manifest artifacts/evaluation/baseline.run-manifest.json `
  --system-id deployed-baseline@exact-revision

python -m about_llm.evaluation.cli score `
  --cases projects/evaluation-gate/cases.example.jsonl `
  --answers projects/evaluation-gate/answers.candidate.example.jsonl `
  --results artifacts/evaluation/candidate.results.jsonl `
  --report artifacts/evaluation/candidate.report.md `
  --manifest artifacts/evaluation/candidate.run-manifest.json `
  --system-id deployed-candidate@exact-revision
~~~

默认计算 normalized exact match 与 token F1。若任务要求逐字保留大小写、标点或空格，显式传 `--metric literal_exact_match`；它的 revision 是 `about-llm.literal-exact-match.v1`，比较 decoded string equality，不声称原始 response bytes 相同。`--metric` 可重复传入，例如同时选择 literal/normalized exact，也可选择 `json_schema`、`json_value_exact`、`citation_syntax` 或 `citation_evidence_span`。Schema 指标要求 case metadata 提供 `output_schema`，引用语法指标要求 `valid_source_ids`；span 指标要求 `citation_sources`。三者都不证明 claim-evidence entailment。

### Strict JSON Schema 与 expected-value equality 分账

五条 authored fixture 把“字符串像不像”“JSON 是否严格合法”“是否符合 schema”“parsed value 是否等于 gold”拆开：

~~~powershell
python -m about_llm.evaluation.cli score `
  --cases projects/evaluation-gate/structured-metrics.cases.jsonl `
  --answers projects/evaluation-gate/structured-metrics.answers.jsonl `
  --results artifacts/evaluation/structured.results.jsonl `
  --report artifacts/evaluation/structured.report.md `
  --manifest artifacts/evaluation/structured.run-manifest.json `
  --system-id authored-structured-fixture@v1 `
  --metric literal_exact_match `
  --metric exact_match `
  --metric token_f1 `
  --metric json_schema `
  --metric json_value_exact
~~~

| Case | literal | normalized | token F1 | schema v2 | JSON value v1 |
|---|---:|---:|---:|---:|---:|
| object key order + whitespace only | 0 | 0 | 1 | 1 | 1 |
| wrong value `43` | 0 | 0 | 0.5 | 1 | 0 |
| duplicate object key | 0 | 0 | 2/3 | 0 | 0 |
| `NaN` | 0 | 0 | 0.5 | 0 | 0 |
| reversed array order | 0 | 0 | 1 | 1 | 0 |

`about-llm.json-schema-metric.v2` 使用 strict JSON：拒绝 duplicate object key 与 `NaN/Infinity`；schema 只允许 local `$ref/$dynamicRef`，拒绝 `$id` 和 external resolution。无效 gold schema 是 case 配置错误，会中止评分，不会把所有输出悄悄记为 0。当前 `format` 仍是 annotation，未启用 `FormatChecker`；它也不做 coercion 或应用 `default`。

`about-llm.json-value-exact.v1` 忽略 JSON object key order 与不重要 whitespace，但保留 array order、string、scalar type 以及 parser 的 integer/float distinction。它不自动调用 schema，也不等于业务语义：单位、资源归属、数据库状态、权限和跨字段规则仍需独立 validator。Fixture 的 `latency_seconds=0.0` 是 authored 非性能占位值；五条 case 不证明真实模型、provider、代表性质量或生产安全。

### Citation ID、exact span 与 entailment 分账

`about-llm.citation-evidence-span-metric.v1` 接受 strict JSON claim 列表。每个 claim 必须有唯一非空 `claim_id`、非空 `text` 和至少一个 evidence；每个 evidence 必须只含 `source_id/start_char/end_char/quote`。Case 的 `citation_sources` 是 scorer 收到的授权来源快照；指标检查 source ID membership、零基/end-exclusive Python string offset、逐字 quote equality、duplicate JSON key、重复 claim/span 和未知字段。它不负责证明这个快照真的来自在线 ACL，也不判断 quote 是否支持 claim。

~~~powershell
python -m about_llm.evaluation.cli score `
  --cases projects/evaluation-gate/citation-evidence-span.cases.jsonl `
  --answers projects/evaluation-gate/citation-evidence-span.answers.jsonl `
  --results artifacts/evaluation/citation-span.results.jsonl `
  --report artifacts/evaluation/citation-span.report.md `
  --manifest artifacts/evaluation/citation-span.run-manifest.json `
  --system-id authored-citation-span-fixture@v1 `
  --metric citation_evidence_span
~~~

| Case | span v1 | 说明 |
|---|---:|---|
| Unicode exact binding | 1 | `0:9` 精确绑定“地球围绕太阳运行。” |
| unknown source `S9` | 0 | 不在 supplied `citation_sources` |
| offset/quote mismatch | 0 | `source[0:3]` 是 `abc`，不是 `bcd` |
| duplicate JSON key | 0 | strict parser 在评分前拒绝 |
| unrelated claim + exact quote | 1 | 故意证明 identity gate 不推断 entailment |

Cases/answers 分别为 1,015/1,138 bytes，SHA-256 为 `ceb3ff9d…89e8` / `c61507ec…2661`。这五条 authored fixture 没有模型调用、人类判断或权限系统；最后一行的明显语义反例仍得 1 是协议设计，不是漏洞。生产评测应把 syntax、authorized source identity、span identity、semantic verdict、source quality 与 publication policy 分开保存。

对已经观测到 binary label 的历史预测做校准分析：

~~~powershell
python -m about_llm.evaluation.cli calibrate `
  --input projects/evaluation-gate/calibration.example.jsonl `
  --bins 5 `
  --output artifacts/evaluation/calibration.json
~~~

每行包含 `case_id`、观测 `label`（0/1）和在看到结果前记录的 `probability`。输出 Brier、equal-width ECE、非空 bin 明细，以及对每个唯一 threshold 的 coverage/risk。相同 confidence 的样本一起接受，不按输入顺序拆 tie。

`calibration.example.jsonl` 是为了验证公式和 CLI 的**合成 fixture**，不是某个模型的真实校准实验。`calibration.manifest.example.json` 记录事件定义、虚构 predictor、输入 SHA-256 与证据边界；测试会核对 manifest hash，防样例变化后仍引用旧证据。真实运行还应记录 predictor/model/revision、生成时间、label protocol、数据切片和 probability 确实先于 outcome 产生的系统证据。

ECE 强依赖 bin 数与分箱方法，不能跨不同配置直接比较；小样本/空 bin 会不稳定。Probability 必须来自可定义、可回放的 predictor/verifier，模型在回答中自述“我有 90% 信心”不自动成为校准概率。用于选择性回答时同时报告 coverage、risk、样本数和关键切片，不能只挑一个低 risk 阈值。

对相同 case 做 paired bootstrap 与发布门禁：

~~~powershell
python -m about_llm.evaluation.cli compare `
  --cases projects/evaluation-gate/cases.example.jsonl `
  --baseline-results artifacts/evaluation/baseline.results.jsonl `
  --candidate-results artifacts/evaluation/candidate.results.jsonl `
  --baseline-manifest artifacts/evaluation/baseline.run-manifest.json `
  --candidate-manifest artifacts/evaluation/candidate.run-manifest.json `
  --quality-metric exact_match `
  --minimum-quality-difference 0 `
  --maximum-latency-increase 0.10 `
  --protected-slice zh `
  --maximum-slice-regression 0 `
  --bootstrap-samples 10000 `
  --seed 7 `
  --output artifacts/evaluation/gate.json
~~~

若有“越高越安全”的 case-level 指标，可用 `--safety-metric` 和 `--maximum-safety-regression` 加入 guardrail。compare 在通过时返回 0、门禁失败时返回 1、artifact/schema 错误时返回 2，适合直接接 CI。安装仓库后也可使用 `about-llm-eval`。

`--output` 写出的不是无 schema 的临时 JSON，而是 `about-llm.evaluation-comparison.v2` artifact。v2 新增 resampling unit 与 cluster 配置，因此不把旧 v1 文件静默解释成新语义。可在发布或历史归档前独立严格重载：

~~~powershell
python -m about_llm.evaluation.cli verify-comparison `
  --input artifacts/evaluation/gate.json
~~~

loader 拒绝 duplicate key、未知/缺失字段、`NaN/Infinity`、非法嵌套类型、内部均值差或 gate 决策不一致、固定 evidence boundary 漂移和 fingerprint 不一致。`verify-comparison` 输出 `verification_scope: artifact_only`、`referenced_manifests_revalidated: false` 与 `statistics_recomputed: false`：其中 `valid` 只表示当前 comparison 文件通过 schema、内部算术/判定和 canonical fingerprint 校验，不会重新打开所引用的 run manifest/results/cases，也不会重跑 bootstrap。`comparison.example.json` 是由仓库两份 scored fixture、run manifest 和固定 gate 配置生成的可复算样例。

需要验证完整本地证据图时使用 `verify-evidence`：

~~~powershell
python -m about_llm.evaluation.cli verify-evidence `
  --cases projects/evaluation-gate/cases.example.jsonl `
  --baseline-answers projects/evaluation-gate/answers.baseline.example.jsonl `
  --candidate-answers projects/evaluation-gate/answers.candidate.example.jsonl `
  --baseline-results projects/evaluation-gate/results.baseline.example.jsonl `
  --candidate-results projects/evaluation-gate/results.candidate.example.jsonl `
  --baseline-manifest projects/evaluation-gate/run.baseline.manifest.example.json `
  --candidate-manifest projects/evaluation-gate/run.candidate.manifest.example.json `
  --comparison projects/evaluation-gate/comparison.example.json
~~~

该命令严格重开 cases、两侧 recorded answers/results/run manifests 与 comparison；按 case 顺序重算 answer fingerprint，用当前仓库中与 manifest revision 精确相同的 metric 实现重新评分，再从 artifact 记录的 resampling/gate 配置重跑 bootstrap、切片和最终判定。只有全部重建为相同 comparison artifact 才输出 `verification_scope: full_local_recomputation`。它会发现 answer 漂移、manifest 自洽但 score 错误，以及 comparison 内部自洽但不对应当前 results 的摘要。

这仍不是“评测真实性证明”：本地文件可以被有权限者整套协同重写；命令不会重新调用模型/provider，不认证 `system_id`，也不验证 sampling/cluster 假设、指标 construct validity 或线上影响。Artifact 认证和历史截断检测属于后文 HMAC ledger + 外部 trusted head 的另一层；即使两层都通过，也只证明已记录 bytes/计算链，不证明模型服务当时真实执行。

### HTML 对比报告

严格 comparison artifact 可生成自包含 HTML：

~~~powershell
python -m about_llm.evaluation.cli render-comparison-html `
  --input projects/evaluation-gate/comparison.example.json `
  --output artifacts/evaluation/comparison.html
~~~

报告显示 pass/fail、system/manifest/case identity、总体与 protected-slice 区间、case/cluster resampling ledger、gate 阈值/原因、metric revisions 和两层 evidence boundary。输出 deterministic，不含 JavaScript、网络链接或外部字体/图片；CSP 默认拒绝所有资源，只允许当前内联 CSS。system id、slice、reason 等动态文本统一 HTML escape，测试用闭合 `td` + `script` payload 验证不会变成节点。

这只是 `artifact_only_render` 派生视图：renderer 先严格加载 comparison，但不会调用 `verify-evidence`、验证 HMAC ledger 或重算统计。HTML 可被覆盖，也不是 canonical identity；发布判断必须回到 JSON artifact/verifier，不从页面颜色或四舍五入后的展示值反推。

也可以直接用仓库中已经评分的 authored fixture 验证 compare 协议：`results.baseline/candidate.example.jsonl` 分别由 `run.baseline/candidate.manifest.example.json` 绑定。fixture 只用于回归 loader、fingerprint 和 gate 数学，不代表任何真实模型质量或延迟。

### Cluster bootstrap 正式门禁

若多条 case 属于同一用户、文档或会话，在每条 case 的 `metadata` 中写入稳定、非空字符串 cluster id，并显式选择 estimand：

~~~powershell
python -m about_llm.evaluation.cli compare `
  --cases artifacts/evaluation/cases.jsonl `
  --baseline-results artifacts/evaluation/baseline.results.jsonl `
  --candidate-results artifacts/evaluation/candidate.results.jsonl `
  --baseline-manifest artifacts/evaluation/baseline.run-manifest.json `
  --candidate-manifest artifacts/evaluation/candidate.run-manifest.json `
  --quality-metric exact_match `
  --cluster-metadata-key user_id `
  --cluster-weighting case `
  --cluster-exact-max 6 `
  --bootstrap-samples 10000 `
  --seed 7 `
  --output artifacts/evaluation/cluster-gate.json
~~~

`case` 回答随机 case 的平均差：每个 resample 用 sampled cluster difference sums / sampled cluster sizes，分母随抽中的 cluster size 变化。`equal` 先求每个 cluster 的 baseline/candidate mean，再回答随机 cluster 的平均差。Safety metric 的 difference 也使用同一 weighting，不能让质量回答“平均用户”而安全悄悄回答“平均请求”。Latency 字段仍明确是逐 case mean，不冒充 equal-cluster latency estimand。

每个 overall/protected-slice cluster result 保存 case/cluster 数、按首次出现顺序的 cluster sizes、weighting、baseline/candidate estimand、confidence、interval、改善比例、`exact|monte_carlo`、实际 resample 数、linear quantile，以及 exact 时 null、Monte Carlo 时有效的 seed。Root `bootstrap` 另绑定 metadata key、exact threshold、requested Monte Carlo samples 和 seed；case metadata 又由 run manifest 的 ordered cases fingerprint 绑定。Artifact-only verification 仍不会重开 cases 检查实际 cluster ids 或重跑统计。

当某个 overall 或 slice 有 \(G\) 个 cluster 且不超过 `cluster-exact-max` 时枚举全部 \(G^G\) 个 ordered resample；否则 `--bootstrap-samples` 控制 seeded Monte Carlo。不同 slice 可因 cluster 数不同而一个 exact、另一个 Monte Carlo，因此 method 和实际 resample 数属于每个 result，而不是只写在 root。运行透明 reference 手算公式：

~~~powershell
python projects/evaluation-gate/clustered_bootstrap_toy.py
~~~

Fixture 的两个 cluster 分别为 5×`+1` 与 1×`-1`。Exact 路径枚举 `AA/AB/BA/BB` 四个 ordered resample。Case-weighted statistic `[1,2/3,2/3,-1]` 得到 linear percentile 95% interval `[-0.875,0.975]`、经验改善比例 3/4；equal-cluster statistic `[1,0,0,-1]` 得到 `[-0.925,0.925]` 和 1/4。Case-weighted 分母是每次抽中 cluster 的 size 总和，不能固定为原始 case 数。

最多可配置 7 个 cluster 的 exact 枚举；大输入用 seeded Monte Carlo，并限制临时 sampled-index matrix。Exact 不等于可信小样本 inference：两 cluster interval 只证明重采样和 quantile 口径。Comparison v2 的通过也不建立 metadata key 真的是正确 sampling unit、cluster 独立/代表性、无 interference、足够功效或 percentile coverage。它不是 BCa/studentized interval；真实分析还应报告最大 cluster sensitivity，并在设计需要时使用 cluster-robust model、randomization inference 或其他方法。

### Paired randomization / sign-flip 对照

Bootstrap 区间之外，可运行一份透明的 hypothesis-test fixture：

~~~powershell
python projects/evaluation-gate/paired_randomization_toy.py
~~~

输入是同 case 的 baseline/candidate score；差值定义为 candidate − baseline。Exact 路径只对非零差值枚举符号，pair 总数与 observed mean 仍包含零差值。默认 two-sided 比较绝对统计量；`greater` 表示预先指定 candidate 更高，`less` 相反。超过 exact 上限后使用 seeded Monte Carlo 和 plus-one correction，并记录 assignments、extreme count、p-value resolution 与 seed。

固定 5-pair fixture 有 4 个 +1 与 1 个 0：greater exact p=1/16，two-sided p=2/16。它只证明枚举、方向、零差值和 Monte Carlo 账本。Sign exchangeability、随机 assignment、独立 case、population sampling、metric construct validity、cluster dependence、multiple testing 和因果结论都没有由 fixture 建立；因此这个 p-value 不接入现有 release artifact 自动决策，也不能替代 effect size、bootstrap interval 与业务阈值。

### Cluster-joint sign-flip 对照

若多条 case 来自同一用户/文档/会话，运行：

~~~powershell
python projects/evaluation-gate/clustered_randomization_toy.py
~~~

Authored difference 为 5 条 user A 的 `+1` 与 1 条 user B 的 `-1`。Naive case-level greater test 把 6 行各自翻转，得到 7/64；cluster-joint case-weighted 路径只给每个 user 一个符号，对 contribution `[5,-1]` 枚举 4 项，得到 2/4。它保留每个 case 的权重和 observed \(4/6\)。若明确要回答“平均用户”，`cluster_weighting="equal"` 会先求每个 cluster mean，observed 变成 \((1-1)/2=0\)，two-sided p=1。

两种 weighting 是不同 estimand，不是看到结果后可切换的 variance option。Joint flip 允许 cluster 内相关，但不证明 cluster 定义正确、cluster-level label exchangeability、cluster 间独立、无 interference 或样本代表流量。Exact 配置上限为 24 个非零 unit，超过阈值使用 seeded Monte Carlo plus-one；这只是防止 reference 意外做不可行枚举，并不会让很少的 cluster 获得足够统计功效。

### Holm multiple-testing 对照

若一个预先定义的 family 同时包含多个 metric/slice hypothesis，可运行：

~~~powershell
python projects/evaluation-gate/holm_correction_toy.py
~~~

输入顺序 `[0.04,0.01,0.03,0.20]` 经稳定升序排序后，multiplier 是 `[4,3,2,1]`。Scaled value `[0.04,0.09,0.08,0.20]` 必须再取前缀最大值，所以 sorted adjusted p-value 为 `[0.04,0.09,0.09,0.20]`；映回输入顺序后为 `[0.09,0.04,0.09,0.20]`，`alpha=0.05` 只拒绝原索引 1。输出同时保留 rank ledger 与 input-order view，避免调用方把排序后的结果贴错 hypothesis。

Holm 在 component p-value 有效时对任意依赖控制 FWER，但不证明 family 是事前定义的，也不修复反复窥视、可选停止、测试集挑指标、cluster unit 错误或无效原始检验。它不估计 effect size、业务重要性或因果。该 reference 不自动接入 comparison artifact；生产 gate 还需把 family、原始/adjusted p-value、alpha、全部 hypothesis 和预注册/选择协议写入版本化 artifact。

### Sequential peeking / optional-stopping 对照

同一 hypothesis 在多个时点反复检验不是 Holm 的 multiple-hypothesis 问题。运行：

~~~powershell
python projects/evaluation-gate/sequential_peeking_toy.py
~~~

Fixture 预设 `[10,20,30,40,50]` 五个 informative-pair looks，以 i.i.d. fair sign 为 null，并固定双侧 p-value 为 doubled smaller inclusive binomial tail。每次都用 0.05 且首次显著即停，exact familywise error 为 `7109832616777/70368744177664 ≈ 0.1010367984`；事前把 familywise 0.05 均分为每次 0.01 时为 `2142139082367/140737488355328 ≈ 0.0152208136`。实现用 `(n, positive_count)` dynamic program 传播精确概率，没有枚举 `2^50` 条 sign sequences。

Bonferroni 对照只在 look 数与阈值事前固定、每个 p-value 在其 null 下有效时由 union bound 控制总体错误；它通常保守，也不允许结果不好看时临时增加 look。本 oracle 没有 tie、effect magnitude、cluster、case sampling、power/sample-size、confidence sequence、模型/judge 或线上随机实验，因此不自动接入 comparison release gate，也不能证明候选模型改善。

### Artifact schema

recorded answer 每行：

```json
{"case_id":"...","output":"...","latency_seconds":0.12,"error":null}
```

scored result 由 CLI 原子写入，每行包含 `case_id`、`output`、`scores`、`latency_seconds` 与 `error`。比较时 baseline、candidate 与 cases 的 id 集必须完全一致；系统错误和缺失 metric 会阻断门禁，而不是被当作 0 分静默平均。所有 JSON/JSONL loader 拒绝 duplicate key、非标准常数和 schema 外字段，避免“后一个 key 覆盖前一个”或字段拼错后继续运行。

run manifest 记录完整 ordered case semantics（input、expected、slices、metadata）的 fingerprint、ordered result values、recorded-answer identity、metric name→revision、`system_id` 和 scorer revision。compare 会用当前 cases/results 重算 identity，并要求 baseline/candidate 对所比较 metric 使用相同 revision；因此“只保留相同 case id、悄悄换 gold/metric”的运行会在 bootstrap 前失败。

这里的 SHA-256 是无密钥 canonical content identity，不是签名。`system_id` 也是调用者提供的 label：manifest 自洽不能证明输出真的来自该系统、模型服务真实执行、case/metric 具有 construct validity 或线上收益成立。生产流水线应由可信 runner 写入 commit/container/model/prompt/index/judge revision，把 manifest 放进签名或 append-only artifact store，并保留访问控制和时间证据。

comparison fingerprint 进一步防止“统计跑完后只改阈值/通过布尔值/失败原因”的静默漂移，但边界相同：攻击者若能同时改写内容并重算无密钥 fingerprint，文件仍会自洽。只有把可信 head/fingerprint 锚定到受控发布系统、签名或 append-only ledger，才能检测整套文件的协同重写或历史截断。

### 认证发布链

仓库提供 HMAC-SHA256 链式 snapshot，把每条记录的连续 sequence、唯一 release/artifact id、artifact kind、原始文件 byte size/SHA-256、decision、caller-supplied RFC 3339 `recorded_at`、`key_id` 和前一条 MAC 一起做 domain-separated MAC。不同记录可使用不同 `key_id`，因此可演示 key rotation；密钥只由调用者 resolver 提供，不写入 ledger。

~~~powershell
$env:PYTHONPATH = "src"
python projects/evaluation-gate/authenticated_release_ledger_toy.py
~~~

`release-ledger.example.json` 精确绑定两份 run manifest 和 comparison 当前文件 bytes。Loader 要求 canonical JSON + 单个结尾 LF，拒绝 duplicate/unknown/missing 字段、`NaN/Infinity`、乱序/断链、重复 identity 和非连续 sequence；writer 只做 exclusive-create + file `fsync`，不会覆盖旧 snapshot。示例的两把 `11...11`/`22...22` 公开测试值不是生产 secret，也不证明 key custody。

验证必须分三层读：

1. `authenticated_chain=true`：仅表示相对 caller-supplied HMAC keys，每条记录及其顺序未被无密钥攻击者改写；
2. `referenced_artifacts_rehashed=true`：只有传入与 ledger artifact id **完全相同** 的 path mapping，并重新读取 size/SHA-256 后才成立；
3. `trusted_head_matched=true`：只有调用者从 ledger 外提供预期 `(sequence, record_mac)` 才成立；没有外部 head 时，删除尾部后留下的合法前缀仍会通过 MAC 验证。

MAC 绑定 `recorded_at` 字符串，但不证明真实发生时间；HMAC 是共享密钥认证，不提供公钥签名/不可否认性。它也不证明 artifact 来自所称 runner、评测统计/指标有效、密钥由谁保管、parent directory 已断电持久化、发布目录原子替换，或 verify 后文件不会发生 TOCTOU 变化。生产系统还需 KMS/HSM、权限与轮换/吊销协议、外部 head/transparency log 或对象锁、可信时间和消费端 verify-then-open 设计。

## 数据记录

每个 case 至少包含 id、输入、期望或 rubric、切片、来源、许可和风险级别。每次运行记录系统版本、原始输出、结构化分数、耗时、token usage 和错误。自动 judge 的 prompt、模型与顺序随机化也属于版本。

## 门禁原则

1. 比较相同 case 的配对结果；
2. 质量提升必须看置信区间，不只看均值；
3. 总体提升不能掩盖关键语言、风险和用户切片退化；
4. 安全与权限是 guardrail，不用平均质量抵消；
5. 延迟比较使用同 workload 和并发；
6. 门禁输出全部失败原因，不只返回一个布尔值。

## LLM-as-judge 校准协议

Judge 只能补充可执行指标，不能成为未经校准的唯一真值。先由至少两名标注者在盲测样本上独立评分并处理分歧，记录 rubric、边界案例和一致性；再固定 judge 模型、版本、prompt、temperature 与解析器，对同一批样本测量与人工标签的相关性、分类 precision/recall 和各切片偏差。成对比较要随机交换 A/B 位置，加入同答案自洽、明显优劣、提示注入和引用伪造控制题。模型、prompt 或任务分布变化后重新校准，并定期人工抽检线上 disagreement。

引用语法分数只检查 `[S1]` 是否来自允许集合和段落覆盖率，不测 claim-evidence entailment。语义忠实度 judge 必须只看到 claim 与对应 evidence，允许返回“不足以判断”，并用人工标注报告误报和漏报。

## 后续里程碑

- KMS/HSM 或非对称签名、外部 transparency/object-lock head 与多发布趋势查询；
- 在线 shadow/canary 数据回流。
