# Evaluation Gate：把“感觉更好”变成发布决定

这个项目把 baseline 和 candidate 的已保存输出放到同一组 cases 上评分，再用配对统计、保护切片和明确阈值决定
发布、拒绝发布，还是判定评测本身无效。

第一次学习请从[项目教学页](../../docs/practice/projects/evaluation-gate.md)开始。那里完整走过 score → compare →
verify → render → tamper；本页只保留运行入口、脚本索引和排错信息。

## 开始前先固定评测问题

看到 candidate 结果之前，至少写下：

- 这次 gate 控制全量发布、canary，还是继续实验；
- Baseline/candidate 的模型、Prompt、RAG、工具和 runtime 身份；
- Case 的来源、抽样单位和 protected slices；
- 主指标、方向和最小有意义差异；
- 可接受的延迟、安全与关键切片退化上限；
- 最大样本量、bootstrap seed，以及是否允许中途查看。

这些定义在看完结果后再改，最终区间和发布结论就不再回答原来的问题。

## 端到端主线 { #run }

先给 baseline 与 candidate 的 recorded answers 分别打分：

```powershell
New-Item -ItemType Directory -Force artifacts/evaluation | Out-Null

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
```

CLI 不调用模型。它消费已经保存的输出，并把评测样例、回答、逐例结果、评分器版本和系统 ID 绑定到运行清单。
先查看同一 `case_id` 上的改善与退化，再看总体均值。

然后做 paired comparison：

```powershell
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
```

退出码有三种含义：

| Exit code | 含义 |
|---:|---|
| 0 | Candidate 满足发布条件 |
| 1 | 评测有效，但 candidate 没有通过门槛 |
| 2 | 输入、schema 或证据有问题，无法作出发布判断 |

CI 必须区分“有效地拒绝发布”和“评测坏了”。

## 验证证据，而不是只信最终 JSON

只检查 comparison 自身的 schema、算术、判定和 fingerprint：

```powershell
python -m about_llm.evaluation.cli verify-comparison `
  --input artifacts/evaluation/gate.json
```

要重新打开 cases、answers、results 与 manifests，并用当前实现重算整张证据图：

```powershell
python -m about_llm.evaluation.cli verify-evidence `
  --cases projects/evaluation-gate/cases.example.jsonl `
  --baseline-answers projects/evaluation-gate/answers.baseline.example.jsonl `
  --candidate-answers projects/evaluation-gate/answers.candidate.example.jsonl `
  --baseline-results artifacts/evaluation/baseline.results.jsonl `
  --candidate-results artifacts/evaluation/candidate.results.jsonl `
  --baseline-manifest artifacts/evaluation/baseline.run-manifest.json `
  --candidate-manifest artifacts/evaluation/candidate.run-manifest.json `
  --comparison artifacts/evaluation/gate.json
```

前者是 artifact-only 检查，后者是本地完整重算。两者都不会重新调用模型，也不能认证调用者填写的 system ID。

给人阅读的报告由 canonical JSON 派生：

```powershell
python -m about_llm.evaluation.cli render-comparison-html `
  --input artifacts/evaluation/gate.json `
  --output artifacts/evaluation/comparison.html
```

## 指标名称相同，不代表测量对象相同

| 问题 | 应使用的检查 |
|---|---|
| 文本必须逐字一致 | Literal exact match |
| 允许大小写、标点或空白归一化 | Normalized exact match |
| 关注 token 重叠 | Token F1，并明确 tokenizer |
| JSON 能否解析 | Strict JSON syntax |
| JSON 是否满足结构 | JSON Schema |
| JSON 值是否与预期一致 | Canonical JSON value |
| 引用 ID 与字符 span 是否准确 | Citation span metric |
| 引用文本是否支持 claim | 独立 entailment judge 或人工标签 |

Schema-valid 的 JSON 仍可能写错值；引用 span 完全匹配也可能与 claim 无关。不要把格式、结构、数值和语义合并成
一个“准确率”。对应的固定反例与命令见[Structured output](../../docs/practice/projects/evaluation-gate.md#structured-output格式通过后还要检查值)。

## 根据当前问题选择入口

| 你想确认什么 | 入口 |
|---|---|
| 两套 recorded outputs 怎样配对比较 | `evaluation.cli score/compare` |
| Comparison 是否自洽、上游是否仍可重算 | `verify-comparison`、`verify-evidence` |
| JSON syntax、schema 与 value 的差别 | `structured-metrics.*.jsonl` |
| Citation span 与 entailment 的差别 | `citation-evidence-span.*.jsonl` |
| Calibration、Brier、NLL 与 ECE | `evaluation.cli calibrate` |
| Reliability、validity 与 power 的反例 | `measurement_toy.py` |
| 同一用户多条 case 怎样重采样 | `clustered_bootstrap_toy.py` 与 cluster compare |
| Paired randomization 与 sign-flip | `paired_randomization_toy.py`、`clustered_randomization_toy.py` |
| Multiple testing 怎样控制 family-wise error | `holm_correction_toy.py` |
| 反复偷看结果为什么膨胀假阳性 | `sequential_peeking_toy.py` |
| 发布历史怎样形成认证链 | `authenticated_release_ledger_toy.py` |
| 固定 Qwen 的七条真实输出表现 | `run_qwen_target_behavior_evaluation.py` |

完整实验解释见[项目教学页](../../docs/practice/projects/evaluation-gate.md)，精确数值与范围见
[评测证据页](../../docs/evidence/project-controls.md)。

## 固定 Qwen 运行应该怎样解读

仓库保存了一组固定 Qwen2.5-0.5B-Instruct 的七条真实 CPU 输出，用来展示 metric choice 如何改变分数：

```powershell
python projects/evaluation-gate/run_qwen_target_behavior_evaluation.py `
  --verify projects/evaluation-gate/target-qwen-behavior.recorded-report.json
```

这组小样例能证明目标 checkpoint 和 generation path 确实执行过，并展示 literal、normalized 与 token F1 的差异。
它没有独立抽样或足够统计功效，不能代表 Qwen 的总体中文、数学、结构化输出或生产质量。

## 主要输入与输出

| 文件 | 用途 |
|---|---|
| `cases.example.jsonl` | Case、reference、slice 与 metric 配置 |
| `answers.*.example.jsonl` | Baseline/candidate 的已保存输出 |
| `results.*.example.jsonl` | 逐 case 指标、延迟和错误状态 |
| `run.*.manifest.example.json` | 系统标签、输入 identity 与 scorer revision |
| `comparison.example.json` | Paired comparison 和发布判定样例 |
| `structured-metrics.*.jsonl` | JSON syntax/schema/value 反例 |
| `citation-evidence-span.*.jsonl` | Citation span 与语义支持反例 |
| `release-ledger.example.json` | 认证发布链教学样例 |
| `artifacts/evaluation/` | 本机 results、manifests、comparison 与 HTML |

Cases、outputs 和逐例 findings 可能包含用户数据。真实系统需要访问控制、脱敏、保留期限和删除流程，不能只依赖
无密钥 fingerprint。

## 常见故障

| 现象 | 先检查 |
|---|---|
| Overall 提升，关键用户退化 | Protected slice、cluster weighting 和逐例 diff |
| Candidate 少了失败 case，均值反而变好 | Exact case join、missing/error 是否保留在分母 |
| Exact match 很低，人工看起来正确 | 任务是否允许归一化；不要事后挑最有利 metric |
| JSON Schema 通过，业务值仍错误 | 再运行 canonical value 与 domain validator |
| 引用 span 完全匹配，claim 仍不可信 | Entailment、来源权威性和时效性是更外层检查 |
| Bootstrap 区间异常窄 | Resampling unit 是否应该是 user/document cluster |
| 多个指标总有一个显著 | 是否预注册主指标并进行 multiple-testing correction |
| 每天查看，显著就停止 | 使用预先设计的 sequential method 或固定样本量 |
| `verify-comparison` 通过，原始结果却变了 | Artifact-only 不重开上游；运行 `verify-evidence` |
| HTML 看起来正常，JSON verifier 失败 | HTML 是派生视图，canonical decision 仍是 JSON |

## 运行检查

```powershell
python -m pytest `
  tests/test_evaluation_cli.py `
  tests/test_evaluation_comparison_artifact.py `
  tests/test_evaluation_structured.py `
  tests/test_evaluation_statistics.py `
  tests/test_evaluation_release_ledger.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

默认检查验证固定输入、统计公式、artifact 关系和失败路径，不运行真实模型或人类标注。上线结论必须使用目标系统、
代表性样本、明确抽样单位和预先约定的发布门槛。
