# Evaluation Gate：从逐例结果到发布决定

**项目导航**：[项目索引](../project-index.md) · [评测总览](../../quality/evaluation.md) ·
[评测方法](../../quality/evaluation-methodology.md) · [评测测量学](../../quality/evaluation-measurement.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/evaluation-gate/README.md) ·
[项目证据](../../evidence/project-controls.md)
{ .doc-nav }

这个项目不再讲一套完整评测教材。它只回答：怎样把 baseline 与 candidate 的固定输出，组装成可复算、可拒绝、
可审阅的发布决定。命令和文件由 README 维护，统计与测量概念回到各自正文。

## 十分钟热身：为什么 24/30 仍然不能发布 {#headline-trap}

```powershell
python projects/evaluation-gate/trace_headline_accuracy_trap.py
```

固定样例里，总分从 22/30 变成 24/30，但六条变化包含四条普通问题改善和两条跨租户拒绝退化。总体均值无法抵消
高风险切片，因此 gate 拦截 candidate。这个脚本只演示决定，不调用模型；作为预期被拦截的教学案例，它完成计算后
仍返回 0。CI 应使用正式 `compare` 命令区分“通过”“有效但未通过”和“评测无效”。

## 开始前先写评测协议

在看到 candidate 结果前固定：决策范围、系统身份、case 与抽样单位、estimand、主指标、最小有意义效果、保护切片、
失败分母、最大样本量和查看计划。构念与标注见[评测测量学](../../quality/evaluation-measurement.md)，配对区间与多重比较见
[评测统计](../../foundations/evaluation-statistics.md)，发布规则见[评测方法](../../quality/evaluation-methodology.md)。

## 端到端主线 {#run}

主线只增加一层产物：

```text
cases + recorded answers
→ per-case results + run manifests
→ paired comparison + gate
→ artifact-only / full-local verification
→ HTML review view + release decision
```

第一次运行的完整参数见[项目 README](https://github.com/NightLemon/about-llm/blob/main/projects/evaluation-gate/README.md#run)。
阅读结果时按以下顺序：

1. 对齐 `case_id`，保留 timeout、refusal、parse error 和 judge failure。
2. 先看逐例改善/退化，再看总体与 protected slices。
3. 核对 baseline/candidate manifest 是否绑定同一 case 顺序与评分器。
4. 比较 effect、区间和事先写下的阈值。
5. 区分 artifact-only 校验与重开上游文件的完整本地重算。
6. 故意改坏 recorded answer、分数或 gate threshold，确认 verifier 会拒绝。

`system_id` 是调用者填写的身份声明，manifest 不会认证真实模型是否执行。HTML 也是从 canonical JSON 派生的阅读视图，
不能代替 comparison 和 verifier receipt。

## Structured output：格式通过后还要检查值

结构化任务至少分开 JSON syntax、closed schema、canonical value 与 domain semantics。Schema-valid 对象仍可能写错金额、
引用无权资源或使用过期 ID；对象字段顺序可忽略，不代表数组顺序或标量类型也可忽略。固定反例的命令在 README，
本页只保留“格式门禁不能升级成业务正确性”这一决策边界。

## Citation span：位置正确不等于支持结论

Citation gate 也要分层：来源是否允许、字符区间是否精确、引文是否匹配原文、引文能否支持 claim、最终是否允许发布。
本项目实现的 span metric 只覆盖前三项中的结构与逐字位置，不自动判断 entailment、来源真实性或 ACL。

## 测量反例 {#measurement-control}

四条标签可以出现 `Cohen's κ=1` 而相对独立外部准则全部错误。这个反例说明 reliability 与 validity 是两件事。
同样，最小可检测效果（MDE）由样本设计决定，不等于业务上值得发布的最小效果。命令、固定数值和功效计算在 README；
这里要求它们先进入 measurement plan，再进入 comparison。

## 可选能力不阻塞主线

| 选修 | 何时需要 | 不能替代 |
|---|---|---|
| Clustered bootstrap / randomization | 同一用户或文档贡献多条 case | 正确的 sampling unit |
| Multiple/sequential controls | 比较多个指标或反复查看 | 事前定义的主要假设 |
| Calibration / risk–coverage | 系统会按置信度拒答 | 置信度事件定义与目标分布校准 |
| Authenticated release ledger | 需要检测工件篡改或历史回滚 | 模型来源认证、可信时间或真实业务结果 |
| 固定 Qwen 行为运行 | 要确认目标 checkpoint 的一条真实路径 | 代表性数据、总体质量或生产性能 |

这些选修应在主线 gate 已可复算后按风险加入，不要一次把所有统计和认证机制都堆进首次运行。

## 最终验收

- 评测协议在结果前固定，所有 case 有终态并进入分母。
- 两侧运行清单、逐例结果、comparison 与 verifier receipt 能互相对账。
- 保护切片退化不会被总体均值抵消。
- Exit code 能区分通过、有效拒绝与无效评测。
- 至少一个 tamper 反例被拒绝。
- 报告明确区分 recorded output、本地重算、真实模型运行与生产结论。

精确命令、输入输出和故障排查只在[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/evaluation-gate/README.md)
维护；固定数值、版本与不可外推范围只在[项目证据](../../evidence/project-controls.md)维护。
