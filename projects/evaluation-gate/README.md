# 统一评测与发布门禁

目标：让 RAG、Agent、微调和模型升级使用同一组 case id、切片、统计与发布决策，而不是各自展示几个成功示例。

## 当前实现

- Recall@k 与 MRR；
- 配对 bootstrap 的均值差、置信区间和改善概率；
- 同时约束质量、安全和延迟的透明 ReleaseGate；
- JSONL runner、原子结果写入、exact match 与 token F1；
- JSON Schema 输出合规和已授权来源的引用语法/覆盖率；
- overall 与语言、风险、用户等切片汇总及 Markdown 表格；
- 固定 seed 的可复现测试。

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

- HTML/Markdown 对比报告；
- CI 回归阈值与历史趋势；
- 在线 shadow/canary 数据回流。
