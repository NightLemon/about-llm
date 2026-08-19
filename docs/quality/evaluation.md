# 评测

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：第一次建立 LLM 质量体系的工程和产品团队。
- **先修**：能描述任务输入、输出和失败；不要求先掌握统计检验。
- **首次阅读**：定义“好” → case → 指标 → judge → 错误分析 → 回归门禁。
- **完成信号**：能建立小型评测集、评分规则、切片和失败 taxonomy。
- **卡住时**：先用 20–30 条真实 case 建基线，再读[评测方法](evaluation-methodology.md)。

</div>

本章给出评测地图。实验设计、统计和 LLM-as-judge 校准详见[评测方法与发布决策](evaluation-methodology.md)，工具型系统另见[Agent 评测、仿真与红队](agent-evaluation.md)。

## 先定义“好”

评测把目标变成可测量行为。一个模型不存在脱离任务的单一“智力分数”。至少区分：能力、指令遵循、事实性、鲁棒性、安全、公平、延迟、成本和真实业务价值。

## 评测集设计

从真实流量分层抽样，补充边界、少数群体、长尾、多语言、对抗和应拒绝案例。每条样本记录来源、许可、难度、标签协议和版本。训练/dev/test 隔离；保留一个开发者无法反复查看的最终集。

稳定 `case_id` 不是不可变评测集。每次运行还要绑定该 ID 下的 input、expected/rubric、slice 和 metadata；metric 名称也必须带实现 revision。否则换了 gold 或评分实现仍可能被误报为“同集配对比较”。canonical manifest 能检测已列字段漂移，但无签名 hash 不能认证来源，也不证明样本或指标有效。

发布证据还要区分 content identity、认证链和外部历史锚点。仓库 HMAC release ledger 可认证连续记录并可选重哈希引用 artifact；但没有 ledger 外 trusted head 时，合法前缀截断仍会通过。HMAC 也不证明 key custody、真实时间、模型实际执行或评测有效性，详见[评测方法与发布决策](evaluation-methodology.md#artifact)。

`verify-comparison` 只重载最终 artifact；`verify-evidence` 则重开 cases/answers/results/manifests，重新评分并重跑 bootstrap/gate。后者能发现更多跨文件不一致，但仍不认证本地 bytes 或重放模型调用，不能把“可复算”写成“来源真实”。

`render-comparison-html` 可把严格 artifact 生成无脚本、无外部资源且动态文本转义的可读报告；页面明确是 `artifact_only_render`，不能代替 JSON identity、全图复算或 HMAC/trusted-head 验证。

黄金答案不一定唯一。开放任务可写 rubric：必须覆盖哪些事实、允许哪些变体、哪些错误致命。高风险领域由合格专家标注，并测量标注者一致性而非强行假设一个客观标签。

## 指标

- 精确匹配、F1：适合答案较确定的抽取/短问答。
- Pass@k：代码/数学多个候选中至少一个通过验证的概率；估计时注意采样数公式。
- ROUGE/BLEU：表面重叠，不能充分代表事实或语义质量。
- BERTScore/Embedding 相似：语义更宽松，但可能忽略关键否定与数字。
- Pairwise win rate：两个系统成对比较；对顺序和长度偏差敏感。
- 校准：Binary Brier score、明确分箱的 ECE、tie-aware risk-coverage 曲线；概率必须对应结果发生前定义的事件。
- 系统：TTFT、TPOT、成功任务成本、工具错误率。

不要把不同维度随意加权成一个总分，除非权重对应真实效用并报告各分项。

指标名称不够，必须固定实现 revision 与 normalization policy。仓库的固定 Qwen 七例 control 给出一个可执行反例：目标 `LLM-2026`、输出 `llm-2026` 时，raw decoded string 的 literal exact 为 0，但 NFKC + `casefold()` + whitespace normalization 后 exact 为 1；目标 `{"answer":42}`、输出 `{"answer": 42}` 时，literal/normalized exact 都为 0，而忽略标点/空白的当前 token F1 为 1。七例汇总因此分别是 `4/7`、`5/7`、`6/7`，不是三个可互换的“准确率”。标识符大小写敏感时不得 case-fold；JSON 应解析后做 schema 与字段语义检查，不能用 token F1 代替结构验证。

通用 Evaluation Gate 保留兼容默认值：不传 `--metric` 时仍计算 normalized `exact_match` 与 `token_f1`；需要逐字契约时显式传 `--metric literal_exact_match`，需要并列审计时再重复传 `--metric exact_match`。三者分别写入独立 metric revision；literal 比较 decoded string，不等同于原始网络响应 bytes。

结构化输出至少分四层：strict JSON syntax、JSON Schema、expected parsed value、业务语义/授权。仓库五条 fixture 中，object key order/whitespace 变化的 literal/normalized/F1/schema/value 为 `0/0/1/1/1`；错误值 `43` 仍 schema-valid，但 value exact 为 0；duplicate object key 与 `NaN/Infinity` 必须在 strict parser 层拒绝；array 逆序仍可 schema-valid 且 token F1=1，却 value exact=0。`about-llm.json-schema-metric.v2` 与 `about-llm.json-value-exact.v1` 因此是两个独立 opt-in 指标，不能用其中一个替代另一个，更不能替代 tenant、单位、时效和副作用规则。

RAG 引用同样不能压成“有 `[S1]` 就忠实”。`citation_syntax` 只覆盖已知 ID 与段落引用；opt-in `about-llm.citation-evidence-span-metric.v1` 进一步验证 strict claim artifact 中 source ID、end-exclusive offset 与 exact quote 一致。固定五例为 `[1,0,0,0,1]`，最后一个明显无关 claim 仍因 exact span binding 得 1，明确说明 span identity 不推断 semantic entailment。ACL snapshot provenance、claim segmentation、support verdict、judge error、source quality 与 publication policy 都要另测。

这七次确实加载固定 Qwen 权重并真实调用 `GenerationMixin.generate()`，只能建立固定执行路径和逐例输出事实；suite 是 authored、未外部预注册、未独立抽样/留出且没有统计功效。真实执行不自动建立 construct validity、sampling validity、总体模型质量或发布有效性。

## Hosted Evals API 与本地发布门禁不是同一层

截至 2026-08-19，OpenAI 官方 [Evals guide](https://developers.openai.com/api/docs/guides/evals) 用 `data_source_config` 描述测试数据 schema，用 `testing_criteria`/graders 描述评分条件；创建 eval 后，再用具体数据源和模型启动异步 run，读取逐条件结果与报告。

这套产品流程可以编排 provider 调用与 grader，但不能替你证明 case 有代表性、gold/rubric 有效、切片完整、阈值符合业务效用或结果可发布。对应关系应这样理解：

| 层 | Hosted Evals API 可承担 | 本仓库仍需承担 |
|---|---|---|
| 数据接口 | item schema、上传/引用数据源 | 来源、许可、case identity、split 与泄漏审计 |
| 执行 | 对每个 item 调模型并排队运行 | 固定 Prompt/tool/environment identity 与失败重放 |
| 评分 | 配置 string/model 等 grader | construct validity、人工校准、指标 revision 与切片 |
| 结果 | run 状态、逐 criterion 结果、报告入口 | paired comparison、统计假设、发布 gate 与回滚证据 |

因此可以把 hosted run 接入 Evaluation Gate，但不能用“run completed”代替“候选通过发布门禁”。异步失败、缺失 case、grader 漂移和多次试验选择仍要进入 artifact 与决策记录。

## LLM-as-judge

适合大规模初筛和复杂 rubric，但会有位置偏差、冗长偏差、自我偏好、提示敏感和知识盲区。使用方式：

1. 给明确 rubric 和证据，不让 judge 猜目标。
2. 随机交换答案顺序，必要时匿名化模型身份。
3. 要求输出结构化分项和引用。
4. 与专家标注在代表性样本上校准，报告一致性。
5. 关键结论用多 judge 或人工复核。

Judge 分数不是事实真值。

Judge 自述 confidence 或输出的 1–5 分不是天然正确概率。若要做 selective automation，先把事件定义为 binary correctness/acceptability，在独立人工集上记录 predictor probability 与最终 label，再检查 Brier、reliability bins 和 risk-coverage；模型、rubric 或流量变化后重新校准。

## 统计

报告样本量、均值、置信区间和效应大小。成对样本用配对 bootstrap/置换检验比独立检验更合适。小提升可能小于随机波动。分层报告可发现总体提升掩盖某语言或用户群体退化。

两者回答不同问题：paired bootstrap 估计 mean difference 的不确定区间；paired sign-flip/randomization test 在 sharp null 与 pair-label exchangeability 下计算 observed statistic 的尾部概率。P-value 不是 posterior probability，bootstrap 改善比例也不是候选更优的后验概率。用户/文档相关数据需按 cluster 整体重采样或联合翻转；case-weighted bootstrap 每个 resample 重算 cluster-sum/cluster-size ratio，equal-cluster 则平均 cluster means，二者必须预先选定。`compare --cluster-metadata-key ... --cluster-weighting case|equal` 会把该选择、cluster sizes、exact/Monte Carlo 方法和实际 resample 数写入 comparison v2；通过工件仍不证明 cluster 假设。同时扫描多个预定义 slice/metric 时，可用 Holm step-down 对明确 family 的有效 p-value 控制 FWER；同一 hypothesis 随数据累积反复查看则是 sequential design，不能用 Holm 替代。仓库 exact sign-test peeking oracle 在 `[10,20,30,40,50]` 五个 looks 上把逐次 0.05 的实际假阳性算为约 0.1010，并把预先均分为每次 0.01 的 Bonferroni 对照算为约 0.0152。case/cluster bootstrap、sign-flip、Holm 与 sequential CPU oracle 都只验证统计口径；它们不建立正确 cluster、抽样、交换性、独立性、coverage、因果、effect importance 或指标有效性。

重复在测试集调 Prompt 会过拟合。维护滚动新鲜集和时间切片。公开基准可能被预训练污染；用 canary、改写、私有数据和过程证据增强可信度。

## 错误分析

每轮抽样失败，建立互斥尽量清晰的 taxonomy：意图理解、知识缺失、检索、推理、格式、工具、权限、安全或评价错误。统计严重度与频率，选最重要类别修复，再增加回归样本。

## 在线评测

A/B 测试看任务完成、留存、人工升级、撤销/纠错等真实指标，同时设安全 guardrail。随机化单位要避免同一用户跨版本污染；考虑新奇效应、学习效应和延迟。最大样本、时长、look schedule、停止规则与异常中止条件要事前固定；每天用固定样本 p-value 偷看并“显著即停”会膨胀假阳性。高风险变更先 shadow/canary，不直接全量。

## Eval-driven development

1. 从失败案例定义可复现测试。
2. 固定基线和版本。
3. 修改一个主要变量。
4. 运行质量、安全、成本回归。
5. 检查分层结果与具体样本。
6. 通过门禁再发布，线上继续监控。

## 自测

1. 为什么开放问答用精确匹配会低估质量？
2. LLM judge 偏爱更长答案时怎样发现和缓解？
3. 总体胜率上升但中文用户下降，应如何报告和决策？
