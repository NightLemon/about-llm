# 评测

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

两者回答不同问题：paired bootstrap 估计 mean difference 的不确定区间；paired sign-flip/randomization test 在 sharp null 与 pair-label exchangeability 下计算 observed statistic 的尾部概率。P-value 不是 posterior probability，bootstrap 改善比例也不是候选更优的后验概率。用户/文档相关数据需按 cluster 整体重采样或联合翻转；case-weighted bootstrap 每个 resample 重算 cluster-sum/cluster-size ratio，equal-cluster 则平均 cluster means，二者必须预先选定。`compare --cluster-metadata-key ... --cluster-weighting case|equal` 会把该选择、cluster sizes、exact/Monte Carlo 方法和实际 resample 数写入 comparison v2；通过工件仍不证明 cluster 假设。同时扫描多个预定义 slice/metric 时，可用 Holm step-down 对明确 family 的有效 p-value 控制 FWER。仓库 case/cluster bootstrap、sign-flip 与 Holm CPU oracle 只验证统计口径；它们不建立正确 cluster、抽样、交换性、独立性、coverage、因果、effect importance 或指标有效性。

重复在测试集调 Prompt 会过拟合。维护滚动新鲜集和时间切片。公开基准可能被预训练污染；用 canary、改写、私有数据和过程证据增强可信度。

## 错误分析

每轮抽样失败，建立互斥尽量清晰的 taxonomy：意图理解、知识缺失、检索、推理、格式、工具、权限、安全或评价错误。统计严重度与频率，选最重要类别修复，再增加回归样本。

## 在线评测

A/B 测试看任务完成、留存、人工升级、撤销/纠错等真实指标，同时设安全 guardrail。随机化单位要避免同一用户跨版本污染；考虑新奇效应、学习效应和延迟。高风险变更先 shadow/canary，不直接全量。

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
