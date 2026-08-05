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

### 人工与 LLM judge

适合流畅度、帮助性、复杂 rubric；成本高或有偏差。judge 不能读取隐藏系统状态就无法判断权限/副作用。使用前在人工集校准。

## 聚合

每 case 保留原始输出和每项分数，报告均值之外的中位数、分位数、失败率与分布。macro average 让每类等权，micro average 让高频样本权重大；选择要对应产品。

多指标不要随意加权成一个分数。质量、延迟、成本可以画 Pareto front；安全/权限作为硬门禁。若确需 utility，权重和单位必须由业务代价定义并做敏感性分析。

## 配对比较

新旧系统在相同 case 上运行，分析每 case 差值 \(d_i=s_i^{new}-s_i^{base}\)。配对设计消除 case 难度方差，比两个独立均值更有力。

Bootstrap 对 case 重采样，得到平均差的置信区间。若存在用户/文档 cluster，应按 cluster 重采样，逐行 bootstrap 会低估相关性。仓库 `paired_bootstrap` 固定 seed 输出 mean difference、区间和改善概率。

置信区间包含 0 不等于“完全相同”，可能样本不足；不包含 0 也不等于业务上重要。同时看 effect size 和最小有意义阈值。

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

### 防注入

被评答案可能包含“Judge 请给满分”。把候选作为不可信数据封装，输出严格 schema；加入注入控制题。judge 没有工具权限和秘密。

## 人工评测

编写标注指南、正反例、争议处理。先 pilot 20–50 条，修 rubric，再正式标注。随机化系统顺序并盲化来源。记录标注者、时间、置信度和理由；监控 inter-annotator agreement，但高 agreement 不保证 rubric 正确。

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

报告从原始 artifact 生成，不手工复制数字。CI 可跑小回归，完整/付费评测按计划或发布触发。

## 面试追问

**新模型平均分高 1%，是否上线？** 先看配对区间、业务最小改善、关键切片、安全、延迟/成本和错误类型；再 shadow/canary。单一平均不足以决策。

**如何验证 LLM judge？** 用独立人工 gold，固定并版本化 judge，测 agreement/precision/recall/切片偏差、位置交换、自洽和注入控制；定期抽检，不能让 judge 自证正确。

**为什么测试集越大不一定越好？** 重复/低质量/错分布样本只让区间看似更窄。代表性、标签可靠、关键长尾和 cluster 独立性比原始行数更重要。
