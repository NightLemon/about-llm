# 持续学习、模型合并与机器遗忘

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：模型持续维护、合并、编辑和删除需求负责人。
- **先修**：训练评测、数据切分、checkpoint 和分布偏移基础。
- **首次阅读**：更新类型 → 评测矩阵 → replay/正则 → 合并 → unlearning。
- **完成信号**：能同时报告新任务收益、旧任务保持、成本和删除证据。
- **卡住时**：回到[训练数据工程](data.md)和[评测](../quality/evaluation.md)。

</div>

先看本章的两任务实验。一个小模型先把 Task A 学到 100% accuracy，再训练规则相反的 Task B。没有 replay 时，
Task A 最终只剩约 2.7%；把旧样本与新样本一起训练，两项任务都能保持 100%。

这组数字不是为了证明 replay 总有效，而是让“更新后旧能力消失”变成可观察的状态变化。真实模型发布后还会遇到
新事实、新术语、新工具、新用户分布和新政策。第一步不是立刻继续训练，而是判断变化究竟应该进入参数、
外部知识库、Prompt，还是确定性业务代码。

把可删除的时效事实写入参数，或用 RAG 修复基础推理能力，都会让系统更难维护。

## 1. 先判断这件事该不该改权重

| 变化 | 首选候选 | 为什么 |
| --- | --- | --- |
| 频繁变化、需来源的事实 | RAG/数据库/API | 可更新、可删除、可引用 |
| Prompt/格式/工作流 | Prompt、schema、deterministic code | 便于回滚和验证 |
| 新领域语言分布 | DAPT/continued pretraining | 学习术语、风格和分布 |
| 新任务行为 | SFT 或 adapter | 监督清晰、部署可隔离 |
| 偏好/安全边界 | preference training + system policy | 同时需要行为与外部约束 |
| 单点参数事实实验 | model editing | 快速但需严格 locality 测试 |
| 大范围能力/数据变化 | 联合重训 | 成本高但控制最完整 |

这些不是互斥选项。例如先用 RAG 提供最新法规，再用 SFT 教模型怎样引用和拒答；事实仍由外部系统提供。

## 2. 只看新任务，会把遗忘藏起来

假设模型依次学习 \(T\) 个任务或时间段。采用从 0 开始的索引，\(R_{i,j}\) 表示完成任务 \(i\) 的训练后，
模型在任务 \(j\) 上的 accuracy；\(b_j\) 是任何顺序训练前的同一模型基线。

要计算 forward transfer，每个阶段还要评测尚未训练的未来任务，而不只是“所有已学任务”。

不能只比较更新前后目标域。至少报告：

- **plasticity**：新任务/新域学到了多少；
- **retention**：旧任务保留多少；
- **forward transfer**：旧知识是否帮助新任务；
- **backward transfer**：学习新任务后旧任务变好或变差；
- **safety retention**：拒答、权限和高风险行为是否回归；
- **calibration/efficiency**：置信度、长度、延迟和成本是否变化。

### 2.1 把 ACC、BWT、FWT 和 forgetting 写成公式

以下口径只适用于“accuracy 越高越好”；loss 等越低越好的指标必须重新处理方向。最终平均准确率为：

\[
\mathrm{ACC}=\frac{1}{T}\sum_{j=0}^{T-1}R_{T-1,j}.
\]

Backward transfer（BWT）比较旧任务最终表现与刚学完该任务时的对角线表现：

\[
\mathrm{BWT}=\frac{1}{T-1}\sum_{j=0}^{T-2}
\left(R_{T-1,j}-R_{j,j}\right).
\]

本仓库把 forgetting 定义为包含最终阶段的非负 peak-to-final drop：

\[
F_j=\max_{j\le i\le T-1}R_{i,j}-R_{T-1,j},
\qquad
\overline F=\frac{1}{T-1}\sum_{j=0}^{T-2}F_j.
\]

因此最终表现超过历史值时 \(F_j=0\)，最后一个任务按定义也是 0；这种口径不会把正向 backward transfer 写成“负遗忘”。有些论文从最大值中排除最终阶段，因而允许负值；复现实验必须说明是否包含当前阶段，不能只写 `forgetting`。

Forward transfer（FWT）比较学习任务 \(j\) 之前的表现与独立的 pretraining baseline：

\[
\mathrm{FWT}=\frac{1}{T-1}\sum_{j=1}^{T-1}
\left(R_{j-1,j}-b_j\right).
\]

对角线 \(R_{j,j}\) 只是“刚学完任务后的 accuracy”，不能单独证明 plasticity；学习增益还需要训练前或只训练到 \(j-1\) 的同任务对照。平均值会掩盖关键任务，应同时报告完整矩阵、逐任务 forgetting、seed 分布与置信区间。

## 3. 旧能力是怎样被覆盖的

新数据梯度更新共享参数，可能覆盖旧任务依赖的表示；优化器状态、学习率、数据顺序与 normalization 也会影响更新路径。遗忘不只表现为准确率下降：

- 输出格式或语言风格变化；
- 校准恶化；
- 安全拒答边界漂移；
- 少数语言先退化；
- 原有工具 schema 不再稳定；
- 事实表面保留但推论关系失效。

目标域性能提高与旧域下降是 trade-off，不应只通过降低训练 loss 判断。

## 4. Replay：训练新任务时别完全丢掉旧分布

训练新数据时混入旧分布样本，是最直接的 retention 方法。

### 4.1 Buffer 设计

- uniform reservoir 保留历史代表样本；
- 按任务/语言/风险分层；
- 选择 boundary/hard examples；
- 使用原始数据、摘要、特征或 synthetic replay；
- 给安全/契约回归样本设置最低配额。

Replay ratio 越高通常越有利于保留，但会减少新域预算。应画 new-quality–retention Pareto，而不是固定一个经验比例。

容量为 \(m\) 的 uniform reservoir 可以单遍处理未知长度的 stream。先放入前 \(m\) 项；看到从 0 开始编号的
第 \(t\) 项时，从 \([0,t]\) 均匀抽一个整数 \(r\)，仅当 \(r<m\) 时替换槽位 \(r\)。

归纳可得，处理 \(N\) 项后，每项进入最终 reservoir 的边际概率都是 \(m/N\)。这只保证对 stream 中的记录均匀，
不保证类别、语言、用户、风险或时间覆盖均衡，也不会自动处理重复记录。

### 4.2 合规边界

Replay buffer 复制用户数据会延长保留周期。删除请求必须覆盖 buffer、cache、shard 和训练 lineage。Synthetic replay 减少直接存储，但可能遗漏长尾、泄露教师记忆或固化旧模型错误。

### 4.3 可运行的两任务控制实验

运行：

~~~powershell
python projects/single-gpu-finetuning/continual_replay_toy.py
~~~

脚本实际执行两阶段 PyTorch SGD，并输出完整 \(R\) 矩阵、ACC、BWT、FWT 与逐任务 forgetting。Task A 与
Task B 的标签规则相反，但输入包含显式 task-id feature，因此同一个 2→16→2 MLP 可以同时解出两项任务。
这样可以避免把“模型容量上不可能同时满足”误称为遗忘。

在固定 seed 的 CPU 样例中，无 replay 在学完 B 后把 A accuracy 从 1.0 降到约 0.027，旧任务 forgetting
约为 0.973；把**全部**旧样本与新样本按 1:1 联合 replay 后，两项 accuracy 都是 1.0，旧任务 forgetting 为 0。

两条路径的 FWT 都约为 -0.418，因为它由训练 B 以前的状态决定，B 阶段是否 replay 无法追溯改变它。
这个负值只说明：在当前样例中，模型学完 A、尚未学习 B 时，B accuracy 低于随机初始化基线。它不是
“replay 产生负迁移”的证据。

这是 task-incremental、单 seed、full-batch、全量旧数据的 CPU synthetic toy。它没有覆盖有限 buffer、
class/domain-incremental 的未知路由、真实 LLM/语料、安全 retention、隐私删除、计算开销或置信区间，
也不支持“replay 总能消除遗忘”的结论。

### 4.4 有限 Reservoir 与 20-seed 配对比较

运行正式 benchmark：

~~~powershell
python projects/single-gpu-finetuning/continual_replay_toy.py --benchmark
~~~

Benchmark 默认使用训练 seed 0–19。每个 seed 的 Task A checkpoint 都复制给 no replay、64/256 uniform reservoir
和 256/256 full replay 三条路径，因此 strategy difference 以 seed 为配对单位。

三条路径在 Task B 都更新 100 步，但每步分别呈现 256、320 和 512 个样本。所以它是
**optimizer-step matched，而不是 example/compute matched**。新样本呈现量都是 25,600，旧样本呈现量分别为
0、6,400 和 25,600。

当前 CPU/PyTorch 样例的聚合结果如下。区间是 5,000 次 percentile bootstrap 得到的 95% paired
seed-level difference interval：

| Strategy | mean old-task final acc | mean new-task final acc | mean final ACC | old-task acc gain vs no replay（95% CI） |
| --- | ---: | ---: | ---: | ---: |
| no replay | 0.1135 | 0.9994 | 0.5564 | — |
| 64-example reservoir | 0.5893 | 0.9922 | 0.7907 | +0.4758 [0.3389, 0.6104] |
| full replay | 0.9824 | 0.9854 | 0.9839 | +0.8689 [0.8340, 0.9025] |

有限 reservoir 的 new-task accuracy difference 为 -0.0072，区间为 [-0.0102, -0.0045]；full replay 为
-0.0141，区间为 [-0.0223, -0.0074]。这里确实出现 retention 提升与轻微 plasticity trade-off，
但 effect 只对固定 task、data 和优化配置成立。

任务数据在 seed 间完全相同，变化只来自初始化与 reservoir 选择。因此这个区间**不覆盖**新任务、数据采样、
超参数、硬件或真实部署不确定性。20 个连续 seed 也不是任意目标总体的概率样本。

每个 run 内部的 `confidence_intervals_computed=false` 表示单个 \(R\) 矩阵没有区间；顶层
`paired_vs_no_replay` 才是跨 seed 比较。Artifact 还逐 seed 保存有限 reservoir 的实际索引和完整矩阵，
但没有测 wall time、energy 或隐私/删除成本。

因此，不能据此断言 64 是最佳容量，也不能忽略额外样本呈现量后宣称“同成本获益”。

## 5. 正则化与蒸馏

### 5.1 参数距离

简单 L2 约束：

\[
\mathcal L
=
\mathcal L_{new}
+\lambda\|\theta-\theta_{old}\|_2^2.
\]

它假设所有参数偏移同样重要。EWC 类方法用重要性 \(F_i\) 加权：

\[
\mathcal L
=
\mathcal L_{new}
+\frac{\lambda}{2}
\sum_iF_i(\theta_i-\theta_{old,i})^2.
\]

Fisher/重要性是数据与近似相关的；大型模型上存储和估计成本高，参数对称性也使“坐标距离”不完全等于函数距离。

### 5.2 Logit distillation

在 replay/anchor prompts 上约束新旧模型输出分布：

\[
\mathcal L_{distill}
=
D_{KL}(p_{old}(\cdot\mid x)\|p_{new}(\cdot\mid x)).
\]

它保留旧模型行为，也会保留旧错误与偏见。Top-k logits、temperature、tokenizer 和 mask 口径必须一致；只蒸馏最终答案无法保护中间 token distribution。

## 6. 参数隔离与 Adapter

冻结底座，只训练 LoRA/adapter/prefix，可减少底座参数遗忘并支持按域切换。但系统仍有边界：

- adapter 可能覆盖底座输出行为；
- 多 adapter 组合会干扰；
- base model 升级后 adapter 不保证兼容；
- tokenizer、chat template 和 target module 必须匹配；
- 冻结底座不代表旧任务质量绝对不变，因为 inference routing/prompt 也可能变化。

高隔离场景可按 tenant/domain 选择 adapter；路由错误则成为新的系统风险。

## 7. DAPT、TAPT 与继续预训练

- **DAPT**：在领域语料继续 pretraining，适应术语、风格和统计分布；
- **TAPT**：在更接近目标任务的未标注语料继续训练；
- **continued pretraining**：泛指从 checkpoint 延续自监督目标。

### 7.1 主要变量

- 新旧数据 mixture 与 replay；
- unique/consumed tokens 和重复 epoch；
- 较小学习率、warmup 与训练步数；
- tokenizer 是否覆盖新语言/符号；
- optimizer state 是恢复还是重置；
- checkpoint 是否保存原训练数据游标；
- 通用/领域 validation loss 与下游回归。

更换 tokenizer 会改变 embedding/output matrix 和 ID 契约，不是普通数据更新。扩词表需要初始化新 token、训练策略和旧文本回归。

### 7.2 何时停止

领域 loss 继续下降时，通用能力可能已经恶化。定期保存 checkpoint，联合观察：领域 loss/任务、通用 benchmark、安全、语言切片与 calibration。停止点是多目标选择，不是只取最后一步。

## 8. 持续 SFT 与偏好更新

新 SFT 可能造成 response style collapse、过度模板化和 refusal shift。偏好更新还会受新旧 rubric、标注者和 judge 漂移影响。

维护：

- 固定 anchor prompts 与历史 preference pairs；
- rubric/version lineage；
- 新旧 policy 的盲测 paired evaluation；
- response length、KL/log-ratio 与 benign refusal；
- current-policy rollouts，避免只在旧候选上评估。

政策发生实质变化时，应明确版本边界，不能把前后冲突标签混成“更多数据”而不记录时间。

## 9. 模型编辑：改一个事实，别顺手改坏邻居

模型编辑试图对少量事实/行为做局部更新。无论使用梯度、closed-form、low-rank update 或 learned editor，都应测：

- efficacy：目标输入更新；
- paraphrase：改写和多语言；
- portability：相关推论；
- locality：无关事实保持；
- specificity：相似实体不误改；
- persistence：保存、量化、继续训练后保持；
- conflict：RAG/旧参数/新编辑冲突处理。

单个 prompt 改对不代表知识全局一致。若事实经常变化或必须引用来源，外部知识存储通常更可控。

## 10. 权重合并不是把能力做加法

### 10.1 线性插值

对共享架构的两个 checkpoint：

\[
\theta_{merge}=(1-\alpha)\theta_A+\alpha\theta_B.
\]

只有当权重位于可兼容的表示 basin/坐标系时，插值才可能工作。相同架构但独立随机初始化的神经网络存在 hidden-unit permutation 和其他对称性，逐坐标平均通常没有语义对齐保证。

### 10.2 Task arithmetic

若多个 finetuned model 都从同一 base \(\theta_0\) 出发，定义 task vector：

\[
\Delta_i=\theta_i-\theta_0,
\qquad
\theta_{merge}=\theta_0+\sum_i\lambda_i\Delta_i.
\]

共享初始化提高坐标兼容性，但 task vectors 仍会冲突；线性相加不是能力线性组合定理。需要调系数、处理符号/幅度冲突并做完整评测。

### 10.3 LoRA 合并

LoRA update 为 \(\Delta W=sBA\)。把多个 adapter 的 dense updates 相加可得到

\[
W'=W+\sum_i\lambda_i\Delta W_i,
\]

但要求相同 base revision、target modules、shape、fan-in/out 和 scaling 约定。若要把和重新压成固定 rank，需要 SVD/近似，会产生误差。多个 adapter 顺序激活、并行加和和先 merge 再量化也可能不同。

## 11. 合并后要重新加载再验收

对每个候选至少比较：

- 每个单任务 checkpoint；
- base model；
- 简单 interpolation/average；
- joint training 或 multi-task baseline（若可行）；
- 不同 merge coefficients 与随机顺序。

测试每项能力、通用回归、安全、校准、生成长度和量化后表现。只展示合并后几个“成功样例”无法发现 destructive interference。

## 12. 每次更新都要能回答从哪里来

每个模型 artifact 记录：

- immutable parent model revision；
- 数据 manifest、过滤/删除状态和时间范围；
- tokenizer/chat template；
- code、dependency、precision 和 hardware；
- optimizer/scheduler 与随机状态；
- adapters/merge recipe；
- evaluation artifact 与 approval；
- compatible RAG index、prompt、tool schema 和 policy version。

回滚权重时也要回滚兼容 tokenizer、adapter、prompt、index 和 tool contract。只回滚 `model.safetensors` 可能产生新的接口错误。

## 13. 部署中的持续更新

推荐阶段：

1. offline replay：固定输入对比新旧 artifact；
2. shadow：新模型不影响用户，只记录差异；
3. canary：小流量、严格 guardrail；
4. gradual rollout：按 tenant/region 扩大；
5. post-deploy monitoring：质量、拒答、工具失败、延迟和成本；
6. rollback/reconciliation：处理模型动作产生的外部状态。

模型更新与 embedding/RAG index 更新可能不同步。Schema、embedding dimension、tokenizer 与 citation contract 都需要兼容检查。

## 14. Machine unlearning 不是删除数据库行

删除源数据与消除已训练模型中的影响是不同问题。

### 14.1 Gold-standard 对照

若可行，从未包含目标数据的 checkpoint/数据集重新训练（retrain-from-scratch 或 retrain-from-safe-checkpoint）是最清晰的对照，但成本高且随机训练差异使逐参数比较没有意义。应比较行为、攻击成功率和任务质量分布。

### 14.2 Approximate unlearning

方法包括 gradient ascent/negative training、scrubbing、influence approximation、adapter removal、model editing 和 distillation。它们可能让常见 prompt 不再复述，却未消除变体提取、membership signal 或内部表示影响。

### 14.3 威胁模型

声明攻击者：

- 只有黑盒 API，还是拥有 logits/weights/gradients；
- 是否知道目标样本和训练算法；
- 可查询次数和 prompt 变体；
- 目标是 verbatim extraction、membership inference、属性推断还是功能影响。

“我们测试的十个 prompt 不再输出”只支持这十个行为，不支持数学意义上的完全删除。

## 15. Unlearning 评价

- forget set 上的 extraction/membership attack；
- paraphrase、翻译、上下文诱导和相邻样本；
- retain set 与通用能力；
- 与从未训练目标数据的 retrained model 比较；
- 多 seed、attack strength 和 confidence interval；
- 训练/部署 artifact、cache、RAG index 与日志是否同步删除。

若只是输出过滤，应明确称为 mitigation，不称为参数遗忘证明。

## 16. 更新决策表

| 问题 | 若答案为“是” | 倾向方案 |
| --- | --- | --- |
| 事实需要来源、频繁更新或删除？ | 外部化更重要 | RAG/API |
| 需要学习新语言统计与术语？ | 参数适配有价值 | DAPT + replay |
| 只是输出格式/流程？ | 不必动全部权重 | Prompt/SFT/adapter |
| 多租户能力需隔离？ | 避免互相污染 | per-tenant adapter/routing |
| 只有一个事实需实验性修改？ | 局部但风险高 | editing + locality suite |
| 数据/能力范围整体变化？ | 局部方法可能不足 | joint retraining |

## 17. 发布门禁

### 学习与保留

- 新域/任务达到预定 gate；
- 旧任务逐项报告 forgetting，不只平均值；
- 安全、语言、工具和 calibration anchor 无回归；
- 对 replay ratio/regularization 做 Pareto 分析。

### Artifact

- parent、data、tokenizer、code 与 merge lineage 完整；
- adapter/base/quantization 兼容性检查；
- checkpoint 恢复后数据 sampler 与 mixture 一致；
- 回滚包包含 prompt/index/tool schema。

### 删除与合规

- replay buffer 和派生数据遵守 retention/deletion；
- unlearning claim 有攻击者与评价范围；
- 无法证明的影响明确标为 unknown；
- 外部 cache、index、日志与备份有单独处置。

## 18. 当前仓库证据边界

仓库已有 LoRA merge equivalence、QLoRA memory estimate、SFT 入口、模型 lineage 指引和评测门禁，
还实际运行了两任务 task-incremental 的 no/finite/full replay 对照与 20-seed 配对区间。

仓库没有在目标 LLM、真实时间序列、多任务/数据采样、compute-matched 配置或安全集上执行完整 benchmark，
也没有目标模型 unlearning 攻击实验。因此，当前结果只展示了几种方法在这些样例上的行为；灾难性遗忘和
机器遗忘是否得到解决，还需要目标模型与目标数据上的验证。

## 19. 常见错误结论

- **“目标域提高，所以更新成功”**：必须同时测旧域、通用、安全和效率。
- **“冻结 base 就不会遗忘”**：adapter、routing、prompt 和 template 仍会改变系统行为。
- **“同架构模型可以直接平均”**：独立初始化的表示坐标不保证对齐。
- **“多个 LoRA 相加就是能力相加”**：更新会冲突，rank 压缩和量化还有误差。
- **“输出不再复述就是完成 unlearning”**：变体攻击和参数影响可能仍存在。
- **“回滚模型文件就完成回滚”**：tokenizer、RAG、prompt 和工具契约必须兼容。

## 自测与实践

1. 为三个顺序任务构造 \(R_{i,j}\) 矩阵，分别计算 ACC、BWT、FWT 和本章口径的逐任务 forgetting。
2. 比较 replay、L2/EWC 和 logit distillation 分别保护什么、会保留什么错误。
3. 为什么两个从同一 base 微调的模型比两个独立初始化模型更适合 task arithmetic？
4. 写出合并两个 LoRA 前必须相同的五个契约。
5. 为一次删除请求定义 black-box 与 white-box 两种 unlearning threat model。
6. 给“最新法规问答”设计 RAG + SFT，而不是直接 DAPT 的更新与回滚方案。
