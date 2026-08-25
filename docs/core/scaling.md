# 规模化规律与预算决策

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：训练预算、模型规划和实验决策负责人。
- **先修**：对数、回归、FLOPs 口径和基本实验设计。
- **首次阅读**：四种口径 → 幂律拟合 → 计算最优点 → 可信外推 → 决策流程。
- **完成信号**：能给出预算账本，并标注拟合区间与不能外推的范围。
- **卡住时**：回到[机器学习与深度学习](../foundations/ml-dl.md)的实验设计部分。

</div>

**学习入口**：[预训练](../training/pretraining.md) · [分布式训练](../systems/distributed-training.md) ·
[评测统计](../foundations/evaluation-statistics.md) · [推理优化](../systems/inference-optimization.md)
{ .doc-nav }

Scaling law（规模化规律）描述的是：在**特定模型族、数据混合、训练目标和规模区间**内，
loss 怎样随参数、数据和计算量呈现可拟合的趋势。它可以帮助分配预算和设计小型试验；
模型是否真的更有用，仍要通过下游任务评测。拟合区间或实验条件改变后，系数也需要重新验证。

## 1. 先区分四种尺度

| 符号 | 含义 | 常见混淆 |
| --- | --- | --- |
| \(N\) | 参数量，通常指被训练的 dense model parameters | 不等于 checkpoint 字节或 MoE 每 token 激活参数 |
| \(D\) | 进入训练目标的 token 数 | 不等于原始字符、去重前 token 或唯一 token |
| \(C\) | 某种定义下的训练计算量 | 不等于 GPU-hours、租用费用或实测有效 FLOPs |
| \(L\) | 固定分布与 tokenizer 下的 loss | 不等于事实性、Agent 成功率或用户价值 |

模型“规模”还可能指层数、hidden size、上下文长度、总专家参数或激活参数。报告结果时必须给出具体定义。

## 2. `6ND` 是预算近似，不是测量值

对常见 dense autoregressive Transformer，训练核心矩阵乘的模型 FLOPs 常粗略写成

\[
C\approx kND,\qquad k\approx 6.
\]

直觉上，每个参数对每个 token 的 forward 约需一次 multiply-add，backward 还需激活梯度与权重梯度；若把 multiply 与 add 分别计作一个 FLOP，会得到约 6 的量级。

这个估计会遗漏或简化：

- attention 随序列长度变化的 \(T^2\) 项；
- embedding、normalization、softmax、loss 与 optimizer kernel；
- padding、packing 低效和被 mask 的 token；
- activation checkpoint 导致的重算；
- MoE 的稀疏激活、router 和 all-to-all；
- 数据加载、通信、故障恢复和空闲时间；
- 硬件实际执行的低精度/稀疏计算计数口径。

因此 `6ND` 适合做同口径的一阶预算，不应被写成 profiler 读数。若比较系统吞吐，应使用实际模型 FLOPs 定义、硬件峰值口径和 wall-clock 测量，参见[分布式训练](../systems/distributed-training.md)。

仓库提供显式命名的预算函数：

```python
from about_llm.scaling import estimate_dense_training_flops

flops = estimate_dense_training_flops(
    num_parameters=1e9,
    training_tokens=20e9,
)
assert flops == 1.2e20
```

把代码中的数字展开：

\[
6\times 10^9\times 20\times10^9
=1.2\times10^{20}\ \text{FLOPs}.
\]

这相当于 120 EFLOP 的**工作量**，不是每秒 120 EFLOP 的吞吐率。`20e9 / 1e9 = 20` 只是本例选择的
20 tokens/parameter；它来自输入参数，不是函数推导出的通用最优比例。实际 GPU 用时还要知道有效吞吐和停机开销。

## 3. 损失的经验幂律

一种常用的可分离拟合形式是

\[
L(N,D)
=L_\infty
+aN^{-\alpha}
+bD^{-\beta},
\]

其中 \(L_\infty\) 表示在该数据分布、表示与目标下的拟合损失下限；\(a,b,\alpha,\beta\) 来自实验。

### 3.1 它实际声称什么

在拟合覆盖的范围内，增加参数或数据后，excess loss 近似按幂律下降，且 marginal return 递减。它没有声称：

- 所有架构共享同一指数；
- 数据质量变化后系数不变；
- 小模型拟合能无限外推到任意大模型；
- token loss 的微小改善必然带来某项能力改善；
- \(L_\infty\) 是语言或智能的物理极限。

“irreducible loss”只是给定问题设定下的模型项，可能随 tokenizer、上下文、数据噪声和训练目标变化。

### 3.2 为什么 log-log 图常近似直线

若忽略其他项，\(L-L_\infty\approx aN^{-\alpha}\)，则

\[
\log(L-L_\infty)
=
\log a-\alpha\log N.
\]

因此 log-log 图斜率约为 \(-\alpha\)。但 \(L_\infty\) 估错会让直线弯曲；在多个项同时主导时也不能直接从两点读指数。

## 4. Compute-optimal 的解析直觉

在计算预算 \(C=kND\) 下，用

\[
D=\frac{C}{kN}
\]

代入 loss 的可变部分：

\[
L-L_\infty
=aN^{-\alpha}
+b\left(\frac{kN}{C}\right)^\beta.
\]

第一项随 \(N\) 增大而下降，第二项因为 token 预算被挤压而上升。令导数为零：

\[
N^*
=
\left[
\frac{\alpha a}{\beta b}
\left(\frac{C}{k}\right)^\beta
\right]^{\frac{1}{\alpha+\beta}},
\qquad
D^*=\frac{C}{kN^*}.
\]

由此可见

\[
N^*\propto C^{\frac{\beta}{\alpha+\beta}},
\qquad
D^*\propto C^{\frac{\alpha}{\alpha+\beta}}.
\]

### 4.1 沿一条固定预算曲线手算

先用一个无量纲小例子看清取舍。设 \(C=100\)、\(k=1\)，并令 \(a=b=\alpha=\beta=1\)、
\(L_\infty=0\)。预算约束给出 \(D=100/N\)，于是：

\[
L(N)=\frac{1}{N}+\frac{1}{D}
=\frac{1}{N}+\frac{N}{100}.
\]

| 参数量 \(N\) | Token 数 \(D=100/N\) | 参数不足项 \(1/N\) | 数据不足项 \(1/D\) | 总 loss |
|---:|---:|---:|---:|---:|
| 1 | 100 | 1.00 | 0.01 | 1.01 |
| 2 | 50 | 0.50 | 0.02 | 0.52 |
| 5 | 20 | 0.20 | 0.05 | 0.25 |
| **10** | **10** | **0.10** | **0.10** | **0.20** |
| 20 | 5 | 0.05 | 0.20 | 0.25 |
| 50 | 2 | 0.02 | 0.50 | 0.52 |
| 100 | 1 | 0.01 | 1.00 | 1.01 |

表中 loss 完全由假设公式算出，并不是训练模型得到的观测值。
从左向右增加参数时，参数不足项持续下降，但可用 token 被挤压，数据不足项持续上升。
本例在 \(N=D=10\) 处平衡两项；这就是导数为零背后的直觉。

同一组数字可以直接交给仓库函数：

```python
from about_llm.scaling import compute_optimal_under_power_law

estimate = compute_optimal_under_power_law(
    compute_flops=100,
    parameter_coefficient=1,
    data_coefficient=1,
    parameter_exponent=1,
    data_exponent=1,
    flops_per_parameter_token=1,
)
assert estimate.num_parameters == 10
assert estimate.training_tokens == 10
assert estimate.modeled_loss == 0.2
```

这里把 \(k\) 设为 1 只是为了手算；训练预算回到 `6ND` 口径时应使用 \(k=6\)。函数要求输入为有限正数，
`tests/test_scaling.py` 会同时检查解析最优点和预算恒等式。

### 4.2 算力增加 16 倍时怎样分配

系数 \(a,b\) 决定当前最优点落在哪里，指数 \(\alpha,\beta\) 决定增加算力后参数与数据增长多快。
例如 \(\alpha=0.4,\beta=0.3\) 时，算力增加 16 倍会得到：

| 数量 | 算力指数 | 增长倍数 |
|---|---:|---:|
| 参数量 \(N^*\) | \(\beta/(\alpha+\beta)=3/7\) | \(16^{3/7}\approx3.281\) |
| Token 数 \(D^*\) | \(\alpha/(\alpha+\beta)=4/7\) | \(16^{4/7}\approx4.876\) |
| 预算乘积 \(N^*D^*\) | 1 | \(3.281\times4.876\approx16\) |

所以“算力 16 倍”并不自动意味着参数和数据各 4 倍；只有两个指数相同时才会这样分配。
这些倍数仍属于同一套拟合模型。换 tokenizer、数据混合、上下文长度或架构后，需要重新取得系数和指数。

## 5. 如何取得可信拟合

### 5.1 IsoFLOP 实验

对多个计算预算 \(C_1,C_2,\ldots\)，分别训练若干不同 \((N,D)\) 组合，找每个预算下 loss 最低的点，再拟合最优 \(N,D\) 随预算的变化。每条曲线只有一两个点无法可靠确定最优区间。

### 5.2 控制变量

至少固定或记录：

- tokenizer、上下文长度与 packing；
- 训练数据快照、混合比例、重复轮次与过滤；
- architecture family、参数计数口径与 initialization；
- optimizer、scheduler、batch/token budget 和停止条件；
- validation 分布、loss reduction 和评测时点；
- 硬件失败、跳步、恢复与实际 consumed tokens。

小模型可能需要不同学习率、depth/width 比和 batch size。若所有尺度机械使用同一超参数，测到的可能是优化不足而不是容量规律；若每个尺度都做不同程度的调参，又会引入调参预算不公平。两者都应披露。

### 5.3 拟合与不确定性

- 在 log space 拟合会改变误差权重，应检查 residual；
- 随机种子、数据抽样和 checkpoint 波动都需要误差条；
- 预留模型尺度或计算预算做 out-of-fit validation；
- 报告参数协方差、bootstrap 区间或至少敏感性分析；
- 不要只展示最贴合直线的尺度点；
- 外推距离越远，结构变化与数据约束越可能破坏规律。

## 6. Token 不等价

把所有 token 视为同质单位只是一阶建模。实际有效性受以下因素影响：

- 语言与领域覆盖；
- 事实准确性、格式和解析质量；
- exact/near duplicate；
- 模板、导航、垃圾和机器生成循环；
- 课程顺序与数据混合；
- 与目标评测的重叠或污染；
- 许可、隐私和可删除性。

高质量 token 的训练收益可能更大，但“质量”通常由启发式或另一个模型打分，本身带偏差。过滤过强会减少多样性、少数语言或困难样本。

### 6.1 重复 epoch

当唯一数据有限时，多次训练同一数据并非自动无效；模型可能在前几轮继续学习。但边际收益会变化，且记忆、过拟合和隐私风险上升。报告 \(D\) 时应区分 unique tokens 与 consumed tokens。

### 6.2 合成数据

合成数据可提供解题轨迹、罕见格式和教师蒸馏信号，但会继承教师错误与风格。若反复用模型输出训练后继模型，又缺少真实分布锚点和质量过滤，分布可能收缩或放大偏差。“token 更多”不能证明信息更多。

## 7. Dense 与 MoE 的尺度口径

Dense 模型每个 token 通常使用绝大部分层参数。MoE 模型有：

- **total parameters**：所有专家和共享层参数；
- **active parameters per token**：router 为某 token 选择的专家加共享部分；
- **training FLOPs**：受 top-k routing、capacity、dropped tokens 和重算影响；
- **memory/communication**：即使专家未对当前 token 激活，总权重仍要存储或分片，all-to-all 也有成本。

MoE 的存储和计算使用不同口径。Checkpoint 大小与分片主要受总参数量影响；单个 token 的主干计算更接近
激活参数量，再加上路由与通信成本。比较 MoE 和 dense 模型时，应同时列出总参数、激活参数和每 token FLOPs，
不能让一个“参数量”数字承担三种含义。

## 8. 上下文长度改变成本模型

在标准 full attention 中，单层 attention score 的计算/存储随序列长度包含二次项，而线性投影和 MLP 更接近按 token 线性增长。于是序列很短时 `6ND` 可能较好，序列很长时 attention 项不可忽略。

相同数量的训练 token 可以被组织成许多短序列，也可以组成少量长序列。两者的 padding、packing 和 Attention
成本不同。长上下文训练还可能使用局部或稀疏 Attention、序列并行和激活重计算，因此最终预算要按实际序列分布
与执行 kernel 重新估算。

## 9. 从 loss 到能力的非线性

### 9.1 “涌现”为什么难判断

若任务只有 exact match 0/1，模型给正确答案的概率从 0.1 平滑升到 0.6，在小样本图上可能像突然跨过门槛。Few-shot prompt、答案解析器和采样次数也会改变观察到的阈值。

这不证明所有能力都平滑，也不否定可能的机制转变。更可信的证据需要：

- 更多规模与训练时点；
- 连续指标或 token-level probability；
- 多 prompt、多 seed 和置信区间；
- 难度分层与独立测试集；
- 排除解析、污染和训练数据阶段变化；
- 若声称机制变化，提供内部或因果证据。

### 9.2 Benchmark saturation

当简单基准接近满分，继续 scale 的收益看不见；换更难基准又可能同时改变领域、格式和污染风险。应使用覆盖不同难度的 item response 曲线，而不是只比较一个总分。

## 10. 训练最优不等于产品最优

Compute-optimal 常优化固定训练预算下的 validation loss。产品目标还包含：

- 推理权重内存与副本数；
- 每 token 延迟和能耗；
- 峰值流量下的并发；
- 上下文与 KV Cache；
- 微调、量化和部署生态；
- 模型生命周期内的总请求量。

高调用量产品可能愿意在训练阶段让较小模型看更多 token，以降低长期 serving 成本。反之，低调用量且训练昂贵的内部模型可能选择不同点。正确问题是生命周期总成本下的质量前沿，而不是单一 `training FLOPs` 最小值。

### 10.1 一个生命周期目标

可把决策写成约束优化：

\[
\min_{N,D,q,s}
\quad
C_{train}(N,D)+Q\,C_{serve}(N,q,s)
\]

满足质量、延迟、显存和风险约束。其中 \(Q\) 是生命周期请求量，\(q\) 是量化/精度策略，\(s\) 是 serving 配置。这个式子提醒我们成本项存在，不声称它们都能被一个精确函数捕获。

## 11. 推理时规模化

Test-time compute 包括更长推理、多候选、beam/search、verifier、工具调用和迭代修正。增加预算只有在以下条件下可能有收益：

1. generator 能产生有意义且不完全相关的候选；
2. verifier/奖励能够区分正确与错误；
3. search 不被错误状态或提示注入劫持；
4. 任务允许验证或分解；
5. 额外延迟与成本可接受。

若 generator 和 verifier 共享同一系统性误解，多采样会重复同类错误。应画 quality–compute curve，报告失败切片，而不是只比较“思考开/关”。

## 12. 墙钟时间与硬件利用率

训练用时近似为

\[
t\approx
\frac{C_{model}}{P_{peak}\cdot MFU},
\]

其中 \(P_{peak}\) 必须与计算精度和 FLOP 计数口径一致，MFU 是模型 FLOPs 利用率的某种定义。该式忽略启动、评测、checkpoint 和故障时间，只适合粗略核算。

增加 GPU 可以缩短固定训练任务，但加速通常达不到设备数量的倍数。损失主要来自三处：

- Collective 通信需要启动时间和带宽；
- 负载不均、流水线空泡和数据等待会让部分设备空闲；
- 切分后矩阵变小，kernel 效率可能下降。

报告结果时，分别给出模型 FLOPs、硬件 FLOPs（若能可靠取得）、每秒 token、有效 token 和实际墙钟时间。
固定任务的强扩展效率应由这些实测量计算。

## 13. 决策流程

1. 定义目标分布、tokenizer、loss 与下游 gate。
2. 在可承受尺度做多点 pilot，检查优化稳定和数据瓶颈。
3. 用 isoFLOP 或 joint fit 估计系数，并保留误差条。
4. 只在已验证区间附近外推，做系数敏感性分析。
5. 把 training optimum 转成 serving 生命周期候选。
6. 对候选做真实单卡/服务基准、量化和下游评测。
7. 记录 consumed tokens、实际 wall clock 与偏离预算的原因。

## 14. 常见错误结论

- **“`6ND` 就是训练实际 FLOPs”**：它是省略多项成本的 dense budgeting approximation。
- **“某篇论文的 Chinchilla ratio 是固定常数”**：最优配比依拟合指数、数据、架构和目标而变。
- **“参数更少就一定推理更便宜”**：上下文、激活、MoE 通信、量化和 serving kernel 同样重要。
- **“MoE 的 active parameters 就是模型总大小”**：激活计算与权重存储使用不同口径。
- **“loss 幂律证明所有能力平滑增长”**：下游指标、阈值和机制都可能非线性。
- **“多采样必然提高答案”**：候选相关性和 verifier 错误会限制收益。

## 自测与实践

1. 在可分离 power law 和 \(C=kND\) 下推导 \(N^*\) 与 \(D^*\)。
2. 为什么不同数据混合拟合出的 \(a,b\) 不能直接用于另一个模型族？
3. 一个 7B dense 模型与 7B active/46B total MoE 应报告哪些不同尺度？
4. 设计至少四个 \((N,D)\) 点的 isoFLOP 实验，并说明如何选择学习率。
5. 使用 `about_llm.scaling` 让 compute 增加 16 倍，验证参数和 token 的理论增长指数。
6. 为一个月请求量 100 万与 10 亿的产品分别讨论 training-optimal 与 lifecycle-optimal 可能如何变化。

运行本章的预算、最优点、增长指数和非法输入对账：

~~~powershell
python -m pytest tests/test_scaling.py -q
~~~
