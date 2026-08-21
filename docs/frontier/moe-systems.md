# MoE 系统：从 Router 到 Expert Parallel

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解 MoE 训练、路由、通信和推理部署的算法与系统工程师。
- **先修**：[Transformer](../core/transformer.md)、softmax、MLP 与分布式 collective 基础。
- **首次阅读**：前向 → 参数口径 → capacity → dispatch → all-to-all → 推理。
- **完成信号**：能画出一个 token 从 source rank 到 expert owner 再返回的完整路径。
- **卡住时**：先用 4 个 tokens、2 个 experts、top-1 在纸上完成 routing。

</div>

**专题导航**：[前沿总览](reasoning-long-context-moe.md) · [DeepSeek](../models/deepseek.md) · [分布式训练](../systems/distributed-training.md) · [证据台账](../evidence/frontier-controls.md)
{ .doc-nav }

Mixture of Experts（MoE）用条件计算扩大模型容量：每个 token 只经过少数 experts，而不是激活全部 MLP。

它没有让系统问题消失。总权重仍需存储和分片，router 可能失衡，tokens 需要跨设备 dispatch，尾部 expert 会决定延迟。

## 一层 MoE 怎样前向

设有 \(E\) 个 routed experts。Router 对 token hidden state \(x\) 输出 scores：

\[
z=W_r x,\qquad p=\operatorname{softmax}(z).
\]

选择 top-\(k\) expert 集合 \(S(x)\)，输出可写为：

\[
y=
\sum_{e\in S(x)}
\tilde p_e(x)f_e(x)
+f_{\mathrm{shared}}(x),
\]

其中 \(\tilde p_e\) 表示选择后可能重新归一化或缩放的 gate，shared expert 是否存在取决于具体架构。

一次前向至少经历：

~~~text
router scores
→ top-k selection
→ capacity decision
→ token dispatch
→ expert MLP
→ weighted combine
→ residual path
~~~

Config 中出现 top-k 或 expert 数，只是静态 marker。Tie-break、normalization、capacity、drop/reroute 和训练梯度必须核对目标代码与 runtime。

## 三本参数账不能混

### Total parameters

全部 experts、router、attention、embedding 和其他模块的唯一权重。它决定 artifact 大小、存储和总体分片需求。

### Active parameters

一个 token 实际经过的 selected experts 与共享模块。它是条件计算口径，不等于实际 FLOPs 或 latency。

### Resident parameters

当前 device/rank 实际持有的权重。Expert parallel 可以让每张卡只持有部分 experts，但 attention、router 或 shared weights 可能复制。

因此“总参数大、激活参数小”不能推出：

- 单卡能加载；
- 显存按 active/total 比例下降；
- 端到端速度更快；
- 通信可以忽略。

## Top-k 是离散选择，也有连续梯度

Router scores 通常通过 softmax/sigmoid 等连续函数产生。Top-k indices 是离散选择；selected gate values 仍参与加权并可以接收 gradient。

没有被选中的 experts 对该 token 通常没有直接 expert-output gradient，但 router loss、load-balance objective 或其他设计可能提供信号。

需要明确：

- score function；
- top-k group/tie-break；
- selected gate normalization；
- straight-through 或其他 estimator 是否存在；
- auxiliary/z-loss；
- shared experts；
- inference 与 training routing 是否相同。

字符串字段名不能自动补出这些语义。

## Capacity 回答“expert 收不下怎么办”

若一个 batch 有 \(N\) 个 tokens，每 token 选 \(k\) 个 routed experts，平均 assignment 数为 \(Nk/E\)。

常见教学容量：

\[
C=
\left\lceil
\text{capacity factor}\times \frac{Nk}{E}
\right\rceil.
\]

真实实现可能按 group、sequence、rank 或全局 batch 计算，也可能 dropless。必须绑定具体 contract。

当 expert assignments 超过 \(C\) 时，需要定义：

- 按 score、位置还是稳定顺序保留；
- overflow 丢弃、reroute 还是送到 backup；
- dropped path 怎样与 residual 组合；
- gate 是否重新归一化；
- 多 rank 是否基于全局竞争；
- 统计是在 selected 还是 accepted 上。

一个 capacity 数字没有这些规则就不能复现实验。

### Selected、accepted、executed 分开

~~~text
selected by top-k
→ accepted by capacity
→ dispatched to owner
→ executed by expert
→ returned and combined
~~~

每一步的计数可能不同。只记录 top-k histogram 会漏掉 capacity drop 和 transport failure。

## Load balance 不只是一个平均数

Router collapse 会让少数 experts 过载，其他 experts 几乎空闲。

观察：

- selected/accepted tokens per expert；
- coefficient of variation、max/mean；
- overflow/drop/reroute；
- gate entropy；
- per-expert compute time；
- all-to-all bytes；
- straggler 与 step time；
- 不同语言/任务/位置的路由切片。

Auxiliary load-balance loss 是 proxy。它下降不保证质量、通信或尾延迟改善。

均匀路由也未必是最终目标：不同 experts 可能学习不同分工。需要在质量与系统平衡之间做 paired experiment。

## Expert parallel 的 token 路径

假设 source rank 持有输入 tokens，expert owner ranks 各持有部分 experts：

~~~text
source rank
→ local router / top-k
→ pack token + gate + metadata by owner
→ all-to-all dispatch
→ owner expert forward
→ all-to-all return
→ source unpacks and combines
~~~

Metadata 至少要让返回结果恢复：

- source rank；
- source token position；
- selected expert；
- gate weight；
- assignment slot；
- padding/capacity status。

错序或 duplicate metadata 可能保持 tensor shape 正常，却把 expert output 加到错误 token。

## Global capacity 与 all-to-all 是不同证据

用 all-gather/all-reduce 统计全局 selected counts，可以实现 global capacity decision，但没有实际把 token 送给 expert owner。

真正 expert dispatch 需要 all-to-all 或等价通信：

1. 每个 source 计算 send splits。
2. 所有 ranks 交换 split sizes。
3. Pack token/gate/metadata。
4. Variable-size all-to-all。
5. Owner 执行本地 experts。
6. Reverse all-to-all 返回结果。
7. Source 按 metadata combine。

“使用了 distributed collective”不能自动写成 expert parallel 已实现。要说明执行的是哪种 collective 和哪条数据流。

## Backward 还要处理两类参数

Expert parameters 只在 owner 上更新；router 可能在每个 data/expert rank 上复制。

训练时需要：

- Reverse dispatch 的 autograd；
- Router gradients 在复制 ranks 间归约；
- Owner expert gradients；
- Capacity/drop mask 与 forward 完全一致；
- Global loss normalization；
- Optimizer state placement；
- Checkpoint sharding 和 resume。

一次 all-to-all forward 对账只检查前向交换与合并。Backward、optimizer 和训练收敛需要各自的实验。

### Loss normalization 很容易错

不同 ranks 接受的 tokens 数可能不同。简单平均每 rank loss，会让 token 少的 rank 权重过大。

目标通常要按 global valid tokens 或明确 assignments 聚合。记录 numerator、denominator 和 world-size factor，
再与单进程参考实现的结果逐项比较。

## MoE 与其他并行怎样组合

可能同时存在：

- data parallel；
- tensor parallel；
- pipeline parallel；
- expert parallel；
- sequence/context parallel。

每个维度拥有不同 process group。Router gradient、expert weight、attention weight、optimizer state 和 activation 应在哪个 group 同步，必须画清楚。

先在小 world size 上分别验证，再组合。一次把五种 parallelism 全开，会让错误难以定位。

## MoE 推理的瓶颈

推理时没有 backward，但仍有：

- 总权重加载与分片；
- token-level routing；
- 小矩阵/低 occupancy；
- all-to-all latency；
- expert imbalance；
- dynamic batching 与 token reordering；
- KV Cache 和 attention；
- quantization/kernel support。

Prefill token 多，可能更容易形成较大 expert batches；decode 每序列每步只有少量 tokens，更容易受通信和小 batch 影响。

因此分别测 prefill/decode，不要只报平均 tokens/s。

## Capacity 与服务过载是两层

MoE expert capacity 决定模型层内如何处理 assignments；服务 admission/queue capacity 决定请求是否进入系统。

即使 MoE 路由内部使用 dropless，服务层仍可能因为 GPU memory、queue deadline 或并发上限而无法接收请求。
路由 capacity 和服务 capacity 位于不同层级，应分别命名和观测。

## Expert 是否“学会了技能”

路由统计可以显示某些 token/语言/任务更常进入某 expert，但不能直接证明 expert 具有可解释技能。

更强证据需要：

- 控制输入特征；
- 跨 seed/checkpoint 稳定性；
- expert ablation 或 intervention；
- 替换/屏蔽后的因果影响；
- 防止 token frequency、position 和 length 混杂；
- 多任务和 out-of-distribution 验证。

一张 top-token word cloud 是描述，不是因果解释。

## 一个纸笔到分布式的实验

### Level 1：手算 routing

4 tokens、2 experts、top-1。手算 scores、selected expert、gate 和 output。

### Level 2：加入 capacity

设置每 expert capacity=1，预测哪些 token 被保留。分别尝试 position priority 和 score priority。

### Level 3：训练 router 与 experts

用 tiny MLP 构造目标，确认 selected gate、router 和 expert gradients。加入一个全部路由同 expert 的 imbalance case。

### Level 4：两 rank dispatch

Rank 0/1 各持有一个 expert，真实交换 token + metadata，再与单进程参考实现对账。

### Level 5：Backward 与 optimizer

对齐 forward output、loss、router gradient、expert gradient 和一步更新。

仓库入口：

~~~powershell
python projects/transformers-basics/moe_routing.py
python projects/transformers-basics/moe_training_control.py
python -m pytest tests/test_moe_routing.py tests/test_moe_training.py -q
~~~

分布式 Gloo/all-to-all 的精确命令和边界见[证据台账](../evidence/frontier-controls.md)。

## 常见错误

- 把 total parameters 当 active parameters，或反过来。
- 用 active/total 比例直接推断显存和速度。
- 只记录 selected experts，不记录 accepted/dropped。
- Capacity 没有定义 priority、reroute 和 normalization。
- Global count collective 被写成 token-to-owner dispatch。
- Forward all-to-all 通过就声称 backward/optimizer 正确。
- Per-rank mean loss 代替 global token-weighted loss。
- 把 CPU/Gloo 固定样例的结果外推成 CUDA/NCCL 性能。
- Router 分布图直接解释为专家技能。

## 面试时怎样回答

面对“解释 MoE”，按 token 路径回答：

1. Router 为 token 打分并选择 top-k。
2. Capacity 决定 assignments 是否接受或 reroute。
3. Expert parallel 将 token 发到 owner。
4. Owner 执行 MLP，再返回 source combine。
5. 训练还要归约 router gradient 和更新 owner experts。
6. 评测同时看 total/active/resident、质量、负载、通信和尾延迟。

继续追问时，应能说明为什么 selected 与 accepted 不同，以及 all-gather counts 为什么不等于 all-to-all dispatch。

## 自测

1. Total、active 与 resident parameters 分别影响什么？
2. Top-k indices 不可导时，router 为什么仍可能得到 gradient？
3. Capacity overflow 的四种关键 policy 是什么？
4. Token 跨 rank 往返需要哪些 metadata？
5. 为什么 decode 阶段比 prefill 更容易出现小 expert batch？
