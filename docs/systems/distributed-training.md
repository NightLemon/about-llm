# 分布式训练：资源、并行与拓扑

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备多 GPU 训练、容量规划或并行拓扑设计的工程师。
- **先修**：[预训练](../training/pretraining.md)、显存状态和基本 collective 概念。
- **首次阅读**：先做资源账本和 global loss，再沿 DP、分片、TP、PP 逐步扩展。
- **完成信号**：能解释每种并行切了什么、通信发生在哪里，并核对设备数与 global batch。
- **卡住时**：暂时忽略框架 API，只画每个 rank 保存哪些 tensor、何时通信。

</div>

一张 24 GB GPU 放不下 7B 模型的全参数 Adam 训练。团队拿到一台 8 卡机器后，打开 FSDP、activation checkpointing
和混合精度，训练终于能够启动，但 loss 与单卡小规模 baseline 不同。

这时问题已经不是“哪个框架参数能省显存”，而是四件事混在了一起：

1. 参数、梯度、optimizer 和 activation 各占多少；
2. 八张卡分别持有什么，通信发生在哪里；
3. Global batch 与 loss denominator 是否仍和单卡相同；
4. Checkpoint、RNG 和 data cursor 能否在所有 rank 上恢复同一训练状态。

本页先把资源、global loss、并行维度和物理拓扑对齐。混合精度、checkpoint、故障恢复与数值验收集中在
[分布式训练正确性](distributed-training-correctness.md)。仓库没有目标多 GPU 或多机集群，GPU 吞吐与拓扑结论仍要在目标环境实测。

## 先做一张显存账本

设参数量为 \(P\)，每个元素占 \(b\) bytes，权重本身约为 \(Pb\)。训练时还可能同时存在：

```text
forward weights
FP32 master weights
gradients
optimizer moments
saved activations
temporary workspaces / communication buckets
allocator fragmentation / runtime context
```

一个常见的混合精度 Adam 配置可以这样逐项记账：

| 状态 | 常见格式 | 每参数字节数 |
|---|---:|---:|
| Forward weights | BF16 | 2 B |
| Master weights | FP32 | 4 B |
| Gradients | FP32 | 4 B |
| 两份 Adam moments | FP32 | 8 B |
| 合计 |  | 18 B |

18 B/P 尚未包含 activation、通信 buffer 和临时 workspace，也只是某一种实现的账本。

18 B/P 只是某一配置的账本。实现可能没有 master weights，gradient 可以是 BF16，moments 也可能量化或分片。
真实报告应列出每项 dtype、复制数和分片组，不能把一个经验常数当成框架规范。

### 训练峰值与 checkpoint payload 是两本账

上表的 18 B/P 描述一种可能同时驻留的训练状态，其中包含 4 B/P 的梯度。
若在完整 optimizer step 结束后保存 checkpoint，下一批梯度可以由 backward 重新生成。
在“BF16 forward weights + FP32 master weights + 两份 FP32 Adam moments”的固定条件下，
持久 tensor payload 为：

\[
2+4+2\times4=14\ \text{bytes/parameter}.
\]

14 B/P 没有包含 scheduler、scaler、RNG、data cursor、shard metadata、压缩和文件对齐，
也没有描述保存时 staging buffer 与训练临时状态的峰值。实现若没有 master weights、保存梯度，
或使用不同 optimizer/dtype，就必须重新枚举实际 state tensors。

### Activation 为什么单独列

Activation 的主要驱动因素是 micro-batch \(B\)、序列长度 \(T\)、隐藏维度 \(H\)、层数 \(L\)、数值格式和算子实现。

朴素 attention 还可能显式保存形状为 \(B\times heads\times T\times T\) 的分数或概率，因此长序列尤其昂贵。

FlashAttention 类算法避免保存完整注意力矩阵，但 Q/K/V、MLP 和残差相关状态仍要保存或重算。排查 OOM 时，
应同时查看框架已分配显存、allocator 预留显存、driver 读数和显存时间线；这些读数描述的不是同一个量。

## 多卡之前先固定 global batch 与 loss { #global-batch-loss-normalization }

每个 data-parallel rank 的 micro-batch 为 \(B_\mu\)，梯度累积次数为 \(A\)，数据并行度为 \(D\)：

\[
B_{global}=B_\mu A D.
\]

TP、PP 与 EP 的多个 rank 共同处理同一批样本，不增加独立样本数，因此不乘进 global batch。

语言模型的序列长度和 padding 常常不同。每个 optimizer update 的有效 target tokens 应按 mask 求和：

\[
N_{tokens}=\sum_{rank,microbatch,t}m_{rank,microbatch,t}.
\]

### Mean 还是 sum { #mean-or-sum }

如果第 \(i\) 个 rank 的 loss 总和是 \(S_i\)，有效 token 数是 \(n_i\)，全局平均 loss 是：

\[
L=\frac{\sum_i S_i}{N},
\qquad
N=\sum_i n_i.
\]

先计算每个 rank 的 \(S_i/n_i\) 再等权平均，会让有效 token 较少的 rank 权重过大。正确做法是聚合 loss 总和
与 token 数，或者使用数学等价的梯度缩放。

### DDP 默认 gradient mean 为什么多一个 world size

假设 DistributedDataParallel（DDP）对 \(D\) 个 rank 的 gradient 求平均。
若每个 rank 直接 backward `local_loss_sum / global_N`，reducer 又除一次 \(D\)，结果会少 \(D\) 倍。

一种等价写法是：每个 rank 在反向传播前，把本地 loss 总和乘以 \(D/N\)。也可以选择明确产生全局总和的 reducer。
先写清目标公式，再核对当前框架究竟求总和还是平均值。

进行梯度累积时，窗口内先累加 loss sums，直到窗口末尾才统一归一化、unscale、clip 并执行参数更新。
AMP、`no_sync`、梯度裁剪和 scheduler 的先后顺序也要写入训练契约。

## 先用一张表选择并行维度

| 并行方式 | 切分对象 | 主要通信 | 首先解决什么 |
|---|---|---|---|
| Data Parallel（DP） | 样本 | Gradient all-reduce | 增加数据吞吐 |
| ZeRO/FSDP | 参数、梯度、optimizer state | Gather / reduce-scatter | 持久训练状态放不下 |
| Tensor Parallel（TP） | 单层矩阵与特征维 | 层内 all-reduce/all-gather | 单层放不下或算不动 |
| Pipeline Parallel（PP） | 连续层 | Stage 间 activation/gradient | 整个模型层数放不下 |
| Context Parallel（CP） | 长序列位置 | K/V 或 attention 统计量 | 单卡序列 activation 过大 |
| Expert Parallel（EP） | MoE experts | Token all-to-all | 专家参数与计算分布到多卡 |

实际系统常常组合多种并行。选择顺序应从“哪一项资源先超限、哪条链路最昂贵”开始，而不是先决定缩写。

## 第一层扩展：Data Parallel

Data Parallel（DP）让每个 rank 保存完整模型并处理不同样本，反向传播后再用 collective 聚合梯度。
只要初始参数、全局 batch 和 loss 归一化相同，一次更新就应与单设备大 batch 在约定浮点容差内一致。

它最容易理解，也最先遇到两个上限：

- 每张卡仍保存完整 weights、gradients 和 optimizer state；
- 每步需要通信与参数规模相当的 gradients，最慢 rank 决定同步时间。

Sampler 要保证一个全局 step 中的样本 ID 不重不漏。Gradient bucket 可以让已经完成反向的层提前通信：
bucket 太大，通信启动得晚；bucket 太小，启动开销又会增加。

`no_sync` 只减少累积窗口中的中间梯度同步。对于 DDP，forward 和 backward 都应位于这个上下文中，因为 reducer
常在 forward 阶段准备同步状态。FSDP 与 ZeRO 的协议不同，迁移时要重新核对。

## 第二层扩展：ZeRO 与 FSDP 分片持久状态

经典 ZeRO 术语把分片逐步加深：

| Stage | 在 data-parallel group 中分片什么 |
|---|---|
| 1 | Optimizer state |
| 2 | Optimizer state + gradients |
| 3 | Optimizer state + gradients + parameters |

Fully Sharded Data Parallel（FSDP）和 ZeRO-3 的目标相近，但参数包裹粒度、预取、重分片、混合精度、状态字典
和通信调度并不完全相同。Stage 名称不能代替目标版本的实现文档和实测。

若分片组大小为 \(D\)，持久状态理想上可以接近原来的 \(1/D\)。峰值不会严格除以 \(D\)，因为计算当前 layer 时还可能出现：

```text
gathered full parameters
communication buffers
prefetched next unit
saved activations
checkpoint staging
allocator fragmentation
```

Wrap 太粗会一次 gather 大块参数；太细又会产生大量小 collective。通常先按 Transformer block 等重复结构切分，
再根据显存时间线和通信 trace 调整，不能只看 `sharding_strategy` 配置是否生效。

## 第三层扩展：Tensor Parallel 切一层矩阵

当单层权重或计算也无法放入一张卡，Tensor Parallel（TP）把矩阵乘法分给多个设备。
以 \(Y=XW\) 为例。

### Column parallel

按输出列切 \(W=[W_1,\ldots,W_p]\)：

\[
Y_i=XW_i,
\qquad
Y=\operatorname{concat}(Y_1,\ldots,Y_p).
\]

每个 rank 产生一部分输出特征。若下一层能继续保持分片，可以推迟 gather。

### Row parallel

按输入维切 \(W\)，同时切 \(X=[X_1,\ldots,X_p]\)：

\[
Y=\sum_i X_iW_i.
\]

各 rank 的部分结果需要 reduce-sum。Transformer 实现常把 column 与 row parallel 成对安排，减少中间 gather。

TP 几乎每层都有对延迟敏感的 collective，因此通常放在 NVLink、NVSwitch 或节点内最快的互联域。
TP degree 过大时，每个 rank 的 GEMM 变小，通信延迟反而可能压过计算收益。

Embedding、词表和 LM head 也可以分片。词表并行的交叉熵需要联合计算全局 log-sum-exp，并找到目标 token 所在分片。
若先 gather 全词表 logits，就没有真正省下这部分峰值内存。

## 第四层扩展：Pipeline Parallel 切连续层

Pipeline Parallel（PP）把连续层分给 \(p\) 个 stages，小批次在 stage 之间传递 activation 和 gradient。

对简化、均衡、无 interleaving 的 pipeline，bubble fraction 可用下面的近似建立直觉：

\[
\frac{p-1}{m+p-1},
\]

其中 \(m\) 是小批次数。增加小批次可以摊薄流水线填充与排空成本，增加 stage 则会加重 bubble。
这个式子只提供直觉，不是所有 1F1B 和 interleaved schedule 的精确模型。

- GPipe 风格先执行多组 forward 再 backward，调度直观但保存更多 activations；
- 1F1B 进入稳态后交替 forward/backward，通常降低峰值；
- Interleaving 让设备持有多个 virtual stages，减少 bubble 但增加通信和调度复杂度。

按层数平均切分不等于按时间均衡。Embedding、LM head、不同 block 和跨 stage 通信都可能成为瓶颈。
最终应根据每个 stage 的 forward/backward 时间和空闲时间调整划分。

## 长上下文还会引入 sequence / context parallel

“沿序列切”可能指不同机制：

- Sequence parallel 常分片 LayerNorm、dropout 或 TP 区域外的 activation；
- Context parallel 把超长 attention 的 tokens 分给多个设备，并交换 K/V 或统计量；
- Ring / blockwise attention 通过分块通信和 online softmax 组合全局 attention。

如果只切分输入 tensor，却不取得其他 rank 的 K/V，全局 attention 就会悄悄变成局部 attention。

设计文档应回答三个问题：结果是否与全局 attention 等价；mask、position 和样本边界怎样跨 rank 对齐；
通信量怎样随序列长度 \(T\) 增长。

## MoE 再增加 expert parallel

Mixture-of-Experts（MoE）用路由器为每个 token 选择 top-k 专家。Expert Parallel（EP）把专家放到不同设备；
token 通过 all-to-all 到达负责它的设备，完成计算后再恢复原始顺序。

资源与性能报告至少包含：

```text
total parameters / active parameters per token
expert count / top-k / shared experts
capacity, overflow and drop policy
per-expert token counts and max/mean load
dispatch + combine all-to-all time
router / auxiliary loss and precision
```

平均负载看起来平衡时，一个特别慢的 expert 仍可能让整个同步 step 等待。Batch 太小还会让每个 expert GEMM
失去效率。EP 因此对 all-to-all、元数据顺序和物理拓扑都很敏感。

仓库提供四个逐层推进的 CPU/Gloo 小实验：全局 capacity 竞争、只在负责设备上执行的 all-to-all 分发、
反向传播时的逆向 split，以及 capacity drop 与训练图的组合。

输入张量由仓库准备，所有进程都运行在同一台机器上。NCCL、多节点、目标 MoE checkpoint、GPU 吞吐
和可扩展 runtime 仍需另行验证。精确 split、gradient 与固定数值见[准确性台账](../evidence/accuracy-ledger.md)。

## 组合并行要映射到物理拓扑

设备总数常见关系是：

\[
N_{devices}=D_{data}D_{tensor}D_{pipeline}D_{expert}.
\]

Sequence/context parallel 有时与 TP 共享 group，有时另建 group，不能机械继续相乘。

一般映射原则是：

1. TP 的高频 collective 放在最低 latency、最高 bandwidth 域；
2. EP all-to-all 控制跨节点范围并观察双向带宽和 tail；
3. PP 传 stage activation，可以跨较慢边界，但消息大小仍要实测；
4. DP gradient reduce 常扩到更多节点，并通过 buckets 和 overlap 摊销；
5. Rank mapping、NUMA、NIC 和物理 topology 写入 run manifest。

“8 卡节点”不代表任意两张 GPU 之间的带宽相同。PCIe switch、NVLink/NVSwitch、网卡亲和性、GPU Direct
和网络超卖都会改变实际路径。

一次传输 \(n\) bytes 的粗略延迟模型是：

\[
T\approx\alpha+\frac{n}{\beta},
\]

其中 \(\alpha\) 是启动 latency，\(\beta\) 是有效 bandwidth。小消息常受 latency 主导，大消息受 bandwidth 主导。

理想 ring all-reduce 每 rank 的总传输量近似：

\[
2\frac{p-1}{p}n.
\]

真实实现还受协议、拓扑、竞争和 compute overlap 影响。异步 API 返回只表示操作已排队，
不表示通信已经被计算隐藏；要从 device trace 观察依赖和实际 overlap。

需要继续手算结果布局、ring 轮次、ready time、分层跨域字节和割集时，进入
[集合通信与数据中心网络](collective-communication-network.md)。该页的算法账本用于建立检查方法，
目标 NCCL/runtime 的带宽和 tail 仍由本节要求的 run manifest 与 trace 验收。

## 用计算换显存时，账本要同时记时间

Activation checkpointing 只保存选定边界，并在 backward 时重新执行部分 forward。它用额外计算和训练时间换取
较少的 activation 存储。Selective recomputation 可以优先选择占内存大、重算相对便宜的区域。

FlashAttention 减少 HBM traffic 和 materialized score，与 activation checkpointing 是两种不同优化。
它们可以组合，收益不会简单相加。

CPU 或 NVMe offload 用主机内存与 IO 换 GPU 容量，可能把计算受限的训练变成 PCIe 或 IO 受限。
报告 GPU 峰值下降时，也要给出传输时间、主机内存和每步耗时的变化。

## 用一个 512 卡例子核对维度

假设 TP=8、PP=4、DP=16、micro-batch=2、accumulation=8：

\[
N_{devices}=8\times4\times16=512,
\]

\[
B_{global}=2\times8\times16=256.
\]

TP 和 PP 共同处理同一批样本，所以全局 batch 不是 2048 或 8192。若每个样本平均有 1800 个有效目标 token，
每次参数更新约消费 460,800 个 token。只知道最大长度为 2048，无法推出真实的 token/update。

方案还要验证 stage 是否均衡、TP group 是否位于高速互联域、DP collective 跨越多少节点，
以及 activation/gathered-parameter 峰值。Checkpoint 间隔还要结合大规模运行的实际故障率选择。

## 把并行方案交给正确性验收

确定 DP、FSDP/ZeRO、TP、PP、CP 或 EP 的组合后，进入
[分布式训练正确性](distributed-training-correctness.md)，固定 dtype、global batch、loss、数据游标和
checkpoint 协议，再比较单步数值、故障恢复与长运行。性能报告只有在这些口径一致后才可比较。

## 自测与面试追问

1. 为什么各 rank 的 local mean 在有效 token 数不同时会改变 global objective？
2. ZeRO-3 理想分片哪些状态，为什么真实峰值不会严格除以 DP degree？
3. Column 与 row tensor parallel 分别需要怎样组合或 reduce？
4. PP、context parallel 与 expert parallel 各增加了哪类通信或尾部等待？
5. 给定 TP、PP、DP、micro-batch 与 accumulation，计算设备总数和 global batch。

## 一手资料

- Rajbhandari 等，[ZeRO](https://arxiv.org/abs/1910.02054)。
- Shoeybi 等，[Megatron-LM](https://arxiv.org/abs/1909.08053)。
- Huang 等，[GPipe](https://arxiv.org/abs/1811.06965)。
- PyTorch，[FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html)。
- DeepSpeed、Megatron Core、JAX sharding 与目标 communication backend 的固定版本官方文档。
