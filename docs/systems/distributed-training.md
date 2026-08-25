# 高效与分布式训练：从单卡 OOM 到多卡一致

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备多 GPU 训练、平台建设或分布式正确性验收的工程师。
- **先修**：[预训练](../training/pretraining.md)、显存状态和基本 collective 概念。
- **首次阅读**：先做资源账本和 global loss，再沿 DP、分片、TP、PP 逐步扩展。
- **完成信号**：能解释每种并行切了什么，并设计单卡到多卡的数值验收。
- **卡住时**：暂时忽略框架 API，只画每个 rank 保存哪些 tensor、何时通信。

</div>

一张 24 GB GPU 放不下 7B 模型的全参数 Adam 训练。团队拿到一台 8 卡机器后，打开 FSDP、activation checkpointing
和混合精度，训练终于能够启动，但 loss 与单卡小规模 baseline 不同。

这时问题已经不是“哪个框架参数能省显存”，而是四件事混在了一起：

1. 参数、梯度、optimizer 和 activation 各占多少；
2. 八张卡分别持有什么，通信发生在哪里；
3. Global batch 与 loss denominator 是否仍和单卡相同；
4. Checkpoint、RNG 和 data cursor 能否在所有 rank 上恢复同一训练状态。

本章沿这条排查顺序学习分布式训练。仓库没有目标多 GPU 或多机集群，
因此本章只能在本地检查算法、CPU 多进程通信和验收方法。GPU 吞吐、MFU 与拓扑结论都要在目标环境实测。

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

## 用计算换显存时，账本要同时记时间

Activation checkpointing 只保存选定边界，并在 backward 时重新执行部分 forward。它用额外计算和训练时间换取
较少的 activation 存储。Selective recomputation 可以优先选择占内存大、重算相对便宜的区域。

FlashAttention 减少 HBM traffic 和 materialized score，与 activation checkpointing 是两种不同优化。
它们可以组合，收益不会简单相加。

CPU 或 NVMe offload 用主机内存与 IO 换 GPU 容量，可能把计算受限的训练变成 PCIe 或 IO 受限。
报告 GPU 峰值下降时，也要给出传输时间、主机内存和每步耗时的变化。

## Mixed precision 是一套分布式状态机

参数存储、forward 计算、梯度、collective、master weights、optimizer state 和 loss 可能使用不同数值格式。
实验报告需要逐项声明。

| 格式 | 训练时要注意什么 |
|---|---|
| FP32 | 范围与精度高，存储和矩阵吞吐成本大 |
| TF32 | 部分 GPU 的 FP32 matmul 路径，不是 parameter storage dtype |
| FP16 | 指数范围小，常配 dynamic loss scaling |
| BF16 | 指数范围接近 FP32，尾数较少 |
| FP8 | 依赖硬件、格式、scale / amax history 和 kernel |

AMP 的常见顺序是：

```text
scaled backward
→ gradient synchronization
→ unscale
→ 跨 rank 汇总 non-finite 状态
→ gradient clipping
→ optimizer step
```

具体位置受框架协议影响，但所有 rank 必须共同决定“执行这一步”还是“跳过这一步”。若一个 rank 跳过更新、另一个
rank 执行更新，参数、Adam moments、scheduler、学习率和 scaler state 都会立即分叉。

有限性检查应位于所有可能产生非有限值的梯度变换之后，并且早于任何 optimizer state 修改。

## Checkpoint 要代表一个真实全局时刻

分片文件都存在，不代表 checkpoint 完成。Manifest 至少绑定：

```text
global step / consumed tokens
model + optimizer shards
scheduler + scaler
RNG states
sampler / data cursors
world size / topology / sharding revision
file size / hash
complete marker
```

如果部分 ranks 保存 step \(t\)，另一些保存 \(t+1\)，拼出的状态没有对应真实训练时刻。
先写 payloads，最后原子发布 manifest；恢复只读取 complete generation。

Rank 0 聚合完整模型时可能再次 OOM，抵消分片训练的内存收益。若需要完整 state dict，应明确在哪个设备或主机上
完成聚合、需要多少内存，以及 world size 改变时怎样重新分片。

Data cursor 要区分三个位置：sampler 已发出、主循环已消费，以及 optimizer 已提交。DataLoader prefetch 可能已经
发出尚未消费的样本；梯度累积窗口崩溃时，还可能留下尚未提交的梯度。

仓库的恢复实验展示两种策略：回到已提交 cursor 重放，或者保存待处理样本、梯度与崩溃时的 RNG sidecar 后继续。

在当前 CPU 样例中，两种路径都能精确恢复。故意漏掉梯度或使用错误 RNG 后，参数会产生可观察漂移。
精确 checkpoint schema、fault snapshots 和数值见[单 GPU 微调项目](../practice/projects/single-gpu-finetuning.md)。

这些本地实验没有把多 rank 数据、collective 和 optimizer 变成一个原子事务。分布式实现仍要定义全局 commit receipt、
失败 rank 的共同回滚或重放方式，以及可接受的重复样本策略。

## 正确性验收从单步开始

性能调优前按四层建立证据。

### 1. 单步数学等价

固定相同 parameters 和 global batch，关闭 dropout 或控制 RNG，比较：

- loss numerator、denominator 与 scalar loss；
- 关键 gradients 与 global norm；
- optimizer step 后 parameters；
- TP / PP boundary tensors 的 shape 和数值。

浮点 reduction 顺序不同会产生微差。先按 dtype 和累积步数定义 `atol/rtol`，
既不能要求跨硬件 bitwise 相同，也不能看到差异后无限放宽容差。

### 2. 数据等价

记录每个 rank 的样本与来源 ID，检查一个全局 step 内不重不漏。恢复后重新核对已发出、已消费、已提交三个 cursor
和重放策略，并保存 shuffle RNG 与 sampler epoch。

### 3. 随机性与 step 共识

Dropout RNG 需要感知 rank 和设备，并且可以重放。TP 区域的某些 mask 需要一致，另一些可以独立。
同时监控 optimizer step、scheduler、loss scale 和参数校验值，及时发现某个 rank 静默跳步。

### 4. 长运行与故障恢复

主动注入 rank 崩溃、checkpoint 中断、worker 重启、collective 超时和非有限梯度。
恢复后比较数据账本、训练状态和接下来的若干 step；模型文件能够加载只是最低要求。

## 用 CPU 小实验逐层排查

从单进程到双进程依次运行：

~~~powershell
python projects/single-gpu-finetuning/gradient_accumulation_toy.py
python projects/single-gpu-finetuning/ddp_token_mean_control.py
python projects/single-gpu-finetuning/ddp_accumulation_no_sync_control.py
python projects/single-gpu-finetuning/amp_grad_scaler_control.py
python projects/single-gpu-finetuning/ddp_amp_overflow_consensus_control.py
~~~

| 实验 | 主要观察 |
|---|---|
| Gradient accumulation toy | Token sum/count 与 local-mean weighting 的差异 |
| DDP token mean | 默认 gradient mean 下 \(D/N\) scaling |
| DDP accumulation | `no_sync` scope、同步后 clipping 和一次 SGD update |
| AMP scaler | unscale-before-clip、overflow skip 与 scaler resume |
| DDP + AMP | Reduction 前 non-finite、reduction 后单-rank fault 与共同 skip |

这些实验使用 PyTorch CPU、Gloo、小参数和仓库准备的输入，能够检查公式、collective 路径和故障状态。

FSDP/ZeRO/TP/PP/EP、CUDA、多节点、目标 Trainer、收敛、吞吐和模型质量需要更高一层的实际运行。

## 性能报告要让两个 run 真能比较

至少保存：

```text
model config / parameter counts
hardware / topology / interconnect
world size and every parallel dimension
precision / kernels / runtime revisions
micro/global batch and effective tokens per update
sequence-length distribution / packing
activation checkpoint / offload
step-time distribution / valid tokens per second
peak allocated / reserved memory
compute / collective / data / idle breakdown
compile / warmup / measurement window
```

模型 FLOPs 利用率（Model FLOPs Utilization，MFU）通常用“模型理论 FLOPs × token throughput”除以硬件峰值 FLOPs。

计算理论 FLOPs 时是否包含 attention、embedding、MoE 和 recomputation，以及硬件峰值使用哪个 dtype，都会改变结果。
只有口径相同的 MFU 才能直接比较。

高 GPU utilization 可能来自无效重算，tokens/s 也可能把 padding 算进分子。应同时报告有效监督 token、
loss/quality 和实际工作分解。

性能诊断按这个顺序进行：

1. 确认 global batch、loss、data 和 RNG 正确；
2. 分解 data、forward、backward、collective、optimizer 与 idle；
3. 检查 GEMM shape、padding 和 kernel path；
4. 观察 communication 是否真实 overlap；
5. 核对 rank、NIC 与 NUMA topology；
6. 找每 rank straggler、热降频和 expert tail；
7. 最后调整 bucket、prefetch、wrap 和 schedule。

| 症状 | 优先检查 |
|---|---|
| 所有卡利用率低 | Data loader、同步点、小 GEMM、compile |
| 一卡慢、全局慢 | 温度/功耗、NUMA/NIC、异常数据、expert load |
| 扩卡吞吐不升 | Collective、TP 粒度、网络 topology 与 workload |
| FSDP 峰值仍高 | Wrap、prefetch、gather overlap、checkpoint staging |
| Loss 与单卡不同 | Token denominator、reducer、sampler、RNG、precision |
| Resume 后突变 | Optimizer、scaler、RNG、data cursor 和 reshard |

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

## 从本地走向目标集群

1. 在单进程小例子上写清 loss numerator、denominator 和预期 update；
2. 用两个 CPU/GPU 进程检查 DDP reduction、样本 ID 与失败共识；
3. 在单节点多 GPU 逐个启用 DP、sharding、TP，不同时改变多个维度；
4. 固定全局 workload 做 strong scaling，固定每设备 workload 做 weak scaling；
5. 注入 rank crash、checkpoint 中断、worker restart 和 non-finite gradient；
6. 扩到多节点后保存 topology、collective trace 和每 rank timeline；
7. 每一级都与上一级做数值、data lineage 和 checkpoint 对照。

未获得目标硬件前，这些命令和公式属于运行计划，不能写成已验证 GPU throughput 或 MFU。

## 自测与面试追问

1. 为什么各 rank 的 local mean 在有效 token 数不同时会改变 global objective？
2. ZeRO-3 理想分片哪些状态，为什么真实峰值不会严格除以 DP degree？
3. Column 与 row tensor parallel 分别需要怎样组合或 reduce？
4. PP bubble 怎样随 stages、micro-batches 和 stage imbalance 变化？
5. Context parallel 如何保持 global attention，而不是悄悄变成 local attention？
6. EP 为什么容易受 all-to-all 和 tail expert 限制？
7. AMP 中一个 rank step、另一个 rank skip 后，哪些训练状态会分叉？
8. World size 改变时，optimizer、RNG、data 和 checkpoint 怎样 reshard 与恢复？
9. 两份报告中的 MFU 为什么可能无法直接比较？

## 一手资料

- Rajbhandari 等，[ZeRO](https://arxiv.org/abs/1910.02054)。
- Shoeybi 等，[Megatron-LM](https://arxiv.org/abs/1909.08053)。
- Huang 等，[GPipe](https://arxiv.org/abs/1811.06965)。
- PyTorch，[FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html)。
- DeepSpeed、Megatron Core、JAX sharding 与目标 communication backend 的固定版本官方文档。
