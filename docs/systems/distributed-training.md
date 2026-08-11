# 高效与分布式训练

## 学习目标与证据边界

读完本章应能建立参数、梯度、optimizer state、activation 与通信的资源账本；解释 DP、ZeRO/FSDP、TP、PP、sequence/context parallel 与 expert parallel；根据节点拓扑组合并行维度，并设计单卡到多卡的正确性与故障恢复验证。

本仓库没有目标多 GPU/多机集群，因此本章给出算法、估算和验收协议，不声称任何拓扑已实测。FSDP、DeepSpeed、Megatron、JAX sharding 与通信库的具体 API/默认值随版本变化，运行命令必须按固定软件栈重新核对。

## 先做资源账本，而不是先选框架

设参数数为 \(P\)，单个元素字节数为 \(b\)。仅权重存储约 \(Pb\)，训练还可能包含：

```text
forward weights
master weights
gradients
optimizer first/second moments
saved activations
temporary workspaces / communication buckets
allocator fragmentation / runtime context
```

一个常见但非通用的 mixed-precision Adam 示例：BF16 权重 2B、FP32 master weight 4B、FP32 gradient 4B、两份 FP32 moment 8B，共约 **18 bytes/parameter**，尚未包含 activation 和临时 buffer。有些实现没有 master weight、梯度用 BF16、moment 被量化或分片，因此不能把 18B/P 当作固定常数。

### Activation 为什么难用每参数字节表示

Activation 取决于 micro-batch \(B\)、sequence length \(T\)、hidden size \(H\)、layers \(L\)、attention/MLP 实现、保存边界和 dtype。朴素 attention 还可能物化 \(B\times heads\times T\times T\) score/probability；FlashAttention 类算法避免保存完整矩阵，但不消除 Q/K/V、MLP 与残差相关 activation。

OOM 诊断要分项测峰值，不能只用 parameter count。allocator reserved、已分配张量和设备驱动显示值也不是同一指标。

## Global batch 与 loss 归一化

设每个 data-parallel rank 的 micro-batch 为 \(B_\mu\)，梯度累积次数 \(A\)，数据并行度 \(D\)：

\[
B_{global}=B_\mu A D
\]

TP、PP、EP 共同处理同一批样本，不乘入 global batch。若长度/padding 可变，每更新有效 token 为：

\[
N_{tokens}=\sum_{rank,microbatch,t}m_{rank,microbatch,t}
\]

不能用 \(B_{global}\times T_{max}\) 替代。

### Mean 还是 sum

若每个 rank 先对本地有效 token 求 mean，再简单平均 rank，当各 rank 有效 token 数不同，会让短/高 padding rank 权重过大。严格 global token mean 应 all-reduce loss numerator 和有效 token denominator，或对梯度使用等价缩放。

梯度累积也要明确每个 micro-batch loss 是否除以 \(A\)。框架可能在 backward、optimizer 或 distributed wrapper 不同位置处理缩放；用一个可手算 batch 对照单卡结果最可靠。

## 数据并行（DP）

每个 rank 持有完整参数，处理不同数据，反向后 collective 聚合梯度。若所有 rank 从相同参数开始，使用相同 global batch 和数学归一化，更新应与单设备大 batch 在浮点容差内一致。

主要成本：

- 每卡完整权重、梯度和 optimizer state；
- 每步通信量随梯度/参数规模增长；
- 最慢 rank 决定同步 step 时间；
- sampler 必须保证 shard 不重不漏。

Gradient bucketing 用较早完成的梯度与反向计算 overlap。bucket 太大延迟 collective 启动，太小增加 latency/launch overhead。`find_unused_parameters` 等图遍历功能也会增加开销，且可能掩盖错误冻结。

## ZeRO 与 FSDP：沿数据并行组分片状态

按经典术语：

- Stage 1：分片 optimizer state；
- Stage 2：再分片 gradients；
- Stage 3：再分片 parameters，并在计算前后按需 all-gather/reshard。

Fully Sharded Data Parallel 与 ZeRO-3 有相近目标，但对象包装、prefetch、reshard、mixed precision、state dict 和通信调度不等同，不能只按 stage 名称推断行为。

### 理想内存与实际峰值

若分片组大小为 \(D\)，理想持久状态可近似缩小到 \(1/D\)。实际峰值还含当前 layer/unit 的 gathered full parameters、通信 buffer、未及时释放的 activation、prefetch 下一单元和 checkpoint staging。

Wrap 粒度过大时一次 gather 很多参数导致峰值；过小时 collective 太多。要按 Transformer block 等重复结构设计，并用 memory timeline 验证而不是只看配置。

### `no_sync`/累积陷阱

某些分片策略在 gradient accumulation 时仍需保留或聚合特定 state；错误使用 no-sync 可能大幅增加内存或产生不正确梯度。必须按目标框架版本验证，不把普通 DDP 经验直接套到 FSDP/ZeRO。

## Tensor Parallel（TP）

TP 把单层大矩阵拆到多个设备，常用于单卡放不下层参数或希望扩大 GEMM。以 \(Y=XW\) 为例：

### Column parallel

按输出列分 \(W=[W_1,\ldots,W_p]\)：

\[
Y_i=XW_i,\quad Y=\operatorname{concat}(Y_1,\ldots,Y_p)
\]

每 rank 计算部分输出特征；后续算子若能继续分片，可推迟 gather。

### Row parallel

按输入行分 \(W=[W_1^T,\ldots,W_p^T]^T\)，同时分 \(X=[X_1,\ldots,X_p]\)：

\[
Y=\sum_i X_iW_i
\]

需要 reduce-sum 部分结果。实际 Transformer 常把 column/row parallel 成对放置，减少中间 gather。

TP 每层有高频 collective，通常优先映射到 NVLink/NVSwitch 或其他节点内高速互联。TP 过大时单 rank GEMM 变小、通信 latency 上升，吞吐可能下降。

Embedding 与 vocab/lm head 也可分片，但 cross entropy 需要处理全局 log-sum-exp 和 target shard；不能先 gather 巨大 vocab logits 再声称 vocab parallel 节省内存。

## Pipeline Parallel（PP）

PP 把连续层分给 \(p\) 个 stage，把 global batch 切成 \(m\) 个 micro-batches，在 stage 间传 activation/gradient。

对简化、均衡、无交错 pipeline，fill/drain bubble 随 stage 增多而上升，随 micro-batch 数增多而下降。常见近似 bubble fraction：

\[
\frac{p-1}{m+p-1}
\]

这不是所有 1F1B/interleaved schedule 的精确公式，只用于直觉。真实效率还受 forward/backward 时间、通信、stage 不均、activation 内存和 optimizer step 同步影响。

### 调度与内存

- GPipe 风格先完成多组 forward 再 backward，bubble 清晰但保存 activation 多；
- 1F1B 进入稳态后交替 forward/backward，降低峰值；
- interleaving 让一个设备持多个虚拟 stage，可减 bubble 但通信和调度更复杂。

Layer 数不能整除 stage 或某些 embedding/head 特别重时，需要显式 balance。按层数均分不保证按 FLOPs、activation 或通信均衡。

## Sequence 与 Context Parallel

“沿序列切”包含不同技术：

- sequence parallel：常把 layer norm、dropout 或 TP 区域外的 activation 沿 sequence 分片，减少复制；
- context parallel：把超长 attention 的 token/context 分给多个设备，并交换 K/V 或部分统计；
- ring/blockwise attention：以分块通信和 online softmax 组合全局注意力。

必须说明 attention 是否全局精确、局部/稀疏近似，mask 与 position 如何跨 rank 保持，以及通信量随 \(T\) 怎样变化。仅把 input tensor 切片而不交换远端 K/V，会改变模型函数。

长序列还要验证 packed sample boundary，避免不同 rank 的 mask/position 错位。

## Expert Parallel（EP）

MoE 将专家分布到设备。Router 为 token 选择 top-k expert，token 经 all-to-all 发往专家，再返回原顺序。

资源账本要同时报告：

- total parameters 与 active parameters/token；
- experts、top-k、shared experts；
- capacity/overflow/drop policy；
- 每 expert token 数和最大/平均负载；
- dispatch/combine all-to-all 时间；
- router/auxiliary loss 与数值精度。

平均负载均衡不代表没有 tail：一个过载 expert 会让整个同步 step 等待。小 batch 时每 expert GEMM 太小也可能低效。EP 通常对网络 all-to-all 特别敏感。

本仓库 `route_topk_capacity` 只在单进程 CPU 上物化 expert ids、capacity keep/drop 与 combine weights，并执行 bias-free linear experts。它可验证 padding、tie-break、assignment count 和 drop 后归一化，不能验证 token packing、rank-to-expert placement、all-to-all bytes/order、grouped GEMM、反向传播或同步 straggler。把该 toy 的 expert count 当通信量或 GPU throughput 属于证据越界。

## 组合并行与拓扑映射

设备总数常满足：

\[
N_{devices}=D_{data}\times D_{tensor}\times D_{pipeline}\times D_{expert}
\]

具体实现可能让 sequence/context parallel 与 TP 共享或另建 group，不能机械再乘。

映射原则：

1. 高频、小延迟敏感 collective（TP）放在最快互联域；
2. EP all-to-all 需要高双向带宽并控制跨节点范围；
3. PP 主要传 stage activation，可跨较慢边界但要看消息大小；
4. DP gradient reduce 可用 bucket/overlap 摊销，常扩到更多节点；
5. rank mapping 与物理 topology 写入 run manifest。

“8 卡节点”不代表所有 pair 带宽相同；PCIe switch、NUMA、NIC 亲和、GPU Direct 与 oversubscription 都会影响。

## 通信成本的简单模型

传输 \(n\) bytes 的一次消息可粗略写成：

\[
T\approx\alpha+\frac{n}{\beta}
\]

\(\alpha\) 是 latency，\(\beta\) 是有效 bandwidth。小消息被 latency 主导，大消息被 bandwidth 主导。

理想 ring all-reduce 对每 rank 的总传输量近似：

\[
2\frac{p-1}{p}n
\]

并经历多轮 reduce-scatter/all-gather。实际还有协议、拓扑、竞争和 overlap；这个公式不能代替 NCCL/目标网络 benchmark。

### Compute/communication overlap

总 step time 不是简单相加，若依赖允许，可在 backward 计算后续层时 reduce 已完成层的 gradient，或 prefetch 下一 FSDP unit。应从 trace 观察真正 overlap；异步 API 返回不代表设备通信已隐藏。

## Activation checkpoint 与内存换计算

Activation checkpoint/recomputation 只保存边界，backward 重做 forward。它降低 activation memory，增加 FLOPs 和 wall time。选择性重计算可优先处理内存大、重算便宜的算子；attention/MLP 策略依 kernel。

FlashAttention 减少 HBM 读写与 materialized score，不改变其支持范围内的 attention 数学目标，但浮点顺序会有差异。它与 activation checkpoint 是不同优化，可以组合，收益不能简单相加。

CPU offload/NVMe offload 用容量换 PCIe/IO，可能让系统从 compute-bound 变成 IO-bound。报告应包含传输时间和 host 内存，而不是只报 GPU 峰值下降。

## 混合精度与 Distributed Reduction

- FP32：精度/范围高，矩阵吞吐和存储成本大；
- TF32：部分 GPU 的 FP32 matmul 加速路径，不是参数存储 dtype；
- FP16：指数范围小，训练常需要 dynamic loss scaling；
- BF16：指数范围接近 FP32，尾数较少；
- FP8：依赖硬件、格式、scale/amax history 和 kernel 支持。

需要分别声明 parameter、forward compute、gradient、collective、master weight、optimizer state 和 loss dtype。某些框架在低精度 all-reduce 前预缩放；顺序错误可能 overflow/underflow。

Loss scaler 的 overflow 决策应在相关 ranks 一致，否则部分 rank step、部分 rank skip 会让参数分叉。

## Checkpoint：分片不是完成标志

分布式 checkpoint 要解决：

- 每个 rank/shard 是否全部写完；
- manifest 是否原子发布；
- 参数、optimizer、RNG、scheduler、sampler 是否一致 step；
- world size/topology 改变时能否 reshard；
- 保存期间内存峰值和存储带宽；
- 文件 hash、版本、保留和灾难恢复。

不要让 rank 0 聚合全模型导致 OOM，再把分片训练的内存收益丢掉。需要 full state dict 时，明确在哪里、用多少 host/GPU 内存完成聚合。

### 故障窗口

若部分 rank 在 step \(t\) 保存、另一部分在 \(t+1\)，拼出的 checkpoint 不对应任何真实全局状态。manifest 应包含 global step、consumed token、shard list/hash 和 complete marker；恢复只读取完整 generation。

## 分布式正确性验证

性能调优前按层级建立证据：

### 1. 单步数学等价

固定相同参数和 global batch，关闭 dropout 或控制 RNG，比较：

- logits 与 loss numerator/denominator；
- 关键 gradient 与 global norm；
- optimizer update 后参数；
- TP/PP 边界 tensor shape 与数值。

浮点 collective 顺序可导致微差，应先定义 atol/rtol 和累积步数，不能要求所有硬件 bitwise 相等，也不能无限放宽容差。

### 2. 数据等价

记录每 rank sample/source id，验证一个 global step 不重不漏；gradient accumulation、worker 重启和 resume 后同样检查。Distributed sampler 的 `set_epoch`/等价状态与 shuffle RNG 要保存。

### 3. 随机性

Dropout 通常需要 rank/device 相关但可重放的 RNG；tensor-parallel 区域中某些 mask 可能需要一致，另一些需要独立。不能只给所有 rank 相同 seed，也不能随机到无法恢复。

### 4. 长运行不变量

监控 rank 间 parameter checksum、step/global token、loss scaler、optimizer step 和 collective timeout。一个 rank 静默跳步会在数百步后才表现为 loss 异常。

## 性能指标与可比较报告

至少报告：

```text
model/config and parameter counts
hardware/topology/interconnect
world size and every parallel dimension
precision and kernel/runtime versions
micro/global batch and effective tokens/update
sequence length distribution / packing
activation checkpoint and offload
step time distribution / tokens per second
peak allocated/reserved memory
compute, collective, data and idle breakdown
compile/warmup and measurement window
```

### MFU 的边界

Model FLOPs Utilization 通常是“模型理论 FLOPs × 实际 token throughput / 硬件峰值 FLOPs”。分子 FLOPs 模型和分母峰值 dtype 必须一致。是否包含 activation recomputation、attention、embedding、MoE 路由会改变数值；不同论文/框架的 MFU 不一定可直接比较。

高 GPU utilization 也可能在做无效重算或通信等待；tokens/s 高也可能因为 padding 被算作 token。质量和有效 token 口径必须同时报告。

## 性能诊断顺序

1. **确认正确**：global batch、loss reduction、数据/RNG；
2. **分解 step**：data、forward、backward、collective、optimizer；
3. **看 shape/GEMM**：是否太小、padding 过多、kernel 未融合；
4. **看 overlap**：collective 是否真正与 compute 重叠；
5. **看拓扑**：rank/NIC/NUMA 映射、跨节点流量；
6. **看 straggler**：每 rank 分位数、热降频、错误重试；
7. **再调 bucket/prefetch/wrap**：每次只改一个主要变量。

常见症状：

| 症状 | 优先检查 |
|---|---|
| 所有卡低利用 | data loader、同步点、小 batch、compile |
| 一卡慢、全局慢 | 热/功率、NUMA/NIC、数据异常、expert 负载 |
| 扩卡吞吐不升 | global batch、collective、TP 粒度、网络拓扑 |
| FSDP 峰值仍高 | wrap 粒度、prefetch、gather overlap、checkpoint staging |
| loss 与单卡不同 | mean/sum、有效 token、sampler、RNG、低精度归约 |
| resume 后突变 | optimizer/scaler/RNG/data position、world-size reshard |

## 容量规划示例

假设 TP=8、PP=4、DP=16、micro-batch=2、accumulation=8：

\[
N_{devices}=8\times4\times16=512
\]

\[
B_{global}=2\times8\times16=256
\]

不是 2048 或 8192，因为 TP/PP 没有增加独立样本。若每样本平均 1800 个有效 target token，则每更新约 \(460{,}800\) token；若只知道 max length 2048，还不能得到真实 token/update。

选择方案时还要验证：每 stage 层数/时间是否均衡、TP group 是否在高速域、DP collective 是否跨多少节点、activation 与 gathered parameter 峰值，以及 512 卡故障率下 checkpoint 间隔是否合理。

## 可执行验收路线

本仓库无法在当前单设备环境证明多卡行为，正确路线是：

1. 2 个 process 的 CPU/GPU toy model，检查 global loss/gradient；
2. 单节点多 GPU，逐个启用 DP → sharding/TP；
3. 固定 global batch 做 strong scaling；
4. 固定每卡 workload 做 weak scaling；
5. 注入 rank crash、checkpoint 中断与 data worker 重启；
6. 再扩到多节点，保存 topology 和通信 trace；
7. 每一层都与上一级做数值和数据 lineage 对照。

未获得目标硬件前，文档命令只能是运行计划，不能标成“已验证吞吐/MFU”。

## 常见错误

- 把 18 bytes/parameter 当作所有 Adam 训练固定值；
- global batch 错乘 TP/PP/EP；
- 各 rank 本地 mean 再等权平均，忽略有效 token 不同；
- 只看配置声称 FSDP state 已按 \(1/D\) 缩小；
- TP 跨慢网络、EP all-to-all 不测 tail；
- 用 PP 层数均分代替实测 stage balance；
- 切 sequence 却不交换远端 attention 信息，悄悄改变模型；
- 首次 compile/warmup 混入稳态吞吐；
- checkpoint 缺 shard/step manifest 或只保存模型权重；
- 追求 MFU 而忽略有效 token、质量和重计算口径；
- 在没有目标 GPU/集群时把设计文档写成已验证结果。

## 面试追问

1. 为什么 local mean loss 在可变 token rank 上会给出错误 global weighting？
2. ZeRO-3 理想节省哪些状态，为什么实际峰值不会严格除以 DP？
3. Column/row tensor parallel 各需要什么 collective？
4. PP bubble 与 micro-batch 数、stage 数、stage imbalance 有什么关系？
5. Context parallel 如何保持全局 attention，而不是变成局部 attention？
6. EP 为什么常被 all-to-all 和 tail expert 限制？
7. 如何设计单卡与多卡单步等价测试？
8. world size 改变时，optimizer/RNG/data/checkpoint 如何 reshard 与恢复？
9. MFU 为什么可能在两个系统间不可直接比较？

## 一手资料

- Rajbhandari 等，[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)。
- Shoeybi 等，[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)。
- Huang 等，[GPipe](https://arxiv.org/abs/1811.06965)。
- PyTorch 官方文档，[FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html)。
- DeepSpeed、Megatron Core、JAX sharding 与目标通信库的固定版本官方文档；具体参数/支持矩阵的最高优先级来源。
