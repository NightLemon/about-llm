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

一张 24 GB GPU 放不下 7B 模型的全参数 Adam 训练。团队拿到一台 8 卡机器后，
第一反应是打开 FSDP、gradient checkpointing 和 mixed precision。训练终于启动，loss 却和单卡小规模 baseline 不同。

这时问题已经不是“哪个框架参数能省显存”，而是四件事混在了一起：

1. 参数、梯度、optimizer 和 activation 各占多少；
2. 八张卡分别持有什么，通信发生在哪里；
3. Global batch 与 loss denominator 是否仍和单卡相同；
4. Checkpoint、RNG 和 data cursor 能否在所有 rank 上恢复同一训练状态。

本章沿这条排查顺序学习分布式训练。仓库没有目标多 GPU 或多机集群，
因此本地证据限于算法、CPU 多进程 controls 和验收协议；任何 GPU 吞吐、MFU 与拓扑结论都要在目标环境实测。

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

一个常见 mixed-precision Adam 配置可能包含 BF16 weights 2 B、FP32 master weights 4 B、
FP32 gradients 4 B 和两份 FP32 moments 8 B，总计约 18 bytes/parameter，尚未包含 activation 与临时 buffer。

18 B/P 只是某一配置的账本。实现可能没有 master weights，gradient 可以是 BF16，moments 也可能量化或分片。
真实报告应列出每项 dtype、复制数和分片组，不能把一个经验常数当成框架规范。

### Activation 为什么单独列

Activation 受 micro-batch \(B\)、sequence length \(T\)、hidden size \(H\)、layers \(L\)、
attention/MLP kernel、保存边界和 dtype 共同影响。朴素 attention 还可能物化
\(B\times heads\times T\times T\) 的 score 或 probability。

FlashAttention 类算法避免保存完整 attention matrix，但仍要保存或重算 Q/K/V、MLP 和残差相关状态。
OOM 排查要同时观察 allocated、reserved、runtime/driver 读数和 memory timeline；这些不是同一个指标。

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

如果第 \(i\) 个 shard 的 loss sum 是 \(S_i\)，有效 token 数是 \(n_i\)，global token mean 是：

\[
L=\frac{\sum_i S_i}{N},
\qquad
N=\sum_i n_i.
\]

先计算各 shard 的 \(S_i/n_i\) 再等权平均，会让短 batch 或高 padding rank 权重过大。
稳妥做法是聚合 numerator 与 count，或使用数学等价的 gradient scaling。

### DDP 默认 gradient mean 为什么多一个 world size

假设 DistributedDataParallel（DDP）对 \(D\) 个 rank 的 gradient 求平均。
若每个 rank 直接 backward `local_loss_sum / global_N`，reducer 又除一次 \(D\)，结果会少 \(D\) 倍。

一种等价写法是每个 rank 对 local loss sum 乘 \(D/N\) 后 backward；
也可以使用明确得到 global sum 的 reducer 路径。关键是先写清目标公式，再核对框架当前 reducer 语义。

在 gradient accumulation 中，窗口内先累计 sums，直到末尾才归一化、unscale、clip 和 optimizer step。
AMP 的 scale/unscale、`no_sync`、gradient clipping 与 scheduler step 顺序也要进入训练契约。

## 第一层扩展：Data Parallel

Data Parallel（DP）让每个 rank 保存完整模型，处理不同数据，反向后 collective 聚合 gradients。
在相同初始参数、global batch 和数学归一化下，一次更新应与单设备大 batch 在约定浮点容差内一致。

它最容易理解，也最先遇到两个上限：

- 每张卡仍保存完整 weights、gradients 和 optimizer state；
- 每步需要通信与参数规模相当的 gradients，最慢 rank 决定同步时间。

Sampler 要保证一个 global step 的 sample IDs 不重不漏。Gradient buckets 可以让已经完成反向的层提前通信；
bucket 太大启动晚，太小则增加 latency 和 launch overhead。

`no_sync` 只减少累积窗口中的中间 gradient synchronization。对于 DDP，forward 和 backward 都应处于目标 context，
因为 reducer 常在 forward 阶段准备同步状态。这个经验不能未经验证地照搬到 FSDP 或 ZeRO。

## 第二层扩展：ZeRO 与 FSDP 分片持久状态

经典 ZeRO 术语把分片逐步加深：

| Stage | 在 data-parallel group 中分片什么 |
|---|---|
| 1 | Optimizer state |
| 2 | Optimizer state + gradients |
| 3 | Optimizer state + gradients + parameters |

Fully Sharded Data Parallel（FSDP）和 ZeRO-3 目标相近，但 wrapping、prefetch、reshard、
mixed precision、state dict 和通信调度并不完全相同。Stage 名称不能代替目标版本的实现文档和实测。

若分片组大小为 \(D\)，持久状态理想上可以接近原来的 \(1/D\)。峰值不会严格除以 \(D\)，因为计算当前 layer 时还可能出现：

```text
gathered full parameters
communication buffers
prefetched next unit
saved activations
checkpoint staging
allocator fragmentation
```

Wrap 太粗会一次 gather 大块参数，太细则产生大量小 collective。通常从 Transformer block 等重复结构起步，
再用 memory timeline 与通信 trace 调整，而不是只看 `sharding_strategy` 配置。

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

TP 在几乎每层都有 latency-sensitive collective，通常放在 NVLink、NVSwitch 或节点内最快互联域。
TP degree 太大时，每个 rank 的 GEMM 变小，通信 latency 反而会压过计算收益。

Embedding、vocabulary 和 LM head 也可以分片。Vocab-parallel cross entropy 要联合计算 global log-sum-exp
和 target shard，不能先 gather 全 vocab logits 后再声称节省了这部分内存。

## 第四层扩展：Pipeline Parallel 切连续层

Pipeline Parallel（PP）把连续 layers 分给 \(p\) 个 stages，micro-batches 在 stage 间传 activation 和 gradient。

对简化、均衡、无 interleaving 的 pipeline，bubble fraction 可用下面的近似建立直觉：

\[
\frac{p-1}{m+p-1},
\]

其中 \(m\) 是 micro-batches 数。Micro-batches 增多会摊薄 fill/drain，stage 增多则加重 bubble。
它不是所有 1F1B 和 interleaved schedules 的精确公式。

- GPipe 风格先执行多组 forward 再 backward，调度直观但保存更多 activations；
- 1F1B 进入稳态后交替 forward/backward，通常降低峰值；
- Interleaving 让设备持有多个 virtual stages，减少 bubble 但增加通信和调度复杂度。

按层数平均切分不等于按时间均衡。Embedding、LM head、不同 block 和跨 stage 通信都可能形成瓶颈，
最终用每 stage forward/backward time 与 idle time 调整划分。

## 长上下文还会引入 sequence / context parallel

“沿序列切”可能指不同机制：

- Sequence parallel 常分片 LayerNorm、dropout 或 TP 区域外的 activation；
- Context parallel 把超长 attention 的 tokens 分给多个设备，并交换 K/V 或统计量；
- Ring / blockwise attention 通过分块通信和 online softmax 组合全局 attention。

只切 input tensor 而不取得远端 K/V，会把 global attention 变成局部函数。
设计必须说明 attention 是否 exact，mask、position 和 packed sample boundary 怎样跨 rank 对齐，
以及通信怎样随 \(T\) 增长。

## MoE 再增加 expert parallel

Mixture-of-Experts（MoE）用 router 为每个 token 选择 top-k experts。
Expert Parallel（EP）把 experts 放到不同设备，token 经 all-to-all 到 owner，计算后再回到 source order。

资源与性能报告至少包含：

```text
total parameters / active parameters per token
expert count / top-k / shared experts
capacity, overflow and drop policy
per-expert token counts and max/mean load
dispatch + combine all-to-all time
router / auxiliary loss and precision
```

平均负载看起来平衡时，一个 tail expert 仍可能让同步 step 等待。Small batch 还会让每个 expert GEMM 太小。
EP 对 all-to-all、metadata order 和拓扑非常敏感。

仓库提供四层 CPU/Gloo controls，分别检查 global capacity competition、owner-only all-to-all dispatch、
reverse-split backward，以及 capacity drop 与训练图组合。它们使用 authored tensors 和本机 processes，
没有验证 NCCL、多节点、target MoE checkpoint、GPU throughput 或可扩展 runtime。
精确 split、gradient 与 fixed values 见[准确性台账](../evidence/accuracy-ledger.md)。

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

“8 卡节点”不代表任意 GPU pair 带宽相同。PCIe switch、NVLink/NVSwitch、NIC affinity、
GPU Direct 和 oversubscription 都会改变路径。

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

Activation checkpointing 只保存边界，backward 时重做 forward。它减少 saved activations，
同时增加 FLOPs 和 wall time。Selective recomputation 可以优先选择占内存大、重算相对便宜的区域。

FlashAttention 减少 HBM traffic 和 materialized score，与 activation checkpointing 是两种不同优化。
它们可以组合，收益不会简单相加。

CPU 或 NVMe offload 用 host/IO capacity 换 GPU capacity，可能把 compute-bound 训练变成 PCIe 或 IO-bound。
报告 GPU 峰值下降时，也要给传输时间、host memory 与 step-time 变化。

## Mixed precision 是一套分布式状态机

需要分别声明 parameter storage、forward compute、gradient、collective、master weights、optimizer state 和 loss 的 dtype。

| 格式 | 训练时要注意什么 |
|---|---|
| FP32 | 范围与精度高，存储和矩阵吞吐成本大 |
| TF32 | 部分 GPU 的 FP32 matmul 路径，不是 parameter storage dtype |
| FP16 | 指数范围小，常配 dynamic loss scaling |
| BF16 | 指数范围接近 FP32，尾数较少 |
| FP8 | 依赖硬件、格式、scale / amax history 和 kernel |

AMP 的常见顺序是 scaled backward → gradient synchronization → unscale → non-finite consensus → clip → optimizer step。
具体位置受框架协议影响，但所有 rank 必须对“这一步执行还是 skip”形成一致决定。

若一个 rank skip、另一个 rank step，parameters、moments、scheduler、LR 和 scaler state 都会分叉。
Finite check 必须放在所有可能产生 non-finite 的 gradient transforms 之后、任何 optimizer mutation 之前。

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

Rank 0 聚合 full model 可能再次 OOM，抵消分片训练的内存收益。若需要 full state dict，
明确在哪个设备或 host、用多少内存完成 gather，以及 world size 改变时怎样 reshard。

Data cursor 还要区分 sampler-emitted、main-loop-consumed 与 optimizer-committed。
DataLoader prefetch 可能已经发出尚未消费的 IDs；gradient accumulation 崩溃时还可能存在未提交 gradients。

仓库的 resume controls 展示两条策略：回到 committed cursor 重放，或保存 pending samples、gradients 与 crash RNG sidecar 后继续。
两条都能在当前 CPU fixture 中 bit-exact 恢复；漏 gradients 或使用错误 RNG 的隔离负例会漂移。
精确 checkpoint schema、fault snapshots 和数值见[单 GPU 微调项目](../practice/projects/single-gpu-finetuning.md)。

这些 controls 没有让多 rank data、collective 和 optimizer 成为一个原子事务。
分布式实现仍要定义 global commit receipt、失败 rank 的共同回滚/重放和可接受的重复样本策略。

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

记录每 rank 的 sample/source IDs，检查一个 global step 不重不漏。
Resume 后重新核对 emitted、consumed、committed 与 replay policy，保存 shuffle RNG 和 sampler epoch。

### 3. 随机性与 step 共识

Dropout RNG 需要 rank/device-aware 且可重放。TP 区域的某些 mask 需要一致，另一些可以独立。
同时监控 optimizer step、scheduler、loss scale 和参数 checksum，发现一个 rank 静默跳步。

### 4. 长运行与故障恢复

注入 rank crash、checkpoint 中断、worker 重启、collective timeout 和非有限 gradient。
恢复后比较 data ledger、训练状态与下一个若干 step，而不只看模型文件能否加载。

## 仓库中的 CPU controls 验证了什么

从单进程到双进程依次运行：

~~~powershell
python projects/single-gpu-finetuning/gradient_accumulation_toy.py
python projects/single-gpu-finetuning/ddp_token_mean_control.py
python projects/single-gpu-finetuning/ddp_accumulation_no_sync_control.py
python projects/single-gpu-finetuning/amp_grad_scaler_control.py
python projects/single-gpu-finetuning/ddp_amp_overflow_consensus_control.py
~~~

| Control | 主要观察 |
|---|---|
| Gradient accumulation toy | Token sum/count 与 local-mean weighting 的差异 |
| DDP token mean | 默认 gradient mean 下 \(D/N\) scaling |
| DDP accumulation | `no_sync` scope、同步后 clipping 和一次 SGD update |
| AMP scaler | unscale-before-clip、overflow skip 与 scaler resume |
| DDP + AMP | Reduction 前 non-finite、reduction 后单-rank fault 与共同 skip |

它们使用 PyTorch CPU、Gloo、小参数和 authored inputs。结果能检查公式、collective 路径和故障状态，
不能拼接成 FSDP/ZeRO/TP/PP/EP、CUDA、多节点、目标 Trainer、收敛、吞吐或模型质量证据。

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

Model FLOPs Utilization（MFU）通常把模型理论 FLOPs × token throughput 除以硬件峰值 FLOPs。
理论 FLOPs 是否包含 attention、embedding、MoE 和 recomputation，硬件峰值使用哪个 dtype，都会改变结果。
不同框架的 MFU 在口径不同时不能直接排名。

高 GPU utilization 可能来自无效重算，tokens/s 也可能把 padding 算进分子。
同时报告 valid objective tokens、loss/quality 和实际工作分解。

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

TP 和 PP 共同处理一批样本，所以 global batch 不是 2048 或 8192。
若每样本平均 1800 个有效 target tokens，每 update 约消费 460,800 tokens；
只有 max length 2048 时，还无法推导真实 token/update。

方案还要验证 stage balance、TP group 是否在高速域、DP collective 跨多少节点、
activation/gathered-parameter 峰值，以及大规模故障率下 checkpoint interval 是否合适。

## 从本地走向目标集群

1. 在单进程 toy 上固定 loss numerator、denominator 和 update oracle；
2. 用两个 CPU/GPU processes 检查 DDP reduction、data IDs 与 failure consensus；
3. 在单节点多 GPU 逐个启用 DP、sharding、TP，不同时改变多个维度；
4. 固定 global workload 做 strong scaling，固定 per-device workload 做 weak scaling；
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
