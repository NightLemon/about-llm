# 高效与分布式训练

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：多 GPU 训练、平台和正确性验收工程师。
- **先修**：[预训练](../training/pretraining.md)、显存状态和基本集合通信。
- **首次阅读**：资源账本 → DP → ZeRO/FSDP → TP/PP → 正确性验证。
- **完成信号**：能按参数、梯度、优化器和激活列账，并设计多卡验收矩阵。
- **卡住时**：回到[硬件性能模型](hardware-edge.md)和[预训练预算](../training/pretraining.md)。

</div>

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

## Global batch 与 loss 归一化 { #global-batch-loss-normalization }

设每个 data-parallel rank 的 micro-batch 为 \(B_\mu\)，梯度累积次数 \(A\)，数据并行度 \(D\)：

\[
B_{global}=B_\mu A D
\]

TP、PP、EP 共同处理同一批样本，不乘入 global batch。若长度/padding 可变，每更新有效 token 为：

\[
N_{tokens}=\sum_{rank,microbatch,t}m_{rank,microbatch,t}
\]

不能用 \(B_{global}\times T_{max}\) 替代。

### Mean 还是 sum { #mean-or-sum }

若第 \(i\) 个 micro-batch/rank shard 有 \(n_i\) 个有效 token、loss sum 为 \(S_i\)，global token mean 是

\[
L=\frac{\sum_i S_i}{N},\qquad N=\sum_i n_i.
\]

先算 local mean \(S_i/n_i\) 再等权平均，会让短批或高 padding rank 权重过大；它只有在所有 \(n_i\) 相同或梯度恰好抵消等特殊情况下才与 global token mean 相同。严格实现应聚合 numerator 与 count，或对梯度使用数学等价缩放。单进程 accumulation 可对每批 `reduction="sum"` 做 backward，窗口结束后把累计梯度除以 \(N\)，随后才做 gradient clipping 与 optimizer step。

DDP 需要再核对 reducer 语义。若默认对 \(D\) 个 rank 的梯度取平均，每个 rank 直接 backward `local_loss_sum / global_N`，最终会额外少一个 \(D\)；可在 backward 前按 `D/global_N` 缩放 local sum，或改用能得到全局 sum 的等价路径。`global_N` 必须来自整个 accumulation window、所有 data-parallel ranks 的有效 token count；不能用 \(A\)、batch size 或 max length 代替。AMP 下还要把 `GradScaler` 的 scale/unscale、手工归一化和 clipping 顺序写进契约。

仓库把证据拆成两层。`gradient_accumulation_toy.py` 固定两个 micro-batch 的有效 token 数为 `[1,3]`；`Fraction` oracle 得到正确 class-aggregate logit gradient `(23/40,-23/40)`，等权 micro-batch mean 为 `(7/20,-7/20)`。PyTorch CPU Float64 的 full batch 与单进程 sum/count 梯度逐元素相同，三个 ignored/padding 位置梯度为零；这只是 reducer 集成的单进程 reference。

`ddp_token_mean_control.py` 再把两个 shard 分给 `D=2` 个真实 OS 进程，并使用 temporary FileStore、Gloo、count `all_reduce` 和 PyTorch `DistributedDataParallel` backward。rank-local loss-sum gradient 分别为 `(-1/10,+1/10)` 与 `(+12/5,-12/5)`，全局 `N=4`：

| backward 路径 | 每 rank loss 定义 | 默认 DDP mean 后梯度 |
|---|---:|---:|
| 正确 global token mean | `(D/N)S_r=(1/2)S_r` | `(+23/40,-23/40)` = `(0.575,-0.575)` |
| 漏掉 world size | `(1/N)S_r=(1/4)S_r` | `(+23/80,-23/80)` = `(0.2875,-0.2875)` |
| rank-local mean | `S_r/n_r` | `(+7/20,-7/20)` = `(0.35,-0.35)` |

两个 rank 都观察到 count 4 和相同同步梯度；正确路径相对单进程 full-batch 的最大绝对误差约 `1.11e-16`，漏 world-size 路径相对 full/2 的误差约 `5.55e-17`。这是当前 PyTorch 2.13.0+cpu/Gloo/default reducer 固定 control 的执行证据，不是所有框架或版本的规范证明。它没有执行 optimizer/parameter update/clipping、gradient accumulation + `no_sync`、AMP/scaler、FSDP/ZeRO/TP/PP/EP、CUDA/GPU、多节点、目标 LLM/tokenizer/dataset、性能或质量评测。

### Accumulation、`no_sync` 与 update control

`ddp_accumulation_no_sync_control.py` 将证据推进到一个完整但极小的 optimizer update。两个 rank 各有两个 micro-batch，valid-token counts 为 `[[1,2],[3,1]]`，global `N=7`。四个 local loss-sum class gradients 是 `(-1/10,+1/10)`、`(+8/5,-8/5)`、`(+12/5,-12/5)`、`(-1/10,+1/10)`；rank sums 为 `(+3/2,-3/2)` 与 `(+23/10,-23/10)`。默认 DDP mean 下每批乘 `D/N=2/7`，无论每批同步还是只在末批同步，精确结果都是：

```
pre-clip gradient = (+19/35,-19/35)
unclipped plain-SGD lr=7/20 parameter delta = (-19/100,+19/100)
```

真实 CPU Float64/Gloo control 分三条路径。第一条使用未注册 hook 的 built-in DDP：首批的 **forward 与 backward** 都在 `ddp.no_sync()` 内，末批正常同步。第二条注册 PyTorch `default_hooks.allreduce_hook` 的透明 reference hook以计数，同样正确包裹，观察 1 次两元素 bucket hook。第三条是负对照：forward 在 context 外、只把 backward 放进 `no_sync`；DDP 已在 forward 决定需要同步，因此观察 2 次 hook。三条路径的 pre-clip gradient 都是约 `(0.542857,-0.542857)`；同步归一化后执行 `max_grad_norm=0.5`，clip 后约 `(0.353553,-0.353553)`，plain SGD `lr=0.35` 后 bias 约 `(-0.123744,+0.123744)`，都与单进程 full batch 逐项相同。

这里的 backward-only 负对照证明“没有减少 collective”，不证明它在此 fixture 上改变数学结果。built-in 路径没有直接插桩 collective count；计数来自另一个使用官方 reference hook 的独立路径。control 只有一个两元素参数和一个 bucket，没有 dropout/BatchNorm/RNG、AMP/scaler/overflow、多参数或多 bucket、AdamW/optimizer state、checkpoint resume、FSDP/ZeRO/TP/PP/EP、CUDA/GPU、多节点、目标 Trainer/LLM、通信字节、性能或质量证据。

### AMP scale、unscale、clip 与 skip control

混合精度是独立状态机，不能仅凭上面的 Float64 DDP control 推断。仓库的 `amp_grad_scaler_control.py` 真实执行 CPU FP16 autocast 与 `torch.amp.GradScaler("cpu")`。两个 micro-batch 的 unscaled sum-gradient 是 3，初始 scale=8 时内存中的 scaled gradient 是 24。正确路径先 `unscale_` 再以 `max_norm=0.5` 做 global-norm clip，得到约 0.5 并与 full batch/SGD update 相同；负对照先 clip 24 再 unscale，optimizer-visible gradient 约为 0.0625。

同一 control 先做一次真实 AdamW step 建立 step=1、`exp_avg` 与 `exp_avg_sq`，再让三个两批窗口中的第二批产生 `inf`。GradScaler 逐次把 scale 从 8 降到 4、2、1，三次都跳过整个 optimizer update，参数与 AdamW state exact 不变。进程内复制 model/optimizer/scaler 后，恢复 scale=1 的 10000 边界梯度执行 step=2，并与不中断路径 exact；故意不加载 scaler、回到 scale=8 时 FP16 scaled gradient overflow，scale 降到 4 且 step 仍为 1。

这条证据只覆盖 PyTorch 2.13.0+cpu、单 FP32 参数、FP16 autocast、单进程和 in-memory state replay；没有把 scaler 纳入 strict 文件 artifact，也没有执行真实进程退出/重启、scheduler/RNG/DataLoader、DDP/FSDP/ZeRO、CUDA kernel、目标 Trainer/model、吞吐、收敛或质量。

### DDP + AMP overflow 共识控制

`ddp_amp_overflow_consensus_control.py` 把 default DDP reducer、`no_sync`、CPU FP16 GradScaler、AdamW 和 StepLR 放进同一条真实双进程 CPU/Gloo control。每条路径先做一次相同的 finite warm-up，建立 parameter≈0.99、AdamW step=1、scheduler epoch=1/LR=0.005、scale=8 和 growth tracker=1，再比较：

| 故障位置与决策 | rank 0 | rank 1 | 结果 |
|---|---|---|---|
| 首批 `no_sync` 内 rank 0 产生 non-finite，末批用 built-in DDP 同步 | 末批后 non-finite | 末批前 local scaled grad=8，末批后 non-finite | 两边都 skip，scale `8→4`、tracker `1→0`；model/AdamW/StepLR 保持一致 |
| DDP 已得到两边 finite scaled grad=8 后，仓库故意在 rank 0 的 `unscale_` **之前**把 grad 改成 Inf；无额外共识 | skip，scale=4、step=1 | unscaled grad=1，scale=8、step=2 | 参数、moments、scheduler、LR、scale/tracker 全部分叉 |
| 同样的 post-reduction fault，`unscale_` 后对 local non-finite flag 做 `all_reduce(MAX)` | local flag=1，global=1 | local flag=0，global=1 | 两边都不调用 `scaler.step`/optimizer/scheduler，并用显式共同 scale policy 回退到 4，训练状态保持一致 |

第一条说明：在这个默认 reducer、单参数/单 bucket fixture 中，**reduction 之前**产生的 Inf 会随梯度同步传播，所以不能据此宣称 vanilla DDP 总要再做一次 finite-flag collective。第二条是 authored fault injection，不是“DDP 正常会在 reduction 后损坏一个 rank”；它代表 per-rank gradient transform、条件参数、自定义 communication path、硬件/内存故障等可能绕开共同 reduction 的风险。finite 共识必须放在所有可能生成 non-finite 的 gradient transform 之后、任何 optimizer mutation 之前；step 后才发现 mismatch 已经无法撤销另一个 rank 的更新。

第三条只证明这个 optimizer-pre gate 阻止了当前故障分叉，不是可直接复制到所有框架的 distributed scaler。`GradScaler.update(new_scale=4)` 是仓库显式 scale policy：两 rank 的完整已序列化 scaler state 一致，但 growth tracker 保持 1；native found-inf transition 则会把 tracker 重置为 0。生产实现应优先使用目标框架公开的 distributed scaler/overflow 协议，或明确定义并测试 scale、growth tracker、optimizer、scheduler、clip 与 checkpoint 的全状态共识，不能修改私有 found-inf 字段，也不能用 `scaler.step()`/`optimizer.step()` 的返回值冒充通用 `did_step`。

该控制仍只有一个 FP32 参数、单 bucket、同机 CPU/Gloo 和人为 Inf；built-in collective count 未直接插桩，也没有 clipping、随机层、自然 overflow、custom hook、conditional graph、checkpoint/elastic restart、CUDA/NCCL、多节点、FSDP/ZeRO、目标 LLM/Trainer、性能、收敛或质量证据。

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

独立的 `moe_distributed_capacity_control.py` 只把证据向前推进一层：两个 same-host CPU/Gloo ranks 用真实 tensor `all_gather` 形成 replicated 4-token global routing input，并用两个 `all_reduce(SUM)` 对账 active count=4、selected counts `[4,0]`。Local-only capacity competition 跨 ranks 合计 kept=2；global score-priority capacity=1 只 kept=1，mask `[F,F,T,F]`，rank-0 output 因失去 local winner 而改变 `0.9640275800758169`。因此它确实执行了 collective capacity-group 反事实，不是只给单进程 group IDs 换名称。

但 hidden `all_gather` 让每个 rank 复制 global batch，router 与 experts 也完全复制；这不具备 scalable EP 的 expert ownership，且没有 token-to-expert `all_to_all`/return combine、distributed backward、optimizer 或性能测量。它用于教学上分开三类 process group：capacity/routing group、expert dispatch group、gradient synchronization group；不能外推到 NCCL、多节点、DeepSeek/Qwen、通信 bytes/tail 或 GPU throughput。

第二条 `moe_all_to_all_control.py` 单独验证 expert dispatch group：expert 0/1 分别只驻留 rank 0/1；source→owner variable splits 为 `[[1,2],[1,0]]`。每 rank 的五次 `all_to_all_single` 依次交换 counts，发送 hidden/gate 与 source metadata，再做 owner→source return 的 output/gate 与 metadata。Rank 0 的 arrival global IDs 是 `[1,0,2]`，按 source-local index scatter 才与单进程 forward oracle 完全相等；忽略 metadata 会产生 `0.8958737432590591` 最大差。

报告的 256/160、合计 416 logical tensor-payload bytes 只是当前 int64/float64 authored tensors 的 `numel × element_size`；不等于 wire bytes，也没有测 collective latency。该控制不含 capacity/drop、backward/optimizer、CUDA/NCCL、多节点、target checkpoint 或性能；不能把它与 replicated-capacity control 的独立证据相加成完整、可扩展或可训练的 EP runtime。

第三条 `moe_all_to_all_training_control.py` 把训练图也做成可执行对照。Authored autograd binding 在 backward 用 reverse-split `all_to_all_single` backward：先把 output/gate gradients送回 owner，再把 hidden/gate gradients送回 source。每 rank 的 local loss sum 都除 global token count；owner expert gradient 不做 data-parallel all-reduce，因为 owner 已处理来自所有 sources 的本 expert tokens；replicated router gradient 做 SUM all-reduce，保证两个副本更新一致。

当前 global MSE `20.78017329703821→19.41091750734501`，gradients、一步 SGD 参数和 post-step forward 都与单进程 oracle exact。但 custom autograd Function 不是 DDP/生产 EP，call ledger 不是 profiler；fixture仍无 capacity、mixed precision、optimizer state、CUDA/NCCL、多节点、目标模型、收敛或性能证据。

第四条 `moe_all_to_all_capacity_training_control.py` 在另一张图中把 global score-priority drop、owner-only dispatch 与 backward 合并。四个 active tokens 的 selected counts `[2,2]` 在 capacity=1 下变为 keep mask `[F,T,T,F]`；kept-only source→owner splits 为 `[[1,1],[0,0]]`，其中 rank 1 的全零 source→owner splits `[0,0]` 仍通过 empty-payload graph edge 触发 reverse collective。Dropped tokens 的 task hidden gradient 为 0；router SUM gradient、owner expert gradients、一步参数和 post-step forward 均与单进程 capacity oracle exact。

该 fixture 的 global MSE 是 `15.253670387373656→14.530264380025987`。每 rank authored ledger 为 payload forward/backward 4/2、count+metadata 6、capacity-route all-gather 4、router reduce 1；仍不是 backend profiler。它没有验证 Gloo 以外 backend、DDP/FSDP/ZeRO/TP/PP、mixed precision、optimizer state、reroute/dropless、shared/fine-grained experts、目标模型、扩展性、收敛或性能。

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

“full batch 与 gradient accumulation 等价”还要求：监督 token/mask 相同；窗口中只在末尾 step/zero-grad；clip 与 scheduler 按 optimizer update 执行；模型没有跨样本 BatchNorm 等 batch-coupled 运算；dropout、gradient checkpoint 等随机路径按已定义的 RNG 契约比较；AMP overflow、`no_sync` 与 collective 边界一致。即使数学目标相同，浮点求和顺序也通常只支持容差等价，不自动支持 bitwise 等价。

本仓库第一条双进程 DDP token-mean control 只覆盖“一次 local backward 后立即同步”的三种 reduction 路径；第二条 accumulation control 已补两个 micro-batch、built-in `no_sync`、同步后 clipping 和 plain SGD update；第三条 DDP+AMP control 再以 AdamW/StepLR 验证同步前 non-finite 的共同 skip、同步后单 rank 故障的分叉和 optimizer-pre flag 共识。三者合起来仍不能证明随机模型、多参数/bucket、自然 overflow、目标 Trainer、FSDP/ZeRO、GPU 或多节点下的 optimizer-update 等价。

### 2. 数据等价

记录每 rank sample/source id，验证一个 global step 不重不漏；gradient accumulation、worker 重启和 resume 后同样检查。Distributed sampler 的 `set_epoch`/等价状态与 shuffle RNG 要保存。

还要区分 sampler-emitted、main-loop-consumed 与 optimizer-committed cursor。`dataloader_prefetch_resume_control.py` 在单进程训练控制面内真实启动两个 spawn workers；`prefetch_factor=2` 时，主循环收完 3 条，sampler 已发到 7。从 emitted cursor 重启会漏掉 4 个 queue 中未消费 IDs，从 consumed cursor 重启可恢复顺序；但 fresh worker RNG tail 仍不同。这个 control 没有 DistributedSampler/DDP，也没有把 sample consumption 与 optimizer commit 做原子事务，因此分布式训练仍须为每 rank/global update 明确 cursor 聚合、drop/replay policy 和失败边界。

独立 `optimizer_commit_resume_control.py` 再执行 main-process inverted-Bernoulli mask、真实两步 accumulation、backward、SGD momentum 与 StepLR：崩溃时 emitted/consumed/committed=`7/3/2`。base 不含 `.grad`，但保存 commit-boundary scheduler/Torch RNG；从 committed=2 恢复 RNG并重放，与 uninterrupted 的 model/optimizer/scheduler/RNG bit-exact。第一个隔离负例从 consumed=3 恢复正确 crash RNG却漏 gradients/sample `1`：未来 RNG 相同，5 次 optimizer/scheduler step 与 LR `0.0125` 也相同，参数仍漂移 `0.005767858566116724`。第五个 PID 加载绑定 base digest 的 pending `[1]`、position/divisor、逐参数 gradients与 crash RNG sidecar，从 consumed=3 继续也 bit-exact；第六个 PID 保留 gradients/ledger 却错误使用 commit-boundary RNG，参数漂移 `0.017878893573032573` 且终态 RNG 不同。sidecar 路径先要求最后发布的 canonical manifest 完整绑定 base/sidecar identity；base-only、payloads-without-manifest、manifest-without-sidecar 与 post-manifest tamper 四种快照均 fail closed，但 base-only 仍可用于 commit replay。

它仍只证明单训练进程的两种恢复协议与两个隔离负例，未执行 DDP/DistributedSampler、rank 间 ledger 共识、collective、sharded optimizer 或 elastic membership；manifest-last 不会让 base+sidecar+manifest 与所有 rank update 原子提交，也没有 directory `fsync`、断电、原子目录/远程快照或来源认证证据。它覆盖 main-process Torch RNG/StepLR，不覆盖 worker/Python/NumPy/CUDA RNG、GradScaler 或原生任意随机层。分布式实现仍须定义全局 commit receipt、sidecar/shard manifest completeness、失败 rank 的共同回滚/重放以及重复样本是否可接受。

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
