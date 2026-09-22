# 分布式训练正确性：混合精度、恢复与验收

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已经选定并行方案，需要证明多卡训练数值与恢复行为可信的工程师。
- **先修**：[分布式训练](distributed-training.md)中的资源账本、global loss 和并行维度。
- **首次阅读**：混合精度状态机 → checkpoint 全局时刻 → 单步等价 → 故障恢复 → 可比较 run。
- **完成信号**：能为一次分布式 run 写出 dtype、提交边界、恢复协议、容差和比较 manifest。
- **卡住时**：先退回单进程固定 batch，记录 loss numerator、denominator 和一次 optimizer update。

</div>

训练能够启动，只证明资源路径可执行。若某个 rank 静默跳步、checkpoint 混合两个时刻，或比较的两个 run
使用不同有效 token 分母，最终 loss 曲线仍可能看起来合理。本页把这些问题收敛为可逐层核对的正确性协议。

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

仓库的 CPU 恢复控制展示两种策略：主循环把已消费、sampler 已发出与 optimizer 已提交位置分开记录。第一条路径从
已提交 cursor 重放；第二条路径把未提交样本、累积梯度与崩溃时 RNG 写入并绑定到基础 checkpoint 的 sidecar，
再从已消费 cursor 继续。`optimizer_commit_resume_control.py` 中，只有 optimizer、scheduler、样本账本和
commit-boundary RNG 都已前进，窗口才算 committed；这些状态由主训练循环持有，不能由 DataLoader 的预取位置代替。

在当前 CPU/float64 样例中，这两条路径都与不中断基线精确恢复。故意丢掉 sidecar 梯度，或使用错误 RNG 恢复后，
参数会产生可观察漂移。这个证据覆盖确定性单机的 DataLoader worker、梯度累积和恢复状态。
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

## 从本地走向目标集群

1. 在单进程小例子上写清 loss numerator、denominator 和预期 update；
2. 用两个 CPU/GPU 进程检查 DDP reduction、样本 ID 与失败共识；
3. 在单节点多 GPU 逐个启用 DP、sharding、TP，不同时改变多个维度；
4. 固定全局 workload 做 strong scaling，固定每设备 workload 做 weak scaling；
5. 注入 rank crash、checkpoint 中断、worker restart 和 non-finite gradient；
6. 扩到多节点后保存 topology、collective trace 和每 rank timeline；
7. 每一级都与上一级做数值、data lineage 和 checkpoint 对照。

未获得目标硬件前，这些命令和公式属于运行计划，不能写成已验证 GPU throughput 或 MFU。

## 完成信号：一张可复核的 run 对比卡

对照 run 必须固定或明确列出模型、数据顺序、global batch 与有效 token、所有并行维度、dtype/scale、
runtime、checkpoint generation 和随机状态。报告单步差值、容差来源、恢复后的数据/参数对照与性能分解；
若目标硬件尚未运行，就把吞吐、MFU 和故障恢复标为未验证。

## 自测与实践

1. AMP 中一个 rank step、另一个 rank skip 后，哪些状态会分叉？
2. 为什么分片文件齐全仍不代表 checkpoint 对应真实全局时刻？
3. 为梯度累积窗口写出“已发出、已消费、已提交”三个 cursor。
4. 两份报告中的 MFU 为什么可能无法直接比较？
5. 设计一次 rank crash、checkpoint 中断和 non-finite gradient 注入，并写出恢复后的对照项。

## 一手资料

- Rajbhandari 等，[ZeRO](https://arxiv.org/abs/1910.02054)。
- Shoeybi 等，[Megatron-LM](https://arxiv.org/abs/1909.08053)。
- Huang 等，[GPipe](https://arxiv.org/abs/1811.06965)。
- PyTorch，[FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html)。
- DeepSpeed、Megatron Core、JAX sharding 与目标 communication backend 的固定版本官方文档。
