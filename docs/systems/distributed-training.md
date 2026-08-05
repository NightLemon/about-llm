# 高效与分布式训练

## 为什么需要并行

训练内存由参数、梯度、优化器状态、激活和临时缓冲组成。以混合精度 Adam 为例，每参数实际可能远超 2 字节：BF16 权重、FP32 master weight、FP32 梯度及两份 FP32 动量已约 18 字节，具体依框架和分片策略。

## 并行维度

### 数据并行（DP）

每卡一份模型，处理不同 micro-batch，反向后 all-reduce 梯度。实现简单，但每卡必须放下完整模型和优化器状态，通信量随参数规模增长。

### ZeRO / FSDP

跨数据并行 rank 分片：

- Stage 1：分片优化器状态；
- Stage 2：再分片梯度；
- Stage 3：再分片参数，计算前按需 all-gather。

FSDP 与 ZeRO-3 思路相近但接口/实现不同。分片节省内存，却增加通信、预取和检查点复杂度。

### 张量并行（TP）

把单个矩阵乘沿行/列切到多设备。每层需要 collective，适合高速节点内互联。规模太大会被通信延迟主导。

### 流水线并行（PP）

把层分给不同 stage，将 batch 切为 micro-batches 形成流水线。存在 pipeline bubble；stage 计算不平衡会产生空闲。调度有 GPipe、1F1B、交错式等。

### 序列/上下文并行

沿序列维切分激活或注意力，适合超长上下文。需要处理跨设备 attention 通信、位置与 mask。

### 专家并行（EP）

MoE 专家分布在不同设备，token 通过 all-to-all 路由。负载不均和网络带宽是核心瓶颈。

大型训练通常组合多种并行，形成 3D/4D 并行。选择受节点拓扑影响：高频通信优先放在 NVLink/高速互联域内。

## 激活与计算权衡

激活检查点只保存部分边界，反向时重算中间激活，用额外 FLOPs 换显存。选择性重计算比整层重算更灵活。FlashAttention 减少注意力中间量和 HBM 访问，也降低显存。

## 精度

- FP32 范围和精度高，成本大。
- TF32 是部分 GPU 上 FP32 矩阵乘的加速格式。
- FP16 范围较小，常需 loss scaling。
- BF16 与 FP32 指数范围相近，LLM 训练通常更稳。
- FP8 需要尺度管理与硬件/内核支持。

混合精度不是把所有张量都降精度；归一化、累加、优化器状态和部分通信可能保留高精度。

## 有效 batch 与 token

\[
B_{global}=B_{micro}\times\text{grad accumulation}\times DP
\]

每次更新 token 数还要乘有效序列 token，并扣除 padding/被 mask 部分。TP 和 PP 不增加数据样本数，不能乘入全局 batch。

## 性能诊断

记录 tokens/s、step time 分解、GPU 利用率、HBM、通信占比、data loader 等待、MFU 与 straggler。常见瓶颈：小算子过多、数据读取慢、网络拓扑错误、bucket 太小、负载不均、频繁同步和 CPU tokenization。

## 分布式正确性

性能提升不能牺牲等价性。验证单卡与多卡的初始 loss、若干步梯度/参数差异；确认梯度缩放、dropout RNG、数据不重复、loss 归一化和 sequence packing mask。异步故障恢复还要避免漏数据或重复训练无法追踪。

## 自测

1. TP=8、PP=4、DP=16、micro-batch=2、累积=8，全局 batch 是多少？
2. 为什么 TP 通常优先放在节点内高速互联？
3. 激活检查点用什么换什么？
