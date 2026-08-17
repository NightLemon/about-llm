# 硬件性能模型与端侧部署

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：性能分析、容量规划和端侧部署工程师。
- **先修**：推理阶段、字节/FLOPs 单位和基础硬件组成。
- **首次阅读**：单位口径 → 容量账本 → Roofline → prefill/decode → benchmark。
- **完成信号**：能为目标设备写可复现 benchmark，而不是只引用峰值规格。
- **卡住时**：先读[推理基础](inference.md)的两个阶段和 KV Cache。

</div>

“模型能运行”只证明容量可能足够，不证明延迟、吞吐、能耗或稳定性满足目标。硬件分析要把权重/KV 容量、计算、内存带宽、互联、kernel、调度和温控分别建模，再用真实 workload 验证。

## 1. 单位与口径

常见量：

| 量 | 单位 | 必须说明 |
| --- | --- | --- |
| 计算吞吐 | FLOP/s、OP/s、TOPS | dtype、dense/sparse、FMA 计数 |
| 内存容量 | byte、GiB/GB | 权重、KV、workspace、allocator 是否包含 |
| 内存带宽 | byte/s、GB/s | 理论峰值还是实测、读写模式 |
| 互联 | GB/s、latency | 单向/双向、拓扑、collective |
| 延迟 | ms/s | TTFT、TPOT、E2E、排队是否包含 |
| 吞吐 | tokens/s、requests/s | 输入/输出 token、并发和 batch |
| 功率/能量 | W、J、Wh | 设备还是整机、采样方法和时间 |

厂商 “TOPS” 可能指 INT8/INT4 稀疏峰值，与 BF16 dense Transformer 不可直接比较。GB 通常是 \(10^9\) bytes，GiB 是 \(2^{30}\) bytes。

## 2. 容量账本

推理显存/内存至少包含：

\[
M_{total}
=M_{weights}
+M_{KV}
+M_{activations/workspace}
+M_{runtime}
+M_{fragmentation}.
\]

### 2.1 权重

理想权重存储约为

\[
M_{weights}\approx N\cdot\frac{b}{8},
\]

其中 \(b\) 是每权重有效 bit。实际量化还包括 scale、zero point、group metadata、未量化层和 alignment。所谓“4-bit 7B = 3.5GB”只是裸权重下界。

### 2.2 KV Cache

标准按层 dense K/V 近似：

\[
M_{KV}
=2L\,B\,T\,H_{kv}\,d_h\,s,
\]

其中 2 表示 key/value，\(L\) 为层数，\(B\) 为 active sequences，\(T\) 为 cached tokens，\(H_{kv}\) 为 KV heads，\(d_h\) 为 head dimension，\(s\) 为每元素字节。

GQA/MQA 通过减少 KV heads 降低该项。Paged/block allocator 还有 block metadata 和内部碎片；prefix cache 共享、sliding window、KV quantization 或 latent cache 需按具体 layout 另算。仓库 `estimate_kv_cache_bytes` 只计算上式理想存储，并明确排除 allocator metadata。

### 2.3 Workspace 与 runtime

Attention/quantization kernel、graph capture、CUDA context、通信 buffer 和临时 logits 都占空间。只根据 checkpoint 文件大小选择 GPU，常会在加载或高并发时 OOM。应保留 headroom 并测峰值 reserved/allocated memory。

## 3. Roofline 模型

算术强度：

\[
I=\frac{F}{Q}
\quad(\text{FLOPs per byte}),
\]

其中 \(F\) 是运算，\(Q\) 是从目标内存层级搬运的字节。给定有效计算上限 \(P\) 与有效带宽 \(BW\)：

\[
t_{ideal}\ge
\max\left(\frac{F}{P},\frac{Q}{BW}\right).
\]

Ridge point 为 \(I_{ridge}=P/BW\)：\(I<I_{ridge}\) 更可能 bandwidth-bound，反之更可能 compute-bound。

### 3.1 可执行下界

```python
from about_llm.inference import roofline_lower_bound

bound = roofline_lower_bound(
    flop_count=100,
    bytes_moved=100,
    effective_flops_per_second=100,
    effective_bytes_per_second=10,
)
assert bound.bottleneck == "memory"
assert bound.lower_bound_seconds == 10
```

该函数使用调用者提供的 **effective ceilings**，并明确排除 launch、依赖、同步、通信、调度、排队和 thermal throttling。因此结果是理想下界，不是 latency prediction。若计算与搬运不能完美重叠，实际时间可能更接近两项之和或更高。

## 4. Prefill 与 Decode 的硬件行为

### 4.1 Prefill

Prompt 的多个位置可并行，大矩阵乘通常有较高 arithmetic intensity。长序列 full attention 还包含二次项，可能受计算、HBM workspace 或 attention kernel 限制。FlashAttention 类算法减少 materialized score traffic，不是把 attention 数学复杂度在所有场景变成线性。

### 4.2 Decode

每个 active sequence 每步通常只产生一个 token。小 batch 时，需要为很少的新 token 读取大量权重，常受权重/缓存带宽和 kernel launch 限制。Continuous batching 增加同时处理的序列，让权重读取被更多 token 摊销，但会增加 KV 容量、排队和 tail latency。

不能只给一个 tokens/s：输入长度、输出长度、并发、batch、cache hit 与 finish reason 都会改变结果。

## 5. 量化：容量收益不等于等比例加速

权重量化降低 storage/traffic，但速度取决于：

- 硬件是否原生支持该 dtype；
- kernel 是否 fused dequantize + matmul；
- group size、scale/zero metadata；
- shape 是否对齐 Tensor Core/SIMD；
- activation/KV 是否仍为高精度；
- batch/sequence 处于 compute 还是 bandwidth 区域；
- CPU/GPU 间是否额外复制。

若 int4 权重先解压到 FP16 再用低效 kernel，容量下降但延迟未必显著改善。量化还需在目标语言、长上下文、代码、工具调用和安全切片测质量，而不只测 perplexity。

## 6. Kernel、shape 与编译

### 6.1 Kernel fusion

融合 bias/activation/norm/quantization 可减少 launch 和 HBM round trip。收益依 shape 与编译器，不是“算子数量减少多少”即可推断。

### 6.2 Tiling 与布局

矩阵 tile 进入 register/shared memory/cache 后复用；维度不对齐可能降低 tensor core 使用率。Transpose、contiguous copy 和 dtype cast 会增加隐藏 traffic。

### 6.3 Graph/compile

Graph capture、AOT/JIT compile 和 kernel autotune 能减少 Python/launch overhead，但 dynamic shape、control flow、custom op 或不同 batch 会触发重新编译/fallback。基准要区分 compile/cold start 和 steady state。

## 7. GPU、TPU、NPU 与 CPU

### 7.1 GPU

通用矩阵/attention 生态成熟，HBM 带宽高；受显存、功率、kernel 与互联限制。消费 GPU 的显存、ECC、NVLink/虚拟化和长期稳定性可能不同于数据中心卡。

### 7.2 TPU/专用加速器

适合编译后规则大图与特定低精度，但 host/device、shape、collective 和编译缓存影响体验。峰值只对支持的 op/dtype 成立。

### 7.3 移动/桌面 NPU

能效可能高，但支持的 op、dynamic shape、context length、量化 format 和 SDK 有限。一个 unsupported op 回退 CPU/GPU 会引入同步和内存复制，必须用 profiler 确认实际 placement。

### 7.4 CPU

优势是容量、普及、低启动门槛和 memory-mapped weights；瓶颈常为 DRAM bandwidth、SIMD kernel、NUMA 和线程调度。小 batch decode 可能接近权重流式读取。

CPU 基准固定：ISA/kernel build、physical/logical cores、thread affinity、NUMA node、memory channels、power mode 和其他负载。线程越多不一定越快；跨 NUMA 访问和 oversubscription 会恶化。

## 8. 单张消费级 GPU 的规划

先做账本：

1. 权重实际加载字节（含量化 metadata）；
2. 最大并发与 prompt/output token 的 KV；
3. runtime/workspace/graph capture；
4. 安全 headroom；
5. context overflow 与 OOM 降级顺序。

### 8.1 合理降级顺序

- 减少 concurrency/max_num_seqs；
- 减少 context/output 上限；
- 关闭过大的 graph capture 或调整 block；
- KV/权重量化（先验证质量/kernel）；
- CPU offload（接受延迟）；
- 选择更小/GQA 模型。

不要在 OOM 后随机改十个参数；一次只改一个并记录峰值、TTFT、TPOT 和质量。

### 8.2 电源与散热

消费设备长时间推理可能因 power/temperature throttle 使前一分钟与一小时后不同。报告 ambient、power limit、clock、temperature 和 sustained throughput。超频结果不代表稳定容量。

## 9. 多设备与拓扑

- **Tensor Parallel**：逐层 collective，延迟敏感，优先同一高带宽域；
- **Pipeline Parallel**：stage 传激活，有 bubble，跨较慢链路有时更合适；
- **Data/Replica Parallel serving**：每副本独立，简单但每卡需完整权重；
- **Expert Parallel**：MoE all-to-all，负载不均和拓扑关键。

PCIe switch、NUMA root、NVLink/NVSwitch 和网络拓扑决定真实路径。模型分到两卡“刚好能放下”可能因每层同步比单卡小模型更慢。

测 strong scaling efficiency：

\[
E_p=\frac{t_1}{p\,t_p}.
\]

若单卡无法运行，则不能伪造 \(t_1\)；改报相邻可测配置和 absolute throughput。

## 10. Offload 与分层存储

GPU↔CPU↔SSD offload 扩大容量，但每 token 若反复跨 PCIe/存储读取权重，性能上限由最慢链路决定。异步 prefetch 只有在传输可被独立计算隐藏时有效。

适合：低吞吐、离线任务、稀疏专家或偶尔访问模块。不适合：严格 TPOT 且每层都需 offload 的交互 decode。

Memory mapping 降低启动复制并允许 OS page cache，但首次访问 page fault、文件格式、随机访问和内存压力会影响 cold run。

## 11. 端侧部署目标

端侧价值：离线、隐私边界、低网络依赖、可控成本；限制：RAM、包大小、存储、功耗、热、后台竞争和更新。

### 11.1 Artifact

明确 model/tokenizer/chat template/quantization/runtime versions、hash/signature 和最低设备。格式名（如某种量化容器）不保证所有 runtime 解释一致。

### 11.2 冷启动与更新

测 app start、model map/load、first-token compile 和 first request。模型更新需要原子下载、签名验证、空间检查、失败回滚和旧版本清理；部分下载不能被当成可加载模型。

### 11.3 Context 与内存压力

移动 OS 可能回收后台应用；动态 KV 增长会越过系统 memory pressure。设置硬 context/output limit、低内存降级和请求取消。不要依赖 swap 在交互延迟下救场。

### 11.4 WebGPU/浏览器

受浏览器实现、adapter、buffer 限制、shader compile、页面生命周期和来源隔离影响。下载大小与 cache policy 也是用户成本。跨站内容和模型 artifact 需要 CSP/integrity/permission 设计。

## 12. Benchmark 协议

### 12.1 固定输入

- model/revision、runtime/commit、dtype/quantization；
- hardware、driver、OS、power/clock；
- prompt/output token 长度分布；
- batch/concurrency、streaming、scheduler；
- sampling 与 stop；
- cache warm/cold、prefix cache 命中；
- 运行时长和重复次数。

### 12.2 指标

- load/cold-start；
- TTFT、TPOT/inter-token latency、E2E；
- request/token throughput；
- p50/p95/p99 和超时；
- peak allocated/reserved/RAM；
- power/energy 与 sustained thermal behavior；
- 输出 token usage 和质量 gate。

平均 latency 会掩盖尾延迟。只用固定 max output 可能让早停策略不公平；记录实际 finish reason。

### 12.3 同步计时

GPU kernel 通常异步。计时前后需要 framework/device synchronize 或正确 event；否则只测 enqueue。编译 warmup 与 steady-state 分开。Energy sampling 的时间窗口也必须覆盖真实执行。

## 13. 能源与环境测量

设备能量：

\[
E=\int P(t)dt.
\]

短 benchmark 的功率采样误差、idle baseline 与外部整机功耗会影响结果。报告：设备/整机边界、采样频率、idle subtraction、PUE 是否包含、地区电力因子和流量总量。

更低 TPOT 不一定更低能量：高功率缩短时间可能改善或恶化总 J/token，需要实测。也不能用 FLOPs 直接推精确碳/水影响。

## 14. 安全与可靠性

- artifact hash/signature 与安全更新；
- 不加载不可信 pickle/remote code；
- runtime/driver 供应链；
- 设备丢失时本地数据/模型保护；
- crash、OOM、thermal 与 watchdog fallback；
- memory-mapped file 权限与临时文件清理；
- 端侧日志/telemetry opt-in、最小化和 TTL。

本地执行减少网络传输，不等于自动隐私：键盘、clipboard、缓存、crash log 和应用备份仍会泄露。

## 15. 选型流程

1. 定义质量、context、并发、TTFT/TPOT、能耗和成本 gate。
2. 用权重 + KV + workspace 账本排除容量不可能方案。
3. 用 roofline 判断优先关注 compute、bandwidth 还是 latency overhead。
4. 确认 target dtype/op 有真实 kernel，不只看峰值规格。
5. 在目标 workload 测 cold/warm、短/长、low/high concurrency。
6. 做量化质量与性能 Pareto。
7. 测 sustained thermal、OOM 和 fallback。
8. 固定 artifact/环境并保留 raw measurements。

## 16. 当前仓库证据边界

仓库已有精确的理想 KV 公式、roofline 下界实现和单测，以及 CPU 侧推理指标/SSE 协议检查。JAX MiniGPT 的当前实跑 device 是 CPU。没有在目标消费 GPU、移动 NPU、WebGPU 或多卡互联上完成硬件基准；因此本文的 GPU/端侧内容是性能模型与验收协议，不是实测性能声明。

## 17. 常见错误结论

- **“模型权重小于显存，所以一定能运行”**：KV、workspace、runtime 和碎片仍占空间。
- **“峰值 TOPS 更高，所以 LLM 更快”**：dtype、op、shape、bandwidth 和 kernel 决定可达性能。
- **“Roofline 下界就是预测延迟”**：它排除了多种不可重叠开销。
- **“Prefill 一定 compute-bound、decode 一定 memory-bound”**：这是常见区域，不是所有长度/batch/kernel 的定理。
- **“4-bit 会让延迟缩短四倍”**：metadata、dequant、activation 和 kernel 限制收益。
- **“端侧运行就自动保护隐私”**：本地日志、缓存、备份和系统权限仍是风险。
- **“两卡能放下就比单卡更快”**：同步与慢链路可能主导。

## 自测与实践

1. 为 7B 4-bit 模型列出裸权重之外的所有内存项。
2. 使用 `roofline_lower_bound` 构造 compute、memory、balanced 三个案例。
3. 为什么提高 continuous batch 会增加吞吐但可能恶化 p99？
4. 设计量化前后同时包含质量、TPOT、峰值内存和 J/token 的实验。
5. 给移动 NPU 设计 operator fallback 的 profiler 验收。
6. 解释为何一次 30 秒桌面 GPU benchmark 不能代表持续一小时的端侧体验。
