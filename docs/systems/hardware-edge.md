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

判断一个模型是否适合某台机器，要依次通过三道门：

| 门槛 | 需要回答的问题 | 主要证据 |
|---|---|---|
| 能否加载 | 权重、KV、workspace 和运行时能否一起放进显存？ | 容量账本与峰值显存 |
| 能否完成请求 | 目标长度和并发下是否会 OOM，TTFT/TPOT 是否可接受？ | 固定 workload 的请求报告 |
| 能否持续运行 | 温度、功耗和时钟稳定后，吞吐与错误率是否仍达标？ | 长时间重复实验 |

“成功生成一次”只通过了第二道门的一小部分。硬件分析还要分别检查计算、显存带宽、互联、kernel、调度和温控，
再用真实工作负载验证。

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

厂商标注的 TOPS 可能是 INT8 或 INT4 稀疏运算峰值，不能直接代表 BF16 稠密 Transformer 的速度。
容量单位也要统一：GB 通常等于 \(10^9\) 字节，GiB 等于 \(2^{30}\) 字节。

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

其中 \(b\) 是每个权重的有效位数。实际量化文件还要保存缩放因子、零点和分组信息；部分层可能保持高精度，
内存布局也可能需要对齐。因此，“4-bit 7B = 3.5 GB”只是裸权重下界。

### 2.2 KV Cache

标准按层 dense K/V 近似：

\[
M_{KV}
=2L\,B\,T\,H_{kv}\,d_h\,s,
\]

其中 2 表示 K 和 V，\(L\) 是层数，\(B\) 是活动序列数，\(T\) 是已经缓存的 token 数，\(H_{kv}\) 是
K/V head 数，\(d_h\) 是每个 head 的维度，\(s\) 是每个元素占用的字节数。

GQA 和 MQA 通过减少 K/V head 数来降低这一项。分页分配器还会引入 block metadata 和内部碎片；前缀共享、
滑动窗口、KV 量化和潜变量 cache 也会改变实际布局。

仓库的 `estimate_kv_cache_bytes` 只计算上式中的理想 K/V 数组，不包含分配器 metadata。

### 2.3 Workspace 与 runtime

运行模型时，注意力与量化算子的临时空间、CUDA 图、CUDA 上下文、通信缓冲区和临时 logits 都会占用显存。
因此，只根据 checkpoint 文件大小选 GPU，可能在加载或提高并发时 OOM。

容量账本要保留安全余量，并分别记录框架报告的峰值 allocated memory 和 reserved memory。

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

这个函数使用调用者提供的**有效上限**，只计算理想的算术与数据搬运时间。Kernel launch、同步、通信、调度、
排队和温度降频都不在公式里。

所以计算结果是延迟下界，不是延迟预测。若计算与数据搬运无法完美重叠，实际时间可能更接近两项之和，或者更高。

## 4. Prefill 与 Decode 的硬件行为

### 4.1 Prefill

Prompt 的多个位置可以并行计算，大矩阵乘通常具有较高的算术强度。长序列的完整 Attention 还包含随序列长度平方
增长的部分，瓶颈可能来自计算、显存 workspace 或 Attention kernel。

FlashAttention 类算法避免把完整 score 矩阵反复写入显存，从而减少数据搬运；它没有把一般 Attention 的算术复杂度
改成线性。

### 4.2 Decode

Decode 时，每条活动序列每步通常只产生一个 token。Batch 很小时，系统为了少量新 token 仍要读取大量权重，
因此常受显存带宽和 kernel launch 限制。

Continuous batching 让更多序列同时参与一步 decode，使一次权重读取服务更多 token。代价是更高的 KV 容量、
更复杂的排队，以及可能恶化的尾延迟。

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

把 bias、激活、归一化或量化融合进同一个 kernel，可以减少启动次数和显存往返。实际收益取决于张量形状与编译器，
不能只按“少了几个算子”来估计。

### 6.2 Tiling 与布局

矩阵分块进入寄存器、共享内存或 cache 后可以重复使用。维度不对齐时，Tensor Core 利用率可能下降；转置、连续化复制
和数值类型转换还会增加不容易从模型公式中看到的数据搬运。

### 6.3 Graph/compile

CUDA Graph、提前编译（AOT）、即时编译（JIT）和 kernel 自动调优，可以减少 Python 与 kernel 启动开销。

动态形状、数据相关控制流、自定义算子或新的 batch 形状，可能触发重新编译或回退路径。基准测试要把首次编译和冷启动
时间，与预热后的稳态时间分开报告。

## 7. GPU、TPU、NPU 与 CPU

### 7.1 GPU

通用矩阵/attention 生态成熟，HBM 带宽高；受显存、功率、kernel 与互联限制。消费 GPU 的显存、ECC、NVLink/虚拟化和长期稳定性可能不同于数据中心卡。

### 7.2 TPU/专用加速器

适合编译后规则大图与特定低精度，但 host/device、shape、collective 和编译缓存影响体验。峰值只对支持的 op/dtype 成立。

### 7.3 移动/桌面 NPU

移动或桌面 NPU 的能效可能很高，但支持的算子、动态形状、上下文长度、量化格式和 SDK 往往更受限。
一个不受支持的算子若回退到 CPU 或 GPU，会引入同步和内存复制；必须用 profiler 确认每个算子实际在哪个设备执行。

### 7.4 CPU

CPU 的优势是容量较大、设备普及、启动门槛低，并且容易使用内存映射权重。常见瓶颈是 DRAM 带宽、SIMD 优化算子、
NUMA 和线程调度。低并发增量解码可能接近“每生成一个 token 都把权重流式读一遍”。

运行 CPU 基准前，要固定：

- 指令集与 kernel build；
- 物理/逻辑核心数和线程绑定；
- NUMA 节点与内存通道；
- 电源模式与同机其他负载。

线程越多不一定越快。跨 NUMA 访问和线程超额订阅都可能让性能变差。

## 8. 单张消费级 GPU 的规划

### 8.1 用你的 3070 Laptop 做第一次判断

不要只凭“RTX 3070 Laptop”这个名称推断容量；先以本机 `nvidia-smi` 显示的专用显存、功耗上限、驱动和温度为准。
针对当前的 Qwen3-0.6B + nano-vLLM 学习实验，可以按以下顺序推进：

1. 用模型参数量和加载 dtype 估算裸权重下界，再为量化 metadata、CUDA context 和 workspace 留空间；
2. 先运行 eager、并发 1 和短输出，确认模型身份、一次 prefill/decode 与 KV 释放；
3. 记录 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()`，不要用模型文件大小代替运行峰值；
4. 再启用 CUDA Graph，对比同一请求的结果、TTFT、TPOT 和额外显存；
5. 最后按实验 7B 的并发 `1/2/4/8` 与 batch token budget `256/1024` 逐档增加负载；
6. 每一档同时观察 OOM、排队、峰值显存、温度、时钟和持续吞吐。

[实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md)固定了 Qwen3-0.6B、nano-vLLM、输入输出长度和对照组，
但仓库还没有你的 3070 Laptop 实测报告。因此，本页现在能给出容量公式和实验顺序，具体“占多少显存、每秒多少 token”
要以你回传并通过 verifier 的报告为准。

先做账本：

1. 权重实际加载字节（含量化 metadata）；
2. 最大并发与 prompt/output token 的 KV；
3. runtime/workspace/graph capture；
4. 安全 headroom；
5. context overflow 与 OOM 降级顺序。

### 8.2 合理降级顺序

- 减少 concurrency/max_num_seqs；
- 减少 context/output 上限；
- 关闭过大的 graph capture 或调整 block；
- KV/权重量化（先验证质量/kernel）；
- CPU offload（接受延迟）；
- 选择更小/GQA 模型。

不要在 OOM 后随机改十个参数；一次只改一个并记录峰值、TTFT、TPOT 和质量。

### 8.3 电源与散热

消费设备长时间推理时，功耗或温度限制可能触发降频，导致第一分钟和一小时后的速度不同。报告中应记录环境温度、
功耗上限、时钟频率、设备温度和持续吞吐。超频后的短时峰值不能代表稳定容量。

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

内存映射可以减少启动时的复制，并利用操作系统 page cache。首次访问的缺页、文件格式、随机读取和系统内存压力，
仍会影响冷启动时间。

## 11. 端侧部署目标

端侧价值：离线、隐私边界、低网络依赖、可控成本；限制：RAM、包大小、存储、功耗、热、后台竞争和更新。

### 11.1 Artifact

端侧工件要明确记录模型、tokenizer、chat template、量化和运行时版本，以及哈希、签名与最低设备要求。
同一个格式名称在不同运行时中也可能存在支持差异，需要用目标运行时实际加载验证。

### 11.2 冷启动与更新

冷启动要拆成应用启动、模型映射/加载、首 token 编译和第一次请求。模型更新还需要原子下载、签名验证、空间检查、
失败回滚和旧版本清理。下载不完整的文件不能进入可加载目录。

### 11.3 Context 与内存压力

移动 OS 可能回收后台应用；动态 KV 增长会越过系统 memory pressure。设置硬 context/output limit、低内存降级和请求取消。不要依赖 swap 在交互延迟下救场。

### 11.4 WebGPU/浏览器

WebGPU 会受到浏览器实现、GPU adapter、buffer 上限、shader 编译、页面生命周期和来源隔离的影响。模型下载大小和
浏览器缓存策略也是用户成本。跨站内容与模型工件还要设计 CSP、完整性校验和权限策略。

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

GPU kernel 通常异步执行。计时边界要调用框架/设备同步，或者使用正确的 GPU event；否则测到的可能只是提交任务所需
时间。编译预热与稳态运行要分开，能耗采样窗口也必须覆盖完整执行过程。

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

仓库已经验证了三类离线内容：理想 KV 容量公式、Roofline 下界计算，以及 CPU 上的推理指标和 SSE 协议检查。
JAX MiniGPT 当前录制的运行设备也是 CPU。

目标消费 GPU、移动 NPU、WebGPU 和多卡互联仍缺少仓库内的实测报告。因此，本页对这些设备提供的是性能模型和
验收协议；具体性能结论要等目标硬件报告通过验证后再写入。

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
