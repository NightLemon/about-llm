# LLM 推理优化：先找瓶颈，再选技术

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已经理解请求生命周期，希望分析 TTFT、TPOT、吞吐和 KV 容量的工程师。
- **先修**：[推理基础](inference.md)与[端到端请求生命周期](inference-request-lifecycle.md)。
- **首次阅读**：症状定位 → Roofline → KV 管理 → batching → 量化与投机解码 → profiling。
- **完成信号**：能根据一个具体性能症状提出测量方案，而不是直接罗列优化技术。
- **卡住时**：回到请求 A 的时间线，只保留 prompt 长度、输出长度、batch 和 KV bytes 四个变量。

</div>

假设一个服务从并发 1 提高到并发 8 后出现三件事：TTFT 变高、TPOT 略好、吞吐增加，偶尔还 OOM。

这不是一个“应该打开哪个优化开关”的问题。它至少包含排队、prefill、decode、batch 和 KV 容量五个因素。
正确顺序是先确定哪个资源限制了目标 workload，再改变一个变量并重新测量。

## 从症状开始，而不是从技术名词开始

| 主要症状 | 第一批检查 | 常见但非唯一原因 |
|---|---|---|
| TTFT 高 | client/server queue、prompt 长度、prefill trace | 过载、长 prompt、tokenization、prefill kernel |
| TPOT 高 | decode batch、权重/KV 读取、launch、preemption | batch 太小、带宽、低效 kernel、频繁重算 |
| 吞吐低 | GPU 利用、batch 空洞、CPU/tokenizer、失败率 | admission 太保守、调度空闲、主机瓶颈 |
| OOM 或拒绝多 | 权重、KV、workspace、峰值长度和并发 | context 上限过大、KV block 不足、量化格式不匹配 |
| 尾延迟高 | 长短请求切片、arrival process、抢占和重试 | Head-of-line blocking、过载、外部依赖 |

平均 tokens/s 会把这些现象混在一起。优化前先按输入长度、输出长度、并发和终态切片。

## 先诊断开头这组症状 { #worked-diagnosis }

现在只使用已经知道的四个现象，不急着改参数：

| 观察 | 当前能作出的判断 | 还不能确定什么 |
|---|---|---|
| TTFT 变高 | 请求在排队或 prefill 阶段等待更久 | 是到达过载、长 Prompt，还是调度偏向 decode |
| TPOT 略好 | 更大的 decode batch 可能提高了权重复用 | 是否还有更快的注意力或矩阵乘内核 |
| 总吞吐增加 | 批处理确实让 GPU 完成了更多 token | 增长是否满足尾延迟与失败率目标 |
| 偶尔 OOM | 某些长度与并发组合超过峰值容量 | 峰值来自 KV、工作区、CUDA Graph 还是其他进程 |

第一轮实验固定模型、输入与输出长度分布、采样参数和请求到达方式，只改变并发 `1/2/4/8`。
每个请求至少保存四段时间：进入服务、开始 prefill、产生第一个 token 和完成；同时记录 KV block 高水位和显存峰值。

结果出来后再选择分支：

1. **服务队列先增长**：降低准入上限，或调整每轮 sequence/token 预算；
2. **长 prefill 挡住 decode**：比较分块 prefill 的不同 token 预算；
3. **KV block 先耗尽**：降低并发或长度上限，再检查分页、抢占和释放是否正确；
4. **队列正常但 TPOT 仍高**：再查看 decode batch、权重与 KV 带宽、内核启动和量化路径；
5. **显存仍有余量却 OOM**：按权重、KV、工作区、CUDA Graph 预留和其他进程重新做峰值账本。

这组症状已经说明批处理带来吞吐收益，也暴露了延迟和容量代价。它还不足以证明应该打开 CUDA Graph、量化，
或直接换更大的 batch；后面的每项技术都对应上面某个具体分支。

## Prefill 和 decode 为什么常有不同瓶颈

Prefill 一次处理许多 prompt positions，矩阵通常更大，并行度更高。
Decode 每条序列每轮通常只增加一个输入位置，却要反复读取权重和越来越长的 KV。

这产生一个常见但不是定律的经验：

- 较长 prompt、合适 batch 的 prefill 常更偏 compute-bound；
- 小 batch decode 常更偏 memory-bandwidth-bound；
- Batch 增大后，权重读取可被更多序列复用，decode 的算术强度会提高。

因此，“prefill 更偏计算、decode 更偏带宽”是一条分析起点。最终瓶颈仍要结合 batch、长度、kernel 和硬件实测。

## 用 Roofline 建立最小性能直觉

算术强度定义为：

\[
I=\frac{\text{FLOPs}}{\text{bytes moved}}.
\]

若 (I) 很低，性能上限更容易受内存带宽限制；若 (I) 很高，则更容易接近计算吞吐上限。

小 batch decode 中，一次权重读取只服务少量序列，(I) 较低。增大 batch 能提高权重复用和总吞吐，
但请求需要等待更大的执行批次，TTFT 或单请求延迟可能上升。

在线服务真正要找的是“满足 SLO 时的持续吞吐”，不是脱离延迟和失败率的最大 batch。

## Attention：先区分计算量与数据搬运

朴素因果自注意力在 prefill 中会形成随长度约 (O(L^2)) 增长的分数区域。
FlashAttention 使用分块计算和在线 softmax，减少完整分数与概率矩阵在 GPU 高带宽显存中的反复读写。

它优化的是 IO 路径，不是把精确 attention 的一般计算量变成线性，也不会减少长期保存的标准 KV 容量。

仓库提供一个 CPU online-softmax 实验，用小块 recurrence 对账 dense reference：

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
~~~

这个实验会逐块维护当前最大值、归一化因子和加权 value 累加值，并用它们重建最终结果。
它只验证在线 softmax 的数学过程。FlashAttention 的 CUDA 内核、GPU 显存和性能需要在目标设备上测量。

## KV 容量：先算理想 payload

每个已缓存 token 的标准 K/V payload 近似为：

\[
M_{KV/token}=2\times L\times H_{kv}\times D\times bytes(dtype).
\]

以 32 层、8 个 KV 头、每头 128 维和 BF16 为例，理想 KV 数据量是每 token 128 KiB。
单条序列缓存 8192 个位置时，理想数据量是 1 GiB。

容量规划还要加：

- 权重与未量化层；
- Block metadata 与对齐；
- 临时 activation、attention workspace 和通信 buffer；
- CUDA Graph 或编译缓存；
- Runtime、driver 和其他进程的保留空间。

“GPU 显存除以模型文件大小”不能得到可服务的并发数。

## Paged KV：把连续空间问题改成映射问题

若每条请求预留最大长度的连续 KV，短请求会浪费尾部空间，动态增删也需要寻找连续区域。
Paged KV 把物理 arena 切成固定 block，并让每条序列保存自己的 block table。

假设 block size 为 4，序列长度为 6：

```text
logical block 0: 4 tokens -> physical block 5
logical block 1: 2 tokens -> physical block 1
block table: [5, 1]
```

物理 block 可以不连续。尾块仍浪费两个 slot，所以分页减少的是预留和外部连续性问题，
不是让碎片自动归零。

### 三份账本不要混在一起

共享前缀和 COW 存在时，至少分开：

| 账本 | 含义 |
|---|---|
| logical tokens | 各序列长度之和，共享 prefix 会按序列重复计数 |
| physical token values | 物理 block 中实际保存的 token positions，共享只算一次 |
| allocated token slots | 已分配物理 block 数乘 block size |

内部碎片是 `allocated token slots - physical token values`，不能直接由 logical tokens 推导。

### Shared partial tail 为什么必须 COW

父子序列可以让 block table 指向同一组 prefix blocks，并用 refcount 管理生命周期。
已填满的共享尾块不再变化，后续 append 可以分配新块。

未填满的共享尾块不能原地写。实现必须先预留新物理块、复制已有 K/V、替换当前序列的尾块映射，
再 append 新 token。若容量不足，整次 append 应在改变旧 tensor 前失败。

用下面的 guided lab 观察真实 CPU K/V 值，而不只看 metadata：

[实验 7A：亲手追踪 Paged KV 与 copy-on-write](../practice/labs/lab-7a-paged-kv.md)

### “Paged KV”与“PagedAttention kernel”不是同一份证据

块分配器可以只管理块 ID、已使用位置和引用计数。张量存储层继续保存真实 K/V 数值。
GPU PagedAttention 内核则读取 block table，直接在不连续的物理块上完成注意力计算。

本仓库当前实验覆盖前两层，并使用稠密因果 GQA 参考实现核对数值。
实验中的注意力仍会收集完整序列并生成稠密分数矩阵，因此没有执行 CUDA PagedAttention 内核。

## Batching：GPU 忙不等于每个请求都快

Static batching 等一批请求一起完成。Continuous batching 在调度边界加入新请求并移除已完成请求，
减少因输出长度不同造成的空位。

Scheduler 仍需明确：

- Waiting 请求何时 admission；
- 一轮最多包含多少 sequence 和 token positions；
- Prefill 是否分块，能否与 decode 混合；
- Decode、prefill 和 priority 的先后关系；
- KV 不足时拒绝、等待、抢占还是交换；
- 完成和取消在哪个 boundary 释放资源。

### Chunked prefill

一个很长的 Prompt 若在单轮完成，可能让所有 decode 请求等待。分块 prefill 把输入拆成多轮，
让 decode 在中间获得调度机会。分块太小，则会增加调度次数和内核启动开销。

因此必须用混合长短输入的真实 workload 测量 p95/p99，而不是只跑固定长度的满 batch。

### Preemption 的账不能只记输出 token

KV 不足时，recompute preemption 会释放某条序列的 cache。它重新 admission 后，需要重跑已经处理的 context。

\[
W_{executed}=W_{logical}+W_{recomputed}.
\]

被抢占请求恢复后，不应再次向用户发送已经返回的 token，但 GPU 确实会重复部分计算。
因此要分开记录 API 用量、用户序列中的逻辑位置和 GPU 实际执行的位置。

端到端的 A/B 请求时间线见[请求生命周期](inference-request-lifecycle.md#request-b)。

## Prefix cache：命中首先是 identity 问题

System prompt、few-shot 或文档前缀相同时，可以复用已经计算的 prefill KV。
但安全命中不能只比较原始文本或 hash。

至少需要绑定：

- 可信 tenant、visibility/security domain 与 authorization policy revision；
- Model、tokenizer、chat template 和 adapter revision；
- RoPE/position config 与 KV dtype；
- Cached token ids 是请求 token ids 的逐项精确前缀。

Hash 只用于缩小候选集合，命中前仍需比较完整 identity 和 token tuple。
未加密 hash 也不能隐藏低熵 prompt。

Prefix cache 只减少可复用的 prefill。它不会减少后续 decode 的权重读取，也不保证命中率足够高。

## Quantization：文件更小不等于服务更快

### Weight-only quantization

若 (N) 个权重从 FP32 变为 4-bit，单看量化编码，理论存储可以缩小 8 倍。
真实格式还要保存每组缩放因子、可选零点、对齐填充、容器和索引，并保留未量化层。

Group 越小，量化尺度越能适应局部范围，但 metadata 和 kernel 处理开销更高。
最终速度取决于硬件是否有匹配的低位 kernel，以及瓶颈是否原本就在权重带宽。

仓库的 CPU 权重量化实验会实际执行整数编码、缩放、位打包、文件重载和反量化矩阵乘：

~~~powershell
python projects/inference-serving/quantization_toy.py `
  --seed 17 --bit-width 4 --group-size 8 `
  --output-features 16 --input-features 33 --batch-size 8
~~~

它最终仍使用 FP32 NumPy 矩阵乘，因此适合检查文件格式和量化误差。
低位 GPU 是否加速、常驻显存是否下降，要使用匹配的 GPU 内核和显存测量回答。

### Activation 与 KV quantization

Activation 分布和 outlier 会影响量化误差。KV quantization 还会把误差带入后续每一步 attention，
需要按长度、任务和 head/layer 切片评价。

若一个长度为 (D) 的 FP32 向量使用 INT8 code 加一个 FP32 scale，理想 payload ratio 是：

\[
\frac{4D}{D+4}.
\]

若原始是 BF16，则分子应为 (2D)。Allocator、alignment 和 workspace 尚未计入。

Weight RMSE 或单层 attention parity 都不能替代目标任务质量、长上下文和安全切片评测。

## Speculative decoding：先保证分布，再谈加速

Draft 模型先提出若干 token，target 模型并行验证。Sampling 版本要保持 target distribution，
不能简单理解为“大模型挑选小模型草稿”。

对同一 proposal token，draft 与 target 概率分别为 (q) 和 (p)：

\[
P(accept)=\min(1,p/q).
\]

第一次拒绝时，要从归一化后的正残差 ((p-q)_+) 中采样，并丢弃草稿中位于它后面的 token。
如果整段草稿全部接受，再从目标模型多计算出的下一个位置采样一个额外 token。

这里要求 (p) 和 (q) 使用相同的分词器、词表、已接受前缀和实际采样变换。
贪心投机解码采用另一套接受规则，不套用随机采样的残差分布。

是否加速取决于接受率、draft 成本、验证长度、batch 和 kernel。分布正确不等于 wall-clock 更快。

## Parallelism、kernel 与编译

### Tensor parallelism

Tensor parallelism 把一层矩阵或 attention heads 分到多个设备，并用 collective 合并结果。
它能解决单设备容量和计算问题，也引入通信、同步和更复杂的故障边界。

小模型或低 batch 下，通信可能抵消并行收益。单张消费级 GPU 的首选通常是先选合适模型、dtype 和 runtime，
而不是把多卡策略当作默认答案。

### Kernel fusion 与 CUDA Graph

内核融合减少中间数据读写和内核启动次数，但通常只支持特定形状、数据类型和硬件。
CUDA Graph 适合重复且形状较稳定的执行路径，因此常用于 decode；动态控制流和频繁变化的形状会降低适用性。

`torch.compile`、Triton、FlashAttention 和 CUDA Graph 解决的问题不同。
看到某个开关可用，不代表它位于当前瓶颈路径。

## 把诊断带回 Qwen3 与 nano-vLLM { #nano-vllm }

你当前的 [实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md) 正好把上面的分支变成四组消融：

| 实验对照 | 主要回答什么 | 应一起观察什么 |
|---|---|---|
| Eager / CUDA Graph | 符合条件的 decode 路径能否减少内核启动开销 | TPOT、执行路径、显存预留；prefill 仍走 eager |
| 精确共享前缀 / 一处 token 漂移 | 前缀身份怎样决定缓存命中 | cached tokens、物理 block、TTFT |
| 每轮 token 预算 256 / 1024 | 分块 prefill 怎样与 decode 分享调度轮次 | scheduled tokens、phase、TTFT 与 TPOT |
| 并发 1 / 2 / 4 / 8 | 批处理收益何时被排队和 KV 容量抵消 | 吞吐、队列、KV block 高水位、失败终态 |

这四组数据不要合成一个“哪个配置最快”的总排名。先回答源码机制是否按预期触发，
再判断它对当前工作负载的延迟、吞吐和显存产生了什么影响。

RTX 3070 Laptop 的功耗模式、显存、驱动、CUDA 和 Torch 版本都要写入报告。
桌面显卡或别人的 4070 结果只能作为问题线索，不能替代本机测量。实验生成的 JSON 先通过离线验证器，
再把具体数字写回教材。

## 单卡优化的推荐顺序

1. 固定模型、revision、tokenizer、sampling 和任务质量基线。
2. 固定输入/输出长度联合分布、arrival process、并发和 SLO。
3. 确认权重、dtype、context 上限和 KV 容量可以安全容纳目标请求。
4. 使用成熟 runtime 与兼容 kernel，记录版本和启动配置。
5. 分解 client queue、server queue、prefill、decode、网络和失败终态。
6. 调整 sequence/token budget，画吞吐与 TTFT/TPOT 的 Pareto curve。
7. 一次只引入 prefix cache、quantization 或 speculation 中的一项，并重新评测质量和失败路径。
8. 保存原始 attempt、server trace、GPU/KV 指标和可回滚配置。

## 怎样做一次可解释的 profiling

先用服务指标缩小范围：

- TTFT 高：先看 offered/dispatch queue 和 prompt 长度，再看 prefill。
- TPOT 高：看 decode batch、权重/KV 带宽、preemption 和 kernel。
- 吞吐低：看 GPU 是否有调度空洞、CPU/tokenizer 是否供不上、失败是否被排除在分母外。
- OOM：按权重、KV、workspace、graph 和其他进程拆显存，不从单一利用率猜测。

再用 PyTorch Profiler 或 Nsight 检查具体内核、GPU 显存流量和集合通信。
保存分析结果时，要同时记录模型、软件版本、张量形状、预热方式、功耗和工作负载；单独一张截图无法支持结论。

每次实验至少保存：

```text
假设：哪个资源是瓶颈
改动：只改变了什么
工作负载：长度、到达、并发、采样和时长
结果：质量、成功率、TTFT、TPOT、吞吐、显存和功耗
失败：OOM、429、timeout、protocol error 与取消
结论边界：结果只适用于哪些模型、硬件和版本
```

## 可运行入口

| 想验证的机制 | 入口 | 证据类型 |
|---|---|---|
| Online softmax | `projects/transformers-basics/online_softmax_demo.py` | 数学 recurrence |
| Continuous batching | `projects/inference-serving/continuous_batching_toy.py` | 离散 policy 状态机 |
| KV preemption | `projects/inference-serving/kv_preemption_batching_toy.py` | Logical/executed work 账本 |
| Paged block 与 COW | `projects/inference-serving/kv_block_allocator_toy.py` | Metadata 状态机 |
| 真实 CPU K/V tensor | `projects/inference-serving/paged_kv_tensor_toy.py` | Tensor 与 dense parity |
| Prefix identity | `projects/inference-serving/prefix_cache_toy.py` | Collision 与 lease 边界 |
| Weight quantization | `projects/inference-serving/quantization_toy.py` | Packing、reload 与误差 |
| KV quantization | `projects/inference-serving/kv_quantization_toy.py` | Code/scale 与 GQA 误差 |
| Speculative sampling | `projects/inference-serving/speculative_decoding_toy.py` | Acceptance/residual 概率 |

具体测试、固定样例数值和目前尚未验证的部分集中在
[推理服务证据页](../evidence/inference-serving-controls.md)。

## 自测与面试表达

1. 为什么提高 decode batch 可能同时提高吞吐并恶化延迟？
2. FlashAttention 与 Paged KV 分别减少什么？为什么前者不会自动减小持久 KV 容量？
3. 两条序列共享未满尾块时，容量检查为什么必须发生在 tensor mutation 之前？
4. 4-bit 权重为什么不等于端到端 8 倍加速？
5. Speculative sampling 在第一次拒绝时为什么不能直接从 target (p) 采样？
6. TTFT 退化但 TPOT 正常时，你会先检查哪三类证据？

面试中不要只列技术名词。更有说服力的结构是：

> 我先把指标拆成排队、prefill、decode 和容量四部分。确认瓶颈后，每次只改变一个变量，
> 同时回归质量、尾延迟和失败率。最终结论会绑定具体模型、硬件、软件版本和工作负载。

下一步进入[vLLM 与单卡服务](vllm-serving.md)，把这些机制放进一个真实部署和验收流程。
