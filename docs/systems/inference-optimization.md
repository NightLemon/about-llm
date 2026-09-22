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

1. **服务端 queue age 先增长**：降低准入上限，或调整每轮 sequence/token 预算；
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

若 \(I\) 很低，性能上限更容易受内存带宽限制；若 \(I\) 很高，则更容易接近计算吞吐上限。

小 batch decode 中，一次权重读取只服务少量序列，\(I\) 较低。增大 batch 能提高权重复用和总吞吐，
但请求需要等待更大的执行批次，TTFT 或单请求延迟可能上升。

在线服务真正要找的是“满足 SLO 时的持续吞吐”，不是脱离延迟和失败率的最大 batch。

## Attention：先区分计算量与数据搬运

朴素因果自注意力在 prefill 中会形成随长度约 \(O(L^2)\) 增长的分数区域。
FlashAttention 使用分块计算和在线 softmax，减少完整分数与概率矩阵在 GPU 高带宽显存中的反复读写。

它优化的是 IO 路径，不是把精确 attention 的一般计算量变成线性，也不会减少长期保存的标准 KV 容量。
在线 softmax 的递推和 CPU 对照实验见[数值计算](../foundations/attention-numerics.md)；是否命中目标 CUDA kernel，
仍要在目标 shape、dtype 与硬件上观察。

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
Paged KV 把物理 arena 切成固定 block，由每条序列的 block table 映射逻辑位置。诊断时分开 logical tokens、实际保存的
physical values 与 allocated slots；共享未满尾块继续写入前必须 copy-on-write。完整状态推导和真实 CPU K/V 观察放在
[实验 7A](../practice/labs/lab-7a-paged-kv.md)。该实验验证 allocator 与张量等价性，没有执行 GPU PagedAttention kernel。

## Batching：GPU 忙不等于每个请求都快

Continuous batching 在调度边界加入新请求、移除已完成请求；chunked prefill 把长 Prompt 拆开，让 decode 获得调度机会。
两者都在吞吐、TTFT 与 token 间延迟之间重新分配等待。KV 不足触发 recompute preemption 时，还要分开 logical work 与
GPU 重算的 executed work。完整时间线与分母见[请求生命周期](inference-request-lifecycle.md#request-b)；本页只把这些机制
作为“队列或 ITL 先恶化”时的候选解释，并要求用混合长短请求实测。

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

Weight、activation 与 KV quantization 改变的对象不同。文件缩小只有在目标 shape/dtype 命中低位 kernel，且瓶颈确实
位于相应带宽时，才可能转成服务收益；metadata、未量化层、allocator 与 workspace 仍要计数。公式口径见
[参考公式](../reference/formulas.md)，packing/reload 实验见下方入口。诊断结论必须同时报告 payload、峰值显存、
TTFT/TPOT 与任务质量，不能用单层误差替代端到端结果。

## Speculative decoding：先保证分布，再谈加速

Draft 模型先提出若干 token，target 模型并行验证。Sampling 版本要保持 target distribution，
不能简单理解为“大模型挑选小模型草稿”。

对同一 proposal token，draft 与 target 概率分别为 \(q\) 和 \(p\)：

\[
P(accept)=\min(1,p/q).
\]

第一次拒绝时要从归一化后的正残差 \((p-q)_+\) 采样；整段接受时再从目标模型的下一位置采样。这里要求 draft 与
target 使用同一 tokenizer、词表、已接受前缀和采样变换。分布正确仍不等于更快：接受率、draft 成本、验证长度、
batch 与 kernel 都要进入同一 workload 的 TTFT、TPOT、E2E、质量和失败分母。

## Parallelism、kernel 与编译

Tensor parallelism 用通信换单设备容量；kernel fusion 减少中间读写和 launch；CUDA Graph 复用形状较稳定的执行路径。
三者处理不同瓶颈，也都可能因小 batch、动态 shape 或通信成本而退化。多卡通信见
[集合通信](collective-communication-network.md)，算子与编译层次见[算子计算栈](operator-stack.md)。

## 把诊断带回 Qwen3 与 nano-vLLM { #nano-vllm }

你当前的 [实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md) 正好把上面的分支变成四组消融：

| 实验对照 | 主要回答什么 | 应一起观察什么 |
|---|---|---|
| Eager / CUDA Graph | 符合条件的 decode 路径能否减少内核启动开销 | TPOT、执行路径、显存预留；prefill 仍走 eager |
| 精确共享前缀 / 一处 token 漂移 | 前缀身份怎样决定缓存命中 | cached tokens、物理 block、TTFT |
| 每轮 token 预算 256 / 1024 | 分块 prefill 怎样与 decode 分享调度轮次 | scheduled tokens、phase、TTFT 与 TPOT |
| 并发 1 / 2 / 4 / 8 | 引擎内 batching 收益何时被等待中的 sequence 和 KV 容量抵消 | 引擎 TTFT/TPOT、KV block 高水位、case 终态 |

这四组数据不要合成一个“哪个配置最快”的总排名。它们先回答源码机制是否按预期触发，
再判断固定 synthetic token 输入在引擎内的延迟、吞吐和显存怎样变化。7B 的 `WAITING` sequence 与
engine TTFT 不包含 HTTP 接收前的客户端等待，也没有给出在线服务的 queue age。

要选择实际单卡部署的配置，还要把候选配置带到 OpenAI-compatible endpoint：固定 prompt/output 长度联合分布、
到达过程、超时和客户端并发，保存每个 attempt 的 `offered_at`、dispatch、首个内容和终态。只有这样才能把
引擎内的机制差异接回用户可见 TTFT、服务端队列与完整失败分母；这一步的具体时钟和报告口径见
[vLLM 服务](vllm-serving.md#client-timestamps)。

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
