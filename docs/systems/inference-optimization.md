# LLM 推理优化：从算子到 KV Cache

LLM 推理不是一个统一工作负载。prefill 对整段 prompt 并行计算，通常更偏计算；decode 每步只产生一个 token、反复读取权重和 KV cache，通常更偏内存带宽。优化前先分阶段测量，否则“平均 tokens/s”会掩盖瓶颈。

## 两个阶段

对长度 \(L_p\) 的 prompt 和 \(L_o\) 个输出 token：

- prefill：一次处理 \(L_p\)，构建每层 KV，决定 TTFT 的大部分；
- decode：循环 \(L_o\) 次，每次读取已有 KV，决定 TPOT/ITL。

长 prompt 可能让 prefill 成为主导；短 prompt 高并发服务通常 decode 和调度更关键。必须按输入/输出长度桶报告。

## Roofline 直觉

算术强度 = FLOPs / 读取字节。若算术强度低，性能受内存带宽限制；高则受算力限制。decode 的小 batch 每 token 都读大部分权重，算术强度低；提高 batch 能让一次权重读取服务多个序列，增加吞吐，但排队与单请求延迟上升。

因此没有“最大 batch 最优”：在线服务选择满足 SLO 下的最大持续吞吐。

## Attention 复杂度

训练/朴素 prefill self-attention 的 score 矩阵随序列长度约为 \(O(L^2)\)，KV/投影计算另计。FlashAttention 通过分块和在线 softmax 减少 HBM 往返，不改变精确 attention 数学结果（浮点顺序会有微差），也没有把理论计算量普遍变成线性。

decode 使用 KV cache 后不再重算历史 K/V，每层每步 attention 对历史长度近似 \(O(L)\)。没有 KV cache 会重复前缀计算，成本巨大。

## KV Cache 容量

标准 MHA 每 token 的 KV 字节近似：

\[
M_{KV/token}=2\times n_{layers}\times n_{kv\_heads}\times d_{head}\times bytes(dtype)
\]

总量再乘 active sequences 的已缓存 token。GQA/MQA 通过减少 `n_kv_heads` 显著降低 cache，但 query heads 可保持更多。模型参数相近不代表长上下文容量相同。

例：32 层、8 KV heads、head dim 128、BF16，则每 token 为 `2*32*8*128*2 = 128 KiB`；单序列 8192 tokens 的理想化 dense KV 存储为 1 GiB。该等式已由 `estimate_kv_cache_bytes` 单测验证，但不包含 block metadata、allocator、对齐和 workspace，也不适用于 MLA 等不同缓存布局。

## KV 管理

传统每请求预分配最大连续空间会碎片化并浪费。PagedAttention 将 KV 分块，用逻辑到物理 block table 映射，像虚拟内存一样按需分配；共享前缀可引用相同 blocks，beam/fork 可 copy-on-write。

block 太小 metadata/调度开销高，太大内部碎片多。实际选择由 runtime 决定。监控 KV 使用率、block fragmentation、eviction/recompute 和因 KV 不足拒绝的请求。

## Prefix caching

system prompt、few-shot 或共享文档前缀相同，可缓存 prefill KV。key 必须包含模型/adapter、token ids、position/rope 设置、dtype 和影响 attention 的配置。只对原始文本 hash 不够：tokenizer/template 变化会改变 token。

多租户场景避免通过 cache timing 泄露前缀存在性，敏感前缀不要跨安全域共享。缓存只减少 prefill，不减少后续 decode 权重读取。

## Batching

Static batching 等整批完成，短请求被长请求拖累。Continuous batching 在每个调度迭代加入新请求、移除完成请求，提高 GPU 利用率。调度器在 prefill chunk、decode token、优先级和 KV 容量间取舍。

Chunked prefill 把长 prompt 分段，避免一次 prefill 阻塞所有 decode；过度切分增加 kernel launch 和调度开销。要用混合长短 workload 测 p95，而不是单一 prompt。

## Quantization

### Weight-only

INT8/INT4 权重量化减少存储和带宽，decode 常受益明显。group-wise scale 提高精度但增加 metadata/dequant。GPTQ/AWQ 等 post-training 方法使用校准数据减少关键权重误差；校准分布应接近任务。

### Activation/KV

权重+激活 FP8/INT8 可利用特定硬件 tensor cores，但对校准和 kernel 要求更高。KV cache 量化显著增加长上下文容量，可能影响 attention/长程召回。必须分长度和任务评测。

### 不能只看 perplexity

量化误差可能集中在代码、数学、稀有 token、长上下文或工具 JSON。比较 base/quantized 的任务质量、格式、安全、首 token 和吞吐；模型文件变小不等于实际服务更快，kernel 不匹配可能反而慢。

## Speculative decoding

小 draft model 一次提出多个 token，大 target model 并行验证并接受连续匹配前缀。输出分布可保持与 target sampling 一致（正确实现下），加速取决于接受率和验证成本。

draft 太弱接受率低，太强自身昂贵。模型/任务、temperature、prompt 长度影响接受率。self-speculative、Medusa/多 token head 等方案改变 draft 来源，但都要报告 accepted tokens、target calls 和端到端 TPOT。

## Parallelism

- Tensor parallel：单层矩阵跨设备切分，需频繁 collective，适合模型单卡放不下。
- Pipeline parallel：层分段，存在 pipeline bubble，在线小 batch 延迟复杂。
- Data parallel：每卡完整副本，不同请求；吞吐扩展简单但模型需单卡容纳。
- Expert parallel：MoE experts 分布，token routing/all-to-all 是关键瓶颈。

单张消费 GPU 优先量化和容量管理，不需要为“分布式完整性”引入多卡复杂度。云 API 用户关注 provider 限流、batch API 和 token 成本。

## Kernel 与编译

融合 RMSNorm、RoPE、MLP、sampling 可减少 launch/HBM 往返。`torch.compile`、CUDA graphs 可降低 Python/launch 开销，但 shape 动态、模型分支和版本会引发 graph break/重新编译。预热和基准要排除首次编译。

使用优化 kernel 前验证 dtype、head dim、mask、滑窗/ALiBi/RoPE 和硬件支持。fallback 路径可能静默变慢；记录实际 backend，而不只记录配置名。

## Sampling 成本

greedy、top-k/top-p、temperature 在 logits 上执行。大词表排序可优化为局部选择。logprobs、多个候选、beam search 会增加内存和计算，且可能限制 batching。API 是否请求 logprobs 应进入 workload。

停止条件处理 EOS、stop strings、max tokens 和结构 grammar。stop string 可能跨 token 边界；在文本层截断要防 UTF-8/结构损坏。

## 单卡优化顺序

1. 固定模型、revision、workload 与质量基线；
2. 确认权重/dtype 和目标 context 能容纳；
3. 使用成熟 runtime 和兼容 attention kernel；
4. 调 max_num_seqs/token budget，画吞吐-延迟曲线；
5. 启用 prefix cache/quantization，并逐项重新评测；
6. 只有 decode 明确受限再尝试 speculative；
7. 用真实长度/并发压测并保留回滚配置。

## Profiling

从服务指标定位：TTFT 高看队列/prefill，TPOT 高看 batch、权重带宽、KV、kernel；吞吐低看 GPU utilization、调度空洞和 CPU/tokenizer。再用 PyTorch profiler/Nsight 看 kernel、collective 和 HBM，不从 profiler 截图猜结论。

基准记录 GPU、driver/CUDA/runtime、功耗、模型 revision、量化、context、输入/输出长度分布、并发、warmup、采样和测试时长。

## 面试追问

**为何 decode 常是 memory-bound？** 每步 FLOPs 相对有限却要读取大量权重/KV；小 batch 权重复用低。batch 增大提高算术强度和吞吐，但增加排队/延迟。

**GQA 如何降低显存？** 多个 query heads 共享更少的 K/V heads，KV cache 按 kv heads 而非 query heads 增长；可能有质量权衡，但现代模型常采用。

**FlashAttention 会减少 KV cache 吗？** 它主要优化 attention 计算中的 IO/中间矩阵；标准 decode KV 的持久容量由层、KV heads、head dim、长度和 dtype 决定，不能混为一谈。

**量化为何可能不加速？** dequant 开销、kernel/硬件不支持、小 batch、其他瓶颈或格式转换会抵消带宽收益；必须测端到端而不是只看权重字节。
