# LLM 推理优化：从算子到 KV Cache

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：推理性能、kernel、KV 和量化工程师。
- **先修**：[推理基础](inference.md)、attention 复杂度和显存单位。
- **首次阅读**：两阶段 → Roofline → attention/KV 容量 → KV 管理 → profiling。
- **完成信号**：能先定位 compute、memory 或 scheduler 瓶颈，再选择优化。
- **卡住时**：回到[硬件性能模型](hardware-edge.md)的单位与容量账本。

</div>

本页属于进阶性能分析。尚不能区分 prefill、decode、TTFT、TPOT 和 KV Cache 时，先完成[推理基础](inference.md)，再按实际瓶颈选择本页小节。

LLM 推理不是一个统一工作负载。prefill 对整段 prompt 并行计算，通常更偏计算；decode 每步只产生一个 token、反复读取权重和 KV cache，通常更偏内存带宽。优化前先分阶段测量，否则“平均 tokens/s”会掩盖瓶颈。

## 两个阶段

对长度 \(L_p\) 的 prompt 和 \(L_o\ge1\) 个输出 token，在标准 decoder-only causal generation、无 prefix reuse/speculation/beam 的口径下：

- prefill：一次处理 \(L_p\) 个位置、构建每层 KV；最后一个 prompt position 的 logits 已用于采样首个输出 token，决定 TTFT 的大部分；
- decode forward：后续循环 \(L_o-1\) 次，每次处理上一个输出 token、读取已有 KV 并产生下一个 token，决定 TPOT/ITL。

因此该简化下 forward positions 是 \(L_p+L_o-1\)，不是机械的 \(L_p+L_o\)。API usage/计费仍可分别报告 prompt 与 completion token；有的 runtime 也会在指标命名上把“首个生成 token”归入 decode 阶段。讨论性能时必须先写清统计的是输出 token、模型输入位置、padded batch slots 还是 kernel work，不能靠阶段名称猜分母。

长 prompt 可能让 prefill 成为主导；短 prompt 高并发服务通常 decode 和调度更关键。必须按输入/输出长度桶报告。

## Roofline 直觉

算术强度 = FLOPs / 读取字节。若算术强度低，性能受内存带宽限制；高则受算力限制。decode 的小 batch 每 token 都读大部分权重，算术强度低；提高 batch 能让一次权重读取服务多个序列，增加吞吐，但排队与单请求延迟上升。

因此没有“最大 batch 最优”：在线服务选择满足 SLO 下的最大持续吞吐。

## Attention 复杂度

训练/朴素 prefill self-attention 的 score 矩阵随序列长度约为 \(O(L^2)\)，KV/投影计算另计。FlashAttention 通过分块和在线 softmax 减少 HBM 往返，不改变精确 attention 数学结果（浮点顺序会有微差），也没有把理论计算量普遍变成线性。

仓库提供一个不依赖 CUDA 的可运行 recurrence oracle：

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
~~~

固定 fixture 的 dense score 有 35 个元素，key block size 为 3 时最大 logical score tile 有 15 个元素，三个 block 的最终输出与 dense reference 在 float64 容差内一致。这里的 15 是按 shape 推导的单个逻辑 tile 上界，不是 Python RSS、allocator peak、HBM bytes 或 kernel workspace 实测；脚本还为了对照而单独运行了会物化 dense score/probability 的 reference。该结果不表示执行了 FlashAttention、CUDA、vLLM，也不能推出 latency 或 throughput。

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

### Block table、共享与 COW

设固定 block 容纳 \(S\) 个 token。长度 \(T\) 的单序列需要 \(\lceil T/S\rceil\) 个逻辑 block reference，最后一块浪费 \((S-T\bmod S)\bmod S\) 个 slot；多序列总碎片不能简单用总 token 再除一次，因为每条序列有独立 tail，共享和 COW 还会改变物理副本。

Prefix fork 可让父子序列的 block table 指向相同 physical blocks，并增加 refcount。已填满的共享 tail 后续 append 只需分配新块，不改旧块；未填满的共享 tail 若直接写入，会污染其他序列，因此必须先 copy-on-write：预留新物理块、复制已有 prefix KV、替换本序列 tail reference，再追加 token。若没有 free block，整个 append 应在填充旧 tail **之前**原子失败，否则会出现“长度没提交但 KV 已改变”的非法状态。

账本至少分开：

- `logical_tokens`：各序列长度之和，共享 prefix 会重复计数；
- `physical_token_values`：物理 block 中实际 materialized 的 token positions，共享只算一次、COW 副本另算一次；
- `logical_block_references` 与 allocated physical blocks：两者之差是当前 sharing saved blocks；
- `allocated_token_slots = allocated_blocks * block_size`；
- `internal_fragmentation_slots = allocated_token_slots - physical_token_values`。

仓库 metadata-only reference 可运行以下确定性场景：

~~~powershell
python projects/inference-serving/kv_block_allocator_toy.py
~~~

它先给 6-token 序列分配 `4+2` slots，再 fork；父序列 append 触发 partial-tail COW，子序列独占旧 tail 后可原地填满。三块耗尽时，父序列一次需要“填 1 个 tail slot + 新增 1 块”的 append 被整体拒绝，block/length/report 均保持不变；release 后 refcount 为零的 block 返回 free pool。

这个实验只维护 block id、occupancy 和 refcount，没有存储或复制真实 K/V tensor；也没有 page table kernel、CUDA 并发、swap/preemption、prefix hash/tenant policy、对齐 metadata bytes 或连续批调度。因此它证明状态机和计数口径，不证明 vLLM/PagedAttention 实现、显存节省、延迟或吞吐。

## Prefix caching

system prompt、few-shot 或共享文档前缀相同，可缓存 prefill KV。安全命中不是“文本 hash 相同”，而是：由可信认证层给出的 tenant 与 visibility/security domain 相同，authorization/policy revision 相同，model/tokenizer/chat-template/adapter revision 相同，RoPE/position config 与 KV dtype 相同，并且缓存 token ids 是请求 token ids 的**逐项精确前缀**。在所有合格项中才选择最长前缀。原始文本相同仍可能因 tokenizer/template 变化得到不同 token；token 相同也不能越过权限域复用。

仓库 `PrefixCache` 是 bounded metadata oracle：用 fingerprint 缩小候选桶，但命中前继续比较完整 identity 与 token tuple；测试把 fingerprint 强制为同一个字符串，仍拒绝跨 tenant 和错 token。Lease/reference count 会 pin 正在使用的 entry，LRU 只淘汰未 leased 项；满容量且全部 leased 时插入在状态改变前失败。运行：

~~~powershell
python projects/inference-serving/prefix_cache_toy.py
~~~

SHA-256 不是授权、真实性或保密边界：未加密 hash 和 token ids 对低熵 prompt 可被字典猜测。真实系统还要从可信服务端状态构造 identity，定义 TTL/删除/加密与共享策略，并缓解 hit/miss timing side channel。该 oracle 不存 K/V tensor，不测 VRAM、命中率、prefill savings、延迟或吞吐，也不等价于任一 vLLM 版本。Prefix cache 只减少可复用 prefill，不减少后续 decode 的权重读取。

## Batching

Static batching 等整批完成，短请求被长请求拖累。Continuous batching 在每个调度迭代加入新请求、移除完成请求，提高 GPU 利用率。调度器在 prefill chunk、decode token、优先级和 KV 容量间取舍。

Chunked prefill 把长 prompt 分段，避免一次 prefill 阻塞所有 decode；过度切分增加 kernel launch 和调度开销。要用混合长短 workload 测 p95，而不是单一 prompt。

KV 不足时，recompute preemption 会释放某条 sequence 的 cache；重新 admission 后必须重跑已经处理过的 context 才能继续。于是 `prompt + output - request_count` 只是无复用/无 speculation 下的 logical causal positions，实际执行还要加 recomputed positions。运行：

~~~powershell
python projects/inference-serving/kv_preemption_batching_toy.py
~~~

固定 metadata-only fixture 把 9 个 logical positions 与 2 个 rebuild positions 分开，实际执行 11；被抢占请求不会在 rebuild 时重复输出 token。它只验证当前 decode-first、stable-FCFS 单向抢占与 lower-priority-youngest victim policy 和 block 账本，不存 K/V、不模拟 swap/prefix cache，也不证明 vLLM 版本行为、GPU VRAM、延迟或吞吐。

## Quantization

### Weight-only

INT8/INT4 权重量化减少存储和带宽，decode 常受益明显。group-wise scale 提高精度但增加 metadata/dequant。GPTQ/AWQ 等 post-training 方法使用校准数据减少关键权重误差；校准分布应接近任务。

最透明的基线是 contiguous row group 的 symmetric absmax quantization。设一个 group 的 bit width 为 \(b\)，并采用对称码域

\[
q_{max}=2^{b-1}-1,\qquad q\in[-q_{max},q_{max}]
\]

则

\[
s_g=\frac{\max_{i\in g}|w_i|}{q_{max}},\qquad
q_i=\operatorname{clip}(\operatorname{round}(w_i/s_g),-q_{max},q_{max}),\qquad
\hat w_i=s_gq_i
\]

全零 group 可约定 \(s_g=1\)，量化码仍全为零。4-bit 时这个约定使用 `[-7, 7]`，故意不用二补码的 `-8`；其他 runtime 可能使用 `[-8, 7]`、asymmetric zero-point、per-channel、非均匀 codebook 或不同 rounding，格式名称相同也不能假定数值契约相同。

对 \(R\times C\) 的 FP32 矩阵、contiguous group size \(G\)，若每个 group 保存一个 FP32 scale，理想 dense-bitstream 账本是

\[
M_{ideal}=\left\lceil\frac{bRC}{8}\right\rceil
  +4R\left\lceil\frac{C}{G}\right\rceil\ \text{bytes}
\]

它仍没包含 row/group alignment、tensor header、zero point、kernel-specific packing、未量化层或 workspace。`int8` NumPy 数组即使只存 `[-7, 7]` 也仍占每元素 1 byte，不能把它的 `nbytes` 当真实 int4 packing。

要把这个下界变成真实 byte stream，还必须固定 signed-code 映射和 bit order。仓库格式令

\[
u_i=q_i+q_{max}\in[0,2^b-2]
\]

因此全 1 的 unsigned code `2^b-1` 保留为非法值。按 C row-major 展平后，每个 (u_i) 的低 (b) 位依次写入 byte 的 low-to-high bit position；最后一个 byte 未使用的高位必须为 0。这个 offset-binary、LSB-first 约定与二补码或某个 GPU kernel 的 nibble/interleave layout 都不等价，读取端不能只凭“int4”猜格式。

仓库 reference 可直接观察 code、scale、反量化误差和线性层输出误差：

~~~powershell
python projects/inference-serving/quantization_toy.py `
  --bit-width 4 --group-size 8 `
  --output-features 16 --input-features 33 --batch-size 8
~~~

输出同时列 `reference_fp32_weight_bytes`、理想 packed weight、实际 dense packed code bytes、FP32 scale metadata、raw code+scale payload 和 NumPy unpacked reference；还保存完整 packed hex、SHA-256、padding bit 数与 code/scale exact round-trip。3×5、4-bit fixture 的 15 个 code 实际占 8 bytes，最后 4 bit 是规范零 padding；加 6 个 FP32 scale 后 raw payload 为 32 bytes，恰好等于该无 header/alignment 格式的账本。

`PackedGroupwiseQuantizedMatrix.to_bytes()` 进一步写出单矩阵 v1 artifact：固定 32-byte little-endian header 保存 magic、version、bit width、mapping/dtype id、shape、group size、code length 和 scale count；随后是 packed code、little-endian FP32 scales，以及对前述 header+payload 的 32-byte SHA-256。Loader 要求 exact length，拒绝未知 version/mapping/dtype、维度与长度不一致、trailing bytes、摘要漂移、非法 code、非零 padding 和非有限/非正 scale。固定 3×5 fixture 的 raw payload 是 32 bytes，完整 tensor artifact 是 96 bytes；这正说明小 tensor 上 header/integrity 开销不能忽略。

SHA-256 没有密钥，只能检测相对可信副本的意外/局部篡改，不能认证作者或来源。该单矩阵格式没有 tensor name、多个 layer、未量化参数、tokenizer/config、分片索引或签名，所以不是 self-contained model artifact。`quantized_linear` 仍先严格 reload/unpack/dequantize 再调用 FP32 NumPy matmul，没有 fused low-bit kernel、calibration/GPTQ/AWQ 或目标模型质量、显存、延迟/吞吐证据。Weight RMSE 小不保证 logits、生成或关键安全切片误差小；输入激活分布、层间放大和 outlier 都会改变最终影响。

#### 多矩阵教学 bundle

单 tensor 格式之上，`QuantizedMatrixBundle` 用 name-sorted manifest 打包多个完整 v1 矩阵 artifact，并记录 model/tokenizer revision 和 architecture config identity。v1 外层是 24-byte little-endian header、canonical UTF-8 JSON manifest、连续 tensor payload 与 32-byte outer SHA-256；每个 descriptor 绑定 name、格式版本、连续 offset、length 和 tensor SHA-256。严格 loader 拒绝 duplicate/unknown JSON key、非 canonical JSON、name/offset/order/digest 漂移、截断、尾随数据和配置上限外的文件、manifest、tensor 数或 tensor 大小。

~~~powershell
python projects/inference-serving/quantized_bundle_toy.py
~~~

默认 two-layer NumPy MLP fixture 的 FP32 权重共 288 bytes，两个 raw quantized payload 共 124 bytes，两个内层 tensor artifact 共 252 bytes；外层 bundle 为 987 bytes，其中相对内层 artifact 的 container overhead 是 735 bytes。严格重载后 quantized forward 与原 bundle 的 quantized forward 按 byte 精确相同；相对 FP32 output 的 RMSE 约 0.0368725、relative L2 约 0.123272。这些数字只属于固定 seed=29、4-bit、group-size=4 fixture，主要说明“小对象上 manifest/integrity 开销可远大于低位 payload”，不是模型压缩率或质量结论。

这个 bundle 仍只接受二维量化矩阵：tokenizer 只有 identity、没有 vocab/merges/chat template payload；bias、norm、embedding 等未量化 state 不受支持，也不含 model forward、shard index 或 runtime-specific GGUF/safetensors/kernel layout。因此它是多 tensor 序列化与重载 control，不是完整 LLM checkpoint。外层和内层 unkeyed SHA-256 都不认证来源。`write_new` 的 exclusive create 防止静默覆盖并对文件执行 flush/fsync，但中途崩溃仍可能留下 partial target，且没有证明 parent-directory fsync 或断电原子发布；生产导出应写临时文件、校验后原子 rename，并使用签名/可信元数据和真实 runtime loader。

#### Repo-native MiniGPT inference checkpoint

要跨过“只有二维矩阵”的边界，仓库另提供只面向当前 `MiniGPT` architecture revision 的完整推理 checkpoint：它保存 Byte-BPE merge payload、严格 GPT config、所有唯一模型参数、量化配置与 tied `lm_head.weight → token_embedding.weight` 契约。二维 embedding/linear weights 使用前述单矩阵 artifact；LayerNorm 与启用后的 linear bias 作为 little-endian FP32 vector 保存。Causal mask 由受信任的架构 loader 按 config 重建，tied LM head 不重复存储。

~~~powershell
python projects/inference-serving/minigpt_checkpoint_toy.py

# 可选：只创建新文件
python projects/inference-serving/minigpt_checkpoint_toy.py `
  --artifact-path .\minigpt.allmgpt
~~~

默认 seed=7 fixture 是 vocab 258、context 8、width 8、1 layer、2 heads、bias=true 的随机 tiny decoder。它有 16 个唯一参数、FP32 参数共 10,976 bytes；checkpoint 是 `24-byte header + 3,904-byte manifest + 4,760-byte parameter payload + 32-byte outer digest = 8,720 bytes`。`abc abc` 的实际 BPE ids 是 `[257,32,257]`；严格重载后 logits shape 为 `[1,3,258]`，重复读取同一 artifact 的 logits bit-exact，相对原 FP32 随机模型的 logit RMSE 约 0.00477277。这个单 prompt/seed 误差只验证端到端 plumbing，不是语言质量结论。

Loader 在构造 PyTorch model 前验证 artifact/manifest/parameter/tokenizer 数量、每个参数 byte、模型总参数量、canonical JSON、完整 name/shape/kind/order/offset/length/digest、architecture revision、tokenizer vocab 与 tied contract；随后把低位矩阵反量化为 FP32 参数并执行仓库 `MiniGPT.forward`。因此 artifact bytes 小于 FP32 参数 bytes **不等于 resident memory 更小或执行低位 kernel**。

这里的“完整”严格限定为“可由本仓库固定 revision loader 恢复推理所需 config、Byte-BPE payload 和全部 MiniGPT 参数”。Forward 源码没有嵌入 artifact，格式也不支持 normalization、special token、chat template、optimizer、RNG、训练 resume、shard/device map、GGUF、safetensors、Transformers、vLLM 或任意 Llama/Qwen checkpoint。Architecture revision 是人工维护的兼容契约，不是源码证明；unkeyed SHA-256 仍不认证来源。文件写入的 crash/durability 边界与上面的 bundle 相同。

#### 固定 Qwen 的真实单矩阵 INT4 control

仓库进一步把同一 packed reference 用到已验证的 Qwen2.5-0.5B-Instruct 权重，但只量化第一层 `o_proj.weight`，不声称生成完整 checkpoint。矩阵 `[896,896]` 含 802,816 个参数；group size 128、码域 `[-7,7]` 时，3,211,264-byte FP32 weight 变成 401,408-byte codes + 25,088-byte FP32 scales，含 strict bundle framing 后是 427,328 bytes，即该 selected matrix 为 7.514752134192002×。

~~~powershell
python projects/transformers-basics/run_qwen_weight_quantization_control.py `
  --local-files-only
python projects/transformers-basics/run_qwen_weight_quantization_control.py `
  --verify projects/transformers-basics/target-checkpoints/qwen2.5-0.5b-instruct.weight-int4.recorded-report.json
~~~

Control 捕获真实 `[1,31,896]` activation，重载 artifact 后执行 dequantized linear，并以仅替换这一个矩阵的模型再跑完整 forward。Weight/selected-output/last-logits relative-L2 分别为 0.1323337087/0.0700015308/0.0851380718；last-logits max-abs 是 1.6255179644，尽管这个 prompt 的 argmax 仍为 17。Argmax 单点相同不能覆盖其余 151,935 个 logits，更不能推断另一 prompt、autoregressive rollout 或任务质量。

该实验的关键分账是：427,328-byte **selected-weight artifact** 不等于整模型文件，artifact compression 不等于 resident/peak memory，反量化 FP32 matmul 不等于 INT4 kernel，单 prompt logits 不等于 quality gate。其余模型权重仍为 FP32；没有 NF4/GPTQ/AWQ/SmoothQuant、校准集、完整量化 loader、generation、GPU/CUDA/vLLM 或性能测量。

### Activation/KV

权重+激活 FP8/INT8 可利用特定硬件 tensor cores，但对校准和 kernel 要求更高。KV cache 量化显著增加长上下文容量，可能影响 attention/长程召回。必须分长度和任务评测。

一个透明的 INT8 KV baseline 是：对每个 `[batch, kv_head, token, :]` 的 K 向量和 V 向量分别取 absmax scale，使用 `[-127,127]`，全零向量令 scale=1，并保留 `-128` 为非法码。它不同于 per-channel、per-block、动态 amax/FP8 或 runtime-specific paged layout；K/V scale 也不能误共用，因为两者分布和 head dimension 可能不同。

若 K/V head dimension 都是 (D)，batch 为 (B)、KV head 数为 (H_{kv})、缓存长度为 (T)，FP32 裸 K/V 是

\[
M_{fp32}=8BH_{kv}TD.
\]

INT8 code 加 K/V 各一个 per-token/per-KV-head FP32 scale 是

\[
M_{int8}=2BH_{kv}TD+8BH_{kv}T=2BH_{kv}T(D+4),
\qquad
\rho=\frac{M_{fp32}}{M_{int8}}=\frac{4D}{D+4}.
\]

因此它只在 (D\to\infty) 时接近 4×；还没计 block alignment、allocator、tensor/container header、workspace 或临时 dequant buffer。GQA 的容量收益来自 (H_{kv}<H_q)，不能把 query head 数代入 K/V 存储公式。

仓库 `QuantizedKVCache` 会真实物化 INT8 codes/FP32 scales，并将反量化 K/V 送入已有 GQA+causal attention oracle。K 误差会改变 logits/softmax，V 误差会改变加权输出，所以 toy 分开报告 K、V、attention probability 和 output RMSE：

~~~powershell
python projects/inference-serving/kv_quantization_toy.py `
  --query-heads 4 --key-value-heads 2 `
  --cached-tokens 8 --query-tokens 3 `
  --key-head-dim 16 --value-head-dim 16
~~~

测试验证 absmax 误差不超过对应 scale/2（加浮点容差）、causal future probability 为 0、GQA 与显式 dequant 一致，以及逐 token prefix cache 与整段量化 causal attention 在 FP32 容差内一致。这仍是 CPU 上“先 dequantize、再 float32 attention”，没有在 INT8 code 上直接算 dot product，没有 paged KV kernel、真实 allocator resident bytes、目标模型长上下文质量或速度证据。

### 不能只看 perplexity

量化误差可能集中在代码、数学、稀有 token、长上下文或工具 JSON。比较 base/quantized 的任务质量、格式、安全、首 token 和吞吐；模型文件变小不等于实际服务更快，kernel 不匹配可能反而慢。

## Speculative decoding

小 draft model 一次提出多个 token，大 target model 并行验证并接受连续前缀。对于同一上下文上的 draft 分布 \(q\) 与 target sampling 分布 \(p\)，一步精确 rejection-sampling 规则是：先采样 \(x\sim q\)，再以

\[
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
\]

接受。因为 \(x\) 确实来自 \(q\)，被抽到的 token 必有 \(q(x)>0\)，比值在该事件上有定义。若拒绝，则不能简单改取 target argmax 或重新从完整 \(p\) 采样，而要从 residual distribution

\[
r(i)=\frac{(p(i)-q(i))_+}{\sum_j(p(j)-q(j))_+}
\]

采样一个修正 token。接受路径输出 token \(i\) 的概率质量是

\[
q(i)\min(1,p(i)/q(i))=\min(q(i),p(i)).
\]

总拒绝概率满足

\[
\beta=1-\sum_i\min(p(i),q(i))
=\frac12\sum_i|p(i)-q(i)|
=\sum_i(p(i)-q(i))_+.
\]

所以最终边际为 \(\min(p(i),q(i))+\beta r(i)=p(i)\)。一步接受率正好是 \(1-\operatorname{TV}(p,q)\)；这条等式不代表多 token block 的接受数可以用独立同分布乘积估算，因为每个位置的分布依赖上下文。

对长度 \(\gamma\) 的 block，target verification 必须为各 proposal position 和额外下一位置提供概率。按顺序检查：接受则继续；第一个拒绝位置发出 residual token，并丢弃它后面的 draft token；若 \(\gamma\) 个都接受，再从额外 target distribution 采一个 bonus token。每个位置的 \(p_i,q_i\) 必须对应相同 tokenizer/vocabulary、相同已接受前缀以及实际 draft proposal 条件，并在 temperature、top-k/top-p 等采样变换后使用真实 proposal/target 概率。否则上面的恒等式没有覆盖实际实现。

仓库提供概率级 CPU oracle：

~~~powershell
python projects/inference-serving/speculative_decoding_toy.py `
  --seed 23 --trials 20000
~~~

固定 authored vector 的解析结果精确恢复 target，并显示 acceptance=`0.6`、TV/rejection=`0.4`；Monte Carlo 只作直觉展示，不是恒等式证据。另一个 fixture 强制 block 的第二个 token 拒绝，验证后续 proposal 被丢弃且不发 bonus。该实验没有模型 forward、tokenizer、KV cache、target verification kernel 或 GPU timing，因此不证明任何模型实现保持分布，也不证明加速。

draft 太弱接受率低，太强自身昂贵。模型/任务、temperature、prompt 长度影响接受率。self-speculative、Medusa/多 token head 等方案改变 draft 来源，但都要报告 accepted tokens、target calls 和端到端 TPOT。

Greedy speculative decoding 是另一份契约：target 逐位置确认与自身 greedy token 一致的 draft prefix，目标是保持 target greedy 输出，不需要把它描述成 sampling residual algorithm。反过来，“大模型给小模型草稿打分后挑一个”也不自动满足上述 rejection rule。

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

Beam width (B) 不只是最后多排几个字符串：每步需要为 active prefixes 计算/筛选候选，保留每条分叉的序列状态和 KV block table；共享 prefix 可以引用相同 full blocks，但分叉后各自增长，shared partial tail 追加还要 copy-on-write。实际成本依 candidate-selection kernel、finished-candidate cap、early stopping、输出长度和 batching policy，不应写成固定 (B\) 倍。比较 beam 配置时同时报告 returned sequence 数与内部 beam 数，不能把只返回 1 条误认为只执行 1 条路径。

Grammar/JSON constrained decoding 需要为每条 active sequence 保存 parser state，并把 tokenizer token 对该状态的完整转移变成 mask。逐 token 解释字符串或遍历全词表可能成为 CPU bottleneck；常见优化会缓存 state→allowed-token set、使用 trie/FSM 与 GPU masking，但收益依 grammar、词表和状态复用。缓存 key 必须包含 grammar/tokenizer revision 与完整 parser state，不能跨不兼容 schema 复用。约束保证的仍只是所编码的语法性质，不可用“JSON 有效”替代业务校验。

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

基准记录 GPU、driver/CUDA/runtime、功耗、模型 revision、量化、context、输入/输出长度分布、arrival process、并发、warmup、采样和测试时长。每个 request 分开保存 offered、HTTP dispatch、first-token 和 terminal 时刻；否则客户端 semaphore 等待可能从 TTFT 中消失。Client queue 仍不能替代 gateway/runtime queue trace。

## 面试追问

**为何 decode 常是 memory-bound？** 每步 FLOPs 相对有限却要读取大量权重/KV；小 batch 权重复用低。batch 增大提高算术强度和吞吐，但增加排队/延迟。

**GQA 如何降低显存？** 多个 query heads 共享更少的 K/V heads，KV cache 按 kv heads 而非 query heads 增长；可能有质量权衡，但现代模型常采用。

**FlashAttention 会减少 KV cache 吗？** 它主要优化 attention 计算中的 IO/中间矩阵；标准 decode KV 的持久容量由层、KV heads、head dim、长度和 dtype 决定，不能混为一谈。

**量化为何可能不加速？** dequant 开销、kernel/硬件不支持、小 batch、其他瓶颈或格式转换会抵消带宽收益；必须测端到端而不是只看权重字节。

**为什么 4-bit 不等于 FP32 的 8 倍压缩？** `4N/0.5N=8` 只比较裸权重；实际还要加每组 scale/zero point、alignment、容器与未量化层。小 group 改善局部拟合却增加 metadata，必须报告实际 artifact/runtime resident bytes。
