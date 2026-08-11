# 单卡推理服务与压测

目标：用同一组 workload 比较 Transformers 与 vLLM，并正确区分 TTFT、TPOT、端到端延迟和系统吞吐。

## vLLM 服务

vLLM 的平台和版本兼容变化较快。先按官方说明在 Linux/WSL2 安装，再选择适合显存和许可证的模型。示例：

~~~bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
~~~

不要未经测量就把上下文和 memory utilization 拉满。先单请求 smoke test，再逐步增加输入长度和并发。

## OpenAI-compatible 压测

安装 API 依赖后运行：

~~~powershell
python -m pip install -e ".[api]"
python projects/inference-serving/benchmark_openai.py --model Qwen/Qwen2.5-0.5B-Instruct --requests 50 --concurrency 4
~~~

默认 `burst` 在基准开始时同时 offer 全部有限请求。要让到达时刻不随服务完成速度变化，可选择 constant 或带固定 seed 的 Poisson inter-arrival：

~~~powershell
python projects/inference-serving/benchmark_openai.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --requests 100 --concurrency 8 `
  --arrival-process constant --request-rate 4

python projects/inference-serving/benchmark_openai.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --requests 100 --concurrency 8 `
  --arrival-process poisson --request-rate 4 --arrival-seed 7
~~~

constant 的第 (i) 条请求在 (i/\lambda) 秒到达。Poisson 模式把第一条请求锚定在 0，后续间隔独立采样自均值 (1/\lambda) 的 exponential distribution；这是有限、seeded realization，实际样本 inter-arrival rate 通常不精确等于 nominal rate。输出保留全部 offsets、nominal/realized rate 与 seed。

客户端使用流式 SSE：

- 首个非空 content delta 记录 first-token time；
- completion_tokens 必须来自服务端 usage；
- SSE chunk 不是 token：缺少 usage 时脚本明确失败，不输出伪精确 TPOT；
- throughput 使用整轮 benchmark wall time，不把单请求速度相加；
- 单 token 输出的 TPOT 为未定义，不伪报为 0。

每个 offered request 都写入 `attempts`，终态分为 `success`、`timeout`、`rate_limited`、`server_error`、`http_error`、`protocol_error` 或 `client_error`。单个预期请求失败不会让整轮 `asyncio.gather` 丢失统计；未预期的程序错误仍会让 benchmark 失败，避免把代码 bug 伪装成服务错误。

脚本把四个时刻分开：`offered_at = benchmark_started_at + scheduled_offset`，`started_at` 是取得 client semaphore 后开始 HTTP dispatch，随后是 `first_token_at` 和 `completed_at`。constant/Poisson 模式会按预生成时刻继续 offer，不等待前一请求完成；`--concurrency` 只限制在途 HTTP attempts，所以服务变慢会累积 client queue，而不是反过来降低 arrival rate。若只从 `started_at` 计时，event-loop lag 与 semaphore 前等待都会消失，形成 client-side coordinated omission。

这仍是有限任务发生器：它会为 `--requests` 一次性物化 coroutine，没有无限运行、bounded pending queue、分布式 worker 或 generator-lag SLO。scheduled `offered_at` 能把本地唤醒迟到计入 client queue，但不证明 Python event loop 实际按时 offer，更不能把 client queue 解释成 gateway/vLLM server queue。高 nominal rate 可能在客户端积压大量任务并产生费用；先用小请求数扫描，生产压测还需监控 generator CPU/lag、设置预算与紧急停止。

输出把 reliability 与成功请求性能分开：

- `success_rate = successful / attempted`；
- attempted requests/s 表示客户端完成的全部尝试；
- successful requests/s 和 successful output tokens/s 只计成功请求；
- `client_queue = started_at - offered_at`，对全部 attempt 统计；
- 既有 TTFT/E2E 是成功请求的 dispatch-to-first/terminal 条件统计；`successful_offered_ttft` 把 client queue 纳入成功请求首 token 体验；
- `offered_to_terminal` 对全部 attempt 统计到成功或失败终态，快速 429 也可能让它变小，因此仍必须与 outcome/success rate 联读；
- 没有成功样本时 latency 是 `null`，不是 0。

失败后若已收到 partial token，真实采集器应保留已知的 first-token/token usage；当前网络脚本只在完整成功且服务端返回 usage 时给出精确 token 指标。

## 离线 attempt 与 SLO 分析

仓库提供包含 3 个成功和 1 个 429 的**合成 client trace fixture**：

~~~powershell
python -m about_llm.inference_analysis_cli `
  --attempts projects/inference-serving/attempts.example.jsonl `
  --benchmark-started-at 0 `
  --benchmark-completed-at 2 `
  --minimum-success-rate 0.75 `
  --maximum-ttft-p95 0.5 `
  --maximum-e2e-p95 1.5 `
  --maximum-tpot-p95 0.3 `
  --maximum-client-queue-p95 0.2 `
  --maximum-successful-offered-ttft-p95 0.6 `
  --maximum-offered-to-terminal-p95 1.5 `
  --output artifacts/inference/workload-report.json
~~~

安装后也可用 `about-llm-inference-analyze`。门禁返回所有失败原因，而不是第一个失败布尔值。Trace 每行最低字段：

```json
{
  "request_id": "req-001",
  "outcome": "success",
  "offered_at": 0.0,
  "started_at": 0.0,
  "first_token_at": 0.2,
  "completed_at": 0.8,
  "prompt_tokens": 128,
  "output_tokens": 4
}
```

时间必须来自同一 monotonic clock/基准窗口；成功行必须有 first token、prompt token 和正 output token。失败行允许 token 未知。`offered_at` 要么每行都有、要么每行都没有；否则工具拒绝从不完整子集计算 queue percentile。Loader 也拒绝重复 JSON key、未知字段与 NaN/Infinity。Fixture 只验证聚合、错误口径和 exit code；client queue 不等于 gateway/vLLM server queue，也不识别 GPU 利用率、KV、网络分段或真实容量。

`attempts.manifest.example.json` 固定输入 SHA-256、基准时间窗、合成时钟和证据边界，测试会验证 hash 与行数。真实 workload manifest 还必须记录 model/tokenizer/runtime/hardware、arrival process、长度联合分布、并发、warmup、网络位置和采集版本。

## Next-token sampling distribution oracle

运行一份完全显式的单步 sampling policy：

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

处理顺序固定为 sign-aware repetition penalty、temperature、exact top-k、top-p、renormalization、token-id-order categorical inverse CDF。Top-k 同分时 token id 小者优先并恰好保留 k 个；top-p 使用 top-k 后重新归一化的概率，并保留第一个使 cumulative probability 达到或越过阈值的 token。

固定原始概率 `[0.4,0.3,0.2,0.1]`、top-k=3、top-p=0.7：top-k 阶段为 `[4/9,3/9,2/9,0]`，top-p 最终为 `[4/7,3/7,0,0]`，uniform=0.6 在 token-id-order CDF 中选择 token 1。另一个 signed-logit fixture 对历史 token 0/1 使用 penalty=2，将 `[2,-2,0.5]` 变为 `[1,-4,0.5]`；重复出现 token 1 不会重复施加，因为这不是 frequency penalty。

测试还覆盖 top-p exact/crossing boundary、top-k/top-p tie、temperature 改变 nucleus、inverse-CDF 边界、大 logits 稳定性、平移不变性、不可变数组和失败输入。该策略是教学契约，不是 Transformers/vLLM/provider 的通用默认；固定 uniform/seed 也不保证跨 RNG/kernel bitwise replay。脚本没有模型 forward、tokenizer、多 token EOS/stop、KV/batching、质量、延迟或吞吐证据。

## Deterministic beam-search oracle

运行一份 table-driven、多 token 且逐步可审计的 beam search：

~~~powershell
python projects/inference-serving/beam_search_toy.py
~~~

每个 probability vector 都由 fixture 显式给出。Oracle 只展开正概率 token；active prefix 先按累计 log probability 降序、再按 token-id sequence 排序，保留前 `beam_width`。EOS expansion 立即进入 finished set，永不再次展开；没有 heuristic early stopping；走到 `max_new_tokens` 的 active prefix 以 `finish_reason=length` 完成。最终分数固定为

\[
s=\frac{\log p(x_{1:T})}{T^\alpha},
\]

其中 (T) 只计生成 token、包含已发出的 EOS、不含 prompt。最终同分依次按 normalized score、raw log probability、token-id sequence 排序。实现保存所有从 active prefix 产生的 EOS，不模拟某些 runtime 的 top-\(2B\) candidate cap。

Pruning fixture 在 root 使用 `A=0.6,B=0.4`，随后 `A→EOS=0.51`、`B→EOS=1`。beam 1 返回 `A,EOS`，概率 0.306；beam 2 返回 `B,EOS`，概率 0.4，直接反驳“有限 beam 必然找到全局最高概率序列”。Length fixture 的短候选概率 0.6/长度 2，长候选概率 0.4/长度 3；alpha 0 选短候选，alpha 2 选长候选，展示 normalization 可以改变目标而不只是搜索宽度。

脚本不执行模型 forward、tokenizer、真实 KV 分叉或 GPU candidate-selection kernel，也不验证文本质量、latency、throughput 或显存。Transformers、vLLM 和 provider 对 length penalty、EOS、finished-candidate cap、early stopping 与 tie-break 的契约可能不同；接入时必须以固定 token-level fixture 对目标版本做差分测试。

## Token-aware finite-language constraint oracle

运行完整 token-fragment 状态转移与 masking fixture：

~~~powershell
python projects/inference-serving/constrained_decoding_toy.py
~~~

Oracle 将有限个 authored literal 编译成 deterministic character trie。对当前状态 (q)，非 EOS token 只有在其**完整 supplied text** 的每个 Python Unicode code point 都能依次转移时才允许；EOS 只在 accepting state 允许。屏蔽后用合法 token 的 raw probability mass 重归一化，再做 greedy selection；同分按较小 token id。合法质量为零会抛出 `ConstraintDeadEndError`，不会静默解除 grammar。

固定 literal 是 `{"x":1}` 与 `{"x":2}`。关键步的 authored raw probabilities 给 `1}` 0.25、`1]` 0.65、`2}` 0.10：`1]` 虽以合法字符 `1` 开始，却在第二个字符 `]` 无转移，因此完整屏蔽；allowed mass 是 0.35，归一化结果为 `5/7` 与 `2/7`，最终选择 `1}`。下一步状态已 accepting，EOS 才开放，最终文本为 `{"x":1}`。

测试另覆盖 Unicode multi-code-point token、首字符反例、tie、输入数组 copy、空串与 prefix literal、zero-mass dead end，以及三项不能混淆的状态：grammar accepting、EOS emitted、finish reason。`length` 可以在 accepting 或 non-accepting state 发生，均不伪装成 EOS 完成。

这不是完整 constrained-generation runtime。它假设 supplied token fragments 可直接拼成 decoded text，不执行 tokenizer byte/incremental state、Unicode normalization、JSON Schema、CFG、regex、模型 forward、KV、GPU mask kernel或性能测试。真实接入需绑定 tokenizer/grammar/runtime revision，用实际 token bytes/decoder semantics 构造 allowed set，并继续做字段语义、权限与副作用校验。

## Incremental UTF-8 stop-string matcher

运行跨 UTF-8 byte 和 stop-prefix 边界的流式 fixture：

~~~powershell
python projects/inference-serving/stop_matching_toy.py
~~~

Matcher 对同一 UTF-8 stream 做 strict incremental decode，并只释放不可能再参与 stop 的文本；内部 pending 长度严格小于配置的最大 stop 字符数。每个 decoded character 后检查完成项，因此 byte chunking 不改变结果；一个 character 同时完成多个 stop 时按配置顺序选。默认从返回文本排除 stop，也可显式 include；匹配 case-sensitive、无 Unicode normalization。

主 fixture 把 `甲🙂乙<END>尾` 切在 emoji bytes 与 `<END>` 中间，返回 `甲🙂乙`、匹配 `<END>` 并记录 1 个 stop 后丢弃字符。Overlap fixture 对配置 `("BC","ABC")` 与 `ABCZ` 在同一字符完成两个 stop，选择 `BC`、返回 `A`、丢弃 `Z`。测试另覆盖 partial prefix disproof/EOF flush、include-stop、first-completion、chunk invariance、invalid/truncated UTF-8 原子失败、大小写/normalization 和 terminal state。

它不接收模型 token ids，不知道 tokenizer decode policy，也不定义 provider usage/finish reason。若只在客户端匹配，它无法证明 stop 命中后远端停止 decode、释放 KV/GPU 或停止计费；真实服务必须把客户端取消、provider terminal event、usage 和 server trace 联合验收。

## Continuous batching 离散调度 oracle

运行不依赖模型或 GPU 的 deterministic scheduler fixture：

~~~powershell
python projects/inference-serving/continuous_batching_toy.py
~~~

它把时间定义成整数 scheduler boundary，并固定一份可逐步复核的 policy：请求按 `(arrival step, input order)` FCFS 进入有限 sequence slots；每轮先给所有 decode-ready sequence 一个 token position，再保证每个 resident prefill 至少前进一个 position，最后按 FCFS 使用剩余 chunked-prefill budget。为保证 resident sequence 每轮都有进展，reference 要求 `max_batch_tokens >= max_running_sequences`。这是一份明确的教学策略，不声称与任意 vLLM release 的 priority、preemption 或 chunked-prefill 实现相同。

固定配置是 `max_batch_tokens=4`、`max_running_sequences=2`、per-request prefill chunk 上限 3：

| interval | admission | prefill positions | decode positions | boundary event | used slots |
|---:|---|---|---|---|---:|
| `[0,1)` | A | A:3 | — | — | 3 |
| `[1,2)` | B | A:1, B:2 | — | A/B 首 token | 3 |
| `[2,3)` | — | — | A, B | B 完成 | 2 |
| `[3,4)` | C | C:1 | A | A/C 完成 | 2 |

C 被 sequence-cap 从 arrival boundary 1 排到 admission boundary 3，因此 queue=2、TTFT=3；A/B 的 prefill 在 boundary 2 结束并同时发出首 token。这里没有把首 token 错算成额外 decode forward：标准 causal LM 的最后一个 prompt position 已产生首个输出分布，所以在没有 prefix reuse/speculation、且每条请求恰好发出 \(O_i\ge1\) 个 token 的约定下，模型位置工作量为

\[
W=\sum_i\left(P_i+O_i-1\right).
\]

fixture 的 prompt/output 分别是 7/6 token、请求数是 3，因此 \(W=7+6-3=10\)，与四轮实际 used slots `3+3+2+2` 完全一致，而不是 13。输出 token throughput、计费 token 和 forward-position work 是不同口径；prefix cache、speculative verification、beam search、padding、scheduler bookkeeping 与实际 kernel 都会改变真实工作账本。

报告同时区分 elapsed capacity（包含两个 arrival 之间的 idle interval）和 active-step capacity，离散 step utilization 也不能解释成 GPU utilization、秒、TTFT/TPOT 或吞吐。该 oracle 不分配 KV、不做 preemption/swap/prefix cache，不执行模型、CUDA 或 vLLM；它只证明当前 policy 的 admission、chunking、首 token、decode、completion 与 token-slot 账本。

## KV-aware recompute preemption oracle

把独立 scheduler 与 block allocator 接成同一份确定性 trace：

~~~powershell
python projects/inference-serving/kv_preemption_batching_toy.py
~~~

策略仍是 authored contract：decode 优先，recompute 其次，initial prefill 最后；每个 position 都向 metadata-only paged KV append。需要新 block 而容量耗尽时，只能抢占“本轮尚未工作且 stable-FCFS 优先级更低”的 resident，再从中选择最近 admission 者并释放其全部 block。优先级边严格单向，避免两个 rebuild 请求反复互相驱逐；状态机另在发现未完成状态重复时 fail closed。被抢占请求以后按 FCFS 重新 admission，从 position 0 重建到既有 logical frontier；重建不改变 logical progress，也不会再次发出已经交付的 token。当轮已执行 work 的 sequence 不再被抢占，完成 sequence 到 interval boundary 才释放 block。

固定请求 A=`prompt 4/output 3`、B=`prompt 2/output 2`，容量为 3 blocks×2 positions。无抢占的 logical work 是 `6+3=9`。A 的第一次 decode 需要第三块时，B 的 2-position KV 被释放；A 完成后 B 重建 2 positions 再 decode，因此：

- used slots：`3,3,1,1,2,1`；
- logical/recomputed/executed positions：`9/2/11`；
- preemption=1，peak blocks=3，完成后 free blocks=3；
- A 输出 boundary 为 `2,3,4`，B 为 `2,6`，rebuild 不伪造输出；
- 容量改为 6 blocks 的对照得到 preemption=0、logical=executed=9。

这不是某版 vLLM 的抢占算法。它没有真实 K/V tensor、swap、prefix cache、priority aging、CUDA/PagedAttention、分布式 worker 或 wall clock，block 数也不能换算成未经测量的目标 VRAM、吞吐或延迟。它只给出一个能逐步复算的 capacity/preemption/recompute 账本。

## Weight-only quantization CPU oracle

在真实 GPU kernel 前，先用透明 baseline 核对 symmetric group-wise code、scale、反量化与存储口径：

~~~powershell
python projects/inference-serving/quantization_toy.py `
  --seed 17 --bit-width 4 --group-size 8 `
  --output-features 16 --input-features 33 --batch-size 8 `
  --artifact-path .\matrix.allmqtz
~~~

每个 output row 的 contiguous group 独立取 absmax scale；4-bit reference 使用 `[-7, 7]`，全零 group 的 scale 约定为 1。Signed code 用 `u=q+qmax` 映射到 offset-binary，按 C row-major、LSB-first 写入连续 bit stream；全 1 code 非法，末 byte 的高 padding bit 必须为 0。输出分开记录 FP32 weight bytes、理想 packed lower bound、实际 packed code bytes、FP32 scale metadata、raw payload、完整 hex/SHA-256 和 NumPy unpacked reference，并在进程内验证 code/scale exact round-trip。

该脚本现在确实执行 dense bit packing，并拒绝 unused all-ones code、错误长度和非零 padding。它还在内存构造严格 v1 tensor artifact：32-byte little-endian header + code + little-endian FP32 scales + 32-byte SHA-256；指定 `--artifact-path` 时用 exclusive create 写入新文件、立即严格重载，已存在路径会失败而不是覆盖。3×5 fixture 的 raw code+scale 是 32 bytes，完整文件是 96 bytes。

这个文件只代表一个矩阵，不包含模型的 tensor name/layer manifest、未量化权重、config/tokenizer、shard index 或签名；unkeyed SHA-256 也不认证来源。`quantized_linear` 仍先严格 reload/unpack/反量化再调用 NumPy matmul，没有执行 calibration、GPTQ、AWQ、低位 tensor core 或任何 GPU kernel。因此 96-byte fixture 不能直接声称完整模型 artifact 大小、resident VRAM、质量或加速。真实比较必须固定量化实现/revision、group/axis/rounding、校准集、未量化层、runtime/kernel 与任务/安全切片，并报告实际 artifact/resident bytes 和端到端性能。

### Multi-matrix quantized bundle control

下面的 toy 把两个 name-sorted v1 矩阵 artifact 放进一个严格 bundle，保存 model/tokenizer revision 与 architecture config identity，再从 bytes 或新文件重载并执行 two-layer NumPy MLP：

~~~powershell
python projects/inference-serving/quantized_bundle_toy.py

# 可选：只创建新文件，已有路径会失败
python projects/inference-serving/quantized_bundle_toy.py `
  --artifact-path .\two-layer.allmqb
~~~

默认 seed=29、4-bit、group-size=4 fixture 固定得到：FP32 weights 288 bytes、raw quantized payload 124 bytes、两个内层 tensor artifacts 252 bytes、完整 bundle 987 bytes、外层 container overhead 735 bytes；strict reload 的 byte 与 quantized-forward round trip 都为 true，FP32-output RMSE 约 0.0368725、relative L2 约 0.123272。外层 v1 包含 24-byte header、canonical JSON manifest、连续内层 artifacts 和 32-byte SHA-256；loader 同时核对 tensor name/order/offset/length/digest 和资源上限。

它只支持二维量化矩阵，不保存 tokenizer payload、bias/norm/embedding 等未量化参数、model forward、shard index、GGUF/safetensors/runtime layout 或 fused kernel，所以仍不是完整 LLM checkpoint，也不证明模型质量、VRAM 或加速。两层 MLP 只验证多 tensor reload/control flow。SHA-256 没有密钥，不认证来源；exclusive-create + file fsync 不覆盖旧文件，但进程/机器中断仍可能留下 partial target，也没有 parent-directory fsync/断电原子性证据。

### Repo-native MiniGPT inference checkpoint

下一层实验真正恢复本仓库 tiny causal LM，而不再停留在 two-layer MLP：

~~~powershell
python projects/inference-serving/minigpt_checkpoint_toy.py

python projects/inference-serving/minigpt_checkpoint_toy.py `
  --artifact-path .\minigpt.allmgpt
~~~

Artifact 包含 Byte-BPE merges、固定 architecture/config、全部唯一参数和 tied LM-head contract。二维 embedding/linear weights 保存为 group-wise quantized matrix artifacts；LayerNorm/linear bias 保存为 FP32 vectors；causal mask 由 loader 重建。默认随机 fixture 固定为 16 个唯一参数、FP32 parameter bytes 10,976、manifest 3,904、parameter payload 4,760、完整 checkpoint 8,720 bytes。Prompt `abc abc` 编码为 `[257,32,257]`，重载 logits shape 为 `[1,3,258]`，重复读取 logits exact，FP32-logit RMSE 约 0.00477277。

严格 loader 在实例化模型前校验资源上限、config/architecture/tokenizer/tied contract 和每个参数的 name/shape/kind/order/offset/length/digest。测试还会在协同重算 outer hash 后注入 unknown/duplicate/non-canonical JSON、错误 tokenizer vocab、name/shape/offset/quantization 漂移、inner artifact 篡改和非有限 FP32 vector。

“完整 checkpoint”只在本仓库 `MiniGPT` inference contract 内成立。Artifact 不嵌入 forward source，需要受信任的固定 revision loader；Byte-BPE 没有 normalizer/special token/chat template；也没有 optimizer/RNG/training resume、sharding/device map、GGUF/safetensors/Transformers/vLLM compatibility。所有低位矩阵先反量化为 FP32 PyTorch parameter，所以不证明 resident VRAM 降低、low-bit kernel、速度或目标模型质量。Unkeyed hash 与 exclusive-create/file-fsync 的来源和 crash 边界仍然存在。

## INT8 KV-cache + GQA error oracle

下面脚本对每个 `[batch, kv_head, token, :]` 的 K/V 向量分别做 symmetric absmax INT8，真实物化 code 与 FP32 scale，再反量化并执行已有 GQA/causal attention oracle：

~~~powershell
python projects/inference-serving/kv_quantization_toy.py `
  --seed 31 --query-heads 4 --key-value-heads 2 `
  --cached-tokens 8 --query-tokens 3 `
  --key-head-dim 16 --value-head-dim 16
~~~

输出分开记录 FP32 K/V bytes、INT8 code bytes、K/V scale metadata、payload compression ratio，以及 K、V、attention probabilities 和 attention output 四组误差。相同 K/V head dimension (D) 时，该 per-token-scale payload 的理想比率是 (4D/(D+4))，不是无条件 4×；还没有计 allocator/block/alignment/workspace。

测试覆盖全零向量 scale=1、`-128` 保留码、误差 ≤ scale/2、GQA head mapping、causal future mass、显式 dequant 等价和 incremental prefix cache。实现仍先 dequantize 再做 FP32 NumPy attention，没有 INT8 dot-product/fused KV kernel、PagedAttention tensor layout、真实 resident VRAM、目标模型长上下文质量或加速证据。

## Speculative sampling probability oracle

下面实验不加载模型，而是用 authored draft/target probability vector 验证精确 rejection-sampling identity：

~~~powershell
python projects/inference-serving/speculative_decoding_toy.py `
  --seed 23 --trials 20000
~~~

解析部分计算 proposal \(x\sim q\) 的接受概率 `min(1,p(x)/q(x))`、positive `(p-q)` residual、一步 acceptance 与 total variation distance，并逐 token 证明输出边际回到 target。默认向量的 acceptance 是 0.6，rejection/TV 是 0.4。Monte Carlo 频率只是带 seed 的直觉检查，不承担正确性证明。

block fixture 的第一个 proposal 必然接受、第二个以 0.25 概率接受但由固定 uniform 强制拒绝；输出改从 residual 取 token，后续 draft 被丢弃且不使用 bonus distribution。另一个单元测试覆盖全接受时额外发一个 target token。所有位置必须使用相同 vocabulary、相同 prefix 和应用 sampling transform 后的真实 \(p,q\)。

这里没有 model forward/tokenizer、draft generation latency、batched target verification、KV rollback 或 GPU kernel。解析恒等式通过不证明某个 runtime 实现正确，更不证明 throughput/TPOT 加速；真实验收还需对固定 target 做 baseline/speculative 同 seed 分布检验、逐请求 accepted-token trace、target calls、峰值显存和端到端负载测试。

## Paged KV block allocator state-machine toy

运行一个 metadata-only 的 prefix sharing/COW/容量故障场景：

~~~powershell
python projects/inference-serving/kv_block_allocator_toy.py
~~~

总容量 3 blocks、每块 4 tokens。`request-a` 的 6 tokens 占用 `4+2`，fork 为 `request-b` 后两块 refcount 都为 2。A 再 append 1 token 时，shared partial tail 从 block 1 COW 到 block 2；B 随后把自己独占的旧 tail 填满。此时 3 blocks 用尽，A append 2 需要先利用 1 个 tail slot、再分配 1 块，因无 free block 在任何 mutation 前失败。输出验证 sequence state 与 report 完全不变；释放 A 后独占 block 回收，共享 block 只减 refcount。

报告同时给 logical block references、sharing saved blocks、logical tokens、physical token values、allocated slots 和 physical fragmentation。Logical tokens 对共享 prefix 重复计数；物理碎片只能用 physical values 计算。

脚本没有分配真实 K/V tensor，所谓 COW 只复制 occupancy metadata；也没有 CUDA page table/PagedAttention、连续批处理、prefix key/ACL、eviction/preemption/swap 或并发压测。因此它是 allocator 状态机 oracle，不是 vLLM 行为、VRAM 或性能证据。

## Prefix cache identity / lease oracle

运行 collision-safe、metadata-only 的 longest-prefix cache：

~~~powershell
python projects/inference-serving/prefix_cache_toy.py
~~~

固定 fixture 存入同一 tenant 的 `(11,12)`、`(11,12,13)` 和另一 tenant 的相同长前缀，并把 fingerprint function 故意固定为字符串 `collision`。对 `(11,12,13,14)` 的查询必须命中本 tenant 长度 3 的最长 exact token prefix；未存 tenant 的同 token 查询必须 miss。输出精确为 1 hit、1 miss、0 eviction，release 后 0 active lease。

安全 identity 包含 trusted tenant、visibility domain、authorization/policy revision、model/tokenizer/chat-template/adapter revision、RoPE/position config 和 KV dtype；full identity 与 exact token tuple 才决定命中，fingerprint 只定位候选。其他测试覆盖每个 identity 字段的隔离、重复 store、LRU、leased entry 不可淘汰、容量全部 leased 时 no-mutation failure，以及 foreign/double release 拒绝。

实现没有存储 K/V tensor，也没有 TTL、加密、分布式一致性或 timing-channel mitigation。Unkeyed SHA-256 不能授权、认证来源或隐藏低熵 prompt；fixture 不证明真实 VRAM、hit rate、prefill savings、延迟/吞吐或任一 vLLM 版本的行为。

## 公平比较协议

固定模型 revision、tokenizer、量化、prompt 集、输入/输出上限、温度、硬件与并发。先做质量等价检查，再比较性能。至少扫描：

- 输入长度：短、中、目标上限；
- 输出长度：短回答与长生成；
- 并发：1、2、4、8，直到尾延迟或 OOM 不可接受；
- prefix cache 开关；
- 量化与 KV dtype；
- Transformers generate 与 vLLM continuous batching。

报告 client queue、dispatch TTFT/E2E、offered TTFT/terminal latency、TPOT、请求/秒、输出 token/秒、峰值显存、错误率和任务质量，并明确每项是 all-attempt 还是 success-conditional。平均延迟不足以做容量规划。

## 容量与故障

压测同时记录服务队列、GPU 利用率、KV cache usage 和 preemption。失败请求不能从统计中静默删除；应单列超时、429、OOM 与取消。客户端断开时验证服务停止生成。

当前在线脚本只能从 HTTP/SSE 分类 client-visible failure，不能仅凭异常类型区分 GPU OOM、scheduler crash 或上游网关 5xx；要把 client attempt 与 server trace/request id 关联。`rate_limited` 也可能是正确的 admission control 行为，SLO 是否允许取决于 offered load 与合同配额。

## 已验证范围

严格 attempt artifact、offered/dispatch 双时钟、finite schedule、SLO/SSE 由 CPU 单元测试覆盖。另有 continuous-batching 离散 policy、真实 dense weight-code packing/单矩阵 artifact、INT8 KV code+scale→dequantized GQA error oracle、speculative sampling，以及 Paged KV block/refcount/COW/atomic-capacity-failure oracle。它们使用 synthetic/authored 数据或 metadata，不是 vLLM scheduler、完整模型、low-bit/KV/speculative/PagedAttention GPU kernel、目标模型质量、VRAM 或性能证据。schedule 测试同样不证明事件循环跟得上 nominal QPS。实际 vLLM 命令需要受支持的 Linux/GPU 环境，当前 Windows CPU fixture 不会伪装成 GPU 实测、容量报告或生产 SLO。
