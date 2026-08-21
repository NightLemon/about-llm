# 单卡推理服务与压测

本页是实现与运行参考，保存脚本参数、固定输入和每项结果的适用范围。第一次学习时不要从这里顺序通读。

- 先读[一次请求如何穿过推理引擎](../../docs/systems/inference-request-lifecycle.md)，建立端到端心智模型。
- 再做[Paged KV 引导实验](../../docs/practice/labs/lab-7a-paged-kv.md)，按预测、运行、负例的顺序学习。
- 项目交付顺序见[Inference Serving 项目页](../../docs/practice/projects/inference-serving.md)。
- 测试与 claim 对照见[推理服务证据页](../../docs/evidence/inference-serving-controls.md)。

本目录的工程目标是用同一组 workload 比较 Transformers 与 vLLM，并正确区分 TTFT、TPOT、
端到端延迟和系统吞吐。下面每个实验只回答标题所述的问题：CPU 公式对账、本地 HTTP 路径和目标 GPU 性能
属于不同证据，不能把其中一个实验的结论直接搬给另一个。

## 固定 Qwen 的真实 HTTP 参考服务

先用已录制报告做离线复核；这条命令不会加载模型或启动服务：

~~~powershell
python projects/inference-serving/run_qwen_target_service.py `
  --verify projects/inference-serving/qwen2.5-0.5b-service.recorded-report.json
~~~

需要重放真实服务时，安装 `torch`、`transformers` 与 `api` 依赖，并使用本机已有的固定 checkpoint cache：

~~~powershell
python projects/inference-serving/run_qwen_target_service.py --local-files-only
~~~

这条路径先按 immutable revision manifest 对 7 个文件、999,586,347 bytes 逐文件重哈希，再以 `trust_remote_code=False` 加载 `Qwen2ForCausalLM`。父进程生成一次性 Bearer token，子进程只监听 `127.0.0.1`；client 真实执行受保护的 health/models、unknown-field/wrong-model 负例，以及 `/v1/chat/completions` 的一次 non-stream 与一次 SSE。两次都得到 31-token prompt、completion IDs `[17,151645]`、`2<|im_end|>`、`finish_reason=stop` 和相同 usage；后端 audit 记录两次 `GenerationMixin.generate()`。Uvicorn 0.52.1 重录 report fingerprint 是 `sha256:63e566ca…617ddb`，不保存 Bearer、raw request、raw response 或 completion 原文。

这是为教学与证据核验设计的 **Transformers CPU FP32 eager reference service**，不是 vLLM，也不声称完整 OpenAI API compatibility。它是单进程、单 admission slot、HTTP loopback，未使用 TLS、proxy、OAuth/JWT/IAM、CUDA、多 worker 或远程 client。SSE 在模型完成完整 generation 后才发出两个文本 delta，因此 `[DONE]`/usage 通过不证明 incremental decode streaming、client disconnect cancellation、KV 释放、容量、SLO 或性能。文件 hash 没有密钥，不认证发布者；verify 后 loader 重新打开路径仍有 TOCTOU。

## 增量 SSE 与断连协作取消

先离线复核录制报告，再按需重放轻量真实 TCP 实验：

~~~powershell
python projects/inference-serving/incremental_streaming_control.py `
  --verify projects/inference-serving/incremental-streaming.recorded-report.json

python projects/inference-serving/incremental_streaming_control.py
~~~

这条独立实验使用本仓库实现的 async pseudo-token backend，不加载 tokenizer 或模型。完整 case 在 backend
完成前依次让 client 收到 `甲`、`🙂`、`终`，随后核对 finish、usage 与 `[DONE]`；取消 case 先发 `首`，
client 在 server audit 仍显示 active=1/backend 未完成时显式关闭 response。真实 Uvicorn subprocess 随后记录
ASGI stream task 与 backend iterator 都观察到 `asyncio.CancelledError`、active 回到 0、cancelled 变为 1，
且 emitted token IDs 仍只有 `[201]`。Uvicorn 0.52.1 重录 report fingerprint 为 `sha256:25846822…2b5d00`。

这个结果说明当前协作式 async iterator 在单进程 IPv4 loopback HTTP 上能接收断连取消，并在后续模拟 delta
前停止。实验没有执行 Transformers blocking generation thread、模型 forward、vLLM、CUDA、KV/GPU 分配或释放，
也没有覆盖 TLS/proxy/IAM、远程 client、多 worker、provider billing、质量、性能或 SLO。不要把它写成
“任意模型 runtime 都会停算”或“云 API 取消后不计费”。

### Tiny Transformers thread 的显式协作退出

下一条实验进入真实 Transformers generation loop，但仍保持模型极小且不下载 checkpoint：

~~~powershell
python projects/inference-serving/transformers_thread_cancellation_control.py `
  --verify projects/inference-serving/transformers-thread-cancellation.recorded-report.json

python projects/inference-serving/transformers_thread_cancellation_control.py
~~~

子进程构造随机 1,272 参数 `GPT2LMHeadModel`，在一个 Python thread 中真实调用 `GenerationMixin.generate()`。仓库提供的 logits processor 强制首 token ID 7；custom streamer 把它投影为 `首` 并故意暂停 generation thread。Client 收到首 delta 时，audit 必须同时显示一次真实 forward、thread alive、`generate()` 尚未返回。Client 断连后 backend 捕获 `CancelledError` 并设置 `threading.Event`；仓库提供的 `StoppingCriteria` 在下一次 termination check 观察事件，`generate()` 以唯一 continuation `[7]` 返回，thread exit/join，第二个 token 未产生。Uvicorn 0.52.1 重录 report 为 `sha256:eadcab54…f62bc7`。

这里故意暂停 streamer，是为了让竞争条件可以稳定复现，并不模拟生产 scheduler。证据只覆盖**显式植入 cooperative event/StoppingCriteria 的随机 tiny CPU 路径**：没有 tokenizer/chat template、公开 checkpoint、目标模型 logits、未修改的 Transformers call、vLLM/CUDA、KV/CPU/GPU memory-release 观测、远程 provider 或计费。已进入某个不可中断 kernel/driver 的 thread 是否退出，仍需目标 runtime 单独验证。

无 `--verify` 的 live execution 会记录当次 UTC 日期与当前 Python/Torch/Transformers/HTTP runtime identity，并按这组显式身份验证行为投影；`--verify` 则只接受仓库中经审阅的固定日期、固定 runtime artifact。两类报告不可互换：依赖升级后的 live 通过只说明新环境观察到同一受限行为，不会自动把旧录制证据升级为已审阅；若要重录，必须同时复核 runtime、报告内容、fingerprint 与上述证据边界。

## Qwen3-0.6B 穿过 nano-vLLM 的目标 GPU study

完整学习顺序见[实验 7B](../../docs/practice/labs/lab-7b-nano-vllm-qwen3.md)。Manifest 固定
`GeeeekExplorer/nano-vllm@bb823b3e06983d71485a8e1f23715ebd87d98ef8` 与
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`：

~~~bash
python projects/inference-serving/nano_vllm_study.py collect \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --source-root /path/to/nano-vllm \
  --model-snapshot /path/to/Qwen3-0.6B/snapshot \
  --output artifacts/inference/nano-vllm-study.json

python projects/inference-serving/nano_vllm_study.py verify \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --report artifacts/inference/nano-vllm-study.json
~~~

`collect` 只接受 clean upstream checkout、固定 model artifacts 与 CUDA 环境。四个独立 worker 隔离
eager/CUDA Graph 和 256/1024 prefill budget；每个 worker 再扫描 exact/one-token-drift prefix 与并发
1/2/4/8。报告保存 warmup、五个 measurement、每步 sequence/KV trace、engine TTFT/TPOT/E2E、输出
token throughput、峰值 allocated/reserved memory 和 typed failure terminal，不保存 raw prompt。

`verify` 是 CPU-only：它会拒绝重复字段、非法数值和未知字段，并检查 identity、时间顺序、指标算术、prefix hit、调度 token budget 和
KV 账本。仓库当前不包含 3070 实测数字；只有用户回传的脱敏 JSON 通过 verifier 和人工边界审查后，
才能新增 recorded evidence。上游 README 的 RTX 4070 Laptop 数字不能替代这项运行。

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

仓库提供包含 3 个成功和 1 个 429 的**合成 client trace 样例**：

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

时间必须来自同一 monotonic clock/基准窗口；成功行必须有 first token、prompt token 和正 output token。失败行允许
token 未知。`offered_at` 要么每行都有、要么每行都没有；否则工具拒绝从不完整子集计算 queue percentile。
Loader 也拒绝重复 JSON key、未知字段与 NaN/Infinity。这份固定输入用于检查聚合、错误口径和 exit code；
client queue 不等于 gateway/vLLM server queue，也不识别 GPU 利用率、KV、网络分段或真实容量。

`attempts.manifest.example.json` 固定输入 SHA-256、基准时间窗、合成时钟和证据边界，测试会验证 hash 与行数。真实 workload manifest 还必须记录 model/tokenizer/runtime/hardware、arrival process、长度联合分布、并发、warmup、网络位置和采集版本。

## 用可手算概率检查 next-token sampling

运行一份完全显式的单步 sampling policy：

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

处理顺序固定为 sign-aware repetition penalty、temperature、exact top-k、top-p、renormalization、token-id-order categorical inverse CDF。Top-k 同分时 token id 小者优先并恰好保留 k 个；top-p 使用 top-k 后重新归一化的概率，并保留第一个使 cumulative probability 达到或越过阈值的 token。

固定原始概率 `[0.4,0.3,0.2,0.1]`、top-k=3、top-p=0.7：top-k 阶段为 `[4/9,3/9,2/9,0]`，top-p 最终为 `[4/7,3/7,0,0]`，uniform=0.6 在 token-id-order CDF 中选择 token 1。另一个固定样例对历史 token 0/1 使用 penalty=2，将 `[2,-2,0.5]` 变为 `[1,-4,0.5]`；重复出现 token 1 不会重复施加，因为这不是 frequency penalty。

测试还覆盖 top-p exact/crossing boundary、top-k/top-p tie、temperature 改变 nucleus、inverse-CDF 边界、大 logits 稳定性、平移不变性、不可变数组和失败输入。该策略是教学契约，不是 Transformers/vLLM/provider 的通用默认；固定 uniform/seed 也不保证跨 RNG/kernel bitwise replay。脚本没有模型 forward、tokenizer、多 token EOS/stop、KV/batching、质量、延迟或吞吐证据。

## 一个反例：候选相关性怎样影响 self-consistency

运行两个边缘单样本成功率都为 0.6 的 exact majority-vote 场景：

~~~powershell
python projects/inference-serving/self_consistency_correlation_toy.py
~~~

独立场景只有一个 \(p=3/5\) regime。相关场景则每题先等概率抽 easy \((p=9/10)\) 或 hard \((p=3/10)\)，再在该 regime 内 conditional i.i.d. 生成 N 个 binary correctness votes；共享 regime 使边缘 pairwise correlation 精确为 \(3/8\)，但跨题单样本平均仍为 \(3/5\)。程序对奇数 N 计算每个 regime 的 binomial upper tail，不枚举 `2^N` 序列：

| N | independent majority | latent-correlated majority |
|---:|---:|---:|
| 1 | 0.60000000000 | 0.60000000000 |
| 3 | 0.64800000000 | 0.59400000000 |
| 5 | 0.68256000000 | 0.57726000000 |
| 11 | 0.75349813248 | 0.53896454244 |

测试用小 N 显式枚举对照闭式结果，并覆盖 deterministic boundary、undefined correlation、odd-N 契约、重复 ID、类型和资源上限。这个 counterexample 只处理两个 canonical answer labels，不处理开放文本中多个错误答案的 plurality/canonicalization。它没有执行 model、tokenizer、dataset、judge、GPU 或 provider，也没有测量 temperature、质量、latency 或 cost；不能据此声称真实 self-consistency 一定下降。

## 用固定概率表逐步检查 beam search

运行一份 table-driven、多 token 且逐步可审计的 beam search：

~~~powershell
python projects/inference-serving/beam_search_toy.py
~~~

每个 probability vector 都在输入中明确给出。这份参考实现只展开正概率 token；active prefix 先按累计 log probability 降序、再按 token-id sequence 排序，保留前 `beam_width`。EOS expansion 立即进入 finished set，永不再次展开；没有 heuristic early stopping；走到 `max_new_tokens` 的 active prefix 以 `finish_reason=length` 完成。最终分数固定为

\[
s=\frac{\log p(x_{1:T})}{T^\alpha},
\]

其中 (T) 只计生成 token、包含已发出的 EOS、不含 prompt。最终同分依次按 normalized score、raw log probability、token-id sequence 排序。实现保存所有从 active prefix 产生的 EOS，不模拟某些 runtime 的 top-\(2B\) candidate cap。

剪枝样例在 root 使用 `A=0.6,B=0.4`，随后 `A→EOS=0.51`、`B→EOS=1`。beam 1 返回 `A,EOS`，概率 0.306；beam 2 返回 `B,EOS`，概率 0.4，直接反驳“有限 beam 必然找到全局最高概率序列”。长度样例的短候选概率 0.6/长度 2，长候选概率 0.4/长度 3；alpha 0 选短候选，alpha 2 选长候选，展示 normalization 可以改变目标而不只是搜索宽度。

脚本不执行模型 forward、tokenizer、真实 KV 分叉或 GPU candidate-selection kernel，也不验证文本质量、latency、throughput 或显存。Transformers、vLLM 和 provider 对 length penalty、EOS、finished-candidate cap、early stopping 与 tie-break 的契约可能不同；接入时必须用固定的 token-level 输入对目标版本做差分测试。

## 用有限字符串集合解释 constrained decoding

运行一组完整的 token-fragment 状态转移与 masking 样例：

~~~powershell
python projects/inference-serving/constrained_decoding_toy.py
~~~

参考实现把本仓库准备的有限个 literal 编译成 deterministic character trie。对当前状态 (q)，非 EOS token 只有在其**完整 supplied text** 的每个 Python Unicode code point 都能依次转移时才允许；EOS 只在 accepting state 允许。屏蔽后用合法 token 的 raw probability mass 重归一化，再做 greedy selection；同分按较小 token id。合法质量为零会抛出 `ConstraintDeadEndError`，不会静默解除 grammar。

固定 literal 是 `{"x":1}` 与 `{"x":2}`。关键步的 raw probabilities 由本仓库设为：`1}` 0.25、`1]` 0.65、`2}` 0.10。`1]` 虽以合法字符 `1` 开始，却在第二个字符 `]` 无转移，因此完整屏蔽；allowed mass 是 0.35，归一化结果为 `5/7` 与 `2/7`，最终选择 `1}`。下一步状态已 accepting，EOS 才开放，最终文本为 `{"x":1}`。

测试另覆盖 Unicode multi-code-point token、首字符反例、tie、输入数组 copy、空串与 prefix literal、zero-mass dead end，以及三项不能混淆的状态：grammar accepting、EOS emitted、finish reason。`length` 可以在 accepting 或 non-accepting state 发生，均不伪装成 EOS 完成。

这不是完整 constrained-generation runtime。它假设 supplied token fragments 可直接拼成 decoded text，不执行 tokenizer byte/incremental state、Unicode normalization、JSON Schema、CFG、regex、模型 forward、KV、GPU mask kernel或性能测试。真实接入需绑定 tokenizer/grammar/runtime revision，用实际 token bytes/decoder semantics 构造 allowed set，并继续做字段语义、权限与副作用校验。

## Incremental UTF-8 stop-string matcher

运行一组跨 UTF-8 byte 和 stop-prefix 边界的固定输入：

~~~powershell
python projects/inference-serving/stop_matching_toy.py
~~~

Matcher 对同一 UTF-8 stream 做 strict incremental decode，并只释放不可能再参与 stop 的文本；内部 pending 长度严格小于配置的最大 stop 字符数。每个 decoded character 后检查完成项，因此 byte chunking 不改变结果；一个 character 同时完成多个 stop 时按配置顺序选。默认从返回文本排除 stop，也可显式 include；匹配 case-sensitive、无 Unicode normalization。

主样例把 `甲🙂乙<END>尾` 切在 emoji bytes 与 `<END>` 中间，返回 `甲🙂乙`、匹配 `<END>` 并记录 1 个 stop 后丢弃字符。重叠样例对配置 `("BC","ABC")` 与 `ABCZ` 在同一字符完成两个 stop，选择 `BC`、返回 `A`、丢弃 `Z`。测试另覆盖 partial prefix disproof/EOF flush、include-stop、first-completion、chunk invariance、invalid/truncated UTF-8 原子失败、大小写/normalization 和 terminal state。

它不接收模型 token ids，不知道 tokenizer decode policy，也不定义 provider usage/finish reason。若只在客户端匹配，它无法证明 stop 命中后远端停止 decode、释放 KV/GPU 或停止计费；真实服务必须把客户端取消、provider terminal event、usage 和 server trace 联合验收。

## 用离散调度器理解 continuous batching

运行一组不依赖模型或 GPU 的 deterministic scheduler 固定输入：

~~~powershell
python projects/inference-serving/continuous_batching_toy.py
~~~

它把时间定义成整数 scheduler boundary，并固定一份可逐步复核的 policy：请求按 `(arrival step, input order)` FCFS 进入有限 sequence slots；每轮先给所有 decode-ready sequence 一个 token position，再保证每个 resident prefill 至少前进一个 position，最后按 FCFS 使用剩余 chunked-prefill budget。为保证 resident sequence 每轮都有进展，这份参考实现要求 `max_batch_tokens >= max_running_sequences`。这是一份明确的教学策略，不声称与任意 vLLM release 的 priority、preemption 或 chunked-prefill 实现相同。

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

固定样例的 prompt/output 分别是 7/6 token、请求数是 3，因此 \(W=7+6-3=10\)，与四轮实际 used slots `3+3+2+2` 完全一致，而不是 13。输出 token throughput、计费 token 和 forward-position work 是不同口径；prefix cache、speculative verification、beam search、padding、scheduler bookkeeping 与实际 kernel 都会改变真实工作账本。

报告同时区分 elapsed capacity（包含两个 arrival 之间的 idle interval）和 active-step capacity，离散 step utilization 也不能解释成 GPU utilization、秒、TTFT/TPOT 或吞吐。这份离散参考程序不分配 KV、不做 preemption/swap/prefix cache，也不执行模型、CUDA 或 vLLM；它只检查当前 policy 的 admission、chunking、首 token、decode、completion 与 token-slot 账本。

## 用逐步账本检查 KV-aware recompute preemption

把独立 scheduler 与 block allocator 接成同一份确定性 trace：

~~~powershell
python projects/inference-serving/kv_preemption_batching_toy.py
~~~

这里明确采用一条教学策略：decode 优先，recompute 其次，initial prefill 最后；每个 position 都向 metadata-only paged KV append。需要新 block 而容量耗尽时，只能抢占“本轮尚未工作且 stable-FCFS 优先级更低”的 resident，再从中选择最近 admission 者并释放其全部 block。优先级边严格单向，避免两个 rebuild 请求反复互相驱逐；状态机发现未完成状态重复时会停止并报错。被抢占请求以后按 FCFS 重新 admission，从 position 0 重建到既有 logical frontier；重建不改变 logical progress，也不会再次发出已经交付的 token。当轮已执行 work 的 sequence 不再被抢占，完成 sequence 到 interval boundary 才释放 block。

固定请求 A=`prompt 4/output 3`、B=`prompt 2/output 2`，容量为 3 blocks×2 positions。无抢占的 logical work 是 `6+3=9`。A 的第一次 decode 需要第三块时，B 的 2-position KV 被释放；A 完成后 B 重建 2 positions 再 decode，因此：

- used slots：`3,3,1,1,2,1`；
- logical/recomputed/executed positions：`9/2/11`；
- preemption=1，peak blocks=3，完成后 free blocks=3；
- A 输出 boundary 为 `2,3,4`，B 为 `2,6`，rebuild 不伪造输出；
- 容量改为 6 blocks 的对照得到 preemption=0、logical=executed=9。

这不是某版 vLLM 的抢占算法。它没有真实 K/V tensor、swap、prefix cache、priority aging、CUDA/PagedAttention、分布式 worker 或 wall clock，block 数也不能换算成未经测量的目标 VRAM、吞吐或延迟。它只给出一个能逐步复算的 capacity/preemption/recompute 账本。

## 用 CPU 参考实现核对 weight-only quantization

在真实 GPU kernel 前，先用透明 baseline 核对 symmetric group-wise code、scale、反量化与存储口径：

~~~powershell
python projects/inference-serving/quantization_toy.py `
  --seed 17 --bit-width 4 --group-size 8 `
  --output-features 16 --input-features 33 --batch-size 8 `
  --artifact-path .\matrix.allmqtz
~~~

每个 output row 的 contiguous group 独立取 absmax scale；4-bit reference 使用 `[-7, 7]`，全零 group 的 scale 约定为 1。Signed code 用 `u=q+qmax` 映射到 offset-binary，按 C row-major、LSB-first 写入连续 bit stream；全 1 code 非法，末 byte 的高 padding bit 必须为 0。输出分开记录 FP32 weight bytes、理想 packed lower bound、实际 packed code bytes、FP32 scale metadata、raw payload、完整 hex/SHA-256 和 NumPy unpacked reference，并在进程内验证 code/scale exact round-trip。

该脚本现在确实执行 dense bit packing，并拒绝 unused all-ones code、错误长度和非零 padding。它还在内存构造严格 v1 tensor artifact：32-byte little-endian header + code + little-endian FP32 scales + 32-byte SHA-256；指定 `--artifact-path` 时用 exclusive create 写入新文件、立即严格重载，已存在路径会失败而不是覆盖。3×5 固定样例的 raw code+scale 是 32 bytes，完整文件是 96 bytes。

这个文件只代表一个矩阵，不包含模型的 tensor name/layer manifest、未量化权重、config/tokenizer、shard index 或签名；unkeyed SHA-256 也不认证来源。`quantized_linear` 仍先严格 reload/unpack/反量化再调用 NumPy matmul，没有执行 calibration、GPTQ、AWQ、低位 tensor core 或任何 GPU kernel。因此这个 96-byte 样例不能直接说明完整模型 artifact 大小、resident VRAM、质量或加速。真实比较必须固定量化实现/revision、group/axis/rounding、校准集、未量化层、runtime/kernel 与任务/安全切片，并报告实际 artifact/resident bytes 和端到端性能。

### 多矩阵 quantized bundle 实验

下面的 toy 把两个 name-sorted v1 矩阵 artifact 放进一个严格 bundle，保存 model/tokenizer revision 与 architecture config identity，再从 bytes 或新文件重载并执行 two-layer NumPy MLP：

~~~powershell
python projects/inference-serving/quantized_bundle_toy.py

# 可选：只创建新文件，已有路径会失败
python projects/inference-serving/quantized_bundle_toy.py `
  --artifact-path .\two-layer.allmqb
~~~

在 seed=29、4-bit、group-size=4 的固定设置下，实验得到：FP32 weights 288 bytes、raw quantized payload 124 bytes、两个内层 tensor artifacts 252 bytes、完整 bundle 987 bytes、外层 container overhead 735 bytes；strict reload 的 byte 与 quantized-forward round trip 都为 true，FP32-output RMSE 约 0.0368725、relative L2 约 0.123272。外层 v1 包含 24-byte header、canonical JSON manifest、连续内层 artifacts 和 32-byte SHA-256；loader 同时核对 tensor name/order/offset/length/digest 和资源上限。

它只支持二维量化矩阵，不保存 tokenizer payload、bias/norm/embedding 等未量化参数、model forward、shard index、GGUF/safetensors/runtime layout 或 fused kernel，所以仍不是完整 LLM checkpoint，也不证明模型质量、VRAM 或加速。两层 MLP 只验证多 tensor reload/control flow。SHA-256 没有密钥，不认证来源；exclusive-create + file fsync 不覆盖旧文件，但进程/机器中断仍可能留下 partial target，也没有 parent-directory fsync/断电原子性证据。

### Repo-native MiniGPT inference checkpoint

下一层实验真正恢复本仓库 tiny causal LM，而不再停留在 two-layer MLP：

~~~powershell
python projects/inference-serving/minigpt_checkpoint_toy.py

python projects/inference-serving/minigpt_checkpoint_toy.py `
  --artifact-path .\minigpt.allmgpt
~~~

Artifact 包含 Byte-BPE merges、固定 architecture/config、全部唯一参数和 tied LM-head contract。二维 embedding/linear weights 保存为 group-wise quantized matrix artifacts；LayerNorm/linear bias 保存为 FP32 vectors；causal mask 由 loader 重建。默认随机样例包含 16 个唯一参数、FP32 parameter bytes 10,976、manifest 3,904、parameter payload 4,760、完整 checkpoint 8,720 bytes。Prompt `abc abc` 编码为 `[257,32,257]`，重载 logits shape 为 `[1,3,258]`，重复读取 logits exact，FP32-logit RMSE 约 0.00477277。

严格 loader 在实例化模型前校验资源上限、config/architecture/tokenizer/tied contract 和每个参数的 name/shape/kind/order/offset/length/digest。测试还会在协同重算 outer hash 后注入 unknown/duplicate/non-canonical JSON、错误 tokenizer vocab、name/shape/offset/quantization 漂移、inner artifact 篡改和非有限 FP32 vector。

“完整 checkpoint”只在本仓库 `MiniGPT` inference contract 内成立。Artifact 不嵌入 forward source，需要受信任的固定 revision loader；Byte-BPE 没有 normalizer/special token/chat template；也没有 optimizer/RNG/training resume、sharding/device map、GGUF/safetensors/Transformers/vLLM compatibility。所有低位矩阵先反量化为 FP32 PyTorch parameter，所以不证明 resident VRAM 降低、low-bit kernel、速度或目标模型质量。Unkeyed hash 与 exclusive-create/file-fsync 的来源和 crash 边界仍然存在。

## 用 CPU 参考实现量化 KV Cache 并测量误差

下面脚本对每个 `[batch, kv_head, token, :]` 的 K/V 向量分别做 symmetric absmax INT8，真实物化 code 与 FP32 scale，再反量化并与 GQA/causal attention 的 NumPy 参考实现比较：

~~~powershell
python projects/inference-serving/kv_quantization_toy.py `
  --seed 31 --query-heads 4 --key-value-heads 2 `
  --cached-tokens 8 --query-tokens 3 `
  --key-head-dim 16 --value-head-dim 16
~~~

输出分开记录 FP32 K/V bytes、INT8 code bytes、K/V scale metadata、payload compression ratio，以及 K、V、attention probabilities 和 attention output 四组误差。相同 K/V head dimension (D) 时，该 per-token-scale payload 的理想比率是 (4D/(D+4))，不是无条件 4×；还没有计 allocator/block/alignment/workspace。

测试覆盖全零向量 scale=1、`-128` 保留码、误差 ≤ scale/2、GQA head mapping、causal future mass、显式 dequant 等价和 incremental prefix cache。实现仍先 dequantize 再做 FP32 NumPy attention，没有 INT8 dot-product/fused KV kernel、PagedAttention tensor layout、真实 resident VRAM、目标模型长上下文质量或加速证据。

## 用手算概率检查 speculative sampling

下面实验不加载模型，而是使用本仓库准备的 draft/target probability vector 验证精确 rejection-sampling identity：

~~~powershell
python projects/inference-serving/speculative_decoding_toy.py `
  --seed 23 --trials 20000
~~~

解析部分计算 proposal \(x\sim q\) 的接受概率 `min(1,p(x)/q(x))`、positive `(p-q)` residual、一步 acceptance 与 total variation distance，并逐 token 证明输出边际回到 target。默认向量的 acceptance 是 0.6，rejection/TV 是 0.4。Monte Carlo 频率只是带 seed 的直觉检查，不承担正确性证明。

固定 block 样例的第一个 proposal 必然接受、第二个以 0.25 概率接受但由固定 uniform 强制拒绝；输出改从 residual 取 token，后续 draft 被丢弃且不使用 bonus distribution。另一个单元测试覆盖全接受时额外发一个 target token。所有位置必须使用相同 vocabulary、相同 prefix 和应用 sampling transform 后的真实 \(p,q\)。

这里没有 model forward/tokenizer、draft generation latency、batched target verification、KV rollback 或 GPU kernel。解析恒等式通过不证明某个 runtime 实现正确，更不证明 throughput/TPOT 加速；真实验收还需对固定 target 做 baseline/speculative 同 seed 分布检验、逐请求 accepted-token trace、target calls、峰值显存和端到端负载测试。

## 一个 verifier-guided best-of-N 的精确反例

运行一个由本仓库准备的有限 candidate distribution 上的精确选择反例：

~~~powershell
python projects/inference-serving/verifier_best_of_n_toy.py
~~~

每次抽样 i.i.d. 来自 `wrong/correct/verifier_hack`，sampling weight 为 `5/4/1`，deterministic verifier score 为 `20/80/99`，只有 `correct` 的 target label 为 true。选择规则固定为最大 `(verifier_score, candidate_id)`。候选按该规则由弱到强排序后，程序用 `P(select i)=F_i^N-F_{i-1}^N` 和 `oracle@N=1-(1-p_success)^N` 做精确 `Fraction` 闭式计算。

N=1/4/16 的 `(oracle@N, selected@N, expected verifier score)` 分别为 `(0.4, 0.4, 51.9)`、`(0.8704, 0.5936, 82.7841)`、`(0.9997178890, 0.1852867601, 95.4783461)`。因此 proxy 分数严格上升并不推出 target success 单调上升；oracle@N 也不等于 verifier-selected@N。测试还用小规模显式序列枚举对照闭式公式，并覆盖 score tie、输入资源上限和失败契约。

N=16 的 `3^16=43,046,721` 只是 logical candidate sequences；实现没有逐序列枚举。报告里的 N 次 logical model samples/scores 也不是 wall-clock、费用或并行度测量。这里没有执行 model、tokenizer、PRM、GPU、provider 或真实 reward pipeline，不证明 verifier calibration、语义正确、目标模型质量、latency、cost 或生产中的 optimizer's curse。

## Paged KV block allocator state-machine toy

运行一个 metadata-only 的 prefix sharing/COW/容量故障场景：

~~~powershell
python projects/inference-serving/kv_block_allocator_toy.py
~~~

总容量 3 blocks、每块 4 tokens。`request-a` 的 6 tokens 占用 `4+2`，fork 为 `request-b` 后两块 refcount 都为 2。A 再 append 1 token 时，shared partial tail 从 block 1 COW 到 block 2；B 随后把自己独占的旧 tail 填满。此时 3 blocks 用尽，A append 2 需要先利用 1 个 tail slot、再分配 1 块，因无 free block 在任何 mutation 前失败。输出验证 sequence state 与 report 完全不变；释放 A 后独占 block 回收，共享 block 只减 refcount。

报告同时给 logical block references、sharing saved blocks、logical tokens、physical token values、allocated slots 和 physical fragmentation。Logical tokens 对共享 prefix 重复计数；物理碎片只能用 physical values 计算。

脚本没有分配真实 K/V tensor，所谓 COW 只复制 occupancy metadata；也没有 CUDA page table/PagedAttention、连续批处理、prefix key/ACL、eviction/preemption/swap 或并发压测。因此它只能检查 allocator 状态机，不能代表 vLLM 行为、VRAM 或性能。

## 使用真实 CPU 张量的 Paged KV 实验

在相同 allocator 状态机上预分配真实 CPU K/V tensor arena：

~~~powershell
python projects/inference-serving/paged_kv_tensor_toy.py
~~~

固定 float64 样例写入 5-token prefix、fork block table，然后让父序列 append 触发 shared partial-tail COW。输出验证子序列 tensor 不受污染、父序列可按逻辑顺序 materialize，并将 4 query heads/2 KV heads 的 causal GQA 与独立 dense reference 对账。`resident_bytes` 只计算两个固定 arena 的 tensor 字节，不包括 allocator metadata、PyTorch runtime 或进程 RSS。

这里存储和复制真实 K/V 数值，但 attention 会 gather 完整 sequence 并物化 dense scores/probability。它没有 CUDA PagedAttention kernel、模型 decode、scheduler、并发、eviction/swap 或性能测量，不能据此声称 VRAM 节省、latency 或 throughput 改善。

## 用固定输入检查 prefix cache 的隔离与 lease

运行 collision-safe、metadata-only 的 longest-prefix cache：

~~~powershell
python projects/inference-serving/prefix_cache_toy.py
~~~

固定样例存入同一 tenant 的 `(11,12)`、`(11,12,13)` 和另一 tenant 的相同长前缀，并把 fingerprint function 故意固定为字符串 `collision`。对 `(11,12,13,14)` 的查询必须命中本 tenant 长度 3 的最长 exact token prefix；未存 tenant 的同 token 查询必须 miss。输出精确为 1 hit、1 miss、0 eviction，release 后 0 active lease。

安全 identity 包含 trusted tenant、visibility domain、authorization/policy revision、model/tokenizer/chat-template/adapter revision、RoPE/position config 和 KV dtype；full identity 与 exact token tuple 才决定命中，fingerprint 只定位候选。其他测试覆盖每个 identity 字段的隔离、重复 store、LRU、leased entry 不可淘汰、容量全部 leased 时 no-mutation failure，以及 foreign/double release 拒绝。

实现没有存储 K/V tensor，也没有 TTL、加密、分布式一致性或 timing-channel mitigation。Unkeyed SHA-256 不能授权、认证来源或隐藏低熵 prompt；这组固定输入不能证明真实 VRAM、hit rate、prefill savings、延迟/吞吐或任一 vLLM 版本的行为。

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
