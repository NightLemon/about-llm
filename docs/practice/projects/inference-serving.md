# Inference Serving

**项目导航**：[返回项目索引](../project-index.md) · [推理基础](../../systems/inference.md) · [服务与可观测性](../../systems/serving.md) · [实验 7](../labs.md#lab-7)
{ .doc-nav }

## 目标

把“解码公式正确”“本地 HTTP 路径执行”“客户端断连传播”“调度/KV 状态机正确”“目标 GPU 服务达到 SLO”拆成不同证据层。这个项目同时提供精确 CPU oracle、固定 Qwen 权重的 Transformers reference service、两种取消 control、离线 workload gate 和真实 vLLM 压测入口；它们不能互相借用证据等级。

| 层级 | 本项目可运行内容 | 能回答的问题 |
|---|---|---|
| 数学/状态机 oracle | sampling、beam、constraint、stop、speculative、batching、KV、量化 | 固定契约的公式、排序、状态转移和字节账本是否正确 |
| 本地集成 control | 固定 Qwen HTTP、authored async SSE、tiny Transformers thread | 指定本地框架/网络路径是否真实执行 |
| Workload contract | offered/dispatch 双时钟、attempt artifact、SLO gate | 给定 trace 的分母、分位数和失败原因是否正确 |
| 目标运行证据 | Linux/GPU vLLM + 固定 workload | 目标模型、硬件、版本与负载下的质量、容量和 SLO |

前 3 层可在仓库中复算；第 4 层必须由你的目标环境产生，不能用 Windows CPU fixture 或 README 命令代替。

## 最小验收路径 { #run }

先离线复核三份录制报告，不加载 0.5B 权重，也不启动服务：

~~~powershell
python projects/inference-serving/run_qwen_target_service.py --verify projects/inference-serving/qwen2.5-0.5b-service.recorded-report.json
python projects/inference-serving/incremental_streaming_control.py --verify projects/inference-serving/incremental-streaming.recorded-report.json
python projects/inference-serving/transformers_thread_cancellation_control.py --verify projects/inference-serving/transformers-thread-cancellation.recorded-report.json
~~~

再分析 3 个成功、1 个 429 的合成 attempt fixture：

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

最后运行两个选择统计反例：

~~~powershell
python projects/inference-serving/self_consistency_correlation_toy.py
python projects/inference-serving/verifier_best_of_n_toy.py
~~~

相同 0.6 单样本正确率下，N=11 的 independent majority 是 `0.75349813248`，latent-correlated fixture 是 `0.53896454244`。Best-of-N 中 oracle@N、verifier-selected@N 与期望 proxy score 分开计算；N=16 时 oracle 约 `0.9997178890`，selected success 却降到 `0.1852867601`。这两个 authored finite/binary oracle 没有运行模型、tokenizer、judge、PRM 或 provider。

## 三层服务与取消证据

### 固定 Qwen 的 Transformers HTTP reference

已有缓存时可重放目标权重 control：

~~~powershell
python projects/inference-serving/run_qwen_target_service.py --local-files-only
~~~

它在加载前按 immutable revision manifest 重哈希 7 个文件、999,586,347 bytes，以 `trust_remote_code=False` 加载 `Qwen2ForCausalLM`，启动只监听 `127.0.0.1` 的受 Bearer 保护子进程，并真实执行 health/models、错误请求、non-stream 与 SSE。两次 chat 都触发一次 `GenerationMixin.generate()`；录制 report fingerprint 为 `sha256:63e566ca…617ddb`。

这仍是 CPU FP32 eager、单进程、单 admission slot 的教学 reference。它先完成 generation 再发送 SSE chunk，所以 `[DONE]`、usage 和两次 generate 只能证明当前 API/execution contract，不证明 incremental decode、断连取消、KV 释放、vLLM/CUDA、容量或性能。

### Authored async iterator 的断连传播

~~~powershell
python projects/inference-serving/incremental_streaming_control.py
~~~

完整 case 在 backend 完成前交付 `甲`、`🙂`、`终`；断连 case 在首个 `首`/token id 201 后关闭 response，真实 Uvicorn subprocess 观察 ASGI stream task 与 backend iterator 的 `asyncio.CancelledError`，active 回到 0，后续 authored token 未产生。录制 fingerprint 为 `sha256:25846822…2b5d00`。

它只证明专门设计的 async iterator 协作取消，没有 tokenizer、模型 forward、Transformers thread、vLLM、CUDA 或 KV memory 观测。

### Tiny Transformers thread 的显式协作退出

~~~powershell
python projects/inference-serving/transformers_thread_cancellation_control.py
~~~

子进程构造随机 1,272 参数 tiny GPT-2，在 Python thread 中真实执行一次 forward 与 `GenerationMixin.generate()`。Client 收到首 token 后断连，backend 设置 `threading.Event`；authored `StoppingCriteria` 在下一次 termination check 观察事件，使 generation 返回并 join。录制 fingerprint 为 `sha256:eadcab54…f62bc7`。

人为 streamer pause 用于固定竞争窗口；证据只覆盖植入 cooperative event/`StoppingCriteria` 的 tiny CPU 路径。它不证明未修改的阻塞调用、目标 Qwen、已进入的不可中断 kernel、vLLM/CUDA 或 KV/CPU/GPU memory 已释放。

## vLLM 与真实压测

vLLM 的支持平台和版本变化较快。按目标版本的官方安装说明在受支持 Linux/GPU 环境安装，先用适合显存与许可证的模型做单请求 smoke：

~~~bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
~~~

这只是启动示例，不是仓库已经取得的 GPU 结果。不要未经测量就把 context length 和 memory utilization 拉满；先保存 model/runtime/driver/hardware identity，再逐步增加长度与并发。

安装 API 依赖后运行 OpenAI-compatible workload generator：

~~~powershell
python -m pip install -e ".[api]"
python projects/inference-serving/benchmark_openai.py --model Qwen/Qwen2.5-0.5B-Instruct --requests 50 --concurrency 4

python projects/inference-serving/benchmark_openai.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --requests 100 --concurrency 8 `
  --arrival-process constant --request-rate 4

python projects/inference-serving/benchmark_openai.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --requests 100 --concurrency 8 `
  --arrival-process poisson --request-rate 4 --arrival-seed 7
~~~

默认 `burst` 同时 offer 全部有限请求；constant 的第 \(i\) 条在 \(i/\lambda\) 秒到达；Poisson 把首条锚定在 0，后续间隔来自 seeded exponential realization。`--concurrency` 只限制在途 HTTP attempt，不改变预生成 arrival schedule。服务变慢时，请求应积累 client queue，而不是通过等待前一请求完成降低 offered rate。

每个 request 必须保留四个时刻：

- `offered_at = benchmark_started_at + scheduled_offset`；
- `started_at`：取得 client semaphore 后开始 HTTP dispatch；
- `first_token_at`：首个非空 content delta；
- `completed_at`：成功或失败终态。

只从 `started_at` 计时会漏掉 event-loop lag 和 semaphore 前等待，形成 client-side coordinated omission。`client_queue = started_at - offered_at` 也不等于 gateway/vLLM server queue。

报告时分开：

- `success_rate = successful / attempted` 与各类 timeout/429/5xx/protocol/client error；
- all-attempt 的 client queue 和 offered-to-terminal；
- success-conditional 的 dispatch TTFT/E2E/TPOT；
- successful-offered TTFT；
- attempted/successful requests/s 与 successful output tokens/s；
- 服务端 GPU、KV、preemption、queue 与 request-id trace。

SSE chunk 不是 token，`completion_tokens` 必须来自服务端 usage；单 token 输出的 TPOT 未定义；没有成功样本时 latency 是 `null`，不是 0。快速 429 可能让 offered-to-terminal 变小，必须和 success rate 一起读。

当前 generator 是一次性物化有限 coroutine 的教学工具，不是无限到达、bounded pending queue 或分布式 load generator。高 nominal rate 可能在客户端堆积任务并产生费用；先小规模扫描，并监控 generator lag/CPU、预算与紧急停止。

## 解码与选择 oracle

下面命令不加载模型，作用是锁定容易写错的算法契约：

~~~powershell
python projects/inference-serving/sampling_toy.py
python projects/inference-serving/beam_search_toy.py
python projects/inference-serving/constrained_decoding_toy.py
python projects/inference-serving/stop_matching_toy.py
python projects/inference-serving/speculative_decoding_toy.py --seed 23 --trials 20000
~~~

- Sampling 顺序固定为 sign-aware repetition penalty → temperature → exact top-k → top-p → renormalization → token-id-order inverse CDF。Top-p 必须保留第一个让 cumulative probability 达到或越过阈值的 token。
- Beam oracle 明确 active/finished set、EOS、length finish、generated-token-only normalization 与 tie-break；有限 beam 不保证找到全局最高概率序列。
- Constraint oracle 对完整 token fragment 做 trie 状态转移，合法质量为零时 fail closed；它不是 JSON Schema/CFG，也没有 tokenizer byte semantics。
- Stop matcher 对同一 UTF-8 stream 做 strict incremental decode，处理跨 byte/chunk/stop-prefix 边界；客户端命中 stop 不证明服务端停算或停止计费。
- Speculative oracle 逐 token 验证 `min(1,p/q)` acceptance 与 positive `(p-q)` residual。默认一步 acceptance=0.6、TV/rejection=0.4；恒等式通过不证明目标 runtime、KV rollback 或加速。

接入 Transformers、vLLM 或 provider 时，必须把 tokenizer、sampling transform、EOS/stop、length penalty、tie-break、grammar/runtime revision 固定到 token-level differential fixture；“参数名相同”不代表默认语义相同。

## 调度、KV 与 prefix cache oracle

~~~powershell
python projects/inference-serving/continuous_batching_toy.py
python projects/inference-serving/kv_preemption_batching_toy.py
python projects/inference-serving/kv_block_allocator_toy.py
python projects/inference-serving/prefix_cache_toy.py
~~~

Continuous-batching fixture 用离散 scheduler boundary、FCFS admission、decode-first 与 chunked prefill 记录每轮 token-position work。固定 3 请求的 prompt/output 为 7/6 tokens，执行工作量是

\[
W=\sum_i(P_i+O_i-1)=10,
\]

不是计费 token 总数 13；最后一个 prompt position 已产生首个输出分布。离散 slot utilization 不能解释成 GPU utilization、秒、TTFT 或吞吐。

KV-preemption fixture 把 scheduler 与 metadata-only block allocator 连接：容量 3 blocks×2 positions 时发生一次抢占与 2 positions recompute，logical/executed work 为 9/11；6 blocks 对照无抢占且二者都是 9。它不分配真实 K/V tensor，也不是任一 vLLM release 的算法。

Allocator fixture 验证 prefix sharing、partial-tail COW、refcount、fragmentation 与 no-mutation capacity failure。Prefix-cache fixture 即使把 fingerprint function 故意固定成碰撞，也必须继续核对 trusted tenant/visibility/policy/model/tokenizer/template/adapter/RoPE/KV-dtype identity 和 exact token tuple；leased entry 不可被 LRU 淘汰。Unkeyed hash 不能授权或隐藏低熵 prompt。

## Weight、checkpoint 与 KV 量化 oracle

~~~powershell
python projects/inference-serving/quantization_toy.py --seed 17 --bit-width 4 --group-size 8 --output-features 16 --input-features 33 --batch-size 8
python projects/inference-serving/quantized_bundle_toy.py
python projects/inference-serving/minigpt_checkpoint_toy.py
python projects/inference-serving/kv_quantization_toy.py --seed 31 --query-heads 4 --key-value-heads 2 --cached-tokens 8 --query-tokens 3 --key-head-dim 16 --value-head-dim 16
~~~

单矩阵 control 真实执行 symmetric group-wise code、scale、dense bit packing、strict binary artifact 与 reload，但 `quantized_linear` 先反量化为 FP32 NumPy matmul。Bundle 再把两个 name-sorted matrix artifacts 与 model/tokenizer/config identity 放进严格容器。Repo-native MiniGPT checkpoint 更进一步保存 Byte-BPE merges、固定 architecture、全部唯一参数和 tied LM-head contract，并用受信任 loader 恢复 tiny causal LM。

这些 artifact 的 SHA-256 是无密钥内容 identity，不认证来源；exclusive-create + file `fsync` 不证明 parent-directory durability 或崩溃原子性。所有低位矩阵最终仍恢复为 FP32 参数，没有 low-bit GPU kernel、resident VRAM 降低、GGUF/safetensors/vLLM compatibility、速度或目标模型质量证据。

KV oracle 对每个 `[batch, kv_head, token, :]` 向量物化 INT8 code + FP32 scale，再反量化并执行 GQA attention。相同 K/V head dimension \(D\) 时，理想 payload ratio 是 \(4D/(D+4)\)，不是无条件 4×；allocator、alignment 和 workspace 尚未计入。

## 最小验证与故意破坏

项目级验证：

~~~powershell
python -m pytest tests/test_inference_analysis_cli.py tests/test_openai_reference.py tests/test_target_service_control.py tests/test_incremental_streaming_control.py tests/test_transformers_thread_cancellation_control.py tests/test_sampling.py tests/test_beam_search.py tests/test_constrained_decoding.py tests/test_stop_matching.py tests/test_continuous_batching.py tests/test_kv_preemption_batching.py tests/test_kv_allocator.py tests/test_prefix_cache.py tests/test_inference_quantization.py tests/test_quantized_bundle.py tests/test_minigpt_checkpoint.py tests/test_kv_quantization.py tests/test_speculative_decoding.py tests/test_self_consistency.py tests/test_verifier_selection.py -q
~~~

故意失败路径至少覆盖：门禁一次返回全部失败原因、success 行缺 token contract、同一 attempt 文件混用 offered/no-offered 时钟、报告协同重哈希后语义漂移、断连后继续产生 token、KV capacity failure 发生部分 mutation、prefix hash collision 绕过 full identity、量化/checkpoint inner/outer tamper：

~~~powershell
python -m pytest tests/test_inference_analysis_cli.py::test_cli_failure_exit_retains_every_gate_reason tests/test_inference_analysis_cli.py::test_cli_rejects_success_row_without_token_contract tests/test_inference_analysis_cli.py::test_cli_rejects_ambiguous_attempt_artifacts -q
python -m pytest tests/test_target_service_control.py::test_offline_verifier_rejects_cooperatively_rehashed_drift tests/test_incremental_streaming_control.py::test_report_verifier_rejects_cooperatively_rehashed_drift tests/test_transformers_thread_cancellation_control.py::test_verifier_rejects_cooperatively_rehashed_semantic_drift -q
python -m pytest tests/test_kv_allocator.py::test_capacity_failure_is_atomic_before_mutating_an_exclusive_tail tests/test_prefix_cache.py::test_injected_hash_collision_never_bypasses_full_identity_or_token_comparison tests/test_minigpt_checkpoint.py::test_checkpoint_rejects_truncation_outer_and_inner_tamper -q
~~~

参数化 drift 测试会展开多个 case；它们通过的含义是“篡改被拒绝”，不是篡改后的报告有效。

真实部署验收还应固定模型/tokenizer/runtime/container/driver/hardware、prompt 与输入/输出长度联合分布、arrival process、warmup、并发和网络位置；先验证任务/安全质量不退化，再扫描并发直到尾延迟、错误率、KV preemption 或 OOM 不可接受。保存 client attempts、server trace、GPU/KV 指标、原始配置和失败样例，而不只保留 dashboard 截图。

## 证据边界

仓库证明了固定 CPU oracle 的公式/状态机、固定 Qwen Transformers loopback reference、authored async cancellation、显式 cooperative tiny-Transformers thread，以及合成 attempt/SLO 聚合。它没有在本机执行 vLLM、CUDA、PagedAttention、低位或 INT8 KV kernel、目标 GPU workload，也不证明真实 tokenizer/model quality、开放文本 self-consistency、verifier calibration、远程 provider cancellation/billing、TLS/IAM、多 worker、峰值显存、容量、性能或生产 SLO。CPU、loopback 与录制报告的结果不得外推为 GPU/NCCL、目标模型或生产性能结论。

完整实现与每个 fixture 的精确账本见 [projects/inference-serving](https://github.com/NightLemon/about-llm/tree/main/projects/inference-serving)。
