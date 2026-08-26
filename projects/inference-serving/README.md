# 单卡推理服务与压测

这个项目帮助你把“模型能生成文本”推进到“我能解释一次请求、测出容量并定位失败”。你会沿同一条请求观察
prefill、decode、KV Cache、流式输出、取消和资源释放，再用明确的工作负载比较延迟、吞吐与失败率。

第一次学习请从[项目教学页](../../docs/practice/projects/inference-serving.md)开始。它给出完整交付顺序；本页只保留
运行入口、脚本索引和排错信息。当前使用 Qwen3-0.6B 与 nano-vLLM 的读者，可以直接进入
[实验 7B](../../docs/practice/labs/lab-7b-nano-vllm-qwen3.md)。

在长 GPU 实验之前，建议先运行一次真实 Qwen3 对话编码：

```powershell
python projects/transformers-basics/trace_qwen3_tokenizer.py --local-files-only
```

它回答“中文 message 怎样变成模型输入”；实验 7B 的 768 个固定生成 ID 则专门回答“相同长度的输入怎样被调度”。

## 第一次运行

先运行一个不需要 GPU 或模型下载的 Paged KV 实验：

```powershell
python projects/inference-serving/paged_kv_tensor_toy.py
```

运行前先预测：五个前缀 token 会占几个块，产生分支后哪些块共享，继续追加 token 时，哪个未写满的块会触发
写时复制（copy-on-write）。输出中的 `dense parity=true` 只表示分页读写与连续张量参考结果一致。GPU
PagedAttention 要在目标运行时中另行观察。

然后离线复核一份已经录制的 HTTP 报告：

```powershell
python projects/inference-serving/run_qwen_target_service.py `
  --verify projects/inference-serving/qwen2.5-0.5b-service.recorded-report.json
```

这一步不会加载模型或启动服务。它让你先看清 model identity、请求、usage、finish reason 和后端调用次数怎样进入报告。

## Qwen3-0.6B 与 nano-vLLM 主线

这个实验追踪一次请求经过 nano-vLLM 的调度器、KV 块管理器、模型执行器、Qwen3 和采样器。
为了让结果可以复现，仓库使用 nano-vLLM commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，以及 Qwen3-0.6B commit
`c1899de289a04d12100db370d81485cdf75e47ca`。

先在任意 CPU 环境运行工作量账本：

```powershell
python projects/inference-serving/generation_work_ledger.py
```

它会根据同一份实验 Manifest 算出无复用、精确前缀和单 token 漂移三种情况下的 prefill、decode 与总计算位置。
这个结果是后续阅读 GPU trace 的预期值，不代表已经执行了模型或测得 GPU 性能。

在有 CUDA 的 Linux/WSL 环境中运行：

```bash
python projects/inference-serving/nano_vllm_study.py collect \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --source-root /path/to/nano-vllm \
  --model-snapshot /path/to/Qwen3-0.6B/snapshot \
  --output artifacts/inference/nano-vllm-study.json

python projects/inference-serving/nano_vllm_study.py verify \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --report artifacts/inference/nano-vllm-study.json

python projects/inference-serving/nano_vllm_study.py explain \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --report artifacts/inference/nano-vllm-study.json \
  --execution-mode eager \
  --max-num-batched-tokens 256 \
  --prefix-variant one_token_drift
```

`collect` 会先确认源码和模型版本，再运行四组对照：eager 与 CUDA Graph、精确前缀与单 token 漂移、两种
prefill 预算，以及并发 1/2/4/8。

报告按步骤记录序列状态、调度的 token 数和 KV block 使用量，最后汇总 TTFT、TPOT、吞吐和峰值显存。

`verify` 不需要 GPU。它会重新检查版本、时间顺序、指标计算、prefix hit、调度预算和 KV 账本。仓库目前没有把
其他机器的数据写成 RTX 3070 Laptop 结果；你在目标机器生成的报告通过验证后，才是这台机器的实测证据。

`explain` 同样不需要 GPU。它先验证整份报告，再把选中的并发 1 样本整理成逐 step 表格，适合核对 sequence 状态、
cached token、提交的输出和活动 KV block。所选 case 如果失败，命令会直接报告失败阶段。

完整的预测题、trace 阅读方法和 3070 Laptop 建议见[实验 7B](../../docs/practice/labs/lab-7b-nano-vllm-qwen3.md)。

## 从参考协议走到真实服务

在启动 vLLM 前，可以先离线复核三条较小的协议路径：

```powershell
python projects/inference-serving/run_qwen_target_service.py --verify `
  projects/inference-serving/qwen2.5-0.5b-service.recorded-report.json
python projects/inference-serving/incremental_streaming_control.py --verify `
  projects/inference-serving/incremental-streaming.recorded-report.json
python projects/inference-serving/transformers_thread_cancellation_control.py --verify `
  projects/inference-serving/transformers-thread-cancellation.recorded-report.json
```

它们分别检查固定 Qwen 的 HTTP 与用量契约、异步流收到断连后的协作取消，以及 tiny Transformers 生成线程如何
通过显式停止条件退出。这些是三条独立实验，不能合并成“目标 GPU 已正确释放 KV”的结论。

在支持的 Linux/WSL2 与 CUDA 环境中启动 vLLM。下面的模型名和参数只是起点，运行时要绑定实际 revision：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

先核对单请求的 token、usage 和 finish reason，再逐级增加并发。负载发生器默认请求本机 OpenAI-compatible endpoint：

```powershell
python -m pip install -e ".[api]"

python projects/inference-serving/benchmark_openai.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --requests 20 --concurrency 1

python projects/inference-serving/benchmark_openai.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --requests 100 --concurrency 8 `
  --arrival-process poisson --request-rate 4 --arrival-seed 7
```

压测报告必须同时保留成功和失败 attempt。TTFT、TPOT、端到端延迟、请求吞吐和输出 token 吞吐使用不同分母，
不能用一个平均延迟代替整条容量曲线。完整方法见[项目阶段 3–5](../../docs/practice/projects/inference-serving.md#阶段-3建立-workload-contract)。

## 离线检查指标口径

仓库准备了 3 个成功请求和 1 个 429 的合成记录，用来检查统计定义：

```powershell
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
```

这份样例只用于手算分母和时钟。快速返回的 429 可能让 offered-to-terminal 变小，所以任何延迟指标都必须与
success rate 和失败类型一起阅读。

## 根据当前问题选择脚本

| 你想理解什么 | 入口 |
|---|---|
| Sampling、beam search 与停止条件 | `sampling_toy.py`、`beam_search_toy.py`、`stop_matching_toy.py` |
| Constrained decoding 与 speculative sampling | `constrained_decoding_toy.py`、`speculative_decoding_toy.py` |
| Self-consistency 或 verifier 选择为何可能失效 | `self_consistency_correlation_toy.py`、`verifier_best_of_n_toy.py` |
| Continuous batching 与 KV-aware preemption | `continuous_batching_toy.py`、`kv_preemption_batching_toy.py` |
| Paged KV、COW、容量失败和 prefix cache | `kv_block_allocator_toy.py`、`paged_kv_tensor_toy.py`、`prefix_cache_toy.py` |
| Weight/KV quantization 的误差来自哪里 | `quantization_toy.py`、`quantized_bundle_toy.py`、`kv_quantization_toy.py` |
| 一个可加载的最小推理 checkpoint 包含什么 | `minigpt_checkpoint_toy.py` |
| HTTP、SSE 和取消终态 | `run_qwen_target_service.py`、`incremental_streaming_control.py`、`transformers_thread_cancellation_control.py` |
| 目标 GPU 上的 scheduler、Paged KV 和 CUDA Graph | `nano_vllm_study.py` |

这些脚本各自隔离一个机制。精确输入、预期结果和适用范围见
[推理服务证据页](../../docs/evidence/inference-serving-controls.md)，不要把多个 CPU toy 拼成一次完整的 vLLM 运行。

## 主要输入与输出

| 文件 | 用途 |
|---|---|
| `nano-vllm-qwen3-0.6b.study.json` | 固定 nano-vLLM/Qwen3 实验版本、参数与对照组 |
| `attempts.example.jsonl` | 用于手算指标的合成请求记录 |
| `attempts.manifest.example.json` | 绑定输入、时间窗和记录数量 |
| `*.control.json` | 固定参考实验的输入与身份 |
| `*.recorded-report.json` | 可离线复核的已录制结果 |
| `artifacts/inference/` | 本机生成的 trace、attempts 和分析报告 |

真实容量报告还要记录 model/tokenizer/template/runtime、driver、硬件、输入输出长度联合分布、到达过程、并发、
warmup、网络位置和质量 cases。

## 常见故障

| 现象 | 先检查 |
|---|---|
| 首 token 很慢，后续 token 正常 | Prompt 长度、prefill 排队、prefix hit 和 chunked-prefill budget |
| TPOT 随并发恶化 | Decode batch、KV 容量、preemption、GPU 利用率与 scheduler trace |
| 吞吐提高但用户体验变差 | 同时查看 success rate、client queue、TTFT/TPOT p95 和失败分布 |
| Prefix 明明相似却未命中 | 比较精确 token IDs，以及 tenant、model、tokenizer/template 和位置配置身份 |
| Client 已断开，GPU 仍在工作 | 分开检查 response task、generation task/thread、sequence 与 KV release |
| 报告缺少 TPOT | 服务是否返回 completion token usage；单 token 输出的 TPOT 本来就未定义 |
| 并发升高后出现 429 | 区分正确 admission control 与容量不足，并保留全部失败 attempt |
| 估算显存足够但仍 OOM | 对照 peak allocated/reserved、KV blocks、临时 buffer 和 allocator reserve |
| nano-vLLM collect 拒绝运行 | 检查源码 commit/dirty 状态、模型 snapshot、CUDA 环境和 manifest identity |

## 运行检查

```powershell
python -m pytest `
  tests/test_inference_benchmark_entry.py `
  tests/test_inference_analysis_cli.py `
  tests/test_nano_vllm_study.py `
  tests/test_paged_kv_torch.py `
  tests/test_prefix_cache.py `
  tests/test_sampling.py `
  tests/test_quantized_bundle.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

默认检查在 CPU 上验证协议、公式、账本和失败路径，不运行真实 vLLM/CUDA 性能测试。GPU 容量、显存与吞吐结论
必须来自目标硬件、固定 workload 和完整失败分母。
