# 推理服务证据与准确性账本

第一次学习推理系统时，不要从本页开始。先读[一次请求如何穿过推理引擎](../systems/inference-request-lifecycle.md)，
再完成[Paged KV 引导实验](../practice/labs/lab-7a-paged-kv.md)与
[Qwen3 + nano-vLLM 实验](../practice/labs/lab-7b-nano-vllm-qwen3.md)。本页面向内容维护者和项目评审者，
集中回答“教材里的结论由什么独立证据支持，以及这些证据没有证明什么”。

**读者入口**：[推理基础](../systems/inference.md) ·
[推理优化](../systems/inference-optimization.md) ·
[Inference Serving 项目](../practice/projects/inference-serving.md)
{ .doc-nav }

## 证据层级

| 层级 | 本仓库实例 | 可以支持的结论 |
|---|---|---|
| 公式与状态机参考 | sampling、batching、KV allocator、量化 | 固定输入下的公式、排序、状态转移和字节账本 |
| 本地集成实验 | Qwen loopback HTTP、ASGI 取消、tiny Transformers thread | 指定框架、进程和网络路径在当前环境真实执行 |
| Workload contract | attempt artifact、offered/dispatch 时钟、SLO gate | 给定 trace 的分母、分位数、终态和门禁计算 |
| 目标运行证据 | Linux/GPU vLLM 与固定 workload | 指定模型、硬件、版本和流量下的容量、质量与 SLO |

前三层可以在仓库中复算。第四层必须在目标环境产生，不能由 CPU 固定样例、README 命令或全绿测试代替。

## 教材结论与对应证据

| 教材结论 | 主要实现或测试 | 判定依据为什么独立于被测实现 | 仍未证明 |
|---|---|---|---|
| 单序列工作量为 \(P+O-1\)；A/B/C 四轮共处理 10 个位置 | `tests/test_continuous_batching.py` | 手算 `3+3+2+2=10`，再与请求总量 `7+6-3=10` 对账 | Padding、speculation、真实 kernel work 或计费 |
| KV 字节随层、KV heads、head dim、长度和 dtype 增长 | `tests/test_inference_memory.py` | 公式 fixture 与边界输入 | Runtime 对齐、workspace、allocator 或峰值 VRAM |
| Shared partial tail append 必须 COW | `tests/test_kv_allocator.py` | append 前后 block/refcount 独立断言 | vLLM allocator、CUDA 并发或 eviction |
| 容量不足不能留下半更新状态 | `tests/test_kv_allocator.py`、`tests/test_paged_kv_torch.py` | 异常前后 allocator 与 tensor 全量比较 | 异步 CUDA 故障的事务回滚 |
| Paged K/V 可按逻辑顺序恢复并保持 GQA causal attention | `tests/test_paged_kv_torch.py` | 独立 dense mask、head expansion 和 softmax | GPU PagedAttention kernel、速度或显存收益 |
| Sampling 需要固定 processor、top-k/top-p 和 RNG 映射 | `tests/test_sampling.py` | 有理/NumPy 概率与边界反例 | Provider 或 runtime 默认语义 |
| Stop string 可能跨 byte、token 和 SSE chunk | `tests/test_stop_matching.py`、`tests/test_sse.py` | 分片、UTF-8 和 overlap 负例 | 服务端停算、KV 释放或计费终止 |
| TPOT 对单 token 输出未定义 | `tests/test_inference_metrics.py` | 时间戳公式和 typed `None` | 目标服务时钟质量与网络归因 |
| Client queue 不能从 dispatch TTFT 中恢复 | `tests/test_inference_workload.py` | offered/started/terminal 四时钟 fixture | Gateway/runtime 内部 queue |
| HTTP 完成不自动证明目标权重执行 | `tests/test_target_service_control.py` | 固定 Qwen manifest、子进程和 generate audit | vLLM/CUDA、质量或生产容量 |
| 断连传播与模型停止、KV 释放是三份证据 | `tests/test_incremental_streaming_control.py` 与 tiny-Transformers recorded verifier | ASGI task 与显式 cooperative generation thread 分开观测 | 未修改 runtime、不可中断 kernel 或 GPU KV release |
| 固定 nano-vLLM trace 可对账 phase、prefix hit 与 KV 释放 | `tests/test_nano_vllm_study.py` + 目标 GPU report | CPU verifier 独立复算 schema、时间、指标和 KV 不变量；GPU runner 观察真实 step | HTTP SLO、模型质量、跨引擎排名或其他硬件版本 |

测试名称表达的是证据类型，不表达“整个系统正确”。例如 attention parity 测试说明当前固定输入的数值等价，
不能证明实现已经使用 PagedAttention GPU kernel。

## 快速正确性门禁

下面一组测试覆盖推理教材最容易写错的公式、状态机和协议边界：

~~~powershell
python -m pytest `
  tests/test_inference_memory.py `
  tests/test_inference_metrics.py `
  tests/test_sampling.py `
  tests/test_stop_matching.py `
  tests/test_sse.py `
  tests/test_continuous_batching.py `
  tests/test_kv_allocator.py `
  tests/test_paged_kv_torch.py `
  tests/test_kv_preemption_batching.py `
  tests/test_nano_vllm_study.py `
  -q
~~~

其中必须保留的失败路径包括：

~~~powershell
python -m pytest `
  tests/test_kv_allocator.py::test_capacity_failure_is_atomic_before_mutating_an_exclusive_tail `
  tests/test_paged_kv_torch.py::test_capacity_failure_preserves_allocator_and_tensor_state `
  -q
~~~

门禁通过只能说明这些已选择的 oracle 没有回归。教材结论仍需检查变量定义、适用假设和外推范围。

## 本地 HTTP 与取消 control { #local-http-cancel }

这三条路径的证据不能相互借用：

~~~powershell
python projects/inference-serving/run_qwen_target_service.py --verify `
  projects/inference-serving/qwen2.5-0.5b-service.recorded-report.json
python projects/inference-serving/incremental_streaming_control.py --verify `
  projects/inference-serving/incremental-streaming.recorded-report.json
python projects/inference-serving/transformers_thread_cancellation_control.py --verify `
  projects/inference-serving/transformers-thread-cancellation.recorded-report.json
~~~

- Qwen control 证明固定权重、tokenizer、HTTP 和两次 `generate()` audit 的当前录制契约。
- Async iterator control 证明专门设计的 ASGI/backend task 能观察断连并停止继续产生 authored token。
- Tiny Transformers control 证明显式植入 event 与 `StoppingCriteria` 后，真实 `generate()` thread 能退出并 join。

三者合起来仍没有执行 vLLM scheduler、CUDA kernel 或 GPU KV release，也没有测量目标服务质量与性能。

## nano-vLLM 目标 GPU study { #nano-vllm-study }

这条路径把“目标运行证据”再拆成 runner 与 recorded evidence 两步：

1. 仓库固定 source/model manifest，提供 GPU collector 与 CPU verifier。
2. 用户在真实 3070 Laptop WSL 环境运行，回传脱敏 `about-llm.nano-vllm-study.v1` JSON。
3. 先离线验证 strict schema、identity、计时/指标算术和 KV 账本，再人工审查硬件、失败和证据边界。
4. 只有前三步都通过，才把具体数字提升为仓库 recorded evidence。

~~~bash
python projects/inference-serving/nano_vllm_study.py verify \
  --manifest projects/inference-serving/nano-vllm-qwen3-0.6b.study.json \
  --report artifacts/inference/nano-vllm-study.json
~~~

CPU 测试使用 synthetic report 检查 verifier 自己会拒绝 duplicate key、NaN、revision 漂移、时间倒流、
指标篡改、prefill budget 超限和 KV 账本不守恒。它不声称 synthetic timing 是 GPU 性能。

GPU collector 才会真实执行固定 Qwen3 权重、nano-vLLM Scheduler/BlockManager/ModelRunner、FlashAttention、
Triton KV store、Sampler 和符合条件的 decode CUDA Graph。报告中的 TTFT/TPOT/E2E 是 engine 内部时钟，
不含 HTTP、tokenization、网络和 client queue。当前仓库尚未录入 3070 report，因此不能引用具体性能数字。

## Workload 与 SLO 证据

离线 attempt fixture 用来验证统计口径，不用来证明服务达到 SLO：

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

真实目标环境还必须保存：

- model、tokenizer、runtime、driver、容器和硬件 identity；
- prompt/output 长度联合分布、arrival process、warmup 和并发；
- 每个 attempt 的原始终态与四个时间戳；
- 服务端 queue、KV、preemption、GPU 和 request-id trace；
- 质量、安全、容量和失败样例，而不只是一组延迟分位数。

## Paged KV fixture 的精确边界

`projects/inference-serving/paged_kv_tensor_toy.py` 使用固定 CPU float64 arena：

```text
K/V shape: [layers, physical_blocks, kv_heads, tokens_per_block, head_dim]
fixture:   [1, 4, 2, 3, 2]
payload:   768 bytes
```

它真实存储、复制和清理 K/V tensor，并将四个 query heads、两个 KV heads 的结果与独立 dense causal reference 对账。
Attention 路径仍先 materialize 完整序列并构造 dense scores，所以必须明确记录：

- `paged_attention_gpu_kernel_executed = false`；
- `scheduler_or_model_decode_integrated = false`；
- `latency_throughput_or_vram_proved = false`。

这里的 `resident_bytes` 仅为两块预分配 tensor 的确定性 payload，不能写成进程 RSS、峰值内存或 VRAM。

## 内容变更时怎样审查

修改推理教材或实现时，按结论审查，而不是机械增加用例：

1. 写出正文声称了什么，以及限定在哪种生成、调度或硬件假设下。
2. 找到对应实现和测试，确认 oracle 没有复制被测实现的同一错误。
3. 检查正常、边界和失败路径，尤其是容量、取消、终态和未定义指标。
4. 检查测试名称与正文是否夸大了证据等级。
5. 把目标模型质量、GPU 性能和生产安全留给目标环境验证。

完整实现、录制报告与专项命令位于
[projects/inference-serving](https://github.com/NightLemon/about-llm/tree/main/projects/inference-serving)。
