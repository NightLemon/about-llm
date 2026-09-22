# Inference Serving：从一个请求到容量报告

**项目导航**：[项目索引](../project-index.md) · [请求生命周期](../../systems/inference-request-lifecycle.md) ·
[推理优化](../../systems/inference-optimization.md) · [vLLM 服务](../../systems/vllm-serving.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/inference-serving/README.md) ·
[证据台账](../../evidence/inference-serving-controls.md)
{ .doc-nav }

这个项目把 KV、调度、HTTP、目标运行时和负载实验接成一份容量报告。本页只负责阶段顺序、升级条件和交付物；
完整命令与脚本选择由 README 维护，机制由系统正文解释。

## 最终交付物

一份可审阅报告至少包含：

1. 模型、tokenizer、template、runtime、driver、硬件与启动参数 identity。
2. 固定输入/输出长度联合分布、到达方式、并发、时限与生成参数。
3. 一条 request ID 串起客户端、HTTP、scheduler、prefill/decode 和 KV 释放的 trace。
4. 多档负载下的完整终态、queue、TTFT、TPOT、吞吐和显存。
5. 至少一个取消、容量不足或 OOM 反例及恢复结果。
6. 只适用于哪些版本和 workload，以及可执行回滚条件。

## 阶段 0：先复述一次请求 {#run}

先画出 `offered → dispatched → admitted → prefill → first token → decode → terminal → KV release`。如果这些状态和
时钟还混在一起，先读[请求生命周期](../../systems/inference-request-lifecycle.md)，不要直接启动 vLLM。

## 阶段 1：观察 Paged KV {#first-kv-run}

完成[实验 7A](../labs/lab-7a-paged-kv.md)。通过条件是能预测 block table、refcount、partial-tail COW、释放和容量不足
时的原子失败。CPU tensor parity 只验证状态与数值顺序，不证明 GPU PagedAttention 性能。

## 阶段 1B：让 Qwen3 穿过 nano-vLLM

完成[实验 7B](../labs/lab-7b-nano-vllm-qwen3.md)，观察固定输入怎样经过 `WAITING/RUNNING/FINISHED`、prefill、decode、
prefix cache 和 block 释放。它是目标 checkpoint/运行时路径，不是 HTTP 服务容量结论。

## 阶段 2：把协议和终态跑通

先运行本地参考 HTTP/SSE 服务，确认非流式、流式、usage、finish、非法输入和断连都有可观察终态。客户端关闭、
生成任务取消与底层资源释放必须分别记录。命令见[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/inference-serving/README.md)。

## 阶段 3：建立 workload contract

在压测前固定：

| 维度 | 必须记录 |
|---|---|
| Identity | 模型/运行时 revision、dtype/quantization、硬件与功耗模式 |
| 请求 | Prompt/output 长度联合分布、模板、采样、stop、流式 |
| 到达 | 同时、固定间隔、Poisson 或回放；客户端并发限制 |
| 终态 | Success、429/503、timeout、cancel、protocol error、OOM |
| 时钟 | offered、started、first token、completed；服务端 queue 与 phase |
| 资源 | KV block 高水位、allocated/reserved 显存、CPU 与功耗 |

只统计成功请求或只报告平均 tokens/s，会把过载和快速失败隐藏起来。

## 阶段 4：在目标 GPU 上逐级加压

先完成非流式单请求与流式单请求，再按并发或到达率逐档增加。每档预热后保留稳定窗口和全部失败；第一个违反质量、
尾延迟、成功率或资源门槛的点也要保存。最后一个同时满足门槛的档位，才是这份 workload 下的容量点。

目标环境的 vLLM 参数、兼容性和观测项见[vLLM 服务](../../systems/vllm-serving.md)。仓库没有随代码提交目标 GPU 数字。

## 阶段 5：一次只改变一个优化变量

在同一 workload 上分别比较 batching/token budget、prefix cache、quantization、CUDA Graph 或 speculative decoding。
每次都重新检查质量、完整终态、TTFT、TPOT、吞吐和显存；payload 变小、引擎内时间变快或某次输出一致，都不能单独
升级成服务结论。

## 故意破坏清单

- 给 Paged KV 一个不足以完成 COW 的 block budget。
- 让共享前缀只漂移一个 token，确认 cache miss。
- 在首段流式文本后断开，分别观察任务、序列和 KV 释放。
- 让负载发生器排队，确认 `offered → started` 没从延迟里消失。
- 让同一模型切换 eager/graph 或量化路径，确认实际 backend 与峰值资源被记录。

## 报告模板

```text
目标与 SLO：
Identity 与 workload：
一次请求 trace：
容量曲线与失败分母：
取消/OOM/过载结果：
单变量优化对照：
证据边界与回滚：
```

## 项目完成标准

报告必须让另一位读者复算时间和容量口径，并能指出瓶颈发生在客户端、服务队列、prefill、decode、KV 还是协议层。
若只有启动截图或单次 tokens/s，本项目尚未完成。
