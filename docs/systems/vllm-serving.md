# vLLM 与 OpenAI-compatible 单卡服务

vLLM 的价值不只是“启动一个 API”，而是 PagedAttention、continuous batching、prefix caching 和调度把多个变长请求高效放到 GPU。生产部署仍需模型许可、容量、限流、观测、升级和回滚。

## 适用边界

vLLM 主要面向 Linux 与受支持 GPU/加速器。Windows 开发者通常用 WSL2 或远程 Linux。模型架构、量化格式、CUDA/driver、PyTorch 和 vLLM 版本有兼容矩阵；先查目标 release 文档并锁版本。

本仓库当前机器无 CUDA/vLLM，因此只离线验证 SSE 解析、请求协议和性能指标。下面命令是目标环境运行路线，不是本机已验证声明。

## 最小启动

以下命令形状已按 vLLM CLI 核对，但参数默认值和支持矩阵随版本变化；生产应使用所安装版本的 `vllm serve --help=all`，并优先查阅 `stable` 文档，而不是 `latest` 开发预览。概念性命令：

~~~bash
vllm serve MODEL_ID \
  --revision COMMIT_HASH \
  --served-model-name my-model \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 \
  --port 8000
~~~

生产不要默认绑定公网；在前面放认证、TLS、限流和 request size 上限。`trust_remote_code` 默认关闭，确需开启时先审计并固定 revision。

## 容量参数

`max-model-len` 决定单序列上限，也影响 KV 容量规划。设成模型理论最大值可能显著减少可并发序列；根据业务 p99 和必要上限选择。超长请求应在网关提前拒绝或路由专用池。

`gpu-memory-utilization` 给 runtime 权重/KV/workspace 的使用目标，过高会在峰值/其他进程下 OOM，过低浪费并发。记录启动后的 KV block 数和最大 concurrency 估计，再用真实压力验证。

调度 token/sequence 上限控制一次迭代容纳多少 prefill/decode。优化吞吐会影响 TTFT/TPOT，必须画 Pareto curve。

## 模型与 tokenizer

固定 model、tokenizer revision、chat template 和 generation defaults。OpenAI-compatible `messages` 由 server template 渲染；客户端若又手工套模板会重复 role token。上线前用 token ids 或 echo 工具核对。

adapter/LoRA serving 要限制允许的 adapter、来源和每请求切换；动态加载是代码/权重供应链边界。多 adapter batching 的性能与显存单独测。

## 量化选择

vLLM 支持的 AWQ/GPTQ/FP8/bitsandbytes 等依版本和硬件变化。选择“runtime 原生高效 kernel + 任务质量通过”的组合，不按文件后缀决定。启动日志要确认实际 quantization/backend，没有静默 fallback。

量化模型仍需 KV cache 和 workspace。长上下文服务可能 KV 成为主导，此时只压权重收益有限；考虑 GQA 模型、KV dtype/quantization（若可靠支持）或更严格 context limit。

## API 与流式协议

OpenAI-compatible 并不保证所有扩展完全相同。契约测试覆盖：model id、messages/content、多模态结构、tools、response_format、logprobs、usage、finish reason、错误 schema 和 SSE `[DONE]`。

流式客户端要处理：一个 TCP chunk 多个 event、一个 event 跨 chunk、空 keepalive、多行 data、UTF-8 分片、错误事件和提前断开。仓库 `about_llm.inference.sse` 用增量 parser 验证这些边界。

客户端取消后服务端应尽快停止 decode 并释放 KV；监控 disconnect-to-release 延迟。代理层必须关闭会破坏 SSE 的缓冲。

## 指标定义

- TTFT：发送请求到收到首个内容 token；包含排队、prefill 和网络。
- TPOT：首 token 后相邻输出 token 的平均时间，通常 `(last-first)/(n-1)`。
- ITL：每个 token 间隔的分布，比单一 TPOT 更细。
- E2E latency：完整请求时间。
- request throughput：完成请求/秒。
- token throughput：prompt/output/total tokens/秒，必须说明分母。

流式 chunk 不等于 token，一个 chunk 可能含多个 token或只有 role/usage。优先用 server usage/tokenizer 计数；不能把 SSE event 数当 token 数。

仓库基准脚本在缺少 `completion_tokens` 时会明确失败。若目标服务不返回流式 usage，应使用与服务端完全相同 revision 的 tokenizer 对完整输出重新计数，并把计数来源写入结果；不能静默退化为 chunk 计数。

## 压测方法

仓库 `benchmark_openai.py` 对 OpenAI-compatible SSE 记录 TTFT、TPOT 和吞吐。生产基准需要 workload manifest：

- 输入/输出长度联合分布，而非固定一句话；
- 并发或到达过程（closed-loop/open-loop）；
- warmup 与稳定测量窗口；
- sampling、stop、tools/logprobs；
- 成功/错误/取消分别统计；
- 客户端与服务端时钟/网络位置。

Closed-loop 每个 worker 完成后再发，系统变慢时自动降低到达率，可能掩盖过载；open-loop 按固定速率发，更适合找饱和点，但要限制队列和总成本。

报告 p50/p90/p95/p99，不只平均。逐步增加 offered QPS，找到队列延迟陡增的 knee；生产容量留余量并考虑故障少一副本。

## Admission control

网关在进入 GPU 前限制认证主体、并发、prompt token、max output、request bytes 和费用。将超长/低优先级请求排队或路由 batch 池。无限队列只把拒绝变成超时并耗尽内存。

按 tenant/user 令牌桶，避免单个用户占满 KV。优先级调度要防低优先级 starvation。返回 429/503 与 `Retry-After`，客户端带 jitter 重试。

## 扩缩容与路由

单卡先一进程一 GPU，避免两个 runtime 抢显存。多副本在网关按可用 KV/队列而非简单 round-robin 路由；prefix cache locality 与负载均衡有冲突。

启动加载权重慢，autoscaling 不能只看当前 GPU utilization；结合队列、KV 使用、arrival rate 和启动时间。scale-to-zero 适合低频离线，不适合严格 TTFT。

模型超过单卡才用 tensor parallel；它增加跨卡通信和故障域。小模型通常 data-parallel 多副本吞吐/隔离更简单。

## 可观测性

服务端监控 request queue、running/waiting、prefill/decode tokens、KV utilization、cache hit、batch size、TTFT/ITL、GPU SM/显存/功耗、OOM/429/取消和 finish reason。版本标签含 model/revision/quant/runtime/config，但 request id 放 trace 不放 metric label。

客户端观测与服务端结合：高 TTFT 可能来自网关/网络，只有服务端 token latency 无法解释用户体验。

## 安全

- 网关认证与 TLS，API key 不写日志；
- 关闭任意 model/adapter/path 参数；
- 限制 prompt 和 output，防资源耗尽；
- 模型权重、remote code 和 tokenizer 供应链校验；
- 对工具调用/结构输出做 schema 和权限验证；
- prompt/response 日志按敏感等级采样与删除；
- 容器非 root、只读权重、最小网络和健康端点隔离。

OpenAI-compatible 只是数据协议，不提供自动安全和多租户隔离。

## 升级与回滚

新模型/runtime/config 启动独立副本，完成 smoke、离线质量、最大 context、量化、SSE、OOM 恢复和性能基准。再 shadow/canary，按 model version 路由。回滚保留旧镜像、权重和 tokenizer；缓存 key 含版本，避免新旧混用。

滚动升级时长连接可能仍在旧副本，先 readiness=false 停新请求，等待/有界终止现有 decode。不能直接杀进程并把断流算成功。

## 故障处理

- CUDA OOM：停止接新请求、让失败明确返回；调查长度/并发/碎片，不无限重启。
- worker 崩溃：readiness 移除，网关仅对安全请求有限重试。
- 模型加载失败：保留旧副本，检查 revision/磁盘/格式。
- 延迟升高：拆 queue、prefill、decode、网络和 tokenizer；看长度分布是否漂移。
- 输出乱码/模板错：比较 tokenizer/template revision 与客户端渲染。

## 单卡验收清单

1. 固定硬件/软件/model revision 和许可；
2. 1/4/8/…并发、短/长输入输出基准；
3. 记录质量、TTFT/TPOT/E2E、吞吐、显存和功耗；
4. max context、取消、超时、429、SSE 分片与 OOM 测试；
5. 认证、限流、日志脱敏和模型供应链检查；
6. canary 与一键回滚演练；
7. 明确哪些结论只对该 GPU/版本/workload 成立。

## 面试追问

**提高并发为何 TTFT 变差？** 请求排队且调度 batch 变大，prefill 争用计算/KV；吞吐提高是以单请求等待为代价。容量选择看 SLO 下吞吐。

**如何定位 TPOT 退化？** 按输出长度/并发切片，看 decode batch、KV 使用、GPU 带宽/利用率、quant kernel、其他进程和调度；TTFT 正常可先排除大部分 prefill/队列问题。

**为什么 OpenAI-compatible 不能保证直接替换？** 基础路径相似，但 tools、JSON schema、多模态、usage、错误、logprobs、finish reason 和扩展字段可能不同，需 provider contract tests。
