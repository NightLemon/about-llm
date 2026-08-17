# vLLM 与 OpenAI-compatible 单卡服务

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：在 Linux/GPU 上部署和验收 vLLM 的工程师。
- **先修**：[推理基础](inference.md)、Linux、GPU 显存和 HTTP 流式协议。
- **首次阅读**：适用边界 → 最小启动 → 容量参数 → tokenizer → 单卡验收。
- **完成信号**：能在目标环境记录 revision、VRAM、TTFT/TPOT 和失败样例。
- **卡住时**：先用[环境矩阵](../guide/environment.md)确认平台与 wheel，不在 Windows 原生环境硬装。

</div>

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

### Continuous batching 要先固定调度语义

“动态把请求放进 batch”还不足以定义可复现实验。至少要说明 arrival/admission boundary、sequence cap、每轮 token budget、prefill chunk、decode 与 prefill 的优先级、preemption，以及 prefill 完成后首 token 在哪个 boundary 可见。仓库的 `simulate_continuous_batching` 固定一份 decode-first CPU 教学策略：FCFS admission；每个 decode-ready request 每轮一个位置；每个 resident prefill 至少一个位置；剩余 prefill budget 再按 FCFS 分配，并要求 `max_batch_tokens >= max_running_sequences` 保证进展。

若请求 \(i\) 有 \(P_i\) 个 prompt token、恰好发出 \(O_i\ge1\) 个输出 token，且无 prefix reuse、speculative verification 或 beam，那么该 oracle 的 causal forward positions 是

\[
W=\sum_i(P_i+O_i-1).
\]

减一是因为 prefill 的最后一个 prompt position 已给出首个输出 token 的分布；后续 \(O_i-1\) 个 token 才各需一个 decode position。不能把 `prompt_tokens + completion_tokens` 直接叫作模型执行 token，也不能反过来用 forward work 改写 API 计费/输出 token。Toy 的 3 个请求有 prompt/output 7/6、实际调度 work 10，逐轮 used slots 为 `3,3,2,2`。

离散 step 不是秒；toy 的 token-slot utilization 不是 GPU utilization，FCFS/decode-first 也不证明 vLLM scheduler equivalence。真实 runtime 还受 KV admission、prefix cache、preemption、priority、padding/kernel shape 和版本策略影响，必须保存 server trace 并在目标硬件测量。

### 抢占后必须把重计算另记一笔

仓库另提供 `simulate_kv_preemption_batching`，把调度 boundary 与 metadata-only `PagedKVAllocator` 接在一起。每处理一个 causal position 就 append 一个 KV position；当新 block 不足时，固定策略只允许抢占本轮尚未工作且 stable-FCFS 优先级更低的 resident，并从中选择最近 admission 者。严格单向的抢占关系防止两个 rebuild 请求互相驱逐。被抢占请求释放全部 block，按 FCFS 再 admission 后先重建已完成的 logical context；rebuild 只恢复 KV，不重复发出已经返回给用户的 token。已经在本轮选中 work 的请求受保护，完成请求到 boundary 才释放 block，避免把同一 interval 内尚未完成的 KV 提前借给别人。

因此逻辑模型工作与实际执行工作要分开：

\[
W_{\text{logical}}=\sum_i(P_i+O_i-1),\qquad
W_{\text{executed}}=W_{\text{logical}}+R,
\]

其中 (R) 是抢占后重复 prefill/recompute 的 position 数。固定 fixture 有 3 个、每块 2 token 的物理 block；A/B 的 logical work 分别为 6/3。A 增长时抢占已缓存 2 positions 的 B，B 恢复后重算 2 positions，所以逻辑 work=9、实际 work=11、overhead=2/9；逐轮 used slots 为 `3,3,1,1,2,1`。B 的 token emission 仍只有 boundary 2 和 6 两次。足够 KV 的对照保持 logical=executed=9、preemption=0。

这份策略不是 vLLM scheduler 的复刻。它没有 K/V 数值、swap、prefix reuse、priority aging、distributed worker、CUDA kernel 或墙钟；block 数也不是目标 GPU 的真实 VRAM。它证明的是在这份明确 policy 下，admission、释放、抢占、重建、token emission 与 work ledger 彼此一致。

### Paged KV 的容量不能只看 token 总数

每条序列有自己的 logical block table；多个表可通过 refcount 共享 prefix physical block。共享的 full block append 不需复制，shared partial tail append 必须 COW，否则会改坏其他序列。容量 admission 要预留完成本次 append 所需的全部新块，再修改 tail/length；只看“当前 tail 还有一个空位”会漏掉同一 append 后续还需新块，造成半更新。

仓库 `PagedKVAllocator` 用 CPU metadata 回归 block partition、occupancy/refcount、prefix fork、partial COW、释放复用、物理碎片和 no-capacity atomic failure。`logical_tokens` 按序列重复共享 prefix，不能拿它直接计算 physical fragmentation；后者用 allocated slots 减 physical materialized positions。它只证明 metadata 状态机：不保存 K/V、未执行 PagedAttention kernel，也没有 vLLM eviction/preemption/swap 或真实 VRAM 证据。

### Prefix cache identity 是安全边界

Prefix reuse 还需要独立于 block allocator 的 identity：可信 tenant/visibility domain、authorization 与 policy revision、model/tokenizer/chat template/adapter revision、RoPE/position config、KV dtype 和 exact token ids。Raw text hash 不包含这些语义；SHA-256 也只能作索引提示，不能替代 full comparison、授权或保密。仓库 `PrefixCache` 用 injected collision、longest exact-prefix、lease-pinned LRU 和全 leased 原子失败测试这些规则，但不声称模拟某版 vLLM 的 block hash、eviction policy、timing 或 GPU storage。部署参数和 cache 行为必须按实际安装版本核对并压测。

## 模型与 tokenizer

固定 model、tokenizer revision、chat template 和 generation defaults。OpenAI-compatible `messages` 由 server template 渲染；客户端若又手工套模板会重复 role token。上线前用 token ids 或 echo 工具核对。

部署前把 tokenizer、model config、generation config 的 BOS/EOS/PAD/decoder-start ID 三方对账，但不要要求机械相等：generation EOS 可以是 tokenizer EOS 的有意 superset，PAD 与 EOS 也可能重合。真正危险的信号是未解释的 disjoint、越过 tokenizer/model vocab 上界，或客户端、server 与离线基线采用不同 stop 集。仓库 generation-protocol inspector 只比较 normalized snapshots；vLLM CLI/request override、server fallback、stop strings/tokenization 与实际 finish reason 仍必须在目标版本做 token-level 契约测试。

仓库的 Transformers generation runtime control 已证明在当前依赖版本和强制 token plan 下 call-level EOS/max-new-token override 的三条路径；它不运行 vLLM，不能外推 vLLM 的 CLI/server/request precedence。跨 runtime 对照应发送完全相同的 prompt token IDs，显式设置 EOS/length/sampling，保存逐 token 输出与服务端 finish reason，并把“token 序列一致”和“停止原因字段一致”分开判定。

adapter/LoRA serving 要限制允许的 adapter、来源和每请求切换；动态加载是代码/权重供应链边界。多 adapter batching 的性能与显存单独测。

## 量化选择

vLLM 支持的 AWQ/GPTQ/FP8/bitsandbytes 等依版本和硬件变化。选择“runtime 原生高效 kernel + 任务质量通过”的组合，不按文件后缀决定。启动日志要确认实际 quantization/backend，没有静默 fallback。

量化模型仍需 KV cache 和 workspace。长上下文服务可能 KV 成为主导，此时只压权重收益有限；考虑 GQA 模型、KV dtype/quantization（若可靠支持）或更严格 context limit。

## API 与流式协议

OpenAI-compatible 并不保证所有扩展完全相同。契约测试覆盖：model id、messages/content、多模态结构、tools、response_format、logprobs、usage、finish reason、错误 schema 和 SSE `[DONE]`。

本仓库的 target-service control 是对这份契约测试方法的窄 reference：固定 Qwen2.5-0.5B-Instruct revision，逐文件重哈希后用 Transformers CPU FP32 eager 加载，真实启动 loopback subprocess，并验证 Bearer、models、unknown field、wrong model、non-stream/SSE、精确 usage/finish 与后端两次 `generate()`。它没有运行 vLLM，接口只实现 closed chat subset；不能把 Uvicorn 0.52.1 重录报告 `sha256:63e566ca…617ddb` 写成 vLLM compatibility、GPU 性能或生产安全结论。迁移到 vLLM 时应以同一 checkpoint/prompt/token trace 重跑，而不是继承 reference 的通过状态。

流式客户端要处理：一个 TCP chunk 多个 event、一个 event 跨 chunk、空 keepalive、多行 data、UTF-8 分片、错误事件和提前断开。仓库 `about_llm.inference.sse` 用增量 parser 验证这些边界。

SSE framing 与 stop matching 仍是两层：前者还原 event，后者在 decoded output text 上匹配可能跨 event/token 的 stop。仓库 `IncrementalStopMatcher` 通过 longest partial-prefix withholding 避免提前泄露 stop 前缀，并固定 first-completion/配置顺序 overlap 语义；它不是 vLLM stop 实现，也不能从客户端截断推断服务端 finish reason、usage、KV release 或计费停止。目标版本必须用 token/event trace 单独做契约测试。

客户端取消后服务端应尽快停止 decode 并释放 KV；监控 disconnect-to-release 延迟。代理层必须关闭会破坏 SSE 的缓冲。

还要辨认“post-completion SSE”：若 backend 先生成完整答案再返回 `StreamingResponse`，client 虽然看见多个合法 event 和 `[DONE]`，首个 event 已经发生在模型计算结束后，既没有增量 TTFT 优势，也无法用断流证明取消 model work。仓库 reference 明确属于这一类，只把它用于协议投影对账。

仓库的第二条 incremental control 使用 authored cooperative async iterator 和真实 Uvicorn loopback subprocess：它验证 content 在 backend 完成前可见，且 client close 会把 `CancelledError` 传播到 ASGI stream task/backend iterator，后续 scripted token 不再产生。Uvicorn 0.52.1 重录 report `sha256:25846822…2b5d00` 没有加载 vLLM、模型、CUDA 或 KV；迁移到 vLLM 时必须重新关联 request id、scheduler sequence、decode step、block allocator release 与 client disconnect，分别验收“停止调度”和“资源已释放”，不能继承这个 control 的通过状态。

第三条 control 真实执行随机 tiny GPT-2 的 CPU `GenerationMixin.generate()` thread，并用 authored streamer pause + `threading.Event` + `StoppingCriteria` 让断连在第一次 forward/首 token 后终止 generation、join thread；Uvicorn 0.52.1 重录报告为 `sha256:eadcab54…f62bc7`。这比纯 async iterator 多证明了一层真实 Transformers loop，但 cooperative hook 与暂停窗口都是 control 自己植入的；它没有 vLLM scheduler、CUDA kernel、Paged KV block 或目标模型。vLLM 验收仍须看目标版本自己的 abort API、scheduler trace 和 block-release trace，不能把 Python thread join 等同于 engine sequence/KV 已释放。

## 指标定义

- Dispatch TTFT：真正开始 HTTP attempt 到收到首个内容 token，包含网关/服务端排队、prefill 和网络，但不含负载生成器本地 semaphore 前等待。
- Offered TTFT：请求按 workload arrival process 进入客户端到首 token；它等于该请求的 client queue 加 dispatch TTFT，但 **p95 不能由两个 p95 相加得到**。
- TPOT：首 token 后相邻输出 token 的平均时间，通常 `(last-first)/(n-1)`。
- ITL：每个 token 间隔的分布，比单一 TPOT 更细。
- E2E latency：完整请求时间。
- request throughput：完成请求/秒。
- token throughput：prompt/output/total tokens/秒，必须说明分母。

流式 chunk 不等于 token，一个 chunk 可能含多个 token或只有 role/usage。优先用 server usage/tokenizer 计数；不能把 SSE event 数当 token 数。

仓库基准脚本在缺少 `completion_tokens` 时会明确失败。若目标服务不返回流式 usage，应使用与服务端完全相同 revision 的 tokenizer 对完整输出重新计数，并把计数来源写入结果；不能静默退化为 chunk 计数。

### 成功延迟是条件统计

若只对成功请求计算 p95 TTFT，就得到 \(P(TTFT\mid success)\)，不是所有用户尝试的体验。失败请求没有完整 TTFT/TPOT 时不能填 0，也不能从行集中删除后只报漂亮 percentile。至少联合报告：

\[
success\ rate=\frac{N_{success}}{N_{attempted}},
\qquad
successful\ RPS=\frac{N_{success}}{wall\ time}.
\]

Attempted RPS、successful RPS 与 offered arrival rate 是不同量。429 可能表示 admission control 正常保护系统，也可能表示容量不足；必须结合租户配额、offered load 和 `Retry-After` 判断。

客户端也可能制造 client-side coordinated omission。若 N 个任务先等待本地 concurrency semaphore，取得槽位后才开始计时，则最慢时请求只是更晚发出，TTFT 样本里看不到此前等待。Trace 应同时保存 `offered_at` 和 HTTP dispatch `started_at`，分别报告 client queue、成功请求的 offered TTFT，以及全 attempt 的 offered-to-terminal time。后者会把快速失败也当作快速终态，不能替代 success rate。

没有成功样本时 percentile 应为 unavailable/null，而不是 0。只有一个输出 token 时没有 post-first-token interval，TPOT 同样未定义。

## 压测方法

仓库 `benchmark_openai.py` 对 OpenAI-compatible SSE 记录 TTFT、TPOT 和吞吐。生产基准需要 workload manifest：

- 输入/输出长度联合分布，而非固定一句话；
- 并发或到达过程（closed-loop/open-loop）；
- warmup 与稳定测量窗口；
- sampling、stop、tools/logprobs；
- 成功/错误/取消分别统计；
- 客户端与服务端时钟/网络位置。

Closed-loop 每个 worker 完成后再发，系统变慢时自动降低到达率，可能掩盖过载；open-loop 按固定速率发，更适合找饱和点，但要限制队列和总成本。

仓库在线脚本支持三种有限 arrival schedule：默认 `burst`、`--arrival-process constant --request-rate λ`，以及 `--arrival-process poisson --request-rate λ --arrival-seed s`。constant 在 (i/\lambda) 到达；Poisson 把第一条请求锚定为 0，后续使用 seeded exponential inter-arrival。两者都预先确定到达时刻，不等待 completion；`--concurrency` 只限制 HTTP dispatch，过载会表现为 client queue 增长。

这里的 open-loop 只描述“到达时刻不由被测服务完成速度反馈控制”，不等于负载生成器无误差。输出将 scheduled offset 加到 benchmark start 作为 `offered_at`，所以 event-loop 唤醒迟到也进入 client queue；但 scheduled timestamp 不证明事件循环按时执行。脚本一次性物化有限 `--requests`，没有无限流、bounded pending queue、分布式同步或独立 generator-lag SLO。应监控负载机 CPU/lag，并且不能从 client timestamp 单独分解 gateway queue 与 vLLM queue。

~~~powershell
python projects/inference-serving/benchmark_openai.py `
  --model MODEL_ID --requests 100 --concurrency 8 `
  --arrival-process constant --request-rate 4
~~~

报告 p50/p90/p95/p99，不只平均。逐步增加 offered QPS，找到队列延迟陡增的 knee；生产容量留余量并考虑故障少一副本。

### Trace artifact 与删失

每个 offered request 保存 request id、到达/开始/首 token/结束时间、terminal outcome、已知 token usage、取消和错误分类。Timeout 是 right-censored experience：我们只知道用户至少等到了 timeout，不知道若继续等待何时完成；不能把 timeout 当作等于成功 latency，也不能忽略。工程容量报告通常直接联合展示 success/timeout curve 与成功 latency；更正式的生存分析需另行定义 estimand。

Client-visible `server_error` 不能证明根因是 OOM；需要与网关、scheduler、engine 和 GPU trace 关联。反之，client timeout 后 server 可能继续 decode，因此还要测 disconnect-to-release 和 wasted tokens。

仓库离线 CLI 可验证 attempt 聚合与 gate：

~~~powershell
python -m about_llm.inference_analysis_cli `
  --attempts projects/inference-serving/attempts.example.jsonl `
  --benchmark-started-at 0 `
  --benchmark-completed-at 2 `
  --minimum-success-rate 0.75 `
  --maximum-ttft-p95 0.5 `
  --maximum-client-queue-p95 0.2 `
  --maximum-successful-offered-ttft-p95 0.6 `
  --maximum-offered-to-terminal-p95 1.5
~~~

该 fixture 是合成 client trace，只证明统计口径，不证明任何 GPU、vLLM 配置或生产 SLO。

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

压测工具自身也会失败。已知网络/HTTP/protocol error 可记录为 attempt；未知代码异常应让基准失败，不要一律捕获后归类为服务故障。错误消息和 response body 可能含 secret/输入，artifact 需脱敏。

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
