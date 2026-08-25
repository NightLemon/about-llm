# vLLM 服务：跟一条请求从 HTTP 走到 GPU

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备在 Linux/GPU 上部署、验收和压测 vLLM 的开发者。
- **先修**：[端到端请求生命周期](inference-request-lifecycle.md)、Linux、GPU 显存与 HTTP 流式协议。
- **首次阅读**：一条请求的三段生命周期 → 启动 → 单请求验收 → 容量扫描 → 取消与回滚。
- **完成信号**：能沿同一个 request ID 解释客户端等待、服务端排队、prefill、decode、流式事件和资源释放。
- **卡住时**：先只完成非流式单请求，不要同时打开量化、prefix cache、并发和公网访问。

</div>

假设用户向 `/v1/chat/completions` 发送一句话：

> 用一句话解释 KV Cache。

片刻后，客户端收到了流式文本。这个结果还留下六个问题：服务加载的是不是目标模型？提示词在服务端排了多久？
首 token 来自哪次 prefill？后续 decode 是否持续占用 KV Cache？客户端断连后生成会不会继续？过载时请求会排队
还是尽快失败？

本页用这条请求回答这些问题。一次成功响应只是起点；验收还要覆盖模型身份、生成协议、容量、失败收口和回滚。

本页按一次真实部署的顺序展开。机制原理见[推理优化](inference-optimization.md)，
完整练习见[Inference Serving 项目](../practice/projects/inference-serving.md)。

## 先看这条请求的完整地图 {#one-request}

同一条请求会同时经历三段生命周期：

| 生命周期 | 从哪里开始 | 到哪里结束 | 主要问题 |
|---|---|---|---|
| 客户端的一次尝试 | 负载发生器计划发送 | 收到终态或本地超时 | 用户等了多久，请求是否成功 |
| 服务端请求与序列 | API 接收；通过校验后才建立序列 | 完成、拒绝或取消 | 排队、调度和 KV 资源怎样变化 |
| GPU 的多轮工作 | 某轮被调度 | 本轮 kernel 完成 | 这轮执行 prefill 还是 decode |

三段生命周期彼此相关，但不能互相代替。客户端关闭连接，不代表服务端序列已经取消；服务端任务结束，也不说明
此前发出的 GPU kernel 能在中途停止。一次完整路线可以先记成：

```text
计划发送
→ 获得客户端并发名额
→ HTTP 请求进入 API
→ 校验模型名、模板、长度和采样参数
→ tokenizer 生成 prompt token
→ admission 判断是否接纳
→ Scheduler 排队并安排 prefill
→ 采样首 token，发送第一个内容事件
→ 多轮 decode，继续发送内容
→ 发送 finish reason 与 usage
→ 序列进入终态，释放 KV block 和并发名额
```

其中 `admission` 是服务端的准入判断，决定请求能否进入执行队列。`Scheduler` 是调度器，决定每轮有哪些序列
获得计算资源。客户端看到的 SSE 数据块则是传输事件；一个数据块可能不含 token，也可能承载多个 token 对应的文本。

后文会反复回到这条请求：第一次只跑非流式，确认模型和 token；第二次打开流式，确认事件顺序；然后逐级增加
同时到达的请求，观察它何时开始排队；最后在首段文本到达后主动断连，检查三个生命周期分别停在哪里。

## 部署前先固定五件事

不要先复制启动命令。先写下：

| 项目 | 示例 | 为什么要固定 |
|---|---|---|
| 模型身份 | 仓库 ID + commit revision | 同名模型可能更新 |
| Tokenizer 与模板 | 与模型使用同一 revision | Token、EOS 与输入长度会改变 |
| 运行时身份 | vLLM、PyTorch、CUDA、驱动 | 命令、kernel 和调度行为会变化 |
| 硬件 | GPU 型号、显存、功耗限制 | 容量和性能不能跨硬件外推 |
| 请求负载 | 输入/输出长度、到达方式、并发 | 单次演示不代表服务负载 |

还要确认模型许可、访问权限和 `trust_remote_code` 策略。
生产环境默认关闭远程代码；确需开启时，先审计并固定 revision。

### 平台边界

本页只讨论 NVIDIA GPU 服务路径。vLLM 还提供其他硬件后端，但安装方式和功能范围要分别查支持矩阵。
Windows 开发者通常在 WSL2 中运行这条路径，也可以连接远程 Linux。

安装以前，要把模型架构和量化格式放进同一张兼容性表。CUDA、显卡驱动、PyTorch 与 vLLM 版本也要一起核对。

本仓库当前 Windows CPU 环境没有执行 vLLM/CUDA。下面是目标环境运行路线，不是已有 GPU 实测结果。

## 第一次启动：保持变量最少 { #first-start }

先使用能放入显存的小模型，不启用量化、adapter、并行或公网绑定。

~~~bash
vllm serve MODEL_ID \
  --revision COMMIT_HASH \
  --served-model-name my-model \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 8000
~~~

参数名称、默认值和支持矩阵会随版本变化。执行前以当前安装版本的 `vllm serve --help=all` 和
对应 stable 文档为准，不要把这里的数值当作生产推荐值。

日志中至少保存：

- 解析后的模型、配置和 tokenizer revision；
- 实际数值类型、量化方式和注意力后端；
- 可用 KV block 或最大并发估计；
- 启动耗时、显存占用和任何回退警告；
- 完整启动参数，而不只保存 shell 历史。

## 单请求验收：沿请求生命周期逐层检查

先把这次调用记作请求 A。确认模型列表后，用非流式接口发送本章开头的问题：

~~~bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "用一句话解释 KV Cache。"}],
    "temperature": 0,
    "max_tokens": 32
  }'
~~~

先不开流式，是为了把响应对象一次看完整。请求结束后检查：

1. 返回的模型身份是否与服务配置一致。
2. `finish_reason` 是否能由 EOS、长度或其他终止条件解释。
3. Prompt 与 completion 的用量是否存在，并与目标 tokenizer 的 token 结果一致。
4. 服务端轨迹是否能用 request ID 关联到这次执行。
5. 错误 model、非法字段、过长请求和空输入是否得到明确失败，而不是静默改写。

建议把结果写在一张请求卡上，不要只保存回答文本：

| 位置 | 请求 A 要记录什么 |
|---|---|
| 客户端输入 | 原始 messages、模型名、采样参数和长度上限 |
| 模板与 tokenizer | 渲染后的文本、prompt token ID 和数量 |
| 服务端序列 | request ID、排队和调度轨迹 |
| 模型执行 | prefill/decode 轮次、实际模型版本和后端 |
| 客户端输出 | 原始响应、输出 token、用量与 `finish_reason` |

如果这五行无法用同一个 request ID 串起来，就还不能判断“返回文本的服务”与“加载目标权重的服务”是否是同一条路径。

### 再验证流式响应

~~~bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "按顺序输出甲乙丙。"}],
    "temperature": 0,
    "max_tokens": 16,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
~~~

记录第一个非空内容增量、终止事件、用量字段和连接关闭的顺序。

SSE 事件或文本数据块不是模型 token。一个事件可能只传角色或终态，也可能包含一段由多个 token 解码出的文本；
因此不能用事件数量代替 completion token 数量。

“OpenAI-compatible”只说明基础 API 形状相近。工具调用、结构化输出、多模态和 logprobs 都要按目标版本
分别验证。用量、错误对象和 `finish_reason` 也要保存真实响应，不能从基础聊天接口反推。

## Model、tokenizer 与生成协议要对账

请求 A 进入模型以前，`messages` 会先由对话模板排版。tokenizer 再把排版后的文本转换为 token ID。

部署前要同时检查模型配置、tokenizer 和生成配置。词表大小、对话模板以及 BOS、EOS、PAD 等特殊 token
都要对得上。

这些特殊 ID 不一定全部相等。例如，生成配置的 EOS 集合可以有意包含多个停止 token，PAD 也可能与 EOS 重合。
真正需要拦住的是下面几类错误：

- ID 超出 tokenizer 或模型词表上界；
- 客户端、服务端与离线基线使用不同的停止集合；
- 对话模板或 adapter revision 没有进入服务身份；
- 请求中的覆盖参数悄悄改变配置里的长度或采样语义。

请求 A 就可以作为升级前后的最小差分样例。保存它的模板文本、输入与输出 token ID、用量和停止原因，
每次升级后重放并比较。

## 两个容量旋钮先用人话理解

### `max-model-len`

它限制单条序列允许的上下文长度，也影响运行时为 KV Cache 规划的空间。
设成模型理论最大值可能显著降低可并发序列数。

根据业务长度分布和必要上限选择。超长请求应在 admission 前拒绝，或路由到专用池，
不要让一条异常长请求挤掉所有正常流量。

### `gpu-memory-utilization`

它为运行时使用 GPU 显存提供目标比例，不等于“KV Cache 独占这部分显存”。
过高会减少峰值余量并增加 OOM 风险，过低会浪费可服务容量。

每次修改后都要重新记录权重、KV blocks、workspace、峰值显存和最大稳定并发。

其他 token 和序列预算会改变调度器每轮能安排多少工作。预算改变后，请求 A 可能更早进入 prefill，也可能
与更多 decode 请求共享一轮计算。吞吐因此可能提高，但排队、首 token 延迟或抢占也可能增加。

## 容量实验必须逐级加压

不要从目标最大并发开始。使用同一模型和 workload 逐级扫描：

```text
并发 1 -> 2 -> 4 -> 8 -> ...
```

每一级先 warm up，再保留稳定窗口和所有失败终态。停止条件可以是：

- Success rate 低于门槛；
- TTFT/TPOT 或 E2E 超过 SLO；
- KV preemption/recompute 持续增加；
- OOM、429、timeout 或其他错误不可接受；
- 质量或生成协议发生漂移。

最后一个仍满足所有门槛的档位，才是这份 workload 下的可辩护容量点。

扫描并发时，可以把请求 A 复制成 A-1、A-2、A-3……，但要为每次尝试分配独立 request ID。发现曲线拐点后，
先判断时间增加在哪一段：

| 先变坏的信号 | 第一轮排查方向 |
|---|---|
| `offered_at → started_at` 增加 | 客户端并发上限或负载发生器跟不上计划 |
| dispatch TTFT 增加，服务端 queue age 也增加 | 服务端接纳速度超过可调度容量 |
| dispatch TTFT 增加，服务端队列稳定 | API、tokenizer、prefill 或网络路径 |
| TPOT 增加 | decode batch、KV 压力、抢占或 GPU 执行 |
| 429/503 增加 | 准入门槛开始拒绝当前到达率 |

这张表只给出排查起点，不是自动诊断。最终判断仍要把客户端 attempt 与同一 request ID 的服务端轨迹对齐。

### 用项目负载发生器

安装 API 依赖后运行：

~~~powershell
python -m pip install -e ".[api]"

python projects/inference-serving/benchmark_openai.py `
  --model my-model --requests 20 --concurrency 1

python projects/inference-serving/benchmark_openai.py `
  --model my-model --requests 100 --concurrency 8 `
  --arrival-process constant --request-rate 4
~~~

先用一小批同时到达的请求做冒烟检查，再用固定间隔或带随机种子的 Poisson 到达观察排队。

脚本会预先生成有限的发送时间表。每个请求的计划时刻不依赖上一条请求何时完成；`--concurrency` 只限制
同时进行的 HTTP 调用。如果并发名额不足，请求会留在客户端等待，这段等待仍会进入报告。

过高的名义到达率可能让客户端堆积任务，并产生模型调用费用。先从小规模开始，同时监控负载发生器落后时间、
CPU、预算和紧急停止路径。

## 把四个客户端时刻写在请求卡上

一次 HTTP 调用称为一个 attempt（尝试）。重试会产生新的 attempt，不能覆盖第一次失败记录。每次尝试至少保存：

| 字段 | 定义 |
|---|---|
| `offered_at` | 按发送时间表，本应可以提交的时刻 |
| `started_at` | 获得客户端并发 permit 并开始 HTTP dispatch |
| `first_token_at` | 收到首个非空 content delta |
| `completed_at` | 成功或失败终态 |

`offered_at → started_at` 是客户端排队。`started_at → first_token_at` 是从实际发送到首段内容的 TTFT。
如果只从 `started_at` 开始计时，事件循环落后和客户端并发名额前的等待就会消失，报告会低估用户经历的压力。

TPOT 表示首 token 之后，平均每个输出 token 等多久。记 (t_{first}) 为 `first_token_at`，
(t_{done}) 为 `completed_at`。若成功请求输出 (O>1) 个 token，本仓库使用：

\[
\mathrm{TPOT}=\frac{t_{done}-t_{first}}{O-1}.
\]

只有一个输出 token 时不存在“后续 token 间隔”，因此 TPOT 应保持未定义，而不是填成 0。

报告中分开：

- 成功数、尝试总数，以及 429、超时、5xx、协议错误和客户端错误；
- 所有尝试的客户端排队与“计划发送到终态”时间；
- 成功请求的 dispatch TTFT、端到端延迟和 TPOT；
- 成功请求从 `offered_at` 到首 token 的时间；
- 尝试请求吞吐、成功请求吞吐与成功输出 token 吞吐；
- 服务端 GPU、KV、抢占、队列与 request ID 轨迹。

快速失败的 429 可能让 offered-to-terminal 变小，所以延迟必须和 success rate 一起解释。

## 先固定请求负载，再比较数字

假设第一次测试使用请求 A，第二次却把输入缩短一半。即使第二次吞吐更高，也不能判断服务真的变快了。
两次压测至少要固定：

- 输入与输出长度的联合分布，而不只是平均长度；
- 对话模板、采样、停止条件和输出上限；
- 同时到达、固定间隔、Poisson 或真实流量轨迹；
- 预热、持续时间、并发和客户端位置；
- 超时、重试、取消和失败是否进入分母；
- 服务是否与其他进程共享 GPU。

输入更短、输出被截断或失败被排除，都可能让吞吐数字“变好”。
没有固定请求负载的跨框架 tokens/s 排名通常不可解释。

## 取消实验要观察三个终点

现在重放请求 A，但在收到第一段内容后主动断开客户端。随后分别确认：

1. HTTP/ASGI 响应任务是否结束；
2. 运行时是否取消对应序列，不再继续 decode；
3. KV block、序列名额和并发名额是否释放。

三个终点不会自动同时发生。客户端可以先断开，而后端仍在生成；Python 任务收到取消，也不能让已经发出的
GPU kernel 在任意位置中断。

验收目标 vLLM 版本时，要使用该版本自己的取消接口、调度轨迹和 block 释放轨迹。
本仓库的异步迭代器与 Transformers 线程验证程序只检查较低层的取消传播，见
[推理服务证据页](../evidence/inference-serving-controls.md#local-http-cancel)。

## Admission、背压与过载

请求进入服务后，准入层只有三种诚实选择：立即接纳、在有界队列中等待，或者明确拒绝。无界队列只是把失败
改写成长时间等待。准入判断至少应考虑：

- 当前序列名额与 KV 容量；
- 输入和最大输出的最坏资源需求；
- 租户配额、优先级和截止时间；
- 队列中最老请求已经等待多久，以及预计还要等待多久；
- 取消、超时和 worker 故障后的资源回收。

过载时返回明确的 429 或 503，通常比先接受所有请求、再让它们大面积超时更可控。调用方如需重试，必须设置
总截止时间和退避策略。一次逻辑请求也要有幂等边界，避免被放大成多次昂贵生成。

单个副本里的 semaphore 只能限制这个进程，不能自动形成整个服务的全局并发上限。

## 可观测性：把协议、调度和 GPU 串起来

一次性能分析至少需要三组信息：

| 层面 | 观察内容 |
|---|---|
| 客户端与 API | 计划发送、实际发送、首 token、终态、状态码和用量 |
| 调度器与 KV | 排队时间、接纳的序列、KV block、抢占和前缀命中 |
| 模型与 GPU | prefill/decode 工作量、batch 形状、kernel、带宽利用率和峰值显存 |

为请求 A 生成一个 request ID，并把它写入三层记录。模型、tokenizer 与模板版本也要进入同一条服务端生成轨迹。
HTTP 200 只说明接口返回成功，不能单独证明目标 checkpoint 或目标 kernel 被执行。

自动扩缩容不能只看 GPU 利用率。准入过严时，GPU 利用率可能很低；队列已经失控时，GPU 又可能长期满载。
因此还要同时观察队列年龄、KV 容量、请求到达率、TTFT 和错误率。

## 安全与部署边界

最小生产边界包括：

- 不默认绑定公网；前置 TLS、认证、授权、限流和 request-size 上限；
- Token、Prompt、输出和 trace 按敏感数据处理，日志默认脱敏；
- 固定 model revision、容器 digest、依赖和远程代码策略；
- 对 tools、JSON schema、多模态和 adapter 建立独立 allowlist；
- Readiness 检查 model/tokenizer/adapter、设备和 scheduler 是否真正可接流量；
- Liveness 只回答进程是否需要重启，不能代替 readiness。

滚动发布时，先从路由摘除、停止 admission、等待或有界取消 in-flight，最后释放模型和 KV。
直接杀进程会把未决 attempts 留给调用方 reconciliation。

## 升级与回滚

升级前保存一份可重放的验收包：

```text
model/tokenizer/template/adapter identity
runtime/container/driver/hardware identity
启动配置
token-level behavior samples
质量与安全 case
固定 workload attempts 和 server trace
容量曲线与失败样例
回滚命令
```

先在隔离环境跑单请求和错误协议，再运行同一 workload。比较的不只是平均性能，还包括 token ids、usage、
finish reason、失败分布、KV/preemption 和尾延迟。

Canary 期间保留旧版本容量，确认回滚真正可执行，而不是只保留旧镜像标签。

## 常见故障从哪里查

### 启动即 OOM

先降低 context 上限和 memory utilization，检查是否加载了预期 dtype/量化格式，
再排查其他进程、workspace、graph capture 和模型架构支持。

### TTFT 退化但 TPOT 正常

先判断等待发生在计划发送、客户端 dispatch，还是服务端队列。然后检查输入长度分布、prefill batch 和
tokenizer 耗时。TPOT 仍然正常时，decode kernel 通常不是第一个排查对象。

### TPOT 退化但 TTFT 正常

查看 decode batch、KV 使用、preemption、权重带宽、quant kernel、其他 GPU 进程和 launch。

### 流式有文本但没有 usage

先确认目标版本是否支持流式 usage。若必须客户端重算，应使用完全相同 revision 的 tokenizer，
并把计数来源写入报告；不能用 SSE chunk 数静默替代 token 数。

### OpenAI client 可以连但高级功能失败

工具调用、结构化输出、logprobs 和多模态输入要分别验证。停止原因与错误对象也要保存原始样例。

基础聊天接口可用，只能证明这条基础路径，不能证明客户端可以完整替换。

## 单卡验收清单

1. 固定硬件、软件、model revision、tokenizer/template 和许可。
2. 非流式、流式、错误 model、长度上限和 stop/token usage 对账。
3. 逐级扫描并发和短/长输入输出，保留全部失败终态。
4. 报告质量、success rate、queue、TTFT、TPOT、E2E、吞吐、显存和功耗。
5. 断连、timeout、429、worker failure 和 OOM 路径能够收口。
6. 认证、限流、日志脱敏、readiness、drain 和供应链检查完成。
7. Canary 与回滚演练完成，并说明结论只适用于哪些版本和 workload。

## 实践入口

- 先跟随[一次请求如何穿过推理引擎](inference-request-lifecycle.md)复述状态和计时边界。
- 用[Paged KV 引导实验](../practice/labs/lab-7a-paged-kv.md)理解 block table 与 COW。
- 在[Inference Serving 项目](../practice/projects/inference-serving.md)完成本地验证和目标 GPU 验收。
- 用[证据页](../evidence/inference-serving-controls.md)核对每条结论的测试与外推边界。

完成后，你应当得到一份可审阅的验收报告，而不只是一个启动成功的终端截图。
