# vLLM 与 OpenAI-compatible 单卡服务

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备在 Linux/GPU 上部署、验收和压测 vLLM 的开发者。
- **先修**：[端到端请求生命周期](inference-request-lifecycle.md)、Linux、GPU 显存与 HTTP 流式协议。
- **首次阅读**：部署前固定身份 → 启动 → 单请求验收 → 容量扫描 → 故障与回滚。
- **完成信号**：能提交一份包含版本、workload、质量、TTFT/TPOT、显存和失败样例的验收报告。
- **卡住时**：先只完成非流式单请求，不要同时打开量化、prefix cache、并发和公网访问。

</div>

成功启动一个 `/v1/chat/completions` 端点，只能说明进程能够响应请求。
一个可验收的推理服务还要证明：加载了目标模型、生成协议正确、容量可控、失败能收口、配置可以回滚。

本页按一次真实部署的顺序展开。机制原理见[推理优化](inference-optimization.md)，
完整练习见[Inference Serving 项目](../practice/projects/inference-serving.md)。

## 部署前先固定五件事

不要先复制启动命令。先写下：

| 项目 | 示例 | 为什么要固定 |
|---|---|---|
| Model identity | repo id + commit revision | 同名模型可能更新 |
| Tokenizer/template | 与 model 同 revision | Token、EOS 与输入长度会改变 |
| Runtime identity | vLLM、PyTorch、CUDA、driver | CLI、kernel 和调度行为会变化 |
| Hardware | GPU 型号、显存、功耗限制 | 容量和性能不能跨硬件外推 |
| Workload | prompt/output 长度、arrival、并发 | 单次 demo 不代表服务负载 |

还要确认模型许可、访问权限和 `trust_remote_code` 策略。
生产环境默认关闭远程代码；确需开启时，先审计并固定 revision。

### 平台边界

vLLM 主要面向 Linux 与受支持的 GPU/加速器。Windows 开发者通常使用 WSL2 或远程 Linux。
模型架构、量化格式、CUDA、driver、PyTorch 和 vLLM 之间存在兼容矩阵。

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

- 解析后的 model/config/tokenizer revision；
- 实际 dtype、量化和 attention backend；
- 可用 KV block 或最大并发估计；
- 启动耗时、显存占用和任何 fallback warning；
- 完整启动参数，而不只保存 shell history。

## 单请求验收：沿请求生命周期逐层检查

先确认模型列表，再发一个非流式请求：

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

第一次不要只看回答是否通顺。检查：

1. 返回的 model identity 是否与服务配置一致。
2. `finish_reason` 是否能由 EOS、长度或其他终止条件解释。
3. Prompt/completion usage 是否存在，并与目标 tokenizer 的 token-level 对照一致。
4. 服务端 trace 是否能用 request id 关联到这次执行。
5. 错误 model、非法字段、过长请求和空输入是否得到明确失败，而不是静默改写。

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

记录首个非空 content delta、终止事件、usage 和连接关闭顺序。
SSE event 或 text chunk 不是模型 token，不能用 chunk 数代替 completion tokens。

OpenAI-compatible 只说明基础 API 形状相近。Tools、JSON schema、多模态、logprobs、usage、错误和
`finish_reason` 的细节仍需针对目标版本做 contract test。

## Model、tokenizer 与生成协议要对账

部署前同时检查 model config、tokenizer 和 generation config 中的 BOS/EOS/PAD、vocabulary size 和 chat template。

不要机械要求所有特殊 token id 完全相等：generation EOS 可以是 tokenizer EOS 的有意 superset，
PAD 也可能与 EOS 重合。真正危险的是：

- ID 越过 tokenizer 或 model vocabulary 上界；
- Client、server 与离线 baseline 使用不同 stop 集；
- Chat template 或 adapter revision 没有进入服务 identity；
- 请求 override 静默改变 config 中的长度或采样语义。

固定一个短 prompt，保存 token ids、渲染后的模板、输出 token ids、usage 和 finish reason，
把它作为升级前后的最小差分样例。

## 两个容量旋钮先用人话理解

### `max-model-len`

它限制单条序列允许的上下文长度，也影响 runtime 为 KV 规划的空间。
设成模型理论最大值可能显著降低可并发序列数。

根据业务长度分布和必要上限选择。超长请求应在 admission 前拒绝，或路由到专用池，
不要让一条异常长请求挤掉所有正常流量。

### `gpu-memory-utilization`

它为 runtime 使用 GPU memory 提供目标比例，不等于“KV 独占这部分显存”。
过高会减少峰值余量并增加 OOM 风险，过低会浪费可服务容量。

每次修改后都要重新记录权重、KV blocks、workspace、峰值显存和最大稳定并发。

其他 token/sequence budget 会改变每轮可容纳的 prefill/decode 工作。
这些参数提高吞吐时，也可能增加 queue、TTFT 或 preemption。

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

先用 burst 做小规模 smoke，再用 constant 或 seeded Poisson 到达观察排队。
脚本会预生成有限 arrival schedule，`--concurrency` 只限制在途 HTTP attempts，不会把 open-loop 变成 closed-loop。

高 nominal rate 可能在客户端堆积任务并产生模型费用。先小规模运行，并监控 generator lag、CPU、预算和紧急停止路径。

## 四个时间戳不能省略

每个 attempt 至少保存：

| 字段 | 定义 |
|---|---|
| `offered_at` | 按 arrival schedule 应提交的时刻 |
| `started_at` | 获得客户端并发 permit 并开始 HTTP dispatch |
| `first_token_at` | 收到首个非空 content delta |
| `completed_at` | 成功或失败终态 |

只从 `started_at` 计时，会漏掉 event-loop lag 和 client semaphore 前等待。
这就是客户端侧 coordinated omission 的一种来源。

报告中分开：

- `success / attempted` 和 429、timeout、5xx、protocol、client errors；
- All-attempt client queue 与 offered-to-terminal；
- Success-conditional dispatch TTFT、E2E 和 TPOT；
- Successful-offered TTFT；
- Attempted/successful requests/s 与 successful output tokens/s；
- 服务端 GPU、KV、preemption、queue 和 request-id trace。

快速失败的 429 可能让 offered-to-terminal 变小，所以延迟必须和 success rate 一起解释。

## Workload contract 决定数字能否比较

两次 benchmark 至少固定：

- Prompt/output 长度的联合分布，而不只是平均长度；
- Chat template、sampling、stop 和输出上限；
- Burst、constant、Poisson 或真实 trace 到达；
- Warmup、持续时间、并发和客户端位置；
- 超时、重试、取消和失败是否进入分母；
- Server 是否与其他进程共享 GPU。

输入更短、输出被截断或失败被排除，都可能让吞吐数字“变好”。
没有 workload contract 的跨框架 tokens/s 排名通常不可解释。

## 取消实验要观察三个终点

在收到第一个内容 delta 后主动断开客户端，然后分别确认：

1. HTTP/ASGI response task 是否结束；
2. Runtime 是否 abort 对应 sequence，不再继续 decode；
3. KV blocks、sequence slot 和并发 permit 是否释放。

三者不是同一事件。客户端断连可以先发生，backend 可能继续生成；Python task 被取消也不能证明已经进入的 GPU kernel 可中断。

目标 vLLM 版本应使用自身 abort API、scheduler trace 和 block-release trace 验收。
本仓库的 async iterator 与 Transformers thread 验证程序只检查较低层的取消传播，见
[推理服务证据页](../evidence/inference-serving-controls.md#local-http-cancel)。

## Admission、背压与过载

无界 queue 只会把失败变成长时间等待。Admission 至少应考虑：

- 当前 sequence slots 与 KV capacity；
- Prompt/maximum output 的最坏资源需求；
- Tenant quota、优先级和 deadline；
- Queue age 与预计等待；
- 取消、超时和 worker 故障后的资源回收。

过载时返回明确的 429/503 和 retry policy，通常比接受所有请求后大面积超时更可控。
Retry 必须有总 deadline、退避和幂等边界，不能让一次逻辑请求无限放大为多次昂贵生成。

Replica-local semaphore 只能限制单进程，不自动构成服务级全局并发上限。

## 可观测性：把协议、调度和 GPU 串起来

一次性能分析至少需要三组信息：

| 层面 | 观察内容 |
|---|---|
| Client/API | offered、dispatch、首 token、终态、status、usage |
| Scheduler/KV | queue age、admitted sequences、KV blocks、preemption、prefix hit |
| Model/GPU | prefill/decode work、batch shape、kernel、带宽/利用率、峰值显存 |

用 request id、model/tokenizer/template revision 和 server-side generation trace 关联三层。
HTTP 200 不能单独证明目标 checkpoint 或目标 kernel 被执行。

Autoscaling 也不能只看 GPU utilization：GPU 可能因 admission 太严而低利用，
也可能在 queue 已失控时长期满载。Queue age、KV 容量、offered rate、TTFT 和错误必须一起看。

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

优先查看 offered/client/server queue、prompt 长度分布、prefill batch 和 tokenizer，
不要先优化 decode kernel。

### TPOT 退化但 TTFT 正常

查看 decode batch、KV 使用、preemption、权重带宽、quant kernel、其他 GPU 进程和 launch。

### 流式有文本但没有 usage

先确认目标版本是否支持流式 usage。若必须客户端重算，应使用完全相同 revision 的 tokenizer，
并把计数来源写入报告；不能用 SSE chunk 数静默替代 token 数。

### OpenAI client 可以连但高级功能失败

对 tools、structured output、logprobs、multimodal、finish reason 和错误字段做独立 contract test。
“基础 chat 可用”不能外推为完整替换。

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
