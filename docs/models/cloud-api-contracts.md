# 云模型 API 的共同点与差异

## 不要把 messages 当成统一标准

不同供应商都能完成对话，但角色、system 位置、tool schema、流式事件、usage、缓存和错误码不同。业务层可以统一最小概念，adapter 必须保留能力差异。

| 维度 | OpenAI-compatible | Anthropic Messages | 本仓库 Gemini `generateContent` adapter |
|---|---|---|---|
| system | messages 中常见 | 顶层 system | systemInstruction |
| assistant role | assistant | assistant | model |
| 内容 | string 或多段结构 | content blocks | parts |
| usage | prompt/completion tokens | input/output tokens | usageMetadata |
| 结束 | finish_reason | stop_reason | finishReason |

表格只描述本仓库 adapter 覆盖的稳定最小契约，实际字段以固定 API 版本的官方文档为准。

## GPT、DeepSeek 与 Qwen

三者可能通过 OpenAI-compatible 形状接入，但兼容不等于完全等价。检查 tool calling、JSON schema、reasoning 字段、stream usage、缓存、错误和模型 id。不要把某个兼容端点的扩展字段无条件传给另一家。

## Claude

Anthropic Messages 将 system 与对话分开，content 是 block 数组。工具结果、thinking/其他 block 与纯 text 不能用同一解析假设。只需要文本时明确过滤 text block，同时保留非文本 block 的审计信息。

## Gemini

截至 2026-08-06，Gemini API 官方文档说明 Interactions API 已 GA，并推荐新项目使用；原有 `generateContent` 仍受支持但已标为 legacy。本仓库 adapter 为教学与兼容性实现 `generateContent`：它使用 `contents`/`parts`、`user`/`model` role 与 `systemInstruction`，多模态输入也是 part。Interactions API 与 `generateContent` 的状态、字段和流式事件必须分别建模，不能把表中契约当成 Gemini 的统一接口。Gemini API 与 Vertex AI 在身份、区域、治理和 endpoint 上也可能不同，应分别配置。

## 可靠客户端

客户端至少记录 provider、model、API version、request id、usage、latency、重试和 finish reason。错误分为认证、配额、内容政策、无效请求、瞬时服务、超时和本地取消。不要对所有 4xx/5xx 自动重试。

一次调用失败后，先回答三个独立问题：

1. 该 status/error 在这个 provider、endpoint 和 API version 下是否被确认是瞬时的？
2. 请求是否可安全重放？工具副作用、batch mutation 与外部状态写入不能仅因 HTTP method 看起来像读取就判定安全。
3. 远端 outcome 是否确定？连接中断或 read timeout 可能发生在服务端已接收甚至完成之后；未知结果不能自动重放。

本仓库 `RetryPolicy` 的教学 allowlist 为 `408/429/500/502/503/504`，明确排除普通 `400/401/403/404`，也不把 `501/505` 仅因是 5xx 而重试。它是本地策略示例，不是供应商通用保证。`max_attempts` 包含首次；exponential backoff 有上限，jitter 由 caller 注入。`Retry-After` 严格解析 non-negative delta-seconds 或 HTTP-date：有效值若超过 policy/deadline 就停止，不能为了赶 deadline 提前重试。

即使 prompt 调用在业务上无副作用，第二次生成仍可能产生另一份 usage 与费用。Provider 的 idempotency key、错误 body 和配额恢复语义不能跨 API 假设统一。

本仓库 `execute_json_request` 使用 caller-owned `httpx.AsyncClient`，以 monotonic clock 计算整个 attempt/sleep deadline，并显式禁止 redirect。请求必须命中 exact origin allowlist；默认 HTTPS-only、URL query disabled，避免把密钥放进 URL。Pool/connect 阶段失败可在 replay-safe 前提下进入本地重试策略；write/read/protocol 或执行器 attempt timeout 的远端 outcome 无法仅由 client 确认，因此 fail closed。Cancellation 原样传播，caller 仍要调查远端是否已执行。

成功响应执行严格 object-JSON 解析，拒绝 duplicate key、非有限 number、错误 Content-Type 和超限 body。这里的 body size 检查发生在 `httpx` 非流式响应已经缓冲之后，只是 acceptance cap，不是下载过程的 ingress/memory 上限。流式客户端必须在读取每个 chunk 时累计并提前关闭超限响应。

流式协议还要拆成两层：SSE framing 只负责从任意 UTF-8 byte chunks 还原 event，provider state machine 才解释 payload。本仓库 `SSEDecoder` 覆盖 BOM、LF/CRLF/CR、多行 data、跨 chunk UTF-8、line/event/total byte 上限与截断 EOF；`cloud_stream` 分别实现 OpenAI-compatible Chat Completions、Anthropic Messages 和 Gemini `streamGenerateContent` 的单 choice/candidate、text-only 子集。OpenAI `[DONE]`、Anthropic `message_stop` 与 Gemini finishReason+EOF 是不同终止契约，不能互换。

应用自定义 stop string 又是第三层：它在 provider text delta 拼接后的 Unicode stream 上匹配，可能跨 event 和 token。仓库增量 matcher 证明严格 UTF-8、partial-prefix withholding 与一份明确 overlap 规则，但客户端命中 stop 后截断显示不能伪造 provider `finish_reason=stop`，也不能自行修改 provider usage；是否取消远端请求、远端是否停止和如何计费仍按具体 API 契约与 request trace 判断。

这些 parser 遇到 tool/refusal/thinking/媒体/未知 block 会失败，不静默丢字段；fragment/event 数不是 token 数，也不自动 reconnect。`execute_sse_request` 已用 MockTransport/AsyncByteStream 验证 response 打开、逐 chunk 消费，以及成功、截断、超限、timeout、协议错误和取消后的关闭。仅在非 2xx headers 阶段允许重试；2xx body 开始后的失败是 partial/outcome-uncertain，不重放。

该测试仍不执行真实 DNS/TLS/TCP/HTTP2。关闭 client response 不证明服务端已收到取消、停止生成或停止计费；`on_update` 已交付文本也无法撤回。`bytes_received` 是 `aiter_bytes` 提供给 parser 的字节量，不是 token 或精确 wire-byte 计量。

## Token 与费用预算为什么要先 reserve、再 reconcile

只在响应后累加 usage 会产生并发超支：十个调用可能同时看到“还剩 1000 tokens”，然后各自发出 1000-token 请求。正确控制流是在 transport 前原子预留 `estimated_input + maximum_output` 与对应估算费用，成功后以 provider-reported usage 结算并释放余量。能证明请求从未发送才 cancel；请求可能已到 provider 但 usage 不可见时，应保守占用完整 reservation 并进入 reconciliation。若实际 usage 超过预留，调用已发生，账本必须先记录真实值，再触发 post-call breach，不能为维持漂亮的 cap 而丢弃超额 usage。

仓库 `UsageBudgetLedger.reserve_request` 从本仓库三类 RequestSpec 的 `max_tokens` 或 `generationConfig.maxOutputTokens` 直接提取唯一正整数 cap，从恰好一个 body/URL 来源提取 model 并与 pricing snapshot 精确匹配，再把 exact URL、完整 body、规范化 header 与显式 billing scope 绑定为 request fingerprint。常见 credential header 值在 canonical hash 前替换；因此同 scope 换 key 不改 identity，而 prompt/cap/scope 漂移会改变 identity。Terminal transition 同时核对 call id 与 fingerprint，错配不能弹出 active reservation。价格快照绑定 provider/model/revision/checked_at，费率单位是 micro-USD per million tokens，整数计算按每调用总估值向上取整。固定 `$1/M input + $2/M output` 的 authored fixture 中，60+10 预留 80 micro-USD，58+4 结算 66 micro-USD。

该结果只是明确费率下的 policy estimate：目标 tokenizer/chat template 不匹配、provider 隐藏/缓存/reasoning tokens、billing tier、最低单位、税费/额度与价格漂移都会破坏“等于发票”的解释。Fingerprint 不认证 caller/pricing/transport，SHA-256 也不为低熵 prompt 提供保密；内存锁不提供多进程、跨区域、持久 quota 或与 HTTP/billing export 的事务性。

仓库另提供 `SQLiteUsageBudgetLedger` 作为 local durable atomic quota reference。它用 `BEGIN IMMEDIATE` 串行化同一文件的 writer，把 pricing+limits singleton config、reservation tombstone 与 event timeline 持久化；测试覆盖两连接争抢最后额度、进程退出后重开、event 失败整笔回滚、配置漂移和 post-call overrun 先持久化再报错。Crash 后 active reservation 继续占额度，不能按 TTL 自动释放，因为“worker 消失”不等于“provider 未收到请求”；必须用 request trace 与 billing export reconciliation，能证明未发送才 cancel，否则按 uncertain 保守入账。

这只把证据从“单进程内存状态机”提升为“本地 SQLite 文件上的 durable quota 状态机”。SQLite commit 不可能与远程 HTTP/provider billing 原子，单文件也不是跨机器/跨区域分布式配额；无密钥 config/request fingerprint 不抵抗能改库并重算 hash 的攻击者。它不认证 pricing、provider usage 或 invoice，不证明取消成功、exactly-once billing、断电 durability、备份恢复或服务可用性。

`execute_budgeted_json_request` 再把单次 JSON HTTP attempt 接入 ledger：target preflight 在 reserve 前；明确的 Pool/Connect 前失败才 cancel；任何 HTTP status、2xx malformed/缺 usage、client cancellation 或发送后的未知错误都按 uncertain；只有严格 parser 返回完整 usage 才 settle。它强制 `max_attempts=1`，因为自动 replay 可能产生多份费用，而“最终一次成功的 usage”不能证明此前 attempt 免费。生产 retry 必须给每个 attempt 独立 reservation/tombstone，再逐次 reconciliation，不能让一个逻辑 call 的一次预留覆盖多次可能计费的调用。

离线 MockTransport + SQLite demo 精确得到 settled 66 与 uncertain 80 micro-USD，但仍有 provider effect 与本地 terminal commit 之间的 crash window；active 记录正是留给重启对账，而不是 TTL 清理。该实验不证明真实网络、provider error usage、request-id 真实性、发票或 cancellation。

密钥不进入 Prompt、日志和异常 repr。流式连接取消必须传播到服务端。产生工具调用时，provider adapter 只解析建议，真正权限与幂等由 Agent runtime 执行。

本仓库的 `about_llm.integrations.cloud_api_cli verify` 只对固定 JSONL fixture 构建请求并解析响应；`retry-matrix` 只计算固定重试决策。HTTP executor 测试也只使用 `httpx.MockTransport`，预算 toy 只运行本地 authored price。它们适合锁定 adapter/策略/控制流回归，不是网络 smoke test，也不证明真实 provider 的当前错误、配额、幂等、计费或 endpoint 语义；真实客户端仍需在显式 network marker、请求/token/费用预算和允许域名下单独验证。
