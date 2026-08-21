# GPT、DeepSeek、Qwen、Claude、Gemini 云 API 契约

目标：统一业务侧 ChatMessage 与 ChatResponse，但显式保留供应商协议差异，不用一个看似通用的 SDK 隐藏 system、usage、finish reason 和错误语义。

第一次阅读只需运行[离线契约验证](#可运行离线契约验证)。输入是三组固定的 request/response JSONL，输出是
规范化后的请求、响应和脱敏检查结果；整个过程不会导入 HTTP client，也不会使用真实密钥。确认三种协议都能
映射到共同对象后，再按需要学习 Responses events、流式、重试和预算。

## 三类协议

- OpenAI-compatible：可用于按配置接入 GPT，以及提供兼容端点的 DeepSeek/Qwen 服务；
- Anthropic Messages：system 位于顶层，消息与 usage 字段独立；
- Gemini `generateContent`：user/model role、parts、systemInstruction 和 usageMetadata；本 adapter 用于教学与兼容性。截至 2026-08-06，官方文档说明 Interactions API 已 GA 并推荐新项目使用，`generateContent` 仍受支持但已标为 legacy；不能混用两套接口的字段、状态与流式事件。

模块在 import 时不读取密钥、休眠或访问网络。RequestSpec 会复制 JSON body/headers，拒绝 `NaN`/`Infinity`，headers 不参与 repr，`sanitized_headers` 会遮蔽认证值。本项目的 ChatResponse 是 text-only 最小契约：tool-call-only 或无文本响应会明确失败；token usage 不接受布尔值、负数或字符串强制转换。生产 adapter 若支持工具、引用、thinking 或媒体 block，应保留并分别建模，不能强转成字符串。

## 可运行离线契约验证

`contracts.example.jsonl` 为三类 adapter 各提供一组 request config、canonical messages、固定 response 和期望的规范化结果。命令只使用 `.invalid` 示例域名和内部假密钥，不导入 HTTP client：

~~~powershell
python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl `
  --output artifacts/cloud-api/contracts.json
~~~

输出必须同时满足：

- `network_performed: false`；
- `real_credentials_used: false`；
- 三个 provider 样例全部通过；
- request headers 中认证值为 `<redacted>`；
- 输出序列化后不含内部假密钥；
- system/role/body、text、model、usage 与 finish reason 符合各自契约。

安装仓库后也可使用 `about-llm-cloud-contract`。输入 schema 为：

```json
{
  "case_id": "...",
  "provider": "openai-compatible",
  "config": {"base_url": "https://...invalid", "model": "...", "max_tokens": 32},
  "messages": [{"role": "user", "content": "..."}],
  "response": {},
  "expected": {"text": "...", "input_tokens": 1, "output_tokens": 1}
}
```

`provider` 允许值为 `openai-compatible`、`anthropic-messages` 或 `gemini-generate-content`。

JSONL loader 拒绝 duplicate key、`NaN`/`Infinity`、未知或缺失的顶层字段，以及带额外字段的 message；
这避免不同解析器对同一输入得出不同对象。

这个结果说明 adapter 对固定输入的构建、解析和脱敏行为符合预期。DNS、TLS、认证、配额、区域、SDK 兼容和
真实端点可用性需要在联网环境另行验证。

## OpenAI Responses typed-event 离线 replay

旧的 `OpenAICompatibleTextStream` 只建模 Chat Completions 风格的单 choice text delta。本项目另提供一条**独立**的 Responses reviewed subset，不把 `response.output[]`、content part、refusal 和 function call 压成一个字符串：

~~~powershell
python projects/cloud-api-contracts/openai_responses_replay.py `
  --events projects/cloud-api-contracts/openai-responses-events.example.jsonl
~~~

固定 3,208-byte JSONL 的 SHA-256 为 `f2947212c1f67adf6f35bc976264db28c30abe1a32310daa284df42ca5a54686`。15 个 SDK-shaped events 形成 2 个 output items，重建 `天气：晴。` 和 `lookup_weather({"city":"上海"})`，并对账 12 input + 9 output = 21 total tokens。Event projection 为 `sha256:9cc5964da2517f2076a1c624c2636bd8ca75077b89f024c7710b1b720cbd713e`，最终 receipt 为 `sha256:c4829c19895dcb4013141da3d11b5dc9befee8189210a0901f0cb14c19942579`。

状态机覆盖 `response.created/in_progress`，message 的 output-text/refusal content lifecycle，function-call arguments delta/done，以及 completed/incomplete/failed terminal。它校验 response id/model、output/item/content index、delta→done→terminal output、usage 三数和 item 完成顺序；reasoning/其他 item 只保留 opaque 生命周期，不解释语义。`sequence_number` 从 0 严格连续是本地 evidence artifact 规则，不是任意网络恢复的公开保证。

Loader 拒绝 duplicate key、`NaN`/`Infinity`、invalid UTF-8、空行、缺少末尾换行、未知或额外 event 字段、4 MiB 文件/1 MiB 单行/10,000 events 超限。Function arguments 无效时保留原字符串并报告 `arguments_is_strict_object=false`，不能冒充已校验工具参数。

该命令没有执行 HTTP/SSE/WebSocket framing、OpenAI SDK 或远程 API，也不会认证样例文件中声明的
`model`、response id 或 usage。真实 OpenAI 服务、完整 Responses API、模型质量、安全、计费和生产可靠性
需要其他证据。当前产品接口与事件 reference 的核对日期为 2026-08-14；完整对象图和生产分层见 [GPT 家族](../../docs/models/gpt.md)。

## Opaque reasoning artifact replay matrix

运行本地上下文绑定实验：

~~~powershell
python -m about_llm.integrations.cloud_api_cli reasoning-replay-matrix `
  --output artifacts/cloud-api/reasoning-replay-matrix.json
~~~

命令使用 `cryptography` 的 AES-256-GCM、固定虚构 key/nonce、虚构 reasoning bytes 和内存 ledger，不解析 provider
signature，不发送请求，也不输出 plaintext/ciphertext。16 个 case 先展示 content-only AEAD 会接受四类错误上下文，
再检查 context-bound envelope 遇到 identity/session/branch/predecessor/model/expiry/key/tamper/replay 漂移时是否拒绝重放。

顶层 `passed: true` 表示所有观察符合实验预期；其中四个 `unsafe_acceptance_demonstrated: true` 是故意保留的弱协议反例，不是安全通过。完整协议、论文时效和故障注入步骤见 [Reasoning 工件安全](../../docs/quality/reasoning-artifact-security.md)与[实验 0D](../../docs/practice/labs/lab-0d-reasoning-artifact-security.md)。

这个实验不模拟任一真实供应商格式或密码系统。内存 nonce/consumption ledger 也不覆盖持久化、多进程、
多区域一致性、KMS/HSM 或 key rotation；predecessor digest 不是完整 fork/compaction/Merkle 协议。

## Trajectory release gate

检查 allowlist publication projection：

~~~powershell
python -m about_llm.integrations.cloud_api_cli trajectory-release-gate `
  --input projects/cloud-api-contracts/trajectory-release.example.json `
  --output artifacts/cloud-api/trajectory-release-report.json
~~~

输入按无歧义的 JSON/JSONL 规则解析：duplicate key、non-finite number 和非对象 candidate 会在解析阶段失败。
发布 schema 只允许 `text/tool_call/tool_result/citation` block；遇到 reasoning/thinking/signature/encrypted、
嵌套禁用字段、未知 block 或未知字段时停止发布。安全投影退出 0，有 finding 的投影退出 1，输入无效退出 2。

报告不回显输入值、未知类型或任意字段名，并固定 `provider_artifacts_interpreted: false`、`plaintext_values_emitted: false`。`secret_pii_scan_performed: false` 表示它不检查允许文本中的 secret/PII、版权、consent 或用途；这个 gate 也不是 raw provider response sanitizer。

## 可运行重试决策表

`about_llm.integrations.retry` 是 provider-neutral 的纯策略层。它不发送请求，而是根据刚结束的 attempt、HTTP status 或本地错误类别、请求能否安全重放、结果是否不确定、剩余 deadline、`Retry-After` 和注入的 jitter 值给出 `RetryDecision`。生成固定决策表：

~~~powershell
python -m about_llm.integrations.cloud_api_cli retry-matrix `
  --output artifacts/cloud-api/retry-matrix.json
~~~

教学默认 allowlist 是 `408/429/500/502/503/504`，不是“所有 4xx/5xx”。`501/505` 不因属于 5xx 就自动重试。`max_attempts` 包含首次请求；本地 backoff 为有上限的 exponential backoff，`jitter_fraction` 由 caller 注入 `[0, 1]` 值，便于使用 seeded RNG 和确定性测试。

`Retry-After` 只严格接受非负整数 delta-seconds 或 HTTP-date。决策显式区分 `absent/valid/malformed`，避免把坏 header 当成从未收到。有效 header 覆盖本地 backoff 且不加 jitter；若等待时间超过策略上限或耗尽 deadline，策略停止，而不是无视服务端要求提前重试。格式错误时才回退本地 backoff。HTTP-date 的当前时间显式传入，测试不依赖墙上时钟。

自动重放还必须同时满足 `replay_safe=true` 和 `outcome_uncertain=false`。若请求可能触发工具副作用、超时后远端结果未知，或 caller 不知道是否安全，默认停止并调查。即使请求没有业务副作用，重复模型调用仍可能重复计费；不能假设不同 provider 或 endpoint 具有统一 idempotency-key 语义。

该矩阵只证明本地 allowlist、退避、header 解析和 fail-closed 分支在固定输入下自洽。它不访问网络，不证明真实 provider 当前错误语义、配额恢复时间、幂等支持、计费去重或端点可用；生产策略必须按 provider、endpoint 和固定 API 版本校准。

## 异步 HTTP 执行层

`about_llm.integrations.cloud_http.execute_json_request` 将 RequestSpec、RetryPolicy 与 caller-owned `httpx.AsyncClient` 接起来。它默认执行以下边界：

- 请求 origin 必须精确命中 allowlist；默认只允许 HTTPS、禁止 URL query/userinfo/fragment；
- `follow_redirects=False`，防止 adapter 在不知情时把调用转向另一 origin；
- 每个 attempt 使用独立 timeout，整个 retry/sleep 循环使用 monotonic deadline；
- PoolTimeout、ConnectTimeout、ConnectError 视为尚未建立可发送 HTTP 请求的 outcome-known 失败；write/read/protocol/执行器整体 attempt timeout 保守视为 outcome-uncertain，不自动重放；
- cancellation 原样传播，不转换成 retry；
- 2xx 必须返回 object JSON；拒绝 duplicate key、`NaN`/`Infinity`、overflow 到 infinity 的 number、数组顶层、错误 Content-Type 与超过 acceptance cap 的 body；
- attempt trace 只记录相对时间、status、稳定错误类别、request id 和 RetryDecision，不记录请求 body、认证 header、响应 body 或原始异常字符串。

先安装可选依赖：`python -m pip install -e ".[api]"`。

最小接线示例：

```python
import httpx

from about_llm.integrations.cloud_http import HttpExecutorConfig, execute_json_request
from about_llm.integrations.retry import RetryPolicy

async with httpx.AsyncClient() as client:
    result = await execute_json_request(
        client=client,
        request=request_spec,
        policy=RetryPolicy(max_attempts=3),
        config=HttpExecutorConfig(
            allowed_origins=frozenset({"https://provider.example"}),
            deadline_seconds=20,
            request_timeout_seconds=10,
        ),
        replay_safe=True,
    )
```

`max_response_bytes` 在当前非流式 `client.send` 已缓冲完整响应后才检查，只是解析 acceptance cap，不是下载过程的 ingress/memory 上限。真正限制接收内存必须使用 streaming、逐 chunk 累加上限并在超限时关闭响应。caller 还需拥有 client 生命周期、连接池、proxy/证书/DNS 配置，以及对取消后未知 outcome 的 reconciliation。

## 有界 SSE framing 与三类文本流状态机

`about_llm.inference.sse.SSEDecoder` 直接接收任意 byte chunk，不假设一次网络读取等于一行、一个事件、一个 token 或一个 UTF-8 字符。它处理 UTF-8 BOM、LF/CRLF/CR、comment、`event/data/id/retry` 字段、多行 data 拼接和跨 chunk 多字节字符，并分别限制 line/event/total bytes。只有空行完成事件；EOF 时残留半行或未由空行结束的事件会失败，不能把断流当正常完成。它不自动 reconnect，因为重新连接模型生成可能重放计算、文本或计费。

`about_llm.integrations.cloud_stream` 在 framing 之上提供三个**互不混用**的 text-only 状态机：

- OpenAI-compatible Chat Completions：单 choice、string content delta、usage、finish_reason 与独立 `[DONE]`；`[DONE]` 不是模型 finish reason；
- Anthropic Messages：校验 SSE event 与 payload type 一致，跟踪 text content block 的 start/delta/stop、message usage/stop reason 与 `message_stop`；
- Gemini `streamGenerateContent`：单 candidate、纯 text part、usageMetadata、finishReason，并在底层 EOF 后显式完成。

三者遇到 tool/function/refusal/thinking/媒体或未知 block 会失败，而不是丢弃后假装获得完整文本。这个旧 text-only SSE 契约不覆盖 OpenAI Responses API；上文的独立 `OpenAIResponsesEventReplay` 只覆盖 reviewed SDK-shaped typed-event subset，也没有接入 HTTP/SSE transport。Gemini Interactions API 和 provider-specific 兼容扩展仍不在三类 text stream 契约中。规范化 `StreamUpdate` 是 wire fragment/usage/finish/transport-end 事件，事件数和 text fragment 数都不是 token 数。

`execute_sse_request` 已把这两层接入 caller-owned `httpx.AsyncClient`：发送时使用 `stream=True`、禁止 redirect，校验 `text/event-stream`，逐个消费 `aiter_bytes` chunk，并在成功、HTTP 错误、截断、超限、idle/overall timeout、协议异常或取消时关闭 response。只有取得非 2xx headers 且尚未读取成功 body 时才允许走 RetryPolicy；一旦 2xx stream 开始，任何失败都终止且不重放。已经通过 `on_update` 交付的 fragment 是 partial output，不能撤回或改标为普通成功。

`on_update` 串行执行，天然形成客户端 backpressure；其耗时也计入 overall monotonic deadline。返回的 CloudStreamResult 累积规范化 text/usage/finish、事件数、更新数和 attempt trace。`bytes_received` 只统计 `httpx.aiter_bytes` 交给 parser 的 bytes，不是 token 数，也不一定等于压缩前/后的网络 wire bytes。当前实现仍会累积最终文本；SSE max_total gate 限制 parser 接收总量，而不是证明任意代理、内核或服务端缓冲有界。

MockTransport/AsyncByteStream 测试证明本地 response-close 与 partial-failure 控制流，不证明真实 TCP backpressure、HTTP/2 行为、服务端收到取消、停止生成或停止计费。RequestSpec 也必须由 caller 选择正确的 streaming endpoint/body；例如某些 Gemini `alt=sse` URL 需要显式 `allow_query=True`，但 API key 仍不应放入 query。

## Token 与估算费用 reservation

`about_llm.integrations.usage_budget` 提供 provider-neutral 的本地预算状态机。调用方先提供一份人工核对的 `TokenPricingSnapshot`，明确 `pricing_id/provider/model/revision/checked_at` 和 input/output 的 micro-USD per million tokens；所有费用计算使用整数，并对每次调用的 input+output 合计只向上取整一次。它不会从品牌名猜价格，也不会联网刷新价格。

发送前用目标 tokenizer 得到 estimated input tokens，再让 `reserve_request` 从实际 RequestSpec 提取最大输出 token 并一起原子预留：

```python
reservation = ledger.reserve_request(
    "stable-call-id",
    request=request_spec,
    billing_scope="stable-account/project-id",
    estimated_input_tokens=input_tokens,
)

# 只有能证明 transport 从未发送时才可释放。
# ledger.cancel_before_send(
#     reservation.reservation_id,
#     request_fingerprint=reservation.request_fingerprint,
# )

# 成功响应使用 provider 报告的 usage 结算并释放未用容量。
ledger.settle(
    reservation.reservation_id,
    request_fingerprint=reservation.request_fingerprint,
    actual_input_tokens=response.input_tokens,
    actual_output_tokens=response.output_tokens,
)
```

当前 helper 严格支持本仓库三类 text request：OpenAI-compatible/Anthropic 的顶层 `max_tokens`，以及 Gemini `generationConfig.maxOutputTokens`；缺失、布尔/非正数或两种字段同时出现都会在预留前失败。它还从前两类 body 或 Gemini URL 提取 model id，要求与 pricing snapshot 的 model 精确相等。Request fingerprint 绑定 identity version、billing scope、exact URL、完整 JSON body 和大小写规范化后的 headers；三类 credential header 的值先替换为占位符。同一 billing scope 换 key 不改变 identity，改 prompt/cap/URL/非敏感 header 或 billing scope 会改变 identity。`RequestSpec` 也拒绝大小写不同的重复 header name，避免 HTTP case-insensitive 语义产生含糊 identity。

若请求可能已发送但 usage 缺失或 outcome 不确定，调用 `mark_usage_uncertain`，按完整 reservation 保守入账；不能用 `cancel_before_send` 假装零费用。`settle/cancel/uncertain` 都要求 call id 与 request fingerprint 同时匹配，错 fingerprint 不会释放 active capacity。若实际 usage 超过估计并穿越 hard limit，ledger 会先提交已发生的 usage，再抛 `PostCallBudgetExceededError`，后续 reservation 保持 fail closed。Reservation 保存 request fingerprint，snapshot 重复记录 `pricing_id` 和三项 hard limit，避免把脱离请求/定价/限额身份的数字当成可审计结果。两线程测试证明同一进程内的并发调用不能重复花同一份 capacity；运行：

~~~powershell
python projects/cloud-api-contracts/usage_budget_toy.py
~~~

固定 authored 价格 `input=$1/M`、`output=$2/M` 时，60 input + 10 max output 预留 80 micro-USD；实际 58+4 结算为 66 micro-USD。这里的 micro-USD 是策略估值单位，不是 provider 发票。输入 estimate 可能因 tokenizer/template/隐藏 token 不匹配而偏低；真实计费还可能区分 cache、reasoning、batch/tier、最低计费单位、税费、额度和币种。Fingerprint 自洽不认证 caller、pricing 或 transport 真的发送了这份 RequestSpec；SHA-256 也不是签名/保密机制，低熵 prompt/hash 仍可能被字典猜测。`UsageBudgetLedger` 是单进程内存对象，不 durable、不跨 worker，也没有与 HTTP executor 做原子事务；重试 attempt、取消确认和 provider billing export 仍需调用方 reconciliation。

### SQLite durable quota 与崩溃恢复

`SQLiteUsageBudgetLedger` 复用同一 pricing、request fingerprint 和 terminal-state 契约，并将 config、reservation tombstone 与 append-only event timeline 放进一个本地 SQLite 文件。每次 reserve/settle/cancel/uncertain 都在 `BEGIN IMMEDIATE` 写事务内校验配置、计算全局快照、更新 reservation 并写 event；因此同一数据库上的多线程/多进程 writer 不能同时花掉最后一份本地 capacity。pricing 全字段、`checked_at`、三项 limit 与 canonical config fingerprint 绑定 singleton config，重开时配置漂移或表面篡改会 fail closed。

~~~powershell
python projects/cloud-api-contracts/sqlite_usage_budget_demo.py `
  --database artifacts/cloud-api/durable-budget.sqlite
~~~

离线 demo 先预留 60 input + 10 output，再用新 ledger 实例打开同一文件；未完成的 active reservation 仍占 80 micro-USD，随后按 usage uncertain 保守提交，并输出 reservation、snapshot、event timeline 和 scope。脚本拒绝覆盖已有数据库，以免把一次新演示和旧 reconciliation state 混在一起。

Active reservation **不按 TTL 自动释放**：进程死亡不能证明请求没发送，超时也不能证明 provider 没执行或没计费。运维必须按 call id、request trace 和 provider usage/billing export 人工或自动 reconciliation；只有能证明未发送才 cancel，否则 mark uncertain。SQLite 只提供单文件所在机器可达范围内的 durable atomic quota；它不与远程 provider call 原子，不证明 server cancellation、provider usage、invoice 或 exactly-once billing，也不是跨区域分布式 quota。Config fingerprint 和 request SHA-256 都没有密钥，能发现意外漂移但不能抵抗可协同改库并重算 fingerprint 的攻击者；数据库权限、加密、备份、签名审计和真实 billing 对账仍由部署层负责。

### 将 reservation 接入一次 HTTP attempt

`execute_budgeted_json_request` 把 RequestSpec、HTTP executor、response parser 与内存/SQLite ledger 接成一个 fail-closed reference。它先完成 exact-origin/HTTPS/query 等本地 target preflight，再 reserve；因此 preflight 错误不会留下 active 记录。发送后的 terminal 规则是：

- 只有 Pool/Connect 阶段且结构化 attempt trace 明确 `status_code=None`、`outcome_uncertain=false` 时 cancel；
- 任何 HTTP response 都证明 request 已越过客户端的“确定未发送”边界；4xx/5xx 即使 outcome known 也可能已产生 usage，当前 wrapper 按 reservation 记 uncertain；
- 2xx 且严格 parser 给出完整非负 input/output usage 时 settle；2xx malformed、缺 usage 或 parser 失败均记 uncertain；
- task cancellation 或 reservation 之后的未知本地异常也记 uncertain，不能把 client cancellation 等同于 provider cancellation；
- settle 若发现 actual usage 穿越 hard limit，ledger 仍先持久化再抛 post-call breach。

~~~powershell
python projects/cloud-api-contracts/budgeted_http_demo.py `
  --database artifacts/cloud-api/budgeted-http.sqlite
~~~

这个 demo 用 `httpx.MockTransport` 完成一个 58+4 usage 的成功结算，再返回一个带 request id 的 HTTP 500；最终 SQLite 中分别是 settled 66 micro-USD 与 uncertain 80 micro-USD，共 committed 146。输出不证明网络、真实 provider error/usage 或 invoice。

旧的单-attempt wrapper 继续强制 `RetryPolicy(max_attempts=1)`。一次逻辑调用预留一次，却在内部自动 replay 三次，会让三次都可能计费而账本只覆盖一个 cap；成功响应 usage 也不能证明前面失败 attempt 为零费用。因此每次 replay 都必须新建独立 reservation（使用唯一 attempt id），并分别 settle/cancel/uncertain；不能复用已成为 tombstone 的 reservation id。

这仍不是跨系统事务：provider 已执行后、SQLite terminal commit 前进程可能崩溃，留下 active 供重启后 reconciliation。`BudgetedCloudCallError` 只给稳定 reason、attempt trace 与预算状态，不把 raw exception/body/密钥写进消息；但部署仍需安全保存 provider request id、billing export 和人工处置。

### 逐 attempt 预算重试 orchestrator

`execute_budgeted_json_request_with_retry` 复用底层 executor 的 bounded retry、`Retry-After`、replay-safe、outcome uncertainty 与 monotonic deadline；它不在外层重写一套 delay 逻辑。默认关闭的 `before_attempt` hook 在每次真正发送前创建 `logical-call:attempt:N` reservation，`after_attempt` hook 在 sleep 或下一次 attempt 前完成 terminal transition：

- 明确 Pool/Connect 前失败为 `cancelled`，释放该 attempt 的 capacity；
- 任意 HTTP status、outcome-uncertain transport、strict-response failure 为 `uncertain`，提交完整 attempt reservation；
- 2xx strict JSON 先保持 active，只有 parser 给出完整 usage 才 `settled`；缺 usage 或 parser failure 改为 `uncertain`；
- reserve 后、trace 前发生 task cancellation，当前 active reservation 仍记 `uncertain`；
- 下一 attempt reserve 若过 hard gate，网络不会再发送；此前 attempt 的 tombstone 保持不变。

~~~powershell
python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/budgeted-retry.sqlite
~~~

离线 fixture 固定为 HTTP 500→200：attempt 1 的 60+10 cap 按 uncertain 提交 80 micro-USD，attempt 2 的 provider-reported 58+4 usage settled 66 micro-USD，逻辑调用合计 146。若 hard limit 设为 140，attempt 1 入账后，attempt 2 的 80 micro-USD reservation 会在 transport 前被 gate 拒绝，因此只发生一次 MockTransport call。429 fixture 另证明原 executor 的有效 `Retry-After` delay 没有被 wrapper 丢失。

这是 JSON-only reference。它不为 streaming partial output 自动重放，不解析 provider-specific error usage，也不声称 HTTP 500 一定收费；`uncertain` 表示本地证据不足以证明零费用。`BudgetedCloudRetryError` 汇总已 terminalize 的 ordered attempts 与最终本地 snapshot，但仍不证明 provider usage、invoice、server cancellation、idempotency 或 exactly-once billing。

## 生产调用层仍需补充

当前 reference executor 仍未实现或证明：

- 按固定 provider/API 版本校准 retryable status/error，不把教学 allowlist 当通用事实；
- 按 provider 解析 error body、request id、配额与 idempotency-key 语义；
- 真实 provider 的取消确认、partial-output reconciliation 与 streaming error body；
- 按 provider/API version 解析 error usage 与 billing receipt；当前通用 orchestrator 对失败 attempt 只能保守记 uncertain；
- streaming partial-output 的 attempt-level reservation、禁止/允许 reconnect 的 provider-specific 协议与 reconciliation；
- proxy、证书 pinning/mTLS、DNS rebinding/egress policy 等进程外网络控制；
- 每次真实调用的 provider/model/API revision/checked_at 记录；pricing snapshot 不能替代 request/runtime manifest。

## 配置原则

base URL、model id、API version 和密钥来自配置/秘密管理，不写死在教材。DeepSeek/Qwen 的云产品与开放权重 checkpoint 不能假定相同。Claude/Gemini 内部架构未公开部分保持未知。

## 离线测试

`tests/test_cloud_api.py`、`tests/test_cloud_api_cli.py`、`tests/test_openai_responses_replay.py`、`tests/test_reasoning_artifact.py`、`tests/test_trajectory_release.py`、`tests/test_cloud_api_retry.py`、`tests/test_cloud_http.py`、`tests/test_sse.py`、`tests/test_cloud_stream.py`、`tests/test_usage_budget.py`、`tests/test_sqlite_usage_budget.py` 和 `tests/test_budgeted_cloud.py` 验证三类字段映射、Responses typed lifecycle、严格 JSON/数值类型、reasoning context binding、trajectory publication allowlist、retry/HTTP 控制、任意 byte framing、三种文本流状态，以及内存/SQLite 预算 reservation、单-attempt 接线和逐 attempt retry orchestration。SQLite 测试覆盖重开、子进程退出、多连接竞争、配置漂移、event 写失败回滚、tombstone 与 post-call overrun 持久化；budgeted HTTP 测试覆盖 500→200、Connect→200、hard gate、`Retry-After`、outcome-uncertain、replay-unsafe、malformed/missing usage、cancel 与 SQLite event 顺序。它们不模拟远程调用原子性、provider artifact 或计费。HTTP 测试使用 `httpx.MockTransport` 或内存 byte fixture；Responses 测试使用 authored JSONL，不执行 OpenAI SDK、真实 DNS/TLS、网络请求或计费。网络 smoke test 必须显式标记 network，并设置请求数、token/费用上限、timeout 与允许的 base URL。
