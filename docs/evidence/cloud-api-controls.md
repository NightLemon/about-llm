# 云模型 API 证据台账：Adapter、Streaming 与预算验证

本页保存多供应商字段核对日期、无歧义 parser、stream/retry/budget 固定输入与精确证据边界，供 adapter 实现和
claim 审计使用。第一次学习请先读[云 API 契约基础](../models/cloud-api-contracts.md)，再读
[可靠性进阶](../models/cloud-api-reliability.md)。

**读者入口**：[契约基础](../models/cloud-api-contracts.md) · [可靠性进阶](../models/cloud-api-reliability.md) · [可运行项目](../practice/projects/cloud-api-contracts.md)
{ .doc-nav }

| 查什么 | 去哪里 |
|---|---|
| 第一次理解云 API 契约与可靠性 | [契约基础](../models/cloud-api-contracts.md)和[可靠性进阶](../models/cloud-api-reliability.md) |
| 核对 provider 对象与 adapter 投影 | 本页协议、canonical core 与 response 分区 |
| 核对 stream、retry、deadline 和预算 | 本页失败、流式、reserve/reconcile 分区 |
| 运行或判断外推边界 | 本页“可运行证据”“真实接入”和故障定位 |

## 适用范围与证据边界

本台账按以下核对项组织：

1. 区分 canonical business model、provider wire protocol 与 transport 三层；
2. 解释为什么 OpenAI-compatible 不等于完整语义兼容；
3. 为 text、refusal、tool、reasoning、citation 和 media 建立 typed representation；
4. 分开判断错误是否瞬时、请求能否重放、远端 outcome 是否确定；
5. 正确处理 SSE framing、provider terminal 与应用 stop string；
6. 在发送前 reserve token/费用，在每个 attempt 后 settle/cancel/mark uncertain；
7. 设计不泄露密钥、Prompt、reasoning 或被拒绝输入的审计工件。

本页结合三类证据：

- **官方接口文档**说明某个检查日期的公开对象与产品状态；
- **本仓库代码与测试**说明固定 authored/MockTransport/SQLite 输入上的本地行为；
- **生产 runbook**说明真实接入还需要收集什么证据。

离线契约不能证明真实账号、网络、模型、usage 或账单。文中出现的教学 allowlist、价格与 token 数只属于明确 fixture，不能外推成跨 provider 事实。

来源登记保存的是语义核对时的原始 `status`、`checked_at`、范围与复核方法；页面上的来源徽章应按构建时的有效状态显示，
它还会受 `next_review_at` 和探测结果影响。历史核对记录不能因为来源后来 stale、unknown 或 pending-review 而被改写，
也不能反过来把登记中的 `verified` 当作当前接口仍有效的保证。

## Provider surface 与对象差异

| Surface | 本仓库已核对的对象 | 不能统一掉的差异 |
|---|---|---|
| OpenAI-compatible Chat | `messages → choices → message/content`；text delta、usage、finish reason 与 `[DONE]` | 不等于 Responses typed items/events，也不代表所有兼容端点语义相同。 |
| OpenAI Responses | `response → output item → content part`，含 message/text/refusal/function-call typed items | 当前模型入口与字段为 2026-09-20 的时间敏感官方事实。[SOURCE:openai-model-catalog] |
| Anthropic Messages | 顶层 `system`、ordered content blocks、input/output usage、`stop_reason` 与 `message_stop` | Text-only projection 不覆盖 tool/thinking/signature 等 block。[SOURCE:anthropic-messages] |
| Gemini `generateContent` | `contents/parts`、`user/model`、`systemInstruction`、`usageMetadata` 与 `finishReason` | Interactions API 是另一套 state/lifecycle；两者不能互作别名。[SOURCE:gemini-interactions-overview] [SOURCE:gemini-generate-content] |
| DeepSeek / Qwen compatible endpoints | 只承认已验证的 Chat-like 字段 | 云服务、开放 checkpoint 与第三方托管是三种身份；response 自报 model 不认证权重。 |

业务层只统一稳定交集；adapter 保留 typed extension，不能用 `supports_tools=True` 一类布尔值吞掉版本与子能力。
至少区分 text、refusal、function call/result、citation、media 与 opaque provider item。若业务只接受 text，
遇到其他 item 应 fail closed，不能丢掉非文本内容后返回“成功”。

## Request、response 与发布 identity

| 对象 | 必须绑定 | 边界 |
|---|---|---|
| Request | Exact scheme/host/port/path、provider/model/API revision、canonical JSON body、非敏感 headers、billing scope、prompt/tool/adapter revision | `provider_options` 必须 closed-schema；SHA-256 不认证 caller、provider、pricing 或 transport。 |
| Response | Provider/request id、model/API、typed output、terminal、usage 与 provider receipt | Text、provider terminal、业务 success 和发票真实性不能互相推导。 |
| Structured output | Strict JSON、schema、domain invariant、trusted identity/ACL、budget/idempotency/approval、effect verifier | JSON/schema 合法不证明值真实、引用存在、资源归属或动作获批。 |
| Tool call | Name/arguments/call id 作为 proposal | Agent runtime 仍执行 schema、resource resolution、policy、approval、budget 与幂等。 |
| Opaque reasoning | Provider/type/version、tenant/session/branch/model audience、expiry/revocation 与 publication policy | 字段名像 signature/encryption 不证明安全；不作为事实或授权依据。 |
| Public projection | Closed-schema allowlist，再做 secret/PII、版权、consent 与用途审查 | 内部 trajectory 和被拒 raw output 不能直接返回用户。 |

2026 年论文 [Stealing Reasoning Traces from Proprietary LLM APIs](https://arxiv.org/abs/2608.09867)
是特定历史 API 版本的架构案例；作者说明截至 2026 年 8 月原攻击流程已因供应商缓解而不可复现。
本台账不据此声称当前 endpoint 有漏洞。完整控制见
[Opaque Reasoning 工件与轨迹安全](../quality/reasoning-artifact-security.md)。

## Retry、transport 与 streaming 矩阵

一次失败要分别判断 `retryable`、`replay safe` 与 `outcome known`；不能写成
`if status >= 400: retry()`。

| 场景 | 本地分类 / 默认动作 | 边界 |
|---|---|---|
| Schema/preflight 失败 | 未发送，修配置 | 仍需保留失败记录。 |
| Pool/connect 前失败 | 当前 policy 可视为 known-not-sent，再按 allowlist/预算重试 | 只适用于结构化的发送前证据。 |
| HTTP 429/5xx | 按固定 provider/version 校准；先记 response 再决策 | HTTP response 不证明 usage/费用为零。 |
| Write/read/protocol/attempt timeout | Outcome uncertain，默认停止并 reconciliation | 不能仅凭客户端异常自动重放。 |
| 2xx stream 截断/超限/parser error/cancel | Partial + uncertain，不自动重放 | 已发布 fragment 无法撤回。 |
| 工具副作用后响应丢失 | 查询 effect ledger；需要稳定业务幂等键 | Provider call id 不是完整业务 identity。 |

本仓库 `RetryPolicy` 的 authored allowlist 是 `408/429/500/502/503/504`，排除普通
`400/401/403/404` 与 `501/505`。`max_attempts` 包含首次请求；有效 `Retry-After` 不再叠加 jitter，
超过 policy cap 或 monotonic logical deadline 时停止。这些是本地策略，不是任何 provider 的现行规范。

Streaming 分三层：arbitrary network bytes → SSE framing → provider typed state → application update。
Decoder 覆盖 BOM、LF/CRLF/CR、多行 `data:`、comment、跨 chunk UTF-8 与 size gates。Provider terminal 分别是：

| Surface | Terminal contract |
|---|---|
| Chat subset | Finish reason + 独立 `[DONE]` |
| Anthropic Messages | Content-block lifecycle + `message_stop` |
| Gemini `streamGenerateContent` | `finishReason` + 底层 EOF |
| Responses | 独立 typed terminal event/state |

应用 stop string 是第四层。它可截断展示，但不能伪造 `finish_reason=stop`、修改 usage，或证明服务端停止生成/计费。
`bytes_received` 只表示 `httpx.aiter_bytes` 交给 parser 的 bytes。当前非流式 `max_response_bytes` 是 body
已缓冲后的 acceptance cap，不是 ingress/peak-memory 限制。

## Token 与费用账本

发送前按每个 attempt 预留，随后只允许进入一个终态：

| 状态 | 所需证据 |
|---|---|
| `cancelled` | 结构化证据证明从未发送 |
| `settled` | 严格解析到可信的完整 usage |
| `uncertain` | 可能已发送，但 usage/outcome 不足 |

合法转换是 `reserved → cancelled/settled/uncertain`，每个 reservation 只能进入一个 terminal state。

估算式为
\(R_i=\widehat T_{in}^{(i)}+T_{out,max}^{(i)}\)，费用按固定 pricing snapshot 做 micro-USD 整数估值。
Actual usage 超过 reservation 时先记录已发生值，再阻断未来调用，不能丢弃超额。

固定 MockTransport pricing fixture 为 input `$1/M`、output `$2/M`：60 input + 10 max output reserve 80 micro-USD，
58 input + 4 output settle 66。HTTP 500→200 时 attempt 1 的 80 记 uncertain，attempt 2 的 66 settled，
logical total 为 146；hard limit=140 时第二次 80 reservation 在 transport 前被拒绝。
这些数字不代表真实 provider 价格、usage 或发票。

`SQLiteUsageBudgetLedger` 用 `BEGIN IMMEDIATE` 串行化同文件 writer 并持久化 tombstone/event。
它不能与远程 generation/billing 原子提交，也不是跨机器 quota。Crash 后 active reservation 不能仅凭 TTL 释放；
需要 stable call/attempt/request id 与 billing export reconciliation。

成本至少同时报告 all-attempt、success-conditional、retry amplification、uncertain reservation、provider-specific usage、
币种/tier/税费和检查日期。平均 request cost 会奖励廉价失败；每成功任务成本的分母必须是 verified task success。

## 本地可执行证据

**三类 strict offline adapters**

~~~powershell
python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl `
  --output artifacts/cloud-api/contracts.json
~~~

它只构建/解析固定 JSONL、检查字段映射、strict JSON 与认证 header 脱敏；报告必须保留
`network_performed=false`、`real_credentials_used=false`。

**Responses typed-event replay**

固定 3,208-byte JSONL（`sha256:f2947212…5a54686`）包含 15 events/2 output items，重建
`天气：晴。`、`lookup_weather({"city":"上海"})` 与 12+9=21 usage。
Projection fingerprint 为 `sha256:9cc5964d…bd713e`，receipt 为 `sha256:c4829c19…42579`。
它没有执行 OpenAI SDK、HTTP/SSE/WebSocket、真实模型或 billing。

**Request/stream tests**

~~~powershell
python -m pytest tests/test_cloud_api.py tests/test_cloud_stream.py -q
~~~

**Retry、HTTP 与预算**

~~~powershell
python -m about_llm.integrations.cloud_api_cli retry-matrix `
  --output artifacts/cloud-api/retry-matrix.json
python -m pytest tests/test_cloud_api_retry.py tests/test_cloud_http.py `
  tests/test_usage_budget.py tests/test_sqlite_usage_budget.py `
  tests/test_budgeted_cloud.py -q
~~~

**Opaque artifact 与公开 projection**

~~~powershell
python -m about_llm.integrations.cloud_api_cli reasoning-replay-matrix `
  --output artifacts/cloud-api/reasoning-replay-matrix.json
python -m about_llm.integrations.cloud_api_cli trajectory-release-gate `
  --input projects/cloud-api-contracts/trajectory-release.example.json `
  --output artifacts/cloud-api/trajectory-release-report.json
~~~

这些命令使用 authored fixture、MockTransport、SQLite 或 AES control，不访问真实付费 endpoint。

## 真实接入证据缺口

真实 provider smoke 必须显式 opt-in，并限制 exact origin、请求/并发、input/output、每 attempt/总费用、
overall/idle timeout、model/API revision、tool 副作用和 artifact 生命周期。至少分栏记录：

1. DNS/TLS/HTTP 建连与认证；
2. 非流式 typed fields、terminal、usage 和 request id；
3. Stream event/terminal 与 partial/cancel 行为；
4. 限流、错误、重试和 unknown outcome；
5. Provider usage/billing export；
6. 代表性质量、延迟、成本与安全切片。

任何一层通过都不能借给其他层。真实单请求成功也不等于生产 SLO。

## Claim 导出矩阵

| 可写 claim | 必须紧邻披露 | 禁止升级 |
|---|---|---|
| 多供应商 adapter | Canonical core + typed extensions 保留三类 request/response/terminal 差异；strict JSON 与未知字段 fail closed | 不是“支持所有 OpenAI-compatible 模型”或完整 SDK/API 兼容。 |
| Stream controls | Byte framing、provider state、terminal 与应用 stop 分层，覆盖 malformed/truncated/duplicate terminal | MockTransport/AsyncByteStream 不证明真实网络、server cancel 或停止计费。 |
| Retry/outcome policy | Bounded retry、`Retry-After`、monotonic deadline、unknown outcome 与 partial stream 不自动 replay | 不是 provider 当前错误/限流/幂等规范。 |
| Budget ledger | 每 attempt reserve/reconcile、durable tombstone 与 hard-limit fault cases | 不证明 provider usage、invoice、distributed quota 或 exactly-once billing。 |
| Publication gate | Credential/body/reasoning/unknown blocks 经过脱敏和 allowlist projection | 不证明输入合规、业务正确或供应商内部安全。 |

故障时按 wire artifact identity → strict parse → provider state → retry/outcome → reservation/request id → effect/账单顺序定位。
不要用 `str(response)` 代替 strict parse，也不要通过打印 raw provider body、扩大重试或覆盖历史报告来制造“成功”。

## 一手资料与运行入口

- OpenAI，[Model catalog](https://developers.openai.com/api/docs/models)，当前模型与 Responses 入口；LLM 复核 2026-09-20。[SOURCE:openai-model-catalog]
- OpenAI，[Create a response](https://developers.openai.com/api/reference/resources/responses/methods/create)，Responses 对象；LLM 复核 2026-09-20。[SOURCE:openai-responses-create]
- OpenAI，[Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)，typed streaming 指南；LLM 复核 2026-09-20。[SOURCE:openai-responses-streaming]
- OpenAI，[Streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events)，Responses event reference；LLM 复核 2026-09-20。[SOURCE:openai-streaming-events]
- OpenAI，[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)，JSON/Schema/refusal/incomplete 边界；LLM 复核 2026-09-20。[SOURCE:openai-structured-outputs]
- Anthropic，[Messages API](https://platform.claude.com/docs/en/api/messages)，Messages request/content/usage/stop；LLM 复核 2026-09-20。[SOURCE:anthropic-messages]
- Google，[Gemini Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)，Interactions lifecycle；LLM 复核 2026-09-20。[SOURCE:gemini-interactions-overview]
- Google，[GenerateContent reference](https://ai.google.dev/api/generate-content)，`generateContent` 字段；LLM 复核 2026-09-20。[SOURCE:gemini-generate-content]
- 可运行项目：[Cloud API Contracts](../practice/projects/cloud-api-contracts.md)。
