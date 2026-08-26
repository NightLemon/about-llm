# Gemini 接入与证据台账

本页保存仓库当前 Gemini adapter、stream parser、预算固定样例和可陈述 claim 的精确边界。第一次学习请从
[Gemini 总览](../models/gemini.md)开始；只有需要复核实现覆盖、测试证据或作品集表述时再查本页。

**证据导航**：[总览](../models/gemini.md) · [Interactions API](../models/gemini-interactions.md) · [generateContent 与多模态](../models/gemini-generate-content.md) · [生产接入](../models/gemini-production.md)
{ .doc-nav }


## 闭源多模态 API 的 L0–L5 证据阶梯

“Gemini 支持某能力”不是一个足够精确的结论。至少要回答：哪个产品、哪个 API、哪个版本、哪个 model id、哪个区域、哪种输入、何时核对、证据来自哪里。

| 层级 | Gemini 场景中的证据 | 能支持的结论 | 仍不能支持 |
|---|---|---|---|
| L0 | `Gemini` 品牌或家族名 | 候选生态 | model id、接口、窗口、质量 |
| L1 | 论文、发布说明、产品页 | 研究/产品声明 | 当前 wire contract、你的账号可用性 |
| L2 | 带日期的官方 API/reference 页面 | 当日文档声明的字段与 lifecycle | raw bytes 不变、endpoint 已接受请求 |
| L3 | authored JSON/SSE/MockTransport/SQLite controls | 本地 adapter、parser、状态机与 policy 行为 | Google SDK、真实网络、provider usage |
| L4 | 在记录好版本和环境后进行一次受限真实调用，并保存原始 receipt | 该时刻单个账号/区域/API/model/workload 的协议行为 | 代表性质量、容量或生产可靠性 |
| L5 | 代表性评测、压测、故障注入、账单对账与 rollout | 明确 workload 下的发布/运维结论 | 其他平台、区域、模型或未来版本 |

这里的 L2 也不是 immutable byte evidence。官方页面会重定向、更新字段、改变导航；`Last updated` 只描述网页，不证明你的请求经过相同服务版本。

本仓库当前为 Gemini 的两个小范围取得 L3 离线证据：`generateContent` 的 text-only 映射，以及
Interactions 的一条 `function_call → requires_action` SSE 生命周期。

- 仓库固定的 request/response 样例；
- 仓库固定的 SSE event 样例；
- provider-neutral strict JSON/SSE/HTTP/retry controls；
- provider-neutral memory/SQLite budget controls；
- `network_performed=false` 的三供应商契约报告。

这些证据不能相加成 L4：

- Interactions 固定回放通过，不表示 request builder、远端 transport 或完整 API 已实现；
- SSE state machine 通过，不表示执行过 streaming HTTP；
- HTTP MockTransport 通过，不表示 Google endpoint 收到 cancel；
- SQLite 账本通过，不表示 provider usage 或 invoice 正确；
- 多模态教材与 text fixture 并存，不表示运行过图片/音频/视频；
- API 产品文档不公开内部参数、层数、训练数据或完整后训练配方。

### 一条 claim 的最小身份

建议把每条外部结论写成：

```text
claim = (
  platform,
  api_surface,
  api_version,
  endpoint_origin,
  region_or_location,
  model_id,
  account_tier,
  request_identity,
  checked_at,
  evidence_level
)
```

任何一项缺失都可能改变语义。尤其不能只记录 SDK 类名，因为同一 SDK 可切换 Gemini API/Google Cloud、API 版本和模型。

## `generateContent` 的完整对象图

`generateContent` 以 request/response 而非 Interaction resource 为中心：

```text
GenerateContentRequest
├── contents[]
│   └── Content(role, parts[])
├── systemInstruction
├── tools[] / toolConfig
├── safetySettings[]
├── generationConfig
└── cachedContent

GenerateContentResponse
├── candidates[]
│   ├── content.parts[]
│   ├── finishReason
│   └── safetyRatings
├── promptFeedback
├── usageMetadata
├── modelVersion
└── responseId
```

官方 reference 说明 prompt 被阻止时可能没有 candidates，并应检查 `promptFeedback`。因此空候选不是空文本成功。


## 当前 request builder 实际覆盖什么 { #request-builder }

`build_gemini_request()` 当前只支持：

- HTTPS base URL；
- `x-goog-api-key` header；
- `/v1beta/models/{model}:generateContent`；
- system text → `systemInstruction.parts[0].text`；
- user/assistant text → `user/model` role；
- 每条 message 一个 text part；
- `maxOutputTokens` 与 temperature。

它没有实现：

- Interactions；
- Google Cloud IAM/region endpoint；
- image/audio/video/document input；
- file/cached content lifecycle；
- function calling/code execution；
- safety settings；
- structured output；
- thinking config；
- streaming HTTP；
- SDK transport 与真实 credentials。


## 当前 response parser 的有损边界

`parse_gemini_response()`：

- 要求恰好一个 candidate；零 candidate 时提示检查 `promptFeedback`；
- 要求 candidate 至少包含一个 part，且每个 part 只能包含非空 `text`；
- 多 candidate、非 text part 或混合 text/function part 都会停止并报错；
- 读取 `modelVersion`；
- 读取 `promptTokenCount` / `candidatesTokenCount`；
- 读取第一个 candidate 的 `finishReason`。

它不支持：

- 第二及后续 candidates；
- `promptFeedback`；
- safety ratings；
- function call/result；
- image/audio/video/document parts；
- thought/signature；
- citations/grounding/provider extension；
- `responseId`；
- cache/tool/thought/total token 明细；
- unknown parts。

因此它是 text-only projection，不是完整 `GenerateContentResponse` adapter。当前实现选择拒绝无法保真表达的结果；
真实生产 adapter 也可以增加明确的 part/candidate 类型，但不能静默吞字段。

## `streamGenerateContent` 与 Interactions stream 不可混写

仓库 `GeminiGenerateContentTextStream` 的固定子集：

```text
SSE message
  → candidates[0].content.parts[text]*
  → candidates[0].finishReason
  → usageMetadata
  → transport EOF
```

它强制：

- SSE `event` 必须是默认 `message`；
- 每个 payload 最多一个 candidate；
- candidate index 必须为 0；
- 只允许单字段 text part；
- `finishReason` 为非空字符串且只能出现一次；
- finish 后不能再有 candidate；
- usage token 为非负整数且 bool 不算整数；
- EOF 前必须见到 `finishReason`。

它没有：

- step start/delta/stop；
- function/multimodal/thought delta；
- prompt feedback；
- 多 candidate；
- HTTP `alt=sse`/URL 选择验证；
- provider error taxonomy；
- resume/reconnect；
- backpressure 与 server cancel 证据。

所以正确表述是“实现并测试 `streamGenerateContent` text-only local state machine”，不是“完成 Gemini 流式 API”。

单独的 `GeminiInteractionsReplay` 读取仓库固定的 named SSE events，并检查：

- named event 与 payload `event_type` 相同；
- `interaction.created → step.start/delta/stop → interaction.completed → done → EOF` 的顺序；
- Interaction id 与 model 在流中保持一致；
- 两段 `arguments_delta` 在 step 关闭后组成严格 JSON object；
- `interaction.completed` event 可以携带 `requires_action` status；
- 未知 event 被记录后跳过，未支持的 step/delta 会留下不完整投影标记。

这条回放没有发送 Interactions request、执行函数、连接 Google、恢复断流或运行 background task。

## 固定预算 control { #budget-control }

仓库为这个实验准备了以下固定输入：

- 60 input + 10 max output；
- authored `$1/M input + $2/M output`；
- reserve 80 micro-USD；
- provider-shaped 58 input + 4 output；
- settle 66 micro-USD；
- 500→200 retry 的两 attempts 合计 146 micro-USD；
- hard limit=140 时第二次发送前被 gate 拒绝。

这些数字只证明整数算术、reservation/reconciliation 与本地 hard gate，**不是 Gemini/Google 价格、usage 或发票**。

## 当前 capability matrix

启动或发布前生成机器可读能力快照：

| 能力 | Interactions | `generateContent` | 本仓库证据 |
|---|---|---|---|
| text request/response | official L2 | official L2 + local L3 subset | `generateContent` 为 local L3 |
| typed steps | official L2 | 不适用 | Interactions function-call step 的 local L3 回放 |
| text streaming | official L2 | official L2 + local L3 subset | 两套 API 都只有离线 state/framing 证据 |
| multimodal input | model/API dependent | model/API dependent | 未执行 |
| function calling | API/model dependent | API/model dependent | 只重建 Interactions proposal，未执行工具 |
| server state | `previous_interaction_id` | 不同历史模型 | 未执行 |
| background | Interactions capability | 不借用 | 未执行 |
| custom safety/config | version/capability probe | surface-specific | 未执行 |
| usage/billing | provider-reported/账单 | provider-reported/账单 | authored budget only |

capability negotiation 失败应阻止发布或走显式降级，不能静默删字段后继续。

## 作品集与简历证据边界

当前仓库可如实写：

> 为 Gemini `generateContent` 构建 text-only canonical→wire adapter 与单 candidate stream state machine；另为
> Interactions 固定 SSE 实现 step 回放，重建分片函数参数，并区分 `interaction.completed` event、
> `requires_action` resource status 与应用成功。

必须紧邻披露：

- 两条路径都使用仓库固定的离线记录；
- Interactions 只实现 reviewed event 子集，没有 request builder、SDK 或远端 transport；
- 未执行 Google GenAI SDK；
- 未使用账号、真实 DNS/TLS/HTTP/SSE；
- 未运行 Gemini model、多模态、tool execution、thought/signature、file/cache；
- 未验证 usage/billing、质量、性能、安全或生产 SLO；
- 80/66/146 micro-USD 是固定样例中的数字，不是 Google 价格。

不能写：

- “接入最新 Gemini”；
- “完成 Interactions API”；
- “支持全量多模态”；
- “实现安全工具调用”；
- “验证长上下文与缓存收益”；
- “降低 Gemini 成本 45%”；
- “生产级 Google Cloud 网关”。

若未来取得真实证据，简历数字必须能回溯到 platform、API/version、model、region、case denominator、artifact、usage/billing 与环境。

## 可运行实验

本仓库 `about_llm.integrations.cloud_api` 对 `generateContent` 做离线 text-only 契约测试：assistant role 映射为
`model`、system 映射到 `systemInstruction`、响应提取 `usageMetadata`；
`GeminiGenerateContentTextStream` 另检查单 candidate 的 text parts、usageMetadata、finishReason 与 EOF。

下面的命令回放另一套 Interactions 协议：

```powershell
python projects/cloud-api-contracts/gemini_interactions_replay.py
```

它从 SSE bytes 重建 `lookup_weather(city="上海")` proposal，并正确停在 `requires_action`。它没有实现真实
streaming HTTP、工具执行、background/resume、多模态 steps 或 Google 端点验证。

建议新增四组 fixture/实验：

1. **契约矩阵**：text、多 text part、无 candidate、多 candidate、function call、安全阻止、未知 part 与错误响应；
2. **API 迁移**：同一 task 分别经 Interactions 和 `generateContent` adapter，检查内部规范化结果与状态差异；
3. **多模态反事实**：原样、遮蔽、局部替换、文本先验四个版本成组评测；
4. **文件治理**：跨租户 id、过期/删除、版本更新和超大文件必须失败或走审批路径。

若运行真实 API，要把结果标为具体平台、区域、model id、API surface、checked_at 和账号配置下的验证；离线 fixture 通过不能证明真实配额、媒体限制或服务端状态行为。

## 一手资料

- Google，[Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)，GA、状态、steps、后台执行与存储边界；核对日期 2026-08-26。
- Google，[Interactions API reference](https://ai.google.dev/api/interactions-api)，resource、status、methods、steps 与 API version；核对日期 2026-08-26。
- Google，[Streaming interactions](https://ai.google.dev/gemini-api/docs/streaming)，SSE interaction/step/terminal lifecycle；核对日期 2026-08-26。
- Google，[GenerateContent API reference](https://ai.google.dev/api/generate-content)，`contents`、`systemInstruction`、candidates、prompt feedback 与 usage；核对日期 2026-08-15。
- Google，[Text generation](https://ai.google.dev/gemini-api/docs/text-generation)，当前入口、`output_text` 有损边界与 stateless step preservation；核对日期 2026-08-15。
- Google Cloud，[Agent Platform model overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models)，云平台模型、访问与治理导航；核对日期 2026-08-15。
- 目标 model page、SDK reference、data retention 与区域文档；生产部署时的最高优先级证据。
