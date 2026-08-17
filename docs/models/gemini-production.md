# Gemini 生产接入：身份、预算、观测与迁移

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备把 Gemini 接入真实业务、建立发布门禁和回滚方案的工程师。
- **先修**：读过 [Gemini 总览](gemini.md)，并选定 Interactions 或 `generateContent`。
- **首次阅读**：身份 → retry/outcome → usage budget → evaluation → rollout。
- **完成信号**：能写出带 model/API/platform identity 的发布工件，并解释超时后为何不一定能安全重试。
- **卡住时**：先完成 text-only smoke，再一次只增加 streaming、tool、state 或多模态中的一项。

</div>

**章节导航**：[总览](gemini.md) · [Interactions API](gemini-interactions.md) · [generateContent 与多模态](gemini-generate-content.md) · [证据台账](../evidence/gemini-controls.md)
{ .doc-nav }

生产接入的核心不是“请求成功过一次”，而是让每个结论都绑定明确身份、分母和故障边界。下面从可复现配置开始，逐步走到评测、灰度和回滚。


## 平台与 endpoint 身份

至少把部署配置拆成四组：

| 组 | 典型字段 | 漂移风险 |
|---|---|---|
| 产品 | Gemini API / Google Cloud 托管 surface | auth、区域、治理和 endpoint 不同 |
| 接口 | Interactions / `generateContent` | object graph、状态和事件不同 |
| 模型 | exact model id / alias / preview | 能力、默认值和下线策略不同 |
| 运行 | API version、SDK version、region、tier | 字段、配额、保留和错误语义不同 |

不要让 `provider="gemini"` 同时承担这些身份。一个更可审计的配置应至少包括：

```yaml
platform: gemini-api
api_surface: interactions
api_version: v1
endpoint_origin: https://generativelanguage.googleapis.com
model_id: deployment-owned-exact-id
region_or_location: platform-defined
checked_at: 2026-08-15
storage_mode: explicit
```

这只是配置形状，不是已运行配置；教材不写入真实密钥、项目号或当前型号榜单。

### alias、preview 与固定 revision

闭源 API 常无法像开放权重仓库那样固定 commit。生产身份至少要保存：

- 请求中的 exact model id；
- 响应中的 model/version 字段（若该 surface 提供）；
- API/SDK 版本；
- 首次与最近验证时间；
- provider request/interaction id；
- capability probe 结果；
- 评测 artifact 与 rollout decision id。

如果 model id 是会漂移的 alias，必须承认它不是 immutable revision。响应字段也只能证明 provider 回报了什么，不能自行认证权重来源。

## Canonical Core 与两套 wire model

业务层可以共享一个窄 canonical core：

```text
CanonicalRequest
├── conversation turns / content blocks
├── system policy
├── output contract
├── tool proposals
├── generation budget
└── trace / tenant / data-policy identity
```

但 wire adapter 必须分开：

```text
canonical request
├── Interactions adapter
│   └── interaction + input/steps + previous_interaction_id
└── generateContent adapter
    └── contents/parts + systemInstruction + generationConfig
```

归一化只应覆盖业务确实共享的语义。以下信息不能在 adapter 中静默丢弃：

- step/part 的原始类型与顺序；
- tool call/result identity；
- thought/signature 等 continuation artifact；
- prompt/candidate safety feedback；
- model version、response/interaction id；
- usage 的分项与计量口径；
- provider terminal 与 transport EOF；
- unknown extension fields 的受控原始投影。

“返回一段 text”是有损 convenience view，不是完整响应模型。

## Safety surface 的版本漂移

2026-08-15 核对时，Interactions overview 的 limitation 文本与 API reference 可见字段对 custom safety surface 存在需要按版本解释的差异。正确工程动作不是选一页当永久真相，而是：

1. 固定 `v1` 或 `v1beta`；
2. 固定 SDK 版本与生成的 wire request；
3. 在目标 endpoint 做 capability probe；
4. 保存 accepted/rejected response 与 checked_at；
5. 将 unsupported field 视为显式能力差异；
6. 发布前用目标模型重跑安全与 over-refusal gate。

provider safety filter 只是 defense-in-depth。应用仍需：

- 输入/文件/URL 安全；
- 数据和工具授权；
- 输出政策；
- 业务 verifier；
- 人工升级；
- abuse monitoring；
- incident/rollback。

## Retry、幂等与 outcome uncertainty

每个失败先回答三个问题：

1. `retryable?`：协议/错误策略允许重试吗？
2. `replay safe?`：相同业务动作可安全重复吗？
3. `outcome known?`：能证明 provider 未接受/执行吗？

| 场景 | outcome | 默认动作 |
|---|---|---|
| target/preflight 失败 | known not sent | 修配置，不记 provider usage |
| connect 前明确失败 | likely not sent，按实现证据 | 可有限重试 |
| 已发送后 timeout/reset | uncertain | 不自动重放副作用任务 |
| HTTP/provider 明确 terminal error | 按固定 allowlist | 保存 request id 后决定 |
| partial stream 已发布 | externally visible | 默认不 replay |
| background id 已取得 | 可查询 | 先 get/reconcile，不新建 |

SSE 断开不等于 cancel；client cancel 不等于 server stop；server stop 不等于零 usage/费用。

## Usage、费用与预算账本

`GenerateContentResponse.usageMetadata` 与 Interactions usage object 的字段和口径不同。接入时按 API surface 保存原始分项，不要只抽取 input/output 两个总数。

预算流程应覆盖每一次 attempt：

```text
exact request identity
  → conservative input estimate + output cap
  → reserve before send
  → one reservation per attempt
  → settle | cancel-before-send | uncertain
  → provider billing reconciliation
```

价格快照要独立版本化，并记录平台、模型、模态、cache、thinking、tool、batch/tier、区域和生效时间。provider-reported usage 用于近实时控制，最终仍需与 billing export 对账。

本地 reservation 只能证明预算代码如何记账，不能证明供应商最终计费。仓库使用的固定算术样例与边界见[证据台账](../evidence/gemini-controls.md#budget-control)。

## 生产 adapter 的目录与能力协商

建议拆分：

```text
integrations/gemini/
├── canonical.py
├── identity.py
├── interactions_request.py
├── interactions_response.py
├── interactions_stream.py
├── generate_content_request.py
├── generate_content_response.py
├── generate_content_stream.py
├── multimodal.py
├── tools.py
├── errors.py
├── retry.py
├── usage.py
├── governance.py
└── fixtures/
```

不要用一个 `if provider == "gemini"` 的巨大 parser 同时处理两个 object graph。

## 用 capability probe 阻止静默降级

同一字段可能因 API 版本、平台、模型或账号能力而不同。发布前对目标 endpoint 运行最小 probe，并保存 accepted/rejected response 与核对时间。

| 能力 | Interactions | `generateContent` | 发布动作 |
|---|---|---|---|
| typed steps / server state | 按目标版本验证 | 不借用该对象图 | adapter 缺失则阻止发布 |
| text streaming | 验证 step lifecycle | 验证 candidate/finish lifecycle | 使用独立 parser |
| multimodal input | model/API dependent | model/API dependent | 对每种 MIME 单独 probe 与评测 |
| function calling | surface-specific | surface-specific | proposal 仍进入本地授权层 |
| safety / structured output | version/capability dependent | version/capability dependent | 不支持时显式拒绝或降级 |
| usage / billing | 保存原始 usage | 保存原始 usage | 与价格快照和账单对账 |

能力协商失败应阻止发布或进入预先设计的降级路径，不能静默删除字段后继续请求。

## 观测与隐私

最小生产 trace 可记录：

- internal request/attempt id；
- provider response/interaction id；
- platform/API/version/model/region/tier；
- request/template/tool/schema fingerprints；
- part/step type 与大小，不默认记录内容；
- lifecycle timestamps；
- finish/status/error taxonomy；
- usage 分项与 budget terminal；
- store/cache/file policy；
- verifier/rollout decision；
- redaction/projection version。

默认不记录：

- API key/authorization；
- raw media；
- thought/signature；
- tool secret；
- 敏感 prompt/result；
- 可跨租户复用的 file/interaction id。

hash 不是匿名化。低熵 prompt、文件名和短 ID 仍可被猜测；需要 keyed fingerprint、权限与 retention policy。

## Evaluation unit 与分母

模型/API 迁移评测的最小单位是 task attempt，不是 text response。建议保存：

```text
TaskResult
├── task/case/slice identity
├── all attempts
├── provider/transport/parse/tool/verifier outcomes
├── final publish outcome
├── latency timestamps
├── usage/cost
└── evidence artifacts
```

至少报告：

- all-attempt provider success；
- parse success；
- tool proposal/authorization/effect success；
- task success；
- citation/grounding；
- safety violation 与 over-refusal；
- multimodal counterfactual consistency；
- latency（offered 与 success-conditional 分开）；
- usage/cost per attempted 与 successful task；
- background completion/cancel/reconciliation；
- unknown/unjudged/pending 分母。

只在成功响应上算质量会隐藏 blocked、timeout、parse failure 和 tool failure。

### paired migration protocol

迁移 Interactions 前后应固定：

- case set/split；
- platform/region/account；
- exact model identity；
- system/tool/schema；
- media bytes/file versions；
- generation budget；
- evaluator/rubric；
- timeout/retry；
- usage/cost口径；
- store/cache/history policy。

如果 API surface 不能保持某字段，应把它记录为 treatment difference，而不是假装同条件比较。

## 生产 rollout / rollback bundle

发布工件至少包含：

- adapter/version manifest；
- official docs checked_at 与 capability probe；
- model/API/platform identity；
- eval dataset/report；
- raw/typed parser fixtures；
- error/retry/cancel matrix；
- tool/ACL/approval policy；
- budget/pricing snapshot；
- logging/redaction/retention review；
- canary/shadow observation；
- rollback trigger 与旧 adapter；
- incident owner 与 reconciliation runbook。

推荐顺序：

```text
offline contract
  → restricted real smoke
  → paired shadow
  → low-volume canary
  → staged rollout
  → steady-state gate
```

回滚不只切 model id。还要恢复 API surface、state ownership、stored interaction/file/cache policy、tool result compatibility 与 parser version。

## 故障定位树

```text
失败
├── identity/preflight
│   ├── platform/API/version/model
│   └── auth/region/capability
├── transport
│   ├── connect/TLS/timeout
│   └── HTTP/SSE framing
├── provider protocol
│   ├── Interaction step/status
│   └── candidate/part/finish
├── application
│   ├── parse/schema
│   ├── state/history/signature
│   ├── tool/ACL/effect
│   └── quality/safety/citation
└── economics/governance
    ├── usage/budget/billing
    └── storage/delete/logging
```

先确认 identity 和层级，再看模型输出。否则会把 endpoint 配错、状态丢失或 parser 有损误判成“模型能力下降”。

## 单张消费 GPU 与 Gemini API

Gemini 闭源 API 本身不在本地 GPU 部署。单张消费 GPU 可以用于：

- 本地 OCR/ASR/媒体抽取 baseline；
- embedding/reranker/RAG；
- 输入去敏与输出 verifier；
- 小模型 fallback；
- synthetic/counterfactual 媒体生成；
- 离线评测与可视化。

这些本地组件的 GPU 指标不能归因给 Gemini；远端 latency/usage 也不能解释本地 GPU 性能。

## 真实接入最小 smoke runbook

只有用户提供账号、合法权限与预算后才执行：

1. 固定 platform/API version/model/region；
2. 使用最小权限 secret，不写入仓库；
3. 设置请求数、token、费用、媒体大小与总时限上限；
4. 先 text-only non-tool case；
5. 保存 redacted request/response/headers/ids/usage；
6. 验证错误、timeout 与 credential redaction；
7. 再分别验证 streaming、state、tool、多模态；
8. 每项独立 artifact，不互借成功；
9. 删除测试 interaction/file/cache；
10. 与 dashboard/billing export 对账。

即使该 smoke 成功，也只得到 L4 单账号/区域/API/model/workload 证据，不得到代表性质量、容量、生产 SLO 或长期兼容性。

## 常见错误

- 把 Gemini 产品、Gemini API 和 Vertex AI 当作同一个 endpoint；
- 因同属 Gemini 就混用 Interactions API 与 `generateContent` 的字段和事件；
- 迁移到 `previous_interaction_id` 后忘记重新发送 interaction-scoped 配置；
- 忽略默认存储与删除策略，直接处理敏感会话；
- 只解析第一个 candidate 的第一个 text part；
- 把无 candidate 的安全阻止解析成空答案成功；
- 信任媒体 MIME、文件 id 或图片中的文字指令；
- 只做主题识别，没有反事实证明模型使用了目标模态；
- 把长窗口/文件上传当作 RAG、ACL 和引用系统的替代品。

## 面试追问

1. Interactions API 与 `generateContent` 的状态和 adapter 边界为什么不能混写？
2. `previous_interaction_id` 带来哪些状态归属、删除和配置继承问题？
3. contents/parts 相比纯文本 messages 怎样改变存储、解析与安全设计？
4. 如何证明视觉问答不是从文件名、OCR 线索或文字先验猜出？
5. 多模态提示注入如何跨提取器、模型和工具执行层隔离？
6. Gemini API 与 Vertex AI 选型要看哪些非模型因素？
7. 文件缓存和长上下文为什么不能替代权限感知 RAG？

## 一手资料

- Google，[Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)，GA、状态、steps、后台执行与存储边界；核对日期 2026-08-15。
- Google，[Interactions API reference](https://ai.google.dev/api/interactions-api)，resource、status、methods、steps 与 API version；核对日期 2026-08-15。
- Google，[Streaming interactions](https://ai.google.dev/gemini-api/docs/streaming)，SSE interaction/step/terminal lifecycle；核对日期 2026-08-15。
- Google，[GenerateContent API reference](https://ai.google.dev/api/generate-content)，`contents`、`systemInstruction`、candidates、prompt feedback 与 usage；核对日期 2026-08-15。
- Google，[Text generation](https://ai.google.dev/gemini-api/docs/text-generation)，当前入口、`output_text` 有损边界与 stateless step preservation；核对日期 2026-08-15。
- Google Cloud，[Agent Platform model overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models)，云平台模型、访问与治理导航；核对日期 2026-08-15。
- 目标 model page、SDK reference、data retention 与区域文档；生产部署时的最高优先级证据。
