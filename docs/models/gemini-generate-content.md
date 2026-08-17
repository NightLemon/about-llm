# Gemini `generateContent`：Content、Parts 与多模态

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：维护 `generateContent` adapter，或需要处理图片、音频、视频与文件输入的工程师。
- **先修**：先读 [Gemini 总览](gemini.md)，并理解 HTTP、JSON 与流式响应。
- **首次阅读**：对象图 → SSE 分层 → Parts 类型 → 多模态反事实评测。
- **完成信号**：能解释为什么 text-only parser 会丢信息，并为一种目标模态设计反事实 case。
- **卡住时**：先只处理一条 text part，再逐项加入 candidate、finish reason 和媒体 part。

</div>

**章节导航**：[总览](gemini.md) · [Interactions API](gemini-interactions.md) · [生产接入](gemini-production.md) · [证据台账](../evidence/gemini-controls.md)
{ .doc-nav }

`generateContent` 以一次请求/响应为中心。它与 Interactions 都能生成内容，但对象图、历史状态和流式 terminal 不同；生产代码应使用两个 adapter，而不是让一个 parser 猜当前协议。


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
├── cachedContent
└── service/store 等版本化字段

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

教学时可以先实现 text-only projection，但必须把它命名为窄 adapter，并对非 text part、多 candidate 和安全阻止显式拒绝。仓库当前 request builder 与 parser 的精确覆盖范围见[证据台账](../evidence/gemini-controls.md#request-builder)。

## `streamGenerateContent` 不是 Interactions stream

`streamGenerateContent` 组织的是 candidate、content part、finish reason 与 usage chunk；Interactions 组织的是 Interaction、step 和 typed delta。二者都可能经 SSE 传输，但共享 framing decoder 不等于共享 provider state machine。

正确复用边界是：底层 UTF-8/SSE/JSON 解码可以共用，上层事件类型、顺序约束、terminal 和错误语义必须分开。仓库当前实现覆盖到哪里，见[Gemini 接入证据台账](../evidence/gemini-controls.md)。

## SSE framing、typed state 与 HTTP 是三层

```text
arbitrary network bytes
  → SSE lines/events
  → provider JSON payload
  → Interactions event or GenerateContent chunk
  → canonical update
```

每层有不同失败模式：

- UTF-8 跨 chunk；
- CR/LF/CRLF 与多行 `data:`；
- line/event/total byte 超限；
- invalid/duplicate/non-finite JSON；
- event name 与 payload discriminator 不一致；
- step/candidate 状态错序；
- provider terminal 缺失；
- EOF、idle timeout、overall deadline；
- callback backpressure 和 partial output 已发布。

## Parts/Content 的安全类型系统

多模态不是把 base64 塞进 prompt。内部模型至少区分：

```text
TextPart
InlineMediaPart(mime, bytes, digest)
FileReferencePart(uri/id, owner, version)
FunctionCallPart(call_id, name, arguments)
FunctionResultPart(call_id, result, trust)
OpaqueProviderPart(type, allowlisted_projection)
```

每类 part 都要有：

- 字节与结构上限；
- MIME allowlist 与 magic-byte 检查；
- tenant/owner/ACL；
- content digest 与版本；
- 来源与解析器版本；
- retention/delete policy；
- 日志/前端 projection；
- prompt-injection trust label；
- 失败与隔离状态。

### inline、file 与 URI 的不同威胁

| 形式 | 优点 | 主要风险 |
|---|---|---|
| inline bytes | request identity 易绑定 | base64 放大、body/日志泄漏、内存峰值 |
| uploaded file id | 复用与大文件友好 | owner/TTL/delete/version 漂移 |
| remote URI | 少搬运 | SSRF、重定向、DNS、内容漂移、凭证泄漏 |

生产层必须先下载/验证还是让 provider 拉取，要按威胁模型选择；不能把用户给出的 URL 直接升级为可信媒体。

## 多模态评测：证明目标模态产生因果影响

单个“看图答对”case 不能证明模型使用图像。建议每个样本建立 paired counterfactual：

| 条件 | 改动 | 预期用途 |
|---|---|---|
| original | 完整媒体与中性文本 | 主任务 |
| no-media | 移除媒体 | 测文本先验 |
| masked | 遮蔽关键区域/时间段 | 测关键证据依赖 |
| swapped | 换关键对象/数值/片段 | 测答案是否随证据变 |
| misleading-text | 加入冲突标题/alt text | 测模态与注入鲁棒性 |
| corrupted | 破损/超限/错误 MIME | 测解析和拒答 |

每个 task 至少记录：

- media hash/version；
- 关键区域坐标、页码或时间戳；
- prompt/template hash；
- model/API/platform identity；
- raw typed output 与解析结果；
- answer/citation/abstention；
- safety/provider/parse errors；
- latency 与 usage 分项；
- paired case id 与统计分母。

### 指标要按能力拆分

- OCR：字符/字段 exact match，按脚本/语言分层；
- 图表：数值、单位、图例与坐标证据；
- 文档：字段、表格、跨页引用与页码；
- 图片：属性、关系、计数、空间与证据框；
- 音频：转写、说话人、事件、时间戳；
- 视频：动作顺序、事件定位、状态变化；
- 工具：proposal schema、授权、effect 与最终任务成功；
- 安全：attack success、over-refusal、正常任务质量。

micro average 会掩盖小模态和难例；报告 macro、slice、failure taxonomy 和置信区间。

## 长上下文不是单一上限

至少区分：

1. **文档声明上限**：当日 model/API page 的 vendor claim；
2. **请求接受上限**：目标账号/区域/配置实际接受；
3. **任务有效上限**：你的 workload 在位置、干扰和输出预算下仍通过 gate。

评测维度包括：

- evidence 在头/中/尾的位置；
- 多证据 join；
- 相互冲突版本；
- 全局聚合；
- recency/order；
- 长输出一致性；
- 多模态 token 分配；
- cache hit/miss；
- timeout/费用/阻止率。

只做 needle retrieval 不能证明多跳、冲突解决、引用或工具任务。

### 文件、cache 与 RAG

`cachedContent`、implicit cache、file API 和 `previous_interaction_id` 解决的是不同问题：

- cache：减少重复处理/计费的 provider 机制；
- file：媒体上传与引用生命周期；
- server state：历史续接；
- RAG：授权过滤、检索、版本、证据定位与发布引用。

它们不能互相替代。尤其 file id 或长窗口不自动提供 tenant ACL、最新版本选择、删除传播、引用正确性或 no-answer gate。

## Structured output：语法约束之后还有语义验证

无论 Interactions 的 response format 还是 `generateContent` 的 generation config/schema surface，结构化输出都只解决受支持 schema 子集中的格式问题。

还必须验证：

- schema/version identity；
- unknown/missing fields；
- duplicate key 与 non-finite number；
- 字符串/数组/嵌套深度上限；
- 单位、范围、交叉字段 invariant；
- 引用是否存在且获授权；
- refusal/block/incomplete 与普通 JSON 分开；
- raw response 与 parsed artifact fingerprint；
- retry 是否会产生不同业务 effect。

“有效 JSON”不等于事实正确、引用正确或工具可执行。

## Function calling：proposal、authorization 与 effect

Interactions 的 `function_call` step 或 `generateContent` function part 都只是候选动作：

```text
provider proposal
  → accumulate/parse strict arguments
  → schema validation
  → trusted resource resolution
  → tenant/subject/ACL/policy
  → approval if required
  → idempotent handler attempt
  → effect verification
  → function result
  → model continuation
```

必须绑定：

- provider call/step id；
- canonical call id；
- tool name 与 schema version；
- exact arguments fingerprint；
- subject/tenant/resource；
- policy/approval version 与 expiry；
- handler attempt/effect id；
- result trust/provenance；
- continuation interaction id。

`requires_action` 不授权工具；`function_result` 不证明 effect；provider 生成最终文本也不证明业务成功。

### server-side tools 与 client-side tools

server-side search/code 等工具由 provider 执行，客户端工具由你的 runtime 执行。两者都要保留 call/result step，但责任边界不同：

- provider tool：核对来源、引用、数据政策、区域与 usage；
- client tool：你负责授权、secret、network egress、sandbox、幂等和审计；
- 混合工具：防止 provider result 与本地 result 的 id/信任级别混淆。

## Thought/signature 是高风险 opaque artifact

官方指南要求某些 stateless continuation 原样保留 thought/tool steps 与 signatures。安全设计应把它们视为：

- provider-originated；
- 会话/模型/API/version context-bound；
- 可能含敏感或不可公开内容；
- 不供业务逻辑解释；
- 不跨用户/租户/模型重放；
- 不进入普通 analytics、前端或简历 artifact。

存储最小化建议：

- 原始密文或受控 blob store；
- 会话与请求 identity；
- allowlisted metadata projection；
- access log 与短 TTL；
- export/redaction policy；
- 重放前 context-binding 校验；
- 删除传播与 incident playbook。
