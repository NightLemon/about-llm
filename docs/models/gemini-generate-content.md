# Gemini `generateContent`：Content、Parts 与多模态

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：维护 `generateContent` adapter，或需要处理图片、音频、视频与文件输入的工程师。
- **先修**：先读 [Gemini 总览](gemini.md)，并理解 HTTP、JSON 与流式响应。
- **首次阅读**：固定请求 → 响应映射 → SSE 分层 → Parts 类型 → 多模态反事实评测。
- **完成信号**：能解释纯文本 parser 为什么要拒绝多 candidate 或非文本 part，并为一种目标模态设计反事实 case。
- **卡住时**：先只处理一条 text part，再逐项加入 candidate、finish reason 和媒体 part。

</div>

**章节导航**：[总览](gemini.md) · [Interactions API](gemini-interactions.md) · [生产接入](gemini-production.md) · [证据台账](../evidence/gemini-controls.md)
{ .doc-nav }

`generateContent` 以一次请求和响应为中心。本章先跟踪一条仓库里可以运行的纯文本样例，
再把它扩展到多候选、流式传输、多模态 Parts 和工具调用。

Interactions API 采用另一套资源、状态和流事件。本章只在需要比较协议时提到它，
生产代码也应为两种 API 使用不同 adapter。

## 先跟一次固定的纯文本请求

从仓库根目录运行：

```powershell
python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl `
  --output artifacts/cloud-api/gemini-page-contracts.json
```

命令会同时核对三种云 API 的教学样例。在输出中找到 `case_id=gemini-generate-content-text`。
这一例从三条内部消息开始：

| 内部角色 | 内容 |
|---|---|
| system | `Be concise.` |
| user | `What is RAG?` |
| assistant | `I will answer.` |

发送给 Gemini API 的请求是：

```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "What is RAG?"}]},
    {"role": "model", "parts": [{"text": "I will answer."}]}
  ],
  "generationConfig": {"maxOutputTokens": 32, "temperature": 0.0},
  "systemInstruction": {"parts": [{"text": "Be concise."}]}
}
```

这里发生了两个容易漏掉的映射：system message 变成顶层 `systemInstruction`，assistant role 变成
Gemini 的 `model` role。模型路径则是 `/v1beta/models/gemini-example:generateContent`。

固定响应如下：

```json
{
  "modelVersion": "gemini-example-001",
  "candidates": [{
    "content": {"parts": [{"text": "part one"}, {"text": " part two"}]},
    "finishReason": "STOP"
  }],
  "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4}
}
```

纯文本 parser 得到：

| 本地字段 | 值 | 来自哪里 |
|---|---|---|
| `text` | `part one part two` | 按顺序连接唯一 candidate 的两个 text part |
| `model` | `gemini-example-001` | `modelVersion` |
| `input_tokens` | 3 | `usageMetadata.promptTokenCount` |
| `output_tokens` | 4 | `usageMetadata.candidatesTokenCount` |
| `finish_reason` | `STOP` | candidate 的 `finishReason` |

这个命令不访问网络，也不运行 Gemini 模型。它验证的是固定 JSON 在 request builder 与 response parser
之间怎样映射，不能证明真实端点、凭证、模型质量或计费。

## 再把样例放回完整对象图

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

固定样例只走了对象图中的一条窄路。完整 adapter 还要区分三类结果：

- 单个或多个 candidate；
- candidate 中的不同 part 类型；
- prompt 被阻止后返回 `promptFeedback`，且 candidates 可能为空。

第三种情况应成为明确的 blocked 终态，而不是成功的空字符串。

### 当前仓库支持到哪里

| 边界 | 当前实现 | 超出边界时怎样处理 |
|---|---|---|
| Request | system text、user/model text、`maxOutputTokens`、temperature | 多模态、工具、安全与结构化输出尚未建模 |
| Response | 恰好一个 candidate；其中每个 part 都是非空 text | 零/多 candidate、非 text part、混合 text/function part 会报错 |
| Stream | 每个 chunk 最多一个 candidate，只接收 text part | 缺失或重复终态、非文本 part、candidate index 非零会报错 |
| Interactions | 未实现 | 使用独立 adapter 和状态机 |

这种限制是有意的。例如响应同时包含说明文字和 `functionCall` 时，只返回文字会让调用方看不到待执行动作；
多个 candidate 只保留第一个，也会丢失选择语义。完整字段清单与测试边界见
[证据台账](../evidence/gemini-controls.md#request-builder)。

## `streamGenerateContent` 不是 Interactions stream

两种 API 都可能通过 SSE 传输，但 JSON payload 不是同一种事件：

| 比较项 | `streamGenerateContent` | Interactions stream |
|---|---|---|
| 主要对象 | candidate、content part | Interaction、step |
| 增量 | text 或其他 part | typed step delta |
| 结束依据 | 候选路径读取 `finishReason`，传输仍以 EOF 结束 | Interaction/step 的状态与 terminal event |
| 当前仓库 | 单 candidate 的 text-only 状态机 | 未实现 |

仓库的固定流式用例在一个 payload 中收到 `text=hello`、`finishReason=STOP` 和 token usage。
状态机依次产出 `text → finish → usage`；随后只有传输 EOF 才产出 `transport_end`。

底层 UTF-8、SSE 行和 JSON 解码可以复用。上层仍要分别实现字段类型、顺序约束和终态，
不能让同一个 provider state machine 猜当前是哪套 API。

具体支持范围见 [Gemini 接入证据台账](../evidence/gemini-controls.md)。

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

单个“看图答对”样本无法区分模型是在看图，还是仅凭文字与常识猜中。可以为每个样本建立 paired counterfactual：

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

`generateContent` 把结构化输出设置放在 generation config 一侧，Interactions 使用自己的 response format。
两者都只能约束各自支持的 schema 子集。成功解析首先说明形状符合约束，还没有验证内容是否真实、获授权或可执行。

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
  → accumulate arguments and validate JSON/schema
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

工具在哪里执行，决定了谁持有凭证、谁能观察副作用，以及谁负责恢复：

| 执行位置 | 应用主要负责什么 |
|---|---|
| Provider 侧的搜索或代码工具 | 核对来源、引用、数据政策、区域和 usage |
| 本地 runtime 调用的工具 | 授权、密钥、网络出口、沙箱、幂等和审计 |
| Provider 与本地混合 | 绑定 call ID，并区分两侧结果的来源和信任级别 |

无论在哪一侧执行，都要保留 proposal、call/result identity 和最终业务验证，不能只保存模型生成的总结文本。

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
