# 云模型 API 契约：先保留差异，再统一业务

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：第一次设计多供应商模型 adapter、网关或 Agent 接入层的开发者。
- **先修**：HTTP/JSON、基本流式传输和工具调用概念。
- **首次阅读**：三层协议 → typed response → streaming → tool proposal → 最小实验。
- **完成信号**：能解释 canonical core 应统一什么，以及哪些 provider 语义必须保留。
- **卡住时**：先实现一次非流式 text request，不要从重试、工具和多供应商同时开始。

</div>

**学习入口**：[模型选型](landscape.md) · [可靠性进阶](cloud-api-reliability.md) · [Cloud API 项目](../practice/projects/cloud-api-contracts.md) · [证据台账](../evidence/cloud-api-controls.md)
{ .doc-nav }

“所有大模型 API 都有 messages 和 text”是一个危险的近似。供应商可能都能完成对话，却使用不同的 system 位置、content graph、tool objects、stream events、usage 和 terminal。

正确的统一方式不是把差异抹平，而是在业务层统一稳定概念，在 provider adapter 中忠实保留 wire semantics。

## 一次调用跨过三层

~~~mermaid
flowchart LR
    A["Canonical task request"] --> B["Provider adapter"]
    B --> C["Provider wire request"]
    C --> D["HTTP / SSE transport"]
    D --> E["Provider service"]
    E --> F["Typed response/events"]
    F --> G["Task projection"]
    G --> H["Policy and quality gate"]
~~~

三层各自回答不同问题：

| 层 | 负责什么 | 不负责什么 |
|---|---|---|
| Canonical business | 任务、用户、预算、稳定业务类型 | 假装所有 provider 能力相同 |
| Provider adapter | request/response/block/terminal/usage 映射 | 工具权限和业务真值 |
| Transport | origin、deadline、bytes、SSE framing、取消 | 猜测 provider 语义 |

最危险的 adapter 不是代码重复，而是返回一个看似统一的字符串，同时丢掉 refusal、tool call、citation、usage 或 incomplete 状态。

## Messages 不是跨供应商标准

下面只展示对象图差异，不把字段表当作永久 API 事实：

~~~text
Chat-like API
request.messages[]
response.choices[] or another surface-specific graph

Anthropic Messages
top-level system
messages[].content blocks
response.content blocks

Gemini generateContent
systemInstruction
contents[].parts
candidates[].content.parts
~~~

即使某个 endpoint 宣称 OpenAI-compatible，也只表示已经验证的字段子集兼容。Model identity、developer/system roles、tools、structured output、stream terminal、usage、错误和计费仍需逐项测试。

同一供应商也可能同时提供多套 API surface。以 OpenAI 为例，Chat Completions 与 Responses 拥有不同对象图；Responses streaming 使用 typed semantic events，而不是只有一个 text delta。

因此 capability 必须绑定 provider、endpoint、model、API revision 与核对日期，不能只保存 supports_tools=true。

## Canonical core 应该很小

一个稳定业务 request 可以只表达：

~~~text
CanonicalRequest
├── provider and API surface
├── model identity
├── ordered messages/content
├── maximum output budget
├── tool proposals allowed by this task
└── validated provider options
~~~

Canonical result 至少保留：

~~~text
CanonicalResult
├── provider/model/request identity
├── typed output items
├── terminal state and reason
├── usage
└── provider receipt
~~~

Provider options 不能成为任意 JSON 透传垃圾桶。每个 adapter 应 closed-schema 校验允许字段，并把 schema revision 纳入 request identity。

未知字段默认失败或进入受控 extension，避免拼写错误和 SDK 行为变化静默进入生产。

## Response 不是一个字符串

生产 response 要分成四个独立维度：

1. **Identity**：provider、request/response ID、model 和 API version。
2. **Typed output**：text、refusal、function call、citation、media 或 opaque item。
3. **Terminal**：completed、stopped、incomplete 或 failed，以及原因。
4. **Usage**：provider 实际返回的 input、output 和其他分类。

它们不能互相推导。拿到一段 text 不表示 stream 正常结束；provider completed 不表示业务任务成功；usage 也不等于发票已经核对。

### 为什么 typed items 很重要

内部至少区分：

~~~text
TextItem
RefusalItem
FunctionCallItem
FunctionResultItem
CitationItem
MediaItem
OpaqueProviderItem
~~~

业务只支持 text 时，遇到 tool/refusal/unknown item 应显式失败或分流，不能丢掉非文本部分后返回“成功”。

Opaque item 只表示系统保留了无法解释的 provider state。它不表示应用可以展示、修改、跨会话重放或把它当作授权凭证。

## Structured output 只解决结构

JSON mode 或 schema-constrained output 可以降低解析失败，但不能保证：

- 字段值真实；
- 引用确实存在；
- 金额和单位合理；
- 用户拥有目标资源；
- 工具动作已经授权；
- 副作用已经完成。

可靠链路是：

~~~text
provider terminal
→ refusal/incomplete/error split
→ validate JSON and schema
→ domain invariants
→ identity and ACL
→ budget / approval / idempotency
→ handler
→ effect verifier
~~~

Schema validation 是必要的一层，不是业务验证的终点。

## Tool call 只是 proposal

模型生成的 tool name、arguments 和 call ID 是候选动作。Provider adapter 负责保真解析；Agent runtime 才负责：

- schema 与 domain validation；
- subject/resource/tool authorization；
- budget 和 approval；
- idempotency 与 duplicate conflict；
- isolated execution；
- effect verification 与 reconciliation。

合法 JSON 不是权限。SDK 的自动 tool loop 也不能绕过业务 policy。

外部网页、RAG 文档和 tool result 都属于低信任数据，不能因为进入模型上下文就获得 system 指令权限。

## Streaming 有三层状态

网络 chunk、SSE event 和 provider event 不是一回事：

~~~text
arbitrary byte chunks
→ SSE framing
→ provider typed events
→ canonical updates
→ task projection
~~~

Byte decoder 负责 UTF-8、换行、多行 data、空行终止和资源上限。Provider state machine 负责 item/block index、delta 次序、usage 与 terminal。

### Terminal 不能统一成一个字符串

不同 API surface 可能用独立 done marker、message stop、finish reason + EOF，或 typed completed/failed event。Adapter 必须按目标协议建状态机。

至少区分：

| 状态 | 含义 |
|---|---|
| content/item done | 某个结构单元闭合 |
| model stop reason | 模型为何停止 |
| provider terminal | 本次 provider response 正常结束 |
| transport EOF | 底层字节流关闭 |
| application stop | 客户端决定不再展示更多文本 |

EOF 缺少协议要求的 terminal 时是截断。客户端命中 stop string 可以截断展示，却不能伪造 provider finish reason、修改 usage 或证明服务端停止计费。

OpenAI 官方文档说明 Responses API 的 HTTP streaming 使用 SSE 和 typed semantic events；常见事件包括 response.created、text delta、response.completed 与 error。完整类型仍以目标版本官方 reference 为准。

## 一条可学习的接入路线

不要一开始就构建“统一大模型网关”。按能力逐层增加：

### Level 1：单供应商非流式 text

固定 endpoint、model、API revision 和 output cap。保存 request identity、typed response、terminal、usage 和 request ID。

验收：非 text response 不会被错误投影为空字符串或 success。

### Level 2：离线错误与类型回放

为多 text、refusal、tool-only、unknown item、invalid JSON、missing usage 和 incomplete response 分别准备
固定样例。

验收：每个输入得到明确 typed state，日志不包含 secret 或 raw user body。

### Level 3：流式状态机

随机切分 bytes，测试 Unicode 边界、错序/重复 event、unknown type、缺 terminal 和超限。

验收：chunk 划分变化不改变最终 typed result；截断永远不会变成 completed。

### Level 4：第二个 provider

只统一已经稳定的 canonical concepts。把新的 content graph、terminal、usage 和 capability 保留为 adapter extension。

验收：两个 provider 的差异能在测试和 artifact 中被看见，而不是被字符串接口吞掉。

### Level 5：可靠性与预算

再加入 deadline、retry decision、per-attempt reservation、outcome uncertain、billing reconciliation 和真实 smoke test。

这一层进入[云 API 可靠性进阶](cloud-api-reliability.md)。

## 选择正确的评测单位

一次 provider response 成功解析，只是 protocol success。完整链路应区分：

~~~text
request attempt
→ provider terminal
→ parsed task result
→ policy decision
→ verified task success
~~~

分别报告 attempted、provider-completed、parsed、admitted 和 verified-success 分母。否则廉价解析失败或过度拒答可能让“成功请求平均成本”看起来更好。

多 provider 比较要固定相同 cases、输入语义、output budget 和 scorer，并披露能力差异导致的不可比部分。

## 常见错误

- 把 OpenAI-compatible 写成所有 API 语义兼容。
- 把 response 压成一个 text 字符串，丢掉 typed items。
- 把 JSON schema 通过当成事实正确和工具授权。
- 把 network chunk、SSE event、provider event 和 token 混为一谈。
- 只看到 EOF 就把 stream 标成 completed。
- 用 application stop string 伪造 provider terminal。
- 让 provider options 接受任意未知字段。
- 把离线样例写成真实账号、网络、usage 或 billing 已经验证。

## 面试时怎样回答

面对“如何设计统一大模型 API”，可以按四层回答：

1. Canonical core 只统一任务、identity、typed output、terminal 和 usage。
2. Provider adapter 保留 request/response object graph 和 capability 差异。
3. Transport 与 provider state machine 分开处理 bytes 和语义。
4. Policy/verifier 再负责权限、工具 effect 和业务成功。

继续追问时，应能解释为什么 typed extension 优于 lowest-common-denominator string，以及为什么 provider completed 不等于 task success。

## 自测

1. 同样叫 messages，为什么不能推断两个 provider 的 role/content 语义相同？
2. 一个 tool-only response 应怎样进入 canonical result？
3. Structured output 之后还需要哪些业务 gate？
4. 为什么 transport EOF 不能单独证明 stream 成功？
5. 加入第二个 provider 时，哪些概念适合统一，哪些应保留 extension？

## 继续学习

- [云 API 可靠性进阶](cloud-api-reliability.md)：retry、deadline、取消、预算和真实 smoke。
- [Claude](claude.md)：Messages blocks 与工具状态机。
- [Gemini](gemini.md)：多套 API surface 的对象辨析。
- [GPT](gpt.md)：Responses typed outputs 与评测。
- [Cloud API 项目](../practice/projects/cloud-api-contracts.md)：离线 adapter 与状态机验证程序。
- [云 API 证据台账](../evidence/cloud-api-controls.md)：具体字段、核对日期、固定样例与命令。

## 官方文档

- OpenAI，[Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)，核对日期 2026-08-17。
- 其他 provider 的带日期来源与本仓库证据范围见[证据台账](../evidence/cloud-api-controls.md)。
