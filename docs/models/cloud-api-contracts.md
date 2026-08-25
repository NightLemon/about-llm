# 云模型 API 契约：先保留差异，再统一业务

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：第一次设计多供应商模型适配器、网关或 Agent 接入层的开发者。
- **先修**：HTTP/JSON、基本流式传输和工具调用概念。
- **首次阅读**：先跟完固定请求，再看三层协议、结构化响应、流式事件和工具提议。
- **完成信号**：能解释业务层应统一什么，以及哪些供应商语义必须保留。
- **卡住时**：先实现一次非流式 text request，不要从重试、工具和多供应商同时开始。

</div>

**学习入口**：[模型选型](landscape.md) · [可靠性进阶](cloud-api-reliability.md) · [Cloud API 项目](../practice/projects/cloud-api-contracts.md) · [证据台账](../evidence/cloud-api-controls.md)
{ .doc-nav }

许多大模型 API 都能接收对话并返回文字，但这不等于它们使用同一套协议。
系统指令放在哪里、返回内容怎样分块、工具调用怎样表示、流式请求怎样结束，都可能不同。

合理的适配层只统一业务真正共享的概念，并在供应商适配器中保留网络协议的原始差异。

## 先跟一条回答穿过适配层 { #worked-request }

仓库准备了三个离线样例，都向模型提问 `What is RAG?`。先从仓库根目录运行：

```powershell
python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl `
  --output artifacts/cloud-api/contracts.json
```

所有域名都以 `.invalid` 结尾，响应也保存在本地文件中，因此这条命令不会访问模型服务或产生费用。
先只看 `openai-compatible-text` 这一条：

| 经过的位置 | 这条样例中能看到什么 |
|---|---|
| 业务请求 | 使用 `model-a`，先发一条 `Be concise.` 系统消息，再提问 `What is RAG?` |
| 供应商请求 | 适配器把两条消息放入该接口的 `messages` 数组，并设置最多生成 32 个 token |
| 固定响应 | 文本位于 `choices[0].message.content`，结束原因是 `stop`，用量是输入 4、输出 3 个 token |
| 共同结果 | `text`、实际模型名、输入与输出用量、结束原因分别进入明确字段 |

这里最重要的不是字段路径，而是信息没有被压成一句字符串。业务拿到了回答文本，同时仍能知道模型身份、
为什么结束，以及服务返回了多少用量。

另外两条样例仍然问同一个问题，但 Anthropic Messages 把系统指令放在顶层，Gemini `generateContent` 使用
`systemInstruction` 和 `parts`。固定答案本身不同，因为这些是仓库准备的协议样例；本实验只比较结构怎样映射，
不比较模型质量。

第一次阅读到这里，可以先回答两个问题：如果只返回文本，哪些结束和计费信息会丢失？
新增一个供应商时，哪些字段属于业务共同概念，哪些路径只能留在它自己的适配器里？

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
| 业务层 | 任务、用户、预算和稳定的业务类型 | 假设所有供应商能力相同 |
| 供应商适配器 | 映射请求、响应块、结束状态和用量 | 判断工具权限和业务事实 |
| 网络传输层 | 目标地址、截止时间、字节、SSE 分帧和取消 | 猜测供应商事件的业务含义 |

适配器最危险的错误不是出现重复代码，而是只返回一个看似统一的字符串。
这样会悄悄丢掉拒答、工具调用、引用、用量或响应未完成等状态。

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

即使某个接口宣称兼容 OpenAI，也只能相信已经实际验证过的字段。
模型身份、消息角色、工具、结构化输出、流式结束信号、用量、错误和计费方式，都要按目标接口逐项测试。

同一家供应商也可能提供多套接口。以 OpenAI 为例，Chat Completions 与 Responses 使用不同的请求和响应对象。
Responses 的流式输出由带类型的语义事件组成，并不只是连续追加文本。

因此，一条“支持工具调用”的能力记录必须同时写明供应商、具体接口、模型、API 版本和核对日期。
单独保存 `supports_tools=true` 无法说明它适用于哪个组合。

## 业务共同对象应该很小 { #canonical-core }

业务请求只需要表达稳定、确实共享的内容：

~~~text
CanonicalRequest
├── provider and API surface
├── model identity
├── ordered messages/content
├── maximum output budget
├── tool proposals allowed by this task
└── validated provider options
~~~

共同结果至少保留：

~~~text
CanonicalResult
├── provider/model/request identity
├── typed output items
├── terminal state and reason
├── usage
└── provider receipt
~~~

供应商专用选项不能成为任意 JSON 的透传入口。每个适配器都应明确列出允许字段，并拒绝未知字段。
这套字段规则的版本也要写入请求身份，便于重放时知道当时采用了什么约束。

确实需要保留的新字段可以进入有类型、有版本的扩展对象。普通未知字段默认失败，
避免拼写错误或 SDK 行为变化悄悄进入生产。

## Response 不是一个字符串

生产响应要把四类信息分开保存：

1. **身份**：供应商、请求与响应 ID、实际模型和 API 版本；
2. **带类型的输出**：文本、拒答、函数调用、引用、媒体或暂时无法解释的对象；
3. **结束状态**：已完成、已停止、未完成或失败，以及具体原因；
4. **用量**：供应商实际返回的输入、输出和其他用量分类。

这四类信息各自回答一个问题：文本说明目前收到了什么内容；结束状态说明协议是否正常收尾；
业务验证器判断用户任务是否成功；用量字段记录供应商在响应中报告的数字。最终账单仍要另行核对。

### 为什么输出项必须带类型 { #typed-items }

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

如果业务当前只支持文本，遇到工具调用、拒答或未知输出项时，应明确失败或转交其他流程。
删除非文本部分后再返回“成功”，会改变供应商响应的真实含义。

不透明输出项（opaque item）表示系统完整保留了暂时无法解释的供应商状态。
默认处理方式是受限保存。展示、修改或跨会话重放都需要该接口的明确规则；授权仍由独立的身份与权限记录提供。

## 结构化输出只解决格式 { #structured-output }

JSON mode 或受 Schema 约束的生成，可以减少格式解析失败。但格式正确以后，仍要检查：

- 字段值是否有事实依据；
- 引用是否真的存在并支持对应结论；
- 金额和单位是否符合业务规则；
- 用户是否有权访问目标资源；
- 工具动作是否获得授权；
- 副作用是否真正完成。

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

Schema 校验只是其中一层，不是业务验证的终点。

## 工具调用只是候选动作 { #tool-call-proposal }

模型返回的工具名、参数和调用 ID 只是候选动作。供应商适配器负责完整解析这些字段；
Agent 运行时再决定它是否可以执行：

- 参数是否符合格式和业务规则；
- 当前用户能否访问资源并调用该工具；
- 预算是否足够，以及是否需要审批；
- 重复调用会命中幂等结果，还是构成冲突；
- 工具在哪个隔离环境中运行；
- 业务效果如何验证，未知结果怎样对账。

合法 JSON 不等于已经获得权限。SDK 提供的自动工具循环也不能绕过业务规则。

外部网页、RAG 文档和工具结果都属于低信任数据。它们进入模型上下文以后，仍然不能获得系统指令的权限。

## Streaming 有三层状态

一次网络读取、一个 SSE 事件和一个供应商事件不是同一回事：

~~~text
arbitrary byte chunks
→ SSE framing
→ provider typed events
→ canonical updates
→ task projection
~~~

字节解码器先处理 UTF-8 边界、换行、多行 `data`、事件间空行和资源上限。
在它之上，供应商状态机再检查输出项与内容块编号、增量顺序、用量和结束事件。

### 结束状态不能统一成一个字符串 { #terminal }

不同接口可能使用独立的完成标记、消息停止事件、结束原因加 EOF，或者带类型的完成与失败事件。
适配器必须按照目标协议建立状态机，不能只查找某个通用字符串。

至少区分：

| 状态 | 含义 |
|---|---|
| 内容块或输出项结束 | 某个结构单元已经闭合 |
| 模型停止原因 | 模型为什么停止生成 |
| 供应商结束事件 | 本次供应商响应按照协议正常结束 |
| 网络 EOF | 底层字节流已经关闭 |
| 应用停止 | 客户端决定不再展示更多文本 |

如果字节流已经关闭，却没有收到协议要求的结束事件，这次响应就是截断。
客户端命中停止字符串只改变本地展示。供应商结束原因和用量继续保留原值；服务端何时停止计算与计费，
要由接口状态或账单记录确认。

以 OpenAI Responses API 为例，HTTP 流式响应使用 SSE，并通过事件类型说明当前发生了什么。
`response.created` 表示响应已创建，文本增量事件携带新增内容，`response.completed` 表示协议正常完成，
`error` 则报告错误。具体事件集合仍要以目标 API 版本的官方文档为准。

## 一条可学习的接入路线

不要一开始就构建“统一大模型网关”。按能力逐层增加：

### Level 1：单供应商非流式 text

先固定一个接口、一个模型和一个 API 版本，并限制最大输出 token。
保存请求身份、带类型的响应、结束状态、用量和供应商请求 ID。

验收时准备一个非文本响应，确认系统不会把它错误地投影为空字符串或成功结果。

### Level 2：离线错误与类型回放

分别准备固定样例：多个文本块、拒答、只有工具调用、未知输出项、无效 JSON、缺少用量，以及未完成响应。

验收时，每个输入都要得到明确状态；日志中不能出现密钥或完整用户请求正文。

### Level 3：流式状态机

把同一字节流随机切成不同大小的网络片段，覆盖 Unicode 字符被切开、事件错序或重复、未知类型、缺少结束事件和超限。

验收时，网络怎样切片都不应改变最终结构化结果；截断也绝不能被标记为完成。

### Level 4：第二个 provider

只把已经验证为共同语义的字段加入业务对象。新的内容结构、结束状态、用量和能力差异，
先保留在该供应商有类型的扩展对象中。

验收时，两个供应商的差异要能在测试和运行记录中被看见，而不是被字符串接口吞掉。

### Level 5：可靠性与预算

最后再加入整次调用的截止时间、重试决策和逐次费用预留。
网络结果不确定时进入对账，真实账号只运行受控的冒烟测试，并与账单记录核对。

这一层进入[云 API 可靠性进阶](cloud-api-reliability.md)。

## 选择正确的评测单位

一次供应商响应成功解析，只能说明协议处理成功。完整链路还要区分：

~~~text
request attempt
→ provider terminal
→ parsed task result
→ policy decision
→ verified task success
~~~

分别报告尝试发送、供应商完成、成功解析、策略允许和业务验证成功的数量。
如果只计算最终成功请求，解析失败和过度拒答会从分母消失，反而让“平均成功成本”看起来更好。

比较多个供应商时，要固定相同的 case、输入语义、输出预算和评分方法。
某个接口独有能力造成的不可比部分，应单独报告。

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

1. 业务共同对象只统一任务、身份、带类型的输出、结束状态和用量；
2. 供应商适配器保留各自的请求、响应结构和能力差异；
3. 网络传输层处理字节和 SSE 分帧，供应商状态机处理事件语义；
4. 权限规则和业务验证器再判断工具效果与任务是否成功。

继续追问时，要说明为什么有类型的供应商扩展优于“所有接口都只返回一个字符串”。
还要能解释：供应商响应完成，只是协议终态，不等于用户任务已经成功。

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
