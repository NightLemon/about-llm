# Claude：把闭源模型接成一个可靠系统

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：要通过 Claude API 构建对话、长上下文、RAG 或 Agent 系统的开发者。
- **先修**：理解 HTTP/JSON、SSE、工具调用和基础模型评测。
- **首次阅读**：先跟一次工单查询走完全程，再理解 Messages、流式事件、工具、预算与迁移。
- **完成信号**：能从一个请求追到最终结算，并指出哪些步骤属于 Claude API、适配器、业务权限和外部工具。
- **卡住时**：先读[云 API 契约](cloud-api-contracts.md)，用一个非流式 text request 建立最小基线。

</div>

**模型导航**：[云 API 契约](cloud-api-contracts.md) · [Agent 总览](../applications/agents.md) · [评测项目](../practice/projects/evaluation-gate.md) · [Claude 证据台账](../evidence/claude-controls.md)
{ .doc-nav }

学习 Claude 的重点不是记忆某一代模型的参数，而是学会连接一个持续变化的闭源产品和 API 契约。

你可以观察请求、响应、usage、错误和任务效果，却不能查看服务端权重。可靠工程因此依赖“保留类型、固定版本、逐层验证”，而不是猜测内部架构。

## 先划清已知与未知

Claude 的公开论文、产品页面和真实 API 运行是三类不同证据：

| 来源 | 可以回答 | 不能直接回答 |
|---|---|---|
| Constitutional AI、RLAIF 等论文 | 特定实验如何组织原则与反馈 | 当前模型的完整训练配方 |
| 带核对日期的 API 文档 | 当时公开的 request/response contract | 未来版本、你的账号与区域行为 |
| 真实受控请求 | 某 model ID 在该请求上的实际结果 | 整体质量、生产 SLO 或内部架构 |

参数量、层数、稠密/稀疏结构、训练数据和未公开的后训练细节都应保持 unknown。不要从输出风格、产品名称或旧论文反推。

具体 model ID、上下文、价格、限额和 beta feature 都会变化。本页讲相对稳定的接入方法；需要核对某个日期的
字段和本仓库实际运行过哪些检查时，再查阅[证据台账](../evidence/claude-controls.md)。

## 研究路线提供什么直觉

Constitutional AI 的学习价值，是把自然语言原则放入数据生成和偏好形成：模型按原则批评、修订回答，再用人类或 AI 反馈构造训练信号。

这让“希望模型遵守什么”变得更显式，但 constitution 不是生产权限系统。原则可能含糊、冲突或遗漏，模型也可能错误应用。

RLAIF 用 AI feedback 扩展部分偏好标注。人仍需选择原则、审计 evaluator、定义不可接受风险并决定发布门槛；共享偏差还可能让自动评审放大盲区。

因此研究方法能解释一种训练思路，不能替代应用侧的 ACL、数据隔离、工具审批、审计和人工升级。

## 先跟一次工单查询走完全程 {#ticket-request}

假设用户提出一个具体任务：

> 查询工单 `T-1042` 的当前状态，并告诉我下一步该做什么。

应用允许 Claude 使用只读工具 `get_ticket`，但模型本身没有数据库权限。一次完整任务可以经过两轮 Messages 调用：

```text
用户请求
→ 应用固定租户、用户、模型版本、预算和可用工具
→ Anthropic adapter 生成顶层 system + messages + tools
→ Claude 返回 tool_use：get_ticket({"ticket_id":"T-1042"})
→ 应用等待 block 完整闭合
→ Schema 校验 + 租户 ACL + 预算检查
→ 应用执行只读查询，得到受控 tool_result
→ 第二轮 Messages 调用携带工具结果
→ Claude 返回面向用户的文字说明
→ 检查 message terminal、usage 和任务结果
→ 结算两次网络 attempt
```

假设工具返回“状态：等待用户补充凭证；下一步：上传付款截图”。第二轮回答就应忠实说明这两个字段。
任务验收程序还要确认回答没有混入其他租户的工单，也没有声称工单已经处理完成。

这条链路里有三个容易混淆的“成功”：

| 阶段 | 成功表示什么 | 还没有证明什么 |
|---|---|---|
| Claude 返回 `tool_use` block | 模型给出了结构完整的工具提议 | 用户有权限、参数真实、工具已经执行 |
| `get_ticket` 返回结果 | 外部查询已经完成 | 最终说明准确、请求整体成功 |
| 第二轮正常结束 | 收到了完整 Messages 响应和 usage | 业务验收、安全检查和引用核验已通过 |

### 模型、适配器和依赖库各做什么 {#dependency-stack}

把一次请求拆开后，各层职责会清楚很多：

| 层 | 主要职责 | 不应交给它的职责 |
|---|---|---|
| 业务应用 | 身份、租户、任务、权限、预算和成功标准 | 猜测 Provider 的 JSON 细节 |
| Anthropic SDK 或 HTTP 客户端 | 认证、连接、超时和字节传输 | 决定业务授权或工具副作用是否可重放 |
| Provider adapter | 顶层 `system`、Messages blocks、headers、stop 和 usage 映射 | 把未知 block 静默压成字符串 |
| SSE decoder | 把任意 byte chunk 还原成完整事件 | 判断某个工具是否获权 |
| Messages 状态机 | 校验 message、block、delta 和 terminal 的次序 | 执行工具或判断业务任务成功 |
| Schema、ACL 与审批层 | 校验参数并决定能否执行 | 相信模型提议天然安全 |
| 工具运行时与账本 | 幂等执行、记录副作用、费用和待对账结果 | 用本地成功替代供应商账单证据 |

仓库目前可以离线运行文本 request/response adapter、文本流状态机、通用 HTTP 重试和预算账本。
上面完整的 `tool_use → tool_result` Claude 链路仍是接入设计：仓库没有用真实 Anthropic 账号执行它，
现有 `AnthropicTextStream` 也会明确拒绝非文本 block。

## Messages 不是一问一答字符串

在工单例子里，系统指令位于请求顶层，对话历史放在 `messages` 中。每条消息的内容又可以由多个有类型的 block 组成。

响应也不是单个字符串。它可以同时带有说明文字、工具提议、停止原因和用量。

~~~text
Message request
├── model / max_tokens
├── system
└── messages[]
    └── role + content blocks

Message response
├── id / model
├── content blocks[]
├── stop reason
└── usage
~~~

第一轮请求的形状可以是：

~~~json
{
  "model": "<reviewed-model-id>",
  "max_tokens": 256,
  "system": "你是工单助手。只能使用授权工具读取当前租户的数据。",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "查询工单 T-1042 的状态，并说明下一步。"}
      ]
    }
  ],
  "tools": [
    {
      "name": "get_ticket",
      "description": "读取一张工单的状态",
      "input_schema": {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}},
        "required": ["ticket_id"],
        "additionalProperties": false
      }
    }
  ]
}
~~~

这个示例用占位 model ID，运行前仍要按目标 API 版本核对字段。它要表达的稳定关系是：`system` 位于顶层，
`messages` 保存对话历史，`content` 和工具结果都应保留类型。

Claude 可能在第一轮返回文字和工具提议。下面只截取 `content` 与停止原因，省略响应 ID、model 和 usage：

~~~json
{
  "content": [
    {"type": "text", "text": "我先读取工单状态。"},
    {
      "type": "tool_use",
      "id": "toolu_example",
      "name": "get_ticket",
      "input": {"ticket_id": "T-1042"}
    }
  ],
  "stop_reason": "tool_use"
}
~~~

生产代码需要保留完整响应，并使用其中实际返回的 usage。应用完成授权和查询后，第二轮消息要带回与
`toolu_example` 对应的工具结果，模型才有证据生成最终说明。

### 为什么要保留 typed blocks

如果代码只读取第一个文本 block，工单例子中的 `tool_use` 就会消失。多个文本块、引用、拒答和未知 block 也会遇到同样问题。

更稳的内部对象是：

~~~text
ProviderResponse
├── provider / model / request id
├── blocks[]
│   ├── text
│   ├── tool proposal
│   └── controlled provider-specific block
├── stop
├── usage
└── raw response identity
~~~

规范化层先保留 provider 语义，再按任务显式投影：

| Provider response | 任务层状态 |
|---|---|
| 一个或多个 text blocks | text result |
| 只有 tool proposal | tool_proposed，不是空字符串 |
| text + tool proposal | mixed typed result |
| refusal/safety outcome | typed refusal |
| unknown block | 停止处理，或交给明确注册的 extension |
| terminal 缺失 | incomplete/error |

“response 可以解析”只证明协议层过关，不证明内容正确、引用有效、工具获权或任务成功。

## 把业务模型与 provider wire 分开

应用不要让领域代码到处读取 Anthropic JSON。使用清晰的 adapter 边界：

~~~text
Canonical task request
→ Anthropic request adapter
→ HTTP/SSE transport
→ lossless provider response
→ task-specific projection
→ policy and quality gates
~~~

业务层表达用户、租户、任务、预算和候选工具。Provider adapter 只负责把这些信息映射成顶层 `system`、
内容 block、请求头、用量和停止原因。

这样更换模型或供应商时，主要变化集中在协议映射，领域代码不必到处读取某个供应商的 JSON 字段。
供应商特有能力仍应作为有类型的扩展保留，不能为了统一 schema 把所有对象压成字符串。

## Streaming 是两层状态机

网络每次读到的只是任意长度的字节片段。一个片段不保证恰好对应模型 token、JSON 对象或 SSE 事件。

因此，流式处理至少分两层：

1. **字节与 SSE 分帧**：处理 UTF-8、换行、事件字段、空行终止与资源上限。
2. **Messages 状态机**：检查消息、block、增量、用量、停止和错误事件的合法次序。

~~~mermaid
flowchart LR
    A["byte chunks"] --> B["SSE events"]
    B --> C["message start"]
    C --> D["block start"]
    D --> E["typed deltas"]
    E --> F["block stop"]
    F --> G["message stop + usage"]
    G --> H["task projection"]
~~~

工单工具参数可能分成多个增量到达。即使某个 JSON 前缀碰巧能解析，也不能把它当作完整参数。
只有 block 正常闭合，并且 schema 与业务语义都通过检查后，提议才可以进入授权流程。

### 四种结束不能混在一起

| 层级 | 结束意味着什么 |
|---|---|
| content block stop | 某个 block 结构闭合 |
| model stop reason | 模型为什么停止 |
| provider message terminal | 消息协议正常结束 |
| transport EOF/close | 字节流结束 |

如果连接到达 EOF 时仍未出现 Provider 的消息终态，这次流就是截断。文本 block 结束只代表该 block 闭合；
程序仍要等待消息终态和完整 usage，再进行结算。

断流后自动重连可能重复文字、工具提议和费用。只有目标 Provider 给出并经过验证的恢复协议时，客户端才能续接原调用。
其他情况应结束当前 attempt；若策略允许重新调用，也要创建新的请求身份和预算记录。

## Tool use：模型只提出动作

模型返回 `get_ticket` 提议，只表示“建议查询这张工单”。权限仍由应用判断。正确链路是：

~~~text
proposal
→ block/JSON completeness
→ schema and semantic validation
→ tenant/resource authorization
→ budget and approval
→ idempotent execution
→ effect verification
→ sanitized tool result
→ next model turn
~~~

至少分开三种状态：

- **proposal**：模型建议调用什么工具和参数；
- **authorization**：业务系统允许谁对哪个资源做什么；
- **effect**：外部系统是否真的完成了副作用。

工具名、参数、schema revision、call ID、policy decision、幂等键和 provider receipt 都应进入审计。审批应绑定规范化参数和资源版本，防止审批后参数漂移。

### 超时后仍要确认远端发生了什么

客户端超时、取消或读取失败后，远端是否收到请求可能仍属未知。对于写操作，应先持久化执行意图或 outbox，
再使用业务幂等键执行。

若远端可能成功但本地没有确认，状态应是 outcome uncertain，并进入 reconciliation。把它当作普通失败自动重试，可能造成重复扣款、重复发信或重复删除。

## Retry 先回答三个问题

每次自动重放前回答：

1. 该 failure 在当前版本 policy 中是否 retryable？
2. 重放是否 replay safe？
3. 能否证明上一次没有被接收、执行或计费？

一个 HTTP 5xx allowlist 不能替代这三个判断。即使没有外部工具，重新生成也会带来不同输出和额外 usage。

流已经向用户发布部分内容时，默认不要透明重放。保留每个 attempt 的 request identity、状态、usage reservation 和最终结算，才能解释成本与用户实际看到的内容。

## Usage 与预算要按 attempt 记账

请求中的 `max_tokens` 是输出上限，不是实际用量。发送前可以按保守上界预留预算：

\[
R=C_{in}(\widehat T_{in})+C_{out}(T_{out,max})+C_{other,max}.
\]

输入 token 仍是目标 tokenizer/template 的估计。缓存、thinking、工具、tier、税费和币种怎样计价，必须来自带日期的正式 pricing contract。

预算流程应是：

~~~text
preflight origin/model/version
→ reserve upper bound
→ send one attempt
→ settle from complete trusted usage
→ otherwise mark reservation uncertain
→ reconcile with billing export
~~~

一次逻辑调用可能包含多个网络 attempt，每次发送都可能生成和计费。因此，每个 attempt 都需要自己的预算预留和终态。

客户端账本只能保证本地状态一致，无法与远端生成和供应商账单组成同一个原子事务。

## 长上下文不是“全部塞进去”

标称 context window 只涉及协议容量的一部分，不证明每个位置和任务同样可靠。至少分开三层：

| 层级 | 要验证的问题 |
|---|---|
| protocol acceptance | 请求是否被 API 接受 |
| runtime completion | 是否在 timeout 和预算内正常结束 |
| effective context | 不同位置和任务是否得到可靠答案 |

有效长上下文评测应覆盖单点检索、多点综合、冲突消解、顺序、引用、全局聚合和长输出约束。一个 needle-in-a-haystack 分数不能代表全部能力。

长上下文与 RAG 不是竞争关系。RAG 帮助更新知识、执行权限过滤并缩小输入；长上下文减少切分损失并支持跨文档综合。

### Prompt caching 也有 identity

Prompt caching 可能降低重复前缀的成本或首 token 延迟。判断两次请求能否复用缓存时，至少要考虑模型版本、
block 顺序、工具 schema、预处理方式、租户、数据等级和缓存生命周期。

只按可见 Prompt 字符串共享缓存，可能把一个租户或权限策略下的前缀用于另一个上下文。

评测时应同时报告冷启动与缓存命中的延迟、可缓存请求数、命中与未命中数、用量、质量变化和失效行为。
单独一个命中率无法说明缓存是否安全、有效。

## 模型选型靠 workload，不靠代际名

先固定代表性样例，再比较候选 model ID：

- 任务质量：抽取、代码、综合、规划和工具参数；
- 协议完整性：typed blocks、unknown fields 和 terminal；
- 长上下文：位置、多跳、冲突、引用与聚合；
- 安全：提示注入、越权工具、敏感数据和 over-refusal；
- 系统：TTFT、terminal latency、限流、重试和取消；
- 成本：每 attempted 与 successful task 的真实 usage；
- 治理：区域、保留、日志、密钥和供应商要求。

基线与候选版本使用同一组样例，并保存逐例输出、错误和统计分母。

模型、API headers、Prompt、工具 schema、解析器、重试策略或价格快照发生变化时，改变的都是整个候选系统，
不能把差异全部归因于模型本身。

先 shadow，再 canary，并保留完整旧 bundle。回滚不是只改回一个 model alias。

## 一个渐进式接入实验

不要从自动工具循环开始。按四级增加能力：

### Level 1：非流式文本 {#level-1-text}

固定 model ID 与 API 版本请求头，发送最短文本请求。保存脱敏后的请求、所有响应 block、停止原因、用量、request ID 和延迟。

验收目标是“协议对象完整”，不是“回答看起来不错”。

先用仓库的固定 Anthropic 样例检查顶层 `system`、文本 block、usage 和停止原因映射：

```powershell
python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl `
  --output artifacts/cloud-api/contracts.json
```

命令会同时验证三种 Provider 的文本子集；其中 Anthropic case 使用 `.invalid` 域名和虚构密钥，不访问网络。

### Level 2：多 block 与失败回放

准备一组离线样例，分别包含多个文本 block、只有工具提议、未知 block、达到输出上限、Provider 错误和缺失消息终态。

逐项确认解析器会保留这些情况，或者返回明确错误。任何分支都不能静默丢失数据后伪装成普通文本成功。

### Level 3：流式状态机

测试字节任意切分、事件重复或错序、Unicode 边界、断流和未知增量。分别记录首个 block、首段文本、
模型停止原因、消息终态与连接关闭时间。

仓库现有的 Anthropic 文本子集可以这样回放：

```powershell
python -m pytest tests/test_cloud_api.py tests/test_cloud_stream.py `
  -k anthropic -q
```

这些测试覆盖文本 block 和事件次序；遇到 `tool_use`、thinking 或其他非文本 block 时会停止，而不是假装已经支持。

### Level 4：受控工具

准备只读、幂等写入和高风险写入三类工具，测参数正确率、权限拒绝、审批、重复副作用、outcome uncertain 与 prompt injection。

本仓库尚未提供一条真实 Anthropic 工具调用 runner。练习这一层时，可以先复用
[Agent Runtime](../applications/agent-runtime.md) 的授权、幂等和恢复状态机，再为目标 Messages 版本实现并验证完整 block adapter。

本仓库提供了 adapter、stream、retry 和预算的离线验证程序，入口见
[Claude 证据台账](../evidence/claude-controls.md)。这些程序不访问真实账号；当前 provider 的实际行为仍要用
受控的真实请求核对。

## 常见错误

- 用 Claude 品牌名代替精确 model ID、API version 和核对日期。
- 从公开论文或输出风格推断当前模型内部架构。
- 只读取第一个 text block，丢弃工具、引用、拒答和 unknown block。
- 把 transport EOF、block stop 和 provider terminal 当作同一种结束。
- block 还没闭合就解析并执行流式工具参数。
- 把 model proposal 当成业务授权，或把 timeout 当成未执行。
- 对所有 429/5xx 或断流透明重试，不记录独立 attempt 和费用。
- 只报长上下文 needle 命中或 prompt-cache hit rate。
- 把离线样例的结果写成真实 Claude 质量、计费或生产可靠性结论。

## 面试时怎样回答

面对“如何生产接入 Claude”，可以按五层回答：

1. 固定 model ID、API 版本请求头、schema 与核对日期。
2. 保留所有 Messages block、停止原因、usage 和原始响应身份。
3. 先完成字节分帧，再用 Messages 状态机处理流式事件。
4. 把工具提议、授权、真实副作用和待对账结果分开。
5. 用代表性任务验证质量、安全、延迟、成本和迁移。

继续追问时，可以回到工单例子解释三件事：未知 block 必须保留或明确报错；截断的流需要独立终态；
客户端取消以后，供应商是否停止生成和计费仍需外部证据。

## 自测

1. 为什么 Constitutional AI 论文不能证明当前 Claude 的完整训练配方？
2. 纯文本 parser 遇到 tool-only response 时应返回什么状态？
3. 为什么 SSE event 数和 token 数不能互相替代？
4. 工具写入 timeout 后，什么证据允许自动重试？
5. 怎样证明一次模型迁移是质量提升，而不是只展示几段更好的回答？

## 继续学习

- [云 API 契约](cloud-api-contracts.md)：跨 provider 的 canonical model 与 adapter。
- [Agent Runtime](../applications/agent-runtime.md)：授权、执行、回放与可观测性。
- [Evaluation Gate](../practice/projects/evaluation-gate.md)：paired cases 与发布决策。
- [Opaque Reasoning 工件安全](../quality/reasoning-artifact-security.md)：不透明状态与轨迹发布。
- [Claude 证据台账](../evidence/claude-controls.md)：具体检查程序、命令和目前尚未验证的部分。
