# Claude：把闭源模型接成一个可靠系统

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：要通过 Claude API 构建对话、长上下文、RAG 或 Agent 系统的开发者。
- **先修**：理解 HTTP/JSON、SSE、工具调用和基础模型评测。
- **首次阅读**：产品边界 → Messages blocks → streaming → tool loop → 预算与迁移。
- **完成信号**：能保真处理 typed response，并区分模型提议、业务授权、真实副作用和最终任务成功。
- **卡住时**：先读[云 API 契约](cloud-api-contracts.md)，用一个非流式 text request 建立最小基线。

</div>

**模型导航**：[云 API 契约](cloud-api-contracts.md) · [Agent 总览](../applications/agents.md) · [评测项目](../practice/projects/evaluation-gate.md) · [Claude 证据台账](../evidence/claude-controls.md)
{ .doc-nav }

学习 Claude 的难点不在记忆某一代模型的参数，而在接受一个事实：你面对的是持续变化的闭源产品和 API 契约。

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

## Messages 不是一问一答字符串

一个 Messages request 可以包含顶层 system、user/assistant 历史和 typed content blocks；response 也可能包含多个不同类型的 block。

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

最小 text request 的形状可以是：

~~~json
{
  "model": "<reviewed-model-id>",
  "max_tokens": 512,
  "system": "你是受约束的工单分析助手。",
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "text": "归纳这份工单"}]
    }
  ]
}
~~~

这只是对象关系示例，不承诺任意模型和版本支持相同的可选字段。system 位于请求顶层；content 也不能假设永远是纯文本。

### 为什么要保留 typed blocks

如果代码只读取第一个 block 的 text，遇到多个文本块、工具提议、引用、拒答或未知 block 时就会静默丢数据。

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

Canonical 层表达用户、租户、任务、预算和工具候选；provider adapter 负责顶层 system、blocks、headers、usage 和 stop。

这能让你更换模型或供应商时只替换协议映射，而不是让业务逻辑依赖某个 wire field。provider 特有能力仍要保留，不能为了统一 schema 把所有东西压成字符串。

## Streaming 是两层状态机

网络每次读到的是 arbitrary bytes，不是一个完整 token、JSON 或 SSE event。流式处理至少分两层：

1. **Byte/SSE framing**：处理 UTF-8、换行、event/data fields、空行终止与资源上限。
2. **Provider state**：处理 message、block、delta、usage、stop 与 error 的合法次序。

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

工具参数分多个 delta 到达时，JSON prefix 即使能解析，也不表示参数已经完整。只有 block 正常闭合、schema 和语义都通过后，才可以进入授权流程。

### 四种结束不能混在一起

| 层级 | 结束意味着什么 |
|---|---|
| content block stop | 某个 block 结构闭合 |
| model stop reason | 模型为什么停止 |
| provider message terminal | 消息协议正常结束 |
| transport EOF/close | 字节流结束 |

只有 EOF 而没有 provider terminal 是截断，不是成功。看到文本 block 结束也不能提前结算 usage 或执行仍未闭合的工具。

断流后自动 reconnect 可能重复文本、tool proposal、usage 和费用。除非 provider 提供经过验证的 resume contract，否则把失败标为新的终态或发起新的、有新 identity 的调用。

## Tool use：模型只提出动作

模型返回 tool proposal，不代表它拥有调用权限。正确链路是：

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

客户端 timeout、cancel 或 read failure 无法证明 provider 或外部工具没有收到请求。对于写操作，应先持久化 intent/outbox，再用业务幂等键执行。

若远端可能成功但本地没有确认，状态应是 outcome uncertain，并进入 reconciliation。把它当作普通失败自动重试，可能造成重复扣款、重复发信或重复删除。

## Retry 先回答三个问题

每次自动重放前回答：

1. 该 failure 在当前版本 policy 中是否 retryable？
2. 重放是否 replay safe？
3. 能否证明上一次没有被接收、执行或计费？

一个 HTTP 5xx allowlist 不能替代这三个判断。即使没有外部工具，重新生成也会带来不同输出和额外 usage。

流已经向用户发布部分内容时，默认不要透明重放。保留每个 attempt 的 request identity、状态、usage reservation 和最终结算，才能解释成本与用户实际看到的内容。

## Usage 与预算要按 attempt 记账

请求中的 max_tokens 是输出上限，不是实际 usage。发送前可以按保守上界做 reservation：

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

一次 logical call 内的每个网络 attempt 都可能生成和计费，因此不能只 reserve 一次。客户端 ledger 也无法与远端 provider billing 建立原子事务。

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

Prompt caching 可能降低重复前缀成本或 TTFT，但 cache identity 至少要考虑 model/version、ordered blocks、tool schemas、预处理、租户、数据等级和 lifecycle。

只按可见 prompt 字符串共享 cache，可能跨租户或跨 policy 错用。评测时同时报告 cold/warm latency、eligible/hit/miss 分母、usage、质量漂移和失效行为，不能只报 hit rate。

## 模型选型靠 workload，不靠代际名

先固定代表性 cases，再比较候选 model ID：

- 任务质量：抽取、代码、综合、规划和工具参数；
- 协议完整性：typed blocks、unknown fields 和 terminal；
- 长上下文：位置、多跳、冲突、引用与聚合；
- 安全：提示注入、越权工具、敏感数据和 over-refusal；
- 系统：TTFT、terminal latency、限流、重试和取消；
- 成本：每 attempted 与 successful task 的真实 usage；
- 治理：区域、保留、日志、密钥和供应商要求。

使用同一 case 集做 paired comparison，保存逐 case 输出、错误和分母。升级时 model、API headers、prompt、tool schema、parser、retry 与 pricing 任一变化，都应视为候选系统变化。

先 shadow，再 canary，并保留完整旧 bundle。回滚不是只改回一个 model alias。

## 一个渐进式接入实验

不要从自动工具循环开始。按四级增加能力：

### Level 1：非流式 text

固定一个 model ID 和 API/version headers，发送最短 text request。保存脱敏 request、typed blocks、stop、usage、request ID 和 latency。

验收目标是“协议对象完整”，不是“回答看起来不错”。

### Level 2：多 block 与失败回放

准备一组离线样例，分别包含多个 text blocks、tool-only、unknown block、max-token stop、provider error 和
缺失 terminal。逐项确认 parser 会保留或明确报告这些情况，而不是静默丢失数据。

### Level 3：流式状态机

测试 byte 任意切分、重复/错序 event、Unicode 边界、断流和未知 delta。分别记录首 block、首 text、model terminal 与 transport close。

### Level 4：受控工具

准备只读、幂等写入和高风险写入三类工具，测参数正确率、权限拒绝、审批、重复副作用、outcome uncertain 与 prompt injection。

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

1. 固定 model、API headers、schema 与 checked_at。
2. 保留 Messages typed blocks、stop、usage 和 raw identity。
3. 用 byte framing + provider state machine 处理 streaming。
4. 把 tool proposal、authorization、effect 和 reconciliation 分开。
5. 用 workload cases 验证质量、安全、延迟、成本和迁移。

继续追问时，应能解释为什么 unknown block 不能静默丢弃、partial stream 默认不能透明 replay，以及 client cancel 为什么不证明 server 停止生成和计费。

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
