# Claude 家族

## 学习目标与证据边界

读完本章应能区分 Anthropic 的公开研究、Claude 产品能力和 Messages API 契约；还能把 content blocks、工具调用、长上下文与 prompt caching 接入一个可观测、可回放、受权限约束的系统。

**先修知识**：decoder-only Transformer、SFT/偏好训练、HTTP/JSON、流式事件、Agent 工具执行与评测。

Claude 是闭源模型产品。Constitutional AI、RLHF/RLAIF 等公开论文可以解释一条研究路线，却不能证明当前某个 Claude 版本采用论文中的完整训练配方。未公开的参数量、层数、训练数据、稀疏/稠密结构、路由和后训练细节应**保持未知**；不要从输出风格、旧论文或产品名称反推内部架构。

本章接口事实按 Anthropic Messages 官方参考于 **2026-08-06** 核对。具体 model id、上下文、价格、区域、限额和 beta header 都是时间敏感产品事实，应在部署时固定检查日期和版本，不在稳定教材中维护“永久最新”表。

## 公开研究路线怎样读

### Constitutional AI

Constitutional AI 的核心学习价值是把一组自然语言原则引入监督与偏好数据生成：模型先根据原则批评、修订回答，再由人类或 AI 反馈形成训练信号。它把“哪些行为值得奖励”显式化，便于讨论原则冲突、覆盖不足和标注规模。

但 constitution 不是可执行安全策略。原则可能含糊、互相冲突或遗漏业务约束；模型也可能错误应用原则。生产系统仍需身份、ACL、数据隔离、工具审批、审计和人工升级通道。

### RLAIF 与人类监督

RLAIF 用模型反馈扩展偏好标注，降低部分人工比较成本。它没有消除人类监督：人仍要选择原则、抽检反馈质量、定义不可接受风险、处理分布外案例并决定发布门槛。若 evaluator 与被评模型共享偏差，自动反馈还可能放大盲区。

阅读论文时把证据拆成三层：论文实际实验、作者提出的机制解释、你对当前产品的外推。只有第一层能直接归属于论文设置；第三层必须标为假设。

## Messages API 心智模型

Messages 不是“把所有内容拼成一个字符串”。请求与响应都应按有类型的 block 处理。

一个教学用请求形状如下：

```json
{
  "model": "<pinned-model-id>",
  "max_tokens": 1024,
  "system": "你是受约束的分析助手。",
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "text": "分析这份工单"}]
    }
  ]
}
```

这段 JSON 只表达稳定的数据模型，不承诺所有模型或 API 版本都支持相同可选字段。关键边界是：

- `system` 位于请求顶层，不是一个普通的 `system` role message；
- `messages` 表示 `user`/`assistant` 对话历史；
- `content` 可以是 block 序列，不能假设永远只有纯文本；
- 响应的 `content` 可能包含 text、tool use 或其他受支持 block；
- `stop_reason`、usage 和 request id 应进入日志，而不是只保存最终文本；
- usage 使用 input/output token 语义；缓存相关 token 字段按实际响应版本单独保留。

如果业务只返回 `response.content[0].text`，一旦第一个 block 不是文本、出现多个文本块或工具调用，就会静默丢数据。更稳的 adapter 先保留原始 block，再按任务投影成文本、引用、工具候选或拒答状态。

## Block 与事件驱动解析

### 为什么不能只做 text parser

统一内部结构可以是：

```text
ProviderResponse
├── provider/model/request_id
├── blocks[]
│   ├── text(text)
│   ├── tool_call(id, name, arguments)
│   └── provider_specific(raw)
├── stop(reason, sequence?)
├── usage(input, output, cache?)
└── raw_response_hash
```

规范化层不应把未知 block 丢弃；可保留 `provider_specific` 供回放和后续迁移。上层任务若要求纯文本，可以显式连接所有 text block；若响应没有文本，应返回类型错误或工具状态，而不是空字符串。

### Streaming 是状态机

流式传输会把消息开始、content block 开始/增量/结束、消息级增量与结束拆成不同事件。健壮实现需要：

1. 用 message/block id 或 index 关联增量；
2. 按 block 类型累积 text 或工具参数；
3. 只在结构闭合后解析工具 JSON；
4. 保存终止原因与最终 usage；
5. 对断流、重复事件和未知事件显式报错或降级。

SSE event/chunk 数不是 token 数。吞吐和费用必须使用服务端 usage 或经声明的 tokenizer 估算，二者要分开标记。

## 工具调用的正确状态机

模型产生 `tool_use` block 只表示**候选动作**。外部 runtime 校验后执行工具，再把对应 `tool_result` 作为后续输入返回；call id 必须关联，不能靠工具名或顺序猜测。

推荐链路：

```text
model proposes tool_use
  -> schema/type validation
  -> resource ownership and ACL
  -> budget / approval / idempotency
  -> isolated execution
  -> sanitize tool_result
  -> append result to conversation
  -> model continues or stops
```

必须处理的失败包括：参数 JSON 不完整、未知工具、同一 call 重复提交、执行超时、结果过大、结果内提示注入和模型再次请求同一副作用。SDK 的自动 tool loop 不能越过业务授权层。

外部网页、邮件、工单和 RAG 文档都属于低信任数据。即使它们被包装成 tool result，也不能获得 system 指令的权限。高风险工具应把审批绑定到**规范化后的参数与资源版本**，避免审批后参数漂移。

## 长上下文与 Prompt Caching

标称 context window 只说明协议上限，不证明所有位置和任务同样可靠。长上下文评测至少分开：

- 单点检索：目标事实在开头、中间、结尾；
- 多点综合：答案需要跨多个片段组合；
- 冲突消解：新旧版本、可信度和时间戳冲突；
- 顺序与引用：事件先后、页码、段落证据；
- 全局聚合：计数、分类和覆盖全部文档；
- 长输出：约束是否在生成后段仍保持。

长上下文与 RAG 互补。RAG 用检索降低输入规模、更新知识并给出证据；长上下文减少切分损失并支持跨文档综合。把整个知识库塞进窗口通常会增加延迟、成本和干扰，也不能替代权限过滤。

Prompt caching 可以降低重复前缀的计算成本或 TTFT，但工程上要记录：哪些 block 可缓存、cache 命中与创建 token、失效条件、敏感数据生命周期、租户隔离、模型/工具 schema 版本和观测字段。缓存命中不代表回答质量不变。

## 模型选型与版本迁移

不要按“最强 Claude”选型，先定义 workload：

| 维度 | 需要测什么 |
|---|---|
| 任务质量 | 抽取、代码、长文综合、规划、工具参数正确率 |
| 结构 | schema 合法率、block 保真、未知 block 处理 |
| 长上下文 | 位置、多跳、冲突、引用和全局聚合 |
| 安全 | 提示注入、越权工具、敏感数据、拒答误伤 |
| 性能 | TTFT、E2E、输出速度、并发、限流与重试 |
| 成本 | input/output/cache/tool/retry 后每成功任务成本 |
| 治理 | 区域、日志、数据保留、密钥与供应商风险 |

升级时固定旧/新 model id、prompt、工具 schema、token 预算和 case 集，执行 paired evaluation；分别报告总体与语言、长度、工具类型等切片。先 shadow，再 canary，保留旧 adapter/parser 与路由以便回滚。模型别名若会漂移，不适合作为唯一可复现标识。

## 可运行实验

本仓库 `about_llm.integrations.cloud_api` 中的 Anthropic adapter 用离线 fixture 检查顶层 system、消息映射、文本解析、usage 和 stop reason；`AnthropicTextStream` 还校验 text block start/delta/stop、message_delta 与 message_stop 的状态次序。它只覆盖 text_delta 子集，不支持 tool/thinking/signature 等 block，也没有接入真实 streaming HTTP 或访问 Anthropic 账号。

建议把实验扩成三组：

1. **契约回放**：为 text、多 text block、tool use、无文本、max token、未知 block 和错误响应保存脱敏 fixture；
2. **长上下文评测**：生成带位置、冲突和跨文档依赖的 case，报告答案与证据定位，不只报 needle 命中；
3. **受控 Agent**：让模型调用只读查询、幂等写入和高风险写入三类工具，测参数正确率、审批触发、重复副作用和注入攻击。

若接入真实 API，保存 provider、model id、API/version header、checked_at、request id、原始 block、usage、stop reason、重试和延迟；密钥、用户内容与工具结果按数据分级脱敏。真实端点结果与离线 fixture 结果必须分栏报告。

## 常见错误

- 把 Constitutional AI 论文写成当前产品的完整内部实现；
- 把 RLAIF 描述成不需要人类定义原则和监督；
- 把顶层 `system` 当作普通 role message；
- 只取第一个 text block，丢掉工具、引用或未知 block；
- 在工具参数尚未流完时执行；
- 用长 context window 数字代替位置鲁棒性评测；
- 认为 prompt caching 自动满足租户隔离和数据删除；
- 把 tool use 当作授权，把 tool result 当作高信任指令；
- 只换 model id，不回归 parser、prompt、token 预算与拒答行为。

## 面试追问

1. Constitutional AI 的批评/修订与偏好训练怎样衔接？边界在哪里？
2. 为什么 RLAIF 不能消除人类监督，且可能放大 evaluator 偏差？
3. content blocks 对数据库 schema、stream parser 和回放系统有什么影响？
4. `tool_use` 到 `tool_result` 的 call id、审批和幂等怎样设计？
5. 长上下文与 RAG 为什么互补？怎样测 lost-in-the-middle？
6. Prompt caching 的命中率、TTFT、成本和敏感数据风险怎样联合观测？
7. 闭源模型升级怎样做到统计可比、可审计和可回滚？

## 一手资料

- Anthropic，[Messages API reference](https://platform.claude.com/docs/en/api/messages)，请求/响应、content blocks、usage 与 stop reason；核对日期 2026-08-06。
- Anthropic，[Tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)，工具 block 与客户端执行循环。
- Bai 等，[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)，原则、批评/修订与 RLAIF 研究路线。
- Bai 等，[Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)，HH-RLHF 研究设置。
