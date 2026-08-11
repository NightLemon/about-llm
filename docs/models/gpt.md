# GPT 家族

## 学习目标与证据边界

读完本章应能区分三件事：公开 GPT 论文证明了什么、当前 OpenAI API 对外承诺了什么、哪些产品内部细节仍然未知。你还应能设计一次模型快照升级评测，而不是只把配置中的 model id 换成新名字。

**先修知识**：decoder-only Transformer、causal language modeling、SFT/偏好训练、HTTP/JSON 与工具调用。

公开论文可用于理解 GPT-1/2/3、InstructGPT 等研究路线；当前产品模型的参数量、层数、训练数据、稀疏/稠密结构和完整后训练配方若未披露，就不能由旧论文、模型名称或输出风格外推。以下产品接口快照核对日期为 **2026-08-06**。

## 从预训练模型到可执行系统

GPT 的稳定核心是自回归条件分布：

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
\]

早期路线展示“通用预训练 + 任务适配”；规模扩大后，few-shot/in-context learning 允许把示例放进上下文而不更新参数。SFT 与偏好优化进一步改变模型遵循指令、拒答和对话的行为分布。工具、检索、代码执行器与 structured output 则把概率生成器接入可验证系统。

这几层不能混为一谈：

| 层 | 改变什么 | 不自动保证什么 |
|---|---|---|
| 预训练 | 语言、代码与世界模式的条件分布 | 指令遵循、事实实时性 |
| SFT/偏好训练 | 输出行为、格式、帮助性与拒答倾向 | 外部权限、业务真值 |
| Prompt/context | 当前请求的条件与示例 | 参数更新、永久记忆 |
| tools/RAG | 可调用能力与外部证据 | 模型一定正确使用证据 |
| schema/grammar | 输出语法空间 | 字段真实、动作已授权 |

## 公开研究怎样读

### GPT-1 到 GPT-3

学习重点不是背参数量，而是观察任务接口的变化：GPT-1 强调生成式预训练后再适配；GPT-2 展示更大规模无监督语言建模的任务迁移；GPT-3 把自然语言说明和示例直接放进上下文，系统性展示 zero/one/few-shot。

论文实验结论绑定当时的数据、模型规模和评测协议。不能用 GPT-3 论文中的架构表描述当前 API 模型，也不能把 benchmark few-shot 提升等价为生产任务可靠性。

### InstructGPT 与偏好训练

InstructGPT 路线把监督示范、偏好排序、奖励模型和强化学习连接起来。它解释了“为什么会续写”与“为什么按意图回答”是两个训练问题，也揭示 reward hacking、标注者代表性、分布外行为和对齐税等新风险。

RLHF 不是一个固定配方。当前产品是否使用某个奖励模型结构、PPO 变体或数据比例，只有官方明确披露时才能写成事实。

## 当前产品接口快照

截至 2026-08-06，OpenAI 官方模型目录把 GPT-5.6 系列列为当前通用 frontier 路线，并按复杂专业任务、智能/成本平衡和高吞吐成本敏感任务区分产品档位。型号、别名、价格、上下文、输出上限与工具支持都属于时间敏感产品事实；本教材不复制会快速过期的价格表，选型时应打开具体 model page 并保存检查日期。

官方目录当前推荐通过 Responses API 使用最新模型。Chat Completions 仍是常见兼容接口，但二者的数据模型不能只靠替换 URL 迁移：Responses 面向带状态、多模态、工具和多个 output item 的工作流；Chat Completions 以 messages 与 choices 为主要心智模型。生产 adapter 应明确支持哪一种，而不是统称“OpenAI API”。

一个稳定的请求配置至少记录：

```json
{
  "provider": "openai",
  "api_surface": "responses",
  "model": "<pinned-model-id-or-snapshot>",
  "checked_at": "2026-08-06",
  "sampling": {"temperature": 0},
  "max_output_tokens": 1024,
  "tool_schema_version": "sha256:...",
  "prompt_version": "git:..."
}
```

字段是否被某个具体模型支持必须按其 model page 与 API reference 核对；不要把示意配置直接当作所有端点的共同 schema。

## 指令、输出项与工具

### 指令层级

不同 API/模型会区分系统、开发者、用户、工具结果等输入来源。业务代码不能把不可信网页或 RAG 文档提升到高权限指令位置。迁移模型时应回归：冲突指令、长上下文中部约束、多轮状态、工具结果注入和语言切换。

### Structured Outputs

JSON mode 的目标是有效 JSON；Structured Outputs 在受支持的 JSON Schema 子集内约束结构。两者都不保证值为真、引用存在、金额合理或工具调用有权限。正确链路是：约束生成 → schema 校验 → 业务规则 → 身份/ACL → 幂等/审批 → 执行。

### Tool call 不是执行

模型输出的 tool call 是候选动作。provider adapter 负责保留 call id、参数与事件；Agent runtime 负责参数类型、资源归属、权限、预算、审批、幂等和审计。不要让 SDK 的自动工具循环绕过业务控制层。

## Reasoning 与 test-time compute

推理型模型可能用更多内部计算或输出 token 换取复杂任务质量。公平比较必须固定或报告：reasoning effort、最大输出、实际 usage、工具/验证器、候选数、wall time 和每成功任务成本。

可见的解释文本不是内部计算的完整忠实转录，也不是正确性证明。数学和代码任务优先用计算器、类型检查、编译、测试和独立 verifier 检验最终结果。

## 工程选型与迁移

不要先问“哪个 GPT 最强”，先定义 workload：

1. 任务质量：抽取、代码、规划、长文综合还是实时对话；
2. 输入模态和工具：文本、图像、音频、文件搜索、函数调用；
3. SLO：TTFT、E2E、吞吐、并发和可接受错误率；
4. 治理：区域、数据保留、日志、敏感信息和人工审批；
5. 成本：输入、缓存、输出、工具和重试后的每成功任务成本。

模型升级使用 paired evaluation：同一 case、同一工具环境和预算运行旧/新快照，报告总体与切片差异、置信区间、格式率、安全 guardrail 和延迟。先 shadow，再小流量 canary；保留旧 prompt、模型 id、解析器与路由以便回滚。

## 可运行实验

本仓库的 `cloud_api` adapter 用离线 fixture 学习 OpenAI-compatible 的 messages、choices、usage 与 finish reason；`cloud_stream.OpenAICompatibleTextStream` 另以离线 SSE fixture 校验单 choice text delta、usage、finish_reason 和 `[DONE]` 的次序。它们不是 Responses API 的完整实现，不支持流式 tool/refusal 等非文本 item，也没有证明真实 OpenAI 端点可用。

建议增加的模型升级实验：

1. 固定 50–200 个含事实、结构化输出、工具、拒答和多语言的 case；
2. 保存原始响应 item/event，而不只保存最终文本；
3. 比较 schema 合法率、任务指标、工具参数正确率和越权调用率；
4. 分开统计 provider error、解析错误、内容错误与预算耗尽；
5. 输出差异报告，并为关键退化保留可回放 fixture。

## 常见错误

- 把公开 GPT-3 架构参数写成当前产品内部结构；
- 把 `OpenAI-compatible` 当作工具、流式事件和错误完全兼容；
- 只解析第一个文本字段，丢掉 tool、citation、reasoning 或拒答信息；
- 用 schema 通过率代替事实正确率和授权检查；
- 用 temperature=0 宣称跨硬件、批处理和服务版本严格确定；
- 升级 model id，却沿用未经回归的 prompt、token 预算和 parser。

## 面试追问

1. next-token prediction 为什么能支持 in-context learning？它和参数更新有什么区别？
2. 预训练、SFT、偏好优化和工具系统分别解决什么问题？
3. Responses API 与 Chat Completions 的 adapter 边界应怎样设计？
4. Structured Outputs 保证什么，为什么仍不能直接执行工具？
5. 如何设计一次有统计把握、可回滚的模型快照升级？
6. reasoning token 增加时，怎样做同预算质量比较？

## 一手资料

- OpenAI，[Model catalog](https://developers.openai.com/api/docs/models)，产品型号与能力；核对日期 2026-08-06。
- OpenAI，[Text generation](https://developers.openai.com/api/docs/guides/text)，Responses/文本生成接口。
- OpenAI，[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)，结构约束边界。
- Brown 等，[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)，GPT-3 与 in-context learning。
- Ouyang 等，[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，InstructGPT。
