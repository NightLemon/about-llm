# Claude 家族

## 能确认什么

Claude 是 Anthropic 的闭源模型产品。可依据公开研究学习 Constitutional AI、RLHF/RLAIF、安全评测与长上下文应用；未公开的当前参数量、层数、训练数据、路由和具体后训练配方保持未知。不要把一篇 Anthropic 论文直接写成某个产品版本的完整实现。

## Messages API 心智模型

- system 通常位于顶层而非普通消息；
- content 是 block 序列，不只纯字符串；
- tool use 与 tool result 有明确 block 协议；
- streaming 包含不同事件类型；
- usage 使用 input/output token 命名；
- stop_reason 需要进入日志和错误分析。

只提取 text 会丢失工具、引用或其他 block。业务 adapter 应按任务显式保留所需结构。

## 长上下文

标称窗口不等于所有位置和任务可靠。用单点/多点检索、跨文档综合、冲突、顺序、全局聚合与长输出一致性测试。Prompt caching 可改变成本与 TTFT，但 cache key、敏感数据和失效策略要明确。

## 工具 Agent

Claude 只提出 tool use；执行权限仍在本仓库 Agent runtime。外部网页、邮件和检索文档是低信任数据。高风险工具需要参数绑定审批与幂等 ledger。

## 选型

在自有评测集比较质量、长上下文、工具 schema、流式体验、限额、区域、数据政策、延迟和每成功任务成本。模型升级走 paired eval 和 canary。

## 面试追问

1. Constitutional AI 的原则、批评/修订与偏好训练怎样衔接？
2. 为什么 RLAIF 不能消除人类监督？
3. content blocks 对 parser 设计有什么影响？
4. 长上下文与 RAG 为什么互补？
5. 闭源模型怎样做可复现评测？
