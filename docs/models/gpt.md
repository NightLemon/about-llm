# GPT 家族

## 核心路线

GPT 将 Transformer decoder 与因果语言建模结合：给定历史 token 预测下一个 token。早期公开工作展示“通用预训练 + 任务适配”，随后扩大参数、数据和上下文，出现 few-shot/in-context learning。再后的指令微调与人类偏好训练把“会续写”转成“会按对话意图回答”。

这条路线的关键不是某个型号：

1. 自回归目标统一文本、代码和任务格式；
2. scale 同时依赖模型、数据与计算；
3. Prompt 把任务转为条件生成；
4. SFT/偏好优化改变行为分布；
5. tool/structured output 把语言模型接入可验证系统；
6. test-time compute、工具与验证器进一步换取质量。

## 公开模型与产品模型

公开 GPT 论文可用于理解 decoder-only、in-context learning 和 RLHF。当前 OpenAI 产品模型的参数量、训练数据和内部架构若未披露，就不能由旧论文外推。API 的 model id、快照、上下文、工具和价格属于时间敏感产品信息，应按固定版本核对官方文档。

## 工程接口

需要区分：

- 文本/聊天输入协议；
- system/developer/user/tool 指令层级；
- structured output 与 JSON Schema；
- tool call 建议和外部执行权限；
- streaming event 与 usage；
- batch、缓存和异步任务；
- 模型快照、限额、数据保留与区域。

OpenAI-compatible 只是请求形状。DeepSeek/Qwen 等兼容端点可能有不同扩展、错误和模型语义。

## 本仓库怎样学习

- 用 MiniGPT 理解 decoder、causal mask 与生成；
- 用 Transformers tiny GPT-2 验证标准框架训练；
- 用 cloud_api adapter 理解 OpenAI-compatible 消息和 usage；
- 用 evaluation runner 比较 Prompt/模型快照；
- 用 Agent runtime 执行工具，不把模型 tool call 当授权。

## 面试追问

1. next-token prediction 为什么能支持 few-shot？
2. in-context learning 与参数更新有什么区别？
3. RLHF 解决了什么，又引入哪些 reward hacking 风险？
4. structured output 保证了什么，没保证什么？
5. API 快照升级时怎样做 paired evaluation 与回滚？
