# 云模型 API 的共同点与差异

## 不要把 messages 当成统一标准

不同供应商都能完成对话，但角色、system 位置、tool schema、流式事件、usage、缓存和错误码不同。业务层可以统一最小概念，adapter 必须保留能力差异。

| 维度 | OpenAI-compatible | Anthropic Messages | 本仓库 Gemini `generateContent` adapter |
|---|---|---|---|
| system | messages 中常见 | 顶层 system | systemInstruction |
| assistant role | assistant | assistant | model |
| 内容 | string 或多段结构 | content blocks | parts |
| usage | prompt/completion tokens | input/output tokens | usageMetadata |
| 结束 | finish_reason | stop_reason | finishReason |

表格只描述本仓库 adapter 覆盖的稳定最小契约，实际字段以固定 API 版本的官方文档为准。

## GPT、DeepSeek 与 Qwen

三者可能通过 OpenAI-compatible 形状接入，但兼容不等于完全等价。检查 tool calling、JSON schema、reasoning 字段、stream usage、缓存、错误和模型 id。不要把某个兼容端点的扩展字段无条件传给另一家。

## Claude

Anthropic Messages 将 system 与对话分开，content 是 block 数组。工具结果、thinking/其他 block 与纯 text 不能用同一解析假设。只需要文本时明确过滤 text block，同时保留非文本 block 的审计信息。

## Gemini

截至 2026-08-05，Gemini API 官方文本生成指南推荐新项目使用已 GA 的 Interactions API。本仓库 adapter 为教学与兼容性实现 `generateContent`：它使用 `contents`/`parts`、`user`/`model` role 与 `systemInstruction`，多模态输入也是 part。Interactions API 与 `generateContent` 的状态、字段和流式事件必须分别建模，不能把表中契约当成 Gemini 的统一接口。Gemini API 与 Vertex AI 在身份、区域、治理和 endpoint 上也可能不同，应分别配置。

## 可靠客户端

客户端至少记录 provider、model、API version、request id、usage、latency、重试和 finish reason。错误分为认证、配额、内容政策、无效请求、瞬时服务、超时和本地取消。不要对所有 4xx/5xx 自动重试。

密钥不进入 Prompt、日志和异常 repr。流式连接取消必须传播到服务端。产生工具调用时，provider adapter 只解析建议，真正权限与幂等由 Agent runtime 执行。
