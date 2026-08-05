# GPT、DeepSeek、Qwen、Claude、Gemini 云 API 契约

目标：统一业务侧 ChatMessage 与 ChatResponse，但显式保留供应商协议差异，不用一个看似通用的 SDK 隐藏 system、usage、finish reason 和错误语义。

## 三类协议

- OpenAI-compatible：可用于按配置接入 GPT，以及提供兼容端点的 DeepSeek/Qwen 服务；
- Anthropic Messages：system 位于顶层，消息与 usage 字段独立；
- Gemini `generateContent`：user/model role、parts、systemInstruction 和 usageMetadata；本 adapter 用于教学与兼容性。截至 2026-08-05，新项目应同时评估官方已 GA、面向最新能力推荐的 Interactions API，不能混用两套接口的字段与流式事件。

代码只构建请求并解析响应，不在 import 时读取密钥或访问网络。RequestSpec 的 headers 不参与 repr，sanitized_headers 会遮蔽认证值。本项目的 ChatResponse 是 text-only 最小契约：tool-call-only 或无文本响应会明确失败；生产 adapter 若支持工具、引用、thinking 或媒体 block，应保留并分别建模，不能强转成字符串。

## 生产调用层仍需补充

真实客户端应实现：

- connect/read/overall timeout；
- 只对明确瞬时错误重试，使用指数退避和 jitter；
- 429/配额与 provider request id；
- 幂等语义，尤其工具副作用；
- 流式取消；
- token/费用预算；
- 原始错误分类和脱敏 trace；
- provider/model/revision/checked_at 版本记录。

## 配置原则

base URL、model id、API version 和密钥来自配置/秘密管理，不写死在教材。DeepSeek/Qwen 的云产品与开放权重 checkpoint 不能假定相同。Claude/Gemini 内部架构未公开部分保持未知。

## 离线测试

tests/test_cloud_api.py 用固定响应验证三类字段映射和密钥脱敏，不产生任何网络请求或费用。网络 smoke test 必须显式标记 network，并设置请求数与 token 上限。
