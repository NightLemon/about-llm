# Gemini 家族

## API 版本边界

截至 2026-08-05，Gemini API 官方文本生成指南推荐新项目使用已 GA 的 Interactions API，以获取最新模型与能力；本仓库实现的是仍需单独理解的 `generateContent` 请求/响应契约。两者的状态管理、字段和流式事件不能混写。下面“generateContent 契约”只描述该接口，不代表 Gemini 当前唯一或首选 API。

## 产品与平台边界

Gemini 涵盖 Google 的多模态模型/API/产品。面向开发者的 Gemini API、Vertex AI 托管和终端产品在身份、区域、治理、配额与功能上不同。选择时先明确使用哪一平台，不把产品界面能力自动等同于 API。

## generateContent 契约

- 对话放在 contents；
- role 常用 user/model；
- 每条内容包含 parts；
- system 使用 systemInstruction；
- 图片、音频、视频或文件也是不同 part；
- generationConfig 控制输出；
- usageMetadata 与 finishReason 进入观测。

多模态 part 可能是 inline data、文件引用或文本；必须校验 MIME、大小、来源和访问权限。

## 多模态

评测不能只问图片主题。按模态测：

- OCR 字符/字段和小字；
- 空间关系与计数；
- 图表数值和单位；
- 文档布局、表格和页码引用；
- 视频时间定位和顺序；
- 音频转写、说话人和事件；
- 文本线索遮蔽，确认模型确实使用目标模态。

图像中的文字也可能是间接提示注入。

## 长上下文与文件

文件上传/缓存降低重复传输，但要管理生命周期、ACL、删除和版本。标称长上下文仍需测试多跳、全局聚合和位置鲁棒性。大文件 token/媒体成本与延迟单独测量。

## 工程选型

比较结构化输出、function calling、媒体限制、streaming、缓存、配额、区域和数据政策。Vertex AI 适合需要云 IAM/区域治理的场景；具体能力以所选 API 版本为准。

## 面试追问

1. Gemini 的 parts 与纯文本 messages 有何架构影响？
2. 怎样证明视觉问答不是从文字先验猜出？
3. 多模态提示注入怎样隔离？
4. 文件缓存的 ACL 与失效如何设计？
5. Gemini API 与 Vertex AI 选型看哪些非模型因素？
