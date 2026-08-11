# Gemini 家族

## 学习目标与证据边界

读完本章应能区分 Gemini 模型家族、Gemini API、Vertex AI 与终端产品；能够分别建模 Interactions API 和 `generateContent`，并设计多模态、长上下文、工具与文件输入的可证伪评测。

**先修知识**：多模态 token/embedding、HTTP/JSON、流式事件、工具调用、RAG 与云 IAM。

Gemini 同时出现在模型研究、开发者 API、Google Cloud 托管和消费产品中。产品界面中可见的能力不自动等于 Gemini API 或 Vertex AI 的公开契约；API 模型的内部参数量、训练数据、完整路由与后训练配方若未披露，应保持未知。

以下 API 状态于 **2026-08-06** 核对。型号、价格、窗口、媒体限制、区域、配额与保留策略都是时间敏感事实，部署时必须重新查看所选平台与 API 版本。

## 先选平台，再选模型

| 层 | 主要问题 | 不能自动外推 |
|---|---|---|
| Gemini 模型 | 文本、多模态、推理和工具能力 | 某平台一定开放全部能力 |
| Gemini API | 开发者接口、AI Studio 相关密钥与配额 | Vertex AI 的 IAM、区域和治理 |
| Vertex AI | Google Cloud 项目、IAM、区域、审计与企业治理 | Gemini API 的 endpoint 与默认值 |
| 终端产品 | 产品内交互、集成与 UI | 可编程 API、数据政策和配额 |

工程配置至少固定 `provider/platform`、API surface、model id、region、checked_at、SDK/API version 和数据保留选择。只写 `provider=gemini` 无法复现身份、端点或治理语义。

## 两套 API surface 必须分开

截至 2026-08-06，官方文档说明 Interactions API 已 GA，并推荐所有新项目使用；原有 `generateContent` 被标为 legacy，但仍受支持。本仓库当前 adapter 教学实现的是 `generateContent`，不是 Interactions API。

| 维度 | Interactions API | `generateContent` |
|---|---|---|
| 核心抽象 | Interaction resource 与可观察 steps | 针对 model 的内容生成请求/响应 |
| 多轮状态 | 可用 `previous_interaction_id` 选择服务端状态，也可无状态 | 客户端通常重发 `contents` 历史 |
| 长任务 | 支持独立的 background execution 语义 | 不应套用 Interactions 的后台状态机 |
| 输出 | model/tool/thought 等步骤按该接口建模 | `candidates`、content/parts、finish reason |
| 数据保留 | 默认存储与 `store=false` 是重要治理选择 | 按该接口及平台的数据政策单独核对 |
| 本仓库状态 | 尚无 adapter/真实端点验证 | 有离线 text-only 契约 adapter |

`previous_interaction_id` 只延续已保存的会话历史；工具、system instruction、generation config 等 interaction-scoped 参数应按本次请求重新设置。`store=false` 会改变可用的状态/后台能力，不能当作无影响的隐私开关。具体保留天数可能变化，本教材不固化数字，生产前应查看官方 data retention 页面。

迁移不能只替换 endpoint。要重新处理：状态所有权、step/event 类型、tool call/result、后台轮询或 webhook、取消、重试、幂等、存储/删除、usage 和错误 taxonomy。

## `generateContent` 契约

官方 GenerateContent reference 的请求核心是 `contents`：单轮是一项，多轮包含历史和最新输入；每项 Content 由 role 与 `parts` 组成。顶层 `systemInstruction`、`generationConfig`、tools 和 safety settings 属于各自字段。

教学用请求形状：

```json
{
  "systemInstruction": {
    "parts": [{"text": "仅依据给定证据回答"}]
  },
  "contents": [
    {
      "role": "user",
      "parts": [
        {"text": "解释图中趋势"},
        {"inlineData": {"mimeType": "image/png", "data": "<base64>"}}
      ]
    }
  ],
  "generationConfig": {
    "temperature": 0
  }
}
```

字段名和媒体传输方式应以实际 SDK/API 版本为准；示例只展示数据模型。响应解析至少保留：

- `candidates`，不能默认永远只有一个成功候选；
- 每个 candidate 的 content/parts；
- `finishReason` 与安全相关反馈；
- `usageMetadata`；
- model version、request id 或 provider 可用的追踪字段；
- 未识别 part 的原始内容。

若 prompt 被阻止，响应可能没有可用 candidate；若 candidate 只有 function call 或其他非 text part，text-only parser 也不应返回空字符串冒充成功。

## Parts 是统一多模态容器

`parts` 让文本、媒体、文件、函数调用与结果进入同一有类型序列。它的工程后果不是“可以传图片”这么简单，而是 parser、存储和安全边界都必须按类型设计：

```text
Content(role)
└── parts[]
    ├── text
    ├── inline/file media reference
    ├── function call / result
    └── provider-specific part
```

每个媒体 part 要校验 MIME、解码后大小、分辨率/时长、来源、租户 ACL、恶意文件和生命周期。不要只相信扩展名或客户端上报的 MIME。内联 base64 会放大请求体；大文件通常应使用受控上传/文件引用，并明确删除和版本策略。

多模态输入也是提示注入载体：图片中的小字、PDF 隐藏层、音频转写和视频字幕都可能包含指令。低信任媒体提取出的文字不能升级为 system instruction；工具执行仍由外部授权层控制。

## 怎样证明模型真的使用了目标模态

“这张图讲什么”容易被标题或常识猜中。高质量评测要构造反事实和遮蔽：

| 模态 | 核心任务 | 反作弊设计 |
|---|---|---|
| 图片 | OCR、小字、空间、计数、属性 | 去掉文件名/alt text，交换局部区域 |
| 图表 | 数值、单位、图例、趋势 | 改一个数据点或单位，保持文字描述相同 |
| 文档 | 布局、表格、页码、跨页引用 | 打乱页序，加入相似干扰段 |
| 视频 | 时间定位、动作顺序、状态变化 | 剪掉关键帧或交换两个片段 |
| 音频 | 转写、说话人、事件与时间 | 替换同文本不同说话人/情绪样本 |

报告 exact match/F1 之外，还要保存证据坐标、页码/时间戳、拒答、媒体解析失败和文本先验对照。视觉模型答对但引用错区域，不能算可审计成功。

## 长上下文、文件与缓存

长上下文要测位置、多跳、冲突、顺序、全局聚合和长输出一致性，不能只放一根 needle。文件/缓存可以减少重复传输或计算，但会引入：

- 文件 owner、租户与项目 ACL；
- 上传、处理、可用、失败、删除等生命周期；
- 内容 hash、版本与解析器版本；
- cache/file id 被错误跨用户复用；
- 原文件删除后派生缓存是否仍存在；
- 媒体 token、延迟与费用的独立计量。

RAG 仍有价值：它在送入模型前做权限过滤、版本选择、证据定位和输入压缩。文件 API 或超长窗口不等于检索系统，也不自动提供引用正确性。

## Structured output 与工具

结构化输出约束语法，不保证字段真实、单位正确或引用存在。function call 只是候选动作；runtime 仍要做 schema、业务规则、身份、ACL、预算、审批、幂等和审计。

Interactions 与 `generateContent` 的工具/step 表达应分别实现 provider adapter，再归一到内部 `ToolCall(id, name, arguments, raw)`。不要把一个 API 的流式事件或 call id 规则复制到另一个 API。工具结果属于低信任输入，尤其要防网页、地图、搜索和代码执行结果中的间接提示注入。

## 工程选型与迁移评测

模型和 API surface 一起评测：

1. 固定平台、区域、API/SDK 版本、model id 与 checked_at；
2. 固定 text/media case、system instruction、工具 schema 与输出预算；
3. 保存原始 part/step/event、usage、finish reason、安全反馈和延迟；
4. 分开统计 provider error、媒体处理失败、解析错误、内容错误和安全阻止；
5. 比较每成功任务成本，而不是只比较每 token 标价；
6. 检查存储、删除、日志、IAM 和跨区域数据流；
7. paired eval 后再 shadow/canary，并保留旧 adapter 以回滚。

Interactions 的服务端状态让客户端更轻，但也增加状态归属、删除、保留和重放问题。无状态 `store=false` 更易由应用掌控历史，却可能失去某些状态/后台能力；应按治理和产品需求选择，而不是只按代码行数。

## 可运行实验

本仓库 `about_llm.integrations.cloud_api` 对 `generateContent` 做离线 text-only 契约测试：assistant role 映射为 `model`、system 映射到 `systemInstruction`、响应提取 `usageMetadata`；`GeminiGenerateContentTextStream` 另检查 `streamGenerateContent` 单 candidate 的 text parts、usageMetadata、finishReason 与 EOF。它没有实现 Interactions API streaming、完整多模态/function parts、真实 streaming HTTP 或 Google 端点验证。

建议新增四组 fixture/实验：

1. **契约矩阵**：text、多 text part、无 candidate、多 candidate、function call、安全阻止、未知 part 与错误响应；
2. **API 迁移**：同一 task 分别经 Interactions 和 `generateContent` adapter，检查内部规范化结果与状态差异；
3. **多模态反事实**：原样、遮蔽、局部替换、文本先验四个版本成组评测；
4. **文件治理**：跨租户 id、过期/删除、版本更新和超大文件必须失败或走审批路径。

若运行真实 API，要把结果标为具体平台、区域、model id、API surface、checked_at 和账号配置下的验证；离线 fixture 通过不能证明真实配额、媒体限制或服务端状态行为。

## 常见错误

- 把 Gemini 产品、Gemini API 和 Vertex AI 当作同一个 endpoint；
- 因同属 Gemini 就混用 Interactions API 与 `generateContent` 的字段和事件；
- 迁移到 `previous_interaction_id` 后忘记重新发送 interaction-scoped 配置；
- 忽略默认存储与删除策略，直接处理敏感会话；
- 只解析第一个 candidate 的第一个 text part；
- 把无 candidate 的安全阻止解析成空答案成功；
- 信任媒体 MIME、文件 id 或图片中的文字指令；
- 只做主题识别，没有反事实证明模型使用了目标模态；
- 把长窗口/文件上传当作 RAG、ACL 和引用系统的替代品。

## 面试追问

1. Interactions API 与 `generateContent` 的状态和 adapter 边界为什么不能混写？
2. `previous_interaction_id` 带来哪些状态归属、删除和配置继承问题？
3. contents/parts 相比纯文本 messages 怎样改变存储、解析与安全设计？
4. 如何证明视觉问答不是从文件名、OCR 线索或文字先验猜出？
5. 多模态提示注入如何跨提取器、模型和工具执行层隔离？
6. Gemini API 与 Vertex AI 选型要看哪些非模型因素？
7. 文件缓存和长上下文为什么不能替代权限感知 RAG？

## 一手资料

- Google，[Interactions API](https://ai.google.dev/gemini-api/docs/interactions)，GA 状态、状态管理、steps、后台执行与存储边界；核对日期 2026-08-06。
- Google，[GenerateContent API reference](https://ai.google.dev/api/generate-content)，`contents`、`systemInstruction`、candidates 与 usage。
- Google，[Text generation](https://ai.google.dev/gemini-api/docs/text-generation)，当前文本生成入口与 API surface 指引。
- Google Cloud，[Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview)，Vertex AI 平台与治理入口。
- 目标 model page、SDK reference、data retention 与区域文档；生产部署时的最高优先级证据。
