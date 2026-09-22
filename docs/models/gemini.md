# Gemini：跟一次图片请求看懂平台、接口与多模态

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：第一次接触 Gemini，或正在比较 Gemini API 与 Vertex AI 的开发者。
- **先修**：理解 HTTP/JSON、流式输出和多模态输入的基本概念。
- **首次阅读**：一个图片任务 → 部署身份 → 两套 API → 有类型的输入输出 → 反事实评测。
- **完成信号**：能为同一任务选择平台与 API，并说明怎样证明模型确实使用了图片。
- **卡住时**：先忽略图片和工具，只保留一次同步文本请求。

</div>

**章节导航**：[Interactions API](gemini-interactions.md) · [generateContent 与多模态](gemini-generate-content.md) · [云 API 可靠性](cloud-api-reliability.md) · [证据台账](../evidence/gemini-controls.md)
{ .doc-nav }

“我们要接入 Gemini”还不是一个可执行的需求。Gemini 既是模型家族名，也出现在开发者 API、
Google Cloud 平台和终端产品中；同一个名字背后可能对应不同接口、身份、区域和数据政策。

本章从一个具体任务开始：

> 维修人员上传一张设备告警截图。系统需要读出错误码，指出截图中的证据位置，并生成一份维修工单建议。
> 工单只允许提交给当前租户名下的设备，高风险操作还要人工批准。

这不是一个单纯的“看图回答”问题。一次完整处理会经过：

```text
图片与问题
→ 文件、租户和大小检查
→ 选择平台、接口与模型
→ 映射成 Gemini 请求
→ 接收有类型的响应
→ 核对错误码和证据位置
→ 将工具调用当作建议
→ ACL 与审批
→ 执行或拒绝
```

沿这条路径学习，你会自然遇到 Gemini 接入中最容易混淆的四件事：模型能力、API 对象、应用权限和实验
证据。它们需要协作，却不能互相代替。

## 第一步：先写完整部署身份 {#deployment-identity}

假设团队说“Gemini 能看图”，这句话至少缺少下面四层信息：

- **Gemini 模型**决定文本、多模态、推理与工具能力。要核对目标模型是否接受图片，以及它有哪些限制。
- **Gemini API**提供开发者接口、密钥与配额。要明确使用 Interactions 还是 `generateContent`。
- **Vertex AI**带来 Google Cloud 项目、IAM、区域与审计。企业身份和数据治理可能要求走这条平台路径。
- **终端产品**面向最终用户。产品界面中可见的功能，未必以相同形式开放为可编程 API。

一份能复现请求的配置通常要写清：

```yaml
platform: gemini-api
api_surface: interactions
api_version: v1
endpoint_origin: https://generativelanguage.googleapis.com
model_id: deployment-owned-exact-id
region_or_location: platform-defined
storage_mode: explicit
checked_at: YYYY-MM-DD
```

这里只展示配置形状，不代表这些占位值已经通过真实调用。生产配置还应绑定账号层级、SDK 版本、能力探测
结果和评测工件。

只写 `provider: gemini` 会丢掉接口、端点、区域和治理语义。某个型号若是会漂移的别名，也应保存请求时的
型号、响应回报的版本字段和验证日期，而不是把它描述成不可变的开放权重 revision。

### 闭源模型不该靠猜架构来填空

Gemini 的公开能力不等于内部实现全部公开。除非目标版本的官方材料明确披露，否则参数量、训练数据、
注意力与缓存布局、多模态编码器、模型路由和后训练配方都应保持未知。

因此：

- 能读图片，只能证明目标接口在本次任务上产生了某种行为，不能据此命名内部视觉编码器；
- 返回 thinking 相关字段，不代表应用拿到了可解释的真实思维链；
- 型号名称相近，不代表权重、tokenizer 或服务端路由相同；
- 接受很长输入，不代表任务在窗口中每个位置都同样可靠。

闭源 API 更适合沿可观察边界学习：输入是什么、状态由谁保存、返回哪些类型、工具由谁执行、费用怎样计量，
以及应用怎样验证结果。

## 第二步：为同一任务选择 API {#choose-api}

当前官方文档推荐新项目使用 Interactions API；原有 `generateContent` 仍受支持，但被标为 legacy。
这是一项会随时间变化的产品状态，部署前仍要重新核对目标版本。[SOURCE:gemini-text-generation]

两套接口都可以参与图片任务，但对象模型不同。

**Interactions API** 把一次执行表示为可查询的 Interaction resource，并用有序 steps 展示过程。

历史既可以由 `previous_interaction_id` 在服务端续接，也可以改为客户端维护。长任务还有独立的后台执行与
状态查询语义。流式输出按交互对象、步骤和有类型的增量事件组织。本仓库有一个固定 SSE 的离线回放器，
它只检查受限的生命周期与投影，不创建 Interaction 请求，也不构成完整的 Interactions adapter。

**`generateContent`** 把一次执行表示为生成请求及其 candidates。多轮对话通常由客户端重新发送
`contents`，也不使用 Interactions 的后台状态机。它的流式输出围绕 candidate、part、finish reason 和
usage 展开。本仓库已经为纯文本子集实现离线适配器和状态机。

如果告警截图只需一次同步识别，现有 `generateContent` 接入可以继续维护。若新系统还需要服务端会话、可观察
工具步骤或后台任务，Interactions 通常更符合当前官方推荐方向。

选择 Interactions 后，`previous_interaction_id` 只延续已保存的对话输入输出。工具、系统指令和生成配置等
本轮参数仍要重新发送。设置 `store=false` 会改变可用的历史续接与后台能力，因此它是一项状态和治理选择，
不是一个毫无副作用的隐私开关。[SOURCE:gemini-interactions-overview]

迁移时也不能只替换 URL。应用需要重新处理状态所有权、事件类型、工具调用与结果、后台查询、取消、重试、
用量和删除流程。两套协议可以归一到同一个业务任务，却不能共用一份猜测当前协议的解析器。

## 第三步：先建立业务对象，再映射到接口 {#canonical-task}

业务层不必直接携带供应商字段。告警截图任务可以先写成一个窄的内部对象：

```text
MaintenanceDiagnosis
├── tenant / subject / device identity
├── text question
├── image bytes / MIME / digest / source
├── expected output schema
├── allowed tool proposals
├── output and cost budget
└── trace / retention policy
```

适配器再把它映射到目标接口：

```text
MaintenanceDiagnosis
├── Interactions adapter
│   └── interaction input + steps + optional previous id
└── generateContent adapter
    └── contents/parts + systemInstruction + generationConfig
```

这样做的目的不是把两套 API 强行变成一样，而是把租户、设备、预算和输出契约留在应用层。供应商特有的
step、part、状态和追踪标识仍应保真保存。

### 图片不是一段更长的 prompt

以 `generateContent` 为例，文本和图片进入同一个有类型的 `parts` 序列：

```json
{
  "systemInstruction": {
    "parts": [{"text": "只报告截图中可见的错误码和证据位置"}]
  },
  "contents": [
    {
      "role": "user",
      "parts": [
        {"text": "识别错误码，并生成维修建议"},
        {"inlineData": {"mimeType": "image/png", "data": "<base64>"}}
      ]
    }
  ],
  "generationConfig": {"temperature": 0}
}
```

这段 JSON 只解释对象关系，不是本仓库已经发往 Google 的请求。实际字段、媒体上限和传输方式要以目标
API、SDK 和模型版本为准。

应用在发送前仍要验证图片的真实类型、解码后大小、分辨率、来源、租户权限和恶意内容。大文件通常更适合
受控上传或文件引用；内联 base64 会放大请求体，也更容易进入不该出现的日志。

图片中的文字同样可能包含提示注入。OCR 或视觉模型读出的指令只能作为低信任数据，不能覆盖系统规则，
更不能绕过工单系统的 ACL 和审批。

## 第四步：解析有类型的响应，而不是只取一段文字 {#typed-response}

`generateContent` 的响应可能包含多个候选，每个候选又包含多个 parts。安全阻止还可能表现为没有可用
candidate，并在 `promptFeedback` 中提供信息。

```text
GenerateContentResponse
├── candidates[]
│   ├── content.parts[]
│   ├── finishReason
│   └── safetyRatings
├── promptFeedback
├── usageMetadata
├── modelVersion
└── responseId
```

因此，以下结果必须分开：

- 正常文本；
- 工具调用或其他非文本 part；
- 安全阻止；
- 不完整或异常终止；
- 解析器不认识的新类型；
- 传输在协议终态前断开。

Interactions 同样不能只保存 `output_text`。这个便捷属性是有损文本视图；更早的文字如果被 thought、图片、
音频或工具步骤分隔，可能不会出现在该属性中。需要重放、审计或无状态续聊时，应保存允许保留的有类型步骤
及其顺序。

### 仓库当前真正运行了什么

下面的命令会构造三家供应商的固定文本请求，并解析仓库准备的响应：

```powershell
python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl

python projects/cloud-api-contracts/gemini_interactions_replay.py
```

Gemini 这一项会检查：

- system 文本映射到 `systemInstruction`；
- assistant role 映射为 `model`；
- 多个 text parts 合并成文本；
- `usageMetadata` 映射为输入、输出 token 数；
- `finishReason` 和 `modelVersion` 被读取。

第一条命令的 `network_performed` 和 `real_credentials_used` 都是 `false`。`generateContent` 解析器只接受一个
candidate 的纯文本 parts；遇到工具或媒体 part 会拒绝。第二条命令回放 Interactions 的固定函数调用事件，
重建 `lookup_weather(city="上海")`，并展示为什么 `interaction.completed` event 仍可能对应
`requires_action` status。

这两条路径都只证明本地协议处理。它们没有运行图片任务、Google SDK 或真实端点；Interactions 回放也没有发送
请求或执行天气工具。

## 第五步：用反事实证明模型真的看了图片 {#multimodal-evaluation}

如果截图文件名叫 `E17_alarm.png`，模型可能完全不看图片就猜中错误码。单个答对样本无法区分视觉理解和文字
先验。

为同一告警样本准备一组只改变关键证据的版本：

| 版本 | 改动 | 想回答的问题 |
|---|---|---|
| 原图 | 保留告警面板与中性文件名 | 主任务能否完成？ |
| 无图 | 移除图片，只保留问题 | 文字先验能猜到多少？ |
| 遮蔽 | 盖住错误码区域 | 答案是否依赖该区域？ |
| 替换 | 把 E-17 改成 E-42 | 输出会不会跟证据改变？ |
| 误导文本 | 文件名或标题暗示另一个错误码 | 模型会信图片还是信提示？ |
| 损坏文件 | 截断内容或伪造 MIME | 系统能否明确拒绝？ |

本例至少记录错误码准确率、证据框位置、无证据时的拒答、工具建议是否合法，以及每种失败发生在哪一层。
如果答案随局部替换而改变、遮蔽后正确拒答，才有更强理由认为目标模态产生了因果影响。

这种实验仍只说明指定模型、接口和样本上的行为。它不能反推出 Gemini 的内部视觉编码器结构。

## 文件、长上下文和 RAG 分别解决什么 {#files-context-rag}

真实维修任务可能上传整本手册，而不是一张截图。此时容易把四种机制混在一起：

- **文件 API** 管理上传、处理、引用和删除生命周期；
- **缓存** 减少某些重复处理或计费；
- **服务端会话状态** 续接历史交互；
- **RAG** 在发送前做授权过滤、版本选择、检索和证据定位。

长窗口或文件引用不会自动提供租户 ACL、最新手册版本和正确引用。RAG 也不负责供应商会话的删除与计费。
工程设计应先写清每种机制的所有者、版本和失效方式，再决定是否组合。

长上下文评测也不应只问“接口接受多少 token”。至少分开：文档声明上限、账号实际接受上限，以及任务在
头部、中部、尾部和冲突证据下仍然有效的长度。

## 模型、SDK 和应用运行时各负责什么 {#dependency-stack}

| 层次 | 在告警任务中负责什么 |
|---|---|
| Gemini 模型 | 根据输入产生文本、媒体或工具建议；内部结构以公开材料为准 |
| Gemini API / Vertex AI | 暴露模型、状态、文件、配额、用量和平台治理能力 |
| Google Gen AI SDK 或 HTTP 客户端 | 把程序对象序列化为目标接口请求，并处理传输 |
| 应用适配器 | 保真解析 part/step、状态、错误和用量，再归一为内部对象 |
| 业务运行时 | 执行 ACL、schema、预算、审批、幂等、验证和审计 |
| 本地模型或单卡 GPU | 可做 OCR、脱敏、检索或验证；它不是 Gemini 服务本身 |

结构化输出只能约束受支持的语法。字段是否真实、设备是否属于当前租户、引用区域是否存在，仍要由业务
运行时验证。模型产生的 function call 也只是候选动作；通过权限、审批和幂等检查后，工具执行器才可以
产生真实副作用。

## Gemini 特有的生产接入差异 {#gemini-production-differences}

通用重试、预算、发布和回滚不再在 Gemini 页面复制；它们分别由
[云 API 可靠性](cloud-api-reliability.md)和[发布、观测与回滚](../applications/llmops-release.md)负责。
接入 Gemini 时仍需保留下面这些供应商与接口差异。

### 冻结四组部署身份 {#production-identity}

- **产品平台**：Gemini API 或 Google Cloud 托管入口；它影响认证、区域和数据治理。
- **API surface**：Interactions 或 `generateContent` 及版本；它决定对象、状态和流式事件。
- **模型**：请求型号、别名性质、响应版本字段和供应商请求标识。
- **运行环境**：SDK、区域或 location、账号层级、存储选择和核对日期。

```yaml
platform: gemini-api
api_surface: interactions
api_version: v1
endpoint_origin: https://generativelanguage.googleapis.com
model_id: deployment-owned-exact-id
region_or_location: platform-defined
account_tier: deployment-owned
storage_mode: explicit
sdk_version: pinned-by-deployment
checked_at: YYYY-MM-DD
```

若型号是可漂移别名，保存精确请求字符串、响应版本字段、请求标识和检查日期；这些信息帮助追查变化，
但不能自行认证供应商内部权重。

### 同一个业务对象使用两套适配器

```text
CanonicalTask
├── tenant / subject / device
├── text + image digest
├── system policy / expected schema
├── allowed tool proposals
└── generation / cost / retention policy
    ├── Interactions adapter → input、steps、previous_interaction_id
    └── generateContent adapter → contents/parts、systemInstruction、generationConfig
```

两套适配器可以共享图片校验、租户身份和内部结果类型，不能静默丢掉 step、part、终态、用量或未知扩展字段。
纯文本便捷属性只能作为明确标注的有损视图。

### 分项探测能力，不用一次大请求猜兼容性

| 探测项 | 成功时保存什么 | 失败时怎样处理 |
|---|---|---|
| 同步纯文本 | 请求、响应、型号和用量 | 停止并检查身份 |
| 图片输入 | MIME、大小、固定图片与有类型响应 | 标记不支持，不能静默删图 |
| 流式文本 | 事件顺序、协议终态与传输结束 | 使用对应接口解析器 |
| 工具建议 | 调用标识、参数和停止状态 | 关闭该能力或阻止发布 |
| 结构化输出 | 实际 schema 与响应 | 进入预设降级 |
| 状态/后台任务 | 创建、查询、取消与保留行为 | 不借用另一接口状态机 |

图片发送前还要从可信会话解析租户、用户和设备；按实际字节核对 MIME、大小与解析结果；隔离不需要的
metadata；给 OCR 与图中文字标记低信任来源；明确原图、派生文本、文件和缓存的删除路径。

### 分开传输、协议和业务终态 {#production-terminals}

```text
传输层：HTTP body 或 SSE 连接结束
协议层：Interaction status，或 candidate finish reason
业务层：错误码、证据位置与工具建议通过验证
```

Interactions 流按 interaction 与 step 事件推进；`streamGenerateContent` 按 candidate、content parts、
finish reason 和 usage 推进。二者可以共享 SSE 解码，上层状态机必须分开。连接结束而没有协议终态时，
结果可能截断；供应商终态完成也不等于业务验证成功。

告警截图任务发布前还要检查：错误码是否落在授权图片证据区域，设备是否属于当前租户，工具 proposal
是否通过 schema、权限、审批和幂等校验，以及最终文字是否与已验证的真实副作用一致。

### 真实账号的最小冒烟 {#real-account-smoke}

取得合法账号、权限和预算后，先固定平台、API、型号、区域与存储选择，再用最小权限 secret 运行同步纯文本。
保存脱敏请求、响应、标识、终态和用量；验证错误、超时、credential 脱敏和预算结算后，才逐项加入流式、
状态、工具与图片。最后清理实验创建的 interaction、文件和缓存，并与控制台及账单导出对账。

一次冒烟只覆盖当时的账号、区域、接口、型号和输入。仓库目前只有固定回放与供应商无关的外围控制；
真实图片、工具、后台任务、长期可靠性和质量仍需在目标环境验证，具体边界见[Gemini 证据台账](../evidence/gemini-controls.md)。

## 下一步怎样读

- 接入有状态对话或长任务：进入 [Interactions API](gemini-interactions.md)，先回答谁保存历史、step 和终态
  怎样推进。
- 处理候选、parts 或媒体输入：进入 [`generateContent` 与多模态](gemini-generate-content.md)，检查解析器
  会丢哪些类型，并设计反事实图片。
- 准备真实发布：用本页的接口差异冻结身份，再进入[云 API 可靠性](cloud-api-reliability.md)和
  [发布、观测与回滚](../applications/llmops-release.md)补齐重试、预算、分母与回滚。
- 核对实现或简历表述：查看[证据台账](../evidence/gemini-controls.md)，确认当前代码实际运行过什么。

第一次学习不必顺序读完所有页面。先用本页的告警截图建立心智模型，再沿自己的任务进入一条路径。

## 自测

1. 为什么只写 `provider: gemini` 仍无法复现一次请求？
2. 告警截图任务在什么情况下更适合 Interactions，而不是继续维护 `generateContent`？
3. 为什么空候选、安全阻止和空文本是三种不同结果？
4. 怎样用局部替换证明模型读取了错误码区域？
5. 文件 API、服务端状态、缓存和 RAG 各自负责什么？
