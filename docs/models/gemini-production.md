# Gemini 生产接入：跟一次请求走到发布与回滚

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备把 Gemini 接入真实业务，并建立发布门禁和回滚方案的工程师。
- **先修**：读过 [Gemini 总览](gemini.md)，并已经选择 Interactions 或 `generateContent`。
- **首次阅读**：发送前检查 → 预算预留 → 一次调用 → 结果验证 → 费用结算 → 灰度与回滚。
- **完成信号**：能解释请求超时后为什么不一定能安全重试，并能画出一次任务的费用和结果账本。
- **卡住时**：先只看同步纯文本请求，再逐项加入流式、工具、状态或图片。

</div>

**章节导航**：[总览](gemini.md) · [Interactions API](gemini-interactions.md) · [generateContent 与多模态](gemini-generate-content.md) · [证据台账](../evidence/gemini-controls.md)
{ .doc-nav }

总览中的任务是：维修人员上传设备告警截图，系统识别错误码、指出证据位置，并生成维修工单建议。

在本地演示里，只要打印出一段答案就像成功了。生产系统还要知道图片属于谁、这次调用用了哪个接口和型号、
超时后供应商是否已经执行、费用该记多少，以及工单建议是否真的获准执行。

把一次任务展开，会得到下面的生命周期：

```text
接收图片与问题
→ 固定身份和业务请求
→ 检查权限、能力与输入
→ 预留本次调用的预算
→ 发起一次供应商调用
→ 接收协议终态或记录未知结果
→ 解析有类型的输出
→ 验证错误码、证据和工具建议
→ 结算本次调用
→ 发布任务结果
→ 持续观测，必要时回滚
```

生产质量来自这条链路能被复现、对账和恢复，而不是某一次请求碰巧返回了正确文字。

## 发送前：冻结这次调用的身份 {#freeze-identity}

同一个 Gemini 型号可能通过不同平台和接口访问。先把一次调用拆成四组身份：

- **产品平台**：记录 Gemini API 或 Google Cloud 托管入口。平台漂移会改变认证、区域和数据治理。
- **API**：记录 Interactions 或 `generateContent` 及其版本。接口漂移会改变请求对象、状态和流式事件。
- **模型**：记录请求型号、别名性质和响应版本字段。型号漂移会改变能力、默认值或下线策略。
- **运行环境**：记录 SDK、区域、账号层级和检查日期。环境漂移会改变字段、配额、保留或错误语义。

可以把它们写进部署配置：

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

这仍只是配置形状。真实发布还要附上能力探测、评测报告和回滚决定。

闭源 API 往往无法像开放权重仓库那样固定 commit。若请求型号是可漂移别名，应保存请求时的精确字符串、
响应回报的版本字段、供应商请求标识和检查日期。它们能帮助追查变化，却不能自行认证供应商内部权重。

### 同一个业务对象，两套发送格式

业务层先形成一份稳定任务：

```text
CanonicalTask
├── tenant / subject / device
├── text + image digest
├── system policy
├── expected output schema
├── allowed tool proposals
├── generation and cost budget
└── trace / retention policy
```

随后才由接口适配器生成供应商请求：

```text
CanonicalTask
├── Interactions adapter
│   └── interaction input、steps、previous_interaction_id
└── generateContent adapter
    └── contents/parts、systemInstruction、generationConfig
```

重试时应复用已经冻结的业务对象和身份。若每次调用都重新读取可变的系统提示、工具列表或型号别名，同一个
逻辑任务可能在第二次尝试中悄悄换了语义。

两套适配器可以共享图片校验、租户身份和内部结果类型，但不能静默丢掉供应商特有的 step、part、终态、
用量或未知扩展字段。方便的纯文本属性只适合做显式标注的有损视图。

## 发送前：确认目标环境真的支持所需能力 {#capability-preflight}

告警任务需要图片输入、结构化结果和工具建议。平台、API 版本、型号或账号层级不同，能力也可能不同。

发布前用最小请求逐项探测，而不是把全部能力塞进一次大请求：

| 探测项 | 成功时保存什么 | 失败时怎样处理 |
|---|---|---|
| 同步纯文本 | 请求、响应、型号和用量 | 停止发布并检查身份 |
| 图片输入 | MIME、大小、固定图片与有类型响应 | 标记不支持，不能静默删图 |
| 流式文本 | 事件顺序、协议终态与传输结束 | 使用对应接口的独立解析器 |
| 工具建议 | 调用标识、参数和停止状态 | 工具能力关闭或阻止发布 |
| 结构化输出 | 实际发送的 schema 与响应 | 走预先设计的降级路径 |
| 状态与后台任务 | 创建、查询、取消和保留行为 | 不借用另一接口的状态机 |

探测结果要绑定平台、API、型号、区域、账号和日期。一个页面写着“支持图片”，并不能证明当前账号、区域和
请求形状已经接受这张图。

安全配置也属于能力探测。官方概览、参考页和不同 API 版本可能出现表述或字段差异。生产代码应固定目标
版本，保存实际发送的请求和接受或拒绝的响应，再用目标型号重跑正常任务与过度拒答评测。

供应商过滤器只是纵深防御的一层。图片安全、设备 ACL、工单审批、输出政策和事件响应仍由应用负责。

## 发送前：先检查图片和业务权限 {#input-preflight}

维修截图进入模型前，至少完成：

1. 从可信会话解析租户、用户和设备，而不是相信请求体自报字段。
2. 检查用户是否有权读取该设备并创建对应类型的工单。
3. 根据 magic bytes 解码并核对 MIME、大小、分辨率和页数。
4. 去除或隔离 EXIF、文件名、隐藏层等不需要的元数据。
5. 为实际字节、解析器版本和业务请求生成稳定摘要。
6. 给 OCR 和图中文字标记低信任来源，防止其覆盖系统指令。
7. 确定原图、派生文本、供应商文件和缓存的保留与删除策略。

通过这些检查只表示“允许把这份输入交给目标流程”，不表示模型答案必然正确。

## 发送前：预留一次尝试可能花掉的预算 {#reserve-budget}

逻辑任务可能因为重试产生多次供应商调用。预算账本应以 **attempt（调用尝试）** 为单位，而不是等最终成功
后只记一笔。

```text
冻结的请求身份
→ 保守估算输入 + 最大输出
→ 发送前原子预留
→ 每次尝试各有一笔 reservation
→ 结算 | 明确未发送后取消 | 结果未知时保留
→ 与供应商账单对账
```

仓库有一组与供应商无关的固定算术：

- 预计输入 60 token，最大输出 10 token；
- 样例单价为输入每百万 1 美元、输出每百万 2 美元；
- 发送前预留 80 micro-USD；
- 固定响应回报实际输入 58、输出 4；
- 按样例价格结算 66 micro-USD。

计算过程是：

```text
预留 = 60 × 1 + 10 × 2 = 80 micro-USD
结算 = 58 × 1 +  4 × 2 = 66 micro-USD
```

这组数字验证的是整数算术和本地账本。

真实成本需要另外保存价格快照。快照要绑定平台、型号与模态，也要写清缓存、thinking、工具和服务层级的
计价方式；区域和生效时间同样不可省略。供应商用量则来自目标接口的真实响应。

供应商响应中的用量适合做近实时控制，最终仍要与账单导出对账。若图片、缓存或工具有独立计量项，就保存
原始分项，不要强行压成“输入 token + 输出 token”两个数字。

固定样例和代码入口见[证据台账](../evidence/gemini-controls.md#budget-control)。

## 发送后：先判断结果是否已知，再决定重试 {#outcome-before-retry}

一次调用失败后，按顺序问三个问题：

1. 协议和错误策略是否允许重试？
2. 同一个业务动作是否可以安全重复？
3. 能否证明供应商没有接受或执行前一次请求？

| 场景 | 当前知道什么 | 默认处理 |
|---|---|---|
| 本地检查失败 | 请求没有发出 | 修正配置，可取消预留 |
| 连接前明确失败 | 根据传输证据判断大概率未发送 | 在限额内重试 |
| 发送后超时或连接重置 | 供应商结果未知 | 保留预算，不自动重放副作用任务 |
| 收到明确协议错误 | 有供应商终态和错误信息 | 按固定允许列表决定 |
| 已向用户发布部分流 | 外部已经看见部分结果 | 通常不自动重放 |
| 已取得后台任务标识 | 可以查询原任务 | 先查询和对账，不创建新任务 |

流连接断开，只说明客户端不再收到事件。供应商是否继续运行，需要取消接口、状态查询或合同语义来确认。
客户端发出取消请求也不等于供应商已经停止；即使最终停止，已经发生的计算和费用仍可能存在。

工单创建尤其不能因为“模型调用超时”就再次执行。模型生成的是建议，真正的业务副作用还要使用独立的
幂等标识、执行记录和查询接口。

### 每次重试都要单独占用预算

仓库另一个固定样例先遇到 500，再收到 200。两次调用各自预留和结算，最终合计 146 micro-USD；若任务硬
上限为 140，第二次尝试会在发送前被阻止。

重试策略和预算策略需要共同决定下一步。第二次尝试只有在协议允许、剩余预算充足且业务副作用可安全重放时
才能继续。

## 接收响应：区分传输结束、协议终态和业务成功 {#three-terminals}

告警任务至少有三层终点：

```text
传输层：HTTP body 或 SSE 连接结束
协议层：Interaction 状态，或 candidate 的 finish reason
业务层：错误码、证据位置和工单建议通过验证
```

只有连接结束，没有协议终态，可能是截断。供应商报告 `completed` 或 `STOP`，也只表示接口生命周期结束；
图片是否读对、schema 是否有效、设备是否获授权，还需要应用验证。

Interactions 流按照交互对象与步骤事件推进。`streamGenerateContent` 则以候选、内容片段、结束原因和用量
推进。两者可以共享底层 SSE 解码，上层状态机仍要分开。

完整解析器至少保存：

- 原始响应或经过允许列表筛选的有类型投影；
- interaction、response、step、candidate 和工具调用标识；
- part/step 的类型、顺序和未知类型处理结果；
- 协议终态、传输终点和时间戳；
- 型号版本、用量和安全反馈；
- 解析器版本与有损投影标记。

## 验证结果：从模型输出走到业务决定 {#verify-result}

假设模型返回错误码 E-17、一个矩形证据框和“创建维修工单”的工具建议。发布前依次检查：

1. 输出能否按预期 schema 解析，是否存在重复字段、未知字段或非法数值。
2. 错误码是否真的出现在授权图片的证据框内。
3. 设备标识是否由服务端解析，并属于当前租户。
4. 工具名称、schema 版本、参数和业务范围是否允许。
5. 该动作是否需要人工审批，审批是否绑定相同参数与设备版本。
6. 工具执行是否带幂等标识，返回结果是否通过独立查询验证。
7. 最终面向用户的文字是否与实际执行结果一致。

模型的 function call 是候选动作，不是授权。工具返回成功字符串也不是副作用已经发生的充分证据。工单
系统的记录和查询结果才是业务层的依据。

如果图片被阻止、没有候选、只返回工具 part、证据框越界或 schema 无效，应记录具体失败类型。把这些情况
都变成空字符串，会让评测和运营误以为请求成功但模型回答为空。

## 完成一次任务：同时结算费用和结果 {#task-result}

生产评测的最小单位不是一段文本，而是一次完整任务。可以保存：

```text
TaskResult
├── task / case / slice identity
├── frozen deployment and request identity
├── all provider attempts
├── transport / protocol / parse outcomes
├── image evidence verification
├── tool proposal / approval / effect outcome
├── final publish decision
├── latency timestamps
├── usage / reservation / settlement
└── retained evidence artifacts
```

有了这份对象，团队才能分别报告：

- 供应商调用成功率；
- 协议与 schema 解析成功率；
- 错误码、证据框和拒答质量；
- 工具建议、授权与实际副作用成功率；
- 尝试任务和成功任务的延迟与成本；
- 安全违规、过度拒答和未知结果；
- 后台任务的完成、取消和对账情况。

只在最终有文本的样本上计算质量，会隐藏安全阻止、超时、解析错误和工具失败。延迟也应同时报告所有收到的
任务与成功任务条件下的分布，不能只保留最快的成功样本。

## 可观测性：记录身份和状态，减少记录内容 {#observability}

告警截图可能包含设备编号、地理位置或内部信息。默认 trace 记录元数据，而不是原始内容：

- 内部任务与 attempt 标识；
- 供应商 interaction 或 response 标识；
- 平台、API、型号、区域和账号层级；
- 请求模板、工具 schema、图片字节和策略的受控摘要；
- part/step 类型与大小；
- 生命周期时间戳、终态和错误分类；
- 用量分项与预算状态；
- 存储、文件和缓存策略；
- 验证与发布决定；
- 脱敏和投影版本。

API key、原始图片、敏感 prompt、工具 secret、thought/signature 和可跨租户复用的文件标识默认不进入普通
日志。

普通 hash 也不是匿名化。低熵设备编号、文件名和短标识可能被枚举，应结合 keyed fingerprint、权限控制、
最短保留期和访问审计。

## 发布：先比较任务，再逐步扩大流量 {#rollout}

接口或型号迁移前，用同一组告警任务做配对比较。至少固定：

- 数据集、切分和关键风险样本；
- 平台、区域和账号；
- 型号与生成预算；
- 系统规则、工具 schema 和媒体字节；
- 超时、重试和预算策略；
- 状态、文件、缓存与保留选择；
- 评价规则和人工复核流程。

若两套 API 无法保持某项条件，例如 Interactions 引入服务端状态，就把它记录为处理差异。不要把不相同的
实验包装成“只替换了接口”。

推荐按下面顺序发布：

```text
离线契约测试
→ 受限真实冒烟
→ 影子流量配对
→ 小比例金丝雀
→ 分阶段扩大
→ 稳态门禁
```

每一步都要预先写明通过阈值、观察时长和停止条件。关键分母包括任务成功、安全、证据正确、工具副作用、
未知结果、延迟和每成功任务成本。

## 回滚：恢复整个协议，而不只是型号 {#rollback}

回滚工件至少包含：

- 新旧适配器与 API 版本；
- 新旧型号和能力探测记录；
- 状态由谁保存，以及旧会话怎样处理；
- 已创建的 interaction、文件和缓存怎样查询或删除；
- 工具调用与结果能否被旧版本理解；
- 预算 reservation 和未知 attempt 怎样对账；
- 日志、脱敏和解析器版本；
- 回滚触发器、负责人和事件手册。

如果只把 `model_id` 切回旧值，新的服务端状态、文件引用或工具结果可能继续留在系统里。真正的回滚要恢复
API 对象、状态所有权、数据政策和解析行为。

## 故障定位：先找层级，再评价模型 {#troubleshooting}

```text
失败
├── 身份与发送前检查
│   ├── 平台 / API / 版本 / 型号
│   └── 认证 / 区域 / 能力 / 图片
├── 传输
│   ├── 连接 / TLS / 超时
│   └── HTTP / SSE framing
├── 供应商协议
│   ├── Interaction step / status
│   └── candidate / part / finish
├── 应用
│   ├── 解析 / schema / 状态
│   ├── 图片证据 / 质量 / 安全
│   └── 工具 / ACL / 副作用
└── 经济与治理
    ├── 用量 / 预算 / 账单
    └── 存储 / 删除 / 日志
```

先确认失败位于哪一层。端点配错、图片解码失败或解析器丢失非文本 part，都可能表面上像“模型能力下降”。

## 单张消费级 GPU 能参与哪里

Gemini 闭源 API 本身不在本地 3070 等消费级 GPU 上运行。本地显卡仍可以承担：

- OCR、ASR 或媒体预处理基线；
- 图片去敏与恶意内容检查；
- embedding、reranker 和权限感知 RAG；
- 小模型 fallback；
- 输出 schema、引用或事实验证；
- 反事实图片生成与离线评测。

这些组件的显存和吞吐属于本地流程，不能归因给 Gemini。反过来，远端 API 的延迟和用量也不能解释本地
GPU 性能。

## 取得真实账号后怎样做第一次冒烟 {#real-smoke}

只有在拥有合法账号、权限和预算后，才运行真实调用：

1. 固定平台、API 版本、型号、区域和存储选择。
2. 使用最小权限 secret，并确保它不写入仓库和日志。
3. 设置请求数、输出、费用、图片大小和总时限上限。
4. 先运行同步、纯文本、无工具任务。
5. 保存脱敏后的请求、响应、标识、终态和用量。
6. 验证错误、超时、credential 脱敏和预算结算。
7. 再分别加入流式、状态、工具和图片，每次只增加一种复杂度。
8. 删除实验创建的 interaction、文件和缓存，并记录 API 返回结果。
9. 与控制台和账单导出对账。

一次冒烟只能说明该时刻、账号、区域、API、型号和输入组合的行为。代表性质量、容量和长期可靠性仍需要
成组评测、压测和持续观测。

## 仓库当前能证明到哪里 {#repository-evidence}

仓库当前为 `generateContent` 的纯文本子集提供离线请求构造、响应解析和单候选流式状态机。另一个固定回放
展示 Interactions 函数调用：流已经结束，资源仍可能等待客户端动作。供应商无关的 HTTP、重试和预算样例则
覆盖调用外围的工程状态。

这些实验都在本地使用固定数据。它们没有连接 Google Gen AI SDK、真实账号与端点，也没有运行图片、工具、文件
或缓存。后台任务和断线恢复同样需要在目标环境验证。

因此，本章中的告警任务是一条生产设计主线，不是已经录制的 Gemini 多模态调用。

可运行的第一步是：

```powershell
python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl

python projects/cloud-api-contracts/gemini_interactions_replay.py

python projects/cloud-api-contracts/usage_budget_toy.py
```

运行后先确认报告中的 `network_performed: false`，再阅读[证据台账](../evidence/gemini-controls.md)里的字段与
适用范围。

## 面试时怎样回答

如果面试官问“怎样把 Gemini 接入生产”，沿一次任务回答：

1. 固定平台、接口、版本、型号、区域、账号和数据政策。
2. 把业务请求与两套供应商对象分开，由专用适配器映射。
3. 发送前检查权限、输入和目标能力，并按 attempt 预留预算。
4. 超时后先判断结果是否已知、动作能否安全重放，再决定重试。
5. 分开传输终点、供应商终态和业务验证结果。
6. 工具调用只作为建议，经过 ACL、审批、幂等与副作用验证后再执行。
7. 同时结算任务结果和费用，用配对评测、影子流量与金丝雀逐步发布。
8. 回滚时恢复适配器、状态、文件、工具结果和预算账本，而不只是切换型号。

## 一手资料

- Google，[Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)，接口定位、状态、后台执行与存储边界；核对日期 2026-08-26。
- Google，[Interactions API 参考文档](https://ai.google.dev/api/interactions-api)，资源、状态、方法与步骤；核对日期 2026-08-26。[SOURCE:gemini-interactions-reference]
- Google，[Streaming interactions](https://ai.google.dev/gemini-api/docs/streaming)，SSE interaction/step 生命周期；核对日期 2026-08-26。
- Google，[GenerateContent API reference](https://ai.google.dev/api/generate-content)，`contents`、candidates、反馈与用量；核对日期 2026-08-15。[SOURCE:gemini-generate-content]
- Google，[Text generation](https://ai.google.dev/gemini-api/docs/text-generation)，当前文本入口和有损文本视图；核对日期 2026-08-15。[SOURCE:gemini-text-generation]
- 目标型号、SDK、数据保留、区域与价格页面；真实部署时需要重新核对。
