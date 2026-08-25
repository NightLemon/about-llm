# GPT、Llama、Qwen、DeepSeek、Claude 与 Gemini

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要选模型、部署路线、API 供应商或设计模型迁移评测的开发者与算法工程师。
- **先修**：知道模型家族、checkpoint、tokenizer、runtime、云 API 和完整应用系统不是同一层对象。
- **首次阅读**：先跟随 3070 Laptop 上的选型案例，再学习怎样扩展候选、比较任务表现和设计迁移。
- **完成信号**：能说明一个候选为什么可以继续测试、为什么被淘汰，以及哪些变化其实来自模板、运行时或应用系统。
- **卡住时**：先读[新手知识地图](../guide/beginner-map.md)、[Transformer](../core/transformer.md)和[云 API 契约](cloud-api-contracts.md)。

</div>

## 从你的机器开始选，而不是从排行榜开始

假设你现在有一台配备 RTX 3070 Laptop GPU 的笔记本，想做两件事：

1. 用中文 RAG 回答自己的技术文档，并给出引用；
2. 让 Agent 生成结构化工具参数，但真正的权限检查和执行仍由程序完成。

你正在运行 `Qwen3-0.6B + nano-vLLM`。这个组合很适合学习一次请求怎样经过调度、KV Cache、
prefill、decode 和 sampling，也能承担本地功能冒烟。不过，这些事实还不能证明它已经达到真实业务所需的回答质量。
固定版本、运行命令和逐步观察项见 [Qwen3 穿过 nano-vLLM](../practice/labs/lab-7b-nano-vllm-qwen3.md)。

因此，第一轮选型可以先作出一个阶段性决定：

| 角色 | 候选 | 这一轮要回答的问题 |
|---|---|---|
| 本地学习基线 | 固定 revision 的 Qwen3-0.6B + nano-vLLM | 链路能否跑通？显存、延迟和失败状态能否解释？ |
| 云端质量参照 | 一个记录了供应商、API、model id、区域和核对日期的候选 | 在同一批 RAG/Agent case 上，质量提升是否值得费用与数据治理成本？ |

这不是提前宣布谁获胜。本地小模型和云端模型先承担不同角色，等它们通过同一组硬约束与任务评测后，
才能决定生产主路、降级路或是否继续扩大本地模型。

这个案例也揭示了“GPT、Qwen 和 Llama 选哪个”为什么不是一个完整问题。品牌名不能直接运行。
程序真正运行的是某个 checkpoint 或 API，连同 tokenizer、对话模板、运行时、精度、量化和服务配置。

## 先看清自己正在比较哪一层

同一个模型名字可能指论文、模型家族、权重文件，也可能指云端产品。它们回答的问题不同：

| 层 | 例子 | 这一层回答什么 | 下一步还要固定什么 |
|---|---|---|---|
| 研究主张 | 某篇 scaling、MoE、MLA 或对齐论文 | 某个方法在论文实验中怎样工作 | 产品是否采用，以及采用到什么程度 |
| 模型家族 | GPT、Claude、Gemini、Llama、Qwen、DeepSeek | 生态、发布方式和研究路线 | 具体型号、版本、架构与接口 |
| 发布文件 | checkpoint revision、tokenizer、processor、license | 你实际取得了哪些文件 | 运行时是否支持，任务表现怎样 |
| 云端 API | provider、region、API surface、model id、snapshot | 供应商对外承诺什么调用契约 | 后端是否更新、怎样路由、当前实测怎样 |
| 执行配置 | runtime、kernel、dtype、quantization、adapter、template | 这次程序到底怎样运行模型 | 目标硬件上的正确性、容量和速度 |
| 应用系统 | RAG、Agent、缓存、权限、重试、评测 | 用户最终经历的完整链路 | 失败发生在哪一层 |

同一个开放权重 checkpoint，换一个对话模板、量化方法或推理运行时，输出与吞吐都可能改变。
同一个云端 model id，换 API、区域或工具协议后，也未必还是同一份调用契约。

### 把候选写成别人可以重新运行的记录

以本地 Qwen 候选为例，模型名之外还要保存：

| 部分 | 要记录什么 | 少了会发生什么 |
|---|---|---|
| 模型文件 | repo、完整 revision、文件哈希 | 无法确认两次运行使用了相同权重 |
| 输入处理 | tokenizer、chat template、processor | 输入 token 或模态处理可能已经变化 |
| 执行方式 | nano-vLLM revision、dtype、量化、kernel | 输出、显存和速度无法复现 |
| 服务配置 | batch、上下文、采样、adapter | 质量或容量差异会被错归因给模型 |
| 测试任务 | case、检索上下文、工具 schema、成功条件 | 两个分数可能根本没有比较同一件事 |

云端候选改为记录供应商、平台、区域、API、model id、请求契约和核对日期。

供应商提供不可变 snapshot 时一并保存。只有动态别名时，就把内部版本记为未知。
请求或响应的哈希只能确认保存下来的字节，无法揭示供应商内部版本。

## 六个家族应该怎样放进候选池

下表不是能力排名。它告诉你把某个家族放进候选池后，第一批实验应该验证什么。

在开头的案例中，Qwen 首先进入本地候选，因为它已经能在目标机器上运行。
nano-vLLM 则让推理链路中的每一步都可以观察。

云端候选可以来自表中的多个模型家族。先根据数据政策、接口和任务需要缩小范围，
再用同一批样本实测；家族名称本身不提供优先级。

| 家族 | 常见取得方式 | 首先固定什么 | 优先验证什么 | 典型误读 |
|---|---|---|---|---|
| GPT | OpenAI 云 API | API surface、model/snapshot、工具与输出契约、检查日期 | output item、structured output、tool、usage、finish/incomplete 与迁移回归 | 用旧 GPT 论文推当前产品架构 |
| Claude | Anthropic 云 API | Messages 版本、model id、顶层 system、content blocks、检查日期 | block/event 状态机、tool use/result、stop reason、usage 与缓存字段 | 把 Constitutional AI 论文当当前完整配方 |
| Gemini | Gemini API 或 Vertex AI | platform、region、Interactions 或 `generateContent`、model id、检查日期 | parts/steps、函数调用、多模态输入是否被使用、状态与存储语义 | 把终端产品、Gemini API、Vertex AI 混成一个接口 |
| Llama | 开放权重/托管服务 | 固定 checkpoint、tokenizer/template、license、runtime | config/权重匹配、量化、KV、单卡容量、Base/Instruct 边界 | 把“开放权重”写成无条件开源或自由商用 |
| Qwen | 开放权重或云 API | 本地/云路线、checkpoint、模板、工具 schema、processor | 中文/多语言 tokenization、dense/MoE、工具轮次、runtime 兼容 | 用本地 checkpoint 的 config 推断云端 SKU |
| DeepSeek | 开放权重或云 API | checkpoint/API 身份、架构字段、remote-code/runtime 边界 | dense/MoE、MLA 需要不同 KV 公式、推理模式和模板 | 因同品牌就给蒸馏模型套上 V3/R1 架构 |

进入各章查看家族特有契约：

- [GPT](gpt.md)：公开研究与当前 OpenAI 产品/API 分账；
- [Claude](claude.md)：Messages、content blocks、工具与流式事件；
- [Gemini](gemini.md)：Gemini API、Vertex AI、Interactions 与 `generateContent`；
- [Llama](llama.md)：开放权重 checkpoint、配置、许可、量化与单卡部署；
- [Qwen](qwen.md)：中文/多语言、dense/MoE、模板、工具与本地/云分层；
- [DeepSeek](deepseek.md)：MoE、MLA、推理后训练与派生 checkpoint 边界。

### 权重下载完成，还不能直接发布

“可以下载权重”解决了取得文件的问题。是否能安全、合法并可重复地使用，还要分别检查：

- 权重许可是否允许目标用途、地域、再分发和衍生模型；
- 训练代码、数据、tokenizer、processor、评测脚本和完整配方是否公开；
- 第三方量化或转换文件是否能追溯到预期 base revision；
- model card 的厂商声明是否经过独立复现；
- remote code、custom kernel 和依赖是否进入供应链审查。

因此报告里最好分别写“开放权重（open-weight）”“开源代码（open-source code）”和“开放数据（open data）”。
一句“开源模型”常常会把许可、数据和实现透明度混在一起。

## 第一步：先排除根本无法使用的候选

先回到 3070 Laptop 的案例。假如私有文档不得离开本机，那么未获数据处理批准的云 API 就应停在这里。
它的回答质量再高，也无法弥补这个冲突。相反，如果目标是学习推理引擎，Qwen3-0.6B 能否在这张 GPU 上稳定加载，
比它在综合榜单上的名次更重要。

这类“必须满足”的条件称为硬约束。开始性能测试前，逐项查证：

| 约束 | 需要的证据 | 常见假阳性 |
|---|---|---|
| 数据驻留与地域 | provider/region 政策或本地部署拓扑 | “企业版”三个字 |
| 许可与用途 | 固定版本 license、法务/负责人结论 | model card 中一句宽泛说明 |
| 输入模态 | 目标 API/processor 的真实输入控制 | 产品网页展示过图片 |
| 输出与工具 | 目标 model/API 的 schema、tool 与 stream 行为 | “OpenAI-compatible” |
| 上下文容量 | tokenizer 后真实长度、输出预留、目标任务质量 | config 中的最大位置数 |
| 延迟/SLO | 固定 workload 的 offered-load 测量 | 单请求 warm latency |
| 本地容量 | 权重、KV、激活、workspace 与 runtime 峰值 | 文件大小或参数量乘 dtype |
| 安全与治理 | IAM、ACL、日志、删除、保留、审批与事件流程 | 模型会拒答 |
| 可运维性 | health、限流、升级、回滚、观测与 on-call | demo 能返回一句话 |

程序里可以把这些条件写成布尔门禁：

\[
feasible(c)=\bigwedge_j g_j(c)
\]

这里的 \(g_j(c)\) 表示候选 \(c\) 是否通过第 \(j\) 项条件。所有条件都通过，`feasible(c)` 才为真。

有些条件暂时只能得到 `unknown`。涉及敏感数据、许可或真实副作用时，未知就意味着暂停；
只做低风险本地探索时，可以记录临时例外、负责人和到期时间。

## 第二步：让剩下的候选完成同一组任务

通过硬约束后，让本地和云端候选回答同一批中文 RAG 问题、生成同一套工具参数。
不要先追求一个“总分”，因为选型同时关心以下几件事：

- 任务质量：正确、完整、忠实、引用、格式和拒答；
- 风险：越权、泄漏、注入、危险动作和受保护切片退化；
- 时延：排队、首 token 时间（TTFT）、后续 token 间隔（TPOT）、完整响应时间与超时；
- 容量：吞吐、并发、峰值内存和降级曲线；
- 成本：token/media/tool 费用、重试、闲置、运维和迁移；
- 可复现与可控：revision、模板、量化、adapter、回滚和数据驻留。

例如，本地候选可能便宜、可查看运行细节，却在复杂问答上落后；云端候选可能质量更高，
却受到网络时延、费用或数据策略限制。这不是一个数字能够诚实概括的取舍。

可以先画 Pareto frontier（帕累托前沿）：如果候选 A 在所有重要指标上都不优于 B，就先淘汰 A。
若业务最终必须汇总为一个分数，可以写成：

\[
U(c)=\sum_k w_k\tilde m_k(c),\qquad \sum_k w_k=1
\]

\(w_k\) 是业务为第 \(k\) 项指标预先设定的权重，\(\tilde m_k(c)\) 是统一方向和量纲后的指标值。
权重、归一化范围和缺失值处理方式应在查看结果前确定。最终报告仍要单列安全门禁、关键用户切片和尾延迟，
这样一个较高的平均分才不会掩盖局部事故。

### 报告不确定性，而不只报均值

两个候选回答同一道题时，保存成对结果和逐题差值。这样能看出改进来自多数题，还是少数极端样本。

若多条样本来自同一用户、文档或会话，它们彼此相关。此时应按用户、文档或会话分组重采样，
而不是把每一行都当成独立证据。区间与显著性只描述当前样本中的不确定性；业务影响、标签质量、
数据污染和生产代表性需要另外判断。具体方法见[评测方法](../quality/evaluation-methodology.md)。

## 第三步：先固定任务，再运行候选

“通用能力强”不能告诉你一次 RAG 引用为什么错，也不能告诉你 Agent 是否会重复退款。
为目标任务固定输入、系统组件和成功条件，才能把失败定位到具体环节。下表给出了常见任务的起点：

| 场景 | 固定变量 | 主要指标/gate | 必须分账的失败 |
|---|---|---|---|
| RAG | corpus/snapshot、ACL、retriever、packing、prompt | answer、citation、claim-evidence、abstention、latency/cost | 语料缺失、召回、rerank、packing、生成未用证据 |
| Agent | tool schema、权限、状态、预算、verifier | task success、unauthorized attempt、重复 effect、pending、成本 | planning、schema、policy、handler、环境观察、verifier |
| SFT/LoRA | base revision、train/held-out、template/mask、seed | held-out task/slice、格式、安全回归、显存/时间 | 数据泄漏、mask、过拟合、base drift、adapter/runtime |
| 推理服务 | checkpoint、dtype/quant、采样、长度/arrival 分布 | TTFT/TPOT/terminal、throughput、OOM、quality delta | client queue、server queue、prefill、decode、取消/超时 |
| 结构化抽取 | schema、业务规则、无答案/冲突 case | parse、schema、field exact、abstention | JSON 合法、schema 合法、值错误、业务冲突 |
| 多模态 | 媒体 bytes、processor、文本对照、扰动集 | task score、grounding、media-use counterfactual | 模态未被读取、OCR、定位、语言先验、媒体安全 |
| 代码 | repo snapshot、sandbox、测试、预算 | patch correctness、pass@k、回归、安全、成本 | 测试污染、环境失败、部分补丁、危险副作用 |

### RAG 选型

如果只想比较生成模型，就给所有候选相同的检索片段和上下文排列。这样，答案差异才主要来自生成阶段。
如果比较完整 RAG 系统，则同时保存检索轨迹与生成轨迹，用来判断正确证据究竟在哪一步丢失。

引用检查至少回答三个问题：引用格式是否可解析、来源是否真实存在、来源内容是否支持对应主张。
还要统计证据不足时能否正确拒答；只看“回答率”会奖励没有证据也敢猜的模型。

### Agent 选型

模型生成的工具调用只是动作建议，执行层仍要独立授权。正常任务之外，至少加入以下失败情形：

- 文档或工具结果中含有提示注入；
- 参数出现未知字段，或工具 ID 指向错误对象；
- 多个动作并行返回，远端超时后状态暂时未知；
- 审批后恢复任务，或同一副作用被重复请求；
- 业务验证程序拒绝模型声称的完成状态。

模型没有调用工具，有时是符合策略的安全拒绝，有时是任务能力失败。预先写出每条 case 期望的行为，评测时才能正确归类。

### 微调选型

先跑 Prompt、RAG 和原始模型基线，再判断是否值得更新参数。每个基础模型都要使用自己兼容的 tokenizer、
对话模板和 LoRA 目标层。训练前完成数据审计，固定训练/留出集，并确认只有目标回答位置参与损失。

训练损失下降说明优化器在拟合训练目标；adapter 能重新加载说明工件可以读取。任务质量是否提高、
推理服务是否兼容，仍需在留出任务和目标运行时中分别验收。

### 推理部署选型

先确认候选在同一输出协议下达到质量门槛，再比较速度。记录冷启动或预热状态、批量与并发、输入输出长度、
采样参数、前缀复用、量化、运行时、kernel 和硬件。

一条短 Prompt 的每秒 token 数只描述这一次测试。长输入的 prefill、多轮对话的 KV Cache、突发请求和频繁工具调用，
都会产生不同的瓶颈。

## 本地单 GPU：容量账本

Qwen3-0.6B 的权重能放进 3070 Laptop 显存，只说明第一道门可以通过。运行时还会为 KV Cache、
中间激活、kernel 工作区和框架自身分配显存。可以先用下面的账本估算峰值：

\[
M_{peak}\approx M_{weights}+M_{KV}+M_{activations}+M_{workspace}+M_{runtime}
\]

做 LoRA 训练时，账本还要增加梯度、优化器状态和可训练参数的高精度副本。
保存或合并模型也可能产生额外副本，数据加载还会使用锁页内存。

QLoRA 主要压缩被冻结的基础权重。适配器、激活、梯度和优化器状态仍可能使用更高精度。
因此“4-bit 基础权重”只描述权重存储的一部分，不能代表整个训练过程的内存。

对于标准 MHA 或 GQA 布局，理想的 K/V 数据量约为：

\[
M_{KV}=2L B T H_{kv} d_{head}s
\]

这里的 2 表示 K 和 V；\(L\) 是层数，\(B\) 是 batch size，\(T\) 是已缓存 token 数，
\(H_{kv}\) 是 K/V head 数，\(d_{head}\) 是每个 head 的维度，\(s\) 是每个元素的字节数。

这个公式只计算理想数据本体。分页分配器、block table、量化 scale、临时张量和 kernel 工作区都会增加开销。
MLA、跨层共享或压缩 cache 采用不同布局，应读取目标 checkpoint 与运行时实现后重新建模。

单卡筛选顺序：

1. 固定 checkpoint、tokenizer、对话模板、许可和运行时版本；
2. 静态估算权重、KV Cache 和训练状态，并写清估算尚未包含哪些开销；
3. 在目标设备加载模型，记录加载峰值和空闲稳态显存；
4. 使用目标输入长度，分别测量 prefill、decode 和峰值显存；
5. 逐步增加 batch 或并发，直到触及受控 OOM 或服务目标边界；
6. 量化后重新检查质量、安全行为和长上下文表现；
7. 保存版本、命令、原始轨迹和失败样本。

## 云 API：接口与费用账本

云端候选的请求成本可以概念化为：

\[
C_{request}=C_{input}+C_{cached}+C_{output}+C_{media}+C_{tool}+C_{retry}
\]

供应商可能分别计算普通输入、缓存输入、输出、媒体和工具费用。每次实验都要按供应商、model id、API 版本和核对日期查价。

正式对账以供应商返回的 usage 或账单导出为准。本地 tokenizer 只能在发请求前估算预算。
如果流式响应中断或请求超时，客户端可能暂时不知道供应商是否已经完成并计费。此时先保留一笔“费用待确认”，
因为后续重试也可能形成独立费用。

“按 token 便宜”也不等于总成本低。还要计入：

- 为达到质量门槛增加的输出、重试、RAG 和工具调用；
- 限额、排队、失败和 fallback provider；
- 日志、评测、合规、网络和数据迁移；
- adapter 维护、版本迁移和双跑窗口；
- 自托管的闲置容量、能源、值班和升级成本。

因此，成本比较的分母应是“成功完成的任务”，而不是请求总数。两个候选还要使用相同的成功定义并通过相同质量门槛，
否则一个便宜但经常失败的方案反而会显得占优。

## 统一云 API 时，哪些差异必须保留

为了让业务代码能够切换供应商，可以在中间增加 provider adapter（供应商适配层）。
它把稳定的公共部分整理成内部请求和结果：

```text
NormalizedRequest
├── trusted instructions
├── typed content parts
├── tools + schema revision
├── generation bounds
└── request identity

NormalizedResult
├── typed output blocks/items
├── tool candidates
├── finish/refusal/incomplete
├── usage + provider request id
├── model/platform identity
└── provider-specific raw projection
```

公共结构减少了业务层的重复代码，但供应商特有语义仍要保留：

- OpenAI 的 Responses 与 Chat Completions 不是只差 URL；
- Anthropic Messages 的顶层 `system`、content blocks 和 stop reason 有自身状态机；
- Gemini Interactions 与 `generateContent` 的状态、steps/parts 和存储语义不同；
- DeepSeek/Qwen 云 API 的扩展字段不由开放 checkpoint config 决定；
- “OpenAI-compatible”通常只表示部分请求/响应 shape 兼容，不保证错误、stream、usage、tool、idempotency 或计费语义一致。

遇到未知的内容块或流事件时，适配层应返回明确的“暂不支持”状态，并在受控存储中保留脱敏后的原始结构。
纯文本业务收到只有工具调用、拒答或未知内容块的响应时，也应返回对应状态；空字符串会让上层误以为请求成功。

## 把选择过程保存下来

下面的 manifest（实验清单）把候选身份、运行方式、任务和决策规则放在一起。
它是本仓库建议的记录格式，不是供应商 API 的请求结构：

```json
{
  "candidate_id": "generator-a",
  "artifact": {
    "provider_or_repo": "<provider-or-repo>",
    "model_id": "<exact-id>",
    "immutable_revision": null,
    "checked_at": "<UTC-date>",
    "tokenizer_template_processor": "sha256:<...>"
  },
  "execution": {
    "api_surface_or_runtime": "<name+version>",
    "region_hardware": "<declared-target>",
    "dtype_quant_adapter": "<exact-config>",
    "generation_config": "sha256:<...>"
  },
  "workload": {
    "cases": "sha256:<...>",
    "slices": "sha256:<...>",
    "arrival_and_length_distribution": "sha256:<...>"
  },
  "decision": {
    "hard_gates": "sha256:<...>",
    "metrics": "sha256:<...>",
    "evidence_boundary": "<what-was-not-tested>"
  }
}
```

本地 checkpoint 应提供不可变 revision；这里若写 `null`，通常说明身份记录还没完成。
云 API 可能只公开动态 model id，此时 `null` 是对供应商可见信息的诚实记录。
读取 manifest 时，应结合 `provider_or_repo` 判断是哪种情况，并在结论中写明版本漂移风险。

## 升级与迁移不是改一个 model id

模型升级至少重放：

1. tokenizer/template 或 provider input 的实际序列化；
2. system/developer/user/tool 的冲突优先级；
3. structured output、refusal、incomplete 和 length termination；
4. 单/并行 tool call、tool result、未知字段与重复 effect；
5. stream event 顺序、UTF-8 边界、usage 和错误 taxonomy；
6. RAG 引用、无答案、冲突证据与 prompt injection；
7. protected slice、越权、安全与内容政策回归；
8. 延迟、限额、成本、峰值内存和 fallback 行为。

先做离线重放或 shadow（只观察、不影响用户结果的影子流量），再进入小流量 canary（金丝雀发布）。
真实请求要经过数据处理授权后才能复制给第二个供应商。发布记录还要写明通过阈值、审批人、回滚触发器和旧版本保留期。

供应商别名会动态更新时，在接入后继续运行 drift detection（漂移检测）。它负责发现同一别名的输出、延迟或协议行为是否发生变化。

## 回到这台 3070：这一轮究竟怎样结束

完成上述检查后，阶段性决定可以有三种：

| 实测结果 | 决定 | 原因 |
|---|---|---|
| 本地候选通过质量、安全和延迟门槛 | 让本地路线承担目标任务，云端保留为可选参照 | 数据与运行时都在自己的控制范围内 |
| 本地候选质量不足，云端候选通过治理和任务门槛 | 云端承担目标任务，本地继续用于学习和回归 | 小模型的教学价值与生产质量要求可以分开 |
| 两者各有未通过的硬约束 | 暂不发布，调整任务、检索、模型或治理条件后重测 | 排名无法覆盖硬约束失败 |

无论得到哪种结果，都要保留逐条样本和运行轨迹。

若本地答案变化，先检查模型 revision、tokenizer、对话模板、nano-vLLM、量化和采样。
若端到端 RAG 变化，还要检查检索与上下文。其他条件保持一致时，差异才有理由主要归因于模型。

## 常见错误

### 把 benchmark 第一名当部署答案

榜单可能使用不同 prompt、采样、工具权限、judge、污染控制和预算。先复现 harness，再运行自己的 case；无法复现时只把榜单作为作者/平台报告。

### 比较不同质量约束下的吞吐

一个候选用更激进量化、更短输出或不同 stop，tokens/s 更高并不构成公平优势。先通过同一 task/safety gate，再比较资源指标。

### 用参数量推能力或显存

闭源参数量可能未知；MoE 还要区分总参数与激活参数。本地显存受 dtype、量化 metadata、KV、workspace 和 runtime 影响，参数量只能做有边界的估算。

### 把支持长窗口等同于会使用长证据

配置/API 接收更多 token 不证明中部信息利用、多跳推理、引用忠实或目标延迟。应按位置、长度、干扰、语言和任务切片实测，并保留 tokenizer 后长度。

### 把结构化输出等同于正确动作

schema 约束语法，不证明字段真实、资源属于当前 tenant、金额合理或动作获批。业务校验、IAM/ACL、幂等、审批与审计必须在模型外执行。

### 用同品牌跨层外推

开放 checkpoint、云 API、蒸馏模型和终端产品即使共享品牌，也可能拥有不同架构、模板、工具和治理契约。每条结论绑定到实际对象。

## 面试与设计题

1. 为什么“GPT 对比 Llama”不是一个完整实验问题？请补齐双方的可执行身份。
2. 公司既要求敏感数据本地驻留，又要求图像输入和低运维成本，你会怎样建立 hard gate、候选集和 exception 流程？
3. 如何公平比较一个 MoE checkpoint 与 dense checkpoint 的质量、加载内存、每 token 计算和吞吐？
4. 为什么两个 OpenAI-compatible endpoint 仍需要独立的 stream、tool、usage、错误和计费契约测试？
5. 从云 API 迁移到本地 Qwen/Llama 时，哪些变化属于模型，哪些属于 tokenizer/template/runtime 和系统边界？
6. 一个模型 structured-output parse rate 更高但 protected slice 准确率更低，为什么不应直接用平均加权分数决定发布？
7. 如何证明多模态模型真的使用了图像，而不是只从问题文本猜答案？
8. 为什么一次成功的 adapter reload、一次训练 loss 下降和一次短 prompt benchmark 都不足以证明生产就绪？

## 实践任务

选择一个云 API 候选和一个可在本地运行的开放权重候选，完成同一项小型 RAG 或结构化抽取任务：

1. 写出两条候选的完整身份与未知字段；
2. 定义至少三个 hard gate、五个质量/风险指标和一个 protected slice；
3. 固定 cases、prompt、schema、retrieval context 和成功定义；
4. 记录 usage/成本或本地内存、TTFT/terminal latency；
5. 保存逐 case 结果，而不只保存平均分；
6. 写出至少五项未验证边界和一个回滚条件；
7. 交换 adapter/runtime 后重跑，解释变化能否归因于模型。

这个任务的完成信号不是宣布赢家，而是让另一位工程师能够复查候选身份、重算指标，并知道结论在哪些边界外失效。

## 型号信息的时间边界

型号、价格、上下文长度、区域、限额和 API 字段都会变化，因此本章只保留稳定的比较方法。

具体项目使用 manifest 记录供应商或仓库、model id、revision 或 snapshot、API 或运行时版本、区域和 `checked_at`。
这些字段应从对应官方文档与固定文件核对。

有日期、有来源的实验记录，比一张声称“永久最新、永远最强”的型号表更可靠。
