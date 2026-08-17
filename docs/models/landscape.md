# GPT、Llama、Qwen、DeepSeek、Claude 与 Gemini

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要选模型、部署路线、API 供应商或设计模型迁移评测的开发者与算法工程师。
- **先修**：知道模型家族、checkpoint、tokenizer、runtime、云 API 和完整应用系统不是同一层对象。
- **首次阅读**：对象分层 → 家族定位 → 硬约束 → 任务协议 → 成本与迁移。
- **完成信号**：能提交一份包含候选身份、淘汰理由、固定 workload、证据等级和回滚条件的模型选型记录。
- **卡住时**：先读[新手知识地图](../guide/beginner-map.md)、[Transformer](../core/transformer.md)和[云 API 契约](cloud-api-contracts.md)。

</div>

## 一句话结论

不要选择一个品牌名，要选择一个**可执行候选**：模型或 API 身份、tokenizer/template/processor、runtime、精度或量化、服务配置和任务协议共同决定实际行为。先用隐私、许可、模态、延迟、显存和接口能力做硬约束过滤，再在同一 workload 上比较质量、失败模式、成本与运维；榜单只能帮助生成候选，不能替代发布决定。

## 先把比较对象分层

“GPT 比 Llama 强吗”通常不是一个可检验问题，因为左右两边都没有确定到可执行对象。至少区分六层：

| 层 | 例子 | 能证明什么 | 不能自动证明什么 |
|---|---|---|---|
| 研究主张 | 某篇 scaling、MoE、MLA、对齐论文 | 论文设置中的方法与结果 | 当前产品完整采用该方法 |
| 模型家族 | GPT、Claude、Gemini、Llama、Qwen、DeepSeek | 生态与研究路线的导航标签 | 固定架构、参数量或接口 |
| 发布 artifact | 固定 checkpoint revision、tokenizer、processor、license | 这组可取得文件的身份和声明 | runtime 正确支持、任务质量 |
| 产品 SKU/API | provider、model id、region、API surface、snapshot | 供应商在检查时承诺的外部契约 | 内部权重、路由和训练配方 |
| 执行配置 | runtime/kernel、dtype、quantization、adapter、template | 一条可运行路径的配置 | 目标硬件上的正确性和性能 |
| 应用系统 | RAG、Agent、缓存、权限、重试、评测 | 端到端用户行为 | 问题可归因于模型本身 |

同一个开放权重 checkpoint 换 tokenizer 模板、量化方法或 serving runtime，输出和吞吐都可能改变；同一个云端 model id 换 API surface、区域、工具协议或供应商别名，也可能不是同一契约。比较报告必须先声明比较的是哪一层。

### 可执行候选的最小身份

本地或自托管候选可以写成：

\[
c_{local}=(repo,revision,files,tokenizer,template,processor,runtime,dtype,quant,adapter,serve)
\]

云端候选可以写成：

\[
c_{api}=(provider,platform,region,api\ surface,model\ id,snapshot,checked\_at,request\ contract)
\]

若供应商没有提供不可变 snapshot，就把该字段记为未知或 `null`，不要把一个可移动别名伪装成权重 revision。请求/响应 hash 可以证明保存的字节一致，不能证明供应商内部没有更新或路由。

## 六个家族应该怎样放进候选池

下表描述的是**首先要验证的证据面**，不是永久能力排名。

| 家族 | 常见取得方式 | 首先固定什么 | 优先验证什么 | 典型误读 |
|---|---|---|---|---|
| GPT | OpenAI 云 API | API surface、model/snapshot、工具与输出契约、检查日期 | output item、structured output、tool、usage、finish/incomplete 与迁移回归 | 用旧 GPT 论文推当前产品架构 |
| Claude | Anthropic 云 API | Messages 版本、model id、顶层 system、content blocks、检查日期 | block/event 状态机、tool use/result、stop reason、usage 与缓存字段 | 把 Constitutional AI 论文当当前完整配方 |
| Gemini | Gemini API 或 Vertex AI | platform、region、Interactions 或 `generateContent`、model id、检查日期 | parts/steps、函数调用、多模态输入是否被使用、状态与存储语义 | 把终端产品、Gemini API、Vertex AI 混成一个接口 |
| Llama | 开放权重/托管服务 | 固定 checkpoint、tokenizer/template、license、runtime | config/权重匹配、量化、KV、单卡容量、Base/Instruct 边界 | 把“开放权重”写成无条件开源或自由商用 |
| Qwen | 开放权重或云 API | 本地/云路线、checkpoint、模板、工具 schema、processor | 中文/多语言 tokenization、dense/MoE、工具轮次、runtime 兼容 | 用本地 checkpoint 的 config 推断云端 SKU |
| DeepSeek | 开放权重或云 API | checkpoint/API 身份、架构字段、remote-code/runtime 边界 | dense/MoE、MLA 拒绝标准 KV 公式、推理模式和模板 | 因同品牌就给蒸馏模型套上 V3/R1 架构 |

进入各章查看家族特有契约：

- [GPT](gpt.md)：公开研究与当前 OpenAI 产品/API 分账；
- [Claude](claude.md)：Messages、content blocks、工具与流式事件；
- [Gemini](gemini.md)：Gemini API、Vertex AI、Interactions 与 `generateContent`；
- [Llama](llama.md)：开放权重 checkpoint、配置、许可、量化与单卡部署；
- [Qwen](qwen.md)：中文/多语言、dense/MoE、模板、工具与本地/云分层；
- [DeepSeek](deepseek.md)：MoE、MLA、推理后训练与派生 checkpoint 边界。

### 开放权重不等于开源软件

“可以下载权重”只说明一种分发方式。还要分别检查：

- 权重许可是否允许目标用途、地域、再分发和衍生模型；
- 训练代码、数据、tokenizer、processor、评测脚本和完整配方是否公开；
- 第三方量化或转换文件是否能追溯到预期 base revision；
- model card 的厂商声明是否经过独立复现；
- remote code、custom kernel 和依赖是否进入供应链审查。

因此报告应使用“开放权重（open-weight）”“开源代码（open-source code）”“开放数据（open data）”等精确词，不用一个“开源模型”覆盖所有维度。

## 第一步：先做硬约束过滤

在跑 benchmark 前建立 feasible set。任何硬约束失败都应淘汰候选，不能用较高平均质量抵消。

| 约束 | 需要的证据 | 常见假阳性 |
|---|---|---|
| 数据驻留与地域 | provider/region 政策或本地部署拓扑 | “企业版”三个字 |
| 许可与用途 | 固定版本 license、法务/负责人结论 | model card 中一句宽泛说明 |
| 输入模态 | 目标 API/processor 的真实输入控制 | 产品网页展示过图片 |
| 输出与工具 | 目标 model/API 的 schema、tool、stream control | “OpenAI-compatible” |
| 上下文容量 | tokenizer 后真实长度、输出预留、目标任务质量 | config 中的最大位置数 |
| 延迟/SLO | 固定 workload 的 offered-load 测量 | 单请求 warm latency |
| 本地容量 | 权重、KV、激活、workspace 与 runtime 峰值 | 文件大小或参数量乘 dtype |
| 安全与治理 | IAM、ACL、日志、删除、保留、审批与事件流程 | 模型会拒答 |
| 可运维性 | health、限流、升级、回滚、观测与 on-call | demo 能返回一句话 |

可以把约束写成布尔 gate：

\[
feasible(c)=\bigwedge_j g_j(c)
\]

只有 `feasible(c)=true` 的候选才进入质量/成本比较。`unknown` 不应自动当作 `true`；高风险约束缺证据时应 fail closed，低风险探索可以显式记录 exception、负责人和到期时间。

## 第二步：再做多目标比较

模型选型通常没有单一最大值。至少同时观察：

- 任务质量：正确、完整、忠实、引用、格式和拒答；
- 风险：越权、泄漏、注入、危险动作和受保护切片退化；
- 时延：排队、TTFT、TPOT、terminal latency 与 timeout；
- 容量：吞吐、并发、峰值内存和降级曲线；
- 成本：token/media/tool 费用、重试、闲置、运维和迁移；
- 可复现与可控：revision、模板、量化、adapter、回滚和数据驻留。

先画 Pareto frontier，查看哪些候选在所有目标上都被支配。若业务必须汇总为一个分数，可写：

\[
U(c)=\sum_k w_k\tilde m_k(c),\qquad \sum_k w_k=1
\]

但权重、归一化范围、方向和缺失值策略必须在看结果前确定。一个加权分数会隐藏 protected slice、安全 gate 和尾延迟，不得替代逐项报告。不要在看到赢家后调整权重。

### 报告不确定性，而不只报均值

同一 case 上比较候选时保存 paired result，给出逐 case 差值、关键切片和适当的配对区间/检验。用户、文档或会话内相关时按 cluster 重采样或随机化；不能把相关 case 当独立样本。统计显著不等于业务重要，区间也不修复错误标签、污染或不代表生产的 workload。具体方法见[评测方法](../quality/evaluation-methodology.md)。

## 第三步：为任务固定比较协议

“通用能力”不能替代目标系统的 failure taxonomy。以下协议用于生成可证伪证据，不预设某一家族获胜。

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

比较 generator 时先冻结检索结果和 packing，防止把检索差异归因给模型；比较端到端系统时则同时保存 retrieval 与 generation trace。答案包含引用不等于引用支持 claim，格式 gate、citation correctness 和 semantic entailment 要分开。证据不足时的正确拒答也应计入 coverage-risk，而不是只算 answer rate。

### Agent 选型

工具调用只是候选动作，不是执行授权。评测既要覆盖正常任务，也要覆盖 prompt injection、未知字段、错 tool id、并行调用、超时后状态未知、审批恢复、重复 effect 和 verifier 拒绝。模型“没有调用工具”可能是安全拒绝，也可能是能力失败，必须按 policy expectation 分账。

### 微调选型

先建立 prompt/RAG/base baseline，再判断更新参数是否必要。比较不同 base model 时，不要复用不兼容的 tokenizer、chat template 或 target modules；数据 audit、split binding、assistant label 和 held-out gate 应先于训练。训练 loss 下降只说明优化目标变化，不证明任务改善；adapter 能 reload 也不证明与 serving runtime、量化基座或 license 兼容。

### 推理部署选型

先用同一输出协议检查质量差，再比较性能。固定 warm/cold、batch/并发、输入输出长度分布、sampling、prefix reuse、量化、runtime/kernel 和硬件。一次短 prompt 的 tokens/s 不能外推到长 prefill、多轮 KV、burst arrival 或 tool-heavy workload。

## 本地单 GPU：容量账本

本地候选的峰值内存不是权重文件大小。推理时至少拆为：

\[
M_{peak}\approx M_{weights}+M_{KV}+M_{activations}+M_{workspace}+M_{runtime}
\]

训练或 LoRA 还要加入梯度、optimizer state、可训练 master weights、保存/合并副本和 dataloader/pinned memory。QLoRA 通常只是冻结 base 权重低位存储；adapter、激活、梯度、optimizer 和部分计算仍使用更高精度，不能写成“整个训练都是 4-bit”。

标准 MHA/GQA 的理想 K/V payload 可按层数、KV heads、head dim、token 数、batch 和 element bytes 计算；MLA、跨层共享、压缩 cache、paged allocator、量化 metadata 与 runtime workspace 需要按实际实现重新建模。看到未知 attention marker 时拒绝套公式，比给出精确但错误的数字更好。

单卡筛选顺序：

1. 固定 checkpoint、tokenizer/template、license 与 runtime support；
2. 静态估算权重/KV/训练状态，明确不含项；
3. 在目标设备加载，记录 load peak 与 steady idle；
4. 用目标长度分布测 prefill/decode 和峰值；
5. 扩大 batch/并发直到受控 OOM 或 SLO 边界；
6. 量化后重新跑质量、安全和长上下文回归；
7. 保存版本、命令、原始 trace 和失败样本。

## 云 API：接口与费用账本

云端候选的请求成本可以概念化为：

\[
C_{request}=C_{input}+C_{cached}+C_{output}+C_{media}+C_{tool}+C_{retry}
\]

每项是否存在、token 分类和单价都按 provider/model/API version/checked_at 核对。计费用量以供应商返回的 usage/billing export 为准；本地 tokenizer 估算适合 preflight，不应冒充发票。缺 usage、partial stream、timeout 或 outcome unknown 时不能猜零费用；重试的每个 attempt 都可能独立产生 effect 与费用。

“按 token 便宜”也不等于总成本低。还要计入：

- 为达到质量门槛增加的输出、重试、RAG 和工具调用；
- 限额、排队、失败和 fallback provider；
- 日志、评测、合规、网络和数据迁移；
- adapter 维护、版本迁移和双跑窗口；
- 自托管的闲置容量、能源、值班和升级成本。

只在同一成功定义和质量 gate 下比较 cost per successful task；否则便宜但失败的请求会被错误地奖励。

## Provider adapter：统一什么，保留什么

内部 adapter 可以统一最小稳定语义：

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

但不能把 provider 差异抹掉：

- OpenAI 的 Responses 与 Chat Completions 不是只差 URL；
- Anthropic Messages 的顶层 `system`、content blocks 和 stop reason 有自身状态机；
- Gemini Interactions 与 `generateContent` 的状态、steps/parts 和存储语义不同；
- DeepSeek/Qwen 云 API 的扩展字段不由开放 checkpoint config 决定；
- “OpenAI-compatible”通常只表示部分请求/响应 shape 兼容，不保证错误、stream、usage、tool、idempotency 或计费语义一致。

未知 block/event 不应静默丢弃；在受控存储中保留脱敏原始投影，并让上层显式决定是否支持。纯文本业务遇到 only-tool、refusal 或未知 block 时应返回 typed state，而不是空字符串成功。

## 可复现选型 manifest

一个候选至少保存以下字段；示例是仓库协议，不是任何 provider 的请求 schema：

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

对本地 checkpoint，`immutable_revision=null` 通常是不合格输入；对没有公开权重 revision 的云 API，它可能是诚实记录。两种情况必须由 `provider_or_repo` 和证据边界区分，不能用同一字段含混处理。

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

先 shadow/offline replay，再小流量 canary；任何真实流量双跑都要满足数据处理授权，不能为了评测把敏感请求复制给未批准的 provider。发布记录应包含 threshold、审批人、回滚触发器和旧版本保留期。若 provider 别名不可固定，还要持续做 drift detection，而不是只在首次接入时评测。

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

型号、价格、上下文长度、区域、限额和 API 字段变化快。本章保留稳定比较方法；具体项目用 manifest 记录 provider/repo、model、revision/snapshot、API/runtime version、region 和 `checked_at`，再从对应官方文档与固定 artifact 核对。不要维护没有日期和来源的“永久最新/最强模型表”。
