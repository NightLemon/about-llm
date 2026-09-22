# 模型选型：从候选身份到发布决定

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要选择 checkpoint、推理运行时或云模型 API 的开发者与算法工程师。
- **先修**：知道模型家族、具体工件、API、运行时和应用系统不是同一个对象。
- **首次阅读**：候选身份 → 硬约束 → 统一任务 → 发布决定。
- **完成信号**：能说明候选为何进入或退出比较，并让另一位工程师复跑同一任务与决策。
- **卡住时**：先读[新手知识地图](../guide/beginner-map.md)和[云 API 契约](cloud-api-contracts.md)。

</div>

模型选型不是回答“哪个品牌最强”，而是为一个已知任务选择可执行、可治理、可回滚的组合。
本页只保留四步决策主线；各模型的接口差异、推理容量、费用重试和发布协议由专题页负责。

假设目标是在 RTX 3070 Laptop 上学习中文 RAG 与结构化工具调用。当前本地基线是固定 revision 的
Qwen3-0.6B + nano-vLLM，云端再选一个质量参照。两者先承担不同角色，只有通过相同硬约束和任务评测后，
才决定生产主路、降级路或继续实验。

## 第一步：把候选写成可执行身份

“GPT、Llama、Qwen、DeepSeek、Claude 或 Gemini”只是家族名。真正参与比较的是下面某一层的具体对象：

| 层 | 必须固定什么 | 它仍不能回答什么 |
|---|---|---|
| 研究主张 | 论文、方法和实验范围 | 当前产品是否采用该方法 |
| 开放权重 | repo、完整 revision、config、tokenizer/processor、license | 目标 runtime 是否正确支持 |
| 云端 API | provider、platform、region、API surface、model id/snapshot、核对日期 | 后端内部实现与未来漂移 |
| 执行配置 | runtime、kernel、dtype、quantization、Adapter、template | 目标任务是否合格 |
| 应用系统 | RAG/Agent 版本、权限、重试、缓存、评测 | 失败究竟来自哪一层 |

候选记录至少包含：

| 部分 | 本地候选 | 云端候选 |
|---|---|---|
| 工件身份 | repo、revision、文件 hash | provider、platform、region、model id/snapshot |
| 输入协议 | tokenizer、chat template、processor | API surface、请求 schema、工具/媒体格式 |
| 执行方式 | runtime、dtype、量化、kernel、Adapter | SDK/HTTP 版本、流式与存储模式 |
| 任务配置 | Prompt、检索上下文、工具 schema、采样 | 同左，并记录 provider 特有字段 |
| 时间边界 | 依赖与运行日期 | 文档、型号、价格、限额的 `checked_at` |

云端只有动态别名时，把不可见 revision 记为 `unknown`；请求哈希只能确认保存的字节，不能揭示供应商内部版本。
本地权重可下载也不等于可发布，还要分别核对许可、数据/代码开放范围、第三方量化来源与 remote code 风险。

### 家族只帮助建立候选池

下表不是排名，而是进入实验前的第一项核对：

| 家族 | 常见取得方式 | 首先核对 |
|---|---|---|
| GPT | 云 API | API surface、model/snapshot、output/tool/usage 契约 |
| Claude | 云 API | Messages、顶层 system、content blocks、stop reason |
| Gemini | Gemini API 或 Vertex AI | platform、region、Interactions 或 `generateContent`、parts/steps |
| Llama | 开放权重或托管 | checkpoint、template、license、量化与 runtime |
| Qwen | 开放权重或云 API | 本地/云身份、dense/MoE、processor、工具模板 |
| DeepSeek | 开放权重或云 API | checkpoint/API、dense/MoE、MLA、推理模式与 runtime |

家族特有边界见 [GPT](gpt.md)、[Claude](claude.md)、[Gemini](gemini.md)、[Llama](llama.md)、
[Qwen](qwen.md)和[DeepSeek](deepseek.md)。

## 第二步：先执行硬约束

开始跑分前，先排除根本不能用于目标场景的候选：

| 硬约束 | 所需证据 | 常见假阳性 |
|---|---|---|
| 数据驻留与地域 | provider/region 政策或自托管拓扑 | “企业版”名称 |
| 许可与用途 | 固定版本 license 与责任人结论 | model card 一句概述 |
| 输入模态 | 目标 API/processor 的真实输入控制 | 产品网页展示过图片 |
| 输出与工具 | schema、tool、stream 的契约测试 | “OpenAI-compatible” |
| 上下文 | tokenizer 后长度、输出预留与任务质量 | 最大位置数 |
| SLO 与容量 | 固定 workload 的排队、尾延迟、峰值与失败 | 单次 warm latency |
| 安全与治理 | IAM、ACL、保留、删除、审批与事件流程 | 模型会拒答 |
| 可运维性 | health、限流、版本、观测与回滚 | Demo 返回一句话 |

[
feasible(c)=igwedge_j g_j(c)
]

任一强制条件失败就停止。敏感数据、许可或真实副作用相关条件为 `unknown` 时也应暂停；
低风险探索若接受例外，要记录范围、负责人和到期时间。

本地容量的权重、KV、activation、workspace 与 runtime 账本见[推理优化](../systems/inference-optimization.md)。
云调用的 attempt、预算、未知结果和账单对账见[云 API 可靠性](cloud-api-reliability.md)。

## 第三步：让候选完成同一任务

硬约束通过后，固定 cases、Prompt、检索上下文、工具 schema、采样与成功定义，再比较逐 case 结果。
不同 tokenizer、模板或 provider 请求格式可以保持各自正确实现，但它们必须服务同一业务任务。

| 场景 | 必须固定 | 主要 gate | 失败要分到 |
|---|---|---|---|
| RAG | corpus/snapshot、ACL、retriever、packing | answer、citation、abstention、延迟/成本 | 语料、召回、重排、packing、生成 |
| Agent | tool schema、权限、状态、预算、verifier | task success、越权、重复 effect、pending | planning、schema、policy、handler、验证 |
| SFT/LoRA | base、训练/留出、模板/mask、seed | 留出任务、格式、安全、资源 | 泄漏、mask、过拟合、工件兼容 |
| 推理服务 | checkpoint、dtype/quant、长度与到达分布 | TTFT、TPOT、terminal、吞吐、OOM | queue、prefill、decode、取消/超时 |
| 结构化抽取 | schema、业务规则、无答案/冲突 case | parse、schema、字段与拒答 | 语法、schema、值、业务冲突 |
| 多模态 | 媒体 bytes、processor、文本对照 | task、grounding、media-use 反事实 | 未读模态、OCR、定位、媒体安全 |

先画 Pareto frontier：候选若在所有关键指标上都不优于另一候选，可以先淘汰。
需要汇总效用时，权重、归一化与缺失值规则必须在看结果前确定；安全门禁、关键切片和尾延迟仍单列。

两个候选回答同一 case 时保存成对差值。来自同一用户、文档或会话的样本不能假装相互独立；
区间与重采样方法见[评测方法](../quality/evaluation-methodology.md)。

## 第四步：形成决定并保存依据

选型输出不是“赢家名称”，而是一张可重放的决策记录：

```json
{
  "candidate_id": "generator-a",
  "identity": {
    "provider_or_repo": "<provider-or-repo>",
    "model_id": "<exact-id>",
    "immutable_revision": null,
    "api_or_runtime": "<name+version>",
    "region_hardware": "<declared-target>"
  },
  "workload": {
    "cases": "sha256:<...>",
    "prompt_and_tools": "sha256:<...>",
    "arrival_and_lengths": "sha256:<...>"
  },
  "decision": {
    "hard_gates": "sha256:<...>",
    "metrics_and_slices": "sha256:<...>",
    "result": "release | limited-release | reject | investigate",
    "rollback_trigger": "<condition>",
    "not_verified": ["<boundary>"]
  }
}
```

对本地 checkpoint，`immutable_revision: null` 通常表示身份记录未完成；对只公开动态 model id 的云服务，
它可能是对供应商可见信息的诚实记录，但必须同时声明漂移风险。

回到开头的 3070 案例，决定只有三类：

| 结果 | 决定 |
|---|---|
| 本地候选通过质量、安全与延迟门槛 | 本地承担目标任务，云端保留参照或降级 |
| 本地质量不足，云端通过治理与任务门槛 | 云端承担目标任务，本地保留学习与回归 |
| 任一方案仍有关键硬约束失败或未知 | 暂不发布，调整任务、候选或治理条件后重测 |

升级不是只改 `model id`。重放输入序列化、指令冲突、结构化输出、tool、stream、RAG 注入、
关键切片、延迟与成本，再经过 shadow、canary 和回滚演练。完整流程见[发布、观测与回滚](../applications/llmops-release.md)。

## 常见误判

- 榜单第一不等于满足数据、接口、质量、延迟和治理硬约束。
- 参数量不能直接推出能力或显存；MoE 还要区分总参数、激活参数与通信。
- 更长的声明窗口不证明模型会利用中部证据，也不证明目标延迟可接受。
- JSON/schema 合法不证明字段真实、资源属于当前租户或动作已经获批。
- 开放 checkpoint、同品牌云 API、蒸馏模型和终端产品不能互相借用结论。
- 不同质量门槛、输出长度或量化条件下的吞吐不能直接比较。

## 自测与实践

1. 为一个本地 checkpoint 和一个云 API 写出完整候选身份与未知字段。
2. 定义三个硬约束、五个任务指标、一个 protected slice 和一个回滚条件。
3. 固定同一批 RAG 或结构化抽取 case，保存逐项结果而不只报平均分。
4. 交换模板或 runtime 后重跑，说明差异为何不能直接归因于模型。
