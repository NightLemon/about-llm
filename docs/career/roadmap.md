# 岗位路线与求职地图

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备 LLM 开发、算法、训练、推理系统、评测数据或安全平台岗位的开发者与算法工程师。
- **先修**：已完成至少一个可运行实验，愿意用原始 artifact 而不是技术关键词证明能力。
- **首次阅读**：先拆一份真实 JD，再选择路线、能力证据和主项目。
- **完成信号**：能为一个真实 JD 写出责任画像、证据矩阵、主项目选择和仍缺失的下一项实验。
- **卡住时**：先看[项目索引](../practice/project-index.md)，找一个与你现有背景最接近的可运行项目。

</div>

你是一名后端工程师，看到一份 “LLM Engineer” 职位描述：

```text
负责企业知识助手的研发与优化；
熟悉 RAG、Agent、vLLM、LangChain，了解 PEFT；
保障系统稳定性和回答质量。
```

如果按关键词学习，几乎整套 LLM 技术栈都要补。更有用的第一步是问清楚：知识助手答错时，
这个岗位先查 retriever、Prompt 还是训练数据？高峰期出现 `504` 时，是否由它承担 on-call？
Agent 重复发出业务操作时，谁负责状态机和恢复？

这三个问题会把同一份 JD 分成应用后端、RAG 算法、Agent 平台或推理系统等不同岗位。
本页帮助你完成这种拆解，并把已有经验组织成面试时可以下钻的证据。

它不是周计划，也不承诺证书或项目数量能换来 offer。公司、团队和职级定义并不统一，
最终判断仍以目标 JD、面试环节和团队实际交付物为准。

**求职导航**：[深度面试题](interview-questions.md) · [系统设计题](system-design.md) · [简历项目与证据包](resume-projects.md) · [完整学习路径](../guide/learning-paths.md)
{ .doc-nav }

## 先问：哪种失败会落到你桌上

“LLM Engineer” 可能负责 API 后端、检索算法、训练、推理平台或客户交付；
“算法工程师” 也可能主要维护数据和评测。岗位名称提供的信号很弱，责任边界更稳定。

仍以企业知识助手为例：

| 线上现象 | 第一个需要回答的问题 | 更接近的责任域 |
|---|---|---|
| 答案引用了错误文档 | gold 文档没召回、被重排丢掉，还是生成器误用证据？ | RAG / 搜索算法 |
| 输出字段错误或接口超时 | Prompt、状态、API、依赖和降级哪一层先偏离？ | LLM 应用 / 后端 |
| 同一退款被请求两次 | 审批、幂等、pending 与 verifier 怎样恢复？ | Agent / 平台 |
| 微调后中文能力退化 | 数据 mixture、loss、checkpoint 或评测切片哪里变化？ | 模型 / 训练算法 |
| 高峰期大量 `504` | client queue、admission、KV 还是 decode 饱和？ | 推理 / 系统 |
| 发布 gate 通过但用户变差 | case、标注、指标或统计口径遗漏了什么？ | 评测 / 数据 |
| 越权文档进入 Prompt | ACL、缓存、tool policy 或审计链在哪一层失守？ | 安全 / 治理平台 |

阅读 JD 或与招聘方沟通时，继续追问：

1. 团队交付什么 artifact、service 或 metric？
2. 上游输入由谁拥有，输出交给谁？
3. 哪类失败由这个岗位首先定位和修复？
4. 是否负责线上 SLO、数据、训练、GPU、权限或发布？
5. 工作停留在 prototype，还是覆盖运行、评测、回滚与事件处理？

一份工作的日常，往往就是反复处理其中一两类失败。优先选择你愿意长期深挖的问题，
再看模型名称和技术栈是否匹配。

## 七类岗位责任画像

下面把常见责任展开。它不是招聘市场占比，也不是统一职级标准；小团队的一个岗位可能横跨两三行。

| 路线 | 主要交付物 | 首先负责的失败 | 核心能力 | 最强作品集信号 |
|---|---|---|---|---|
| LLM 应用/后端 | API 产品、RAG/Agent workflow、业务集成 | 答案错误、接口失败、权限/成本/延迟失控 | Python/后端、Prompt、RAG、Agent、云 API、测试与可观测 | 有 ACL、引用、typed state、预算、故障注入和逐 case 评测的服务 |
| RAG/搜索算法 | ingestion、索引、retriever/reranker、context | 召回、排序、过滤、packing 或引用失败 | IR、embedding、学习排序、数据、检索评测、在线服务 | 分层 retrieval→rerank→answer 评测、消融、切片和索引更新/删除 |
| Agent/平台 | tool/runtime、状态、审批、恢复、互操作 | 越权、重复 effect、循环、pending、跨系统不一致 | 状态机、schema、IAM、幂等、队列/事务、MCP/A2A、verifier | proposal/execution 分离、typed approval、崩溃恢复、outbox 与安全 trace |
| 模型/训练算法 | 数据配方、SFT/PEFT、RM/DPO/PPO、checkpoint | loss/梯度异常、泄漏、过拟合、能力/安全回归 | Transformer、PyTorch/JAX、优化、数据、分布式、实验设计 | 数据门禁、训练恢复、baseline/消融、held-out 与多切片回归 |
| 推理/系统 | runtime、serving、kernel、调度、容量 | OOM、吞吐不足、尾延迟、取消/抢占/缓存错误 | GPU/内存、KV、量化、batching、并行、profiling、网络服务 | 固定 offered-load 下的 TTFT/TPOT/吞吐、显存账本和瓶颈归因 |
| 评测/数据 | case、标注、benchmark、统计、发布 gate | 指标无效、标注偏差、污染、切片退化、错误发布 | sampling、taxonomy、统计、judge 校准、数据治理、实验平台 | 可复算 case/result/manifest、人工 judgment、区间、切片和门禁 |
| 安全/治理平台 | policy、红队、审计、模型/系统卡、事件流程 | 注入、泄漏、滥用、越权、供应链与证据断裂 | 威胁建模、IAM、隐私、安全测试、治理流程、跨团队沟通 | 攻击路径、模型外控制、可重放证据、exception/incident/retirement 闭环 |

### 应用工程与算法工程怎样区分

两者都可能写 RAG、训练脚本和评测。差别更常出现在主要优化对象：

- 应用工程把模型作为不稳定依赖，优化端到端任务、状态、接口、可靠性和成本；
- 算法工程把数据、目标函数、模型行为和实验作为主要对象，优化可归因的质量与泛化；
- 搜索算法位于两者之间，既要做 ranking/数据实验，也要承担索引和在线检索约束；
- 平台/系统岗位把可复现执行、容量、隔离和故障恢复作为主要对象。

LangChain、Transformers 或 vLLM 都可能出现在多条路线中。真正区分岗位的，是你要守住哪类 invariant，
以及出现失败时由你做什么决定。

## 能力等级：知道、实现、验证、负责

本页用四级证据描述能力，不按工作年限或职称划线：

| 级别 | 能回答什么 | 证据 |
|---|---|---|
| K — Know | 机制是什么、适用边界在哪里 | 能推导/解释并指出反例 |
| I — Implement | 能否从输入到输出实现最小闭环 | 代码、单元测试、固定 fixture |
| V — Validate | 怎样证明实现和改动有效 | baseline、逐 case 结果、消融、统计、失败测试 |
| O — Own | 出现漂移、过载、越权或事故时怎样处置 | SLO、监控、runbook、回滚、权限与事件证据 |

以一个 RAG 项目为例：能解释 BM25 和 reranker 属于 K；从摄取到引用跑通闭环属于 I；
用逐 case baseline、消融和 held-out 证明改动有效属于 V；负责 SLO、权限、发布、回滚和事故处理才进入 O。
这四级描述的是证据范围，不按工作年限或职称自动升级。

### 角色能力矩阵

下表只是准备基线，应按目标 JD 调整。`V/O` 表示面试中通常需要能给出验证或 ownership 证据，而不是所有公司都使用相同门槛。

| 能力域 | 应用/后端 | RAG/搜索 | Agent/平台 | 模型/训练 | 推理/系统 | 评测/数据 | 安全/治理 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python、测试、Linux、Git | V | V | O | V | O | V | V |
| Transformer/tokenizer/generation | I | I | I | V | V | I | K |
| 数据、切分、lineage | I | V | I | O | K | O | V |
| RAG/IR/ranking | V | O | I | K | K | V | I |
| Agent state/tool/IAM | V | I | O | I | I | V | O |
| 训练/PEFT/对齐 | I | K | K | O | I | V | I |
| serving/KV/量化/容量 | V | I | V | I | O | V | I |
| 统计、切片、实验设计 | V | V | V | O | V | O | V |
| 安全、隐私、治理 | V | V | O | V | V | V | O |
| 产品约束与跨团队交付 | O | V | O | V | O | O | O |

矩阵的价值是暴露证据缺口。例如自评“训练 O”却没有可恢复 checkpoint、数据 lineage、held-out regression 或失败诊断，说明等级写高了，而不是应该再背更多名词。

## 从已有背景迁移

不必从头平均学习所有主题。先复用已有强项，再补相邻的 LLM 特有边界。

### 后端/全栈工程师

优先路线：LLM 应用、Agent 平台、评测平台。可复用 API、数据库、队列、可观测和运维；重点补 tokenizer/template、非确定生成、RAG 分层评测、模型外权限和 evidence-driven release。不要把成熟后端能力丢掉，只展示 Prompt demo。

### 搜索、推荐或数据工程师

优先路线：RAG/搜索、评测数据、训练数据。可复用 retrieval/ranking、特征/数据管道、A/B 与指标；重点补生成忠实度、context packing、LLM judge 边界、token 成本和多租户 ACL。

### 传统 ML/深度学习工程师

优先路线：模型训练、RAG 算法、评测。可复用训练、优化和实验设计；重点补 chat template、tool/RAG 系统、服务约束、数据/模型治理和目标 workload，而不是只跑公开 benchmark。

### 系统、HPC、编译器或 SRE 工程师

优先路线：推理系统、训练平台、Agent 平台。可复用 profiling、并行、内存、调度和可靠性；重点补 Transformer/KV、generation quality、量化回归和模型版本契约。性能优化必须经过质量 gate。

### 安全、隐私或治理工程师

优先路线：AI 安全平台、Agent/RAG 控制面、评测治理。可复用 threat modeling、IAM、审计和 incident response；重点补模型输出的不确定性、Prompt injection、tool candidate/action 分离、数据污染和可执行评测。不要把模型拒答当作安全边界。

### 研究生或研究背景

优先路线取决于能否把假设转成工程证据。论文复现应补环境、数据、代码质量、失败测试、成本与部署边界；单个 best seed 或作者 benchmark 不能替代可复算实验和软件交付。

## 怎样读一份 JD

先把营销名词删掉，只抽取五类信息：

| 信息 | 从 JD 找什么 | 应形成的问题 |
|---|---|---|
| 动词 | 研究、训练、构建、部署、优化、运营、评测、治理 | 日常主要是在提出方法、写系统还是承担服务？ |
| 对象 | 数据、模型、retriever、runtime、Agent、平台、客户方案 | 主要 artifact 和上下游是谁？ |
| 规模 | 单卡/集群、QPS、数据量、租户、区域、端侧 | 哪些容量/分布式能力是真的必要？ |
| 生命周期 | prototype、上线、on-call、迭代、事故、合规 | 是否需要 O 级证据？ |
| 评价 | 质量、收入、SLO、GPU 利用率、安全、交付速度 | 团队怎样判断工作成功？ |

### 建立 JD—证据映射

对每条职责写四列：

```text
JD 原文 | 我的证据 | 证据等级/边界 | 缺口与下一实验
```

例如“优化大模型推理性能”不能只映射到“使用 vLLM”。更强的映射是：固定 checkpoint/runtime/workload，保存 TTFT/TPOT/吞吐/峰值显存和 quality delta，定位 prefill/decode/queue/KV 瓶颈，并写出没有执行 CUDA kernel profiling 时不能声称什么。

JD 中的技术栈是约束，不是完整能力定义。会某个框架但无法解释输入契约、失败模式和替代方案，证据仍然弱；没有使用同名框架但拥有同层 invariant、benchmark 和迁移能力，可能是可解释的相邻证据。

### Must-have 与 nice-to-have

不要只数关键词命中率。优先判断：

1. 是否具备岗位主要失败责任对应的基础能力；
2. 是否有一项可以深挖到原始 artifact 的主证据；
3. 缺失技术是同层工具迁移，还是缺少整个能力域；
4. 对地域、许可、学历、工作授权或 on-call 等硬条件是否满足；
5. 能否诚实解释未做过的规模和目标环境。

工具 API 的短期缺口通常比“从未做过实验设计/系统 ownership”更容易补，但具体判断仍由目标团队决定。

## 作品集组合：一条主线加一条相邻能力

可信组合通常是**一个主项目 + 一个相邻项目 + 一项底层实现**，而不是四个换壳聊天 demo。

| 主路线 | 主项目 | 相邻项目 | 底层实现 |
|---|---|---|---|
| 应用/RAG | [可诊断 RAG](../practice/projects/rag-foundations.md) | [评测门禁](../practice/projects/evaluation-gate.md) | BM25、context packing 或引用 verifier |
| Agent/平台 | [Safe Agent](../practice/projects/safe-agent.md) | [云 API 契约](../practice/projects/cloud-api-contracts.md) | typed state、outbox、budget ledger |
| 模型/训练 | [单 GPU 微调](../practice/projects/single-gpu-finetuning.md) | [合成数据审计](../practice/projects/synthetic-data-audit.md) | attention、LoRA、checkpoint resume |
| 推理/系统 | [推理服务](../practice/projects/inference-serving.md) | [Transformers 基础](../practice/projects/transformers-basics.md) | sampling、KV allocator、batch scheduler |
| 评测/数据 | [评测门禁](../practice/projects/evaluation-gate.md) | RAG 或 Agent 主项目 | paired/cluster statistics、artifact verifier |

主项目证明你能把模糊需求变成系统和发布决定；相邻项目证明你理解上下游；底层实现证明你不是只会调用框架。它们可以共享代码和数据，但必须避免把同一次实验包装成三份独立证据。

### 证据等级与简历措辞

本仓库使用：L0 文档、L1 最小实现、L2 可复现实验、L3 工程样例、L4 生产设计。L4 表示容量、安全、监控、回滚和成本设计完整，仍不自动证明真实生产流量、组织控制或 availability。

| 当前证据 | 可以写 | 不应写 |
|---|---|---|
| CPU deterministic fixture | 实现并验证公式/状态机/错误路径 | GPU 加速、生产吞吐 |
| authored/offline case | 在固定 N 条回放上得到某结果 | 真实用户整体提升 |
| 目标 checkpoint 单点 control | 在固定 revision/环境/输入下执行 | 模型总体质量、跨硬件兼容 |
| loopback HTTP integration | 真实本机进程/TCP/HTTP 路径 | 公网、TLS、IAM、多区域可用 |
| 生产设计与故障演练 | 设计/演练某项回滚或控制 | 未实际承担的 SLA、事故或用户规模 |

没有原始 artifact、代码 revision、运行配置和失败样本时，删除精确数字或补证据。截图只能辅助展示，不能替代可重算结果。

## 简历 bullet 的证据语法

一个可审计 bullet 可以写成：

> 在 `[数据/流量/硬件/权限约束]` 下，设计并实现 `[关键机制]`；相对 `[基线]`，在 `[固定 workload 与切片]` 上将 `[指标]` 从 A 改为 B，并以 `[安全/统计/回滚 gate]` 限定发布；未覆盖 `[目标环境边界]`。

其中 A/B 只能替换为实际结果。不要为了句子短而删除分母、baseline 或环境；可以把完整证据放在项目 README/report，在简历保留可下钻链接。

数字反向审计链：

```text
resume bullet
  -> summary report
  -> per-case/per-request result
  -> workload/data manifest
  -> code/config/model revision
  -> raw failure examples
```

链路断在任一层，就不能把数字写成已验证事实。无密钥 SHA-256 只能帮助发现漂移，不认证执行者、时间或来源。

## 面试环节怎样准备

| 环节 | 所有路线共同要求 | 路线差异 |
|---|---|---|
| 编码 | 正确性、边界、测试、复杂度、可读性 | 算法偏张量/统计，系统偏并发/缓存，应用偏 API/data |
| 基础机制 | Transformer、tokenizer、generation、训练与评测边界 | 深度按岗位主要对象增加 |
| 项目深挖 | 数据、baseline、指标、失败、取舍、本人贡献 | 必须能打开 artifact 或重建关键数字 |
| 实验设计 | 假设、控制变量、切片、不确定性、停止条件 | 算法/评测权重更高，但任何路线都不能缺失 |
| 系统设计 | 需求量化、数据流/控制流、容量、安全、降级 | 应用/Agent/推理强调不同瓶颈 |
| 故障排查 | 从现象到分层假设、最小观测、止损、复盘 | RAG 召回、训练 NaN、OOM、尾延迟、重复 effect 等 |
| 行为/协作 | 决策、冲突、失败、ownership、跨团队沟通 | 只讲成功故事无法证明事故处理能力 |

### 项目深挖的六层回答

1. **问题**：用户/系统目标和硬约束是什么？
2. **基线**：为什么选择它，成功定义是什么？
3. **机制**：你改变了哪一层，哪些层保持固定？
4. **证据**：case、指标、区间、切片和原始失败在哪里？
5. **取舍**：质量、风险、延迟、成本和复杂度怎样变化？
6. **边界**：当前证据不能支持什么，下一项最可能推翻结论的实验是什么？

回答“我们用了某框架所以效果更好”会在第二、三和四层同时断裂。

### 故障题的回答顺序

先止损，再分层，再观测，最后改动：

```text
影响范围/安全风险
-> rollback、限流、禁用副作用或降级
-> input/data/model/runtime/system 分层假设
-> 最小日志、trace、metric、replay
-> 一次只改变一个主要变量
-> 回归集与发布 gate
-> postmortem 和长期预防
```

不要一上来“换更大模型”或“调 Prompt”。

## 如何判断职级与 ownership

职级不能仅由年限、模型规模或论文数量推断。可以用责任范围自查：

- **在明确问题内交付**：能实现、测试并解释一个模块；
- **独立闭环**：能澄清需求、选 baseline、验证、上线/交付并处理常见故障；
- **跨系统 ownership**：能定义接口/SLO、协调上下游、管理迁移和高风险失败；
- **组织级杠杆**：能建立平台、标准、评测或决策机制，让多个团队减少重复风险。

这些不是统一晋升标准。面试时用实际决策和结果说明范围，不要自封“架构师”或把团队成果全部写成个人贡献。

## 选择团队时也要做反向评估

岗位是否值得选择不只看模型名称。可以询问：

- 团队拥有的是 demo、内部平台还是有 SLO 的产品？
- 是否能访问失败样本、用户反馈、数据 lineage 和真实评测？
- 模型/数据/GPU/API 预算如何分配，谁决定发布？
- 是否有权限、安全、隐私、合规和 incident owner？
- 工作主要是短期客户定制、长期平台还是研究探索？
- 失败时能否回滚，是否需要 on-call，如何做复盘？
- 个人能否拥有端到端结果，还是只维护不可观测的中间环节？

没有生产流量不代表岗位没有价值，研究或平台工作也可以形成强证据；关键是团队是否诚实定义成功、保留原始结果并允许失败被发现。

## 常见准备误区

### 同时准备所有路线

四处各做一个浅 demo 会削弱项目深挖。先选主要失败责任，再补一条相邻能力；通用底座仍包括 Python、Transformer、测试、实验设计和系统边界。

### 用课程完成度代替能力证据

看完章节、拿到证书或运行 notebook 不等于能独立验证和负责。把学习产出转换为代码、测试、报告、失败样本和 decision record。

### 只展示最好结果

面试官更关心失败如何分类、为什么归因、怎样回滚。保存负结果、被淘汰方案和阈值选择，不要只放最好的一次 seed。

### 夸大证据等级

CPU fixture 不证明 GPU 吞吐，loopback 不证明公网服务，作者数据不证明真实用户，schema-valid 不证明业务正确，模型拒答不证明权限安全。

### 把框架熟练度当核心壁垒

框架会变，稳定能力是识别输入输出契约、构造正确基线、定位失败、验证修改并控制发布。熟练使用框架仍有价值，但必须能解释它替你做了什么。

## 求职决策记录模板

为每个目标岗位维护一页，而不是维护一个无限增长的“LLM 技能清单”：

```markdown
# Role decision record

## Target
- company/team/JD snapshot:
- primary failure ownership:
- main deliverables and scale:

## Evidence map
- strongest K/I/V/O evidence:
- flagship project and artifact links:
- adjacent evidence:

## Gaps
- hard missing domain:
- tool-level migration:
- claim I must not make:
- smallest falsifiable next experiment:

## Interview
- likely coding/mechanism/experiment/system loops:
- three project decisions to defend:
- two failures and one conflict story:

## Decision
- fit, learning surface and risks:
- questions for the team:
```

这个模板不要求公开公司机密。真实工作证据应脱敏、获得授权，并避免上传客户数据、密钥、内部 Prompt、日志或专有代码。

## 完成检查

- [ ] 用失败责任而不是 title 选择了一条主线。
- [ ] 对一个真实 JD 完成了动词、对象、规模、生命周期和评价拆解。
- [ ] 每个 must-have 都有证据、相邻证据或明确缺口。
- [ ] 有一个主项目能下钻到逐 case/逐请求 artifact。
- [ ] 能解释当前证据的 L0–L4 等级和不能外推的边界。
- [ ] 简历每个数字都能反向追到 workload、配置、代码 revision 和失败样本。
- [ ] 能完成机制、实验、系统设计、故障和项目深挖五类追问。
- [ ] 没有把 authored fixture、CPU control 或生产设计写成真实线上结果。
- [ ] 准备了对团队的数据、预算、ownership、发布和事件流程问题。

完成这些检查不保证录用；它证明的是你的求职材料内部一致、可核验，并与目标岗位责任相匹配。
