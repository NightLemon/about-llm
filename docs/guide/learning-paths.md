# 学习路径

路径不是门槛。遇到不懂的数学时按需回补，不必先修完全部微积分才开始。

第一次学习时先用[新手知识地图](beginner-map.md)完成自检和最小运行；准备环境时看[环境与硬件矩阵](environment.md)，已有明确目标时可直接选择下面的路径。完整覆盖范围保留在[完整知识地图](knowledge-map.md)。

## 怎样判断自己真的学会了

不要用“读完章节”作为进度。每一阶段都完成下面四步：

1. **解释机制**：不用术语堆砌，说明输入如何经过中间状态得到输出，并写出关键张量或数据结构的形状。
2. **运行基线**：先运行仓库中的确定性 CPU control，保存命令、环境、原始输出和随机种子。
3. **制造反例**：主动破坏一个假设，例如移除 causal mask、把 ACL 放到检索后、把超时请求当作未发送。
4. **通过门禁**：用测试、指标阈值或不变量判定结果；“输出看起来合理”不是通过条件。

推荐为每个阶段保留一份简短实验记录：`问题 → 假设 → 基线 → 改动 → 观察 → 结论边界`。其中结论边界必须写明当前证据不能外推到哪些模型、数据、硬件或生产环境。

## 入门路径：从使用者到理解者

建议 6 周，每周 5–8 小时。

| 周 | 主题 | 可交付成果 |
|---|---|---|
| 1 | [Python/张量与机器学习](../foundations/ml-dl.md)、[NLP 基础](../foundations/nlp.md) | 实现词频与简单分类器；解释训练/验证/测试集 |
| 2 | [Tokenization](../core/tokenization.md)、Embedding 与语言建模 | 检查不同文本的 token 数；计算交叉熵和困惑度 |
| 3 | [Transformer、位置与注意力](../core/transformer.md) | 手算一次缩放点积注意力；标注各张量形状 |
| 4 | [生成入门](../core/generation-basics.md)、[Prompt](../applications/prompting.md)、结构化输出 | 比较温度与 top-p；建立 30 条小型评测集 |
| 5 | [RAG](../applications/rag.md) 与[向量检索](../applications/rag-retrieval.md) | 构建带引用问答；测 Recall@k 和答案忠实度 |
| 6 | [安全](../quality/safety.md)、[评测](../quality/evaluation.md)与[综合项目](../practice/labs.md) | 完成威胁模型、错误分析和项目报告 |

### 入门路径的阶段门禁

**第 1–2 周：从文本到损失。** 需要解释字符、UTF-8 byte、token id 和 embedding 向量不是同一层对象；能够从一组 logits 手算 softmax、单 token NLL 与序列平均 NLL。常见误判是把 tokenizer 的词表大小当作模型能表达的语义数量，或直接比较不同 tokenizer 下的 PPL。通过标准是：对中英文、数字、代码和 emoji 给出 token/byte 统计，并解释比较口径。

**第 3 周：从张量到注意力。** 需要写出 `Q: [B,H,T,D]`、`K/V: [B,H_kv,T,D]` 与 score `[B,H,T,T]` 的关系，解释 causal mask 在 softmax 前为何使用负无穷，以及 GQA 中 query head 如何映射到 KV head。常见误判是只比较最终输出而不检查 mask、位置和 cache。通过标准是：手算一个两 token 样例，并验证逐 token KV Cache logits 与完整因果前向在容差内一致。

**第 4 周：从生成到评价。** 温度改变 logits 尺度，top-k/top-p 改变候选 support，两者不是“创造力按钮”。结构化输出的 JSON 可解析也不代表字段合法、事实正确或动作已授权。通过标准是：固定 logits 和 uniform sample，复算至少两种采样配置；为 30 条样例预先定义评分规则，并保存失败样例而非只报平均分。

**第 5–6 周：从检索到系统。** RAG 要把“相关文档是否进入 top-k”“答案是否受证据支持”“引用是否指向正确 span”分开评价；安全测试要覆盖无答案、冲突证据、提示注入和跨权限检索。通过标准是：报告 Recall@k、答案/引用错误分类和至少一个权限负例；如果检索失败，不能用生成端措辞优化掩盖召回缺口。

## 工程路径：从原型到生产

建议 8–12 周，有 Python、HTTP、数据库和基本深度学习经验。

1. **模型接口层**：先读[云 API 契约](../models/cloud-api-contracts.md)，掌握消息协议、流式输出、超时、重试、幂等和结构化输出；再读 [Opaque Reasoning 工件安全](../quality/reasoning-artifact-security.md)并完成[实验 0D](../practice/labs/lab-0d-reasoning-artifact-security.md)，区分 provider API、有状态交互接口、客户端 opaque state 与本地内部契约。
2. **[RAG 数据层](../applications/rag-production.md)**：解析、切分、元数据、索引、混合检索、重排、引用。
3. **[Agent 执行层](../applications/agent-runtime.md)**：工具 schema、权限、状态机、循环上限、人工确认；依次运行 [MCP official-SDK memory](../applications/agent-interoperability.md#mcp-sdk-memory-control)、[official-SDK stdio](../applications/agent-interoperability.md#mcp-sdk-stdio-control)、[official-SDK Streamable HTTP](../applications/agent-interoperability.md#mcp-sdk-streamable-http-control)、[authored strict stdio](../applications/agent-interoperability.md#mcp-stdio-control)、[authored Streamable HTTP](../applications/agent-interoperability.md#mcp-streamable-http-control) 与 [A2A 1.0 loopback](../applications/agent-interoperability.md#a2a-loopback-control)，比较官方 SDK、memory/pipe/HTTP transport、raw framing、JSON/SSE/session/cancel、Agent Card/binding、错误分层和 verifier；任何协议连接或 schema-valid 都不能当作授权。
4. **[评测层](../quality/evaluation-methodology.md)**：黄金集、LLM-as-judge 校准、成对比较、线上实验和回归门禁。
5. **[服务层](../systems/serving.md)**：批处理、缓存、限流、[KV Cache 与量化](../systems/inference-optimization.md)、GPU 容量规划。
6. **[治理层](../quality/governance-impact.md)**：PII、提示注入、供应链、审计日志、数据保留和事故响应。

毕业标准：一个不是“演示即成功”的系统——有明确 SLO、版本化评测集、可追踪引用、失败降级、成本仪表盘和红队报告。

建议顺序：先完成云 API 契约、reasoning artifact replay matrix 与单 Agent runtime，再阅读 [Agent 互操作](../applications/agent-interoperability.md)。互操作层会扩大信任边界，不应早于身份、授权、上下文绑定、幂等、审批和 verifier。可从[工程项目索引](../practice/project-index.md)选择实现，并用[生产检查表](../practice/production-checklist.md)验收。

### 工程路径的决策门禁

| 阶段 | 必须回答的工程问题 | 失败信号 | 最低通过证据 |
|---|---|---|---|
| API 契约 | 哪些失败确定未发送，哪些结果未知且可能计费？opaque reasoning state 绑定谁、哪条会话和哪些模型？ | 超时后一律重试；只记录最终成功 usage；合法 signature 在任意上下文重放 | 请求 fingerprint、attempt ledger、uncertain reconciliation 与 context-bound replay matrix |
| RAG | 身份从哪里来？ACL 在正文进入 scorer、cache、trace 前还是后执行？ | tenant/principal 来自 body 或 Prompt；只测最终答案 | 未授权文档在检索前不可见；召回、生成、引用分层指标 |
| Agent | 模型输出是建议还是授权？副作用如何确认、幂等和恢复？ | schema 校验后直接执行；把远端 `completed` 当本地成功 | proposal/execution identity、typed approval、verifier、崩溃恢复测试 |
| 评测 | case、scorer、基线和统计方法是否版本化？差异是否有实际意义？ | 反复试指标直到显著；只看总体均值 | 预注册门槛、paired/cluster 统计、切片与完整失败清单 |
| 服务 | 排队、prefill、decode 和客户端观察分别怎样计时？超时是否停止底层工作？ | 只报平均延迟；504 后后台工作继续但 permit 已释放 | TTFT/TPOT 分位数、admission/backpressure、取消与容量故障测试 |
| 发布 | artifact、配置、tokenizer、Prompt 与代码 revision 如何绑定和回滚？ | 只保存模型名或单文件 hash；验证后重新打开可变路径 | immutable manifest、完整性/来源分离、canary 门禁与回滚演练 |

工程毕业标准不是所有项目都达到 L4，而是能准确陈述当前等级：哪些结论由本地确定性 control 支持，哪些需要真实网络、模型、GPU、并发流量或组织控制才能成立。

## 研究路径：从复现到提出问题

建议先掌握线性代数、概率、优化和 PyTorch。

1. 复现一个小型 decoder-only Transformer，从数据加载到生成全部自己写。
2. 复现一项单变量结果，例如位置编码外推、LoRA 秩或量化位宽的影响。
3. 阅读相互矛盾或结论不同的论文，找出数据、规模、预算和评价协议差异。
4. 写研究问题：假设、可证伪预测、基线、消融、算力预算、风险与停止条件。
5. 预先登记评价指标，报告负结果、方差和限制，避免只展示最好的一次运行。

### 研究路径的最小实验协议

一个可复查实验至少包含：

- **可证伪假设**：例如“在相同训练 token 和 optimizer steps 下，提高 LoRA rank 会改善领域 exact match，但可能增加通用能力回归”，而不是“研究 LoRA rank”。
- **受控变量**：固定数据 snapshot、样本顺序、tokenizer、初始化、训练预算和评价代码；如果无法同时固定训练 token 与 wall-clock，要声明选择了哪种公平性。
- **基线与消融**：至少包含最简单基线、目标方法和移除关键机制的消融。不同参数量或计算量的方法要单独报告成本。
- **重复与不确定性**：保存逐 seed 原始结果，报告均值、离散程度和配对差异；样本具有用户、文档等簇结构时，不把 case 当独立样本。
- **停止条件**：在运行前定义最大预算、无效结果条件和失败处理；不能看到中间结果后只延长表现较差的配置。
- **结论强度**：toy 数据上的机制复现只支持机制层结论；单模型、单数据集的提升不自动支持普遍优越性或生产收益。

建议在实验报告最后加入“最可能推翻本结论的下一项实验”。它通常比再增加一张最好结果表更有价值。

## 数学补给路径

按需求回补：

- 看不懂张量形状 → [数学基础](../foundations/math.md)中的向量、矩阵乘法、转置与广播。
- 看不懂 softmax/损失 → [概率与信息论](../foundations/math.md)以及[机器学习损失](../foundations/ml-dl.md)。
- 看不懂训练 → [导数、链式法则与自动微分](../foundations/math.md)，再读[预训练](../training/pretraining.md)。
- 看不懂采样 → [条件概率、熵与 KL 散度](../foundations/math.md)，再回到[生成与解码入门](../core/generation-basics.md)。
- 看不懂规模化实验 → [统计与实验设计](../foundations/math.md)、[规模化规律](../core/scaling.md)和[评测方法](../quality/evaluation-methodology.md)。

## 完成路径后

- 想把知识变成可运行证据：进入[实验与项目](../practice/labs.md)。
- 想了解代码、Notebook 与项目如何对应：阅读[仓库地图与实现契约](repo-map.md)。
- 想检查是否达到工程交付标准：使用[生产检查表](../practice/production-checklist.md)。
