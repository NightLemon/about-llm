# 学习路径

学习路径用来减少选择，不是给所有人安排同一张课表。先选最近要解决的问题，做出一个能运行、能解释的成果；
遇到缺口时再回补数学或相邻主题。

| 你现在想解决的问题 | 路线 | 第一个可交付成果 |
|---|---|---|
| 想知道 LLM 到底怎样工作 | [基础路线](#beginner) | 一次从文本、token 到生成结果的手算与实验 |
| 想构建 RAG 或 Agent | [应用工程路线](#application) | 一个会引用、会拒答、能处理权限失败的小系统 |
| 想训练或适配模型 | [模型工程路线](#model-engineering) | 一次数据、loss、adapter 和 held-out 结果可对账的训练 |
| 想部署和压测模型 | [系统工程路线](#systems) | 一份包含 TTFT、TPOT、吞吐、容量和失败终态的报告 |
| 想复现论文结论 | [研究路线](#research) | 一项带基线、消融和负结果的复现实验 |

## 基础路线 { #beginner }

这条路线适合第一次系统学习 LLM 的读者。六站按依赖排列，进度由右栏的成果决定，不按周数决定。

| 站点 | 先回答的问题 | 离开这一站时留下什么 |
|---|---|---|
| [机器学习与 NLP](../foundations/ml-dl.md) | 模型从什么数据学习，怎样知道它没有只记住训练集？ | 一次 train/validation/test 切分解释和 logits→NLL 手算 |
| [Tokenization](../core/tokenization.md) | 文本为什么会变成不同长度的整数序列？ | 中英文、数字、代码和 emoji 的 byte/token 对照 |
| [Transformer](../core/transformer.md) | 一个 token 怎样读取前文信息？ | 两 token Attention 手算和 `Q/K/V/score` shape 图 |
| [生成入门](../core/generation-basics.md) | 下一 token 怎样被选中，循环为什么停？ | Greedy、temperature、top-p 的同分布对照 |
| [RAG 请求](../applications/rag-request-lifecycle.md) | 模型怎样获得外部证据，并在证据不足时停下？ | 一次回答和一次拒答的完整证据链 |
| [评测与安全](../quality/evaluation.md) | 怎样判断系统变好，同时没有制造新的高风险失败？ | 小型 case set、错误分类和一个越权负例 |

完成标准：能从文本输入一路解释到模型输出，并为一个小任务建立可重复的评价方法。

## 应用工程路线 { #application }

应用路线最终要做出一个可诊断的助手。开始前应熟悉 token 与生成循环，并能使用 Python、HTTP 和数据库。
下面每一步都在给同一个系统补一项能力。

1. **先让任务可检查**：用 [Prompt](../applications/prompting.md) 写清输入、输出、拒答和结构化字段。
2. **给回答接上证据**：走完[请求生命周期](../applications/rag-request-lifecycle.md)与
   [实验 5](../practice/labs/lab-5-rag-request.md)，再学习[摄取](../applications/rag-ingestion.md)、
   [检索](../applications/rag-retrieval.md)、[检索表示学习](../applications/retrieval-learning.md)和
   [引用](../applications/rag-generation.md)。
3. **处理证据之外的现实问题**：在[生产 RAG](../applications/rag-production.md)中加入 ACL、索引更新和无答案路径。
4. **确实需要开放决策时再引入 Agent**：先跟完[一次退款任务](../applications/agent-task-lifecycle.md)，再学习
   [架构](../applications/agent-architecture.md)、[决策理论](../applications/agent-decision-theory.md)与
   [Runtime](../applications/agent-runtime.md)，把模型输出视为提议而不是授权。
5. **让失败可以被统计**：用[评测测量学](../quality/evaluation-measurement.md)检查 rubric 和标注，再按
   [评测方法](../quality/evaluation-methodology.md)分别记录任务成功、引用、工具执行和安全指标。
6. **收束成一个项目**：从 [RAG Foundations](../practice/projects/rag-foundations.md) 或
   [Safe Agent](../practice/projects/safe-agent.md) 选择一个，把前面的 trace 和失败样例保留下来。

完成标准：系统对无答案、冲突证据、提示注入、越权请求、工具超时和重复请求都有可测试的行为。

## 模型工程路线 { #model-engineering }

模型工程路线围绕一次权重更新展开：数据从哪里来，哪些 token 产生 loss，参数怎样改变，以及 held-out 行为
是否真的改善。开始前需要 PyTorch、反向传播和基本优化算法。

1. 从[数据工程](../training/data.md)建立 schema、切分、去重和污染检查。
2. 用[预训练](../training/pretraining.md)理解 token budget、优化器状态和 checkpoint 为什么必须一起保存。
3. 阅读[微调总览](../training/finetuning.md)，先确认问题是否需要改权重，还是 Prompt 或 RAG 已经足够。
4. 在 [SFT 数据流水线](../training/sft-data-pipeline.md)中打印最终 token、mask、截断和 held-out identity。
5. 进入 [LoRA/QLoRA](../training/peft-qlora-engineering.md)，建立显存预算、基线和 adapter 发布流程。
6. 需要偏好优化时，再学习[偏好对齐](../training/alignment-basics.md)中的 DPO/RLHF 与 reward hacking。
7. 最后用 [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md)把一个样本从模板追到独立重载。

完成标准：报告数据版本、训练预算、曲线、基线、held-out 结果、失败样例和资源消耗；“loss 下降”不能单独作为完成信号。

## 系统工程路线 { #systems }

系统工程路线追踪“一次请求怎样占用机器，又怎样释放资源”。开始前应理解 token、生成循环，并熟悉 Linux、
HTTP 和基本性能测量。

1. [推理基础](../systems/inference.md)：区分 prefill、decode 和 KV Cache。
2. [请求生命周期](../systems/inference-request-lifecycle.md)：沿一次请求串起调度、KV、采样、流式和终态。
3. [Paged KV 实验](../practice/labs/lab-7a-paged-kv.md)：先预测 block table，再验证 COW 与容量失败。
4. [Qwen3 + nano-vLLM 实验](../practice/labs/lab-7b-nano-vllm-qwen3.md)：在真实 GPU 上追踪
   waiting/running/finished、chunked prefill、prefix hit、decode 与 KV 释放。
5. [推理优化](../systems/inference-optimization.md)：从 TTFT、TPOT、吞吐和容量症状选择技术。
6. [vLLM 服务](../systems/vllm-serving.md)：启动服务并固定模型、请求和采样配置。
7. [服务与可观测性](../systems/serving.md)：测量排队、错误、资源和回滚。
8. [硬件与端侧](../systems/hardware-edge.md)：用算力、带宽和容量账本解释瓶颈。
9. 完成 [Inference Serving](../practice/projects/inference-serving.md) 项目的一次压测与故障实验。

完成标准：能说明负载怎样到达、每只时钟从哪里开始、资源上限在哪里，以及失败后何时真正释放。
容量结论应来自一段分布和一组终态，而不是某次请求的延迟截图。

## 研究路线 { #research }

研究路线从一个可能被实验推翻的问题开始。热门模型可以是实验对象，但不应代替研究问题。

1. 写出一个可证伪假设和最小实验。
2. 固定数据、模型、训练预算、评价代码和随机种子。
3. 设置简单基线、目标方法和至少一项关键消融。
4. 运行多个 seed，报告原始结果、差异和失败情况。
5. 解释证据只支持哪一层结论，并提出最可能推翻它的下一项实验。

可从[近期论文](../papers/index.md)、[规模化](../core/scaling.md)、[可解释性](../core/architectures-interpretability.md)或[前沿主题](../frontier/reasoning-long-context-moe.md)选题。

完成标准：别人能够根据你的记录重做实验，并得到相同方向的结论；负结果也必须保留。

## 数学按需补给 { #math-supplement }

- 看不懂矩阵和广播：读[数学基础](../foundations/math.md)的线性代数部分。
- 看不懂 softmax、交叉熵或 KL：读概率与信息论部分。
- 看不懂反向传播：读链式法则与自动微分，再回到[预训练](../training/pretraining.md)。
- 分不清一致性、正确性和指标有效性：先读[评测测量学](../quality/evaluation-measurement.md)。
- 看不懂显著性、功效和置信区间：读[评测测量学](../quality/evaluation-measurement.md)与[评测统计](../foundations/evaluation-statistics.md)，再回到[评测方法](../quality/evaluation-methodology.md)。

## 每个阶段怎样验收

无论走哪条路线，最后都检查桌面上是否留下了四样东西：

1. 一个能运行的最小基线；
2. 一个主动制造的失败案例；
3. 一组事先定义的评价标准；
4. 一段用自己的话写出的结果解释。

下一步：查看[知识地图](knowledge-map.md)确认先修，或直接从[项目索引](../practice/project-index.md)选择与路线匹配的成果。
