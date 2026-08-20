# 学习路径

路径不是考试，也不要求一次补齐所有数学。选择最接近当前目标的一条，完成一个可展示的成果后再扩展。

| 目标 | 路线 | 建议投入 |
|---|---|---:|
| 从零理解 LLM | [基础路线](#beginner) | 6 周 |
| 构建 RAG 或 Agent | [应用工程路线](#application) | 6–8 周 |
| 学习微调与训练 | [模型工程路线](#model-engineering) | 6–10 周 |
| 学习推理部署 | [系统工程路线](#systems) | 6–10 周 |
| 做论文复现 | [研究路线](#research) | 按项目 |

## 基础路线 { #beginner }

适合第一次系统学习 LLM 的读者。每周约 5–8 小时。

| 周 | 学习内容 | 可交付成果 |
|---|---|---|
| 1 | [机器学习](../foundations/ml-dl.md)与 [NLP](../foundations/nlp.md) | 解释训练/验证/测试集，并从 logits 算一次 NLL |
| 2 | [Tokenization](../core/tokenization.md) | 比较中英文、数字、代码和 emoji 的 byte/token 差异 |
| 3 | [Transformer](../core/transformer.md) | 手算两 token 注意力，标注 `Q/K/V/score` 形状 |
| 4 | [生成入门](../core/generation-basics.md) | 比较 greedy、temperature 与 top-p 的输出分布 |
| 5 | [RAG](../applications/rag.md)与[请求生命周期](../applications/rag-request-lifecycle.md) | 跟踪一次回答与一次拒答的证据链 |
| 6 | [评测](../quality/evaluation.md)与[安全](../quality/safety.md) | 完成 30 条样例、错误分类和一个越权负例 |

完成标准：能从文本输入一路解释到模型输出，并为一个小任务建立可重复的评价方法。

## 应用工程路线 { #application }

先修：基础路线第 2–6 周，外加 Python、HTTP 和数据库基础。

1. **定义任务**：用 [Prompt](../applications/prompting.md) 写清输入、输出、拒答和结构化字段。
2. **建立 RAG 基线**：先走完[请求生命周期](../applications/rag-request-lifecycle.md)与
   [实验 5](../practice/labs/lab-5-rag-request.md)，再学习[摄取](../applications/rag-ingestion.md)、
   [检索](../applications/rag-retrieval.md)、[检索表示学习](../applications/retrieval-learning.md)和
   [引用](../applications/rag-generation.md)。
3. **加入权限与失败处理**：在[生产 RAG](../applications/rag-production.md)中处理 ACL、索引更新和无答案情况。
4. **引入 Agent**：先跟完[一次退款任务](../applications/agent-task-lifecycle.md)，再学习
   [架构](../applications/agent-architecture.md)、[决策理论](../applications/agent-decision-theory.md)与
   [Runtime](../applications/agent-runtime.md)，把模型输出视为提议而不是授权。
5. **建立评测**：先用[评测测量学](../quality/evaluation-measurement.md)验证 rubric、标注与指标解释，再按[评测方法](../quality/evaluation-methodology.md)区分任务成功、引用、工具执行和安全指标。
6. **完成项目**：从 [RAG Foundations](../practice/projects/rag-foundations.md) 或 [Safe Agent](../practice/projects/safe-agent.md) 选择一个项目。

完成标准：系统对无答案、冲突证据、提示注入、越权请求、工具超时和重复请求都有可测试的行为。

## 模型工程路线 { #model-engineering }

先修：基础路线第 1–4 周，熟悉 PyTorch 和基本优化算法。

1. [数据工程](../training/data.md)：建立数据 schema、切分、去重和污染检查。
2. [预训练](../training/pretraining.md)：理解 token budget、优化器、稳定性和 checkpoint。
3. [微调总览](../training/finetuning.md)：先判断问题是否真的需要改权重。
4. [SFT 数据流水线](../training/sft-data-pipeline.md)：处理模板、mask、截断和 held-out 数据。
5. [LoRA/QLoRA](../training/peft-qlora-engineering.md)：建立显存预算、训练基线和 adapter 发布流程。
6. [偏好对齐](../training/alignment-basics.md)：理解偏好数据、DPO/RLHF 和 reward hacking 风险。
7. 完成 [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md) 项目中的一个小实验。

完成标准：报告数据版本、训练预算、曲线、基线、held-out 结果、失败样例和资源消耗；“loss 下降”不能单独作为完成信号。

## 系统工程路线 { #systems }

先修：基础路线第 2–4 周，熟悉 Linux、HTTP 和性能测量。

1. [推理基础](../systems/inference.md)：区分 prefill、decode 和 KV Cache。
2. [请求生命周期](../systems/inference-request-lifecycle.md)：沿一次请求串起调度、KV、采样、流式和终态。
3. [Paged KV 实验](../practice/labs/lab-7a-paged-kv.md)：先预测 block table，再验证 COW 与容量失败。
4. [推理优化](../systems/inference-optimization.md)：从 TTFT、TPOT、吞吐和容量症状选择技术。
5. [vLLM 服务](../systems/vllm-serving.md)：启动服务并固定模型、请求和采样配置。
6. [服务与可观测性](../systems/serving.md)：测量排队、错误、资源和回滚。
7. [硬件与端侧](../systems/hardware-edge.md)：用算力、带宽和容量账本解释瓶颈。
8. 完成 [Inference Serving](../practice/projects/inference-serving.md) 项目的一次压测与故障实验。

完成标准：能说明负载模型、计时边界、资源上限和失败恢复；不使用单次延迟或平均值代表容量。

## 研究路线 { #research }

研究路线从问题开始，不从“复现一个热门模型”开始。

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

无论选择哪条路线，都保留四样东西：

1. 一个能运行的最小基线；
2. 一个主动制造的失败案例；
3. 一组事先定义的评价标准；
4. 一段用自己的话写出的结果解释。

下一步：查看[知识地图](knowledge-map.md)确认先修，或直接从[项目索引](../practice/project-index.md)选择与路线匹配的成果。
