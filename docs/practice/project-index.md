# 工程项目索引

项目用于把一组知识变成可以运行、解释和改进的系统。本页帮助你选项目、获得第一次成功并完成一次验收。
精确版本、固定输入、hash 和历史运行结果统一放在[项目实验与证据台账](../evidence/project-controls.md)。

**项目导航**：[实验目录](labs.md) · [学习路径](../guide/learning-paths.md) · [环境配置](../guide/environment.md) · [生产检查表](production-checklist.md)
{ .doc-nav }

## 先按成果选择项目

不要从技术栈名称开始。先决定你希望交付什么，再选择最小项目。

| 想完成的成果 | 建议项目 | 第一次成功 | 做深一层 |
|---|---|---|---|
| 理解 Transformer 内部 | [Transformers Basics](projects/transformers-basics.md) | 跑通 BPE、attention 和 generation toy | 比较 cache、量化或 MoE 的正反例 |
| 理解函数式训练 | [JAX MiniGPT](projects/jax-minigpt.md) | 在 CPU 上让 tiny model 过拟合小数据 | 对账 PyTorch/JAX 梯度并验证恢复 |
| 构建可诊断 RAG | [RAG Foundations](projects/rag-foundations.md) | 跑通一次 answer 与一次 abstain | 加入持久化、模型、服务与分层评测 |
| 比较 RAG 框架 | [RAG Framework Adapters](projects/rag-framework-adapters.md) | 同一检索结果通过两个框架往返 | 检查 metadata、rank 与权限是否漂移 |
| 构建安全工具 Agent | [Safe Agent](projects/safe-agent.md) | 跑通一次退款的九阶段生命周期 | 再接真实 Planner、框架、协议与外部服务 |
| 完成单卡微调闭环 | [Single-GPU Finetuning](projects/single-gpu-finetuning.md) | 审计一批最终 labels | 比较 Prompt、RAG、LoRA 与 held-out 结果 |
| 理解云 API 契约 | [Cloud API Contracts](projects/cloud-api-contracts.md) | 解析一次 typed response | 加入流式、重试、预算和不确定结果 |
| 部署和测量推理服务 | [Inference Serving](projects/inference-serving.md) | 跑通采样与最小 HTTP 服务 | 测量排队、TTFT、TPOT、显存和取消 |
| 建立发布评测门禁 | [Evaluation Gate](projects/evaluation-gate.md) | 对固定 cases 重算指标 | 加入配对比较、切片与发布判断 |
| 审计合成数据 | [Synthetic Data Audit](projects/synthetic-data-audit.md) | 验证一份 lineage artifact | 加入去重、verifier 和训练暴露账本 |

如果这是你的第一个工程项目，优先选 RAG Foundations。它能同时练习数据、检索、权限、生成和评测，又不要求先拥有 GPU。

## 一个项目分三次完成

### 1. 最小成功

只运行一条主命令，确认输入、输出和一个关键中间状态。此时目标是理解链路，不是把所有测试跑一遍。

交付物：运行命令、原始输出，以及三句话说明“发生了什么、为什么、还不能说明什么”。

### 2. 探索实验

主动改变一个变量并制造一个失败。例如移除 RAG 证据、改变 LoRA rank、压低服务容量，或让 Agent 工具返回非法结果。先写预测，再运行。

交付物：基线、变量、结果、失败样例和解释。没有对照的成功截图不算实验。

### 3. 工程验收

最后再检查权限、恢复、并发、回滚和发布门禁。CPU 固定样例可以验证控制流；GPU 性能、云端计费和
真实组织权限仍必须在目标环境重新测量。

交付物：验收表、机器可读结果和清晰的证据边界。精确运行身份记录在项目 README 或[证据台账](../evidence/project-controls.md)，不塞回学习笔记。

## 怎样阅读证据等级

| 等级 | 读者可以相信什么 | 仍不能外推什么 |
|---|---|---|
| L0 文档 | 设计与边界写清楚 | 代码已经工作 |
| L1 最小实现 | 固定输入能执行 | 代表性质量与可靠性 |
| L2 可复现实验 | 局部机制和负例可复算 | 目标硬件或真实流量表现 |
| L3 工程样例 | 多组件闭环和故障路径可运行 | 组织级生产责任已经落实 |
| L4 生产设计 | SLO、权限、容量和回滚方案完整 | 未实际运行的生产结果 |

等级描述的是已有证据，不是项目“高级程度”。不能用更多单元测试弥补缺失的真实硬件、数据或权限证据。

## 三条最小验收路径

### 路径 A：RAG 权限与超时 { #arag }

~~~powershell
python projects/rag-foundations/rag_service_control.py
python -m pytest tests/test_rag_service.py -q
~~~

运行前先预测不同身份可见的 source。运行后确认未授权正文不会进入排序、缓存或生成；再区分“客户端收到超时”“底层工作终止”和“并发许可释放”，三者不是同一事件。

故意破坏：把 ACL 移到 rerank 后，或在同步工作尚未退出时释放 permit。最终答案可能仍正常，但安全或容量契约应失败。

### 路径 B：Agent 授权与恢复

~~~powershell
python projects/safe-agent/refund_lifecycle.py
python -m pytest tests/test_agent_refund_lifecycle.py -q
~~~

沿 trace 标记 observation、proposal、schema、ACL、approval、execution、idempotency、verifier 和 recovery。
重点解释为什么远端已受理而本地超时时，不能报告失败或直接重试。完成后再运行
`model_planner_control.py`，把固定 proposal 换成带完整请求、响应与结构校验记录的模型边界。

故意破坏：用旧审批批准新参数，或把远端 `completed` 直接当成本地成功。两种情况都必须被控制面拒绝。

### 路径 C：评测差异是否足以发布 { #acceptance-evaluation }

~~~powershell
python projects/evaluation-gate/paired_randomization_toy.py
python projects/evaluation-gate/clustered_bootstrap_toy.py
python -m pytest tests/test_evaluation_statistics.py -q
~~~

先手算 paired difference，再比较 case-level 与 cluster-level 重采样。统计显著不自动代表差异足够大、指标有效或用户受益。

故意破坏：删除零差异 case、反复尝试指标只保留显著结果，或看到结果后修改门槛。程序仍可能输出合法数字，但发布结论已经失效。

## 推荐顺序

1. 用 RAG Foundations 理解数据、召回和 ACL。
2. 用 Evaluation Gate 固定 case、基线和错误分类。
3. 在微调或服务项目中只改变一个主要变量。
4. 最后让 Safe Agent 调用已经验证的检索和工具。

## 项目完成标准

一个可展示项目至少包含：

- 清楚的输入、输出、数据和权限边界；
- 一个简单基线和一个主动制造的失败样例；
- 质量、安全、延迟或成本中的相关指标；
- 可复制的运行命令与原始结果；
- 已知限制，以及下一项最可能推翻当前结论的实验。

只有 README 和架构图不算完成；只有程序能运行但没有评价方法，也不算完成。
