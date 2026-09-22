# LLM 面试题与回答方法

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备 LLM 应用、算法、训练、评测或推理岗位的开发者。
- **先修**：学过 Transformer、RAG、微调和基础工程实践中的至少一条主线。
- **首次阅读**：回答结构 → GQA 示例 → 题目地图 → 与目标岗位最相关的五题。
- **完成信号**：能在 30 秒给出结论，在两分钟内补充机制、边界和验证方法。
- **卡住时**：先用自己做过的一个项目回答，不要从术语定义开始背。

</div>

**求职导航**：[岗位路线](roadmap.md) · [应用与治理题](applied-questions.md) ·
[编码轮](coding-round.md) · [系统设计](system-design.md) · [行为面试](behavioral.md) ·
[简历项目](resume-projects.md) · [深挖验证台账](../evidence/interview-controls.md)
{ .doc-nav }

面试官通常在确认三件事：你是否理解机制，是否知道结论何时失效，以及能否设计实验把争论变成证据。
本页负责答题方法和选题，不再重复各专题教材。

## 一道题怎样回答

按“结论—机制—边界—验证”组织回答：

1. 用一句话直接回答；
2. 用公式、数据流或状态变化解释原因；
3. 指出成立条件、代价和最可能的反例；
4. 给出能推翻当前说法的最小实验或生产指标。

回答深度跟随追问逐层增加。30 秒说结论、一个机制和一个边界；两分钟再加入公式、权衡与验证；
白板深挖才展开 shape、复杂度和正反例；项目追问必须给环境、分母、原始结果和失败样例。

## 一场追问怎样逐层展开 { #gqa-walkthrough }

以“GQA 为什么能降低推理成本”为例，先说：GQA 让多个 query heads 共享较少的 K/V heads，
因此标准稠密 KV Cache 的理想容量和 decode 时的 K/V 读取量会下降。

如果面试官要求计算，再写：

\[
M_{KV}=2LBTH_{kv}d_hs.
\]

当 \(L=32\)、\(B=1\)、\(T=4096\)、\(d_h=128\)，且 K/V 使用 BF16 时，
32 个 KV heads 的理想 payload 是 2 GiB，8 个 KV heads 是 0.5 GiB。

最后主动收住边界：这 4 倍只属于理想 K/V payload。Q/O 投影、MLP、权重读取、分页分配、工作区和临时张量
不会同步缩小，所以端到端吞吐不能直接写成 4 倍。容量要检查实际 cache layout 和峰值显存，性能要固定 checkpoint、
长度分布、并发、kernel 与硬件后测 TPOT 和吞吐，质量则要另用留出任务集。

## 30 题路线索引

每题只保留“回答焦点”和权威入口。先沿入口理解机制，再回本页口述；需要可执行数字和反例时查
[深挖验证台账](../evidence/interview-controls.md)。

### Transformer 与生成

| # | 问题与回答焦点 | 权威入口 / 动手练习 |
|---:|---|---|
| <a id="attention-scaling"></a>1 | Attention score 为什么除以 \(\sqrt d\)：点积方差、softmax 饱和与尺度稳定。 | [Transformer](../core/transformer.md) · [编码题 1](coding-softmax-attention.md) |
| 2 | Causal mask 怎样排除未来信息：区分 causal、padding、packing 和 loss mask，用未来 token 不变性反证。 | [Transformer](../core/transformer.md) · [编码题 1](coding-softmax-attention.md) |
| 3 | MHA、MQA、GQA 怎样取舍：比较 K/V heads、理想 cache、decode 带宽与质量，避免把局部缩减外推成整机收益。 | [Transformer](../core/transformer.md) · [推理优化](../systems/inference-optimization.md) |
| 4 | Prefill 与 decode 为什么瓶颈不同：分别观察 TTFT、TPOT、算力利用率和内存带宽。 | [推理优化](../systems/inference-optimization.md) · [Serving](../systems/serving.md) |
| 5 | Temperature、top-k、top-p 改变什么：先声明处理顺序、重归一化、tie 与 crossing-token 契约。 | [生成与解码](../core/generation.md) · [编码题 2](coding-sampling.md) |
| 6 | “支持 1M context”是否等于有效利用：把请求上限、任务有效长度、成本和延迟分开验证。 | [长上下文系统](../frontier/long-context-systems.md) |

### 训练与对齐

| # | 问题与回答焦点 | 权威入口 / 动手练习 |
|---:|---|---|
| 7 | 什么时候用 RAG，什么时候微调：先按知识、检索、行为和格式错误分类，再比较 Prompt、RAG、LoRA 与组合基线。 | [微调](../training/finetuning.md) · [RAG](../applications/rag.md) |
| 8 | LoRA 的公式与参数量：解释 \(W+(\alpha/r)BA\)、初始化、冻结基座和可训练参数账本。 | [LoRA/QLoRA 工程](../training/peft-qlora-engineering.md) · [编码题 5](coding-lora.md) |
| <a id="qlora-scope"></a>9 | QLoRA 为什么不等于“全部 4-bit”：区分冻结权重的存储精度、计算精度、adapter、梯度、optimizer 与 activation。 | [LoRA/QLoRA 工程](../training/peft-qlora-engineering.md) |
| 10 | Assistant-only loss 怎样避免监督错位：在目标 chat template 和 tokenizer 之后检查最终 token labels。 | [SFT 数据闭环](../training/sft-data-pipeline.md) · [实验 4A](../practice/labs/lab-4a-sft-sample.md) |
| 11 | 可变长度 micro-batch 怎样正确累计梯度：累积 loss sum 与有效 token 数，并处理 DDP reduction、AMP 和 clip 顺序。 | [SFT 数据闭环](../training/sft-data-pipeline.md) · [分布式正确性](../systems/distributed-training-correctness.md) |
| 12 | DPO 与 PPO/RLHF 的训练信号有何不同：比较偏好对、reference、在线 rollout、advantage、ratio clip 与 KL。 | [对齐进阶](../training/alignment.md) |
| 13 | Loss 下降为什么不保证产品质量：检查泄漏、错误 labels、shortcut、reward hacking 和能力回归。 | [微调](../training/finetuning.md) · [评测方法](../quality/evaluation-methodology.md) |

### RAG 与 Agent

| # | 问题与回答焦点 | 权威入口 / 动手练习 |
|---:|---|---|
| 14 | 检索到正确文档但答案仍错：沿召回、重排、packing、生成、引用和发布 gate 分层定位。 | [RAG 请求生命周期](../applications/rag-request-lifecycle.md) |
| 15 | 多租户 RAG 怎样防泄漏：授权必须先于共享统计、排序、缓存和生成，并在边界处重新检查。 | [RAG 检索](../applications/rag-retrieval.md) · [编码题 3](coding-bm25.md) |
| 16 | Agent 与 Workflow 的边界：固定高风险流程用状态机，开放语义决策才交给模型。 | [Agent 架构](../applications/agent-architecture.md) |
| <a id="effect-idempotency"></a>17 | 怎样避免重复副作用：稳定 identity、审批绑定、pending、幂等键与 reconciliation 缺一不可。 | [任务生命周期](../applications/agent-task-lifecycle.md) · [编码题 6](coding-idempotent-effect.md) |
| 18 | 提示注入为什么不能只靠 system prompt：可信指令和外部文本共享上下文，权限边界必须在模型外。 | [安全](../quality/safety.md) · [Agent 运行时](../applications/agent-runtime.md) |
| 19 | Agent 怎样判断完成并安全恢复：模型只提出完成，verifier 检查业务事实；checkpoint 保存预算、pending effect 与身份。 | [Agent 任务生命周期](../applications/agent-task-lifecycle.md) |

### 评测与实验

| # | 问题与回答焦点 | 权威入口 / 动手练习 |
|---:|---|---|
| 20 | 评测最小单位是什么：用稳定 case/task-attempt 绑定输入、版本、输出、终态、指标和分母。 | [评测方法](../quality/evaluation-methodology.md) |
| 21 | LLM-as-a-Judge 有哪些偏差：检查位置、篇幅、风格、自偏好和 Prompt 敏感性，并与盲化人工或确定性 verifier 校准。 | [评测测量](../quality/evaluation-measurement.md) |
| 22 | 为什么使用 paired design：在同一 case 上计算差值，再按真实抽样单位估计区间或做检验。 | [评测统计](../foundations/evaluation-statistics.md) |
| 23 | 怎样防止测试集被反复调参污染：隔离 dev/test 权限、记录访问，按用户或 problem family 分组并检查近重复。 | [评测方法](../quality/evaluation-methodology.md) |
| <a id="slice-regression"></a>24 | 总体提升但中文用户下降怎样决策：关键 slice 若是发布硬约束，不能由总体平均抵消；样本不足就保持 canary。 | [评测方法](../quality/evaluation-methodology.md) · [应用题](applied-questions.md) |
| 25 | 正确候选已经生成，系统为何仍失败：把候选覆盖、选择器能力、验证器有效性和全链路终态分账。 | [代码 Agent 工作流](../applications/code-agent-workflow.md) · [评测总览](../quality/evaluation.md) |

### 25. 候选生成与选择必须分账 { #25 }

设四个任务各生成三个候选。若三个任务至少有一个正确候选，`oracle@3=3/4`；选择器只在其中两个任务选对，
则 `selected@3=2/4`，可解任务上的 selector recall 是 `2/3`。前者回答生成覆盖，后两者回答实际选择损失。

`pass@k` 估计随机给 \(k\) 次候选机会时至少一次通过当前验证器的概率，也不包含真实选择器。
若系统会合并候选、调用工具或重新生成，最终产物已不属于原候选集，不能继续把 `oracle@k` 当直接上界。
线上还要把生成/选择超时、预算耗尽、工具失败和发布拒绝放回全部合格任务的分母。

### 推理与生产系统

| # | 问题与回答焦点 | 权威入口 / 动手练习 |
|---:|---|---|
| 26 | p95 TTFT 突然升高怎样排查：先统一计时起点，再拆客户端、网关、队列、tokenization、prefill 和首 token 回传。 | [Serving](../systems/serving.md) · [推理服务项目](../practice/projects/inference-serving.md) |
| 27 | 4-bit 模型为什么不一定更快：文件、resident memory、峰值显存与目标 kernel 吞吐是不同命题。 | [推理优化](../systems/inference-optimization.md) |
| 28 | 云 API 为什么不能对所有 429/5xx 自动重试：同时判断 provider 语义、是否已发送、远端 outcome、幂等、deadline 与预算。 | [云 API 可靠性](../models/cloud-api-reliability.md) · [实验 0C](../practice/labs/lab-0c-cloud-budget.md) |
| 29 | 客户端断开 SSE 是否证明服务端停止生成：连接、task、backend、scheduler、KV 释放和计费需要分别观测。 | [Serving](../systems/serving.md) |
| <a id="replay-identity"></a>30 | 模型版本相同为什么仍未必可重放：还要绑定 template、adapter、runtime、RNG、索引、策略和外部状态。 | [准确性方法](../reference/accuracy.md) · [云 API 契约](../models/cloud-api-contracts.md) |

## 代码题怎样准备

索引页只教时间分配与表达方法；完整题面、参考实现和边界测试在独立练习页：

| 练习 | 最容易漏掉的边界 |
|---|---|
| [Softmax / Causal Attention](coding-softmax-attention.md) | 极大 logit、fully masked row、未来 token 不变性 |
| [Top-k / Top-p](coding-sampling.md) | crossing token、tie、全非法 logits、可重放 RNG |
| [带权限 BM25](coding-bm25.md) | ACL 必须先于语料级统计 |
| [RRF](coding-rrf.md) | 同列表重复、排名起点与稳定 tie-break |
| [LoRA Linear](coding-lora.md) | 冻结 base、缩放、合并与独立重载 |
| [幂等工具执行](coding-idempotent-effect.md) | timeout unknown、参数漂移、并发同键与对账 |

## 怎样使用深挖验证台账

当核心题能脱稿回答后，再到[深挖验证台账](../evidence/interview-controls.md)按领域查 claim、可执行 control 和边界。
选择与你岗位和项目相关的十项，把每项改写成自己的“结论—机制—边界—验证”，并准备一个真实失败样例。
