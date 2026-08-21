# LLM 面试题与回答方法

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备 LLM 应用、算法、训练、评测或推理岗位的开发者。
- **先修**：学过 Transformer、RAG、微调和基础工程实践中的至少一条主线。
- **首次阅读**：回答结构 → 题目地图 → 每个方向任选五题口述。
- **完成信号**：能在 30 秒给出结论，在两分钟内补充机制、边界和验证方法。
- **卡住时**：先用自己做过的一个项目回答，不要从术语定义开始背。

</div>

**求职导航**：[岗位路线](roadmap.md) · [系统设计](system-design.md) · [简历项目](resume-projects.md) · [深挖题与验证台账](../evidence/interview-controls.md)
{ .doc-nav }

面试不是关键词召回测试。面试官通常在确认三件事：你是否真的理解机制，是否知道结论何时失效，以及是否能设计一个实验把争论变成证据。

## 一道题怎样回答

推荐使用四层结构：

1. **结论**：先直接回答，不绕背景。
2. **机制**：用公式、数据流或状态变化解释为什么。
3. **边界**：指出成立条件、代价和常见反例。
4. **验证**：给一个最小实验或生产指标。

以“GQA 为什么能降低推理成本”为例：

> GQA 让多个 query heads 共享较少的 KV heads，因此主要减少 KV Cache 容量和 decode 时读取 KV 的带宽。理想 KV payload 约为
> \(2LBTH_{kv}d_{head}s\)，其中 2 对应 K/V，\(s\) 是元素字节数。
> 它不会同比减少 query projection、模型权重或所有临时内存，质量也可能变化。
> 我会在固定模型、batch 和序列长度下比较峰值显存、TPOT、吞吐和任务质量。

这个回答先给因果链，再给不能外推的部分，最后说明怎样证伪。不要一上来罗列 MHA、MQA、GQA 的定义。

## 深度怎样控制

| 时间 | 应完成什么 | 不要做什么 |
|---|---|---|
| 30 秒 | 结论、一个关键机制、一个边界 | 背完整教材 |
| 2 分钟 | 公式或状态流、主要 trade-off、验证方法 | 抛出十个未解释术语 |
| 白板深挖 | 明确假设，推导 shape/复杂度，设计正反例 | 用“框架会处理”结束 |
| 项目追问 | 给环境、分母、原始结果和失败样例 | 把本地 toy 写成生产结论 |

如果面试官继续追问，优先加深当前因果链；不要横向跳到更多名词。

## 高频题地图

| 方向 | 必须掌握的主线 | 深入章节 |
|---|---|---|
| Transformer / 生成 | attention、KV Cache、prefill/decode、采样 | [Transformer](../core/transformer.md)、[推理原理](../systems/inference.md) |
| 训练 / 对齐 | loss mask、LoRA/QLoRA、分布式归一化、偏好优化 | [微调](../training/finetuning.md)、[对齐](../training/alignment.md) |
| RAG / Agent | 检索归因、ACL、授权、幂等与恢复 | [RAG](../applications/rag.md)、[Agent](../applications/agents.md) |
| 评测 | evaluation unit、配对比较、切片、judge 校准 | [评测方法](../quality/evaluation-methodology.md) |
| 推理系统 | TTFT/TPOT、容量、量化、取消与重试 | [Serving](../systems/serving.md)、[推理优化](../systems/inference-optimization.md) |

下面的题不是背诵答案，而是示范怎样把回答落到机制和实验。

## Transformer 与生成

### 1. 为什么 attention score 要除以 \(\sqrt{d}\)？

**30 秒回答**：若 Q/K 各维近似独立且方差稳定，点积方差会随 head dimension \(d\) 增长。除以 \(\sqrt{d}\) 可稳定 score 尺度，避免 softmax 过早饱和。

**展开时说明**：

- LayerNorm 约束单个向量，不保证 QK 点积方差与 \(d\) 无关。
- 这是尺度控制，不是推理时调节随机性的 temperature。
- 实际分布不完全独立，但该缩放仍给初始化和优化提供稳定起点。

**怎样验证**：固定 Q/K 分布，改变 \(d\)，比较缩放前后的 score 方差、softmax entropy 和 gradient norm。

### 2. causal mask 怎样证明没有未来信息泄漏？

**30 秒回答**：mask 让位置 \(t\) 对未来 key 的注意力权重为零，但还要同时检查数据构造、position、cache 和 loss mask。

**怎样验证**：只修改 \(t\) 之后的 token；在 eval mode 下，位置 \(0..t\) 的 logits 应保持不变。若不变性失败，再定位 mask shape、广播方向和 cache。

### 3. MHA、MQA 与 GQA 怎样取舍？

**30 秒回答**：减少 KV heads 会降低 KV Cache 与 decode 带宽；MHA 表达最自由，MQA 最省 KV，GQA 位于两者之间。最终取舍要同时看质量、显存和 TPOT。

**展开时说明**：KV 公式只估理想 payload，不含 block allocator、对齐、量化 scale、workspace 和临时 tensor。遇到 MLA 或未知 attention layout 时，不应套标准 GQA 公式。

### 4. Prefill 与 decode 为什么瓶颈不同？

**30 秒回答**：Prefill 一次处理多个 token，矩阵乘规模较大，通常更容易吃满算力；decode 每步只生成少量 token，却反复读取模型权重和 KV，常受 memory bandwidth 限制。

**怎样验证**：分开记录 TTFT、TPOT、每步 active sequences、算力利用率和 memory bandwidth。总延迟无法告诉你是哪一阶段退化。

### 5. temperature、top-k 与 top-p 分别改变什么？

**30 秒回答**：temperature 改变整个 logit 分布的尖锐程度；top-k 限制候选排名；top-p 保留累计概率达到阈值的最小前缀。顺序和重新归一化时点属于生成契约的一部分。

**边界**：`temperature=0` 通常是 greedy 特例。即使参数相同，seed、RNG、并列规则、kernel 和版本漂移也可能让输出无法逐 token 重放。

### 6. “支持 1M context”是否等于有效利用 1M token？

**30 秒回答**：不是。文档上限、请求可接受上限和任务有效上限是三件事。模型可能接受请求，却在中部证据、多跳、冲突版本或长输出上明显退化。

**怎样验证**：改变证据位置、干扰数量和任务类型，报告准确率、引用、拒答、延迟和费用，而不是只做单一 needle retrieval。

## 训练与对齐

### 7. 什么时候用 RAG，什么时候微调？

**30 秒回答**：易变、私有、需要引用的事实优先 RAG；稳定的行为、格式和风格可以微调。先按错误类型判断问题来自知识、检索、指令遵循还是输出格式。

**验证路径**：至少比较 Prompt baseline、RAG、LoRA 和组合方案，并使用同一 held-out set。不要因为训练 loss 下降就跳过更便宜的基线。

### 8. LoRA 的公式与参数量是什么？

对线性层 \(y=Wx\)，LoRA 冻结 \(W\)，学习低秩更新
\(\Delta W=(\alpha/r)BA\)。若 \(W\in\mathbb{R}^{d_{out}\times d_{in}}\)，新增参数约为 \(r(d_{in}+d_{out})\)。

**边界**：参数少不自动意味着显存按同一比例下降。activation、optimizer、quantization buffer、target modules 和 sequence length 仍会影响峰值。

### 9. QLoRA 为什么不等于“用 4-bit 做全部训练”？

**30 秒回答**：QLoRA 通常把冻结的 base weights 以低比特存储，在计算时反量化到计算 dtype，并训练较高精度的 LoRA adapters。梯度和 optimizer state 并非都变成 4-bit。

**怎样验证**：打印每类参数的 storage dtype、compute dtype、`requires_grad`、optimizer state 和峰值显存，不要只看加载参数 `load_in_4bit=True`。

### 10. assistant-only loss 怎样避免监督错位？

**30 秒回答**：先用目标 chat template 渲染并 tokenize，再构造只覆盖 assistant content 的 labels；system、user、padding 和不应学习的控制 token 设为 ignore index。

**怎样验证**：打印最终 token ID、解码片段、label mask 和有效监督 token 数。只检查原始文本边界不够，因为模板和 tokenizer 会改变位置。

### 11. 可变长度 micro-batch 怎样做正确 gradient accumulation？

**30 秒回答**：每个 micro-batch 累积 loss sum 与有效 token count，在整个 optimizer update window 结束时按全局 token 数归一化。平均各 micro-batch mean 会让短 batch 权重过大。

**分布式追问**：若 DDP reducer 对 \(D\) 个 rank 的梯度取 mean，全局 token mean 需要让每个 rank backward \((D/N)S_r\)。还要在完整累计后 unscale、clip 和 step。

### 12. DPO 与 PPO/RLHF 的训练信号有何不同？

**30 秒回答**：DPO 直接从 chosen/rejected 对和 reference policy 构造偏好目标；PPO 通常从当前 policy 采样，用 reward/value 估计 advantage，再以 ratio clip 和 KL 约束更新。

**边界**：DPO 简单不代表数据无偏，PPO reward 上升也不代表人类效用改善。两者都需要 held-out preference、任务 verifier、安全回归和分布漂移检查。

### 13. 训练 loss 下降为什么不保证产品质量？

Loss 只衡量训练目标在当前数据和 reduction 下的改善。数据泄漏、错误 labels、shortcut、reward hacking、格式过拟合和通用能力回归都可能与 loss 同时发生。

**怎样验证**：冻结 held-out set，按任务和失败类型报告指标；同时保留简单 baseline、人工复核和真实约束下的延迟/成本。

## RAG 与 Agent

### 14. RAG 检索到了正确文档，答案仍错，怎样排查？

按数据流逐层定位：

1. 正确 chunk 是否真的进入 top-k；
2. rerank 是否保留它；
3. packing 后是否被截断；
4. prompt 中证据和问题是否对应；
5. 模型是否引用正确 span；
6. 答案是否被独立 verifier 判为 supported。

“文档在数据库里”与“模型当次看到了证据”不是同一个结论。

### 15. 多租户 RAG 怎样防止数据泄漏？

ACL 必须进入 retrieval query，或者在候选进入共享排序、缓存和生成前停止未授权数据继续流动。只在最终答案层
过滤已经太晚，因为未授权文本可能已经影响 rerank、cache 或模型上下文。

**负例**：让两个租户拥有相似文档，使用同一 query 和 cache key；验证未授权 source ID、正文和 embedding-derived result 都不会跨边界出现。

### 16. Agent 与 Workflow 的边界是什么？

固定步骤和少量已知分支优先 Workflow；只有下一动作确实依赖开放语言理解和中间结果时才使用 Agent。Agent 仍应运行在确定性的状态、权限和预算外壳中。

**追问**：多 Agent 只有在权限隔离、上下文隔离、并行或独立验证有价值时才值得，否则只是增加协调失败和成本。

### 17. 怎样避免 Agent 重复转账或重复发消息？

不能只让模型“记住不要重复”。应为逻辑动作生成稳定 identity，持久化 proposal/approval/attempt/effect 状态，并让外部系统支持 idempotency key 或可查询 receipt。

超时后的 outcome 可能是 `unknown`。此时先 reconcile，不能把 timeout 写成失败后直接重放。Transactional outbox 改善本地投递可靠性，也不能一般性地保证远端 exactly-once。

### 18. 提示注入为什么不能只靠 system prompt？

外部网页、邮件和工具结果与可信指令进入同一模型上下文，模型可能错误遵循低信任文本。分隔符和提示词能降低风险，却不是权限边界。

真正的边界在模型外：最小权限、资源级 ACL、工具 allowlist、参数校验、秘密隔离、沙箱、审批和 effect verifier。

### 19. Agent 怎样判断完成并安全恢复？

模型可以提出“已完成”，但 verifier 应根据业务系统状态判断。每一步都持久化 task state、tool result、预算和 pending effect；恢复时从最后一个已确认状态继续。

停止条件至少包含最大步数、deadline、费用、重复动作和无进展。达到限制时应报告停止原因，而不是生成一个看似成功的答案。

## 评测与实验

### 20. 评测的最小单位应该是什么？

通常是带稳定 ID 的 case 或 task attempt，而不是一段孤立文本。一个结果要绑定输入、版本、输出、错误类型、指标和是否进入最终分母。

如果同一用户或文档贡献多条 case，统计不再独立，应按相应 cluster 重采样或至少报告 cluster slice。

### 21. LLM-as-judge 有哪些偏差，怎样校准？

Judge 可能有 position、verbosity、style、自偏好和 prompt sensitivity。先保留双顺序判断、tie 和原始理由，再与盲化人工标注或确定性 verifier 对照。

报告 agreement、混淆矩阵和按语言/长度/难度的切片。高总体一致率不能掩盖某个关键 slice 的系统误判。

### 22. 为什么比较模型要用 paired design？

同一 case 上比较 baseline 与 candidate，可以消除大量 case 难度差异。先计算逐 case difference，再报告 effect size、区间和失败切片。

Paired bootstrap 用于估计差异的不确定性；randomization/sign-flip test 检验交换标签后的零假设。`p < 0.05` 不说明提升足够大、指标有效或用户受益。

### 23. 怎样防止测试集被反复调参污染？

把 train/dev/test 权限和用途分开；Prompt、threshold 和错误规则只在 dev 上迭代，最终 test 只在预注册 gate 执行。每次查看 test 都会泄露信息，应记录访问与决策。

还要按用户、thread、来源或 problem family 分组切分，并检查 exact、near-duplicate、语义改写和时间穿越。Hash 相同门禁只覆盖字节身份。

### 24. 总体提升但中文用户下降，怎样决策？

先确认 slice 是预定义还是事后发现，检查样本量、区间和流量权重。然后判断中文是否是发布硬约束，而不是用总体平均把它抵消。

发布 gate 可以同时要求 overall non-inferiority、关键 slice 不退化和故障率上限。若数据不足，保持 canary 或收集更多样本，不应把不确定写成“无影响”。

### 25. pass@k、oracle@k 与线上成功率有什么区别？

pass@k / oracle@k 问 k 个候选中是否至少有一个正确；线上系统还必须用实际 verifier 选出并返回它。候选相关性和 verifier 错误会让 selected@k 远低于 oracle@k。

增加 k 也会增加 token、延迟和发现 verifier 漏洞的机会。应联合报告 oracle、selected、单次成功率、成本和 tail latency。

## 推理与生产系统

### 26. p95 TTFT 突然升高，怎样排查？

先拆分 offered load、queue、prefill、调度和网络时间。检查到达率、并发、prompt 长度、batch policy、KV 压力、cache hit、GPU 利用率和错误/取消分母。

只看成功请求的 p95 会隐藏 rejected 和 timeout。对比同一 workload 下的时间序列，并确认指标定义和采样没有变化。

### 27. 4-bit 模型为什么不一定更快？

量化减少权重存储和带宽，但可能引入反量化、scale 读取、kernel 不匹配和小 batch 开销。某些硬件或 shape 没有高效低比特 kernel，甚至会 fallback。

在目标 GPU 上同时测质量、峰值显存、TTFT、TPOT 和吞吐。单矩阵压缩比不能外推为整个 checkpoint 或端到端加速比。

### 28. 云 API 为什么不能对所有 429/5xx 自动重试？

HTTP class 不足以决定 replay 是否安全。还要检查 provider 错误语义、请求是否有副作用、远端 outcome 是否已知、`Retry-After`、attempt/deadline 和费用预算。

connect 前的明确失败与发送后 timeout 不同。后者可能已经生成、计费或执行工具；若有 request/background ID，应先查询和 reconcile。

### 29. 客户端断开 SSE，是否证明服务端停止生成？

不证明。连接关闭、应用 task 取消、backend iterator 停止、GPU work 结束、KV 释放和停止计费是不同层级。

验收时用同一 request ID 关联客户端、server task、scheduler、allocator 和 usage。至少报告 disconnect-to-work-stop 与 disconnect-to-resource-release latency。

### 30. 模型版本相同，为什么请求仍未必可重放？

结果还依赖 tokenizer、template、generation config、seed/RNG、runtime、kernel、adapter、tool、retrieval index、缓存和 provider routing。闭源 alias 甚至可能在 model ID 不变时漂移。

先定义你需要的是字节级、token 级、指标级还是业务决策级重放，再保存相应 artifact。不要用一个 config hash 代表未被序列化的全部外部状态。

## 代码题怎样准备

至少能现场写出一个最小实现，并为它补正例、边界和失败例：

| 代码题 | 必须测的边界 | 仓库练习 |
|---|---|---|
| stable softmax / causal attention | 极大 logit、fully masked row、未来 token 不变性 | [Transformers Basics](../practice/projects/transformers-basics.md) |
| top-k / top-p sampling | crossing token、tie、全非法 logits、seed | [推理服务项目](../practice/projects/inference-serving.md) |
| BM25 / RRF | 空查询、重复文档、排序 tie、ACL | [RAG Foundations](../practice/projects/rag-foundations.md) |
| LoRA Linear | shape、scale、冻结 base、保存/重载 | [单卡微调](../practice/projects/single-gpu-finetuning.md) |
| paired metric / bootstrap | case 对齐、空分母、cluster | [Evaluation Gate](../practice/projects/evaluation-gate.md) |
| 幂等工具执行 | 重复 call、timeout unknown、审批变化 | [Safe Agent](../practice/projects/safe-agent.md) |

代码能跑只是起点。面试时要主动说明复杂度、数值稳定性、错误契约和怎样验证。

## 怎样使用深挖题库

当核心题能脱稿回答后，再进入[深挖题与验证台账](../evidence/interview-controls.md)。那一页保留分布式 AMP、
MoE、PPO、MCP、统计检验和 serving 的完整追问，用于二面或专项岗位。

不要背其中的固定数字。选择与你岗位和项目相关的十题，把每题改写成自己的“结论—机制—边界—验证”，并准备一个真实失败样例。
