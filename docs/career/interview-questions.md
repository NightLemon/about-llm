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

## 一场追问怎样逐层展开 { #gqa-walkthrough }

以“GQA 为什么能降低推理成本”为例。

**面试官**：GQA 为什么更省？

**你，先用 30 秒回答**：GQA 让一组查询头（query heads）共享较少的 K/V heads。

标准稠密 KV Cache 的理想容量与 \(H_{kv}\) 成正比，因此它最直接地减少缓存容量，以及 decode
时读取 K/V 的数据量。

模型权重、Q/O 投影和 MLP 不会按同一比例缩小，所以不能从 KV 缩小倍数直接推出端到端加速倍数。

这时已经完成“结论—机制—边界”。如果面试官追问“能不能算一个例子”，再上白板。

标准 dense K/V 的理想 payload 是：

\[
M_{KV}=2LBTH_{kv}d_hs,
\]

其中 2 代表 K 和 V，\(L\) 是层数，\(B\) 是活动序列数，\(T\) 是每条序列缓存的 token 数，
\(d_h\) 是 head dimension，\(s\) 是每个元素的字节数。

假设 \(L=32\)、\(B=1\)、\(T=4096\)、\(d_h=128\)，并用 BF16 保存 K/V：

| Attention 设置 | \(H_{kv}\) | 理想 KV payload |
|---|---:|---:|
| MHA | 32 | 2 GiB |
| GQA | 8 | 0.5 GiB |

这组设置让理想 KV payload 缩小 4 倍。它没有计算 block allocator、对齐、workspace 和临时 tensor，
也没有描述质量变化。

**面试官继续追问**：那吞吐会提高 4 倍吗？

**你**：不能这样推。只有 K/V projection、缓存容量和相关读取接近这个缩放；Q/O projection、MLP、权重读取、
调度和其他运行时开销仍然存在。最终收益还取决于 batch、序列长度、kernel 和硬件。

**面试官最后追问**：怎样验证？

**你**：固定 checkpoint、请求长度分布、并发和硬件，先确认真实 cache layout，再同时测峰值显存、TPOT、吞吐和
任务质量。若只测短 Prompt 或只看显存，无法回答长上下文 decode 和质量取舍。

这就是一条完整的回答链：先用一句话指出主因，再按追问加入算术、反例和实验。不要一上来横向罗列
MHA、MQA、GQA 的全部定义。

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

**30 秒回答**：MHA 为每个查询头保留独立 K/V head，MQA 让所有查询头共享一组 K/V，GQA 位于两者之间。
减少 \(H_{kv}\) 会降低 K/V 投影、KV Cache 和 decode 时的 K/V 读取量。最终仍要同时比较质量、显存和 TPOT。

**展开时说明**：KV 公式只估算理想 K/V 数组，不含分页分配器、内存对齐、量化 scale、工作区和临时张量。
遇到 MLA 或未知 cache layout 时，应先确认实际存储结构，不能套用标准 GQA 公式。

完整的数值追问见[开头的 GQA 白板示范](#gqa-walkthrough)。

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

**边界**：可训练参数少，不表示峰值显存按同一比例下降。训练激活、优化器状态、量化缓冲区、适配的层数和序列长度
都会占用显存。

### 9. QLoRA 为什么不等于“用 4-bit 做全部训练”？

**30 秒回答**：QLoRA 通常以低比特保存冻结的基础权重，计算时再反量化到 BF16 或 FP16 等计算精度。
训练更新的是较高精度的 LoRA adapter；梯度和优化器状态并非全部变成 4-bit。

**怎样验证**：打印每类参数的 storage dtype、compute dtype、`requires_grad`、optimizer state 和峰值显存，不要只看加载参数 `load_in_4bit=True`。

### 10. assistant-only loss 怎样避免监督错位？

**30 秒回答**：先用目标对话模板（chat template）渲染并分词，再让监督标签只覆盖 assistant 回复。
系统消息、用户消息、补齐位置及不应学习的控制 token 全部设为 ignore index。

**怎样验证**：打印最终 token ID、解码片段、label mask 和有效监督 token 数。只检查原始文本边界不够，因为模板和 tokenizer 会改变位置。

### 11. 可变长度 micro-batch 怎样做正确 gradient accumulation？

**30 秒回答**：每个微批次累积损失总和与有效监督 token 数。等整个参数更新窗口结束，再除以全局有效 token 数。
直接平均各微批次的 mean loss，会让短批次获得过高权重。

**分布式追问**：设 \(D\) 个进程一共有 \(N\) 个有效 token，本进程的损失总和为 \(S_r\)。
若 DDP 会平均各进程的梯度，本进程应反向传播 \((D/N)S_r\)。完整累计后再执行反缩放、梯度裁剪和参数更新。

### 12. DPO 与 PPO/RLHF 的训练信号有何不同？

**30 秒回答**：DPO 从 chosen/rejected 样本对和参考策略直接构造偏好目标。PPO 通常从当前策略采样，
再用奖励与价值估计 advantage，并通过概率比裁剪和 KL 约束限制更新幅度。

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

不能只让模型“记住不要重复”。应为逻辑动作生成稳定 ID，并持久化提议、审批、尝试和外部效果四阶段状态。
外部系统还要支持幂等键（idempotency key）或可查询的执行回执。

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

Judge 常见位置、篇幅、风格、自偏好和 Prompt 敏感性等偏差。先把候选顺序交换后再评一次，并保留平局与原始理由。
然后与盲化人工标注或确定性验证程序对照。

报告 agreement、混淆矩阵和按语言/长度/难度的切片。高总体一致率不能掩盖某个关键 slice 的系统误判。

### 22. 为什么比较模型要用 paired design？

在同一个 case 上比较基线与候选系统，可以消除大量题目难度差异。先计算每条 case 的成对差值，
再报告效应大小、置信区间和失败切片。

Paired bootstrap 用于估计差异的不确定性；randomization/sign-flip test 检验交换标签后的零假设。`p < 0.05` 不说明提升足够大、指标有效或用户受益。

### 23. 怎样防止测试集被反复调参污染？

把训练集、开发集和最终测试集的权限与用途分开。Prompt、阈值和错误规则只在开发集上迭代；
最终测试集只按事先登记的发布判据执行。每次查看最终测试结果都会泄露信息，因此要记录访问和后续决策。

还要按用户、thread、来源或 problem family 分组切分，并检查 exact、near-duplicate、语义改写和时间穿越。Hash 相同门禁只覆盖字节身份。

### 24. 总体提升但中文用户下降，怎样决策？

先确认 slice 是预定义还是事后发现，检查样本量、区间和流量权重。然后判断中文是否是发布硬约束，而不是用总体平均把它抵消。

发布 gate 可以同时要求 overall non-inferiority、关键 slice 不退化和故障率上限。若数据不足，保持 canary 或收集更多样本，不应把不确定写成“无影响”。

### 25. pass@k、oracle@k、selected@k 与线上成功率有什么区别？

先把四个问题分开：

| 指标 | 它实际询问什么 |
|---|---|
| pass@k | 在常见代码评测中，从同一题的 \(n\) 个采样里给 \(k\) 次机会，至少一个通过测试的估计概率 |
| oracle@k | 当前 \(k\) 个候选中，是否存在一个满足正确性判据的候选 |
| selected@k | 系统生成 \(k\) 个候选后，真实选择器最终挑出的那个是否正确 |
| 线上成功率 | 在真实流量、超时、费用和失败终态下，最终交付是否成功 |

同一题有 \(n\) 个采样、其中 \(c\) 个通过验证时，常用 pass@k 估计是：

\[
\operatorname{pass@k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
\qquad 1\le k\le n.
\]

例如 \(n=10\)、\(c=2\)、\(k=2\) 时，pass@2 是 \(17/45\approx0.378\)。这个组合估计采用常见的
独立同分布采样解释，并把测试通过当作“正确”判据；它不是线上用户只请求一次时的成功率。

Oracle 只判断正确候选是否存在。选择器还必须在看不到 oracle 答案时把它找出来，所以在同一候选集合和判据下，
selected@k 不会高于 oracle@k。验证程序漏判、候选高度相关或选择策略偏置，都会改变实际收益。

增加 \(k\) 还会增加 token、延迟、费用和验证程序的攻击面。

报告至少包含：pass@1、oracle@k、selected@k、实际 \(k\)、采样设置和验证判据；性能侧再给出成本与尾延迟。

## 推理与生产系统

### 26. p95 TTFT 突然升高，怎样排查？

**30 秒回答**：先确认请求入口和统计口径有没有变化，再把 TTFT 拆成客户端排队、网关/网络、服务端排队、
tokenization、prefill 和首 token 回传。只有 dispatch 之后的时间变慢，才继续把重点放在 tokenizer、调度和 GPU。

按症状继续追：

| 同时观察到什么 | 优先检查什么 |
|---|---|
| Offered rate 上升，服务队列变长 | Admission、并发上限、长短请求混排和副本健康 |
| Prompt 变长，dispatch TTFT 上升 | Tokenization、chunked prefill 和 prefill kernel |
| KV 接近容量，出现抢占或重算 | Block 使用、最大长度、并发和调度策略 |
| TPOT 也变慢 | Decode batch、权重/KV 带宽、kernel 和功耗 |
| 客户端 TTFT 高，服务端 trace 正常 | 网关、网络、客户端队列和流式缓冲 |

只看成功请求的 p95 会隐藏 rejected、timeout 和排队失败。对比同一 workload 下的时间序列，并同时保存
all-attempt 分母、输入长度、终态和 offered-to-first-token 时间。

### 27. 4-bit 模型为什么不一定更快？

量化减少权重存储和带宽，但可能引入反量化、scale 读取、kernel 不匹配和小 batch 开销。某些硬件或 shape 没有高效低比特 kernel，甚至会 fallback。

在目标 GPU 上同时测质量、峰值显存、TTFT、TPOT 和吞吐。单矩阵压缩比不能外推为整个 checkpoint 或端到端加速比。

### 28. 云 API 为什么不能对所有 429/5xx 自动重试？

HTTP class 不足以决定 replay 是否安全。还要检查 provider 错误语义、请求是否有副作用、远端 outcome 是否已知、`Retry-After`、attempt/deadline 和费用预算。

connect 前的明确失败与发送后 timeout 不同。后者可能已经生成、计费或执行工具；若有 request/background ID，应先查询和 reconcile。

### 29. 客户端断开 SSE，是否证明服务端停止生成？

不证明。连接关闭、应用 task 取消、backend iterator 停止、GPU work 结束、KV 释放和停止计费是不同层级。

验收时用同一个 request ID 关联客户端请求、服务端任务、调度器、内存分配器和计费用量。
至少分别报告“断连到计算停止”和“断连到资源释放”两段延迟。

### 30. 模型版本相同，为什么请求仍未必可重放？

模型名称只覆盖了输入身份的一小部分。一次结果还取决于：

| 层次 | 需要绑定的内容 |
|---|---|
| 模型输入 | Tokenizer、chat template、完整消息、工具 schema 和生成参数 |
| 模型工件 | Checkpoint revision、adapter、量化格式与加载配置 |
| 执行环境 | Runtime、kernel、硬件、随机数状态和并行拓扑 |
| 外部状态 | Retrieval index、权限策略、缓存、工具结果和 Provider routing |

闭源 alias 甚至可能在 model ID 不变时切换实际版本。先说明目标是字节级、token 级、指标级还是业务决策级重放，
再保存对应工件。一个 config hash 只能覆盖被明确序列化的字段，不能代表遗漏的外部状态。

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
