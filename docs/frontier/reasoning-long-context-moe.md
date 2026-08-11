# 推理增强、长上下文与 Mixture of Experts

这三类技术都在扩大模型可用计算或容量，但扩大的是不同维度：推理增强增加 test-time search/verification，长上下文增加一次调用可访问的信息，MoE 增加总参数而控制每 token 激活计算。它们都不保证质量单调提升。

## 1. “推理”需要可操作定义

研究中可能指：

- 多步组合、约束满足或算法执行；
- 生成可见 rationale；
- 使用更多 test-time tokens/search；
- 调用计算器、代码、检索或 theorem prover；
- 在难题上的正确率提升。

可见 chain-of-thought 更长不等于内部推理更强；答案正确也可能来自记忆或 shortcut。任务应包含 counterfactual、held-out template、难度分层和可执行 verifier。

## 2. 训练时推理增强

### 2.1 Reasoning SFT

用问题—轨迹—答案示范训练。轨迹可由专家、程序或强模型生成。风险：

- teacher 的错误步骤被模仿；
- style/verbosity 被当成 reasoning quality；
- 训练集模板泄漏；
- 只会复述轨迹格式，不会适应新状态；
- rationale 与真实内部计算不忠实。

用最终答案 verifier、逐步 checker、执行测试与人工抽样过滤；保留 rejected/error trajectories 也可训练 verifier 或纠错。

### 2.2 Outcome reward

只根据最终答案给 reward，便宜且在数学/代码等可验证任务上清晰。但稀疏 reward 难以 credit assignment；模型也可能 exploit parser/test loophole。

### 2.3 Process reward

对中间 step/state 打分，提供更密监督。需要定义“局部正确”“可修复”“第一处错误”和多条合法路径。标注 step 不是证明模型内部按相同步骤计算。

### 2.4 Verifiable reward

编译器、unit tests、symbolic algebra、simulator、database constraint 和 formal proof checker 提供比纯 LLM judge 更强证据。但 verifier 只覆盖所编码性质：通过公开 tests 不代表隐藏输入正确，proof checker 正确也不保证 theorem specification 是用户想要的。

## 3. Test-time compute 方法

### 3.1 Self-consistency

从模型采样多条候选，对可 canonicalize 的最终答案投票。若单样本正确概率 \(p>0.5\) 且错误近似独立，多数票可提高成功率；真实候选错误高度相关，独立假设通常不成立。

投票还需要答案归一化。把等价数学表达识别错，会把正确票拆散；开放文本没有自然 majority label。

### 3.2 Best-of-N

生成 \(N\) 个候选，由 reward/verifier 选最大分数：

\[
\hat y=\arg\max_{y_i\sim\pi}v(x,y_i).
\]

随着 \(N\) 增大，候选 oracle quality 可能提高，selection quality 却受 verifier 限制。弱 verifier 会更频繁选到高分漏洞，产生“优化者诅咒”。必须同时报告 oracle@N、selected@N 和 verifier calibration。

### 3.3 Search

Tree/graph search 定义 state、action/expansion、value、backup、branching 和 stop。Monte Carlo Tree Search、beam、A* 等各有假设。自然语言 state 容易重复、不可判等且分支巨大；模型 value 也可能自信错误。

记录 expanded nodes、model/verifier calls、tokens、wall time 和 memory。只报告最终准确率会隐藏百倍成本。

### 3.4 Reflection / revision

模型根据 feedback 修订。若没有新证据，第二次回答可能只是换措辞或更自信；若 feedback 来自 tests/tool，则更可验证。保留每轮 delta 与停止理由，防止无限循环。

### 3.5 Tool-augmented reasoning

计算器、Python、SQL、检索和 proof assistant 把可验证子任务外包。工具结果仍可能错误、过期或被注入；tool schema、权限、timeout 和 result validation 属于推理系统的一部分。

## 4. Test-time scaling 曲线

不要只比较 “thinking on/off”。对预算 \(B\) 画：

- quality/success vs tokens；
- quality vs model/verifier calls；
- quality vs wall-clock/cost；
- p95 latency 与 failure rate；
- 不同难度/领域切片；
- oracle candidate quality vs selected quality。

收益可能先升后降：上下文被低质量轨迹污染、search 走偏、verifier exploit 或超时。停止策略应基于 marginal utility、confidence 或硬预算，不假定越长越好。

## 5. 推理评测的泄漏与指标

- GSM/代码题等模板可能在预训练/合成数据中出现；
- exact answer parser 可被格式影响；
- pass@k 与 pass@1 回答不同问题；
- 同一题多次采样不是独立 test samples；
- judge 可能偏好更长 rationale；
- 解题轨迹正确不保证 final，final 正确也不保证轨迹。

报告题目级 paired result 和 bootstrap CI；对可验证任务优先执行 verifier。若公开 benchmark 被反复用于 prompt/算法选择，保留独立 holdout。

## 6. 可见推理与隐私

Chain-of-thought 可能包含用户敏感数据、系统提示、攻击内容或不应暴露的内部规则。产品可输出简洁理由、证据和可验证步骤，而不承诺展示完整内部 token。隐藏 rationale 也不自动提高安全；仍需外部审计和结果验证。

## 7. 长上下文的三层含义

必须区分：

1. **API acceptance length**：请求最多接收多少 token；
2. **trained/evaluated length**：模型在哪些长度上训练和验证；
3. **effective context**：在具体任务/位置/干扰下真正可利用的信息。

128k 请求不报错只证明第 1 项。它不证明能准确聚合 128k，也不证明输出预算与 KV capacity 足够。

## 8. 位置表示与外推

### 8.1 Absolute position

Learned absolute embedding 在训练最大位置外没有可靠已学参数。简单扩表/插值需要继续训练或验证。

### 8.2 Relative bias

按 token distance/bucket 加 bias，可更自然表达相对关系；bucket 范围和训练长度仍限制外推。

### 8.3 RoPE

RoPE 对 query/key 维度施加位置相关旋转，使 dot product 带相对位置信息。长上下文方法可能调整 frequency/base、position interpolation 或分段 scaling，并做 long-context continued training。

修改 RoPE 配置会改变 checkpoint 的位置行为；不能把任意 scaling factor 视为免费扩窗。需测试短上下文回归、长位置检索、多跳和生成稳定性。

## 9. Attention 复杂度与替代

标准 full attention 的 score matrix 对长度 \(T\) 有 \(O(T^2)\) 元素，QKV/MLP 还有按 token 线性项。实际成本依 hidden size、batch 和 memory-efficient kernel。

### 9.1 Sliding/local attention

每个 token 只看窗口 \(w\)，attention 连接约 \(O(Tw)\)。堆叠层扩大 receptive field，但远距离精确信息可能需要多层传播。

### 9.2 Block sparse / global tokens

局部 block 配合少量 global/landmark connection。Sparse pattern 必须和任务匹配，并有高效 kernel；理论 sparsity 不保证 wall-clock 加速。

### 9.3 Linear attention

通过 kernel feature map 重排 \(QK^\top V\)，避免显式 score matrix。它通常改变 softmax attention 的函数族或使用近似，稳定性、causal scan 与 state size 需验证。

### 9.4 Recurrent/SSM memory

把历史压缩到 state，推理 memory 可受限，但存在信息瓶颈。精确 retrieval、copying 与 state reset 需要单独测。

## 10. KV Cache 与长上下文

标准 cache 理想字节：

\[
M_{KV}=2LBTH_{kv}d_hs.
\]

上下文翻倍，理想 KV 线性翻倍；并发也线性放大。Paged allocation、prefix sharing、sliding window、KV quantization、eviction 和 latent compression 改变真实占用/质量。

压缩/淘汰策略应按 position、query type 和 multi-turn 测试，不能只测平均 perplexity。丢掉早期 system/tool state 可能造成安全回归。

## 11. 长上下文、RAG 与 memory hierarchy

三者不是二选一：

- model context：低延迟直接访问，但昂贵且会受干扰；
- RAG：按 query 选择、可更新/ACL/引用，但召回可能漏；
- summary/memory：压缩历史，但摘要会丢信息并累积错误；
- external structured store：可精确查询，但需要 schema/tool。

合理流程：先按权限检索文档/章节，再在足够长上下文中整合；对稳定状态写结构化 store，对原始证据保留 source pointer。

## 12. 长上下文评测矩阵

### Retrieval

- single needle、多 needle；
- key/value、exact string、语义改写；
- 开头/中间/结尾与等距位置；
- distractor 数量和相似度；
- 冲突、过期和无答案。

### Integration

- 跨段 multi-hop；
- count/sum/global aggregation；
- temporal/order reasoning；
- entity coreference；
- 多文档 contradiction 与 source priority。

### Generation

- 长文一致性和约束保持；
- 引用覆盖与位置；
- early instruction retention；
- context copying/attribution；
- 输出长度与 repetition。

### Systems

- TTFT、prefill throughput、TPOT；
- peak KV 与 OOM；
- cache hit/eviction；
- concurrency 下 tail latency；
- truncation policy 和实际 accepted tokens。

“Lost in the middle”是可能的位置效应，不是所有模型/任务的固定曲线。逐位置报告，而不是只引用标签。

## 13. MoE 前向与口径

对 token representation \(x\)，router logits \(r(x)\) 选择 top-k experts：

\[
y(x)=
\sum_{e\in TopK(r(x))}
g_e(x)E_e(x).
\]

总参数包括所有 experts，共享 attention/norm/router；active parameters 只包含当前 token 使用部分。二者分别影响 weight memory 和 token compute，不能混称“模型大小”。

## 14. Router 与 capacity

先固定 routing group。设其中有 \(N\) 个有效 token（padding 不计）、\(E\) 个 routed expert、每 token 选 \(k\) 个 expert。本仓库 CPU reference 使用

\[
C=\left\lceil\phi\frac{Nk}{E}\right\rceil
\]

作为**每个 expert**的 assignment capacity，其中 \(\phi>0\) 是 capacity factor。这只是明确的一种约定：真实实现可能按 device、sequence、micro-batch 或全局 token group 计算，可能设置最小 capacity，也可能 dropless。即使总 slot 数看似足够，routing 偏斜仍会让 hot expert overflow。

Reference 对每个 token 的 softmax probability 做稳定 top-k，同分时 expert id 小者优先；expert 内按 router probability 降序占用 capacity，再以 token index、top-k rank 打破平局。生产实现也可能按 token 到达顺序、grouped kernel layout 或 reroute policy 处理，不能把这份 score-priority 规则叫作通用 MoE 语义。

### 14.1 Load balance

没有约束时，少数 experts 可能吸收大部分 token。Auxiliary load-balancing loss、router z-loss、noise 或 capacity policy 用于稳定分配。它们改变优化目标，系数过大可能损害 task routing。

为提供可复算诊断，reference 定义 pre-capacity assignment fraction 与平均 router probability：

\[
f_e=\frac{n_e}{Nk},\qquad
p_e=\frac1N\sum_{i=1}^{N}\operatorname{softmax}(r_i)_e,
\qquad
L_{bal}^{ref}=E\sum_e f_ep_e.
\]

它是本仓库对 top-k 的**广义诊断**，不是所有论文/框架的 training loss；有的定义只用于 top-1、使用 post-capacity count、按 sequence/device 分组、stop-gradient 或不同缩放。单看该标量也不够：均匀 probability 配合 deterministic tie-break 仍可能只选择较小 expert id，因此必须同时报告 per-expert counts、overflow 与 entropy。

Reference 另计算

\[
L_z=\frac1N\sum_i\left(\log\sum_e\exp r_{i,e}\right)^2,
\]

只作为 router logit scale 诊断；是否加入训练、系数和 reduction 必须以目标实现为准。

### 14.2 Capacity

每 expert 的 buffer 常与平均 token/expert 数乘 capacity factor 相关。溢出 token 可能 dropped、rerouted、padding 或走 shared expert；不同实现语义不同。Token dropping 若未进入 loss/metric 统计，会静默改变训练样本。

还要区分 **dropped assignment** 与 **all-assignments-dropped token**。Top-2 token 丢掉一个 expert 后仍可能由另一个处理；两个都丢才是 routed expert 输出为零。Residual/shared expert 可能让最终 block 输出不为零，但这不等于 routed 分支没有丢失。报告两种分母，不能把 3/8 assignment drop 写成“3 个 token 被丢”。

### 14.3 Top-k 与 gate

Top-1 计算/通信较低，top-2 等提供冗余/表达但增加 active compute。Gate probability 是 mixture weight，不是“这个专家正确的概率”。

Top-k 后通常先对被选 gate 归一化；capacity drop 后是否再次归一化是另一份契约。重归一化让至少保留一个 assignment 的 token 权重和回到 1，但放大幸存 expert；不重归一化则保留丢失 mixture mass。两者都不能静默选择。若所有 routed assignment 被丢，本仓库 sparse expert toy 输出零，残差或 shared expert 必须由上层另行加入。

### 14.4 可运行 routing fixture

~~~powershell
python projects/transformers-basics/moe_routing.py
python -m pytest tests/test_moe_routing.py -q
~~~

固定 \(N=4,E=3,k=2,\phi=0.75\)，所以每 expert capacity 为 2。Pre-capacity counts 是 `(3,4,1)`，score-priority 后为 `(2,2,1)`；8 个 assignment 保留 5、丢 3，但 4 个 token 都至少保留一个 expert。Toy 真实执行 kept assignment 的 bias-free linear expert 与 weighted combine；测试还覆盖 expert/token tie-break、整 token drop、padding mask、drop 后重归一化开关、解析 balance/z-loss/entropy、输入失败和数组不可变。

这不是训练过的 router/MLP，不执行 backward、expert-parallel all-to-all、distributed capacity、GPU grouped GEMM，也不加载 DeepSeek/Qwen checkpoint。它证明当前公式与 CPU control flow，不证明模型质量、expert specialization、显存、通信或吞吐。

## 15. Expert parallel 与通信

Expert weights 分布在设备上时，token 按 routing 做 all-to-all dispatch，再返回原设备。成本取决于 token 数、hidden size、top-k、dtype、网络、负载不均和消息大小。

即使 theoretical active FLOPs 低，all-to-all、packing/unpacking、小 expert GEMM 和 straggler 可吞噬收益。报告：

- total/active parameters；
- tokens/expert distribution、coefficient of variation；
- overflow/drop rate；
- router entropy 与 aux losses；
- all-to-all bytes/time；
- shared vs expert compute；
- quality、tokens/s、memory 和 tail latency。

## 16. MoE 推理

所有 expert 权重通常需驻留集群或被动态加载；小 batch 难以将同一 expert 的 token 合成大 GEMM。Hot experts 造成负载倾斜。可用 expert replication、routing-aware batching、quantization、cache 或 shared expert，但每项引入内存/延迟 trade-off。

“每 token 只激活 10B，所以部署等于 10B dense”是错误的：总权重、router、共享层、通信和并发都不同。

## 17. 专家是否可解释

按高 routing token 给 expert 命名为“代码/中文/数学”只说明相关性。Expert 可能是 polysemantic，routing 还受位置、频率和负载机制影响。需要 intervention、counterfactual routing 和跨数据稳定性，才能支持更强功能结论。

## 18. 前沿结果的审计

面对新 SOTA，检查：

1. 参数是 total 还是 active，训练/推理 FLOPs 怎样算；
2. 数据、tokenizer、污染和训练 tokens 是否相同；
3. test-time tokens、N candidates、verifier/tool 是否计入；
4. context acceptance 还是有效 integration；
5. latency、memory、energy 和失败率是否报告；
6. baseline 是否用同等调参/推理预算；
7. 多 seed/CI、ablation、code/data 是否存在；
8. 负结果、适用长度/任务和硬件是否披露。

## 19. 当前仓库证据边界

仓库已有生成采样、paired evaluation、KV 公式、Roofline、Agent 工具安全、scaling 计算器，以及 top-k/capacity/drop/sparse-linear-combine 的 NumPy MoE oracle，可用于验证部分数学/系统概念。但没有训练 PRM/online reasoning policy，没有目标长上下文 checkpoint 的全长度矩阵，也没有训练 MoE router/MLP 或 expert-parallel GPU 实跑。因此 CPU routing fixture 不是 DeepSeek/Qwen 复现，本章其余能力仍是实验协议与机制教材。

## 20. 常见错误结论

- **“更长 CoT 一定更会推理”**：长度可能只是风格或重复。
- **“Best-of-N 随 N 单调提高实际质量”**：弱 verifier 会选择高分漏洞。
- **“128k API 就有 128k 有效记忆”**：接受长度、训练长度和有效利用不同。
- **“稀疏 attention 的 Big-O 更低就一定更快”**：kernel、常数和硬件决定实测。
- **“长上下文可以替代 RAG”**：ACL、更新、引用和降噪仍需检索系统。
- **“MoE active 参数就是权重内存”**：总 experts 仍需存储/分片。
- **“专家激活主题就是专家的单一功能”**：相关 token 不构成因果解释。

## 自测与实践

1. 分别报告 Best-of-16 的 oracle@16 与 verifier-selected@16，为什么二者不同？
2. 为 128k 模型设计位置 × distractor × integration 的评测矩阵。
3. 推导 sliding window 每层连接数，并解释多层 receptive field。
4. KV eviction 为什么可能造成安全而不只是质量回归？
5. 为 top-2 MoE 列出 total/active/memory/communication 四种口径。
6. 设计 counterfactual routing 实验，避免只按 token 主题给 expert 命名。
