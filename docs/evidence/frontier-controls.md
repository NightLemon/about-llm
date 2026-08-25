# 前沿系统证据台账：Reasoning、Long Context 与 MoE

本页保存 self-consistency、best-of-N、长上下文与 MoE routing/collective 实验的精确公式、录制结果和边界。
第一次学习请从[前沿总览](../frontier/reasoning-long-context-moe.md)进入三条独立路线。

**读者入口**：[前沿总览](../frontier/reasoning-long-context-moe.md) · [推理系统](../frontier/reasoning-systems.md) · [长上下文](../frontier/long-context-systems.md) · [MoE 系统](../frontier/moe-systems.md)
{ .doc-nav }

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：推理增强、长上下文和 MoE 研究与系统工程师。
- **先修**：[生成](../core/generation.md)、评测、attention 和并行基础。
- **首次阅读**：推理定义 → test-time compute → 长上下文三层含义 → MoE → 审计。
- **完成信号**：能分开模型能力、搜索预算、上下文有效性和容量证据。
- **卡住时**：先回到[Transformer](../core/transformer.md)与[评测](../quality/evaluation.md)。

</div>

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

#### 相同单样本正确率，不同多数票结果

先把问题限定为只有“正确/错误”两个 canonical label，且 N 为奇数。若候选真正 i.i.d.、每次正确概率都是 \(p\)，多数票成功率是 binomial upper tail；当 \(p>0.5\) 时它随奇数 N 增大。但真实题目有共享难度、模板和错误模式：同一道题的多次采样可能只在**给定题目状态后**条件独立，边缘上并不独立。

用一个 finite latent-regime 模型表达这种差别。每题先抽一次 \(R=r\)，概率为 \(q_r\)；随后该题的 N 个 correctness indicators 在给定 R 后 i.i.d. Bernoulli\((p_r)\)。奇数 N 的多数票成功率为

\[
P(\operatorname{majority\ success})=
\sum_r q_r
\sum_{k=(N+1)/2}^{N}
{N\choose k}p_r^k(1-p_r)^{N-k}.
\]

边缘单样本正确率只有 \(\bar p=\sum_rq_rp_r\)，不足以确定上式。两次候选 correctness 的边缘相关系数为

\[
\rho=
\frac{\sum_rq_rp_r^2-\bar p^2}{\bar p(1-\bar p)},
\]

其中分母非零。运行精确控制：

~~~powershell
python projects/inference-serving/self_consistency_correlation_toy.py
~~~

对照场景只有一个 regime、\(p=0.6\)，所以候选无条件独立、\(\rho=0\)。相关场景以相同概率抽 easy \((p=0.9)\) 或 hard \((p=0.3)\)，其边缘单样本正确率仍是 0.6，但共享 R 令 \(\rho=3/8\)：

| N | i.i.d. majority success | latent-correlated majority success |
|---:|---:|---:|
| 1 | 0.60000000000 | 0.60000000000 |
| 3 | 0.64800000000 | 0.59400000000 |
| 5 | 0.68256000000 | 0.57726000000 |
| 11 | 0.75349813248 | 0.53896454244 |

相关场景中，增加采样会让 easy 题更稳，也会让 \(p=0.3\) 的 hard 题更稳定地投错；只看跨题平均 \(\bar p>0.5\) 会隐藏这个异质性。N=11 对应 `2^11=2,048` 条 logical binary vote sequences，程序按每个 regime 的 binomial tail 用 `Fraction` 闭式计算，没有枚举。

这只是 authored binary-answer counterexample：每题恰好一次 latent regime draw，候选只在 regime 内 conditional i.i.d.。它没有模拟多个不同错误答案的 plurality、答案 canonicalization、temperature 对相关性的影响、模型、tokenizer、dataset 或 judge，也没有测量 latency、费用、provider 或目标质量。开放文本 self-consistency 必须保存逐题候选与规范化结果，按 item cluster 估计增益和不确定性，不能用这个 toy 宣称真实系统必然退化。

### 3.2 Best-of-N

生成 \(N\) 个候选，由 reward/verifier 选最大分数：

\[
\hat y=\arg\max_{y_i\sim\pi}v(x,y_i).
\]

随着 \(N\) 增大，候选 oracle quality 可能提高，selection quality 却受 verifier 限制。弱 verifier 会更频繁选到高分漏洞，产生“优化者诅咒”。必须同时报告 oracle@N、selected@N 和 verifier calibration。

#### 一个可精确复算的 verifier 反例

运行有限支持集上的闭式控制：

~~~powershell
python projects/inference-serving/verifier_best_of_n_toy.py
~~~

这个 fixture 不是模型输出，而是作者明确写下的三个 outcome class。每次从同一固定分布独立同分布（i.i.d.）抽样，verifier score 是确定常数；选择规则固定为最大 `(verifier_score, candidate_id)`，因此同分时较大的 canonical ID 胜出：

| candidate | sampling probability | verifier score | target success |
|---|---:|---:|---:|
| `wrong` | 0.5 | 20 | false |
| `correct` | 0.4 | 80 | true |
| `verifier_hack` | 0.1 | 99 | false |

按 `(verifier_score, candidate_id)` 从弱到强排序，令 \(F_i\) 为截至候选 \(i\) 的累积抽样概率。best-of-N 最终选择该候选的概率是

\[
P(\operatorname{select} i)=F_i^N-F_{i-1}^N.
\]

它来自“\(N\) 次抽样都不超过 \(i\)”减去“\(N\) 次都严格低于 \(i\)”，无需枚举所有序列。另令单次抽到任意 target-success candidate 的概率为 \(p_s\)，则

\[
\operatorname{oracle@N}=1-(1-p_s)^N.
\]

这里 oracle@N 只问候选集合中是否**存在**正确项；selected@N 才问 verifier 最终选中的项是否正确：

| N | oracle@N | selected@N | oracle-selection gap | expected verifier score |
|---:|---:|---:|---:|---:|
| 1 | 0.4000000000 | 0.4000000000 | 0 | 51.9000000 |
| 4 | 0.8704000000 | 0.5936000000 | 0.2768000000 | 82.7841000 |
| 16 | 0.9997178890 | 0.1852867601 | 0.8144311289 | 95.4783461 |

随着 N 从 1 增至 16，oracle success 和期望 verifier score 都严格上升，但 selected success 先升后降；N=16 时 `verifier_hack` 至少出现一次的概率已约为 0.814698。`3^16=43,046,721` 只是三个 outcome class 的 logical candidate sequences 数，程序使用 `Fraction` 闭式计算，没有枚举这些序列。

这份控制只证明上述 authored finite distribution、i.i.d.、deterministic score 和 tie-break 下的数学反例。逻辑上的 N 次 model sample / verifier score 不等于 wall-clock、费用或并行行为；脚本没有执行 model、tokenizer、PRM、GPU 或 provider，也不证明 verifier calibration、语义正确性、目标模型质量或真实系统中的 optimizer's curse 强度。

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

### 14.5 可运行 router/MLP 梯度 fixture

~~~powershell
python projects/transformers-basics/moe_training_control.py
python -m pytest tests/test_moe_training.py -q
~~~

这个独立 PyTorch CPU Float64 control 使用 5 个 token、3 个 MLP experts 与稳定 top-2。Sparse dispatch 只运行实际选中的 token—expert pair；dense oracle 运行所有 pair，再用完全相同的 gate mask 合并。两条路径对 `task + 0.05L_{bal}^{ref} + 0.001L_z` 的输出最大差为 0，所有 router/expert 参数的 backward 最大差约为 (6.94\times10^{-18})。一次真实 SGD step 同时改变 router 和三个 experts；当前 authored task loss 从约 0.0886473 降到 0.0875580，只能证明该单步链路执行，不能推断收敛。

同一训练图还执行 score-priority capacity/drop。固定 \(\phi=0.5\) 时 capacity 为 2，pre/post counts 为 `[4,3,3]→[2,2,2]`，10 个 assignments 丢 4 个但无整 token 全丢；capacity-enabled sparse/dense 的输出与全参数梯度最大差均为 0。Drop 后重归一化会把有幸存 assignment 的 token 权重和恢复到 1；保留丢失 mass 的策略不这样做，当前两者输出最大差约为 0.125542。独立同分 fixture 还让后两个 token 的 assignments 全丢，并观察 routed expert 输出精确为零；这不包含残差或 shared expert。

另一个固定 control 显式提供 token mask 与 CPU-local routing-group IDs。最后一个 padding token 不参与 capacity、drop 分母、expert dispatch、balance/z-loss 或 routed output；修改其 hidden value 和 group id 后，active output 与 aux diagnostics 差仍为 0，padding hidden gradient 也为 0。两个 active groups 各有 2 tokens，因此各组 capacity 都是 1；逐组 pre/post counts 为 `[2,1,1]→[1,1,1]` 和 `[1,1,2]→[1,1,1]`。逐组 \(L_{bal}^{ref}\)/z-loss 按 active-token 数加权。与把四个 active tokens 放入一个 capacity=2 的 group 相比，kept assignments 改变，输出最大差约 0.329387；这说明 routing group 是公式输入，不是可省略的实现细节。

Hard top-k indices 是离散选择。当前主任务到 router 的可微路径来自 selected softmax probabilities：若只 detach combine weights，expert index 与 expert forward 仍完全相同，三个 experts 都有非零 task gradient，但 router 的 task gradient 为 `None`。这不是说所有 MoE 都必须使用同一 gate 公式；它证明实现者若无意 detach，就会把 router 从主任务梯度中切断。

训练 control 对 (f_e) 使用 stop-gradient，对 (p_e) 保留梯度。在 collapsed top-1 fixture 中，一次只更新 router 的 balance step 把 (L_{bal}^{ref}) 从约 2.567724 降到 2.552751，而四个 hard assignments 仍全选 expert 0。这说明连续 probability pressure 可以先于离散 route change，但不证明后续一定均衡、不会 oscillate、task quality 不受损或 experts 会形成可解释专门化。

v3 又把 overflow policy 做成可执行反事实。4 个相同 token 的 top-1 都先选 expert 0，完整稳定排名为 `[0,2,1]`，每 expert nominal capacity=2。`drop` 保留前两个并丢 2 个；deterministic `reroute` 按原 selected score/token/rank 处理 dropped slots，扫描完整 ranking、禁止同 token 重复 expert，最终 dispatched experts 为 `[0,0,2,2]`、post-policy excess=0；`dropless` 保留 `[0,0,0,0]` 并显式报告 expert 0 超额 2，而不是声称 capacity 得到满足。Reroute 还分别执行 post-policy renormalization 与“以原 selected top-k mass 为分母”的保留策略；后两 token 的 weight sum 为 1 或约 0.449329，输出确实不同。两条新策略的 sparse/dense forward 与 materialized-zero backward 对账为 0。

这仍只是本仓库的 deterministic full-ranking reroute 与 dropless nominal-capacity-excess contract，不是某个框架或模型的默认行为。它没有跨 device/process 的 distributed capacity-group collective、shared/fine-grained expert、all-to-all、grouped GEMM、GPU、目标 checkpoint、收敛、质量或性能证据。整数 group IDs 不证明通信域真的一致，也不执行任何 collective；它与 NumPy oracle 使用不同 fixture，不能合并外推为 DeepSeek/Qwen 已复现。

### 14.4 Collective capacity group 与 expert parallel 不是同一件事

仓库另有一条独立的 two-process CPU/Gloo control，专门补“整数 group ID 不等于 collective”这条边界。Rank 0/1 各持有两个 token；真实 `all_gather` 形成 `[2,1,3,0.5]` 的 4-token replicated global routing batch，两个 `all_reduce(SUM)` 又分别确认 global active count=4、selected counts=`[4,0]`。在 `E=2,k=1,φ=0.5` 下，每 rank 独立计容会各保留 1 个、合计 2 个；global score-priority competition 的 capacity=1，只保留 score 最高的 rank-1/token-0，mask `[F,F,T,F]`、counts `[4,0]→[1,0]`、drop=3。Rank 0 的 local-only 与 collective routed output 最大差为 `0.9640275800758169`，因此这不是只打印 collective 元数据的空对照。

但这条 control 为了隔离 capacity group，复制了 router、experts 与 gathered hidden states；它没有把 token 发到 owner expert，也不执行 `all_to_all`、`reduce_scatter`、distributed autograd 或 backward。因此它证明“当前全局 routing input/competition 经真实 collective 建立”，不证明 expert parallel。Same-host Gloo/FileStore 也不能外推为 NCCL、多节点、GPU grouped GEMM、通信量、tail latency、目标模型策略、收敛或质量。

## 15. Expert parallel 与通信

Expert weights 分布在设备上时，token 按 routing 做 all-to-all dispatch，再返回原设备。成本取决于 token 数、hidden size、top-k、dtype、网络、负载不均和消息大小。

### 15.1 真实 token-to-owner all-to-all control {#moe-all-to-all-control}

仓库的独立 CPU/Gloo fixture 让 expert 0/1 只驻留在 rank 0/1，并用 variable splits 完成 count exchange、token/gate 与 metadata dispatch、owner forward、output/gate 与 metadata return。Source→owner counts 是 `[[1,2],[1,0]]`；每 rank 共执行五次 `all_to_all_single`。Rank 0 的 return arrival 的 global token 顺序为 `[1,0,2]`，不是 source-local `[0,1,2]`，所以必须用 source rank/local index metadata scatter 后再乘 gate。正确输出与单进程 oracle 精确对账；按 arrival row 直接合并的最大差为 `0.8958737432590591`。

这份 authored top-1 fixture选择保留 selected softmax probability，因此输出不是简单的 raw expert output；目标实现也可能把幸存 top-1 gate 归一化为 1。逻辑张量账本为 416 bytes，但没有测 wire/protocol bytes。它只证明同机 CPU/Gloo 前向通信与顺序恢复，不含 capacity/drop、distributed backward、CUDA/NCCL、多节点、目标模型、性能、收敛或质量；上一节的 replicated global-capacity control 与本节也不能拼成一个已验证的完整 EP 实现。

### 15.2 Authored all-to-all autograd 与梯度归约

另一条独立 training control 用 authored autograd Function 包装 variable-split payload：forward 按 source→owner splits 通信，backward 交换 reverse splits，再把 output/gate 梯度送回 owner、hidden/gate 梯度送回 source。每个 rank 的 local loss contribution 都按 global-token mean 的分母 4 缩放。Owner expert 已处理来自所有 sources 的本 expert tokens，所以其参数梯度留在 owner；replicated router 只积累本 source token 的 gate 梯度，必须 SUM all-reduce。

固定 fixture 的 router global gradient 为 `[[2.2904292655042227],[-2.290429265504225]]`，一步 SGD 后两 rank 的 router 和各 owner expert 都与单进程 global-batch oracle 对齐；前后分布式 MSE 为 `20.78017329703821→19.41091750734501`。Call ledger 中 payload forward/backward 为 4/2、count+metadata 为 6、router all-reduce 为 1，但这只是 authored wrappers 计数。它不使用 DDP、`torch.distributed.autograd` RPC、capacity/drop、CUDA/NCCL、目标模型或性能测量；不能把单步 loss 下降解释为收敛、专门化或质量改善。

### 15.3 Capacity、owner dispatch 与 backward 同图

第四条独立 control 在同一个 two-process CPU/Gloo autograd 图中加入 global score-priority drop。四个 active tokens 的 top-1 selected counts 为 `[2,2]`；`φ=0.5` 时每 expert capacity=1，global keep mask `[F,T,T,F]`，仅 token 1/2 分别发往 owner 0/1。两 rank 的 source→owner splits 为 `[[1,1],[0,0]]`，所以 rank 1 的 source 侧 dispatch/return 都是零行，但 owner 1 仍处理来自 rank 0 的 token。

零行并不表示该 rank 可以跳过 collective。实现保留一个依赖 returned empty tensor 的 zero-size graph edge，使 rank 1 backward 仍按相同顺序执行 reverse all-to-all；否则 rank 0 会等待未参与的 peer。Dropped token 0/3 的 routed output 与 task hidden gradient 为 0，而 rank-1 router local gradient也为零；router SUM gradient、owner expert gradients、一步参数及 post-step forward 都与单进程 capacity oracle 对齐。Global MSE `15.253670387373656→14.530264380025987`，strict fingerprint `sha256:33f11f199b9668c…`。

这证明当前 authored drop policy、capacity group、kept-only owner dispatch 与 reverse backward 可在这个同机 Float64 fixture 中组合；不证明 reroute/dropless、shared/fine-grained experts、DDP/FSDP/ZeRO、mixed precision、CUDA/NCCL、多节点、目标 DeepSeek/Qwen policy、通信性能、扩展性、收敛或质量。Call ledger 仍只是源码包装器计数，不是 backend profiler。

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

仓库已有生成采样、paired evaluation、KV 公式、Roofline、Agent 工具安全与 scaling 计算器。MoE 侧除 NumPy top-k/capacity/drop/combine oracle 外，已有 CPU Float64 trainable router/MLP 的 sparse—dense forward/backward、overflow policy 对账，以及四条相互隔离或递进的同机 Gloo controls：replicated global capacity competition、owner-only token dispatch/return、无 capacity 的 authored reverse-split training，以及 capacity-aware kept-only reverse-split training。仍没有目标 DeepSeek/Qwen MoE checkpoint、shared/fine-grained experts、reroute/dropless distributed training、DDP/FSDP/ZeRO、CUDA/NCCL、多节点、GPU grouped GEMM 或性能/质量实跑；因此这些 CPU fixtures 不是目标模型或生产 EP 复现。推理侧也没有训练 PRM/online reasoning policy，长上下文侧没有目标 checkpoint 的全长度矩阵，本章相应内容仍是实验协议与机制教材。

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
