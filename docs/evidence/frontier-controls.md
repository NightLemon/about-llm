# 前沿系统证据台账：Reasoning、Long Context 与 MoE

本页保存 self-consistency、best-of-N、长上下文与 MoE routing/collective 实验的精确公式、录制结果和边界。
第一次学习请从[前沿总览](../frontier/reasoning-long-context-moe.md)进入三条独立路线。

**读者入口**：[前沿总览](../frontier/reasoning-long-context-moe.md) · [推理系统](../frontier/reasoning-systems.md) · [长上下文](../frontier/long-context-systems.md) · [MoE 系统](../frontier/moe-systems.md)
{ .doc-nav }

## 证据索引

| 主题 | 本页保留 | 首次学习 |
|---|---|---|
| Reasoning / test-time compute | Self-consistency 与 verifier best-of-N 的闭式反例、命令和边界 | [推理系统](../frontier/reasoning-systems.md) |
| Long context | Acceptance/runtime/effective 三层、评测维度和当前证据缺口 | [长上下文系统](../frontier/long-context-systems.md) |
| MoE | Routing/capacity/collective 的固定输入、精确输出与不可外推范围 | [MoE 系统](../frontier/moe-systems.md) |
| 三条路线的关系 | 它们分别增加搜索计算、可访问信息或总参数容量，不保证质量单调提高 | [前沿总览](../frontier/reasoning-long-context-moe.md) |

## 1. Reasoning controls

“推理”必须落到可判定任务、candidate/search budget、verifier、终态和成本。Reasoning SFT、outcome/process reward、
search、reflection 与 tool use 的机制由[推理系统](../frontier/reasoning-systems.md)负责；本页只记录两个可精确复算的反例。

### Self-consistency 与候选相关性

若题目只有正确/错误两个 label，N 为奇数，且每次采样真正 i.i.d.、成功率为 `p`，多数票是 binomial upper tail。
仓库另设 latent regime：每题先以相同概率进入 easy `p=0.9` 或 hard `p=0.3`，边缘成功率仍为 `0.6`，
候选 pairwise correlation 为 `3/8`。

~~~powershell
python projects/inference-serving/self_consistency_correlation_toy.py
~~~

| N | I.i.d. majority | Latent-correlated majority |
|---:|---:|---:|
| 1 | 0.60000000000 | 0.60000000000 |
| 3 | 0.64800000000 | 0.59400000000 |
| 5 | 0.68256000000 | 0.57726000000 |
| 11 | 0.75349813248 | 0.53896454244 |

N=11 对应 `2^11=2,048` 条 logical vote sequences，程序用 `Fraction` 闭式计算，没有枚举。
它只证明 authored binary/conditional-i.i.d. 反例；开放文本 plurality、canonicalization、模型、成本和真实相关性都未执行。

### Verifier best-of-N

~~~powershell
python projects/inference-serving/verifier_best_of_n_toy.py
~~~

| Candidate | Sampling probability | Verifier score | Target success |
|---|---:|---:|---:|
| `wrong` | 0.5 | 20 | false |
| `correct` | 0.4 | 80 | true |
| `verifier_hack` | 0.1 | 99 | false |

| N | oracle@N | selected@N | Gap | Expected verifier score |
|---:|---:|---:|---:|---:|
| 1 | 0.4000000000 | 0.4000000000 | 0 | 51.9000000 |
| 4 | 0.8704000000 | 0.5936000000 | 0.2768000000 | 82.7841000 |
| 16 | 0.9997178890 | 0.1852867601 | 0.8144311289 | 95.4783461 |

N=16 时 verifier hack 至少出现一次的概率约为 `0.814698`；`3^16=43,046,721` 只是 logical candidate
sequences 数，程序同样用闭式计算。该 fixture 假设有限分布、i.i.d. candidates、确定性 score 与固定 tie-break；
没有运行 model/tokenizer/PRM/provider，也没有校准、语义、wall-clock 或费用证据。

生产比较应固定总 token/candidate budget，同时报告 pass@1、oracle@N、selected@N、candidate correlation/duplicate、
truncation、model/verifier calls、tail latency 和每成功任务成本。可见 reasoning 还可能包含敏感数据、系统提示或
攻击内容；public rationale、内部 trace 与业务证据必须分开。

## 2. Long-context claim 矩阵

| Claim | 必须记录 | 当前边界 |
|---|---|---|
| API 接受长输入 | Target tokenizer token count、request/response terminal 与拒绝/截断 | 128k 请求不报错只说明 acceptance，不证明完成或有效利用。 |
| Runtime 能完成 | Prefill/TTFT/TPOT、KV peak、OOM/timeout、输出预算与并发 | Config length 不等于 runtime capacity。 |
| 任务有效 | Start/middle/end、single/multiple needle、multi-hop、冲突、顺序、聚合、长输出 | “Lost in the middle”不是所有模型/任务的固定曲线。 |
| 位置外推 | Exact position scheme、RoPE/base/scaling、继续训练与短上下文回归 | 改 `max_position_embeddings` 或 scaling factor 不是免费扩窗。 |
| 稀疏/线性/SSM 替代 | Connectivity/state、kernel、常数、质量与目标硬件 | 更低 Big-O 不自动变成更快，也可能形成检索/复制瓶颈。 |
| KV 与记忆层级 | `M_KV=2LBTH_kv d_h s` 的假设、page/prefix/quantization/eviction 与 ACL | Context、RAG、summary 和 structured store 各有不同更新、授权和信息损失。 |

当前仓库没有目标 checkpoint 的全长度矩阵。任何 context claim 都要保留 wrong answer、refusal、truncation、OOM 与
timeout 的完整分母，不能只展示 needle 命中。

## 3. MoE controls

通用参考先固定 routing group。对 N 个有效 token、E 个 routed experts、每 token top-k，
本仓库用 `C=ceil(phi Nk/E)` 作为每 expert assignment capacity，并区分 dropped assignments 与
all-assignments-dropped tokens。真实框架可能使用不同 group、minimum、dropless 或 reroute 语义。

### 单进程 routing 与 gradient

~~~powershell
python projects/transformers-basics/moe_routing.py
python -m pytest tests/test_moe_routing.py tests/test_moe_training.py -q
python projects/transformers-basics/moe_training_control.py
~~~

| Fixture | 精确记录 | 边界 |
|---|---|---|
| NumPy routing | `N=4,E=3,k=2,phi=0.75`，capacity 2；counts `(3,4,1)→(2,2,1)`；8 assignments 保留 5、丢 3，4 tokens 均有幸存 expert | 不含训练、collective、GPU 或目标 checkpoint。 |
| Sparse/dense gradient | CPU Float64，5 tokens、3 MLP experts、top-2；目标 `task + 0.05L_{bal}^{ref} + 0.001L_z`；output diff 0、gradient diff 约 `6.94×10^-18`；loss `0.0886473→0.0875580` | 单步下降不证明收敛、专门化或质量。 |
| Capacity/drop | `phi=0.5`、capacity 2；`[4,3,3]→[2,2,2]`，10 assignments 丢 4；renormalize 与保留 mass 的 output diff `0.125542` | Routed output 为零不代表带 residual/shared expert 的 block 输出为零。 |
| Routing groups | 两个 2-token groups 各 capacity 1；合并成 capacity-2 group 后 output diff `0.329387` | Integer group IDs 不证明分布式通信域。 |
| Balance step | `L_bal` `2.567724→2.552751`，hard assignments 仍全选 expert 0 | 连续 pressure 先变化不保证最终均衡或质量。 |
| Reroute/dropless | Pre-policy `[4,0,0]`；reroute dispatch `[0,0,2,2]`、counts `[2,0,2]`；dropless dispatch `[0,0,0,0]`、expert-0 excess 2；weight sums 为 1 与约 `0.449329` | Authored policy 不是任意框架默认。 |

### Collective capacity

两个 CPU/Gloo ranks 通过 `all_gather` 形成 `[2,1,3,0.5]` 的 4-token global batch，并用两个
`all_reduce(SUM)` 对账 active count=4、selected `[4,0]`。在 `E=2,k=1,φ=0.5` 下，local-only 合计保留 2，
global competition 只保留 1，mask `[F,F,T,F]`、counts `[4,0]→[1,0]`、drop=3；rank-0 output 最大差
`0.9640275800758169`。Router/experts 仍被复制，未执行 expert all-to-all 或 backward。

### Token-to-owner all-to-all {#moe-all-to-all-control}

Owner-only fixture 的 source→owner counts 为 `[[1,2],[1,0]]`，每 rank 执行五次 `all_to_all_single`。
Rank 0 return arrival 为 `[1,0,2]`；必须按 source rank/local index metadata scatter，漏掉时最大差
`0.8958737432590591`。Logical tensor ledger 为 416 bytes，不是 wire/protocol bytes。
该 control 不含 capacity/drop、backward、CUDA/NCCL、多节点、目标模型或性能。

### Backward 与 capacity-aware dispatch

| Control | 精确记录 | 边界 |
|---|---|---|
| Reverse-split autograd | Router global gradient `[[2.2904292655042227],[-2.290429265504225]]`；MSE `20.78017329703821→19.41091750734501`；payload forward/backward calls 4/2、count+metadata 6、router all-reduce 1 | Authored wrapper counts 不是 profiler；不含 DDP、`torch.distributed.autograd`、capacity 或 GPU。 |
| Capacity-aware graph | Selected `[2,2]`、`phi=0.5`、capacity 1、keep mask `[F,T,T,F]`、splits `[[1,1],[0,0]]`；MSE `15.253670387373656→14.530264380025987`，fingerprint `sha256:33f11f199b9668c…` | Zero-assignment rank 仍须参加 reverse collective；不证明目标 MoE、扩展性、收敛或质量。 |

这些 controls 分别隔离 routing、global capacity、owner dispatch 和 backward，不能拼成已经复现的完整 expert-parallel stack。
仓库没有目标 DeepSeek/Qwen MoE checkpoint、shared/fine-grained experts、DDP/FSDP/ZeRO、CUDA/NCCL、多节点、
GPU grouped GEMM 或性能/质量实跑。

## 4. Claim 审查

| 常见表述 | 证据要求 |
|---|---|
| “更长 CoT 一定更会推理” | 固定任务、总预算、终态、verifier 与切片；长度可能只是风格或重复。 |
| “Best-of-N 随 N 单调提高质量” | 同时报 oracle 与 selected；弱 verifier 会更常选中高分漏洞。 |
| “128k API 就有 128k 有效记忆” | 分开 acceptance、runtime completion 与位置/任务有效性。 |
| “稀疏 attention 更快” | 目标 kernel、常数、硬件、质量和 wall-clock profile。 |
| “长上下文可替代 RAG” | 仍需 ACL、更新、引用、降噪与失败分母。 |
| “MoE active 参数等于模型大小” | 同时报 total/active、resident memory、通信、FLOPs 与 tokens/s。 |
| “Expert 主题等于单一功能” | 需要 intervention、counterfactual routing 和跨数据稳定性。 |

复核前沿结果时同时检查 total/active 口径、training/test-time tokens、候选/verifier/tool 预算、context 任务、
latency/memory/energy、baseline 公平性、多 seed/CI/ablation，以及负结果和适用范围。
