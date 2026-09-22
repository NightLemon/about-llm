# LLM 面试深挖验证索引

本页是跨主题证据索引，不是第二套教材。第一次准备面试请先读
[回答方法与核心 30 题](../career/interview-questions.md)；理解机制时回到专题正文，核对运行结果时再进入对应证据页。

**证据导航**：[回答方法](../career/interview-questions.md) · [系统设计](../career/system-design.md) ·
[简历项目](../career/resume-projects.md) · [项目实验台账](project-controls.md) · [准确性总台账](accuracy-ledger.md)
{ .doc-nav }

## 查用方法

每一行固定回答四件事：要追问的 claim、权威讲解、可执行证据，以及证据不能外推到哪里。
编号沿用原深挖题，便于旧笔记继续定位。链接页保存完整公式、命令、版本、原始数字和失败记录；
本页只保留跨页面检索所需的最小摘要。

## Transformer 与生成

| 题号 / claim | 权威讲解 | 可执行证据 | 边界 |
|---|---|---|---|
| 1–3：缩放、causal mask、MHA/MQA/GQA | [Transformer](../core/transformer.md) | [Transformers 证据](transformers-controls.md)；`tests/test_attention_numpy.py` | 数学与 CPU fixture 不证明 GPU kernel、峰值显存或质量。 |
| 3.1：能否从 `config.json` 估 KV | [推理优化](../systems/inference-optimization.md) | [Qwen](qwen-controls.md)与 [Llama](llama-controls.md)的 config/weight 账本 | 理想式为 `2 × L × H_kv × d_head × tokens × batch × bytes(element)`；MLA、allocator、scale 和 workspace 要另算。 |
| 3.2：online softmax 是否把 attention 变线性 | [推理优化](../systems/inference-optimization.md) | `projects/transformers-basics/online_softmax_demo.py` | 它减少 score/probability 中间存储；精确 dense QK/AV 算术仍是 \(O(T_qT_kD)\)。 |
| 4：prefill 与 decode 的瓶颈 | [推理优化](../systems/inference-optimization.md) | [Serving 证据](inference-serving-controls.md) | TTFT、TPOT、带宽与利用率要在目标 workload/runtime/hardware 上测。 |
| 5：temperature、top-k、top-p | [生成与解码](../core/generation.md) | [编码题 2](../career/coding-sampling.md)；`tests/test_sampling.py` | 固定 CPU 顺序不代表 Transformers、vLLM 或 provider 默认，也不证明生成质量。 |
| 5.1：tokenizer/model/generation EOS 不一致 | [生成与解码](../core/generation.md) | `projects/transformers-basics/generation_runtime_control.py` | 静态 ID 一致不证明运行时 stop、finish reason 或服务覆盖参数正确。 |
| 5.2：activation patching 的因果范围 | [机制可解释性](../core/mechanistic-interpretability.md) | [Qwen patching 台账](qwen-controls.md)；`tests/test_activation_patching.py` | 只支持给定干预、样本、site 与 metric 的局部因果作用，不证明唯一或完整回路。 |
| 5.3–5.3.1：stop string 与 SSE 断连 | [生成与解码](../core/generation.md)；[Serving](../systems/serving.md) | [本地 HTTP/取消 control](inference-serving-controls.md#local-http-cancel) | 客户端截断、ASGI 取消、backend 停止、KV 释放和停止计费是不同证据层。 |
| 5.4–5.5：token/费用 hard cap 与 retry | [云 API 可靠性](../models/cloud-api-reliability.md) | [云 API 证据](cloud-api-controls.md)；[实验 0C](../practice/labs/lab-0c-cloud-budget.md) | 每个 attempt 独立 reserve/settle；本地账本不证明 provider usage、发票或全局配额。 |

### 已录制观察值

| Control | 记录 | 只允许推出 |
|---|---|---|
| 随机 MiniGPT activation patching | joint recovery `1`，future control `0` | 当前 hook 与 causal-mask 控制流成立；token pair 是事后选择。 |
| Tiny Transformers cooperative cancel | 随机 GPT-2 共 `1,272` 参数；thread 在显式 stop check 后返回 | 显式 cooperative hook 可工作；不能强杀 thread，也未观察 GPU/KV。 |
| Attempt-level 费用预留 | cap `80` micro-USD；limit `140` 时第二次 `80` 会把 projected total 推至 `160`；若第二次实际 usage `66`，逻辑调用合计 `146` | 本地预算会在下一次 transport 前阻断；不证明 provider 真实计费。 |

## 训练、对齐与 MoE

| 题号 / claim | 权威讲解 | 可执行证据 | 边界 |
|---|---|---|---|
| 6–10：loss、LoRA/QLoRA、assistant-only labels、RAG 或微调 | [SFT 数据闭环](../training/sft-data-pipeline.md)；[LoRA/QLoRA](../training/peft-qlora-engineering.md) | [Qwen 证据](qwen-controls.md)；[项目台账](project-controls.md) | 单步、单样本和独立重载证明 plumbing，不证明收敛、质量、CUDA 或 QLoRA。 |
| 9.0：模板 mask 到 Trainer labels | [SFT 数据闭环](../training/sft-data-pipeline.md) | `projects/single-gpu-finetuning/run_qwen_target_sft_label_control.py` 对应记录见 [Qwen 台账](qwen-controls.md) | 必须核对模板、padding/truncation、collator 三层；no-grad loss 不是训练成功。 |
| 9.0.1：目标 Qwen LoRA backward/export/reload | [LoRA/QLoRA](../training/peft-qlora-engineering.md) | [Qwen 台账](qwen-controls.md)；`tests/test_lora.py` | Base 冻结和 reload exact 不等于 objective 改善或部署兼容。 |
| 9.0.2–9.0.4：token mean、DDP `D/N`、`no_sync` | [分布式训练](../systems/distributed-training.md) | [准确性台账](accuracy-ledger.md)中的 accumulation/DDP controls | `loss/accumulation_steps` 对可变 token 数可能错误；`no_sync` 要在 forward 前改变 `require_backward_grad_sync`。 |
| 9.0.5–9.0.6：AMP unscale/clip、overflow 共识 | [分布式正确性](../systems/distributed-training-correctness.md) | [准确性台账](accuracy-ledger.md)中的 GradScaler controls | 单 rank overflow 是否传播取决于故障位置和 collective；scheduler 只能随已提交 update 前进。 |
| 9.1–9.2：split 泄漏、MinHash/LSH | [SFT 数据闭环](../training/sft-data-pipeline.md) | [准确性台账](accuracy-ledger.md)的数据审计记录 | Lexical Jaccard/LSH 可能假阴性，不覆盖语义改写、翻译、许可或隐私。 |
| 11–11.5：DPO、PPO、RM、GAE、sampled KL | [对齐进阶](../training/alignment.md) | [对齐证据](alignment-controls.md)及 `smoke_trl_dpo.py`、`smoke_text_ppo.py` | Tiny objective/control-flow 证据不等于人类偏好、目标模型质量或训练稳定。 |
| 11.6：RM proxy 与真实目标 | [对齐进阶](../training/alignment.md) | `projects/single-gpu-finetuning/smoke_learned_rm_ppo.py` | 冻结 tiny support 的可穷举反例证明 proxy exploitation 可发生，不是生产实证。 |
| 11.6.1–11.7：self-consistency、oracle@N、selected@N | [推理系统](../frontier/reasoning-systems.md) | [前沿证据](frontier-controls.md)；项目中的两个闭式 toy | i.i.d. 与 deterministic verifier 是教学假设；真实系统还要算相关性、校准、延迟和费用。 |
| 12：scaling law | [规模与缩放](../core/scaling.md) | [准确性方法](../reference/accuracy.md) | `6ND` 是相近数据、架构和口径下的一阶经验式，不含全部通信、重跑和产品约束。 |
| 13–13.3：MoE capacity、drop/reroute、EP、router gradient | [MoE 系统](../frontier/moe-systems.md) | [前沿证据](frontier-controls.md)及 `moe_*_control.py` | 分开 `dropped assignments / (Nk)` 与 `all-assignments-dropped tokens / N`；CPU/Gloo 不证明 NCCL 或性能。 |
| 14：长上下文 | [长上下文系统](../frontier/long-context-systems.md) | [前沿证据](frontier-controls.md) | API 接受长度、位置外推、任务有效长度和成本必须分别测。 |
| 14.1：持续学习与 replay | [持续学习](../training/continual-learning.md) | [项目台账](project-controls.md) | 多 seed 只覆盖实际变化的随机源；任务、数据和额外样本/FLOPs 预算仍可能不公平。 |

### 已录制观察值

| Control | 记录 | 只允许推出 |
|---|---|---|
| Qwen assistant labels | 三条输入长度 `47 / 301 / 200`；assistant tokens `8 / 51 / 31`；collator `[3,301]`、监督 labels `90`、loss `1.251716` | 固定模板和 TRL collator 消费了预期 labels；不证明 backward 或质量。 |
| Qwen LoRA 单步 | `270,336` adapter 参数，reload logits max error `0`；loss 约 `0.003864 → 0.584557` | 训练/导出链路执行；loss 反而说明不能把 plumbing 当改善。 |
| 单进程 token mean | counts `[1,3]`；full/count-scaled `(23/40,-23/40)`，naive `(7/20,-7/20)` | 等权 micro-batch mean 的局部反例。 |
| 默认 DDP token mean | `D=2,N=4`；`D/N=1/2` 得 `(23/40,-23/40)`，漏 world size 得 `(23/80,-23/80)` | 当前 default mean reducer 需要匹配全局 token 分母。 |
| DDP accumulation | counts `[[1,2],[3,1]]`、`N=7`；正确 pre-clip gradient `(+19/35,-19/35)`；正确 scope 1 次 hook，只包 backward 为 2 次 | `no_sync` 必须覆盖 forward+backward；错误 scope 在该线性 fixture 上未必改数值。 |
| AMP accumulation/recovery | 正确路径 `24→3→约0.5`，错误路径 `24→约0.5→约0.0625`；overflow scale `8→4→2→1` | 先 unscale 再 clip；漏 scaler 状态会改变后续是否提交 update。 |
| LSH snapshot | 64 hashes、`16×4` bands；10 pairs 得 3 candidates、1 TP、2 FP，precision `1/3`；1-hash 反例漏掉 Jaccard `2/3` | 当前 lexical fixture 的候选行为，不是“无泄漏”证明。 |
| Qwen DPO 单步 | 96 个 finite gradients、两条 positive margin；reference replay drift `0.547077` | 参数/state/config 冻结与浮点 bitwise replay 是不同命题。 |
| Learned-RM PPO | train accuracy `1`、margin `5.57`；chosen 在 57 条中排第 38；proxy `2.739→4.652`，严格成功 `1/64→4.99×10^-4`，partial credit `15/64→0.566` | Authored tiny support 中 proxy 与外部目标可背离。 |
| Self-consistency | 边缘 `p=0.6`；N=11 时 independent `0.75349813248`，latent-correlated `0.53896454244` | 平均单样本准确率不能代替候选相关结构。 |
| Verifier best-of-N | `wrong/correct/verifier_hack` 概率 `0.5/0.4/0.1`；N=`1/4/16` 的 oracle 为 `0.4/0.8704/0.9997178890`，selected 为 `0.4/0.5936/0.1852867601` | 增加 N 可提高 oracle/proxy 并降低真实选择成功。 |
| MoE capacity/grouping | factor `0.5` 时 capacity `2`，counts `[4,3,3]→[2,2,2]`，drop `4/10`；两组 capacity `1` 改成单组 capacity `2` 后最大输出差约 `0.329387` | Capacity 必须绑定 routing group 与 assignment/token 分母。 |
| MoE reroute/EP | top-1 drop `[2,0,0]` 丢 2，reroute `[2,0,2]` 无 drop，dropless `[4,0,0]` 超额 2；return arrival `[1,0,2]`，漏 metadata 最大差 `0.8958737432590591`；logical payload `416` bytes | Dispatch policy、metadata scatter 与 wire/性能账本是三件事。 |
| MoE router gradient | 两 rank local gradients 为 `[[1.8045724],[-1.8045724]]` 与 `[[0.4858569],[-0.4858569]]`，SUM 后对齐单进程 oracle | Replicated router 需要跨 source ranks 汇总；owner-only expert 参数是否归约取决于 ownership 与 loss 分母。 |

## RAG

| 题号 / claim | 权威讲解 | 可执行证据 | 边界 |
|---|---|---|---|
| 15–16：chunk 与 BM25/dense | [RAG 检索](../applications/rag-retrieval.md) | [RAG 证据](rag-answer-controls.md)；[BM25/RRF 练习](../career/coding-bm25.md) | Chunk、召回器和融合策略必须用同一语料、权限、qrels 与预算比较。 |
| 17–19：失败分层与 Recall/MRR/nDCG | [RAG](../applications/rag.md) | `projects/rag-foundations/rag_request_walkthrough.py` | 检索指标不代表 packing、生成、引用或发布成功。 |
| 19.1–19.3：Prompt token 预算、请求 identity、reranker ACL | [RAG 请求生命周期](../applications/rag-request-lifecycle.md) | [RAG 证据](rag-answer-controls.md) | 分段 token 长度不能代替完整 rendered prompt；recorded score 不认证 scorer 执行。 |
| 19.4–19.4.1：extractive baseline 与 citation 分层 | [RAG 生成](../applications/rag-generation.md) | `tests/test_rag_request_walkthrough.py` | Syntax、授权 source、exact span 与 `supported/contradicted/insufficient` entailment 是不同判断。 |
| 19.5：原生、LangChain、LlamaIndex 公平比较 | [RAG 生产](../applications/rag-production.md) | [项目台账](project-controls.md) | Adapter parity 不证明框架默认 ACL、learned quality 或生产性能。 |
| 19.6：FastAPI 504 后后台 work | [Serving](../systems/serving.md) | `tests/test_rag_service.py` | `asyncio.wait_for(asyncio.to_thread(...))` 取消等待，不会强制停止 thread、SQLite、模型 kernel 或费用。 |
| 19.7–19.8：空证据、abstain/reject/error | [RAG 请求生命周期](../applications/rag-request-lifecycle.md) | `tests/test_rag_guarded_transformers_control.py` | Audit/public projection 必须分开；直接 `return decision.to_dict()` 可能把被拒 raw output 泄露给用户。 |

固定 citation 反例把 “The moon is cheese.” 的 quote 精确指向 `Earth is round.` 中的 `Earth`，span identity 仍可通过；
它专门说明 exact quote 不等于 claim entailment。真实 guarded Qwen control 中，有证据 case 调用生成一次后 reject，
空证据 case 的 callback/framework method 均为 `0` 次并 abstain；这只约束固定 CPU API 路径。

## Agent 与互操作

| 题号 / claim | 权威讲解 | 可执行证据 | 边界 |
|---|---|---|---|
| 20–23：Agent/workflow、幂等、注入、停止 | [Agent 任务生命周期](../applications/agent-task-lifecycle.md) | [Safe Agent 项目](../practice/projects/safe-agent.md)；`projects/safe-agent/refund_lifecycle.py` | 模型 proposal 不拥有执行权；local completed/timeout 都不证明远端 effect。 |
| 23.1–23.1.1：effect verifier 与 outbox | [Agent 运行时](../applications/agent-runtime.md) | `tests/test_agent_outbox.py` | Outbox 保证本地原子提交与 at-least-once 投递，不保证远端 exactly-once。 |
| 23.2–23.6：评测、fingerprint、cache auth、snapshot、resume | [Agent 评测](../quality/agent-evaluation.md) | [准确性台账](accuracy-ledger.md) | SHA-256 绑定所选 bytes，不保密、不认证主体，也不能替代每次授权。 |
| 23.7：自由文本到 typed loop | [Agent 运行时](../applications/agent-runtime.md) | `projects/safe-agent/model_planner_control.py` | Closed JSON/schema 只检查结构；resource、tenant、policy、approval 和 effect 仍在模型外。 |
| 23.8–23.8.3：MCP memory/stdio/HTTP | [Agent 互操作](../applications/agent-interoperability.md) | `mcp_sdk_memory_control.py`、`mcp_sdk_stdio_control.py`、`mcp_sdk_streamable_http_control.py` | Schema error 可成为 `isError: true`，unknown tool 仍需应用 gate；局部集成不等于 conformance 或授权。 |
| 23.9：A2A Agent Card 与 completed task | [Agent 互操作](../applications/agent-interoperability.md) | [项目台账](project-controls.md) | Capability/card/task 状态不认证 endpoint、主体、资源或业务效果。 |

MCP 固定记录使用 `mcp==1.29.0`、协议 `2025-11-25`；stdio control 走真实 OS pipe，HTTP control 在 loopback
观察到 `7 POST / 1 GET SSE / 1 DELETE`。这些数字只描述该固定 SDK/transport control，未运行 conformance suite，
也未覆盖 TLS、OAuth、远程 server、多 worker 或生产 supervisor。

## 评测与发布证据

| 题号 / claim | 权威讲解 | 可执行证据 | 边界 |
|---|---|---|---|
| 24–24.2：Judge 偏差、文本指标、JSON 层级 | [评测测量](../quality/evaluation-measurement.md) | [准确性台账](accuracy-ledger.md) | Exact、schema-valid、value exact 和业务语义不可互换；`{"answer":43}` 也可通过只约束 integer 的 schema。 |
| 25–25.3：paired/cluster bootstrap 与 sign flip | [评测统计](../foundations/evaluation-statistics.md) | `clustered_bootstrap_toy.py`、`paired_randomization_toy.py` | 每个 slice 记录 `exact|monte_carlo`；统计必须匹配 case/cluster 单位，p-value 不是 posterior。 |
| 25.4：Holm correction | [评测统计](../foundations/evaluation-statistics.md) | `tests/test_evaluation_statistics.py` | Holm 控制预定义 family 的 FWER，不修复事后选指标、错误单位或无效检验。 |
| 25.5：重复查看与可选停止 | [评测统计](../foundations/evaluation-statistics.md) | `projects/evaluation-gate/sequential_peeking_toy.py` | 固定样本 p-value 不能直接用于“每周显著就停”。 |
| 26–29：test 污染、slice gate、pass@k、Judge 校准 | [评测方法](../quality/evaluation-methodology.md) | [Evaluation Gate](../practice/projects/evaluation-gate.md) | 平均值和通过样本分母不能隐藏关键 slice、失败、拒答或 timeout。 |
| 29.1–29.3：配对 identity、HMAC 链、证据图复算 | [推理 artifact 安全](../quality/reasoning-artifact-security.md) | `tests/test_model_release_evidence.py` | 自洽 fingerprint、authenticated chain、外部 trusted head 与完整复算是不同命题。 |
| 29.4：HTML 报告安全与 scope | [推理 artifact 安全](../quality/reasoning-artifact-security.md) | [项目台账](project-controls.md) | 对 `script/img/onerror` 等动态文本统一 escape；receipt 要保留 `statistics_recomputed=false` 与 `artifact_authentication_verified=false`。 |

### 已录制观察值

| Control | 记录 | 只允许推出 |
|---|---|---|
| Qwen 文本指标 | 七例 literal/normalized/token-F1 为 `4/7、5/7、6/7`；`LLM-2026 → llm-2026` 为 `0/1/1`，`{"answer":42} → {"answer": 42}` 为 `0/0/1` | 三种 construct 在固定小集上会给不同结论；不是总体质量。 |
| Cluster sign flip | 用户 A 有 5 个 `+1`、用户 B 有 1 个 `-1`；逐 case p=`7/64`，case-weighted cluster p=`2/4`，equal-cluster effect=`0` | 随机化单位与 estimand 会改变问题本身。 |
| Sequential peeking | looks `n=[10,20,30,40,50]`；naive 首次拒绝约 `0.1010`，最终单看约 `0.03284`；每次阈值 `0.01` 后总体约 `0.01522` | 当前 fair-sign exact toy 展示可选停止膨胀；不证明真实 case 独立。 |

## 系统、量化与恢复

| 题号 / claim | 权威讲解 | 可执行证据 | 边界 |
|---|---|---|---|
| 30–30.1.1：TTFT、open/closed loop、admission | [Serving](../systems/serving.md) | [Serving 证据](inference-serving-controls.md) | `estimated_prompt + output_cap - 1` 只是一阶 position reservation；client queue 不等于 server queue。 |
| 30.2–30.4：Paged KV、continuous batching、preemption、prefix cache | [推理优化](../systems/inference-optimization.md) | [Serving 项目](../practice/projects/inference-serving.md) | `prompt+output-request_count` 是受限条件下的 logical work；CPU state machine 不证明真实 KV/VRAM/性能。 |
| 30.5–30.6：beam 与约束解码 | [生成与解码](../core/generation.md) | [Transformers 证据](transformers-controls.md) | 必须对 token 完整片段执行 grammar 转移；只检查片段首字符会错误放过 `1]` 一类 token。 |
| 30.7：OpenAI-compatible demo 是否执行目标权重 | [Serving](../systems/serving.md) | [目标服务 control](transformers-controls.md) | Loopback Transformers CPU 证据不等于 vLLM/CUDA、TLS、远程、性能或兼容性。 |
| 31–31.2：4-bit、bundle、单矩阵与 INT8 KV | [推理优化](../systems/inference-optimization.md) | [Qwen 量化记录](qwen-controls.md) | Row-group FP32-scale 理想下界为 `ceil(bRC/8) + 4R*ceil(C/G)`；artifact、内存、kernel 与质量仍分别验证。 |
| 31.3：speculative decoding | [推理优化](../systems/inference-optimization.md) | [Transformers 证据](transformers-controls.md) | 接受质量为 `min(p_i,q_i)`，拒绝概率为 `1-sum(min(p,q)) = TV(p,q) = sum((p-q)_+)`；恒等式不保证加速。 |
| 31.4–31.4.4：checkpoint、sampler cursor、commit、JAX、跨框架 RNG | [分布式正确性](../systems/distributed-training-correctness.md) | [准确性台账](accuracy-ledger.md)；[项目台账](project-controls.md) | “能打开”只证明解析；bit-exact resume 必须绑定具体状态清单、提交边界和随机源。 |
| 32–36：消融、重放、记忆、delimiter、云 API retry | [评测方法](../quality/evaluation-methodology.md)；[云 API 可靠性](../models/cloud-api-reliability.md) | [云 API 证据](cloud-api-controls.md) | 单变量实验仍受数据/预算/seed 约束；Prompt 结构不是安全边界，HTTP class 也不是重试策略。 |

### 已录制观察值

| Control | 记录 | 只允许推出 |
|---|---|---|
| KV recompute | `executed = logical + recomputed`；本例为 `9 + 2 = 11`，请求 B 只发出 boundary `2/6` 的输出 | 重算增加 forward work，不应重复向用户发 token。 |
| Beam 反例 | root `A=0.6,B=0.4`；beam 1 返回概率 `0.306`，beam 2 返回 `0.4` | 有限 beam 不保证任意生成树全局最优。 |
| Qwen 单矩阵 INT4 | `[896,896]`、`802,816` 参数；FP32 `3,211,264` bytes，bundle `427,328` bytes，比例 `7.514752×`；仅占全模型 `0.1625%` | 只能称 selected-weight artifact/forward control，不能称整模型压缩 7.5×。 |
| 同一量化 control 的数值误差 | output relative-L2 `0.0700`；last-logits relative-L2/max-abs `0.0851/1.6255`；argmax `17→17` | Argmax 相同不能替代完整误差与任务评测。 |
| PyTorch/JAX 恢复 | JAX 6-step 在 step 3 恢复 bit-exact；漏 dropout key 差 `0.037261832505464554`，wrong cursor 差 `0.03700308472616598` | 当前 authored 单设备状态清单足够；不覆盖 Orbax、分布式或 accelerator。 |
| Accumulation crash | consumed=`3`、committed=`2`；漏 pending gradient 时参数差 `0.005767858566116724`，错用 commit-boundary RNG 时差 `0.017878893573032573` | 恢复点由是否保存半窗口 gradients/RNG 决定，不能只看 consumed cursor。 |
| 共享 dropout mask | NumPy PCG64 三张 mask 对齐三步；wrong-mask 参数差 `0.06900620367377996` | 共享输入可隔离框架计算差异，但绕开了 native RNG 等价问题。 |

## 常用执行入口

这些命令选择一条最短证据链；完整参数、依赖和 recorded verification 以链接的专题证据页为准。

```powershell
python -m pytest tests/test_attention_numpy.py tests/test_sampling.py -q
python -m pytest tests/test_evaluation_statistics.py -q
python -m pytest tests/test_rag_request_walkthrough.py tests/test_rag_service.py -q
python -m pytest tests/test_agent_refund_lifecycle.py tests/test_model_planner.py -q
python projects/transformers-basics/online_softmax_demo.py
python projects/inference-serving/self_consistency_correlation_toy.py
python projects/inference-serving/verifier_best_of_n_toy.py
python projects/evaluation-gate/sequential_peeking_toy.py
python projects/safe-agent/refund_lifecycle.py
```

## 统一解释边界

任何 control 都先标明输入、配置、脚本、判定依据和反例，再说结论。`authored fixture`、fake tool、recorded replay、
CPU control、loopback HTTP 和 Python API invocation 都适合验证局部控制流；它们不能自动升级成目标 GPU 性能、
真实 provider 调用、账单、代表性质量或生产安全。Hash 绑定所选 bytes，不认证来源，也不能证明未记录的外部状态。
