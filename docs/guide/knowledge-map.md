# 完整知识地图

这张地图是面向进阶读者和维护者的覆盖契约与成熟度台账。第一次学习请先看[新手知识地图](beginner-map.md)，不要把下面的完整主题和证据清单当作入门前置。`✅` 表示已有可独立学习的正文，并至少有对应实现、实验或详细工程案例；`🟡` 表示已有准确的概述，但深度、例子或可执行证据仍需扩充；`⬜` 表示尚未成文。状态描述当前仓库证据，不评价主题本身是否“研究完成”。

**使用这张地图**：[新手入口](beginner-map.md) · [选择学习路径](learning-paths.md) · [查看仓库实现层次](repo-map.md) · [进入实验](../practice/labs.md) · [查看项目成熟度](../practice/project-index.md) · [核对证据台账](../reference/accuracy.md)
{ .doc-nav }

<details class="evidence-summary" markdown="1">
<summary>展开：当前实现与证据摘要</summary>

RAG 基线已能分开评 graded relevance、多证据完整召回、答案型和无答案型 query，并诊断语料/ACL/召回边界；authorization-first rerank core 会在 scorer 前二次过滤 tenant/ACL，并以 strict recorded-score artifact 验证 query/chunk/content/scorer identity、排序与篡改拒绝，CrossEncoder adapter 复用同一核心。Fixture 分数由作者构造，不证明 learned model 质量、tokenizer/truncation 或延迟。另有 optimistic-version SQLite 事务持久 chunk store、upsert/delete/retrieve 与 backup/verify/restore JSON CLI、故障回滚及物理/逻辑篡改测试，但不是 ANN/分布式向量库。灾备只证明 schema-v1 authored fixture 的单文件闭环，不认证来源、RPO/RTO 或远端依赖。上述检索、重排和存储 controls 本身没有调用 LLM，因此不构成生成忠实度或生产部署证据。

RAG 还提供端到端非 LLM extractive baseline：授权检索、byte packing、exact source span、短引用、lexical coverage 拒答与独立 artifact，并通过 5 条 authored fixture 的 3-answer/2-abstain 回归。它只证明固定语料上的 exact-substring/control-path，不证明语义相关、来源真实、答案完整、阈值校准、模型忠实或生产部署。另有 recorded answer/abstain/error 与 supplied claim judgment gate，可验证引用、权限、判断覆盖和分母；fixture verdict 是手写协议样例，不是独立人工或模型忠实度实证。

固定 Qwen RAG control 则复用不可变 checkpoint snapshot，真实执行 ACL-before-BM25、目标 tokenizer packing、逐步 greedy logits/KV cache 与 `generate()`。attempt-1 的有证据 case 漏引、空证据 case 幻觉，behavior gate 为 0/2；closed-schema report 保存失败而不 repair。policy replay 只做反事实 reject/abstain；后续不同 query 的 guarded runtime 才真实让 policy 包住 Qwen callback，观察有证据 `GenerationMixin.generate` API=1、空证据=0，且 public projection 不泄露 rejected raw。它们都不证明内部 forward/provider 计费、claim-evidence entailment、总体质量、来源认证、GPU/vLLM 或生产安全。

LangChain/LlamaIndex 不再只有对象转换 demo：offline parity control 真实调用两个框架的 Retriever 与 PromptTemplate API，把 canonical BM25 的 tenant/principal ACL、连续 rank、有限 score、正文和 metadata 逐项对账，并验证 engineering/anonymous 可见集、Prompt hash、extractive artifact、Recall@4/nDCG@4 一致。LlamaIndex node 保留控制面 metadata 供审计，但从默认 embed/LLM content 排除。它不证明框架默认 ACL、supplied canonical result 来源、learned embedding/vector index/reranker、LLM query engine、网络或生产性能。

RAG 现有 persistent extractive ASGI service 真实执行 FastAPI/Starlette/HTTPX dispatch、body 外 bearer identity resolution、SQLite 每请求重开、ACL-before-BM25、closed schema、readiness、request id、artifact response、queue saturation、execution timeout 与脱敏 500。同步 thread 在 504/client cancellation 后不会被强杀，reference 一直持有 semaphore permit 到真实 work 结束；这只保证单 event-loop 账本。Static token、ASGITransport 和单进程 fixture 不证明 JWT/IAM、TCP/TLS/proxy、全局 admission、cache、learned retrieval/LLM、负载容量或生产 SLO，项目因此标记 L2+ 而非完整 L3。

Context packing 除可注入完整 prompt cost 与 UTF-8 byte CLI 外，已有 `pack-tokenized` 目标 tokenizer/chat-template 路径：逐候选重渲染完整 prompt、预留输出、记录模板/prompt hash 和最终 token IDs。它不认证 tokenizer 与部署权重/context window 匹配；本地 WordLevel 测试不是目标模型、生成忠实度或生产吞吐证据，greedy reference 也不是最优/高吞吐实现。

Safe Agent 已有默认拒绝 exact-capability policy、server-resolved tenant、cache 重新授权、typed approval、proposal/execution 双 identity、严格结果快照、pending reconciliation、verifier-driven budgeted loop、strict JSON model-text planner、同源 Draft 2020-12 Planner/runtime schema、严格 JSON checkpoint/resume 和 trajectory gate。Recorded planner control 锁定 prompt/state/tool/budget request、schema/validator revision、normalized response 与 typed decision identity，并运行 parser→standard schema→resolver/policy→handler→verifier；LangChain/LlamaIndex tool adapter control 又把真实 framework API 限定为 proposal transport，用 direct-call schema 差异负例证明 disclosure 不等于授权。两者都无网络/真实模型；response/request id/usage/cost 是 authored fixture。Schema 不 coercion/default/authorize，scripted/recorded planner、local exact verifier、unsigned approval、离线 resolver、无密钥 hash 与非原子文件+ledger 都不能冒充真实模型/计费、开放语义判断、分布式 durable workflow、集中 IAM、签名审批或 provider effect 证据。

当前重点 RAG、Agent、SFT/QLoRA、推理部署、评测和系统安全已有进阶专章与 CPU/离线可执行基线。SFT 已有严格 JSONL、exact/group/lexical/governance gate、deterministic MinHash/LSH candidate 与 exact recheck/recall audit、有序 train/combined binding、held-out 审计进程与 train-only trainer 权限边界、tokenizer mask/截断 preflight，以及随机 tiny GPT-2 上真实 TRL label/overfit 闭环。偏好对齐已有 pairwise annotation/split contract、binary-train/combined binding、跨 A/B candidate lexical gate、source/sensitive governance、held-out-free readiness、目标 tokenizer preflight、LoRA/QLoRA DPO 入口、tiny TRL DPO 闭环，以及逐标注者 raw judgment、agreement、Fleiss’ κ 和 case-cluster position diagnostic。

PPO 证据从 NumPy GAE、两状态 MDP、integer-token Transformer 延伸到本地 tokenizer/chat-template 文本 rollout，覆盖 EOS、generation-cap truncation、padding、分离 actor/critic、冻结 reference 与精确两 token oracle；另有 sparse tiny learned RM 驱动 PPO，在 generation allowlist 的完整 57-response support 上复现 proxy expectation 上升、strict target success 下降而 dense partial credit 上升。后者只有一个 authored pair 和随机 tiny 模型，不能冒充真实人类偏好、目标模型 reward hacking、CUDA 或生产稳定性；有限时域 cap 也不默认要求 bootstrap。

readiness 未签名 hash 不认证来源；registry 不是法律结论，敏感扫描未做真实域 precision/recall 校准且不证明无 PII/secret。Lexical gate 已有 deterministic MinHash/LSH candidate、exact recheck、理想 band probability、snapshot recall audit 与固定漏检反例，但 readiness 仍用全对比较；它不覆盖语义/翻译级污染、真实域校准或规模化 recall ground truth。偏好与 judgment fixture 不是人类标签、随机实验或目标模型对齐证据。

推理部署已有单步 repetition/temperature/exact-top-k/post-top-k-top-p/CDF oracle、逐步记录 pruning/EOS/length normalization 的 deterministic beam oracle，以及完整 token-fragment 状态转移、zero-mass dead end、EOS-acceptance 分账的 finite-language constrained oracle；有限 authored distribution 上的 verifier-guided best-of-N oracle 又用精确分数分开 oracle@N、selected@N 与 expected proxy score，N=16 时前两者为 0.9997178890/0.1852867601。它只证明 i.i.d.、deterministic score 与固定 tie-break 下的闭式反例，不是 model/tokenizer/PRM/GPU/provider 或 calibration 实证。仓库还能把 failed attempt、成功 latency、吞吐和联合 SLO 分开分析，并支持 offered/dispatch 双时钟、client queue、burst/constant/seeded-Poisson finite open-loop schedule。云 API 另有 cap/model/fingerprint、内存/SQLite usage reservation、保留原重试决策/`Retry-After`/deadline 的逐 attempt JSON orchestrator：固定 500→200 先 uncertain 80、再 settled 66，hard cost limit=140 时第二次 reserve 在 transport 前失败。它不执行真实网络，不支持 streaming partial-output replay，不与 provider 原子，也不认证 usage/invoice 或提供分布式 quota/exactly-once billing。另有 FCFS/decode-first continuous-batching 离散 oracle，精确绑定 admission、chunked prefill、首 token、decode、完成与 `prompt+output-1` forward work。服务总章已把 data/control/evidence plane、workload contract、eligible offered denominator、token/KV admission、queue fairness、多副本隔离、autoscaling、clock domain、发布/回滚和 runbook 串成生产设计，但这些设计不替代目标 GPU/runtime 容量实测。Authored logits/probability table/text fragment、固定 uniform、有限任务、scheduled timestamp 与 CPU step 都不证明目标模型质量、runtime 默认等价、真实 tokenizer/JSON Schema、event loop、vLLM scheduler、server queue 或 GPU 容量。评测包含 calibration/risk-coverage、严格 run/comparison artifact 和 HMAC release ledger；后者只有配合 artifact rehash 与外部 trusted head 才分别证明当前 bytes 和尾部未截断，仍不证明 key custody、真实时间或执行来源。Prompt、代码、LLMOps、typed conversation memory、产品设计与治理已有协议或 reference core，但仍缺真实组织签署、长期服务与法律审查。基础数学、ML/NLP、生成、规模化、数据、硬件、多模态、预训练和分布式训练已补关键推导与部分解析/CPU 证据；目标多 GPU、真实端侧和多项前沿环境实证仍待补。不要把“导航中存在一页”理解成已达到研究生教材深度。

Self-consistency 现有独立与 latent-correlated binary majority exact oracle：两者边缘单样本正确率都为 0.6，N=11 多数票却分别为 0.75349813248 与 0.53896454244；相关场景每题抽一次 easy/hard regime、regime 内 conditional i.i.d.，边缘 pairwise correlation 为 3/8。它不处理开放文本 plurality/canonicalization，也没有 model/tokenizer/dataset/judge/provider 或目标质量证据。

</details>

## 1. 基础与核心模型

- ✅ 数学：张量/线性代数、概率/信息论、自动微分、优化、数值精度与配对统计实验
- ✅ 机器学习：统计风险、泛化、数据划分、损失、优化、表示学习、分布偏移与可复现训练
- ✅ NLP：语言层次、文本表示、语言模型、困惑度边界、解码、数据工程与错误分析
- ✅ Tokenization：Unicode/UTF-8 边界、BPE/WordPiece/Unigram、字节回退、词表权衡、从零 byte-BPE 与 checkpoint/template 契约
- ✅ Transformer：Embedding、mask taxonomy、MHA/GQA/MQA、RMSNorm、RoPE、MLP、残差、KV Cache 等价与 kernel 边界；NumPy blockwise online-softmax oracle 已在 causal prefill/decode/稀疏 mask/多 block size 下对齐 dense reference，并把完整 score 元素数与最大 logical tile 分账，但没有执行或测量 FlashAttention/CUDA/HBM。PyTorch↔JAX parity 又在显式 LayerNorm contract 下对齐 logits/loss、20 个参数梯度和 plain-SGD 单步，并用原生 RMSNorm 反事实阻止默认架构等价误报；不覆盖 AdamW/RNG/JIT/GPU。strict config inspector 除 authored standard-GQA/MoE-GQA fixture 外，以不可变 revision/raw hash 绑定 Qwen2.5-0.5B-Instruct 与 DeepSeek-V3 config：前者复算 32,768-token 理想 K/V payload 为 402,653,184 bytes，后者因 MLA+MoE markers 即使含 KV-head 字段也 fail closed。另有独立 target-checkpoint control 重哈希并真实执行固定 Qwen2.5-0.5B-Instruct 的 CPU FP32 prefill/cache/full/generate，不能把这个单 prompt observation 外推为质量、有效上下文、CUDA/vLLM、显存峰值或性能。Llama 3.2 仍只绑定官方 model-card fragments，DeepSeek-V3 仍只到 config evidence；均不冒充权重执行
- ✅ 生成：greedy/采样、temperature、top-k/top-p、beam search、约束解码、停止与流式协议；已有单步 processor/tie/crossing/CDF、beam pruning/EOS/length-ranking、finite-language 完整 token-transition/mask/renormalization oracle，以及跨 UTF-8 byte/event 的 partial-prefix withholding、overlap-priority stop matcher。另有 tokenizer/model/generation 三方 special-token set 对账与越界诊断，以及真实 Transformers GenerationMixin 的 forced-token EOS-set/call-override/length-cap control；后者不执行真实 tokenizer或正常模型 token 选择，仍不证明目标 checkpoint、vLLM/provider、质量或性能
- ✅ 规模化：参数/数据/算力口径、power-law 拟合、compute-optimal 解析解、生命周期成本与涌现争议
- ✅ 架构谱系：encoder/decoder、mask 变体、RNN/CNN/SSM、MoE、混合架构与公平比较；MoE 已有 padding-aware top-k、capacity/drop、gate diagnostic 与 sparse-linear combine CPU oracle
- 🟡 机制可解释性：探针、归因、activation/path patching、SAE 与模型编辑协议完整；除 seeded random MiniGPT hook fixture 外，固定 Qwen2.5-0.5B-Instruct 已在 CPU FP32 下执行单事实/单模板 source-position patch，first/lower-middle/final recovery 为 1.000024/0.992244/0，并以完整 prefix、final readout、future position 验证 1/1/0 结构控制。它仍是 authored fixed protocol 而非外部预注册，且缺多样本/模板/语言、random source、component/path、SAE 与 held-out replication，不能写成唯一自然 circuit

## 2. 数据与训练

- 🟡 采集与许可：source registry、授权范围、隐私、地域、退出与删除；具体法律审查/数据登记待项目落实
- ✅ 数据流水线：不可变快照、解析、质量过滤、exact/near dedup、污染、PII、混合、packing 与 lineage
- ✅ 预训练：next-token prediction、数据混合、token/FLOPs 预算、优化器、稳定性、检查点与继续训练
- ✅ 微调：已按失败类型区分 Prompt/RAG/SFT/DAPT/runtime 控制，并把数据接口→机制执行→训练目标→held-out 行为→发布运行拆成五层证据。覆盖 SFT、全参微调、LoRA/QLoRA、灾难性遗忘与数据配比；已有 CPU FP32 MiniGPT strict training checkpoint/bit-exact resume、单参数 CPU FP16 GradScaler overflow/state omission，以及统一 model/AdamW/StepLR/scaler/Torch+Python RNG/stateful-shuffle checkpoint 的真实跨 PID bit-exact control。独立 DataLoader control 真实启动 2 个 spawn workers/prefetch：consumed=3 时 sampler emitted=7，从 7 恢复漏 `[7,0,9,4]`，从 3 恢复 ID exact；fresh worker RNG tail 不同而 sample-ID-keyed tail exact。optimizer-commit control 再以 seed `20260815` 的 main-process stochastic mask、真实 backward、SGD momentum、StepLR 与两步 accumulation固定 emitted/consumed/committed=`7/3/2`：8,985-byte base 不含 `.grad`，从 2 恢复 commit RNG并 replay bit-exact；从 3 使用正确 crash RNG却漏 gradients/sample `1`，在同为 5 steps/LR `0.0125` 且终态 RNG 相同下参数漂移 `0.005767858566116724`。7,905-byte sidecar 绑定 base digest、pending/position/divisor、逐参数 gradients 与 crash RNG，从 3 恢复也 bit-exact；只把 RNG 换回 commit boundary 的独立反例在 ledger/steps/LR 完整时仍漂移 `0.017878893573032573`。827-byte manifest-last completeness gate 绑定两个 payload，并在反序列化前拒绝 base-only、两 payload 无 manifest、manifest 缺 sidecar与 tamper；base-only 仍可 replay。它仍未保存 queue/worker/Python/NumPy/CUDA RNG，也未实现 sample—optimizer—base+sidecar+manifest 原子事务，且无 directory `fsync`/断电、来源认证或不可变快照证据。另有 tiny GPT-2 的 13-file PEFT export，以及固定 Qwen 的 final labels/no-grad forward、LoRA backward/单步 AdamW/冻结基座/adapter reload。目标 QLoRA/CUDA AMP、完整 worker/adapter/optimizer 一致提交、distributed shard 恢复、量化 merge、目标 runtime 发布和真实数据评测仍待补
- ✅ JAX parity 与 JAX cross-process resume：除 LayerNorm/plain-SGD forward/backward 外，shared materialized masks 的三步 AdamW/clipping/schedule parity 已对账 moments/count、参数和 post-step forward，并用 wrong-mask 反例阻止把共享输入写成 native RNG 等价。strict artifact 另保存参数/Optax state、dropout/data PRNG、shuffle permutation/cursor 与 step；两个独立 CPU spawn process 在第 3/6 步交接后，sample/loss/gradient 与完整终态相对 uninterrupted bit-exact。两条证据均不包含 Orbax/TensorStore、Python/NumPy/worker/accelerator RNG、directory durability、来源认证、CUDA/TPU、多设备 sharding、目标模型、收敛或性能
- ✅ Loss reduction 正确性：已有变长 masked-token `Fraction` oracle 与单进程 PyTorch CPU Float64 backward 对照；`[1,3]` 个有效 token 时 sum/count 与 full batch 的 class-aggregate gradient 都为 `(23/40,-23/40)`，等权 micro-batch mean 为 `(7/20,-7/20)`。两条双进程 CPU/Gloo controls 锁定 default-DDP `D/N`、`no_sync` forward+backward scope、同步后 clip 与 plain SGD update；独立 CPU FP16/GradScaler control 锁定 `24→unscale 3→clip 约 0.5`、overflow scale `8→4→2→1` 与 AdamW skip。新增同路径 DDP+AMP control 证明 reduction 前 Inf 在当前 default reducer 中传播并共同 skip，也以 post-reduction rank-0 authored fault 观察 step `[1,2]` 分叉，再用 optimizer-pre `all_reduce(MAX)` gate 共同 skip。显式 `new_scale` policy 保留 tracker=1，不等同 native distributed scaler。仍无随机层、多参数/bucket、自然 overflow、FSDP/ZeRO、GPU、多节点或目标模型证据，不证明完整训练等价、性能或质量
- 🟡 偏好与对齐：RM、RLHF/PPO、DPO、RLAIF、拒答与发布门禁完整；严格 pairwise audit/binding、lexical/governance held-out-free readiness、raw judgment coverage/agreement/position 诊断、linear/tiny GPT-2 RM shortcut、masked GAE/PPO 与 sampled-ratio/KL 反例、两状态/integer-token/本地文本 PPO、冻结 sparse learned RM 后完整 support 的 proxy-exploitation 对照、真实 TRL tiny-DPO，以及固定 Qwen checkpoint 的 CPU FP32 TRL/PEFT DPO 单步已补。仍缺真实域阈值/detector 校准、语义近重复、真实人类标签/随机化实验、签名审计、可靠目标 RM、目标 checkpoint PPO、CUDA/QLoRA 与 held-out 对齐质量证据
- 🟡 持续学习：DAPT、replay、正则/蒸馏、模型合并、编辑与 unlearning 协议完整；已有显式 task-id 的两任务 CPU no/64-example reservoir/full replay、20-seed paired interval 与样本呈现成本，目标 LLM、真实时间序列、task/data 重采样、compute-matched 与安全 retention 实证待补
- ✅ 合成数据：provenance、拒绝采样、自训练、hard/soft 蒸馏、多代反馈、真实锚点、去重、混合暴露与离线审计

## 3. 系统与推理

- ✅ 训练并行：数据、张量、流水线、序列、专家并行、ZeRO/FSDP、通信模型与正确性验收
- ✅ 数值与内存：状态/激活账本、FP32/TF32/FP16/BF16/FP8、混合精度、激活检查点
- ✅ 推理：prefill/decode、KV Cache、连续批处理、PagedAttention；离散 scheduler oracle 已验证 FCFS admission、sequence/token cap、chunked prefill、prefill 首 token、decode/completion、queue/TTFT/TPOT step 和 `P+O-1` logical work；metadata allocator 已验证固定 block、prefix fork/refcount、partial-tail COW、物理碎片、释放复用与 no-capacity 原子失败；集成 oracle 另验证 block-pressure preemption、FCFS re-admission、context rebuild、无重复 emission，以及 logical/recomputed/executed=`9/2/11`。这些 CPU policy 都未执行真实模型/KV tensor、swap/prefix reuse、vLLM scheduler、GPU page table 或性能测量
- ✅ 压缩：已有 symmetric group-wise code/scale/error、dense bit packing/单矩阵 artifact、带 identity/manifest 的多矩阵 bundle，以及含 Byte-BPE merges、config、全部量化/FP32 参数、tied contract 并能恢复 forward 的 repo-native MiniGPT inference checkpoint；另有 KV INT8 GQA/incremental oracle、剪枝/蒸馏和 speculative `min(1,p/q)`/positive-residual/block oracle。固定 Qwen 又新增真实 selected-weight INT4：第一层 802,816-parameter `o_proj` strict bundle 为 427,328 bytes/7.514752×，真实 activation/整模型末位 logits relative-L2 为 0.070002/0.085138，单提示 argmax 17→17；但它只覆盖全模型 0.1625%，并用反量化 FP32 forward。MiniGPT 的完整性也只对 repo tiny revision 成立。尚未完成目标 Llama/Qwen 的完整 low-bit checkpoint/runtime、KV/speculative GPU kernel、GPTQ/AWQ、resident/peak memory 或目标模型质量/性能实验
- ✅ 服务：吞吐/延迟、TTFT/TPOT、调度、缓存、弹性、可观测性；固定 Qwen selected snapshot 已经真实进入 Transformers CPU FP32 subprocess，经 IPv4 loopback HTTP/Bearer 执行 models、拒绝负例、non-stream、SSE 与两次 `generate()` audit，但该 SSE 在完整 generation 后才发送。独立 async control 验证 content-before-completion 与 ASGI/backend cooperative cancellation；随机 1,272 参数 tiny GPT-2 control 又真实执行一次 CPU forward/`GenerationMixin.generate()` thread，并以 streamer pause + event + `StoppingCriteria` 验证退出/join。后者不证明未修改/已阻塞调用、目标 checkpoint/logits、vLLM/CUDA、KV/CPU/GPU release、TLS/IAM、多 worker、provider billing、性能或质量
- ✅ 硬件：容量账本、GPU/TPU/NPU/CPU、Roofline、带宽、kernel、互联与测量协议
- 🟡 边缘部署：CPU/移动 NPU/WebGPU、更新、内存、功耗和热约束协议完整；目标设备实测待补

## 4. 应用工程

- ✅ Prompt：任务契约、真实 chat template、few-shot/分解、结构化输出、版本化、评测与安全边界
- ✅ RAG：摄取、切分、Embedding、向量/关键词检索、重排、目标-tokenizer packing、引用、模型外 fail-closed publication policy，以及 packing→output→evaluation trace identity gate；通用 scorer 新增 strict claim→authorized-source exact-span identity metric，五例 `[1,0,0,0,1]` 用无关 claim 正对照证明 span pass 不是 entailment。canonical-first LangChain/LlamaIndex parity 与 persistent ASGI service 仍用 deterministic extractive baseline，另有固定 Qwen failure control（2 case gate 0/2）、counterfactual policy replay 和真实 guarded callback 1/0 invocation control，明确不把 method count、语法/span gate 或少量 authored case 写成语义、质量、计费或生产修复通过
- ✅ Agent：typed planner loop、strict model-text→proposal boundary、同源版本化 Draft 2020-12 Planner/runtime schema、request/response/decision identity、verifier 完成判定、step/token/cost/time 预算、循环/错误停止、工具调用、状态、记忆、权限、审批暂停与工作流；proposal-only control 先以真实 LangChain/LlamaIndex direct tool API 锁定 schema enforcement、跨 tenant、unknown-tool 与 cache replay 边界，独立 Agent-loop control 再真实执行 `create_agent()`/LangGraph 与 `FunctionAgent.run()` 的 model→tool→model，并让本地 verifier 拒绝无可接受 canonical receipt 的模型成功文本。后者仍是 scripted in-process model，LlamaIndex call ID 是可信 fixture action hash且存在 Pydantic deprecation；没有 provider、网络、remote effect、persistent resume、streaming/parallel/cancel、质量/性能或默认框架安全证据。另有 local SQLite transactional outbox 的原子 enqueue、lease、崩溃重投、dead letter 与模拟 provider 幂等实验，准确边界是 at-least-once delivery，不是远端 exactly-once
- 🟡 Agent 互操作：已覆盖 provider API、MCP 与 A2A 的分层、版本和 verifier 边界。MCP official-SDK memory control 以 `mcp==1.29.0` 执行 client/server/generated types/schema validation 但无 transport；official-SDK stdio 又经真实 OS pipe 验证 schema-invalid 未进 handler、unknown tool 进入应用 gate 与 graceful EOF；official-SDK Streamable HTTP 则让官方 client/session manager/low-level server 在独立 subprocess 与真实 loopback TCP/HTTP 上执行 stateful POST/GET SSE、DELETE 和 graceful shutdown。后两者把 SDK 与具体 transport 放进同次运行，但未借到自写 negative-control matrix。另两个自写 2025-11-25 control 分别覆盖严格 LF/UTF-8/duplicate/nonfinite/byte-cap stdio 子集，以及 Origin/Bearer、session/version、JSON/SSE、显式 cancellation 与 DELETE。五者仍都没有完整 conformance、OAuth、TLS、远程、跨厂商或业务授权证据；官方 HTTP 的私有 shutdown token 也不是 MCP auth。A2A 1.0 control 用官方 Python SDK 1.1.2 真实执行 IPv4 loopback TCP/HTTP、Agent Card、JSON-RPC SendMessage/GetTask、legacy/version 错误与可选冻结官方 Schema 校验，但没有 TCK、SSE、REST/gRPC、TLS、认证、签名 card、远程或跨厂商互操作
- ✅ 代码模型：补全/FIM、仓库检索、补丁、执行反馈、沙箱、测试、pass@k 与成本边界
- ✅ 对话系统：typed state、来源/TTL/consent、修正/撤回、摘要边界、上下文压缩与租户隔离 reference core
- ✅ LLMOps：artifact graph、canonical identity、trace、离线门禁、发布、A/B、漂移与成本管理
- ✅ 云 API 可靠性：三类 adapter、strict JSON/SSE、bounded retry/`Retry-After`/deadline/outcome guard、request-bound token/费用 reservation、SQLite durable quota，以及 JSON logical call 的逐 attempt reserve/reconcile；固定 500→200 control 记 uncertain 80 + settled 66 = 146 micro-USD，只证明 MockTransport/本地账本，不证明 provider usage、invoice、server cancellation、streaming replay 或跨系统 exactly-once
- 🟡 产品设计：状态机、不确定性、证据、确认、用户控制、可访问性和实验协议完整；真实 UI/用户研究待补

## 5. 评测、安全与治理

- ✅ 评测设计：能力、质量、事实性、鲁棒性、安全、效率、业务价值；固定 Qwen behavior control 已真实执行 7 条 authored case，并以 raw output + metric revision 对账 literal exact `4/7`、NFKC+casefold normalized exact `5/7`、token F1 `6/7`。大小写复制与 JSON 反例证明 normalization/tokenization 会改变 construct；strict JSON Schema v2 / canonical value v1 五例矩阵拒绝 duplicate/nonfinite 并分开 schema-valid 与 value-equal；citation evidence-span v1 再以 `[1,0,0,0,1]` 分开 source/span identity 与 entailment。三组 suite 都是 authored、非代表性证据，不是总体准确率、业务语义或发布结论
- ✅ 数据污染、基准饱和、提示敏感性和统计显著性；已有 paired/cluster exact-or-Monte-Carlo percentile bootstrap、case/cluster-joint sign-flip、case/equal-cluster estimand、Holm FWER 与 repeated-look exact sign-test oracle。固定 `[10,20,30,40,50]` looks 上，逐次 0.05 的首次拒绝概率约 0.1010，事前 Bonferroni 0.01 split 为约 0.0152；同时明确 coverage、exchangeability、cluster/interference、prespecified family/look/selection、confidence-sequence 与 effect-size 边界
- ✅ 评测 artifact：严格 run/comparison schema，完整本地 evidence graph 可重算 answer/case identity、scores、bootstrap 与 gate；另有 deterministic/XSS-safe/script-free HTML 派生报告、HMAC key-rotation chain、可选 artifact rehash 与外部 trusted-head rollback 检测；KMS/HSM、可信时间、透明日志和真实执行 provenance 待生产集成
- ✅ 人工评价、自动指标、LLM-as-judge、离线/在线实验
- ✅ 安全：威胁模型、滥用、注入、RAG ACL、工具副作用、SSRF、泄露、供应链、红队与事件响应；opaque reasoning artifact 已补 content-only/context-bound AEAD 对照、跨主体/租户/会话/模型负例、key/replay 边界和公开 trajectory 门禁。它是 authored 离线协议 control，不解析 provider block，也不证明当前云端行为、生产 key custody 或多区域 replay protection
- 🟡 公平、偏见、隐私、版权、透明度、问责与审计
- 🟡 法规与标准：风险分级、影响评估、数据/模型/系统卡、第三方、发布、事件与退役模板完整；正式地区/用途审查和组织签署待补
- 🟡 环境与社会影响：能源、水、劳工、集中化、信息生态

## 6. 多模态与前沿

- ✅ 视觉语言：patch/动态分辨率、投影/重采样/cross-attention、OCR/图表、grounding 与评测
- ✅ 音频/语音、视频与文档：ASR/TTS、streaming、时空采样、定位指标、数据与安全边界
- 🟡 长上下文：位置外推、稀疏/线性注意力、RAG/memory、KV 与评测协议完整；目标长上下文实测待补
- 🟡 MoE：路由、负载均衡、capacity、all-to-all 与口径完整；已有 deterministic top-k、score-priority capacity、assignment/token drop、重归一化和 sparse linear expert NumPy oracle。PyTorch CPU Float64 control 又在同一训练图真实执行 top-2 router/三组 MLP experts 与 score-priority capacity/drop，对齐 sparse/dense forward/backward，并以 detached-gate、collapsed top-1 balance、两种 post-drop policy、全丢 token、padding exclusion 与 CPU-local routing groups 拆出梯度和 overflow 语义；v3 再执行 authored full-ranking reroute、dropless excess 与 materialized-zero backward。第一条 two-process CPU/Gloo control 用 hidden `all_gather`/count `all_reduce` 建立 replicated global capacity group，证明 local-only kept=2 与 global kept=1/mask `[F,F,T,F]`；第二条执行 owner-only expert 的 variable-split token `all_to_all` forward/return 与 metadata scatter；第三条 two-process Gloo training control 用 authored autograd reverse all-to-all hidden/gate gradients、replicated-router SUM reduce、owner-expert gradients和一步 SGD 对齐 global-batch oracle；第四条 capacity-aware two-process Gloo training control 把 global drop、kept-only dispatch/backward 与一步 SGD 放在同图，并覆盖 zero-assignment source。具体 DeepSeek/Qwen checkpoint、shared/fine-grained experts、distributed reroute/dropless、grouped GEMM、DDP/FSDP/ZeRO、GPU/NCCL、多节点、收敛、专门化、质量与性能实跑仍待补，仓库策略不代表框架默认行为
  - 分布式 MoE 证据再增加同机两进程 CPU/Gloo token dispatch/return + metadata scatter：source→owner `[[1,2],[1,0]]`，owner-only experts 的 forward 与单进程 oracle 对齐。它仍不含 CUDA/NCCL、capacity/backward、目标 checkpoint 或性能，不能与其他独立 fixtures 拼接成生产 EP。
- 🟡 推理增强：过程/结果监督、self-consistency、verifier、搜索和 test-time scaling 协议完整；binary latent-regime oracle 精确展示相同单样本 0.6 在 N=11 可得到 0.75349813248/0.53896454244 两种多数票结果，有限 authored i.i.d. distribution 的 best-of-N oracle又展示 oracle/proxy 上升而 selected success 非单调。两者都明确 logical samples 不等于 wall-clock/cost；开放文本 canonicalization、真实 model/tokenizer/PRM、verifier calibration、目标质量、GPU/provider 与 online policy 实跑待补
- 🟡 世界模型、具身与 GUI：闭环、安全控制、sim-to-real、state verifier 协议完整；模拟器/硬件实测待补
- 🟡 小模型、路由、蒸馏、speculative decoding、联邦/本地隐私协议完整；目标设备级联实测待补

## 横切能力

无论学习哪个主题，都持续追问：

1. 输入输出与张量形状是什么？
2. 训练信号来自哪里，可能包含什么偏差？
3. 质量、算力、内存、延迟和成本怎样权衡？
4. 指标是否真的代表用户目标？是否存在数据污染？
5. 在恶意输入、外部工具失败或证据冲突时会怎样？
6. 哪些结论只适用于某个规模、语言、数据集或时间点？

## 从地图到行动

- 选择一组 `✅` 主题形成主线，再用相邻 `🟡` 主题定义探索边界。
- 按[学习路径](learning-paths.md)安排顺序，在[实验与项目](../practice/labs.md)中留下可复查证据。
- 实现前查看[仓库地图](repo-map.md)，不要把 L0 文档、L1 实现与 L2 实验混成同一证据等级。
