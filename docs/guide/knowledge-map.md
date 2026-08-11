# 完整知识地图

这张地图既是覆盖契约，也是成熟度台账。`✅` 表示已有可独立学习的正文，并至少有对应实现、实验或详细工程案例；`🟡` 表示已有准确的概述，但深度、例子或可执行证据仍需扩充；`⬜` 表示尚未成文。状态描述当前仓库证据，不评价主题本身是否“研究完成”。

RAG 基线已能分开评 graded relevance、多证据完整召回、答案型和无答案型 query，并诊断语料/ACL/召回边界；authorization-first rerank core 会在 scorer 前二次过滤 tenant/ACL，并以 strict recorded-score artifact 验证 query/chunk/content/scorer identity、排序与篡改拒绝，CrossEncoder adapter 复用同一核心。Fixture 分数由作者构造，不证明 learned model 质量、tokenizer/truncation 或延迟。另有 optimistic-version SQLite 事务持久 chunk store、upsert/delete/retrieve 与 backup/verify/restore JSON CLI、故障回滚及物理/逻辑篡改测试，但不是 ANN/分布式向量库。灾备只证明 schema-v1 authored fixture 的单文件闭环，不认证来源、RPO/RTO 或远端依赖。它没有调用 LLM，因此不构成生成忠实度或生产部署证据。

RAG 还提供端到端非 LLM extractive baseline：授权检索、byte packing、exact source span、短引用、lexical coverage 拒答与独立 artifact，并通过 5 条 authored fixture 的 3-answer/2-abstain 回归。它只证明固定语料上的 exact-substring/control-path，不证明语义相关、来源真实、答案完整、阈值校准、模型忠实或生产部署。另有 recorded answer/abstain/error 与 supplied claim judgment gate，可验证引用、权限、判断覆盖和分母；fixture verdict 是手写协议样例，不是独立人工或模型忠实度实证。

LangChain/LlamaIndex 不再只有对象转换 demo：offline parity control 真实调用两个框架的 Retriever 与 PromptTemplate API，把 canonical BM25 的 tenant/principal ACL、连续 rank、有限 score、正文和 metadata 逐项对账，并验证 engineering/anonymous 可见集、Prompt hash、extractive artifact、Recall@4/nDCG@4 一致。LlamaIndex node 保留控制面 metadata 供审计，但从默认 embed/LLM content 排除。它不证明框架默认 ACL、supplied canonical result 来源、learned embedding/vector index/reranker、LLM query engine、网络或生产性能。

RAG 现有 persistent extractive ASGI service 真实执行 FastAPI/Starlette/HTTPX dispatch、body 外 bearer identity resolution、SQLite 每请求重开、ACL-before-BM25、closed schema、readiness、request id、artifact response、queue saturation、execution timeout 与脱敏 500。同步 thread 在 504/client cancellation 后不会被强杀，reference 一直持有 semaphore permit 到真实 work 结束；这只保证单 event-loop 账本。Static token、ASGITransport 和单进程 fixture 不证明 JWT/IAM、TCP/TLS/proxy、全局 admission、cache、learned retrieval/LLM、负载容量或生产 SLO，项目因此标记 L2+ 而非完整 L3。

Context packing 除可注入完整 prompt cost 与 UTF-8 byte CLI 外，已有 `pack-tokenized` 目标 tokenizer/chat-template 路径：逐候选重渲染完整 prompt、预留输出、记录模板/prompt hash 和最终 token IDs。它不认证 tokenizer 与部署权重/context window 匹配；本地 WordLevel 测试不是目标模型、生成忠实度或生产吞吐证据，greedy reference 也不是最优/高吞吐实现。

Safe Agent 已有默认拒绝 exact-capability policy、server-resolved tenant、cache 重新授权、typed approval、proposal/execution 双 identity、严格结果快照、pending reconciliation、verifier-driven budgeted loop、strict JSON model-text planner、同源 Draft 2020-12 Planner/runtime schema、严格 JSON checkpoint/resume 和 trajectory gate。Recorded planner control 锁定 prompt/state/tool/budget request、schema/validator revision、normalized response 与 typed decision identity，并运行 parser→standard schema→resolver/policy→handler→verifier；但 response/request id/usage/cost 是 authored fixture，无网络/真实模型。Schema 不 coercion/default/authorize，scripted/recorded planner、local exact verifier、unsigned approval、离线 resolver、无密钥 hash 与非原子文件+ledger 都不能冒充真实模型/计费、开放语义判断、分布式 durable workflow、集中 IAM、签名审批或 provider effect 证据。

当前重点 RAG、Agent、SFT/QLoRA、推理部署、评测和系统安全已有进阶专章与 CPU/离线可执行基线。SFT 已有严格 JSONL、exact/group/lexical/governance gate、deterministic MinHash/LSH candidate 与 exact recheck/recall audit、有序 train/combined binding、held-out 审计进程与 train-only trainer 权限边界、tokenizer mask/截断 preflight，以及随机 tiny GPT-2 上真实 TRL label/overfit 闭环。偏好对齐已有 pairwise annotation/split contract、binary-train/combined binding、跨 A/B candidate lexical gate、source/sensitive governance、held-out-free readiness、目标 tokenizer preflight、LoRA/QLoRA DPO 入口、tiny TRL DPO 闭环，以及逐标注者 raw judgment、agreement、Fleiss’ κ 和 case-cluster position diagnostic。

PPO 证据从 NumPy GAE、两状态 MDP、integer-token Transformer 延伸到本地 tokenizer/chat-template 文本 rollout，覆盖 EOS、generation-cap truncation、padding、分离 actor/critic、冻结 reference 与精确两 token oracle；另有 sparse tiny learned RM 驱动 PPO，在 generation allowlist 的完整 57-response support 上复现 proxy expectation 上升、strict target success 下降而 dense partial credit 上升。后者只有一个 authored pair 和随机 tiny 模型，不能冒充真实人类偏好、目标模型 reward hacking、CUDA 或生产稳定性；有限时域 cap 也不默认要求 bootstrap。

readiness 未签名 hash 不认证来源；registry 不是法律结论，敏感扫描未做真实域 precision/recall 校准且不证明无 PII/secret。Lexical gate 已有 deterministic MinHash/LSH candidate、exact recheck、理想 band probability、snapshot recall audit 与固定漏检反例，但 readiness 仍用全对比较；它不覆盖语义/翻译级污染、真实域校准或规模化 recall ground truth。偏好与 judgment fixture 不是人类标签、随机实验或目标模型对齐证据。

推理部署已有单步 repetition/temperature/exact-top-k/post-top-k-top-p/CDF oracle、逐步记录 pruning/EOS/length normalization 的 deterministic beam oracle，以及完整 token-fragment 状态转移、zero-mass dead end、EOS-acceptance 分账的 finite-language constrained oracle；能把 failed attempt、成功 latency、吞吐和联合 SLO 分开分析，并支持 offered/dispatch 双时钟、client queue、burst/constant/seeded-Poisson finite open-loop schedule。云 API 另有 cap/model/fingerprint、内存/SQLite usage reservation，以及单-attempt HTTP reconciliation；后者锁定 pre-send cancel、HTTP/缺 usage/cancel uncertain、成功 settle 与每次 replay 独立 reservation，但不执行真实网络、不与 provider 原子，也不认证 usage/invoice 或提供分布式 quota/exactly-once billing。另有 FCFS/decode-first continuous-batching 离散 oracle，精确绑定 admission、chunked prefill、首 token、decode、完成与 `prompt+output-1` forward work。Authored logits/probability table/text fragment、固定 uniform、有限任务、scheduled timestamp 与 CPU step 都不证明目标模型质量、runtime 默认等价、真实 tokenizer/JSON Schema、event loop、vLLM scheduler、server queue 或 GPU 容量。评测包含 calibration/risk-coverage、严格 run/comparison artifact 和 HMAC release ledger；后者只有配合 artifact rehash 与外部 trusted head 才分别证明当前 bytes 和尾部未截断，仍不证明 key custody、真实时间或执行来源。Prompt、代码、LLMOps、typed conversation memory、产品设计与治理已有协议或 reference core，但仍缺真实组织签署、长期服务与法律审查。基础数学、ML/NLP、生成、规模化、数据、硬件、多模态、预训练和分布式训练已补关键推导与部分解析/CPU 证据；目标多 GPU、真实端侧和多项前沿环境实证仍待补。不要把“导航中存在一页”理解成已达到研究生教材深度。

## 1. 基础与核心模型

- ✅ 数学：张量/线性代数、概率/信息论、自动微分、优化、数值精度与配对统计实验
- ✅ 机器学习：统计风险、泛化、数据划分、损失、优化、表示学习、分布偏移与可复现训练
- ✅ NLP：语言层次、文本表示、语言模型、困惑度边界、解码、数据工程与错误分析
- ✅ Tokenization：Unicode/UTF-8 边界、BPE/WordPiece/Unigram、字节回退、词表权衡、从零 byte-BPE 与 checkpoint/template 契约
- ✅ Transformer：Embedding、mask taxonomy、MHA/GQA/MQA、RMSNorm、RoPE、MLP、残差、KV Cache 等价与 kernel 边界；另有 strict config inspector 对 authored standard-GQA/MoE-GQA fixture 复算理想 payload，并在已知 MLA marker、缺字段或不自洽 head layout 上 fail closed。它不识别所有专有 attention，也不检查权重、来源、有效上下文、runtime layout 或 GPU 峰值
- ✅ 生成：greedy/采样、temperature、top-k/top-p、beam search、约束解码、停止与流式协议；已有单步 processor/tie/crossing/CDF、beam pruning/EOS/length-ranking、finite-language 完整 token-transition/mask/renormalization oracle，以及跨 UTF-8 byte/event 的 partial-prefix withholding、overlap-priority stop matcher。另有 tokenizer/model/generation 三方 special-token set 对账与越界诊断，以及真实 Transformers GenerationMixin 的 forced-token EOS-set/call-override/length-cap control；后者不执行真实 tokenizer或正常模型 token 选择，仍不证明目标 checkpoint、vLLM/provider、质量或性能
- ✅ 规模化：参数/数据/算力口径、power-law 拟合、compute-optimal 解析解、生命周期成本与涌现争议
- ✅ 架构谱系：encoder/decoder、mask 变体、RNN/CNN/SSM、MoE、混合架构与公平比较；MoE 已有 padding-aware top-k、capacity/drop、gate diagnostic 与 sparse-linear combine CPU oracle
- 🟡 机制可解释性：探针、归因、activation/path patching、SAE 与模型编辑协议完整；已有 seeded random MiniGPT 的真实 forward-hook、联合 prefix 恢复和未来位置负对照，但 post-hoc metric/toy model 只证明管线与 causal mask，目标模型预注册因果实验待补

## 2. 数据与训练

- 🟡 采集与许可：source registry、授权范围、隐私、地域、退出与删除；具体法律审查/数据登记待项目落实
- ✅ 数据流水线：不可变快照、解析、质量过滤、exact/near dedup、污染、PII、混合、packing 与 lineage
- ✅ 预训练：next-token prediction、数据混合、token/FLOPs 预算、优化器、稳定性、检查点与继续训练
- ✅ 微调：SFT、全参微调、LoRA/QLoRA、灾难性遗忘、数据配比；已有 CPU FP32 MiniGPT + 单 AdamW group 的 strict training checkpoint/bit-exact resume，以及 tiny GPT-2 base/adapter/safe-merged/tokenizer 的 13-file strict deployment export、完整 file-set verifier、safetensors/config/tensor-signature/LoRA-target 结构校验与重载等价；目标 LoRA/QLoRA/CUDA/AMP/worker/sharded 恢复、量化 merge 和目标 runtime 发布仍待补
- 🟡 偏好与对齐：RM、RLHF/PPO、DPO、RLAIF、拒答与发布门禁完整；严格 pairwise audit/binding、lexical/governance held-out-free readiness、raw judgment coverage/agreement/position 诊断、linear/tiny GPT-2 RM shortcut、masked GAE/PPO 与 sampled-ratio/KL 反例、两状态/integer-token/本地文本 PPO，以及冻结 sparse learned RM 后完整 support 的 proxy-exploitation 对照已补；另有无 combined 文件的 RM trainer、目标 tokenizer preflight、本地 RewardTrainer/LoRA step+save、目标 RM LoRA/QLoRA 入口与真实 TRL tiny-DPO 闭环。仍缺真实域阈值/detector 校准、语义近重复、真实人类标签/随机化实验、签名审计、可靠目标 RM、目标 checkpoint/CUDA/QLoRA 和目标模型 DPO/PPO
- 🟡 持续学习：DAPT、replay、正则/蒸馏、模型合并、编辑与 unlearning 协议完整；已有显式 task-id 的两任务 CPU no/64-example reservoir/full replay、20-seed paired interval 与样本呈现成本，目标 LLM、真实时间序列、task/data 重采样、compute-matched 与安全 retention 实证待补
- ✅ 合成数据：provenance、拒绝采样、自训练、hard/soft 蒸馏、多代反馈、真实锚点、去重、混合暴露与离线审计

## 3. 系统与推理

- ✅ 训练并行：数据、张量、流水线、序列、专家并行、ZeRO/FSDP、通信模型与正确性验收
- ✅ 数值与内存：状态/激活账本、FP32/TF32/FP16/BF16/FP8、混合精度、激活检查点
- ✅ 推理：prefill/decode、KV Cache、连续批处理、PagedAttention；离散 scheduler oracle 已验证 FCFS admission、sequence/token cap、chunked prefill、prefill 首 token、decode/completion、queue/TTFT/TPOT step 和 `P+O-1` logical work；metadata allocator 已验证固定 block、prefix fork/refcount、partial-tail COW、物理碎片、释放复用与 no-capacity 原子失败；集成 oracle 另验证 block-pressure preemption、FCFS re-admission、context rebuild、无重复 emission，以及 logical/recomputed/executed=`9/2/11`。这些 CPU policy 都未执行真实模型/KV tensor、swap/prefix reuse、vLLM scheduler、GPU page table 或性能测量
- ✅ 压缩：已有 symmetric group-wise code/scale/error、dense bit packing/单矩阵 artifact、带 identity/manifest 的多矩阵 bundle，以及含 Byte-BPE merges、config、全部量化/FP32 参数、tied contract 并能恢复 forward 的 repo-native MiniGPT inference checkpoint；另有 KV INT8 GQA/incremental oracle、剪枝/蒸馏和 speculative `min(1,p/q)`/positive-residual/block oracle。MiniGPT checkpoint 的完整性只对本仓库 tiny architecture revision 成立，forward code 仍由 trusted loader 提供，也没有 normalizer/special/chat template、训练状态、shard/runtime layout 或通用格式兼容；尚未完成目标 Llama/Qwen checkpoint、low-bit/KV/speculative GPU kernel、GPTQ/AWQ 或目标模型质量/性能实验
- ✅ 服务：吞吐/延迟、TTFT/TPOT、调度、缓存、弹性、可观测性
- ✅ 硬件：容量账本、GPU/TPU/NPU/CPU、Roofline、带宽、kernel、互联与测量协议
- 🟡 边缘部署：CPU/移动 NPU/WebGPU、更新、内存、功耗和热约束协议完整；目标设备实测待补

## 4. 应用工程

- ✅ Prompt：任务契约、真实 chat template、few-shot/分解、结构化输出、版本化、评测与安全边界
- ✅ RAG：摄取、切分、Embedding、向量/关键词检索、重排、目标-tokenizer packing、生成、引用，以及 packing→output→evaluation trace identity gate；另有 canonical-first LangChain/LlamaIndex parity 与 persistent extractive ASGI service，但两者生成仍是 deterministic extractive non-LLM baseline
- ✅ Agent：typed planner loop、strict model-text→proposal boundary、同源版本化 Draft 2020-12 Planner/runtime schema、request/response/decision identity、verifier 完成判定、step/token/cost/time 预算、循环/错误停止、工具调用、状态、记忆、权限、审批暂停与工作流；recorded-response control 不是真实模型实测，schema 也不是授权/业务验证。另有 local SQLite transactional outbox 的原子 enqueue、lease、崩溃重投、dead letter 与模拟 provider 幂等实验，准确边界是 at-least-once delivery，不是远端 exactly-once
- 🟡 Agent 互操作：已覆盖 provider API、MCP 与 A2A 的分层，tools/resources/prompts、Agent Card、task/message/artifact、版本协商、信任与 verifier 边界；尚无真实 MCP/A2A client/server、SDK conformance、网络认证或跨厂商互操作实验
- ✅ 代码模型：补全/FIM、仓库检索、补丁、执行反馈、沙箱、测试、pass@k 与成本边界
- ✅ 对话系统：typed state、来源/TTL/consent、修正/撤回、摘要边界、上下文压缩与租户隔离 reference core
- ✅ LLMOps：artifact graph、canonical identity、trace、离线门禁、发布、A/B、漂移与成本管理
- 🟡 产品设计：状态机、不确定性、证据、确认、用户控制、可访问性和实验协议完整；真实 UI/用户研究待补

## 5. 评测、安全与治理

- ✅ 评测设计：能力、质量、事实性、鲁棒性、安全、效率、业务价值
- ✅ 数据污染、基准饱和、提示敏感性和统计显著性；已有 paired/cluster exact-or-Monte-Carlo percentile bootstrap、case/cluster-joint sign-flip、case/equal-cluster estimand 与 Holm FWER oracle，并明确 coverage、exchangeability、cluster/interference、prespecified family/selection 和 effect-size 边界
- ✅ 评测 artifact：严格 run/comparison schema，完整本地 evidence graph 可重算 answer/case identity、scores、bootstrap 与 gate；另有 deterministic/XSS-safe/script-free HTML 派生报告、HMAC key-rotation chain、可选 artifact rehash 与外部 trusted-head rollback 检测；KMS/HSM、可信时间、透明日志和真实执行 provenance 待生产集成
- ✅ 人工评价、自动指标、LLM-as-judge、离线/在线实验
- ✅ 安全：威胁模型、滥用、注入、RAG ACL、工具副作用、SSRF、泄露、供应链、红队与事件响应
- 🟡 公平、偏见、隐私、版权、透明度、问责与审计
- 🟡 法规与标准：风险分级、影响评估、数据/模型/系统卡、第三方、发布、事件与退役模板完整；正式地区/用途审查和组织签署待补
- 🟡 环境与社会影响：能源、水、劳工、集中化、信息生态

## 6. 多模态与前沿

- ✅ 视觉语言：patch/动态分辨率、投影/重采样/cross-attention、OCR/图表、grounding 与评测
- ✅ 音频/语音、视频与文档：ASR/TTS、streaming、时空采样、定位指标、数据与安全边界
- 🟡 长上下文：位置外推、稀疏/线性注意力、RAG/memory、KV 与评测协议完整；目标长上下文实测待补
- 🟡 MoE：路由、负载均衡、capacity、all-to-all 与口径完整；已有 deterministic top-k、score-priority capacity、assignment/token drop、重归一化和 sparse linear expert CPU oracle，训练 router/MLP、具体 checkpoint 与 expert-parallel GPU 实跑待补
- 🟡 推理增强：过程/结果监督、verifier、搜索和 test-time scaling 协议完整；PRM/online policy 实跑待补
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
