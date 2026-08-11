# 工程项目索引

项目代码位于仓库根目录 projects/。教材解释原理，项目负责把原理变成可运行、可测量、可恢复的系统。

| 项目目录 | 当前能力 | 可运行入口 | 当前等级与证据 |
|---|---|---|---|
| rag-foundations | 结构切分/增量更新、SQLite 事务持久 store 与可验证灾备、BM25/dense/RRF、authorization-first rerank、tenant+principal ACL、目标-tokenizer packing、exact-span answer/abstain、引用/answer/trace gate、persistent extractive ASGI API | `about-llm-rag` CLI；`serve_extractive.py` 提供 body 外 identity、closed schema、readiness、request id、bounded concurrency/queue/deadline；`rag_service_control.py` 真实执行 FastAPI/Starlette/HTTPX ASGI + SQLite reopen；另有故障/篡改/超时测试 | L2+ 服务边界：透明 CPU/单机 SQLite + lexical/extractive API/CLI 证据；504 后后台 thread 继续占 permit。无 production auth/cache/真实 TCP/TLS/全局 admission/learned retrieval/LLM/SLO，exact substring 也不证明语义、来源、完整性或生成质量，因此尚不声称完整 L3 |
| rag-framework-adapters | Canonical-first LangChain/LlamaIndex 对象转换、ACL-bound Retriever adapter、严格 round-trip、Prompt/answer identity parity | `demo.py`；`parity_control.py` 真实调用 `BaseRetriever.invoke()`/`retrieve()`，比较 engineering/anonymous 可见集、Recall@4/nDCG@4 与 extractive artifact | L2：真实框架 core API + authored BM25/ACL/Prompt/non-LLM extractive control；不执行 learned embedding/vector index/reranker、provider/local LLM、框架默认 ACL/query engine、网络或生产性能，因此不是目标模型完整框架 RAG |
| safe-agent | 默认拒绝 policy、可信 subject/resource、cache 重新授权、proposal/execution identity、typed approval、严格结果快照、SQLite reconciliation、verifier-driven loop、strict JSON model-text planner、同源 Draft 2020-12 Planner/runtime schema、JSON checkpoint/resume、trajectory/effect gate、transactional outbox | `python -m about_llm.agents.cli run/loop/pause-loop/resume-loop/pending/resolve/inspect/evaluate`、`model_planner_control.py` 与 `outbox_demo.py`；含 request/response/decision/schema identity、closed JSON、完成校验、循环/错误停止、审批恢复、崩溃重投、lease/stale ack、dead letter 与 recorded trace | L2：可运行 scripted/recorded-response planner、标准 schema validator、exact verifier、counter/usage-preserving restart、exact-capability reference、跨进程 ledger 与 local SQLite at-least-once outbox；schema 不 coercion/default/authorize，recorded/supplied usage、unsigned approval、离线 resolver/receipt/effect、无密钥 hash 和模拟 provider 不等于真实模型/API/计费、签名审批、开放语义判断、集中 IAM、broker/provider 或 exactly-once external effect |
| single-gpu-finetuning | 严格 SFT、pairwise preference 与 raw judgment JSONL；exact/group/lexical/governance gate；deterministic MinHash/LSH + exact recheck/recall audit；两类 train/combined 有序绑定与 held-out-free readiness；linear/tiny Transformer RM control 与目标 RM LoRA/QLoRA；masked GAE/PPO、两状态/integer-token/text/learned-RM 对照；judgment agreement/κ/position；SFT mask、RM/DPO preflight、LoRA/TRL/QLoRA/DPO；continual-learning replay；MiniGPT strict training checkpoint/exact resume；PEFT adapter/base/merged/tokenizer strict export | `about-llm-sft-data audit/near-audit/governance-audit/prepare-training`、preference CLI、`minhash_lsh_toy.py`、RM/PPO/TRL smoke、`continual_replay_toy.py --benchmark`、`minigpt_resume_toy.py`、`smoke_peft.py --artifact-root ...` 与训练入口 | L2：CPU SFT/preference gate、MinHash/RM/PPO/TRL/replay、FP32 MiniGPT bit-exact split-run，以及 tiny GPT-2 adapter/safe-merge/tokenizer 磁盘 round trip。PEFT strict manifest 覆盖 13-file 完整目录，并在 load 前校验 safetensors 可解析性、base/merged config 与 tensor signature、LoRA A/B target coverage 及 file-set/hash/tokenizer 漂移；结构一致不证明权重数值正确，且 PEFT 自身不自动强制，无训练状态、量化 merge、目标 checkpoint/CUDA/runtime/质量/来源或原子发布证据；其余 MinHash、learned-RM 与治理边界见项目 README |
| transformers-basics | 确定性 raw-byte BPE、attention、MoE top-k/capacity/sparse-linear combine、离线 tiny GPT、residual activation patching、strict config/KV contract、generation-protocol 三方对账、Transformers generate stop control、checkpoint config/tokenizer/generation 检查 | `train_byte_bpe.py`、`moe_routing.py`、`smoke_tiny.py`、`activation_patching.py`、`inspect_config.py`、`inspect_generation_protocol.py`、`generation_runtime_control.py`、`inspect_checkpoint.py` | L2：CPU BPE/attention/MoE routing、Transformers smoke、真实 forward-hook、authored standard-GQA/MoE-GQA/MLA 与 special-token aligned/drift fixture；另真实执行 GenerationMixin 的 forced-token EOS-set/call-override/length-cap。配置对账不裁决 EOS superset/PAD=EOS；runtime control 的权重不决定 token且 finish reason 为推断。所有 fixture 均非发布模型，不证明来源、目标停止行为、质量、许可、vLLM/provider 或 GPU 性能 |
| jax-minigpt | core JAX 前向、Optax、梯度裁剪、JIT train step 与 tiny-batch overfit | `python projects/jax-minigpt/train_tiny.py` | L2：当前 CPU JAX device 实测与单测；GPU/TPU/sharding 未验证 |
| cloud-api-contracts | 三类严格 text-only 请求/响应/stream；bounded retry；async JSON/SSE HTTP executor；request-bound token/估算费用 reservation；SQLite durable quota；单-attempt HTTP reconciliation | verify/retry-matrix、execute_json_request/execute_sse_request、SSEDecoder/cloud_stream、`usage_budget_toy.py`、`sqlite_usage_budget_demo.py`、`budgeted_http_demo.py`；cap/model/fingerprint、reserve/settle/cancel/uncertain、重开/event audit | L2：MockTransport/AsyncByteStream、内存/SQLite ledger、target-preflight、明确未发送 cancel、HTTP/usage/cancel uncertain 与成功 settle；wrapper 强制每次 replay 独立 reservation。hash 不认证 caller/transport，SQLite 不与 provider 原子且不是分布式 quota，不执行真实 DNS/TLS/TCP，不证明 provider usage/发票/取消或 exactly-once billing，真实 smoke、provider error usage 与非文本 block 待补 |
| inference-serving | SSE/finite schedule/attempt/SLO、sampling/beam/finite-language constraint、continuous batching、group-wise/KV quantization、strict 多矩阵 bundle、repo-native MiniGPT inference checkpoint、speculative rejection sampling、Paged KV allocator、vLLM 路线 | `benchmark_openai.py`、`about_llm-inference-analyze`、decoding/scheduler toys、`quantization_toy.py`、`quantized_bundle_toy.py`、`minigpt_checkpoint_toy.py`、`kv_quantization_toy.py`、`speculative_decoding_toy.py`、`kv_block_allocator_toy.py` | L2：CPU 协议/统计、deterministic decoding/scheduler、单/多矩阵 artifact，以及含 Byte-BPE/config/全部参数/tied forward 的 tiny checkpoint；完整性只对 trusted repo MiniGPT loader 成立，不含训练状态或通用 runtime layout。Synthetic/authored reference 不证明目标 checkpoint/质量、runtime/provider 默认、vLLM scheduler、low-bit/speculative/PagedAttention kernel、GPU 服务、VRAM、加速或容量 |
| evaluation-gate | 严格 scorer/run manifest/comparison v2、完整本地证据图复算、script-free HTML report、HMAC release ledger、指标/校准/切片、paired/cluster bootstrap、sign-flip、Holm FWER、门禁 | `about-llm-eval score/compare/verify-comparison/verify-evidence/render-comparison-html/calibrate`；统计与 authenticated-ledger toys | L2：可运行 answer/case rehash、score/statistics/gate 重算、XSS-safe artifact-only 报告、HMAC key rotation、artifact rehash 与 trusted-head truncation control；本地复算/共享密钥不证明模型真实执行、custody、sampling/指标有效性、目录原子发布或线上影响，KMS/透明日志/真实 shadow/canary 待补 |
| synthetic-data-audit | Synthetic lineage、required verifier、self-verification、exact duplicate、generation round 与 mixture repetition | `python -m about_llm.synthetic_data_cli`，含候选与 mixture fixture | L2：可运行 CPU/offline artifact audit；真实 teacher/student、人工校准和多代训练待补 |

补充：`inference-serving` 另有 `stop_matching_toy.py`、`prefix_cache_toy.py` 与 `kv_preemption_batching_toy.py`。它们分别固定跨 UTF-8 byte/event 的 partial-stop/overlap 语义，完整安全 identity/collision-safe longest prefix/lease/LRU，以及 KV capacity/preempt/rebuild 的 logical/executed work 分账。连同表中的 sampling/beam/constraint，它们都是 authored/metadata CPU oracle，不证明目标模型生成质量、runtime/provider 默认等价、完整 grammar/schema、服务端取消/计费、真实 K/V、timing-channel mitigation 或 GPU 性能。

该项目的 continuous-batching oracle 执行固定 FCFS/decode-first 离散 policy，证明 `P+O-1` logical work 与 queue/first-token/completion 账本；KV-aware 扩展进一步执行一次 metadata block 抢占与 2-position rebuild，固定 logical/recomputed/executed=`9/2/11`，但仍不证明 vLLM、真实 K/V、VRAM 或 wall-clock performance。group-wise weight quantization 已加入 2–8 bit dense packing、strict 单矩阵 artifact 与 exact reload；多矩阵 bundle 验证 name/order/offset/digest 和 identity，但仍不是模型。新增 MiniGPT checkpoint 则保存 Byte-BPE merge payload、config、全部唯一量化/FP32 参数与 tied contract，并由固定 repo loader 恢复真实 causal forward。它不嵌入 forward source，也没有 normalizer/special/chat template、训练状态、通用 shard/runtime layout或低位 kernel。KV quantization 另有 INT8 GQA/incremental-prefix oracle。整体仍不证明目标 LLM checkpoint、GPU kernel、质量或性能，项目等级保持 L2。

这里的 L2 表示“可测试模块或可复现实验”，不等于 L3 工程样例或 L4 生产系统。目录中的 README 可以描述目标架构，但只有相应入口、配置、故障测试和可复现实验全部落地后，才提升等级。

`evaluation-gate` 另有 `clustered_bootstrap_toy.py`：枚举完整 cluster resample，并区分 case-weighted ratio 与 equal-cluster mean。`paired_randomization_toy.py` exact 枚举非零 paired differences 的 sign assignment；`clustered_randomization_toy.py` 对同 cluster 的 case 联合翻转；`holm_correction_toy.py` 固定 rank multiplier、running maximum 与 input-order remap。它们不建立真实 sampling/exchangeability/cluster/family 设计，也不证明 coverage、因果、effect importance 或模型质量。

`authenticated_release_ledger_toy.py` 则把两份 run manifest 与 comparison 的当前 bytes 纳入三条 HMAC chain，并在末条轮换 fixture key。它把 chain authentication、artifact rehash 和外部 trusted-head match 分开报告：缺 trusted head 时无法发现合法前缀截断。公开 key、caller timestamp 与 exclusive-create/file-fsync 分别不证明 key custody、真实时间、不可否认性或目录原子发布。

## 推荐顺序

1. 先运行 rag-foundations，理解召回与 ACL。
2. 用 rag-framework-adapters 比较抽象成本，不改变检索结果。
3. 用 evaluation-gate 固定 case 与基线。
4. 用 single-gpu-finetuning 比较 Prompt、RAG 与 LoRA。
5. 用 inference-serving 测真实服务质量和性能。
6. 最后让 safe-agent 调用经过验证的检索与工具。

## 项目完成标准

每个生产项目最终应有：

- 明确的输入输出、数据与权限契约；
- 非 LLM 或最简单可行基线；
- 质量、安全、延迟和成本指标；
- 单元、集成、对抗和故障恢复测试；
- 可复现配置、版本和运行命令；
- 可观测 trace、发布门禁与回滚；
- 已知限制和不适用范围。

只有 README 和架构图不算项目完成；只有能运行但没有评测也不算。
