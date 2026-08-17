# 工程项目索引

项目代码位于仓库根目录 projects/。教材解释原理，项目负责把原理变成可运行、可测量、可恢复的系统。

**项目导航**：[实验顺序与交付物](labs.md) · [仓库实现契约](../guide/repo-map.md) · [环境配置](../guide/environment.md) · [生产检查表](production-checklist.md) · [内容准确性台账](../reference/accuracy.md)
{ .doc-nav }

| 项目 | 学习主线 | 当前证据 | 详情 |
|---|---|---|---|
| [RAG Foundations](projects/rag-foundations.md) | 版本化摄取/备份、ACL、检索/重排、packing/trace、ASGI、Qwen failure/replay/guard | L2+：CPU/SQLite/ASGI + 固定 Qwen 三层 controls；远端向量库/GPU 待实测 | [运行与验收](projects/rag-foundations.md#run) |
| [RAG Framework Adapters](projects/rag-framework-adapters.md) | canonical ACL/rank→两框架 Retriever/Prompt→round-trip/artifact parity | L2：真实 LangChain/LlamaIndex core API + 16 个字段/安全/漂移测试；native index、LLM 与性能未执行 | [运行与验收](projects/rag-framework-adapters.md#run) |
| [Safe Agent](projects/safe-agent.md) | proposal/policy/approval/verifier、LangChain/LlamaIndex tool/Agent-loop controls、pending/resume、outbox、MCP/A2A | L2：真实 framework tool API/控制流 + scripted model + 离线 planner/SQLite/outbox + official/authored loopback controls；真实模型、生产 IAM/副作用待实测 | [运行与验收](projects/safe-agent.md#run) |
| [Single-GPU Finetuning](projects/single-gpu-finetuning.md) | train-only readiness→mask/final labels→SFT/QLoRA/DPO→adapter→held-out gate | L2+：零下载 preflight、tiny/CPU/Gloo/跨 PID controls + 固定 Qwen SFT labels/forward、LoRA/DPO 单步；CUDA/QLoRA 峰值与业务质量待目标环境实测 | [运行与验收](projects/single-gpu-finetuning.md#run) |
| [Transformers Basics](projects/transformers-basics.md) | BPE→attention/online softmax→generation/config→固定 Qwen forward/单矩阵 INT4→activation patching→六级 MoE routing/training | L2：NumPy/PyTorch CPU controls + immutable release evidence + 固定 Qwen 真实权重/activation；完整 low-bit checkpoint、CUDA/NCCL/生产性能未执行 | [运行与验收](projects/transformers-basics.md#run) |
| [JAX MiniGPT](projects/jax-minigpt.md) | 纯函数 PyTree/Optax/JIT→SGD/AdamW parity→strict resume | L2：632 参数 CPU overfit + PyTorch/JAX 全梯度/三步 optimizer 对账 + 两进程 bit-exact control；accelerator/sharding 未执行 | [运行与验收](projects/jax-minigpt.md#run) |
| [Cloud API Contracts](projects/cloud-api-contracts.md) | 三类 adapter、Responses typed-event replay、HTTP/SSE、重试、逐 attempt 预算、reasoning/trajectory gate | L2：authored events/AES-GCM/MockTransport/SQLite controls；OpenAI SDK、真实 provider/计费待实测 | [运行与验收](projects/cloud-api-contracts.md#run) |
| [Inference Serving](projects/inference-serving.md) | 解码/选择、调度/KV/量化、HTTP/取消、vLLM runbook、压测与 SLO | L2+：精确 CPU oracle/统计 + 固定 Qwen/async/tiny-Transformers controls；GPU 仍待目标环境实测 | [运行与验收](projects/inference-serving.md#run) |
| [Evaluation Gate](projects/evaluation-gate.md) | target-Qwen behavior、strict JSON schema/value、citation evidence-span→score、comparison v2、完整本地复算、统计控制与发布账本 | L2+：固定 Qwen 七次真实 generate + 两组五例 structured/span metric control + 可复算证据图/ledger；非代表性、非 held-out、无性能/entailment 结论 | [运行与验收](projects/evaluation-gate.md#run) |
| [Synthetic Data Audit](projects/synthetic-data-audit.md) | strict JSON→lineage graph→verifier gate→exact identity→target exposure→v2 full recomputation | L2：CPU/offline、输入/policy-bound artifact + 40 tests；teacher/verifier、训练、observed ledger 与质量未执行 | [运行与验收](projects/synthetic-data-audit.md#run) |

这里的 L2 表示“可测试模块或可复现实验”，不等于 L3 工程样例或 L4 生产系统。每个详情页给出复制即运行的最小命令、定向测试和当前结论边界；完整能力矩阵保留在对应项目 README 中。

JAX MiniGPT 的站点入口和根项目 README 现按四层证据组织：原生纯函数 JAX tiny overfit、同解析权重 LayerNorm/plain-SGD parity、shared-mask 三步 AdamW trajectory、`ALLMJAX1` 跨进程 strict resume；README 另补运行矩阵、故障树、生产扩展、验收与求职边界，并纳入独立防退化 gate。`train_tiny.py` 实际只报告 loss/gradient/timing，不生成文本；`block_until_ready()` 后的 CPU 计时也不是 accelerator benchmark。

JAX MiniGPT 又新增 PyTorch↔JAX 同权重 control：显式对齐 affine LayerNorm、epsilon、GELU、mask、tied embedding、masked loss 与 plain SGD 后，初始 logits、20 个 unique parameter gradients、一步参数及 post-step forward 都在 `2e-6` 内。原生 JAX RMSNorm 反事实 logits 最大差 `0.37747739627957344`，因此默认两个 MiniGPT 不被误写成等价。该 L2 证据不比较 AdamW state、PRNG/JIT、GPU/TPU、sharding、收敛或性能。

JAX MiniGPT 的三步 AdamW parity control 再用三张共享 materialized embedding masks，对账真实 clipping、first/second moments、count、schedule、参数与 post-step forward，最大参数差 `2.5480985641479492e-06`；wrong-mask 反例为 `0.06900620367377996`。它没有证明 native RNG/state advance、norm/bias decay mask、JIT 或 accelerator。

JAX MiniGPT 的 strict resume control 进一步把 params、Optax state、typed PRNG、shuffle permutation/cursor 与 step 放进 **13,476-byte artifact**，由两个独立 spawn process 在 step 3 交接；六步 trace 和最终 full state 与 uninterrupted bit-exact，wrong-PRNG/wrong-cursor 负例则产生可测漂移。它仍只是 authored CPU 单设备 control，不证明 Orbax/TensorStore、directory durability、来源认证、accelerator/sharding、目标模型、收敛或性能。

Transformers Basics 详情页已按四层证据重构为独立教程：先用 byte-BPE、dense/online/cache attention 锁定机制，再用随机 tiny Transformers 验证训练与 generation API，随后把 immutable model-card/config 投影和固定 Qwen 真实权重执行严格分账，最后以 activation patching 正负对照和六级 MoE 矩阵讲清从 routing 到 collective/backward 的递进关系。页面同时给出故障定位、面试问题与作品集证据包；这增强的是可学习性和可审计性，不新增 CUDA/NCCL、目标 MoE checkpoint、质量或性能证据。

Transformers Basics 的前两条 MoE 证据是单进程 controls：NumPy oracle 固定 padding-aware top-k/capacity/drop/combine；PyTorch CPU Float64 control 在同一训练图真正执行 top-2 router/三组 MLP experts 与 score-priority capacity/drop，对齐 sparse—dense forward/backward，并以 detached gate、collapsed balance step、两种 post-drop policy、全丢 token、padding exclusion 与 CPU-local routing groups 拆出梯度和 overflow 语义。v3 的独立 fixture 又执行 authored deterministic full-ranking reroute、token 内 duplicate-expert avoidance 与 dropless nominal-capacity-excess policy，reroute/dropless sparse—dense forward/materialized-zero backward 均对齐。这些单进程 fixtures 不执行 all-to-all；int64 group IDs 也不是 distributed collective，authored policy 不是目标框架默认。整个项目仍没有目标 MoE checkpoint、shared/fine-grained experts、目标级 distributed reroute/dropless、GPU training、收敛、专门化、质量或性能证据。

第三条 distributed capacity fixture 启动两个真实 CPU/Gloo workers：hidden `all_gather` 建立 4-token replicated global routing batch，两个 count `all_reduce` 得到 active=4/selected=`[4,0]`。Rank-local 独立 capacity 合计 kept=2；global competition 只 kept 全局最高分的 1 个，mask `[F,F,T,F]`、drop=3，rank-0 output 反事实差 `0.9640275800758169`。它证明 same-host collective capacity-group 边界，但没有 expert ownership/token `all_to_all`、distributed backward、CUDA/NCCL、多节点或性能证据，不能借给前两条升级为 distributed training/目标模型声明。

第四条 all-to-all fixture 才把 expert 0/1 分别只放在 rank 0/1，并以 source→owner `[[1,2],[1,0]]` 的 variable splits 发送 token/gate/metadata、执行 owner forward 再返回 source。Rank 0 arrival global IDs 为 `[1,0,2]`；metadata scatter 后与单进程 oracle一致，metadata-free 错序差 `0.8958737432590591`。其 416-byte logical tensor账本不等于 wire bytes；不证明 CUDA/NCCL 或生产性能，也没有 capacity、backward或目标模型证据，不能与前三条拼成完整 EP。

第五条 all-to-all training fixture 用 authored reverse-split autograd把 hidden/gate gradients 返回 source，对 replicated router gradient做 SUM all-reduce，owner expert直接持有本 expert global-mean gradient；一步 SGD 的 gradients、参数和 post-step forward 与单进程 oracle exact，global mean loss `20.78017329703821→19.41091750734501`。它没有 capacity、DDP、optimizer state、CUDA/NCCL 或目标 checkpoint，不借用为 CUDA/NCCL 或目标模型训练证据，也不证明收敛或性能。

第六条 capacity-aware all-to-all training fixture 在独立两进程图中加入 global score-priority drop：selected `[1,0,1,0]`、capacity=1、mask `[F,T,T,F]`，kept-only splits `[[1,1],[0,0]]` 覆盖 zero-assignment source backward。Dropped outputs/hidden task gradients为 0；其余 gradients、一步参数和 post-step forward与单进程 capacity oracle一致，global MSE `15.253670387373656→14.530264380025987`。它不执行 reroute/dropless、DDP、CUDA/NCCL、目标 checkpoint 或性能测量，也不证明收敛与质量。

MCP 与 A2A 当前都是 Safe Agent 的 L2 局部 control。MCP official-SDK memory 固定 `mcp==1.29.0` 但没有 transport；official-SDK stdio 与 Streamable HTTP 又分别让官方 client/server 在同一次运行中经独立 subprocess/OS pipe 或真实 loopback TCP/HTTP 完成 lifecycle/schema/handler gate。它们没有借到自写 controls 的畸形 framing、header/session/cancel negative matrix，也没有 conformance、OAuth/TLS、远程或业务授权证据；官方 HTTP 的私有 shutdown token 不是 MCP auth。两个 authored controls 分别覆盖 strict stdio 子集与 IPv4 loopback Streamable HTTP JSON/SSE/session/cancel，但不继承官方 SDK 身份。五个 MCP controls 都没有远程、跨厂商或生产身份。A2A 以官方 Python SDK 1.1.2 真实执行 IPv4 loopback TCP/HTTP、Agent Card、JSON-RPC `SendMessage`/`GetTask`、错误分层与可选冻结官方 Schema 校验，但没有 TCK、SSE、REST/gRPC、TLS、认证、签名 card、远程或跨厂商互操作。不同 control 不自动建立业务授权。

Safe Agent 的 framework adapter control 另把 LangChain `StructuredTool` 与 LlamaIndex `FunctionTool` 放在 proposal transport 层：同一 strict Pydantic model、授权/跨 tenant/重放/unknown-tool cases 最终进入 canonical runtime。当前 LlamaIndex direct `FunctionTool.call()` 对 `key=7` 不先执行 `fn_schema` validation，而 canonical Draft 2020-12 gate 会在 resolver 前拒绝；这条负例说明 schema disclosure 不等于 effect authorization。它没有执行 LangGraph/LlamaIndex Agent loop、模型、网络或真实副作用，也不证明框架默认安全。

独立的 framework Agent-loop control 则真实执行 LangChain `create_agent()`/LangGraph 与 LlamaIndex `FunctionAgent.run()` 的 model→tool→model；authorized/replay 由本地 canonical receipt 验收，cross-tenant/unknown-tool 即使模型文本声称成功也被独立 verifier 拒绝。LangChain 用 injected tool-call ID，当前 LlamaIndex 路径从可信 fixture action 派生 canonical ID，并捕获每 case 73 次 Pydantic deprecation。它仍只使用 scripted in-process model，没有 provider、网络、remote effect、persistent resume、streaming/parallel/cancel、性能或质量证据。

Transformers Basics 的 target-checkpoint control 已重哈希并加载固定 Qwen2.5-0.5B-Instruct revision，在 CPU FP32 下执行单个 prefill/cache/full/generate case；同一 snapshot 另有单事实 France/Germany source-position activation-patching control，真实执行 layer 0/11/23 与 full-prefix/readout/future 结构对照。它们比 config-only release evidence 多了真实权重与框架路径，但仍只是 L2 局部运行证据：单 pair/整 residual 高恢复不是唯一自然 circuit、质量基准、有效上下文、许可审查、GPU/vLLM、峰值内存或生产性能证明。

同一 snapshot 的 selected-weight INT4 control 又真实量化第一层 `[896,896]` `o_proj.weight`：802,816 参数的 strict bundle 为 427,328 bytes，相对该矩阵 FP32 为 7.514752×；真实 activation output/末位 logits relative-L2 分别为 0.070002/0.085138，尽管单提示 argmax 仍为 17。它只覆盖全模型 0.1625% 参数并以反量化 FP32 forward，不能写成完整 Qwen low-bit checkpoint、RAM/VRAM/速度缩减、量化无损或 GPU kernel 证据。

Synthetic Data Audit 的 v2 control 将 1,457-byte records、341-byte mixture、required verifier、known external parent 与 fingerprint profile 绑定进 canonical artifact，并从 caller-supplied 输入完整本地复算；固定报告为 4 candidates / 2 eligible / 1 eligible unique、25% synthetic target / 5.0 expected repetition。它还分开报告 cycle、nonmonotonic round、unresolved parent、missing/failed verifier 与 revision overlap，并用 input drift、cooperative rehash 和 no-overwrite 反例锁定边界。无密钥 hash 不认证来源，file `fsync` 不证明目录持久化；当前没有执行 teacher/verifier model、训练、observed exposure、许可隐私审查、语义去重、质量、collapse 或收益评测。

Inference Serving 先用 binary latent-regime oracle 展示相同单样本 0.6 不决定多数票：N=11 时 independent/latent-correlated 分别为 0.75349813248/0.53896454244；它只处理两个 canonical labels、regime 内 conditional i.i.d.，不处理开放文本 plurality。另一个 authored finite distribution 的闭式 `Fraction` oracle 分开 oracle@N、verifier-selected@N 和期望 proxy score；N=1/4/16 显示 oracle 与 proxy 上升时 selected success 可从 0.5936 降到 0.1852867601。两者均没有执行 model/tokenizer/dataset/judge/PRM/GPU/provider，logical N 也不代表延迟、费用或并行度。

同项目复用 selected snapshot，把权重放进只监听 IPv4 loopback 的 Transformers reference subprocess，以随机 Bearer 真实调用 models/chat，并让 non-stream/SSE 各触发一次 `generate()`；但它先完整生成再发 SSE。第二条无模型 async control 验证 content-before-completion 与 ASGI/backend cancellation。第三条随机 tiny GPT-2 control 真实执行一次 forward/`GenerationMixin.generate()` thread，并以人为 streamer pause、event 与 `StoppingCriteria` 验证退出/join。三者分别补目标权重 HTTP 集成、async 传播和显式 cooperative thread 三层证据；合起来仍不是未修改/目标模型增量取消、vLLM/CUDA、KV/CPU/GPU release、完整 OpenAI compatibility、TLS/IAM、多 worker、provider billing、性能或质量证据。

Single-GPU Finetuning 的站点入口现在先给出一条完整交付主线：审计身份生成不含 held-out 原文的 readiness，训练身份做零下载 identity preflight，再进入目标 template/mask/final-label 审计、tiny PEFT 发布 control、LoRA/QLoRA 或 DPO、独立 held-out gate。容量公式只用于筛配置；不同 CPU/Gloo/tiny/recorded control 不拼接成 CUDA/QLoRA、目标 GPU 峰值或业务质量证据。

Single-GPU Finetuning 先复用同一 selected snapshot 对账 tool-aware SFT final labels：原生 Qwen 模板在三条多轮/并行 tool fixture 上返回全零 assistant mask，审核模板保持 47 / 301 / 200 个 input IDs 一致并在 Arrow 前生成 masks；真实 TRL collator 得到 `[3,301]`、90 个监督 labels，目标权重 no-grad loss 为 `1.251716`。它没有 backward/optimizer，只证明固定 Qwen schema，不证明数据合法性、任意 provider schema、multimodal、收敛或质量。

同项目再在目标 Qwen 上执行 41-token prompt/3-token supervision、270,336-parameter `q_proj/v_proj` LoRA backward 与一次 AdamW step；冻结基座 fingerprint 不变，48 个 B tensors 均非零，1.09 MB adapter 在新基座上 bit-exact reload。该步 loss 从约 0.003864 升至 0.584557，所以 L2+ 表示目标框架/权重链路证据，不是训练质量、代表性数据、QLoRA/CUDA、显存或性能证据。

同项目另用两条 authored preference pair 在固定 Qwen 上执行一次真实 TRL DPO step：loss `0.693147→0.333352`、两条 relative margin 为正、96 个梯度张量 finite，冻结 parameter/state/config 指纹 exact。`0.547077` reference replay drift 被单列且不等同于权重漂移；它不证明人类偏好、held-out 质量、收敛、安全或生产对齐。

同项目的 controls 与目标 Qwen controls 相互独立：单进程 toy 证明 masked-token sum/count；第一条双进程 control 证明 default DDP `D/N`；第二条执行 `no_sync`、clip 与 SGD；第三条单进程 AMP 证明 `unscale→clip`、overflow skip 和 scaler omission；第四条双进程 DDP+AMP 验证共同 skip、post-reduction fault 分叉与 flag gate；第五条统一 CPU resume 保存 model/AdamW/StepLR/scaler/RNG/custom shuffle，由不同 PID bit-exact 恢复。第六条 DataLoader control 以两个 spawn workers 证明 consumed=3 时 sampler emitted=7、从 emitted 恢复漏 queue IDs、fresh worker RNG 不重放。第七条扩展为六进程 main-process stochastic mask、真实 backward、SGD momentum、StepLR 与 accumulation：committed=2/consumed=3 时，从 2 恢复 commit RNG并 replay bit-exact；从 3 用正确 crash RNG却漏 gradients/sample `1`，在相同 5 steps/LR 与终态 RNG 下仍漂移；从 3 加载绑定 base digest 的 pending/position/divisor/gradients/crash-RNG sidecar 也 bit-exact；完整 gradients/ledger/steps/LR 但使用错误 RNG 的负例再次漂移。sidecar 协议最后发布 canonical manifest，并以四种 incomplete/tamper snapshots 验证 completeness fail-closed；base-only 仍可走 replay。第七条仍没有 queue/worker/Python/NumPy/CUDA RNG、sample—optimizer—base+sidecar+manifest 原子事务、directory `fsync`/断电、来源认证/不可变快照、GradScaler、DistributedSampler 或目标 adapter。所有路径仍未覆盖目标 Qwen Trainer 的原生随机层/多 bucket/自然 overflow、distributed resume、CUDA、性能或质量；各条不能借用彼此升级证据等级。

RAG Foundations 复用同一固定 Qwen snapshot，进一步串起 ACL-before-ranking BM25、tokenizer-measured packing、逐步 greedy logits 与 `generate()`。attempt-1 的 answerable case 漏引、empty-context case 幻觉，行为 gate 为 0/2；这证明失败被忠实记录，不是宣称模型质量通过。publication policy 的首次回放只是反事实；后续不同 query 的 guarded control 才真实观察有证据 `GenerationMixin.generate` method=1、空授权证据=0，并把 audit/public projection 分开。API method count 不等于内部 forward/provider attempt 或计费；少量共享 authored corpus/checkpoint 的 case 与 unsigned report仍不证明语义蕴含、总体质量、来源认证或生产安全。

RAG Framework Adapters 将同一 authorization-first BM25 与 `SearchResult[]` 分别绑定到 LangChain `BaseRetriever.invoke()` 和 LlamaIndex `BaseRetriever.retrieve()`，逐项对账 ID、正文、保护 metadata、rank/score、Prompt SHA-256 和 deterministic extractive answer artifact。当前四文档 fixture 中 engineering/anonymous 分别看到 2/1 条授权证据，16 个测试覆盖保护键覆盖、rank/ID/finite-score、mutation、metadata exclusion 与安全上下文。根项目 README 已把运行、证据、负例、扩展和求职验收整理为 standalone 教程并纳入防退化 gate；这不提高运行证据等级。该 control 证明当前 core API 适配不漂移，不证明框架默认 ACL、native embedding/index/query engine、LLM、网络或生产性能。

## 如何阅读项目证据

项目脚本通常是 **control/oracle**：它用固定输入和透明实现锁定一个局部契约。control 通过，可以证明该契约在当前 fixture 与环境中成立；它不能自动证明真实模型质量、第三方服务行为、GPU 性能或生产安全。

评审一个项目时，把证据分成四层：

1. **机制证据**：手算值、确定性脚本或单元测试能否复现局部不变量。
2. **集成证据**：真实框架、数据库或 ASGI 边界是否被执行，而不是全部 mock 掉。
3. **运行证据**：目标模型、网络、硬件与并发负载下，质量和 SLO 是否达到门槛。
4. **治理证据**：身份、密钥、审批、发布链、事故恢复是否由真实控制面保证。

前两层可以在本仓库离线学习；后两层通常需要部署环境和组织流程。项目等级按最弱的关键证据判断，不能用大量单元测试补偿缺失的生产身份或容量证据。

## 三条最小验收路径

下面三条路径分别覆盖“数据与权限”“模型决策与副作用”“指标与发布判断”。命令均从仓库根目录执行。

### 路径 A：RAG 的权限、背压与超时

~~~powershell
python projects/rag-foundations/rag_service_control.py
python projects/rag-foundations/run_qwen_rag_control.py --local-files-only
python projects/rag-foundations/run_qwen_guarded_rag_control.py --local-files-only
python -m pytest tests/test_rag_service.py -q
~~~

运行前先预测 engineering 与 anonymous 各自可见的 source；运行后检查身份是否来自 body 外、未知字段是否 422、缺认证是否 401。并发测试中要区分三件事：请求收到 504、底层同步 thread 终止、并发 permit 释放。第一件发生不代表后两件已经发生。

**通过门槛**：未授权正文在 scorer/cache 前不可见；readiness 能发现 store 不可用；容量耗尽会在有界 queue deadline 后拒绝；超时工作在真正结束前仍占 permit。

**故意破坏**：把 ACL 移到 rerank 后，或在 timeout 时立即释放 semaphore。即使最终答案和 happy-path 测试仍可能通过，也应判定安全或容量契约失败。

**结论边界**：ASGITransport + SQLite control 不含 TCP/TLS、反向代理、JWT/IAM、多 worker 或多副本 admission；extractive exact span 也不证明生成质量。

### 路径 B：Agent 的 proposal、授权与恢复

~~~powershell
python projects/safe-agent/model_planner_control.py
python projects/safe-agent/mcp_sdk_memory_control.py
python projects/safe-agent/mcp_sdk_stdio_control.py
python projects/safe-agent/mcp_sdk_streamable_http_control.py
python projects/safe-agent/mcp_stdio_control.py
python projects/safe-agent/mcp_streamable_http_control.py
python -m pytest tests/test_model_planner.py tests/test_mcp_sdk_memory.py tests/test_mcp_sdk_stdio.py tests/test_mcp_sdk_streamable_http.py tests/test_mcp_stdio.py tests/test_mcp_streamable_http.py -q
~~~

沿 trace 分开标记 model response、decision、action proposal、approval、execution 与 verifier result。严格 JSON schema 只说明结构符合约束，不说明资源已授权、内容可信或副作用可以执行。模型看到的工具结果和网页文本都属于不可信 observation。

**通过门槛**：非法或漂移 schema 在 resolver 前拒绝；模型只能产生 proposal；不可逆动作需要绑定具体 subject/resource/action 的 approval；只有 verifier 通过才能完成；重复投递和进程恢复不会静默重复副作用。

**故意破坏**：复用旧 approval 批准新参数，或让远端 `completed` 直接转换为本地成功。前者破坏 approval identity，后者绕过本地 verifier；两种情况都必须 fail closed。

**结论边界**：recorded planner、离线 resolver 和 SQLite ledger 不证明真实 provider JSON 稳定性、集中 IAM、签名审批、外部 broker 或 exactly-once effect。MCP official-SDK stdio/HTTP 已分别同时执行 SDK 与本地 pipe/loopback TCP，但未继承 authored controls 的畸形 framing/body/header/cancel 负例或 conformance；authored stdio/HTTP 也只是本地固定子集。所有 MCP controls 都没有生产认证、授权、远程或跨厂商证据；A2A control 虽有官方 SDK 与 loopback HTTP，也没有 TCK、TLS、认证、远程或跨厂商证据。

### 路径 C：评测差异是否足以发布 { #acceptance-evaluation }

~~~powershell
python projects/evaluation-gate/paired_randomization_toy.py
python projects/evaluation-gate/clustered_bootstrap_toy.py
python projects/evaluation-gate/sequential_peeking_toy.py
python projects/evaluation-gate/run_qwen_target_behavior_evaluation.py --verify projects/evaluation-gate/target-qwen-behavior.recorded-report.json
python -m pytest tests/test_evaluation_statistics.py -q
~~~

先手算 paired difference，再比较 case-level 与 cluster-level 重采样。若同一用户贡献多个 case，它们通常相关；把它们当独立样本会低估不确定性。统计显著也只回答“在给定抽样假设下差异是否难以由随机性解释”，不回答差异是否足够大、指标是否有效或线上用户是否受益。

**通过门槛**：case identity 与 scorer 版本固定；比较使用配对样本；存在簇时按簇重采样或联合翻转；多指标门禁说明 family 并控制多重比较；反复查看时事前固定最大样本、look schedule 与 stopping rule；同时报告 effect size、区间、切片与失败样例。

**故意破坏**：删除所有零差异 case、反复尝试指标只保留显著结果、每 10 条样本按固定 0.05 偷看并显著即停，或在看到结果后改变非劣门槛。这些操作即使命令仍输出合法数字，也使发布结论失效。

**结论边界**：toy 的 exact enumeration 能验证统计实现，不验证真实数据的 exchangeability、样本代表性、judge 有效性、业务重要性或因果影响。

上述命令只走统计 oracle。完整的 `score → comparison v2 → verify-comparison → verify-evidence → HTML → HMAC ledger` 路径及故意破坏用例见 [Evaluation Gate 详情页](projects/evaluation-gate.md#run)；其中完整本地复算仍不重放模型，HTML 只是派生视图，HMAC 链也必须配合 artifact rehash 与 ledger 外 trusted head 才能检测引用 bytes 漂移和合法前缀截断。

Target-Qwen recorded control 则是另一条独立证据：固定同一 7-file/999,586,347-byte snapshot，以 CPU FP32 greedy 对 7 条 authored case 真实执行 `GenerationMixin.generate()`；英文算术输出 `112`，大小写复制输出 `llm-2026`，JSON 输出 `{"answer": 42}`，literal exact/normalized exact/token F1 汇总为 `4/7`、`5/7`、`6/7`。`--verify` 只重开 strict suite/report、重算指标/聚合并确认 reviewed fingerprints，不重放这七次生成。suite 未外部预注册、未独立留出、非代表性且无统计功效；没有 latency、系统对照、judge、人评或发布 gate，因此三种分数都不能写成 Qwen 总体准确率、性能或生产质量。

完成任一路径时，交付物不应只有终端截图。至少保存环境与命令、原始机器输出、一个故意失败的反例、通过/失败判断，以及不超过五行的结论边界。

`evaluation-gate` 另有 `clustered_bootstrap_toy.py`：枚举完整 cluster resample，并区分 case-weighted ratio 与 equal-cluster mean。`paired_randomization_toy.py` exact 枚举非零 paired differences 的 sign assignment；`clustered_randomization_toy.py` 对同 cluster 的 case 联合翻转；`holm_correction_toy.py` 固定 rank multiplier、running maximum 与 input-order remap；`sequential_peeking_toy.py` 用 exact dynamic program 将五次 naive 0.05 peeking 的总体假阳性算为约 0.1010，并与预设 Bonferroni split 的约 0.0152 对照。它们不建立真实 sampling/exchangeability/cluster/family/sequential 设计，也不证明 coverage、因果、effect importance 或模型质量。

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
