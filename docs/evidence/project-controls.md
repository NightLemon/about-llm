# 项目实验与证据台账

本页保存项目的精确版本、固定输入、运行结果和证据边界，供复核与维护使用。第一次学习请从
[工程项目索引](../practice/project-index.md)开始，不要把本页当作课程顺序。

**证据导航**：[工程项目索引](../practice/project-index.md) · [仓库实现契约](../guide/repo-map.md) · [生产检查表](../practice/production-checklist.md) · [内容准确性台账](../reference/accuracy.md)
{ .doc-nav }

| 项目 | 学习主线 | 当前证据 | 详情 |
|---|---|---|---|
| [Training Data Lineage](../training/data.md#training-data-lineage-run) | 帖子版本→解析与规范化→选择去重代表项→定位 shard token 区间→反查训练与 checkpoint→计算删除影响 | L2：CPU 离线关系图可以完整复算，12 个测试覆盖关键关系和输入错误；没有执行真实抓取、构建、删除或 unlearning | [运行与解释](../training/data.md#training-data-lineage-run) |
| [RAG Foundations](../practice/projects/rag-foundations.md) | 版本化摄取/备份、ACL、检索/重排、packing/trace、ASGI、Qwen failure/replay/guard | L2：CPU/SQLite/ASGI 与固定 Qwen 三层可复现 controls；远端向量库/GPU 待实测 | [运行与验收](../practice/projects/rag-foundations.md#run) |
| [RAG Framework Adapters](../practice/projects/rag-framework-adapters.md) | canonical ACL/rank→两框架 Retriever/Prompt→round-trip/artifact parity | L2：真实 LangChain/LlamaIndex core API + 16 个字段/安全/漂移测试；native index、LLM 与性能未执行 | [运行与验收](../practice/projects/rag-framework-adapters.md#run) |
| [Safe Agent](../practice/projects/safe-agent.md) | 一笔退款的九阶段生命周期、LangChain/LlamaIndex tool/Agent-loop controls、pending/resume、outbox、MCP/A2A | L2：closed schema/ACL/approval + SQLite recovery + 真实 framework 控制流 + scripted model + official/authored loopback；真实模型、生产 IAM/副作用待实测 | [退款生命周期](../practice/projects/safe-agent.md#refund-lifecycle) |
| [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md) | train-only readiness→mask/final labels→SFT/QLoRA/DPO→adapter→held-out gate | L2：零下载 preflight、tiny/CPU/Gloo/跨 PID controls，以及固定 Qwen SFT labels/forward、LoRA/DPO 单步；CUDA/QLoRA 峰值与业务质量待目标环境实测 | [运行与验收](../practice/projects/single-gpu-finetuning.md#run) |
| [Transformers Basics](../practice/projects/transformers-basics.md) | 教学 BPE/labels→固定 Qwen3 tokenizer/template→attention/online softmax→generation/config→固定 Qwen forward/单矩阵 INT4→activation patching→六级 MoE routing/training | L2：NumPy/PyTorch CPU controls + 固定 Qwen3 tokenizer + immutable release evidence + 固定 Qwen 真实权重/activation；完整 low-bit checkpoint、CUDA/NCCL/生产性能未执行 | [运行与验收](../practice/projects/transformers-basics.md#run) |
| [JAX MiniGPT](../practice/projects/jax-minigpt.md) | 纯函数 PyTree/Optax/JIT→SGD/AdamW parity→strict resume | L2：632 参数 CPU overfit + PyTorch/JAX 全梯度/三步 optimizer 对账 + 两进程 bit-exact control；accelerator/sharding 未执行 | [运行与验收](../practice/projects/jax-minigpt.md#run) |
| [Cloud API Contracts](../practice/projects/cloud-api-contracts.md) | 三类 adapter、Responses typed-event replay、HTTP/SSE、重试、逐 attempt 预算、reasoning/trajectory gate | L2：authored events/AES-GCM/MockTransport/SQLite controls；OpenAI SDK、真实 provider/计费待实测 | [运行与验收](../practice/projects/cloud-api-contracts.md#run) |
| [Inference Serving](../practice/projects/inference-serving.md) | 解码/选择、调度/KV/量化、HTTP/取消、vLLM runbook、压测与 SLO | L2：精确 CPU oracle/统计、固定 Qwen/async/tiny-Transformers controls；GPU 仍待目标环境实测 | [运行与验收](../practice/projects/inference-serving.md#run) |
| [Evaluation Gate](../practice/projects/evaluation-gate.md) | target-Qwen behavior、strict JSON schema/value、citation evidence-span→score、comparison v2、完整本地复算、统计控制与发布账本 | L2：固定 Qwen 七次真实 generate、两组五例 structured/span metric control 与可复算证据图/ledger；非代表性、非 held-out、无性能/entailment 结论 | [运行与验收](../practice/projects/evaluation-gate.md#run) |
| [Synthetic Data Audit](../practice/projects/synthetic-data-audit.md) | strict JSON→lineage graph→verifier gate→exact identity→target exposure→v2 full recomputation | L2：CPU/offline、输入/policy-bound artifact + 40 tests；teacher/verifier、训练、observed ledger 与质量未执行 | [运行与验收](../practice/projects/synthetic-data-audit.md#run) |

## 怎样解释项目等级

| 等级 | 已建立的证据 | 仍需补齐 |
|---|---|---|
| L0 | 机制、术语和边界已写清 | 可执行实现 |
| L1 | 固定 CPU 输入与局部不变量可检查 | 真实框架或目标对象 |
| L2 | 版本、输入、参数与结果可复算 | 代表性数据、目标硬件/网络 |
| L3 | 多组件、错误和恢复路径已接通 | 组织级容量、安全与运维验收 |
| L4 | SLO、权限、监控、成本和回滚方案完整 | 目标生产环境的实际结果 |

等级只绑定具体 claim，不代表整个项目、模型质量或 GPU 性能都达到同级。多条 L1/L2 control 不能用“相加”
升级为 L3/L4。录制报告保留当时的日期、输入和 runtime；`--verify` 只复核 artifact 身份、结构和范围，
不重新执行模型。SHA-256 也只在已有可信 digest/manifest 时检测 bytes 漂移，不认证发布者、执行者或时间。

## 权威证据入口

| 项目 | 精确结果与边界 | 运行手册 |
|---|---|---|
| Training Data Lineage | [准确性总台账](accuracy-ledger.md) | [站点项目](../training/data.md#training-data-lineage-run) |
| RAG Foundations | [RAG 请求与回答证据](rag-answer-controls.md) | [站点项目](../practice/projects/rag-foundations.md#run) |
| RAG Framework Adapters | 本页顶部当前等级；字段 parity 见项目测试 | [站点项目](../practice/projects/rag-framework-adapters.md#run) |
| Safe Agent | [面试验证索引](interview-controls.md)与[准确性总台账](accuracy-ledger.md) | [退款生命周期](../practice/projects/safe-agent.md#refund-lifecycle) |
| Single-GPU Finetuning | [Qwen](qwen-controls.md)、[对齐](alignment-controls.md)与[准确性总台账](accuracy-ledger.md) | [站点项目](../practice/projects/single-gpu-finetuning.md#run) |
| Transformers Basics | [Transformers](transformers-controls.md)、[Qwen](qwen-controls.md)与[前沿](frontier-controls.md)台账 | [站点项目](../practice/projects/transformers-basics.md#run) |
| JAX MiniGPT | [准确性总台账](accuracy-ledger.md) | [站点项目](../practice/projects/jax-minigpt.md#run) |
| Cloud API Contracts | [云 API 证据](cloud-api-controls.md) | [站点项目](../practice/projects/cloud-api-contracts.md#run) |
| Inference Serving | [推理服务证据](inference-serving-controls.md) | [站点项目](../practice/projects/inference-serving.md#run) |
| Evaluation Gate | [准确性总台账](accuracy-ledger.md) | [站点项目](../practice/projects/evaluation-gate.md#run) |
| Synthetic Data Audit | [准确性总台账](accuracy-ledger.md) | [站点项目](../practice/projects/synthetic-data-audit.md#run) |

站点项目页负责学习顺序与首次成功；项目 README 负责完整命令、输入输出和故障排查；专题证据页保存固定数字、
fingerprint、oracle 与不可外推范围。本页不再复制这三层内容。

## 跨项目唯一观察

| 项目 | 固定观察 | 边界 |
|---|---|---|
| JAX MiniGPT 单步 | Seed 11、学习率 0.02、632 个参数、8 个监督位置；loss `2.1085913181→1.9567849636`，裁剪前梯度范数 `1.6481350660` | 固定 CPU tiny 输入，不证明泛化、收敛或 accelerator 性能。 |
| Framework Agent adapter | 当前 LlamaIndex direct `FunctionTool.call()` 对 `key=7` 不先执行 `fn_schema` validation；canonical gate 会在 resolver 前拒绝 | 只说明该固定版本和调用路径，不能概括框架默认安全。 |
| Finetuning AMP | 完整更新边界遵循 `unscale→clip`；overflow、optimizer、scheduler 与 scaler state 共同决定是否提交 | 各 CPU/Gloo/tiny controls 不拼成目标 Qwen CUDA/QLoRA 证据。 |
| Evaluation Gate | 完整路径是 `score → comparison v2 → verify-comparison → verify-evidence → HTML → HMAC ledger` | HTML 是派生视图；本地复算不重放模型，HMAC 链还需要外部 trusted head。 |
| Paired randomization toy | Difference `[1,1,1,1,0]` 的 mean `0.8`、interval `[0.4,1.0]`；单/双侧 p-value `0.0625/0.125` | 固定枚举只核对计算，不建立真实抽样、exchangeability 或业务重要性。 |

## 跨项目复核规则

1. 先写最小 claim，再选择同层级的 oracle；不要从“脚本跑完”推导质量、性能或安全。
2. 区分 authored fixture、框架 control、目标模型/硬件运行与真实生产观测。
3. 保存失败、拒绝、timeout、unknown 和未完成结果的全分母。
4. 多个 control 必须保持各自的对象和边界，不能拼成未执行过的端到端系统。
5. Hash、recorded report、loopback、MockTransport、SQLite 与 CPU/Gloo 各自只证明明确记录的局部契约。
6. 需要更新精确数字时，在权威台账中新建运行记录，不用当前环境覆盖历史 artifact。

若要选择项目，从[工程项目索引](../practice/project-index.md)进入；若要判断生产准备度，使用
[生产检查表](../practice/production-checklist.md)。
