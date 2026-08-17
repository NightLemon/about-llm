# 仓库地图与实现契约

**相关导航**：[如何使用](how-to-use.md) · [学习路径](learning-paths.md) · [知识地图](knowledge-map.md) · [环境配置](environment.md) · [工程项目索引](../practice/project-index.md)
{ .doc-nav }

## 四层结构

### 教材层：docs/

解释 Why、What 与 Trade-off。每章以直觉为入口，给张量形状、机制、工程选择、失败模式和面试追问；公式服务于理解，不以推导长度衡量深度。

### 实验层：notebooks/

展示一个可观察现象：token 切分、attention mask、训练曲线、检索召回、量化误差等。Notebook 必须能 Restart & Run All；默认使用小数据和固定随机种子，不把大段核心逻辑藏在单元格中，而是调用 src/。

### 实现层：src/about_llm/

提供可测试模块。from_scratch/ 为教学实现，追求透明和等价性；工程模块追求清晰接口、错误处理和可组合性。教学实现不会冒充高性能生产 kernel。

### 项目层：projects/

围绕真实交付组织，包含 README、配置、数据契约、基线、评测、测试、故障注入和部署说明。框架版本和外部服务是可替换 adapter，核心领域逻辑不绑定 LangChain/LlamaIndex。

## 实现矩阵

| 方向 | 从零/原生基线 | 主流框架 | 生产项目 | 关键验收 |
|---|---|---|---|---|
| Tokenizer | UTF-8 byte、确定性 byte-level BPE | checkpoint tokenizer/chat template | tokenizer 机制与契约实验 | round-trip、document boundary、merge rank、special token/offset/版本边界 |
| Transformer/MoE | NumPy RMSNorm/RoPE/GQA/cache 与 top-k/capacity/sparse-linear oracle、PyTorch trainable top-k router/MLP 与 PyTorch/JAX MiniGPT | Transformers + immutable Llama/Qwen/DeepSeek release evidence + 固定 Qwen CPU FP32 权重/activation-patching controls | 微型 GPT/路由/配置/目标 checkpoint 证据实验 | 局部代数/因果/cache 等价；PyTorch/JAX LayerNorm parity 对齐 forward/backward/SGD，并以 RMSNorm 反事实证明默认模型不等价，但不含 AdamW/RNG/JIT/GPU。MoE 单进程 controls 覆盖 assignment/drop/combine、sparse—dense forward/backward、gate/balance gradient、padding/group competition、full-ranking reroute、dropless excess 与 materialized-zero gradient。独立 two-process CPU/Gloo control 用 hidden `all_gather`/count `all_reduce` 对比 local kept=2 与 replicated-global kept=1；该 capacity control 仍无 expert ownership/token `all_to_all`。后续同机 Gloo controls 分别执行 owner-only variable-split dispatch/return、无 capacity 的 reverse-split training，以及 global drop + kept-only all-to-all backward + owner/router SGD；仍无 distributed reroute/dropless、GPU/NCCL、目标 MoE checkpoint或性能。固定 Qwen 权重只覆盖单 prompt/pair 的 prefill/cache/generate 与 activation patching；不声称 authored MoE policy 是框架默认，也不声称 shared/fine-grained expert、完整生产 expert-parallel/grouped-GEMM、事实存储层、唯一 circuit、Llama/DeepSeek 权重、有效上下文、总体质量、CUDA/vLLM 或 GPU hook/kernel 等价 |
| JAX 训练与恢复 | 纯函数 MiniGPT、JIT/Optax、PyTorch parity 与 JAX strict checkpoint/resume | JAX + Optax | JAX MiniGPT | tiny-batch overfit；LayerNorm SGD parity；shared-mask AdamW trajectory 对账 raw/clipped gradients、moments/count、schedule 与三步更新；13,476-byte artifact 保存 params/Optax/PRNG/data cursor，跨独立进程 bit-exact。共享 mask 不证明 native RNG 等价；当前不含 Orbax/TensorStore、sharding、accelerator、目标模型或性能证据 |
| 微调 | loss mask、tool-aware SFT、LoRA、DPO、token-mean reduction | PEFT/TRL + 固定 Qwen CPU FP32 target controls + PyTorch CPU/Gloo DDP + CPU FP16 GradScaler/resume controls | 单卡领域 SFT/偏好训练 | 固定 Qwen final labels/no-grad forward、LoRA/adapter reload 与 DPO 单步均已执行；另有 masked-token oracle、DDP `D/N`/`no_sync`/clip/SGD、独立 AMP 与单参数 DDP+AMP 共识 gate。统一 6 参数 CPU resume control 保存 model/AdamW/StepLR/scaler/RNG/custom shuffle；独立 2-worker prefetch control 证明 emitted cursor 可跳样本与 fresh worker RNG 不重放；后续六进程 2-worker/stochastic-forward/backward/SGD/StepLR accumulation control 在 consumed=3/committed=2 时证明 commit-RNG replay 与完整 gradients+crash-RNG sidecar resume bit-exact。隔离负例又证明：正确 RNG、相同 steps/LR 仍会因漏 gradients/sample 漂移；完整 ledger/gradients/steps/LR 仍会因错误 RNG 漂移。sidecar 路径新增 manifest-last completeness 与四种 incomplete/tamper fault snapshots，但 base-only 仍可 replay。仍无 queue/worker/Python/NumPy/CUDA RNG、原子 sample—optimizer—base+sidecar+manifest 事务、directory `fsync`/断电与来源认证/不可变快照、分布式 checkpoint、自然 overflow、多 bucket、CUDA 或目标 Trainer；SFT 无 backward，LoRA loss 上升，DPO 仅同 batch 下降。均不证明数据合法性、held-out 质量/收敛，QLoRA/CUDA/峰值仍未验证 |
| RAG | BM25/dense/hybrid/RRF + fail-closed publication policy | LangChain/LlamaIndex ACL-bound Retriever/Prompt adapter + 固定 Qwen generation/guarded controls | SQLite + persistent extractive ASGI API | framework/API 前授权、closed body schema、readiness、queue/deadline、ordered result/metadata/Prompt/artifact identity、Recall@k、nDCG、忠实度、权限；固定 Qwen attempt-1 真实执行得到漏引/拒答失败（gate 0/2），policy replay 仅为反事实；独立 guarded runtime 又真实观察有证据 `generate()` method=1、空证据=0，但不计内部 forward/provider billing，也不证明语义或生产修复 |
| Agent | 显式状态机、strict proposal/schema、SDK-memory/stdio/HTTP protocol fixtures | MCP 2025-11-25 official-SDK memory + stdio + Streamable HTTP，以及 authored local transport controls、A2A 1.0 official SDK | 可恢复工具执行与 transactional outbox | 幂等、确认、预算、注入；official SDK stdio/HTTP 同时覆盖 SDK+具体 transport，但不继承 authored negative matrix/conformance；私有 control token 不是 MCP auth，所有 loopback/A2A 证据都不证明 OAuth/TLS/远程或业务授权 |
| Cloud API 安全 | strict provider fixture、retry oracle、context-bound AES-GCM 与 trajectory allowlist | HTTPX、cryptography | Cloud API Contracts | subject/tenant/session/predecessor/model/expiry/key/replay 负例，reasoning/signature/unknown block 发布拒绝；authored 离线 control 不证明当前 provider 协议、KMS/HSM、分布式 ledger 或 secret/PII 清理 |

Transformer/MoE 还有一条独立的 two-process CPU/Gloo dispatch fixture：variable-split `all_to_all_single` 建立 owner-only expert placement，完成 token/gate/metadata dispatch、owner forward、return 与 source scatter；source→owner counts 为 `[[1,2],[1,0]]`，当前 416-byte logical payload 与单进程 oracle 对账。它不测 wire bytes，也不含 capacity、backward、CUDA/NCCL、目标模型或性能；不能借用表内其他 controls 的证据升级。

后续 training fixture 再用 authored autograd reverse-split 把 gradient 发回 owner/source，对 owner expert + synchronized replicated router执行一步 SGD，并与单进程 global MSE 对账。它自身仍不含 capacity、DDP 或 CUDA。

最新 capacity-aware training fixture 对 global mask `[F,T,T,F]` 执行 kept-only all-to-all backward，覆盖全零 source splits 与 dropped-token zero task gradient，并再次对齐单进程 oracle。它仍不证明目标 MoE、distributed reroute/dropless、生产拓扑、CUDA/NCCL、收敛或性能。
| 推理 | 单步 sampling、UTF-8 stop matcher、continuous batching、KV Cache、量化实验、repo-native MiniGPT checkpoint | Transformers reference + authored incremental ASGI/thread controls + vLLM 目标路线 | OpenAI-compatible 服务 | processor/top-k/top-p/CDF、partial stop/overlap、admission/work conservation、严格 tokenizer/config/全参数 reload、TTFT、TPOT、吞吐、显存、质量；固定 Qwen 已走真实 post-completion Transformers HTTP，纯 async control 验证断连传播，tiny GPT-2 又验证显式 event/`StoppingCriteria` 下的真实 generate-thread join；仍不冒充未修改/目标模型取消、vLLM/CUDA、KV/CPU/GPU release、完整 API、性能或生产证据 |
| 评测 | 指标与 bootstrap | dataset/runner adapter | 发布门禁 | 可复现、分层、置信区间、回归 |

## 质量等级

- **L0 文档**：解释准确，有术语、边界和自测。
- **L1 最小实现**：CPU 可运行，单元测试覆盖核心不变量。
- **L2 可复现实验**：Notebook/脚本固定输入、种子和指标。
- **L3 工程样例**：配置化、日志、错误处理、集成测试。
- **L4 生产设计**：容量、安全、监控、回滚和成本齐全。

同一主题只有达到标注等级才能宣称完成。外部 API/GPU 测试必须显式 opt-in，CI 默认不产生费用。

## 代码约定

- Python 3.10–3.12，路径使用 pathlib，配置与密钥分离。
- 公共函数有类型标注和 docstring；错误消息包含可操作上下文。
- 浮点测试使用容差；随机测试固定种子但不只测一个样本。
- 不在 import 时下载模型/数据、访问网络或初始化 GPU。
- 任何执行模型输出的代码都先做 schema、权限和副作用校验。

## 数据约定

小型教学数据可随仓库分发，但要标注来源和许可。大型、受限或可能变化的数据只提供下载说明和校验值。评测数据与训练数据隔离；生成物写入 outputs/，权重写入 checkpoints/，两者默认不提交。

## 下一步

- 运行前先完成[环境检查](environment.md)。
- 从[实验与项目](../practice/labs.md)选择任务，再到[工程项目索引](../practice/project-index.md)核对入口和证据等级。
- 准备交付时使用[生产检查表](../practice/production-checklist.md)，并在[准确性台账](../reference/accuracy.md)确认结论边界。
