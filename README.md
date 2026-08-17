# About LLM

面向开发者与算法工程师的中文 LLM 教材与工程实验室。

本仓库不是论文链接目录，也不是只会调用 API 的教程。它把研究生层次的核心原理、从零实现、主流框架实践、生产系统设计和求职准备放在一套可验证的学习路径中。

## 你能学到什么

- 用直觉和张量形状理解 tokenization、Transformer、训练与生成。
- 用 PyTorch 和 JAX 从零实现语言模型的关键组件。
- 运行 PyTorch↔JAX MiniGPT parity control，在显式对齐 LayerNorm/RMSNorm 差异、epsilon、GELU、mask 与 tied embedding 后对账 logits、loss、20 个参数梯度和一步 SGD；当前 CPU Float32 梯度最大差为 `2.384185791015625e-07`。
- 运行 PyTorch↔JAX 三步 AdamW parity：共享三张外部物化的 embedding inverted-dropout masks，对账 raw/clipped gradients、first/second moments、count、`0.02→0.01→0.005` schedule、参数与 post-step forward；最大参数差为 `2.5480985641479492e-06`，wrong-mask 反例为 `0.06900620367377996`，但不声称两个框架原生 RNG 等价。
- 运行 JAX/Optax cross-process bit-exact resume：13,476-byte strict artifact 保存参数、Optax moments、typed PRNG、shuffle permutation/cursor 和 step；错误重置 PRNG 的参数最大差为 `0.037261832505464554`。
- 使用 Transformers 完成微调、量化、评测和本地推理，并在固定 Qwen2.5-0.5B-Instruct 真实权重上执行 SFT token/mask/final-label 与 no-grad forward 对账、CPU FP32 LoRA backward/adapter 重载、TRL DPO 单步和 source-position activation-patching 结构对账。
- 使用 Transformers reference service 与 vLLM 面向单卡/服务场景做 API 契约、吞吐、延迟和显存优化；仓库已让固定 Qwen 权重经真实 loopback TCP/HTTP 完成 non-stream 与 SSE 调用，但不把 CPU reference 冒充 vLLM/GPU 性能证据。
- 分别用原生代码、LangChain 和 LlamaIndex 构建并评测 RAG，用固定 Qwen 权重观察真实漏引/拒答失败，先反事实重放发布策略，再以真实 guarded Qwen runtime 验证有证据 1 次、空证据 0 次的 framework `generate()` 边界。
- 用显式状态机和框架构建安全、可恢复的 Agent。
- 运行 MCP 2025-11-25 official-SDK memory/stdio/Streamable HTTP、authored strict stdio/Streamable HTTP 与官方 SDK A2A 1.0 loopback controls，理解 SDK、transport、协议证据和授权边界。
- 为云 API 的 opaque reasoning state 建立上下文绑定、重放保护和公开轨迹发布门禁。
- 设计离线评测、线上指标、错误分析和发布门禁。
- 理解 GPT、Llama、Qwen、DeepSeek、Claude、Gemini 的公开架构与产品差异；对 Llama/Qwen/DeepSeek 的代表性发布证据按不可变 revision、raw hash 与保守 projection 分层核验，并用固定 Qwen2.5-0.5B-Instruct 权重做 CPU FP32 forward/cache/generate 控制实验。
- 用两个同机 CPU/Gloo 进程执行 MoE token-to-owner `all_to_all_single`、owner-only expert forward、owner-to-source return 与 metadata scatter；当前账本是 416 logical tensor-payload bytes，不等于 wire bytes，也不外推为 CUDA/NCCL 或目标模型性能。
- 在独立训练 fixture 中用 authored autograd-enabled reverse-split all-to-all 返回 hidden/gate 梯度，对 replicated router gradient 做 SUM all-reduce、对 owner expert 执行 SGD；global-mean MSE `20.78017329703821→19.41091750734501`，但不外推 CUDA/NCCL、DDP 或目标 MoE 训练。
- 在另一条 capacity-aware all-to-all training control 中，对全局 4-token routing group 执行 score-priority drop、kept-only dispatch/backward 与一步 SGD；它覆盖全零 split 的 zero-assignment source rank，global-mean MSE `15.253670387373656→14.530264380025987`，仍只证明同机 CPU/Gloo authored fixture。
- 准备算法/LLM 应用岗位面试、系统设计和简历项目。

## 仓库组成

| 目录 | 内容 | 验收标准 |
|---|---|---|
| docs/ | MkDocs 教材与工作手册 | 严格构建、内部链接有效 |
| site/ | 由 `mkdocs build` 生成的静态站点，不是编辑源 | 关键页面内容与源码版本一致 |
| src/about_llm/ | 可复用的教学与工程代码 | 类型清晰、单元测试通过 |
| notebooks/ | 交互实验 | 可从头执行，固定种子与轻量默认配置 |
| projects/ | RAG、Agent、微调、推理、评测项目 | 有配置、测试、指标与故障说明 |
| tests/ | 单元/集成/回归测试 | CPU 默认可跑；GPU/API 测试显式标记 |
| scripts/ | 文档、环境和质量检查工具 | 无隐藏网络调用 |

完整导览见[仓库地图](docs/guide/repo-map.md)，学习安排见[开发者与算法工程师路线](docs/guide/learning-paths.md)，事实分类与验证范围见[内容准确性与核验台账](docs/reference/accuracy.md)。

教材修改以 `docs/`、`mkdocs.yml` 和 `overrides/` 为准。`site/` 在本地构建和 CI 部署时重新生成且被 Git 忽略；不要直接编辑其中的 HTML。

## 四档本地安装

先选择今天要完成的任务，不需要为阅读文档安装整个训练栈。所有档位都从仓库根目录执行；Windows PowerShell 的完整起步路径如下：

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

| 档位 | 安装命令 | 就绪检查 | 首个任务 |
|---|---|---|---|
| docs | `python -m pip install -c constraints/ci.txt -e ".[docs]"` | `python scripts/doctor.py --profile docs` | `mkdocs serve` |
| cpu-starter | `python -m pip install -c constraints/ci.txt -e .` | `python scripts/doctor.py --profile cpu-starter` | `python projects/transformers-basics/train_byte_bpe.py` |
| notebooks | `python -m pip install -c constraints/ci.txt -e ".[dev,torch,jax]"` | `python scripts/doctor.py --profile notebooks` | `python scripts/execute_notebooks.py` |
| full-ci | `python -m pip install -c constraints/ci.txt -e ".[docs,dev,torch,jax,transformers,finetune,rag,langchain,llamaindex,api,evaluation,agents]"` | `python scripts/doctor.py --profile full-ci` | `pytest -m "not gpu and not network"` |

`doctor` 对缺少的必需包、Python 版本、导入或 Notebook kernel 返回非零退出码，并给出修复命令；未进入虚拟环境只标记为 `warn`。默认 `python scripts/doctor.py` 仍只输出不含密钥值的环境报告。

Python 支持 3.10–3.12。`constraints/ci.txt` 固定 CI 已复核的直接依赖版本，同时保留平台相关传递依赖由 pip 解析；它不是 CUDA 或生产部署的全平台 lock。MkDocs 当前显式保持在 1.x，因为 Material 9.x 主题、插件和 overrides 尚未声明兼容 MkDocs 2。

## GPU 与云路线

### 单张消费级 GPU

在 `notebooks` 或 `full-ci` 环境上，按本机 CUDA 版本从 PyTorch 官方源安装 GPU wheel，再安装目标实验依赖：

~~~powershell
python -m pip install -e ".[transformers,finetune,qlora,rag]"
~~~

显存预算和推荐实验见[环境与硬件矩阵](docs/guide/environment.md)。不要盲目复制 CUDA 安装命令：驱动、CUDA runtime 和 wheel 必须匹配。

### 云 API

复制 .env.example 为 .env，只填写实际使用的 provider。示例默认不读取、不打印密钥；所有产生费用的测试均标记为 network，不会在普通测试中运行。

## 快速检查

~~~powershell
python scripts/check_docs.py
python scripts/check_content_accuracy.py
python scripts/doctor.py --profile full-ci
python -m ruff check .
python -m pytest -m "not gpu and not network"
mkdocs build --strict
~~~

## 设计原则

1. **先有基线和评测，再引入框架。**
2. **从零实现用于理解，成熟库用于生产。**
3. **公开信息和推测分开写。**闭源模型只描述官方公开能力与接口，不臆测内部参数；官方文档核对也不冒充真实 API/GPU 运行证明。
4. **每个优化同时报告质量、延迟、吞吐、显存和成本。**
5. **默认可在 CPU 或小数据上验证机制；重型实验给出单卡缩放方案。**
6. **模型输出、检索内容和工具参数都不被默认信任。**

## 当前状态

最新的目标权重可解释性 control 固定 France/Germany 单事实 pair、`Paris−Berlin` metric 与 layer 0/11/23，真实执行 Qwen2.5-0.5B-Instruct CPU FP32 hooks；source recovery 为 1.000024/0.992244/0，full-prefix/readout/future 结构对照为 1/1/0，report 为 `sha256:3f8410f5…ebb18c`。它只证明这个 authored fixed protocol 的执行与计算图边界，不证明事实存储层、唯一自然 circuit 或总体事实性。

微调侧的固定 Qwen SFT final-label control 已升级到 tool-aware v2：原生模板在多轮、并行 tool calls、带 preamble 的 tool call 三条 fixture 上均返回全零 assistant mask；审核模板保持 47 / 301 / 200 个 input IDs 完全相同，并在进入 Arrow 前生成 8 / 51 / 31 个 assistant-mask tokens。真实 TRL 0.29.1 collator 固定 `[3, 301]`、90 个监督 labels 与 813 个 `-100`，CPU FP32 no-grad loss 为 `1.251716`。report 为 `sha256:8b61fa58…10421a`；没有 backward/optimizer、QLoRA/CUDA、held-out 质量、任意 provider schema 或数据合法性证据。

首版教材、三本 Notebook、十个重点工程项目和求职材料已经形成可运行闭环，状态与本地/外部验证边界见[实现矩阵](docs/guide/repo-map.md#实现矩阵)。仓库已对一个固定 Qwen2.5-0.5B-Instruct revision 执行真实 CPU FP32 权重控制，并让同一组选定权重经真实子进程、IPv4 loopback TCP/HTTP、Bearer、`/v1/models` 与 `/v1/chat/completions` 完成一次 non-stream 和一次 SSE 请求；Uvicorn 0.52.1 重录的 service report 为 `sha256:63e566ca…617ddb`。该 SSE 在完整 generation 后才发块，不证明 incremental decode/cancel。另一条轻量 control 以 authored cooperative async backend 真实证明 content 先于 backend 完成，并在 client close 后让 ASGI task/iterator 观察取消且不再产生后续 scripted token；重录 report 为 `sha256:25846822…2b5d00`。第三条 control 在随机 1,272 参数 tiny GPT-2 上真实执行 CPU forward 与 thread 内 `GenerationMixin.generate()`，经人为 streamer pause、event 和 authored `StoppingCriteria` 让断连后的 continuation 保持 `[7]` 并 join thread；重录 report 为 `sha256:eadcab54…f62bc7`。它只证明显式 cooperative tiny path，不证明未修改/已阻塞调用、目标 Qwen、vLLM/CUDA、KV/CPU/GPU 释放或 provider 计费。RAG control 原样记录有证据漏引、空证据幻觉这两个失败；行为 gate 0/2 说明“真实执行”不等于“质量通过”。模型外 policy 先对同一 attempt 做 reject/abstain **反事实回放**，不冒充 guard 当时已经与 Qwen 同跑；随后用不同 query 真实包裹 Qwen callback，记录有证据 `GenerationMixin.generate` API 调用 1 次、空授权证据调用 0 次，report 为 `sha256:00706d00…f29ede`。微调侧在同一固定 checkpoint 上真实执行 270,336-parameter LoRA backward/单步 AdamW，验证冻结基座 fingerprint 与新基座 adapter bit-exact reload；report 为 `sha256:8a3897b1…026230`，但 loss 上升，只证明链路。另一条 TRL DPO control 使用两条 authored pair 完成一次真实 step，loss `0.693147→0.333352`、report 为 `sha256:3cafbade…b549b7bc`；冻结 parameter/state/config 指纹 exact，但 reference replay 有 `0.547077` 数值漂移，故不声称 bitwise 确定性、人类偏好或 held-out 改善。真实 CUDA/vLLM/QLoRA 峰值和付费云 API 仍需在目标环境执行。变更见 [CHANGELOG.md](CHANGELOG.md)。

## License

源码、可执行示例、测试和配置采用 [MIT License](LICENSE-CODE)；教材正文、图表和其他文字内容采用 [CC BY 4.0](LICENSE-DOCS)。Notebook 的代码单元采用 MIT，Markdown 单元采用 CC BY 4.0。完整适用边界见 [LICENSE](LICENSE)；第三方模型、数据、图片和依赖仍遵守各自许可证与条款。
