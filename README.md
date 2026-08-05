# About LLM

面向开发者与算法工程师的中文 LLM 教材与工程实验室。

本仓库不是论文链接目录，也不是只会调用 API 的教程。它把研究生层次的核心原理、从零实现、主流框架实践、生产系统设计和求职准备放在一套可验证的学习路径中。

## 你能学到什么

- 用直觉和张量形状理解 tokenization、Transformer、训练与生成。
- 用 PyTorch 和 JAX 从零实现语言模型的关键组件。
- 使用 Transformers 完成微调、量化、评测和本地推理。
- 使用 vLLM 面向单卡/服务场景做吞吐、延迟和显存优化。
- 分别用原生代码、LangChain 和 LlamaIndex 构建并评测 RAG。
- 用显式状态机和框架构建安全、可恢复的 Agent。
- 设计离线评测、线上指标、错误分析和发布门禁。
- 理解 GPT、Llama、Qwen、DeepSeek、Claude、Gemini 的公开架构与产品差异。
- 准备算法/LLM 应用岗位面试、系统设计和简历项目。

## 仓库组成

| 目录 | 内容 | 验收标准 |
|---|---|---|
| docs/ | MkDocs 教材与工作手册 | 严格构建、内部链接有效 |
| src/about_llm/ | 可复用的教学与工程代码 | 类型清晰、单元测试通过 |
| notebooks/ | 交互实验 | 可从头执行，固定种子与轻量默认配置 |
| projects/ | RAG、Agent、微调、推理、评测项目 | 有配置、测试、指标与故障说明 |
| tests/ | 单元/集成/回归测试 | CPU 默认可跑；GPU/API 测试显式标记 |
| scripts/ | 文档、环境和质量检查工具 | 无隐藏网络调用 |

完整导览见[仓库地图](docs/guide/repo-map.md)，学习安排见[开发者与算法工程师路线](docs/guide/learning-paths.md)，事实分类与验证范围见[内容准确性与核验台账](docs/reference/accuracy.md)。

## 三条执行环境

### 基础 CPU 路线

用于文档、tokenizer、注意力、微型 GPT、检索与评测：

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[docs,dev,torch,jax]"
python scripts/doctor.py
pytest -m "not gpu and not network"
mkdocs serve
~~~

### 单张消费级 GPU 路线

在基础环境上按本机 CUDA 版本从 PyTorch 官方源安装 GPU wheel，再安装：

~~~powershell
python -m pip install -e ".[transformers,finetune,qlora,rag]"
~~~

显存预算和推荐实验见[环境与硬件矩阵](docs/guide/environment.md)。不要盲目复制 CUDA 安装命令：驱动、CUDA runtime 和 wheel 必须匹配。

### 云 API 路线

复制 .env.example 为 .env，只填写实际使用的 provider。示例默认不读取、不打印密钥；所有产生费用的测试均标记为 network，不会在普通测试中运行。

## 快速检查

~~~powershell
python scripts/check_docs.py
python scripts/check_content_accuracy.py
python scripts/doctor.py
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

首版教材、三本 Notebook、八个重点工程项目和求职材料已经形成可运行闭环，状态与本地/外部验证边界见[实现矩阵](docs/guide/repo-map.md#实现矩阵)。真实 CUDA/vLLM/QLoRA 峰值和付费云 API 仍需在目标环境执行，仓库不会把离线协议测试表述为线上验证。变更见 [CHANGELOG.md](CHANGELOG.md)。

## License

文字内容采用 [CC BY 4.0](LICENSE)。代码示例后续如需独立软件许可证，将在代码目录明确标注；引入第三方模型、数据和依赖时请遵守各自许可证。
