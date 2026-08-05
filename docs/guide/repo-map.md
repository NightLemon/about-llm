# 仓库地图与实现契约

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
| Transformer | NumPy/PyTorch/JAX | Transformers | 微型 GPT | 前向/梯度等价、可过拟合小批次 |
| 微调 | loss mask、LoRA | PEFT/TRL | 单卡领域 SFT | 格式率、领域质量、通用回归、显存 |
| RAG | BM25/dense/hybrid/RRF | LangChain/LlamaIndex adapter | 带 ACL 与引用问答 | Recall@k、nDCG、忠实度、权限 |
| Agent | 显式状态机 | 可选框架 adapter | 可恢复工具执行 | 幂等、确认、预算、注入测试 |
| 推理 | KV Cache、采样、量化实验 | Transformers/vLLM | OpenAI-compatible 服务 | TTFT、TPOT、吞吐、显存、质量 |
| 评测 | 指标与 bootstrap | dataset/runner adapter | 发布门禁 | 可复现、分层、置信区间、回归 |

## 质量等级

- **L0 文档**：解释准确，有术语、边界和自测。
- **L1 最小实现**：CPU 可运行，单元测试覆盖核心不变量。
- **L2 可复现实验**：Notebook/脚本固定输入、种子和指标。
- **L3 工程样例**：配置化、日志、错误处理、集成测试。
- **L4 生产设计**：容量、安全、监控、回滚和成本齐全。

同一主题只有达到标注等级才能宣称完成。外部 API/GPU 测试必须显式 opt-in，CI 默认不产生费用。

## 代码约定

- Python 3.10+，路径使用 pathlib，配置与密钥分离。
- 公共函数有类型标注和 docstring；错误消息包含可操作上下文。
- 浮点测试使用容差；随机测试固定种子但不只测一个样本。
- 不在 import 时下载模型/数据、访问网络或初始化 GPU。
- 任何执行模型输出的代码都先做 schema、权限和副作用校验。

## 数据约定

小型教学数据可随仓库分发，但要标注来源和许可。大型、受限或可能变化的数据只提供下载说明和校验值。评测数据与训练数据隔离；生成物写入 outputs/，权重写入 checkpoints/，两者默认不提交。
