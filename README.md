# About LLM

一套面向开发者和算法工程师的中文 LLM 教材。这里既讲清楚模型为什么这样工作，也提供可以运行、修改和验证的实验。

如果你第一次来，直接从[新手路线](docs/guide/beginner-map.md)开始。已经具备机器学习基础的读者，可以按下面的方向进入。

## 选择一条路线

| 目标 | 建议起点 | 你会完成什么 |
|---|---|---|
| 理解语言模型 | [机器学习基础](docs/foundations/ml-dl.md) → [Tokenization](docs/core/tokenization.md) → [Transformer](docs/core/transformer.md) | 能解释训练、注意力和生成过程，并读懂常见模型结构 |
| 构建 LLM 应用 | [RAG](docs/applications/rag.md) → [Agent](docs/applications/agents.md) → [评测](docs/quality/evaluation.md) | 完成一个可评测的 RAG 或 Agent 系统，而不止是 API demo |
| 训练与部署模型 | [微调](docs/training/finetuning.md) → [推理](docs/systems/inference.md) → [服务](docs/systems/serving.md) | 理解数据、显存、吞吐和可靠性之间的工程权衡 |

完整的阶段安排见[学习路径](docs/guide/learning-paths.md)。按岗位或项目选题时，参考[知识地图](docs/guide/knowledge-map.md)和[项目索引](docs/practice/project-index.md)。

## 内容怎么组织

仓库采用三个层次，每一层解决不同问题：

1. **教材**解释概念、公式和工程判断，正文位于 `docs/`。
2. **实验**用小规模输入验证关键机制，入口位于 `notebooks/` 和 `docs/practice/labs/`。
3. **项目**把多个机制组合成可运行系统，代码与说明位于 `projects/`。

学习时不必先安装全部依赖。先读一章，再运行对应的最小实验；能解释实验结果后，最后进入项目。

## 十分钟开始

阅读文档不需要 Python。要在本地浏览完整站点：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -c constraints/ci.txt -e ".[docs]"
mkdocs serve
```

要先运行一个不下载模型的实验：

```powershell
python -m pip install -c constraints/ci.txt -e .
python projects/transformers-basics/train_byte_bpe.py
```

环境有问题时运行：

```powershell
python scripts/doctor.py --profile cpu-starter
```

Python 支持 3.10–3.12。GPU、CUDA 和云 API 都不是入门路线的前置条件；需要它们时再参考[环境与硬件矩阵](docs/guide/environment.md)。

## 仓库地图

| 路径 | 用途 |
|---|---|
| `docs/` | 教材正文、学习路线和参考资料 |
| `notebooks/` | 可交互的注意力、MiniGPT 与 RAG 实验 |
| `projects/` | RAG、Agent、微调、推理与评测项目 |
| `src/about_llm/` | 项目复用的 Python 实现 |
| `tests/` | 核心代码的正确性与回归测试 |
| `artifacts/` | 机器可读的实验结果，不作为正文入口 |

更详细的代码入口见[仓库地图](docs/guide/repo-map.md)。实验结果、适用范围和复现条件集中在[内容准确性与核验台账](docs/reference/accuracy.md)，版本变化记录在 [CHANGELOG](CHANGELOG.md)；README 不重复这些机器审计信息。

## 学习约定

- 先理解问题和基线，再引入框架或复杂优化。
- 从零实现用于建立直觉，成熟库用于工程实践。
- 代码跑通不等于方案有效；每个项目都需要独立评测。
- 外部模型、API 和硬件结论必须注明来源与运行条件。

## 参与贡献

教材应优先服务读者：一段只讲一个核心命题，公式必须解释变量，实验必须说明预期现象。固定哈希、完整测试账本和发布记录应放在证据页或 changelog，而不是学习正文。

提交前的写作与最小检查要求见 [CONTRIBUTING](CONTRIBUTING.md)。

## License

源码、可执行示例、测试和配置采用 [MIT License](LICENSE-CODE)；教材正文、图表和其他文字内容采用 [CC BY 4.0](LICENSE-DOCS)。完整适用边界见 [LICENSE](LICENSE)。
