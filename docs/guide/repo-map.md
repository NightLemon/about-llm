# 仓库地图：想学一个主题时，应该从哪里进去

**相关导航**：[如何使用](how-to-use.md) · [学习路径](learning-paths.md) · [知识地图](knowledge-map.md) ·
[环境配置](environment.md) · [工程项目索引](../practice/project-index.md)
{ .doc-nav }

这个仓库同时放了教材、实验、可复用实现和端到端项目。它们不是同一内容的四份拷贝，而是回答四种问题：

```text
docs/      我为什么需要这个机制，它在什么条件下成立？
notebooks/ 我能否亲眼看到一个最小现象？
src/       这个机制怎样实现成可测试模块？
projects/  怎样把模块组成一次可交付的工程任务？
```

第一次学习一个主题时，通常按这个顺序走；定位 bug 时则经常从 project trace 反向进入 `src/` 与测试。

## 四层内容各自承担什么责任

### 教材层：`docs/`

正文负责建立直觉、因果链、公式和工程取舍。关键章节会给出“完成信号”和下一步实验，但不会把所有 fixture、
hash 与测试排列组合塞进主线。精确验证结果集中在 `docs/evidence/`。

### 实验层：`notebooks/` 与 `docs/practice/labs/`

实验让读者观察一个现象，例如 token merge、causal mask、训练 loss、检索排序或量化误差。
Notebook 应能 Restart & Run All；核心逻辑放在 `src/` 或 `projects/`，避免隐藏在不可测试的长单元格中。

### 实现层：`src/about_llm/`

`from_scratch/` 优先透明、可手算和 correctness oracle；其他模块提供明确接口、错误类型和可组合状态。
教学实现不会冒充生产 kernel，工程模块也不会在 import 时下载模型、访问网络或初始化 GPU。

### 项目层：`projects/`

项目围绕一次交付组织：输入契约、运行命令、artifact、测试、故障注入和验收。LangChain、LlamaIndex、Provider SDK
等框架放在 adapter 边界，核心数据/权限/状态语义保持可独立验证。

## 按主题找入口

| 想学什么 | 先读 | 再运行 | 完成信号 |
|---|---|---|---|
| Tokenization | [Tokenizer](../core/tokenization.md) | 实验 1 / tokenizer tests | 能解释 round-trip、merge 与 chat-template identity |
| Transformer | [Transformer](../core/transformer.md) | `transformers-basics` | 能手算 shape，并对账 causal/cache 不变量 |
| 生成 | [生成基础](../core/generation-basics.md) | 实验 0A/0B | 能复算采样并说明停止原因 |
| 推理服务 | [请求生命周期](../systems/inference-request-lifecycle.md) | `inference-serving` | 能分解 queue、prefill、decode 与 KV 容量 |
| RAG | [一次 RAG 请求](../applications/rag.md) | `rag-foundations` | 能追踪 ACL、检索、packing、引用与拒答 |
| Agent | [一次退款任务](../applications/agent-task-lifecycle.md) | `safe-agent` / 实验 6 | 能处理 timeout unknown、幂等与 reconciliation |
| SFT/QLoRA | [SFT 数据闭环](../training/sft-data-pipeline.md) | `single-gpu-finetuning` | 能审计 final labels、训练、重载与 held-out gate |
| JAX | [JAX/Optax](../training/jax-optax.md) | `jax-minigpt` | 能解释 PyTree、JIT、PRNG 与完整 resume state |
| 评测 | [评测总览](../quality/evaluation.md) | `evaluation-gate` | 能从 recorded answers 重建发布决定 |
| Cloud API | [云 API 契约](../models/cloud-api-contracts.md) | `cloud-api-contracts` | 能判断 partial、retry、usage 与 uncertain outcome |

项目目录与完整命令可从[项目索引](../practice/project-index.md)进入。

## 三类实现不要混成同一份证据

仓库经常同时提供：

1. **数学 oracle**：NumPy/精确分数实现，用于检查公式和边界；
2. **框架 control**：真实调用 PyTorch、JAX、Transformers 或 SDK，但输入很小且固定；
3. **目标环境实验**：在指定 checkpoint、GPU、runtime 与 workload 上测行为或性能。

例如 NumPy GQA 可以证明 head mapping，不能证明 CUDA kernel 快；固定 Qwen CPU forward 说明真实权重路径执行过，
不能证明总体质量；MockTransport 的 response close 说明客户端释放资源，不能证明云端停止计费。

想查看每条 control 的精确版本、fixture 与未覆盖项，请用：

- [项目控制台账](../evidence/project-controls.md)：项目级可执行证据；
- [准确性台账](../evidence/accuracy-ledger.md)：重要结论与验证入口；
- 各主题的 `docs/evidence/*-controls.md`：精确结果与边界。

## 质量等级怎样理解

| 等级 | 表示什么 | 尚不能默认说明什么 |
|---|---|---|
| L0 文档 | 机制、术语、边界和自测完整 | 代码可运行 |
| L1 最小实现 | CPU 实现与核心单元测试通过 | 框架/目标模型兼容 |
| L2 可复现实验 | 固定输入、seed、artifact 和结果 | 真实 workload 代表性 |
| L3 工程样例 | 配置、日志、错误与集成路径完整 | 已满足生产 SLO |
| L4 生产设计 | 容量、安全、监控、回滚和成本方案齐全 | 目标组织已实地验收 |

同一主题可以在数学机制上达到 L2，而在 CUDA 性能上仍只有“待目标环境实测”。证据等级必须绑定具体结论，
不能给整个目录贴一个笼统等级。

## 一个推荐的工作循环

```mermaid
flowchart LR
  Q["从一个具体问题开始"] --> D["读对应主线章节"]
  D --> L["运行最小 lab"]
  L --> P["运行端到端 project"]
  P --> F["故意触发一个失败"]
  F --> E["核对 evidence / tests"]
  E --> N["记录结论与未覆盖项"]
```

例如学习 Agent 时，不必先跑所有 MCP transport tests。先运行退款生命周期，看懂为什么 `pending` 不能直接重试；
需要接框架时再运行 LangChain/LlamaIndex adapter；准备协议互操作时才进入 MCP/A2A controls。

## 代码与数据约定

- Python 支持 3.10–3.12；路径优先使用 `pathlib`，配置与 secret 分离。
- 公共函数提供类型标注和可操作错误信息。
- 浮点测试使用有理由的 tolerance；随机实验固定 seed，也保留至少一个反例。
- Import 不下载模型/数据、不访问网络、不初始化 GPU。
- 执行模型 proposal 前，外部 Runtime 继续做 Schema、权限、审批和副作用校验。
- 小型 authored 教学数据可以随仓库分发；大型或受限数据只保存获取说明、identity 与校验方式。
- 训练与评测数据分权；生成 artifact 默认不提交到 Git。

## 下一步

1. 先运行[环境检查](environment.md)。
2. 按[学习路径](learning-paths.md)选择一条主线。
3. 到[实验与项目](../practice/labs.md)完成最小可运行练习。
4. 准备交付时使用[生产检查表](../practice/production-checklist.md)。
