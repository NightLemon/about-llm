# 仓库地图：遇到一个问题时，下一步打开哪里

**相关导航**：[如何使用](how-to-use.md) · [学习路径](learning-paths.md) · [知识地图](knowledge-map.md) ·
[环境配置](environment.md) · [工程项目索引](../practice/project-index.md)
{ .doc-nav }

这个仓库同时包含教材、实验、代码和工程项目。第一次使用时不用先记目录结构，只需判断自己下一步想做什么：

| 你现在想做什么 | 打开哪里 | 你应该带走什么 |
|---|---|---|
| 理解一个机制为什么成立 | `docs/` 中的主题正文 | 直觉、公式、适用条件和常见误判 |
| 亲眼观察一个变量怎样改变结果 | `docs/practice/labs/` 或 `notebooks/` | 预测、原始输出和自己的解释 |
| 阅读机制怎样实现 | `src/about_llm/` | 数据结构、状态变化和错误处理 |
| 把多个部件接成可交付系统 | `projects/` | 运行入口、输入输出、失败路径和验收方法 |
| 核对某个数字或实验版本 | `docs/evidence/` | 固定输入、实际结果和结论范围 |

学习时通常按“正文 → 小实验 → 项目”前进。只有在想追代码或核对结论时，才进入 `src/`、测试和证据页。

## 用一个 KV Cache 问题走遍仓库

假设你正在问：**为什么生成下一个 token 时可以复用 KV Cache，推理框架又怎样保存这些状态？**

不要一次打开十篇文章。按下面五步走，每一步只多回答一个问题。

### 第一步：先看真实文本怎样变成模型输入

阅读 [Qwen 请求主线](../models/qwen.md#local-request-stack)，然后运行固定 Qwen3 tokenizer：

~~~powershell
python projects/transformers-basics/trace_qwen3_tokenizer.py --local-files-only
~~~

固定版本已在本地缓存时，这条命令会展示 chat template、29 个输入 ID 和每个 token 的可读片段。
第一次尚未缓存时可以去掉 `--local-files-only`。完成这一步后，你应该能指出哪些 ID 来自用户正文，哪些来自模板。

### 第二步：用小张量看懂 block 分配

阅读 [Paged KV 实验](../practice/labs/lab-7a-paged-kv.md)，运行：

~~~powershell
python projects/inference-serving/paged_kv_tensor_toy.py
~~~

这个 CPU 实验把 block size 缩小到 3。你可以亲手复算五个前缀 token 占用几个块、分支为什么共享前缀，以及
继续写入时为什么触发写时复制。此时关注的是状态和账本，GPU kernel 留到后面。

### 第三步：读懂它怎样写成代码

小实验的核心实现位于：

- `src/about_llm/inference/kv_allocator.py`：分配、引用计数、分支与容量错误；
- `src/about_llm/inference/paged_kv_torch.py`：把逻辑 block table 接到真实 PyTorch K/V 张量；
- `tests/test_paged_kv_torch.py`：正常路径、容量失败和张量一致性检查。

先沿 `append()` 或 `fork_sequence()` 读一条路径，再看相应测试。无需从包入口开始逐文件阅读。

### 第四步：换到真实 Qwen3 与 nano-vLLM

[实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md)把同一概念放进真实模型运行时。它使用 256-token block，
并观察 prefill、decode、prefix hit、序列状态和 KV 释放。

实验输入是 768 个固定生成的 ID，目的是让 block 账本容易复算。第一步的 29 个 ID 来自真实中文请求；两组输入
分别解释“文本怎样编码”和“运行时怎样调度”。

### 第五步：最后进入服务容量

完成 [Inference Serving 项目](../practice/projects/inference-serving.md)，再把真实业务 Prompt、并发和请求长度分布
接回来。到这里才测 TTFT、TPOT、吞吐、显存、取消和失败终态。

这五步展示了目录之间的关系：正文先解释问题，小实验隔离机制，`src/` 暴露实现，目标实验运行真实模型，项目再处理
服务交付。以后学习 RAG、Agent 或微调时，也可以沿同一顺序找入口。

## 按主题直接进入第一步

| 想学什么 | 先读 | 第一次运行 | 学会后能解释什么 |
|---|---|---|---|
| Tokenization | [Tokenizer](../core/tokenization.md) | [实验 1A/1B](../practice/labs.md#lab-1) | 教学 BPE 与目标 tokenizer 的职责差异 |
| Transformer | [Transformer](../core/transformer.md) | [Transformers Basics](../practice/projects/transformers-basics.md#run) | 张量形状、因果 mask 与 cache 一致性 |
| 生成 | [生成基础](../core/generation-basics.md) | [实验 0A](../practice/labs/lab-0a-sampling.md) | 概率怎样变成 token，循环为什么结束 |
| 推理服务 | [请求生命周期](../systems/inference-request-lifecycle.md) | [Paged KV 实验](../practice/labs/lab-7a-paged-kv.md) | 排队、prefill、decode 与容量怎样连接 |
| RAG | [一次 RAG 请求](../applications/rag-request-lifecycle.md) | [实验 5](../practice/labs/lab-5-rag-request.md) | 权限、召回、重排、引用与拒答 |
| Agent | [一次退款任务](../applications/agent-task-lifecycle.md) | [实验 6](../practice/labs/lab-6-agent-lifecycle.md) | 审批、幂等、超时未知与恢复 |
| SFT/QLoRA | [SFT 数据闭环](../training/sft-data-pipeline.md) | [实验 4A](../practice/labs/lab-4a-sft-sample.md) | 模板、labels、LoRA 更新与独立重载 |
| JAX | [JAX/Optax](../training/jax-optax.md) | [JAX MiniGPT](../practice/projects/jax-minigpt.md) | PyTree、JIT、PRNG 与恢复状态 |
| 评测 | [评测总览](../quality/evaluation.md) | [实验 8](../practice/labs.md#lab-8) | 指标分数怎样变成发布决定 |
| Cloud API | [云 API 契约](../models/cloud-api-contracts.md) | [实验 0C](../practice/labs/lab-0c-cloud-budget.md) | 重试、流式、用量和不确定结果 |

所有项目及其第一次成功命令都集中在[工程项目索引](../practice/project-index.md)。

## 三类实验分别回答什么

仓库经常为同一个主题提供三种实验。它们的区别在于问题不同，不在于哪一种“更高级”。

| 实验类型 | 典型例子 | 可以回答的问题 | 还需要什么后续证据 |
|---|---|---|---|
| 公式与参考实现 | NumPy attention、手写 BPE | 给定小输入时，公式和状态怎样变化 | 框架接口与目标硬件行为 |
| 框架小实验 | tiny PyTorch/JAX、Transformers tokenizer | 当前框架版本能否执行这条固定路径 | 目标 checkpoint、代表性数据或性能 |
| 目标环境实验 | 固定 Qwen、3070 Laptop、真实 API | 指定版本、输入和机器上的实际行为 | 更广负载、质量样本和生产验收 |

例如，NumPy GQA 能核对查询头与键值头的映射。CUDA 内核速度要在目标 GPU 上测量。

固定 Qwen CPU 前向计算能说明特定权重和输入实际运行过。模型质量要用独立评测集回答。

`MockTransport` 可以观察客户端是否关闭响应。云端是否停算和计费要由真实供应商协议与账单核对。

想追溯精确版本和结果时再打开：

- [项目实验与证据](../evidence/project-controls.md)：每个项目运行到了哪一层；
- [内容准确性证据](../evidence/accuracy-ledger.md)：重要结论由什么实现或实验支持；
- 各主题的 `docs/evidence/*-controls.md`：固定输入、录制结果和未覆盖条件。

## 证据等级怎样读

等级必须和具体结论写在一起。例如，“KV block 分配达到 L2”并不表示同一项目的 GPU 性能也达到 L2。

| 等级 | 已经做了什么 | 读者接下来还要验证什么 |
|---|---|---|
| L0 文档 | 机制、术语和边界已经写清 | 代码能否运行 |
| L1 最小实现 | 固定 CPU 输入和核心检查通过 | 框架或目标模型兼容性 |
| L2 可复现实验 | 版本、输入、参数和结果可以复算 | 样本与真实 workload 是否匹配 |
| L3 工程样例 | 多个组件、错误和恢复路径已经接通 | 目标组织的容量、安全与运维验收 |
| L4 生产设计 | SLO、权限、监控、成本和回滚方案完整 | 方案在目标生产环境的实际结果 |

## 学习时怎样使用测试

先运行当前页面给出的单个脚本，再改一个变量。能解释变化后，只运行离该机制最近的测试。

例如学习 Agent 时，先运行退款生命周期，观察远端已经受理但本地超时的分支。理解 `pending` 状态后，再打开
相应恢复测试。框架适配、MCP 和 A2A 属于后续专题，无需作为第一次运行的前置条件。

完整测试套件主要用于维护仓库。它能检查已经写下的预期，无法代替你对输入、状态和结论范围的解释。

## 实现与数据约定

- Python 支持 3.10–3.12；路径优先使用 `pathlib`，配置与 secret 分离。
- 公共函数提供类型标注和可操作的错误信息。
- 浮点比较写明容差来源；随机实验固定 seed，并保留至少一个反例。
- 导入模块时不下载模型或数据，也不访问网络、初始化 GPU。
- 模型只能提出工具动作。执行层继续检查 Schema、权限、审批和副作用。
- 仓库内只保存可公开分发的小型教学数据；大型或受限数据保存获取方法与校验信息。
- 训练集与评测集分开管理；本机生成的报告和模型工件默认不提交到 Git。

## 下一步

1. 先运行[环境检查](environment.md)。
2. 按[学习路径](learning-paths.md)选择一条主线。
3. 到[实验与项目](../practice/labs.md)完成最小可运行练习。
4. 准备交付时使用[生产检查表](../practice/production-checklist.md)。
