# 实验与项目

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：希望把教材概念变成可运行观察，并逐步进入微调、RAG、Agent 或推理服务的开发者。
- **先修**：实验 0–2 只要求基本 Python；GPU、云 API 和模型下载都有本地或 CPU 前置实验。
- **首次阅读**：在选择矩阵中找到目标，只打开对应实验，按“预测—运行—解释—写边界”完成。
- **完成信号**：能保存一次失败，并说明它来自公式、数据、模型、运行时还是实验设计。
- **卡住时**：把变量缩减到一个 CPU 样例，先手算预期结果，再查看程序输出。

</div>

**实践导航**：[选择学习路径](../guide/learning-paths.md) · [配置环境](../guide/environment.md) · [工程项目索引](project-index.md) · [生产检查表](production-checklist.md)
{ .doc-nav }

本页只负责选择实验和给出第一次成功入口。每个独立 Lab 负责预测、命令、观察与反例；项目页负责组装路线，
项目 README 负责完整运行参数。不要从这里连续执行所有命令。

## 怎样选择实验

| 当前问题 | 先做 | 第一次成功 |
|---|---|---|
| logits 怎样变成 token | [0A：采样](labs/lab-0a-sampling.md) | 运行页面中的固定 logits 例子并手算候选集 |
| 生成何时结束 | [0B：生成协议](labs/lab-0b-generation-protocol.md) | 画出 EOS、长度、stop string 与断流终态 |
| 云调用失败后能否重试 | [0C：预算与对账](labs/lab-0c-cloud-budget.md) | 解释两次 attempt 的预算终态 |
| 不透明推理工件能否跨会话重放 | [0D：工件安全](labs/lab-0d-reasoning-artifact-security.md) | 预测并运行 16 个固定案例 |
| 文本怎样变成模型输入 | [实验 1](#lab-1) | 运行教学 BPE，再核对真实 Qwen tokenizer |
| Attention 与模型配置怎样对应 | [实验 2](#lab-2) | 手算 attention；再按需选 2A–2D |
| 训练或微调怎样闭环 | [实验 3](#lab-3)、[实验 4](#lab-4) | 先让 tiny batch overfit，再检查一条 SFT 样本 |
| RAG 哪一层先失败 | [实验 5](#lab-5) | 对比有证据回答与证据不足拒答 |
| Agent 副作用怎样收尾 | [实验 6](#lab-6) | 追踪一笔退款的 pending 与 reconciliation |
| KV 与真实服务怎样连接 | [实验 7](#lab-7) | 先做 CPU Paged KV，再进入目标 GPU |
| 分数怎样变成发布判断 | [实验 8](#lab-8) | 构造一个结构正确、语义错误的反例 |

每次至少留下运行配置、原始观察、一个失败样例和自己的解释。CPU 基线回答局部逻辑，目标模型、GPU 或远端服务
回答实际兼容性与性能；两种证据不能互相替代。

## Qwen3 + nano-vLLM 最短路线

按 0A → 1A/1B → 0B/2/2B → 7A → 7B → 服务项目推进。前五步分别固定采样、输入 identity、
attention/停止协议和 KV 状态；最后才在目标 GPU 测容量。训练、RAG 与 Agent 路线可以按目标插入，不是前置条件。

## 实验 0：先观察语言模型协议 {#lab-0}

| 层级 | 实验 | 唯一问题 |
|---|---|---|
| 必做 | [0A：从 logits 到采样](labs/lab-0a-sampling.md) | top-k、top-p 与 repetition penalty 怎样改变候选分布？ |
| 推荐 | [0B：生成、停止与流式协议](labs/lab-0b-generation-protocol.md) | EOS、长度、stop string 和断流怎样形成终态？ |
| 工程选修 | [0C：云 API 预算、重试与对账](labs/lab-0c-cloud-budget.md) | 一次逻辑调用为什么可能产生多次发送和费用？ |
| 安全选修 | [0D：不透明推理块安全](labs/lab-0d-reasoning-artifact-security.md) | 工件在哪些身份变化后必须拒绝？ |

## 实验 1：从教学 tokenizer 走到真实 Qwen3 输入 {#lab-1}

先运行教学样例：

~~~powershell
python projects/transformers-basics/trace_language_model_sample.py
python projects/transformers-basics/trace_minigpt_training_step.py
~~~

记录 bytes、token IDs、labels、loss mask、逐位置 NLL 与一次更新。教学词表只用于解释算法，不属于任何真实 checkpoint。

### 1B：真实 Qwen3 tokenizer {#lab-1b}

~~~powershell
python projects/transformers-basics/trace_qwen3_tokenizer.py --local-files-only
~~~

缓存不存在时按[项目运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/transformers-basics/README.md)
选择联网或本地 snapshot。交付物是 message、模板文本、token IDs 与目标 revision，不是只抄一个 token 数。

## 实验 2：从 Attention 到运行栈 {#lab-2}

先手算两个 token 的 score、causal mask、softmax 与输出，再比较 full causal forward 和逐 token cache。

### 2A：MoE routing 与 capacity {#lab-2a}

沿少量 token 预测 top-k、capacity、drop/reroute 和 combine；随后进入[MoE 系统](../frontier/moe-systems.md)选择单进程或
双进程控制。CPU/Gloo 只核对数据流，不代表 NCCL 性能。

### 2B：配置与生成协议 {#lab-2b}

用固定配置复算标准 GQA 的 KV，再制造字段缺失、head 不整除和 MLA 三个反例。交付一张“可推导 / 必须实测 /
信息不足”表；扩大配置里的最大位置数不能证明有效长上下文。

### 2C：真实 checkpoint 选修 {#lab-2c}

本地已有固定 snapshot 时，比较 prefill、cached decode、完整重算和 `generate()`；保存 revision、模板、输入 IDs、
容差与资源。单个 prompt 或 argmax 一致不能证明整体质量。

### 2D：从 RMSNorm 走到框架算子 {#lab-2d}

进入[实验 2D](labs/lab-2d-operator-stack.md)，观察同一次 RMSNorm 的非连续布局、FX、ATen 与 profiler 事件。

## 实验 3：训练微型 GPT {#lab-3}

固定数据切分、tokenizer、上下文、seed 和硬件，让 tiny batch overfit；只改变一个模型变量，并同时保存训练/验证 loss
和最差生成样例。Activation patching 属于[机制可解释性](../core/mechanistic-interpretability.md)选修，不在此展开。

## 实验 4：LoRA 领域适配 {#lab-4}

先完成[4A：追踪一个 SFT 样本](labs/lab-4a-sft-sample.md)，核对模板、右移标签、assistant mask、可训练参数和 adapter
独立重载。完整训练与恢复进入 [Single-GPU Finetuning](projects/single-gpu-finetuning.md)。

### 训练系统选修 {#lab-4-systems}

按需选择 MinHash/LSH、continual replay、checkpoint resume、token-weighted accumulation、DDP `no_sync`、AMP 或
DataLoader cursor。每题都要有完整批次/不中断正对照和故意遗漏一个状态的负例。

## 实验 5：可诊断的 RAG {#lab-5}

先完成[独立实验 5](labs/lab-5-rag-request.md)。它只要求你预测、运行和解释授权→召回→重排→packing→引用→发布，
然后把失败归到第一个错误阶段。完整系统组装见 [RAG Foundations](projects/rag-foundations.md)。

### 实验 5A：框架公平对照 {#lab-5a}

让仓库实现、LangChain 与 LlamaIndex 使用同一语料、问题、主体和候选数；检查转换前后 identity 与 ACL，
不要只比较最终文档 ID。项目入口见 [RAG Framework Adapters](projects/rag-framework-adapters.md)。

### 实验 5B：服务与真实模型选修 {#lab-5b}

在固定本地基线之后再接 API 与目标模型，分别观察 readiness、身份、queue、生成器调用次数、引用和拒答。
回环服务不证明 TLS、IAM 或远端行为。

## 实验 6：安全的工具 Agent {#lab-6}

完成[独立实验 6](labs/lab-6-agent-lifecycle.md)，只追踪 proposal、schema、ACL、approval、pending、replay、verifier 和
reconciliation。框架、outbox 与交付路线见 [Safe Agent](projects/safe-agent.md)。

### 实验 6A：MCP 与 A2A 互操作 {#lab-6a}

按进程内→stdio→本机回环 HTTP 逐层改变传输，始终分开 schema、允许列表、认证与业务授权。
[互操作专题](../applications/agent-interoperability.md)维护协议边界。

## 实验 7：量化与服务基准 {#lab-7}

先完成[7A：Paged KV 与 COW](labs/lab-7a-paged-kv.md)，再完成
[7B：Qwen3 穿过 nano-vLLM](labs/lab-7b-nano-vllm-qwen3.md)。理解状态后进入
[Inference Serving](projects/inference-serving.md)，在目标 GPU 固定 workload，报告完整终态、TTFT、TPOT、吞吐、
显存和质量；CPU 状态实验不能替代性能测量。

## 实验 8：评测与指标冲突 {#lab-8}

构造可手算反例，区分原文相等、规范化相等、token F1、schema 合法、字段正确和引用支持。
随后用独立留出集报告逐例结果与错误分母；完整发布证据链见 [Evaluation Gate](projects/evaluation-gate.md)。

## 综合项目验收

实验解释一个变量，项目才组装交付。项目索引负责路线，README 负责命令；最终报告应包含问题、基线、配置、逐例结果、
失败、安全边界、SLO、成本和回滚，并指出下一项最可能推翻当前结论的实验。
