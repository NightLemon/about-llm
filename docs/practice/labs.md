# 实验与项目

实验的目的不是把命令跑绿，而是隔离一个机制：先写预测，只改变一个变量，保留失败样例，再说明结果不能推出什么。固定 hash、录制报告和完整参数位于对应项目目录，不在本页展开。

**实践导航**：[选择学习路径](../guide/learning-paths.md) · [配置环境](../guide/environment.md) · [工程项目索引](project-index.md) · [生产检查表](production-checklist.md)
{ .doc-nav }

## 怎样选择实验

| 当前目标 | 从这里开始 | 预计时间 |
|---|---|---:|
| 第一次观察生成模型 | [实验 0A](labs/lab-0a-sampling.md) | 30–60 分钟 |
| 理解 tokenizer 与注意力 | [实验 1](#lab-1)、[实验 2](#lab-2) | 2–4 小时 |
| 训练或微调模型 | [实验 3](#lab-3)、[实验 4A](labs/lab-4a-sft-sample.md) | 1–3 天 |
| 构建 RAG 或 Agent | [RAG 实验 5](labs/lab-5-rag-request.md)、[Agent 实验 6](labs/lab-6-agent-lifecycle.md) | 1–3 天 |
| 学习服务与评测 | [实验 7](#lab-7)、[实验 8](#lab-8) | 1–3 天 |

每次实验至少提交四项：运行配置、原始观察、一个负例、自己的解释。模型下载、GPU 和网络实验都是选修；先用 CPU 基线建立正确性分母。

## 实验 0：观察语言模型，而不是只和它聊天 { #lab-0 }

第一次只完成 0A。后续三个实验分别进入协议、云成本和工件安全，不是入门前置。

| 层级 | 实验 | 你要回答的问题 |
|---|---|---|
| 必做 | [0A：从 logits 到采样](labs/lab-0a-sampling.md) | temperature、top-k、top-p 怎样改变候选分布？ |
| 推荐 | [0B：生成、停止与流式协议](labs/lab-0b-generation-protocol.md) | EOS、长度上限、stop string 和断流如何形成终态？ |
| 工程选修 | [0C：云 API 预算、重试与对账](labs/lab-0c-cloud-budget.md) | 一次逻辑请求为什么可能产生多次调用和费用？ |
| 安全选修 | [0D：Opaque Reasoning 工件安全](labs/lab-0d-reasoning-artifact-security.md) | 哪些内部字段可以保存、重放或返回给用户？ |

交付物：一张手算表、一张状态图，以及至少一个“输出看似合理但协议已经失败”的例子。

## 实验 1：手写 tokenizer 和语言模型 { #lab-1 }

从 `projects/transformers-basics/train_byte_bpe.py` 开始：

1. 在小语料上手算一次 pair count、tie-break 和非重叠 merge。
2. 比较字符、UTF-8 byte 与 grapheme；验证 encode/decode round trip。
3. 用 bigram 建立最小语言模型，报告验证集 NLL/PPL。
4. 比较中文、英文、代码、数字和 emoji 的 bytes/token 与序列长度。
5. 修改一条输入，先预测 merge 和概率怎样变化，再重新运行。

交付物：token 表、长度分布、一个未见 bigram 或边界字符失败样例。Round trip 只证明本 tokenizer 自洽，不证明它兼容任意现有 checkpoint。

## 实验 2：从零实现注意力 { #lab-2 }

只用张量基础算子实现单头 causal attention，然后扩展到多头：

1. 为 `Q/K/V/score/output` 标出形状。
2. 手算两个 token 的 score、mask、softmax 和输出。
3. 验证修改未来 token 不会改变过去位置输出。
4. 关闭 dropout，比较逐 token cache 与完整 causal forward。
5. 让序列长度翻倍，分别记录理论中间元素数和实际运行时间。

`projects/transformers-basics/online_softmax_demo.py` 可用来理解 running max、normalizer 和 value accumulator。NumPy 结果用于验证代数，不代表 FlashAttention、GPU 内存或性能。

### 实验 2A：MoE routing 与 capacity { #lab-2a }

先在纸上为少量 token 完成 top-k routing，再改变 capacity factor、drop/reroute 策略和 combine normalization。关注三个问题：离散 expert index 如何产生、selected probability 如何传梯度、容量溢出怎样改变输出。

进阶时再比较单进程 sparse/dense oracle、跨 rank token dispatch 与 router/expert gradient。每一步都画出 token owner、expert owner 和 collective；同机 CPU/Gloo 对账不能外推为 NCCL 性能或目标 MoE checkpoint 复现。

### 实验 2B：配置与生成协议 { #lab-2b }

用项目中的 config fixtures 手算标准 GQA 的 KV Cache，再让字段缺失、head 数不可整除或 attention 语义变成 MLA，观察何时必须拒绝估算。随后比较 tokenizer、model config 和 generation config 的 BOS/EOS/PAD、长度与停止规则。

交付物：一张“配置可推导 / 必须实测 / 信息不足”的表。扩大 `max_position_embeddings` 不能证明有效长上下文。

### 实验 2C：真实 checkpoint 选修 { #lab-2c }

本地已有固定小模型 snapshot 时，可比较 prefill、cached decode、full recompute 和 `generate()` 的 token trace；也可只量化一层矩阵，观察局部误差如何传播到 logits。

交付物：模型 revision、tokenizer/template、输入 token IDs、数值容差和资源记录。单个 prompt、单层量化或 argmax 不变都不能证明整体质量、显存收益或加速。

## 实验 3：训练微型 GPT { #lab-3 }

在公开小语料训练一个可在本机完成的 decoder：

1. 固定 train/validation split、tokenizer 和上下文长度。
2. 记录参数量、训练 token、优化器、学习率和硬件。
3. 绘制 train/validation loss，不只保存最后一个数字。
4. 只改变层数、宽度或上下文中的一个变量。
5. 保存最差生成样例，并判断问题来自数据、优化还是解码。

选修 activation patching：预先固定 clean/corrupt pair、连续 metric 和 hook 位置，加入未来位置与随机来源负对照。热图或单样本高 recovery 不足以定位“事实存储层”。

## 实验 4：LoRA 领域适配 { #lab-4 }

第一次做微调时，先完成[实验 4A：追踪一个 SFT 样本](labs/lab-4a-sft-sample.md)。它在 CPU 上把 template、shifted labels、LoRA backward、adapter-only reload 和 held-out comparison 串成一条可观察路径。

选择一个可以自动评价的任务，例如分类、结构抽取或受限 SQL。比较同一评测集上的 base + Prompt、RAG（若适用）和 LoRA；固定 template 与 decoding。

实验顺序：

1. 检查 train/validation/test 的来源隔离、重复和模板泄漏。
2. 打印最终 token IDs、assistant labels 与截断位置。
3. 核对可训练参数、冻结基座和 optimizer step。
4. 比较训练曲线与 held-out 行为，不用同 batch loss 代替质量。
5. 独立重载 adapter，并检查通用能力与安全回归。

交付物：数据说明、基线、曲线、失败分类、资源使用和发布边界。详细实现见 [Single-GPU Finetuning](projects/single-gpu-finetuning.md)。

### 训练系统选修 { #lab-4-systems }

以下题目用于理解训练正确性，不要求基础路线全部完成：

| 题目 | 核心问题 |
|---|---|
| MinHash/LSH | 候选召回率和 exact recheck 怎样共同决定去重质量？ |
| Continual replay | 新任务质量与旧任务遗忘如何形成 Pareto 权衡？ |
| Checkpoint resume | model、optimizer、scheduler、RNG 和 data cursor 缺一项会怎样？ |
| Token-weighted accumulation | 不同有效 token 数的 micro-batch 应怎样归一化？ |
| DDP 与 `no_sync` | world-size 因子、同步边界和 global token count 在哪里进入？ |
| AMP | 为什么必须先 unscale 再 clip，overflow 时哪些状态不得前进？ |
| DataLoader prefetch | emitted、consumed 和 optimizer-committed cursor 为什么不同？ |

每题都应有 full-batch 或 uninterrupted 正对照，以及故意遗漏一个状态的负对照。具体脚本和参数在项目页维护，本页不固定录制数字。

## 实验 5：可诊断的 RAG { #lab-5 }

先完成[独立实验页：追踪一次 RAG 问答](labs/lab-5-rag-request.md)。
它用请求 A/B 串起授权、BM25、重排、packing、exact span、citation 与 non-empty retrieval 拒答。

完成 walkthrough 后，再把固定小语料替换成自己的 corpus：

建立一个小而可人工审阅的语料库：

1. 为问题标注 answer-bearing chunk，而不只标文档主题。
2. 先做 BM25，再加入 dense、hybrid 或 reranker。
3. 在任何正文读取和打分前执行 tenant/ACL 授权。
4. 加入无答案、冲突、过期、注入文本和跨权限案例。
5. 分别报告 retrieval、context、answer、citation 与 system 指标。

错误必须归入“语料无答案、解析错、召回漏、排序错、上下文丢失、生成越界、引用错误”之一。只报端到端总分无法指导修复。

### 实验 5A：框架公平对照 { #lab-5a }

在同一 corpus、query、授权主体和 top-k 下比较 canonical 实现、LangChain 与 LlamaIndex adapter。先预测每个主体能看到的文档，再检查无权正文是否曾进入 scorer、cache 或 callback，而不只比较最终 ID。

交付物：框架前后的 canonical request/result、排序差异和一个故意把 ACL 后移的安全负例。Adapter API 对齐不证明框架默认安全，也不证明 learned retrieval 质量。

### 实验 5B：服务与真实模型选修 { #lab-5b }

为 RAG API 增加 readiness、认证主体、schema、queue deadline 和后台工作取消。然后用一个固定小模型检查“有证据需引用、无证据需拒答”。保留第一次失败，不要调好 Prompt 后只展示成功版本。

交付物：typed terminal、generator 调用次数、public/audit 两种投影和失败样例。Loopback API 不等于 TLS/IAM；策略重放也不等于当时线上真实省掉了模型调用。

## 实验 6：安全的工具 Agent { #lab-6 }

先完成[独立实验页：追踪一次 Agent 退款](labs/lab-6-agent-lifecycle.md)。它用一笔“远端已受理、
本地响应超时”的退款，串起 proposal、closed schema、资源 ACL、审批、pending fence、provider verifier
和 reconciliation。完成后再把固定 Planner 换成模型或框架，不要先把控制状态藏进一个 `invoke()`。

交付物：九阶段状态图、字段信任边界、授权负例、effect receipt、恢复时间线和证据边界。

### 实验 6A：MCP 与 A2A 互操作 { #lab-6a }

按 memory transport → stdio → loopback HTTP 的顺序学习 MCP，再观察 A2A 的 discovery、message 和 task 状态。每种协议都先画消息序列，并分别记录 schema validation、应用 allowlist、transport、认证与业务授权发生在哪里。

交付物：请求/响应序列、生命周期错误、未知工具、取消和来源边界。官方 SDK + loopback 只能证明当前路径真实执行，不等于通过 conformance、TLS/OAuth、远程或跨厂商测试。

## 实验 7：量化与服务基准 { #lab-7 }

先完成[实验 7A：Paged KV 与 COW](labs/lab-7a-paged-kv.md)，用一条父子序列追踪真实 CPU K/V tensor；
再完成[实验 7B：Qwen3 穿过 nano-vLLM](labs/lab-7b-nano-vllm-qwen3.md)，把相同概念放进固定模型、
真实 scheduler 与 GPU runtime。然后用小型 oracle 理解 preemption 与量化，最后进入真实服务：

1. 固定模型、runtime、硬件和 prompt/output 长度分布。
2. 先验证单请求 token/usage/finish，再做 open-loop 多档负载。
3. 同时报告 success rate、queue、TTFT、TPOT、吞吐和资源。
4. 断开流式客户端，分别观察请求终止、底层停算和 KV 释放。
5. 比较 BF16/FP16、8-bit、4-bit 时联合检查质量与性能。

交付物：Paged KV 预测表、nano-vLLM 逐步 trace、workload contract、原始终态、容量曲线和故障记录。
CPU toy 的离散 step、逻辑 bytes 或本地取消不能冒充 GPU 性能、显存或远端计费证据。
详细入口见 [Inference Serving](projects/inference-serving.md)。

## 实验 8：评测与指标冲突 { #lab-8 }

构造少量能手算的反例，比较 literal exact、normalized exact、token F1、JSON schema/value 和 citation span：

1. 大小写 ID、JSON 空格和数组顺序是否应该视为等价？
2. schema 合法但字段值错误时，哪个指标必须失败？
3. 引用 span 存在但不支持 claim 时，句法与语义评价如何分开？
4. 多次查看结果、挑 slice 或比较多模型时，显著性怎样失真？

再用独立 held-out 集评价自己的系统，报告逐例结果和错误 taxonomy。七个 authored case 或一个较高分数都不能代表通用能力；指标首先要匹配任务 construct。

## 综合项目验收

一个合格项目应让别人回答：问题是什么、非 LLM 基线是什么、数据从哪里来、系统怎样失败、评价分母是什么、权限在哪里执行、服务如何降级、成本多少、怎样回滚。

最终报告至少包含：

- 问题定义、数据说明和架构图；
- 可运行基线与复现实验配置；
- 逐例结果、错误分类和负例；
- 安全边界、SLO、成本与已知限制；
- 下一项最可能推翻当前结论的实验。

项目目录可保存机器报告和可复现命令，但学习结论应写成普通人能够审阅的短文，而不是一串 hash 或“所有测试通过”。
