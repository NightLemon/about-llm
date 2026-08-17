# Qwen：从模型选择到中文工程落地

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要在中文、多语言、RAG、工具调用或单卡微调中使用 Qwen 的开发者。
- **先修**：理解 decoder-only Transformer、tokenizer、RAG 与 LoRA 的基本概念。
- **首次阅读**：对象识别 → checkpoint 检查 → 中文输入 → 应用路线 → 评测。
- **完成信号**：能固定一个 Qwen 对象，跑通最小基线，并说明结果不能外推到哪些版本或环境。
- **卡住时**：先读[模型选型](landscape.md)和[Tokenization](../core/tokenization.md)，只处理一个 text-only Instruct checkpoint。

</div>

**模型导航**：[模型全景](landscape.md) · [Transformers 项目](../practice/projects/transformers-basics.md) · [单卡微调](../practice/projects/single-gpu-finetuning.md) · [Qwen 证据台账](../evidence/qwen-controls.md)
{ .doc-nav }

Qwen 是一个不断扩展的模型与产品家族，不是一个固定架构。文本、代码、视觉、音频、dense、MoE、Base、Instruct、本地权重和云 API 可能共享品牌名，却拥有不同的输入、模板、运行时和许可边界。

学习 Qwen 的第一步不是记型号，而是把“我要运行什么”写成可验证对象。

## 先把需求写成一个对象

至少回答五个问题：

| 问题 | 常见选择 | 为什么重要 |
|---|---|---|
| 在哪里运行 | 本地开放权重 / 云 API | 身份、协议、费用与治理完全不同 |
| 处理什么模态 | 文本 / 代码 / 图像 / 音频 | tokenizer、processor 和模型头不同 |
| 需要什么行为 | Base / Instruct / reasoning | 模板、停止条件和后训练目标不同 |
| 使用什么结构 | dense / MoE | 总参数、激活参数、显存和通信口径不同 |
| 怎样交付 | Prompt / RAG / LoRA / 服务 | 数据、评测和回滚工件不同 |

例如“使用 Qwen 做客服”仍然不够。更可执行的描述是：

~~~text
本地 text-only Instruct checkpoint
+ 固定 immutable revision
+ Transformers runtime
+ 中文知识库 RAG
+ 单张消费级 GPU
+ 带引用与拒答的 held-out 评测
~~~

型号和上下文窗口会变化，因此本页讲检查方法，不维护“永久最新”榜单。

## 一次 checkpoint 检查

本仓库用固定的 Qwen2.5-0.5B-Instruct revision 演示检查流程。它只是教学样本，不能代表其他 Qwen 代际、尺寸、MoE、多模态或云端产品。

### 1. 固定发布身份

不要只保存 `Qwen2.5-0.5B-Instruct` 这个短名。至少记录：

- model ID 与完整 immutable revision；
- config、tokenizer、template、generation config 和权重文件清单；
- runtime、dtype/quantization、device 与 adapter；
- model card、许可审查和核对日期；
- evaluation manifest 与发布决策。

文件 SHA-256 可以确认已知 bytes 是否变化，但不能认证发布者，也不能证明这些文件已经成功前向。

### 2. 从 config 读取结构

对于标准 decoder checkpoint，先检查：

~~~text
model_type / architectures
hidden_size / intermediate_size / num_hidden_layers
num_attention_heads / num_key_value_heads
vocab_size / tied embeddings
position and RoPE fields
dense or MoE-specific fields
special token IDs
~~~

若 `hidden_size` 能被 query heads 整除，可得到候选 head dimension；若 K/V heads 更少，通常是 GQA/MQA。这个静态推导仍依赖实现采用标准 layout。

看到 MLA、专有 attention 或 remote code 时，应停止套用标准 KV 公式，转而检查同 revision 实现和真实 tensor shape。

### 3. 对账 tokenizer 与 template

中文字符常对应多个 token，也可能与前后空格、标点或 Unicode 形式共同切分。不要用“一个汉字约等于一个 token”估算容量。

至少保存：

- 原始文本与 Unicode normalization policy；
- tokenizer revision；
- chat template 渲染后的完整文本；
- 最终 token IDs；
- BOS/EOS/PAD 与 turn-end token；
- generation 时是否添加 assistant prompt。

Base 模型通常学习续写；Instruct 模型依赖特定对话模板。把普通字符串直接送给 Instruct checkpoint，程序可能运行，但任务接口已经变化。

### 4. 跑最小真实前向

最小 smoke 应使用本地已验证文件，执行：

1. 一次 prefill，检查 logits shape 和有限值；
2. 一次带 cache 的单 token decode；
3. 一次不带 cache 的同序列 forward；
4. 比较最后位置 logits；
5. 一次固定 generation，保存 token trace 与停止原因。

这能证明指定环境和输入的执行路径，不证明中文能力、有效长上下文、GPU 性能或生产稳定性。精确 revision、hash 和录制报告见[证据台账](../evidence/qwen-controls.md)。

## 架构怎样读

### Dense 与 MoE 不能共用参数口径

Dense 模型的每层通常激活全部 MLP 参数。MoE 每个 token 只路由到部分 experts，但设备仍需存放更多总权重，并承担 routing、capacity、通信和负载不均。

比较时同时报告：

- total parameters 与 active parameters；
- 每 token 激活 experts；
- 权重显存、KV Cache 与 activation；
- all-to-all 或 expert placement；
- 端到端吞吐和质量。

“总参数很大但激活参数较小”不自动表示单卡可加载，也不表示服务一定更快。

### GQA 主要改变 KV 口径

在标准 layout 下，理想 KV payload 约为：

\[
2 \times L \times B \times T \times H_{kv} \times d_{head} \times s
\]

其中 2 对应 K/V，`s` 是元素字节数。它不包含 allocator、block table、量化 scale、workspace、临时 tensor 或 prefix-cache metadata。

因此 config 公式用于容量预估，目标 runtime 的峰值显存才用于发布。

## 中文与多语言任务

中文能力不是一个总分。至少拆成：

| 能力 | 代表任务 | 常见失败 |
|---|---|---|
| 语言理解 | 分类、抽取、问答 | 否定、指代、长句关系 |
| 生成 | 摘要、改写、写作 | 事实漂移、重复、风格失控 |
| 知识 | 领域问答、时效事实 | 过期知识、无来源自信回答 |
| 结构 | JSON、表格、函数参数 | 字段遗漏、单位和 enum 错误 |
| 中英混合 | 代码、产品名、术语 | token 激增、实体边界错误 |
| 安全 | 拒答、隐私、越权 | over-refusal 或漏拦截 |

评测数据要按简繁体、地区表达、领域、长度和 code-switching 切片。中文总体平均不能掩盖繁体或专业领域退化。

## 四条工程路线

### RAG：先固定检索证据

Qwen 不会让 RAG 自动正确。先用 extractive 或模板化 baseline，确认 ACL、召回、rerank、packing、引用和无答案拒答，再让模型生成自然语言。

至少区分：

~~~text
没召回正确证据
→ 召回后被 rerank/packing 丢失
→ 模型看到了证据但回答错误
→ 答案正确但引用 span 错误
~~~

入口见 [RAG Foundations](../practice/projects/rag-foundations.md)。

### Tool calling：输出只是 proposal

模型生成合法 JSON 或 function call，不等于动作已获授权。执行层仍要验证 schema、主体、租户、资源、金额、审批、幂等和 effect receipt。

模板、工具 schema 和 runtime parser 必须一起版本化；云 API 的 tool contract 不能从本地 checkpoint 模板推断。

### LoRA/QLoRA：先检查 labels

微调前先比较 Prompt 与 RAG baseline，并打印最终 input IDs、labels 和监督 token 数。QLoRA 通常低比特存储冻结 base，在较高精度计算并训练 adapter；它不是“所有训练状态都是 4-bit”。

单卡实验按顺序增加复杂度：

1. 零下载 preflight 与数据审计；
2. tiny CPU batch 过拟合；
3. 目标 tokenizer 的 labels 检查；
4. 小规模 LoRA backward 与 adapter 重载；
5. 目标 GPU 的 QLoRA 显存测量；
6. held-out 质量与通用回归。

### 推理服务：模型与协议分开验收

模型 forward 正确，不证明 HTTP/SSE、取消、过载和计费正确；OpenAI-compatible 请求能解析，也不证明实际调用了目标权重。

服务 trace 至少绑定 request ID、model/revision、template、sampling、queue/prefill/decode 时间、usage 和 outcome。发布时同时测 TTFT、TPOT、吞吐、峰值显存、拒绝和取消。

## 怎样做模型评测

使用同一组 case 比较 baseline 与 candidate，并保留逐 case 输出。至少覆盖：

- 中文任务质量与关键切片；
- 格式、tool proposal 和业务 verifier；
- RAG 引用、拒答与越权负例；
- 通用能力与安全回归；
- offered-load 下的延迟、吞吐和失败；
- 每 attempted/successful task 的 token 与成本。

一次 generation、同 batch loss 下降或单矩阵压缩都只是局部机制证据。它们不能互相拼成“已经完成生产微调和部署”。

## 发布与回滚

发布工件至少包含：

- base checkpoint、tokenizer、template 和 adapter identity；
- quantization/runtime/container 与硬件；
- data/eval manifest 和完整分母；
- RAG index/tool/policy version；
- capability probe、容量结果和已知限制；
- canary 指标、回滚触发器与旧版本工件。

回滚不只切换 model ID。template、adapter、index、tool schema 和 parser 若不同，也要作为一个 bundle 恢复。

## 常见错误

- 用“Qwen”代替具体 checkpoint 或 API 产品。
- 把一个固定小模型的 config 外推到整个家族。
- 把 Instruct 模型当 Base 模型直接续写，或忽略 chat template。
- 只看中文平均分，不看领域、简繁体和中英混合切片。
- 把 4-bit 文件、单矩阵压缩或 CPU demo 写成整模型 GPU 结论。
- 把 JSON 可解析、tool call 或模型“完成”文本当作业务成功。
- 把多条共享 checkpoint 的实验拼成一条未实际执行的生产故事。

## 下一步怎样学

| 目标 | 建议入口 |
|---|---|
| 理解 tokenizer、attention、量化与真实权重 | [Transformers Basics](../practice/projects/transformers-basics.md) |
| 完成单卡 SFT/LoRA/DPO 路线 | [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md) |
| 建立中文权限感知 RAG | [RAG Foundations](../practice/projects/rag-foundations.md) |
| 部署并测量服务 | [Inference Serving](../practice/projects/inference-serving.md) |
| 核对仓库精确运行证据 | [Qwen 证据台账](../evidence/qwen-controls.md) |

## 自测

1. 为什么同一 Qwen 品牌下的本地 checkpoint 和云 API 必须分开建模？
2. 哪些条件满足时可以从 config 估算 KV payload？哪些情况必须拒绝？
3. 怎样证明 chat template 的 assistant 区域和训练 labels 对齐？
4. 一次 LoRA loss 下降后，还缺哪些证据才能决定发布？
5. 服务返回兼容格式时，怎样证明它加载了目标 revision？
