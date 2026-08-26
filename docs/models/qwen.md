# Qwen：跟一句中文请求穿过本地推理栈

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要在中文、多语言、RAG、工具调用或单卡微调中使用 Qwen 的开发者。
- **先修**：理解 decoder-only Transformer、tokenizer、RAG 与 LoRA 的基本概念。
- **首次阅读**：中文请求 → 模型与运行时分工 → checkpoint 检查 → 应用路线 → 评测。
- **完成信号**：能解释一条请求怎样变成 token、怎样经过 prefill 和 decode，并能为目标 checkpoint 选择匹配的运行时。
- **卡住时**：先读[模型选型](landscape.md)和[Tokenization](../core/tokenization.md)，只处理一个 text-only Instruct checkpoint。

</div>

**模型导航**：[模型全景](landscape.md) · [Transformers 项目](../practice/projects/transformers-basics.md) ·
[Qwen3 + nano-vLLM 实验](../practice/labs/lab-7b-nano-vllm-qwen3.md) ·
[单卡微调](../practice/projects/single-gpu-finetuning.md) · [Qwen 证据台账](../evidence/qwen-controls.md)
{ .doc-nav }

Qwen 是一个模型与产品家族，而不是某一种固定架构。纯文本、视觉语言和音频模型都可能使用这个名字，
但它们接收的输入并不相同。

稠密模型（dense）与混合专家模型（Mixture of Experts，MoE）对显存和运行时的要求也不同。
本地权重与云 API 更是两种不同的交付方式。

学习 Qwen 的第一步不是记型号，而是把“我要运行什么”写成可验证对象。

本页用两个具体对象讲方法。已有的 Qwen2.5-0.5B-Instruct 记录用来演示怎样检查 checkpoint；
Qwen3-0.6B 则用来追踪一次本地推理。前者的结果不会被当成后者的结果，Qwen3 实验也不会凭空写出
RTX 3070 Laptop 的速度和显存数字。

## 先跟一句中文穿过本地栈 {#local-request-stack}

先看一条普通请求：

```json
{
  "messages": [
    {"role": "user", "content": "请用一句话解释：为什么生成下一个 token 时可以复用 KV Cache？"}
  ],
  "max_tokens": 8
}
```

模型不能直接读取这个 JSON。请求会依次经过下面几步：

在真实程序中，应用适配层（adapter）会先调用 Qwen tokenizer 的 chat template，再把 `max_tokens` 转成
nano-vLLM 的采样参数。下面省略外围 HTTP 协议，只保留本地生成路径。

```text
messages
→ chat template 组成模型真正看到的文本
→ tokenizer 把文本切成 token ID
→ nano-vLLM 把请求放入 waiting 队列
→ Scheduler 分配本轮计算和 KV block
→ prefill 一次处理提示词
→ Qwen3 forward 计算下一个 token 的 logits
→ Sampler 选出 token
→ decode 每轮继续生成一个 token
→ 请求结束，释放 KV block
→ tokenizer.decode 还原可读文本
```

这条链路里最容易混淆的是“模型”和“推理框架”。Qwen checkpoint 提供配置、tokenizer 与权重；
nano-vLLM 负责请求状态、调度、缓存和执行。

第一次 forward 会处理完整提示词，这一阶段叫 **prefill（预填充）**。

之后每轮只输入最新 token，并从 KV Cache 读取历史注意力状态。这一阶段叫 **decode（解码）**。

请求刚加入时处于 `waiting`。调度器完成提示词调度后，请求进入 `running`。采样到结束 token 或达到长度上限后，
它进入 `finished`。状态变化属于运行时，不是 Qwen 权重自己完成的工作。

### 这条链路由谁完成

| 组件 | 在这次请求中负责什么 | 它不负责什么 |
|---|---|---|
| Qwen3 checkpoint | 提供模型配置、词表、对话模板和权重 | 排队、分配显存或实现 HTTP 服务 |
| Transformers | 读取 Qwen 配置与 tokenizer | 在本实验中执行模型生成 |
| nano-vLLM | 实现 Qwen3 模型类，并管理序列、调度、KV block 和采样 | 取代 checkpoint 中的权重与词表 |
| PyTorch | 提供张量、模块、CUDA 接口与显存分配器 | 决定请求先后顺序 |
| FlashAttention | 执行 prefill 与 decode 的注意力计算 | 管理序列生命周期 |
| Triton | 运行把本轮 K/V 写入缓存槽位的自定义 kernel | 实现完整的 Qwen 模型 |
| CUDA | 让上述张量和 kernel 在 NVIDIA GPU 上执行 | 决定 chat template 或采样参数 |
| xxhash | 为可复用的完整前缀块生成键 | 判断两个不同 token 前缀语义是否相近 |
| NCCL | 为张量并行准备进程间通信；本实验只用一张 GPU | 证明多卡路径已经运行 |

这张表也解释了为什么“模型支持某种架构”不等于“任意推理框架都能运行它”。运行时必须认识配置中的
模型类型，拥有对应的模型类，并能管理它需要的缓存状态和计算 kernel。

### 先把模板和 token IDs 真正跑出来

仓库提供了一条只运行 Qwen3 tokenizer 的命令。它不会加载模型权重，也不需要 GPU：

~~~powershell
python projects/transformers-basics/trace_qwen3_tokenizer.py --local-files-only
~~~

若模型在单独的 snapshot 目录中，使用 `--model-snapshot <path>`。如果固定版本还没有进入本地缓存，第一次运行时
去掉 `--local-files-only` 即可。脚本会把这条中文 message、chat template 渲染出的完整提示词、29 个输入 ID 和
每个 token 的可读片段放在一起。

Transformers 会用 `Qwen2TokenizerFast` 加载这个固定版本。Qwen3 复用了兼容的 tokenizer 实现，模型权重仍然
属于 Qwen3-0.6B。因此，不能只看 tokenizer 的 Python 类名判断模型家族。

脚本还会显示：`<think>` 和 `</think>` 是 added tokens，但不在当前 `all_special_ids` 中。模板控制词与
tokenizer 元数据中的 special token 是两个不同概念。

### 教材主线与可复现实验怎样对应

先在[实验 1B](../practice/labs.md#lab-1b)中编码上面的中文问题，观察真实模板和输入 ID。

[实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md)改用 768 个固定生成的 token ID，并只生成 8 个 token。
这组整齐的长度用来观察 256-token KV block、分块 prefill、前缀复用和逐轮 decode。

进入服务基准后，再把真实业务 Prompt 和长度分布接回去。

实验使用 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`，以及 nano-vLLM
commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。runner 会记录序列状态和缓存账本；真实耗时、吞吐与
显存只能由你的 3070 Laptop 实际运行后填写。

## 先把需求写成一个对象

至少回答五个问题：

| 问题 | 常见选择 | 为什么重要 |
|---|---|---|
| 在哪里运行 | 本地开放权重 / 云 API | 身份、协议、费用与治理完全不同 |
| 处理什么模态 | 文本 / 代码 / 图像 / 音频 | 分词器、输入处理器和模型头不同 |
| 需要什么行为 | 基础模型 / 指令模型 / 推理模型 | 模板、停止条件和后训练目标不同 |
| 使用什么结构 | 稠密模型 / MoE | 总参数、激活参数、显存和通信口径不同 |
| 怎样交付 | Prompt / RAG / LoRA / 服务 | 数据、评测和回滚工件不同 |

例如“使用 Qwen 做客服”仍然不够。更可执行的描述是：

~~~text
本地纯文本指令模型
+ 固定到完整 revision
+ 使用 Transformers 读取配置和 tokenizer
+ 中文知识库 RAG
+ 单张消费级 GPU
+ 带引用与拒答的留出集评测
~~~

型号和上下文窗口会变化，因此本页讲检查方法，不维护“永久最新”榜单。

## 一次 checkpoint 检查

本仓库用一个固定版本的 Qwen2.5-0.5B-Instruct 演示检查流程。这份记录只回答“这个 checkpoint
怎样核对”。其他尺寸、MoE、多模态模型和云端产品都需要重新检查。

### 1. 固定发布身份

不要只保存 `Qwen2.5-0.5B-Instruct` 这个短名。至少记录：

- 模型 ID 与完整 revision；
- 配置、tokenizer、对话模板、生成配置和权重文件清单；
- 运行时、数值类型、量化方式、设备与 adapter；
- 模型卡、许可审查和核对日期；
- 评测清单与发布决策。

文件的 SHA-256 摘要可以发现内容变化。它既不能认证发布者，也不能代替一次真实前向。

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

如果隐藏维度能被查询头数整除，就能算出候选的单头维度。如果 K/V 头更少，模型通常使用 GQA 或 MQA。
这些结论仍需与实现和真实张量形状对账。

遇到 MLA、专有注意力或远程自定义代码时，不要继续套用标准 KV 公式。此时应读取同一版本的实现，并检查
真实张量形状。

### 3. 对账 tokenizer 与 template

中文字符常对应多个 token，也可能与前后空格、标点或 Unicode 形式共同切分。不要用“一个汉字约等于一个 token”估算容量。

至少保存：

- 原始文本与 Unicode 规范化规则；
- tokenizer 的版本；
- chat template 渲染后的完整文本；
- 最终 token ID；
- BOS/EOS/PAD 与 turn-end token；
- 生成前是否添加 assistant 起始标记。

Base 模型通常学习续写；Instruct 模型依赖特定对话模板。把普通字符串直接送给 Instruct checkpoint，程序可能运行，但任务接口已经变化。

### 4. 跑最小真实前向

最小冒烟检查应使用本地已经核对过的文件，并执行：

1. 做一次 prefill，检查 logits 的形状与有限值；
2. 带 KV Cache 生成一个 token；
3. 不使用缓存，对同一序列重新做 forward；
4. 比较两条路径最后位置的 logits；
5. 固定生成参数，保存 token 轨迹与停止原因。

这次运行说明指定环境能够加载这些权重，并按给定输入完成前向和生成。中文能力、有效长上下文、GPU 性能和
生产稳定性需要各自的任务与运行环境。具体版本、文件摘要和录制报告见[证据台账](../evidence/qwen-controls.md)。

## 架构怎样读

不要按型号代际背架构。拿到一个 checkpoint 后，按下面四步阅读：

1. **先看输入**：纯文本模型使用 tokenizer；多模态模型还需要处理图像或音频的 processor。
2. **再看主体**：确认它使用标准注意力、MoE、递归状态、线性注意力，还是几种结构的混合。
3. **找出跨步状态**：decode 时究竟要保存 K/V、递归状态，还是两者都要保存。
4. **映射到运行时**：检查框架是否有对应模型类，能否管理这些状态，并具备所需的调度器和 kernel。

完整方法和兼容性阶梯见[从架构推导运行时依赖](../core/architectures-interpretability.md#architecture-runtime-dependencies)。

以本页的 Qwen3-0.6B 为例，配置中的模型类型是 `qwen3`，入口类是 `Qwen3ForCausalLM`。

它是纯文本稠密解码器。主要部件包括 RMSNorm、RoPE、GQA、门控 SiLU MLP 和语言模型输出头。
nano-vLLM 的固定版本实现了这些部件，因而能够加载该 checkpoint 的权重。

本仓库的 Qwen2.5 验证程序同样只运行纯文本解码器。遇到多模态模型、MoE、混合状态或额外预测头时，
必须重新选择模型类和运行库，不能只替换模型名称。

### Dense 与 MoE 不能共用参数口径

稠密模型的每层通常激活全部 MLP 参数。MoE 会把每个 token 路由给少数专家，但设备仍要存放更多总权重。
它还会引入路由、容量限制、通信和负载不均等问题。

比较时同时报告：

- 总参数量与每个 token 的激活参数量；
- 每个 token 激活几个专家；
- 权重、KV Cache 与中间激活各占多少显存；
- 是否需要全互连通信，以及专家怎样放置；
- 端到端吞吐和质量。

“总参数很大但激活参数较小”不自动表示单卡可加载，也不表示服务一定更快。

### GQA 主要改变 KV 口径

在标准张量布局下，KV 数据的理想大小约为：

\[
2 \times L \times B \times T \times H_{kv} \times d_{head} \times s
\]

其中 2 代表 K 和 V，`s` 是每个元素占用的字节数。这个公式只计算 KV 张量本身。显存分配器、block table、
量化 scale、临时工作区和前缀缓存元数据都不在其中。

因此，配置公式适合做第一轮容量预估；发布时仍要测目标运行时的峰值显存。

## 中文与多语言任务

中文能力不是一个总分。至少拆成：

| 能力 | 代表任务 | 常见失败 |
|---|---|---|
| 语言理解 | 分类、抽取、问答 | 否定、指代、长句关系 |
| 生成 | 摘要、改写、写作 | 事实漂移、重复、风格失控 |
| 知识 | 领域问答、时效事实 | 过期知识、无来源自信回答 |
| 结构 | JSON、表格、函数参数 | 字段遗漏、单位和枚举值错误 |
| 中英混合 | 代码、产品名、术语 | token 激增、实体边界错误 |
| 安全 | 拒答、隐私、越权 | 过度拒答或漏拦截 |

评测数据要按简繁体、地区表达、领域、长度和中英混合方式切片。总体平均分不能掩盖繁体中文或专业领域的退化。

## 四条工程路线

### RAG：先固定检索证据

换成 Qwen 不会让 RAG 自动变正确。先用摘录式或模板化基线，逐项确认权限过滤、召回、重排和上下文装配。
引用与无答案拒答都可靠以后，再让模型生成自然语言。

至少区分：

~~~text
没召回正确证据
→ 召回后被重排或上下文装配丢失
→ 模型看到了证据但回答错误
→ 答案正确但引用位置错误
~~~

入口见 [RAG Foundations](../practice/projects/rag-foundations.md)。

### 工具调用：输出只是动作建议

模型生成合法 JSON 或 function call，只说明它提出了一个结构化动作。执行层仍要检查字段结构、调用者身份、
租户、资源、金额、审批和幂等键。动作执行后，还要保存可核对的结果回执。

对话模板、工具 Schema 和运行时解析器必须一起版本化。云 API 的工具协议要以云端文档为准，不能从本地
checkpoint 的模板反推。

### LoRA/QLoRA：先检查 labels

微调前先比较 Prompt 与 RAG 基线，并打印最终输入、labels 和参与监督的 token 数。QLoRA 通常以低比特形式
存放冻结的底座权重，再用较高精度计算并训练 adapter。它不表示所有训练状态都是 4-bit。

单卡实验按顺序增加复杂度：

1. 运行不下载模型的数据预检；
2. 在 CPU 上让一个微小 batch 过拟合；
3. 目标 tokenizer 的 labels 检查；
4. 小规模 LoRA 反向传播与 adapter 重载；
5. 目标 GPU 的 QLoRA 显存测量；
6. 留出集质量与通用能力回归。

### 推理服务：模型与协议分开验收

模型 forward 正确，只说明模型计算路径通过。HTTP、流式传输、取消、过载和计费仍要分别检查。
同样，兼容 OpenAI 格式的请求能够解析，也不代表服务实际加载了目标权重。

服务轨迹至少要绑定请求 ID、模型版本、对话模板和采样参数。排队、prefill 与 decode 的时间也要分开记录，
同时保存 token 用量和请求终态。

发布测试再关注用户真正感受到的结果：首 token 延迟、后续 token 延迟、吞吐、峰值显存、拒绝和取消。

本章开头只追踪了引擎内部的一条请求。[实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md)
会继续比较 eager 与 CUDA Graph、完整前缀与单 token 漂移，以及两种 prefill 预算。它帮助你解释性能变化来自
哪段执行机制。HTTP 服务的容量与可靠性则要到[推理服务项目](../practice/projects/inference-serving.md)中继续验证。

## 怎样做模型评测

使用同一组样例比较基线与候选模型，并保留每条样例的输出。至少覆盖：

- 中文任务质量与关键切片；
- 格式、工具动作建议和业务结果检查；
- RAG 引用、拒答与越权负例；
- 通用能力与安全回归；
- 固定到达负载下的延迟、吞吐和失败；
- 每次尝试和每次成功任务的 token 与成本。

一次生成、一个 batch 的 loss 下降或单个矩阵压缩，都只能说明局部机制已经运行。不能把这些结果拼成
“已经完成生产微调和部署”。

## 发布与回滚

发布工件至少包含：

- 底座 checkpoint、tokenizer、对话模板和 adapter 版本；
- 量化方式、运行时、容器与硬件；
- 数据和评测清单，以及没有被过滤掉的完整分母；
- RAG 索引、工具和策略版本；
- 能力探针、容量结果和已知限制；
- 灰度指标、回滚触发器与旧版本工件。

回滚不能只切换模型 ID。如果对话模板、adapter、索引、工具 Schema 或解析器也发生了变化，就要把它们
作为同一个发布包恢复。

## 常见错误

- 用“Qwen”代替具体 checkpoint 或 API 产品。
- 把一个固定小模型的 config 外推到整个家族。
- 把 Instruct 模型当 Base 模型直接续写，或忽略 chat template。
- 只看中文平均分，不看领域、简繁体和中英混合切片。
- 把 4-bit 文件、单矩阵压缩或 CPU demo 写成整模型 GPU 结论。
- 把 JSON 可解析、工具调用或模型生成的“完成”文本当作业务成功。
- 把多条共享 checkpoint 的实验拼成一条未实际执行的生产故事。

## 下一步怎样学

| 目标 | 建议入口 |
|---|---|
| 观察中文 message 怎样变成 Qwen3 输入 IDs | [实验 1B](../practice/labs.md#lab-1b) |
| 理解 tokenizer、attention、量化与真实权重 | [Transformers Basics](../practice/projects/transformers-basics.md) |
| 完成单卡 SFT/LoRA/DPO 路线 | [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md) |
| 建立中文权限感知 RAG | [RAG Foundations](../practice/projects/rag-foundations.md) |
| 部署并测量服务 | [Inference Serving](../practice/projects/inference-serving.md) |
| 追踪 Qwen3 在 nano-vLLM 中的真实执行 | [实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md) |
| 核对仓库精确运行证据 | [Qwen 证据台账](../evidence/qwen-controls.md) |

## 自测

1. 为什么同一 Qwen 品牌下的本地 checkpoint 和云 API 必须分开建模？
2. 哪些条件满足时可以从 config 估算 KV payload？哪些情况必须拒绝？
3. 怎样证明 chat template 的 assistant 区域和训练 labels 对齐？
4. 一次 LoRA loss 下降后，还缺哪些证据才能决定发布？
5. 为什么 Qwen checkpoint、Transformers 和 nano-vLLM 在一次请求中承担不同职责？
6. 服务返回兼容格式时，怎样证明它加载了目标 revision？
