# DeepSeek：把 MoE、MLA 与推理模型分开学

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解 DeepSeek 技术路线，并在 API 或单卡环境中做可靠实验的开发者和算法工程师。
- **先修**：理解 decoder-only Transformer、KV Cache、SFT 和基础推理评测。
- **首次阅读**：对象识别 → 三条技术线 → checkpoint 检查 → 单卡/API 路线 → 评测。
- **完成信号**：能解释 MoE、MLA 和 reasoning post-training 的边界，并为一个具体对象设计实验。
- **卡住时**：先复习 [Transformer](../core/transformer.md)，再只选择一个小型 distill checkpoint 做检查。

</div>

**模型导航**：[模型全景](landscape.md) · [前沿专题](../frontier/reasoning-long-context-moe.md) · [单卡微调](../practice/projects/single-gpu-finetuning.md) · [DeepSeek 证据台账](../evidence/deepseek-controls.md)
{ .doc-nav }

DeepSeek 最容易学乱，是因为一个名字同时覆盖研究路线、开放 checkpoint、蒸馏模型和云 API。

这些对象可能共享训练数据或行为来源，却不一定共享基础架构、权重、上下文协议和运行时。先分清对象，再讨论技术，能避开大多数错误。

## 先判断你面对的是什么

| 对象 | 你真正拿到的东西 | 首要检查 |
|---|---|---|
| V2/V3 技术报告 | 对 MLA、MoE 和训练系统的公开描述 | 公式、实验设置与适用版本 |
| 开放 checkpoint | config、tokenizer、代码和 weight shards | immutable revision 与真实架构 |
| R1 系列 | 经过推理后训练的模型发布 | base、训练阶段与生成协议 |
| R1 Distill | 用推理数据训练的 Qwen/Llama 系学生 | 学生自己的 config，而非 teacher 架构 |
| 云 API | provider 暴露的模型与协议 | model ID、接口契约、计费与观测 |

“我要部署 DeepSeek”还不是可执行需求。一个可验证的对象应写成：

~~~text
DeepSeek-R1-Distill-* 的具体 model ID
+ immutable revision
+ Transformers/vLLM 的具体版本
+ tokenizer 与 chat template
+ dtype 或量化方案
+ 单张目标 GPU
+ 固定任务集和生成预算
~~~

若使用云 API，则把 revision、权重和本地 runtime 换成 provider、model ID、catalog 日期、endpoint 与请求协议。两条路线的证据不能互相代替。

## 用三条技术线建立地图

DeepSeek 的代表性工作可以先拆成三条线：

1. **MoE** 关注怎样扩大模型容量，同时让每个 token 只激活部分 experts。
2. **MLA** 关注怎样改变注意力中的 K/V 表示与缓存口径。
3. **Reasoning post-training** 关注怎样通过 SFT、强化学习、可验证奖励和蒸馏塑造推理行为。

它们解决的是不同问题。MoE 不自动产生推理能力，MLA 不是量化，推理后训练也不会把学生模型的 attention 换成 teacher 的结构。

~~~mermaid
flowchart LR
    A["模型容量与计算"] --> B["MoE"]
    C["解码缓存与带宽"] --> D["MLA"]
    E["输出行为与搜索"] --> F["Reasoning post-training"]
    B --> G["共同出现在一条模型路线中"]
    D --> G
    F --> G
~~~

## MoE：少算一部分，不等于少存一部分

Dense Transformer 的每个 token 通常经过同一套 MLP。Mixture of Experts（MoE）把 MLP 替换为多个 experts，并让 router 为每个 token 选择少数 experts。

可以把一次路由理解成四步：

1. router 为 token 计算 expert scores；
2. 选择 top-k experts，并得到 gate weights；
3. dispatch token 到对应 expert；
4. 合并 expert 输出。

如果有 \(E\) 个 routed experts、每个 token 选择 \(k\) 个，那么 token 只执行一部分 expert 计算。但设备或集群仍可能需要保存全部 expert 权重。

因此至少要分开三本账：

- **总参数**：加载、存储和分片要容纳多少权重；
- **激活参数**：单 token 实际走过多少参数；
- **系统成本**：routing、capacity、通信、负载不均和 kernel 效率。

### 为什么路由是系统问题

某个 expert 突然收到过多 token 时，会出现排队、丢弃、reroute 或显存压力。多 GPU expert parallel 还要把 token 发给持有目标 expert 的设备，再把输出送回来源。

所以“激活参数较少”不能推出“服务一定更快”。验收 MoE 服务时要观察：

- 每个 expert 的 selected、accepted 和 dropped token；
- load balance、capacity policy 与 overflow；
- all-to-all payload 和跨设备等待；
- prefill/decode 的吞吐、尾延迟和峰值显存。

DeepSeek-V3 固定 config 中的 expert 数、top-k 和 routing 字段只是静态 markers。真实选择顺序、梯度、通信和 capacity 行为必须由同 revision 代码与 runtime execution 证明。

## MLA：缓存的是 latent，不再是标准 K/V 账本

自回归解码会复用历史 token 的注意力状态。标准 MHA/GQA 的理想 KV payload 常写成：

\[
2 \times L \times B \times T \times H_{kv} \times d_{head} \times s
\]

这里的 2 对应 K/V，\(L\) 是层数，\(T\) 是缓存长度，\(s\) 是每个元素的字节数。

Multi-head Latent Attention（MLA）的关键直觉是：先把与 K/V 有关的信息压到较低维 latent，缓存 latent 与必要的位置分量，再通过投影参与注意力计算。

若把压缩表示记为 \(c_t^{KV}\)，可用简化关系理解：

\[
c_t^{KV}=W^{DKV}h_t, \qquad
k_{t,i}^{C}=W_i^{UK}c_t^{KV}, \qquad
v_{t,i}^{C}=W_i^{UV}c_t^{KV}.
\]

这不是在标准 K/V tensor 上简单减少 head 数，而是改变了缓存对象。部分 projection 还可吸收到 query 或输出侧计算中，避免在每一步完整展开历史 K/V。

### 为什么这里不能套用标准 KV 公式

看到 config 同时存在 query/KV head 数，并不证明它采用标准 MHA layout。若同一 checkpoint 还声明 latent rank、独立的 non-RoPE/RoPE 维度或自定义 attention，实现可能拥有完全不同的 cache tensor。

因此 MLA 的容量估算应遵循：

1. 固定 config 与实现代码；
2. 检查 runtime 实际返回的 cache 结构；
3. 记录每个 tensor 的 shape、dtype 和随 token 增长量；
4. 再加入 page、block table、scale、workspace 和 allocator 开销；
5. 用目标硬件上的峰值显存校准。

本仓库看到已知 MLA markers 时会停止计算，并提示标准 KV 公式不适用。这样可以避免得到一个形式精确、
实际口径错误的数字。触发这一判断的字段和验证样例见[证据台账](../evidence/deepseek-controls.md)。

## R1：训练行为与基础架构是两层

R1 路线应从“怎样得到推理行为”来理解，而不是只观察输出中是否出现长思维文本。

一个抽象流程可能包含：

~~~text
base model
→ cold-start / SFT data
→ sampled reasoning trajectories
→ verifiable or learned rewards
→ policy optimization
→ filtering and distillation
~~~

可验证奖励适合有确定判据的任务，例如最终答案、测试用例或格式约束。它降低了部分 reward model 误差，但仍可能奖励投机格式、错误 verifier 或数据泄漏。

Group-relative 方法的直觉是：对同一问题采样一组答案，用组内相对结果构造 advantage，再更新策略。具体 reward、normalization、clip、KL 和 token mask 必须以固定版本报告或实现为准。

### 蒸馏复制行为，不复制内部结构

R1 Distill 学生可能基于 Qwen 或 Llama。它可以学习 teacher 生成的数据与推理风格，但不会因此自动获得 DeepSeek-V3 的 MLA、MoE、FP8 kernel 或训练轨迹。

部署和微调学生时，应完全按学生 config 决定：

- attention 与 KV Cache 公式；
- LoRA target modules；
- 权重显存和量化兼容性；
- chat template 与停止条件；
- runtime 支持矩阵。

“输出看起来像 teacher”只是一种行为观察，不是架构复制证据。

## FP8 与 MTP：字段不等于加速

DeepSeek-V3 路线还常与 FP8 和 Multi-Token Prediction（MTP）一起讨论。

| 机制 | 想解决的问题 | 仅凭 config 不能证明 |
|---|---|---|
| FP8 | 降低部分训练/推理计算和存储成本 | kernel 覆盖、误差、峰值显存与 speedup |
| MTP | 增加多步预测训练信号，可能支持候选生成 | 服务端使用该 head 或获得解码加速 |

FP8 的效果取决于格式、scale granularity、累加 dtype、异常值处理、硬件和 fallback。MTP head 出现在训练结构中，也不代表当前服务路径会加载或调用它。

工程报告应分别记录 artifact、kernel coverage、数值误差、质量、显存和端到端性能，不能把一个字段直接翻译成“已加速”。

## Config 能证明到哪里

把证据分层，能阻止静态观察被写成运行结论：

| 层级 | 证据 | 合理结论 |
|---|---|---|
| L1 | 固定技术报告或发布说明 | 该来源公开主张了什么 |
| L2 | 固定 config/tokenizer/code bytes | 字段、模板和静态结构候选 |
| L3 | weight inventory 与成功加载 | 指定权重已完整读取 |
| L4 | 目标 runtime execution | forward、cache、routing 或 kernel 实际行为 |
| L5 | 目标任务和硬件评测 | 质量、性能、容量与 SLO |

config 中出现 auto_map 只表示声明了自定义模块映射。若没有固定并审阅 remote code，不能声称该执行路径安全或可复现。

本仓库目前只核对了指定 DeepSeek-V3 config 的架构字段，尚未运行对应权重。通用 MoE 样例只能解释机制，
不能代替这个 checkpoint 的前向执行。具体 revision、hash 和已经运行的检查见
[证据台账](../evidence/deepseek-controls.md)。

## 单卡与云 API 怎样选择

在单张消费级 GPU 上，完整 V3 级 MoE 权重通常不是合适的入门对象。更稳妥的学习路线是选择资源可承受的具体 Distill checkpoint，或使用云 API 研究模型行为。

### 单卡路线

下载较大文件以前，先运行一次预检；缺少必要条件时直接停止：

1. 固定 model revision，检查 weight shard 总字节数；
2. 读取学生 config，确认 dense/MoE 与 MHA/GQA/MLA；
3. 估算权重、KV、activation、workspace 和量化 metadata；
4. 验证 tokenizer、chat template 与 EOS；
5. 确认 runtime、量化格式和目标 GPU 的兼容性。

先跑小输入 smoke，再逐步增加上下文和并发。不要用文件大小直接代替运行峰值，也不要为填表而给未知自定义架构套标准公式。

### 云 API 路线

把 API 当作一个受版本化协议约束的远程系统，而不是本地 checkpoint 的网络入口。保存：

- provider 与精确 model ID；
- 请求中的 sampling、max tokens、tools 和 reasoning 相关字段；
- 响应文本、usage、finish reason、错误与重试；
- catalog 核对日期、价格快照与数据治理设置；
- 业务 verifier 和完整分母。

兼容 OpenAI-style schema 不证明服务端使用某个开放权重 revision。reasoning 字段也应按 provider contract 处理，不能据此推断隐藏训练过程。

## 一个可运行的学习实验

选择一个资源可承受的 Distill checkpoint，按同一 manifest 完成四轮实验：

1. **身份检查**：保存 revision、config、tokenizer、template、权重清单和环境。
2. **执行检查**：运行 prefill、带 cache decode 和固定 generation，保存 token trace。
3. **行为评测**：比较 direct answer、reasoning prompt 和固定预算采样。
4. **服务检查**：在 Transformers 与目标 serving runtime 中对齐输入、停止和输出。

每轮开始前先写预测。例如，增加采样预算通常会提高“候选中至少有一个正确答案”的比例，但 verifier
未必更容易选中它。运行后逐个 case 查看候选和选择结果，而不只看平均分。

实现入口：

- [Transformers Basics](../practice/projects/transformers-basics.md)：config、tokenizer、真实权重与 generation；
- [Evaluation Gate](../practice/projects/evaluation-gate.md)：逐 case 评测与发布判定；
- [Inference Serving](../practice/projects/inference-serving.md)：协议、流式、取消和 offered-load；
- [Single-GPU Finetuning](../practice/projects/single-gpu-finetuning.md)：学生 checkpoint 的 SFT/LoRA/DPO。

## Test-time compute 要固定预算

推理模型常通过多采样、self-consistency、verifier selection 或工具调用增加 test-time compute。比较两个系统时，必须固定或同时报告：

- prompt 与模板；
- sampled token、候选数 \(k\) 和最大轮数；
- temperature、top-p 与 seed policy；
- verifier、工具、超时和重试；
- attempted 与 successful case 的成本和延迟。

pass@k 回答“候选中是否至少有一个正确答案”；self-consistency 和 verifier selection 回答“系统最终能否
选出正确候选”。一个衡量候选覆盖，另一个衡量选择能力。

若候选数增加但 verifier 很弱，成本会增加，最终答案却可能不变甚至变差。输出更长只说明使用了更多 token，
推理能力是否提高仍要看最终任务结果。

## 常见错误

- 把 DeepSeek 品牌下所有 checkpoint 都写成 MLA + MoE。
- 把 Distill 学生的推理行为当作 teacher 架构复制。
- 看到标准 head 字段就对 MLA checkpoint 套 MHA/GQA KV 公式。
- 从 expert config 直接推断真实 routing、active FLOPs 或通信效率。
- 把 FP8/MTP 字段写成已经执行的 kernel、显存收益或解码加速。
- 只比较最终答案，不固定 token、候选、工具和 verifier 预算。
- 用 API 兼容格式推断 provider 内部权重和训练实现。
- 把通用 MoE 或 RL 教学样例写成 DeepSeek checkpoint 复现。

## 面试时怎样回答

面对“介绍 DeepSeek 的关键技术”，可以用四步回答：

1. 先声明具体研究版本或 checkpoint，避免品牌级泛化。
2. 分别解释 MoE 的条件计算、MLA 的 cache 变化和 R1 的行为后训练。
3. 给出一个工程后果，例如 expert parallel 通信或 MLA 公式拒绝。
4. 说明验证方法：config 只到静态证据，runtime trace 和目标评测才回答执行与效果。

继续追问时，应能说明：

- total parameters、active parameters 与服务成本为何不同；
- MLA 为什么不能只看 num_key_value_heads；
- Distill 为什么按学生 base 选择 LoRA modules；
- pass@k、self-consistency 与 verifier selection 分别测什么；
- FP8 config、FP8 artifact、FP8 kernel 和 speedup 如何分层。

## 自测

1. 为什么同名的研究报告、开放 checkpoint、Distill 和云 API 不能共用一份架构结论？
2. MoE 每 token 激活更少参数，为什么单卡加载和分布式服务仍可能更难？
3. MLA 改变了什么缓存对象？你会怎样在目标 runtime 中测量它？
4. R1 Distill 学生能继承哪些东西，又不能据此声称继承了什么？
5. 怎样公平比较 direct answer、self-consistency 和 verifier selection？

## 一手资料入口

- DeepSeek-AI，[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)。
- DeepSeek-AI，[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)。
- DeepSeek-AI，[DeepSeek-V2](https://arxiv.org/abs/2405.04434)。
- 具体 config、revision 与本仓库运行过的检查见[DeepSeek 证据台账](../evidence/deepseek-controls.md)。
