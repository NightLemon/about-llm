# DeepSeek：跟一枚 token 看懂 MLA、MoE 与推理训练

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解 DeepSeek 技术路线，并在 API 或单卡环境中做可靠实验的开发者和算法工程师。
- **先修**：理解 decoder-only Transformer、KV Cache、SFT 和基础推理评测。
- **首次阅读**：一枚 token 的前向路径 → MLA → MoE → R1/MTP/FP8 → 运行时 → 单卡学习路线。
- **完成信号**：能顺着一次前向解释 MLA 与 MoE 的位置，并能区分模型架构、训练方法和服务实现。
- **卡住时**：先复习 [Transformer](../core/transformer.md)，再回到“从一枚 token 开始”逐步画出张量流。

</div>

**模型导航**：[模型全景](landscape.md) · [前沿专题](../frontier/reasoning-long-context-moe.md) ·
[单卡微调](../practice/projects/single-gpu-finetuning.md) · [DeepSeek 证据台账](../evidence/deepseek-controls.md)
{ .doc-nav }

DeepSeek 不是一种固定的模型结构。这个名字可以指技术报告、开放权重、R1 推理模型、
基于 Qwen 或 Llama 训练的 Distill 学生，也可以指云 API。它们可能有关联，却不一定使用相同的
attention、专家层、tokenizer 或服务协议。

本页主要用 DeepSeek-V3 的公开架构讲清 MLA 与 MoE，再说明 R1 后训练怎样改变模型行为。
为了避免把讲解写成“已经复现”，先说明本仓库实际检查过什么：

- 仓库保存了 `deepseek-ai/DeepSeek-V3` revision
  `e815299b0bcbac849fa540c768ef21845365c9eb` 的 `config.json`；
- 文件大小为 1,660 bytes，SHA-256 为
  `cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9`；
- 检查结果能确认这份配置出现了 MLA、MoE、FP8、YaRN 和 MTP 相关字段；
- 仓库没有下载对应权重，也没有运行它声明的自定义模型代码、前向计算或 GPU kernel。

因此，下面的前向路径来自公开架构与固定配置的联合解读。它帮助你理解模型应当怎样工作，
但不是本仓库已经录制的 DeepSeek-V3 运行轨迹。精确证据见[DeepSeek 证据台账](../evidence/deepseek-controls.md)。

## 从一枚 token 开始 {#one-token-forward}

假设模型正在生成回答，当前 token 在某一层的隐藏状态是 \(h_t\)。根据固定配置，前 3 层使用稠密
前馈网络；这里选择后面的一层，观察它怎样依次经过 MLA attention 和 MoE 前馈网络。

```text
h_t
→ RMSNorm
→ MLA 构造 query 与压缩后的 K/V 表示
→ 读取并更新历史 attention 状态
→ attention 输出 + 残差 + RMSNorm
├─ 共享专家 ──────────┐
└─ router → 8 个专家 ─┴→ 合并输出 + 残差
→ 送入下一层
→ 最后一层之后得到 logits
→ 运行时采样下一个 token
```

这条路径先给出三个关键结论：

1. **MLA 在 attention 内部工作**，它让历史 token 的 K/V 信息可以用压缩表示参与计算；
   运行时是否真的缓存这种表示，还要看具体实现。
2. **MoE 替换后半层的前馈网络**，它让当前 token 只执行一部分路由专家，同时保留共享专家路径。
3. **R1 不是图中的某一层**。R1 所代表的后训练过程会改变参数和输出行为，不会在推理时额外插入一个
   “推理模块”。

下面把这枚 token 的每一步展开。

## 第一步：MLA 把 K/V 信息压到潜在表示

标准多头注意力（Multi-Head Attention，MHA）会在每一层为历史 token 保存 key 和 value。

分组查询注意力（Grouped-Query Attention，GQA）让多组查询头共享较少的 K/V 头，
但缓存对象仍然是标准 K/V。

Multi-head Latent Attention（MLA）的思路不同：先把与 K/V 有关的信息压缩成更小的潜在表示
（latent representation），再由投影恢复 attention 计算需要的分量。用简化符号可以写成：

\[
c_t^{KV}=W^{DKV}h_t, \qquad
k_{t,i}^{C}=W_i^{UK}c_t^{KV}, \qquad
v_{t,i}^{C}=W_i^{UV}c_t^{KV}.
\]

\(c_t^{KV}\) 是压缩后的表示，\(W^{DKV}\) 完成下投影。\(W_i^{UK}\) 和 \(W_i^{UV}\) 为第 \(i\) 个
注意力头恢复内容相关的 K/V 分量，位置相关分量则单独经过 RoPE。

面向高效推理的实现还可以利用矩阵吸收等价变换，避免在每轮解码时为全部历史 token 完整展开 K/V。

### 用固定配置读懂张量角色

DeepSeek-V3 的固定配置给出了这些字段：

| 字段 | 值 | 它提示了什么 |
|---|---:|---|
| `num_attention_heads` | 128 | attention 有 128 个 query heads |
| `q_lora_rank` | 1536 | query 路径包含低秩压缩 |
| `kv_lora_rank` | 512 | K/V 内容路径使用 512 维潜在表示 |
| `qk_nope_head_dim` | 128 | 每个 head 中不使用 RoPE 的 Q/K 分量维度 |
| `qk_rope_head_dim` | 64 | 每个 head 中使用 RoPE 的 Q/K 分量维度 |
| `v_head_dim` | 128 | 每个 value head 的维度 |

这张表也揭示了一个常见错误：`hidden_size=7168` 除以 128 得到 56，但 56 不是这里的 Q/K head
维度。配置已经把 Q/K 拆成 128 维内容分量和 64 维位置分量。只沿用标准 Transformer 的
`hidden_size / num_attention_heads` 经验公式，会在这里得到错误结论。

### prefill 与 decode 分别发生什么

在 **prefill（预填充）** 阶段，模型并行处理提示词 token，为每一层计算 attention 状态并写入缓存。
进入 **decode（逐 token 解码）** 后，每轮只处理最新 token，同时读取此前保存的状态。

对于标准 MHA/GQA，理想化的 K/V 张量大小常写成：

\[
2 \times L \times B \times T \times H_{kv} \times d_{head} \times s,
\]

其中 2 对应 K 和 V，\(L\) 是层数，\(B\) 是批量，\(T\) 是缓存长度，\(s\) 是每个元素的字节数。
MLA 的数学结构允许运行时保存压缩表示和必要的位置分量，因此不能直接沿用这本标准账。

这里必须区分**架构提供的可能性**和**运行时真正采用的布局**。一个兼容实现可以先把潜在表示展开成
完整 K/V，再接入通用缓存接口；它也可能专门保存潜在表示，并把部分投影吸收到查询或输出计算中。
两种实现都可能成功加载同一个 checkpoint，但缓存占用和解码带宽会明显不同。

固定配置让我们知道 `kv_lora_rank=512`、位置分量为 64 维，却仍不足以给出真实显存数字。
要得到容量结果，还需要在目标运行时检查：

1. 实际缓存了哪些 tensor，以及各自的 shape 和 dtype；
2. 每增加一个 token，哪些 tensor 随之增长；
3. page、block table、量化 scale、workspace 和分配器额外占用多少内存；
4. prefix cache、beam、speculative decoding 和请求取消怎样改变生命周期。

因此，本仓库的配置检查器看到 MLA 字段后，会明确停止标准 K/V 估算。停止计算不是“缺少一个公式”，
而是在提醒读者：缓存对象已经变了，应先读取目标实现和运行时张量。

## 第二步：MoE 为当前 token 选择专家

注意力输出与残差相加后，token 进入本层的前馈网络。稠密 Transformer 会让所有 token 经过同一套 MLP。

混合专家（Mixture of Experts，MoE）准备多套 MLP，再由路由器（router）选择其中一部分。

固定配置声明：

| 字段 | 值 | 学习时怎样理解 |
|---|---:|---|
| `n_routed_experts` | 256 | 有 256 个可由 router 选择的专家 |
| `n_shared_experts` | 1 | 另有 1 个共享专家路径 |
| `num_experts_per_tok` | 8 | 每个 token 选择 8 个路由专家 |
| `n_group` | 8 | 路由专家被组织成 8 组 |
| `topk_group` | 4 | 先保留 4 个候选组 |
| `scoring_func` | `sigmoid` | 配置指定 sigmoid 打分 |
| `topk_method` | `noaux_tc` | 配置声明无辅助损失的 top-k 路由方法 |

`noaux_tc` 容易被误读成“不做负载均衡”。公开方法的要点是：用额外的校正偏置影响专家选择，
同时让实际 gate weight 仍来自原始分数；训练过程再根据专家负载调整偏置。配置文件只给出了方法名，
偏置怎样更新、怎样参与分组选择，仍应以同版本训练与推理代码为准。

从这枚 token 的视角，可以把过程理解为：

1. 共享专家处理当前隐藏状态，为所有 token 提供共同路径；
2. router 为 256 个路由专家计算分数；
3. 路由策略先缩小候选组，再选择 8 个专家；
4. token 被送到这些专家执行 MLP；
5. 各专家输出按 gate weight 合并，再与共享专家输出共同形成本层结果。

这解释了为什么 MoE 被称为**条件计算**：模型保存了很多专家，但单个 token 只执行其中一小部分。
不过，“少算一部分”不等于“少存一部分”。工程上至少要分开三本账：

- **总参数**决定权重文件、加载内存和分片规模；
- **激活参数**描述单个 token 实际经过多少参数；
- **系统成本**还包括路由、容量限制、负载不均、通信和 kernel 效率。

### 为什么多卡 MoE 很依赖运行时

如果专家分布在不同 GPU，路由器选完专家后，运行时要把 token 发送到拥有该专家的设备。
专家完成计算，输出还要回到 token 的来源位置。

这条往返路径通常需要全互连通信（all-to-all）和分组矩阵乘法（grouped GEMM）。
某些专家收到过多 token 时，系统还要按容量策略决定保留、丢弃或重新路由哪些分配。

因此，验证一个 MoE 服务不能只看 `num_experts_per_tok=8`。还应观察每个专家收到、接受和丢弃的 token，
设备间通信量，prefill 与 decode 的负载差异，以及尾延迟和峰值显存。

本仓库里的 NumPy、PyTorch 和 CPU/Gloo MoE 实验可以解释路由、容量与通信机制；它们没有加载
DeepSeek-V3 权重，也没有执行这份配置对应的 router。固定配置告诉我们“有哪些字段”，同版本模型代码与
运行轨迹才能告诉我们“这些字段怎样执行”。

## 第三步：从层输出到下一个 token

MoE 输出经过残差连接后进入下一层。所有层完成后，最终隐藏状态由语言模型输出头映射成词表上的 logits，
也就是每个候选 token 的未归一化分数。

模型前向到这里结束。接下来，推理运行时应用温度、top-p、top-k 或贪心规则，选出下一个 token。
它会更新序列状态，再发起下一轮解码。

模型权重负责计算 logits；排队、缓存分配、批处理和采样属于运行时。

## 模型、框架和 kernel 各负责什么 {#runtime-dependencies}

理解 DeepSeek 的另一条主线，是问清楚“一次前向究竟由谁完成”。

| 层次 | 在这条路径中负责什么 | 需要核对什么 |
|---|---|---|
| Checkpoint | 提供 config、tokenizer、权重与发布说明 | model ID、revision、文件摘要、许可与模板 |
| 模型实现 | 把 MLA、MoE、RMSNorm 等结构写成张量计算 | 是否与同一 revision 的权重和字段匹配 |
| Transformers | 读取配置和 tokenizer，并可按映射加载模型类 | 版本、`trust_remote_code`、自定义代码来源 |
| 推理运行时 | 管理请求、batch、KV 状态、并行、采样和取消 | 是否原生支持该架构及其 cache layout |
| CUDA 与计算 kernel | 在 GPU 上执行 attention、GEMM、通信和量化计算 | dtype、硬件能力、fallback、数值与性能 |

固定配置中的 `auto_map` 指向 `configuration_deepseek` 和 `modeling_deepseek`。这只说明发布者声明了哪些
Python 类映射，并不等于这些文件已经被仓库下载、审阅或执行。

同样，某个运行时能够加载普通 decoder，也不代表它已经支持 MLA cache、细粒度专家、FP8 权重格式和
目标并行方式。兼容性应沿着“配置能识别 → 权重能加载 → 最小前向能运行 → 缓存与并行语义正确 →
目标 workload 达标”逐级验证。

## R1：后训练改变行为，不替换基础架构

R1 路线关注的是怎样训练出更强的推理行为。一个抽象流程可以写成：

```text
base model
→ cold-start 或 SFT 数据
→ 为同一问题采样多条推理轨迹
→ 用答案、测试用例或学习到的模型给出奖励
→ 更新策略
→ 过滤高质量轨迹并训练 Distill 学生
```

可验证奖励适合数学答案、程序测试或格式约束等有明确判据的任务。它能减少一部分奖励模型误差，
但验证程序本身仍可能写错，也可能被模型利用。

组内相对优化的直觉是：对同一问题采样一组答案，比较组内结果来构造优势值（advantage），再更新策略。
具体的奖励、归一化、裁剪、KL 惩罚和 token 掩码必须以目标实现为准。

推理阶段看不到一个独立的“R1 层”。后训练已经把行为写进参数；test-time compute 再通过更长生成、
多候选、验证程序或工具调用增加求解预算。

### Distill 学生继承什么

R1 Distill 学生可能使用 Qwen 或 Llama 作为基础模型。学生可以从 teacher 生成的数据中学习答案分布、
表达方式和一部分求解策略，但内部结构仍由学生自己的 checkpoint 决定。

所以部署 Distill 模型时，要读取学生 config 来决定：

- 它使用 MHA、GQA 还是其他 attention；
- 它是稠密模型还是 MoE；
- KV Cache 公式和 LoRA target modules 是什么；
- tokenizer、chat template、量化格式和运行时是否匹配。

输出风格像 teacher，只能说明行为相似，不能据此写成“继承了 DeepSeek-V3 的 MLA、MoE、FP8 kernel
或训练轨迹”。

## MTP 与 FP8 在哪里

多 token 预测（Multi-Token Prediction，MTP）主要发生在训练目标一侧。除预测下一个 token 外，
附加预测头还学习更远位置的 token，为训练提供额外信号。

固定配置中的 `num_nextn_predict_layers=1` 表明发布结构声明了一个额外预测层。它没有告诉我们普通服务路径
是否加载或调用这个预测头。若要把它用于推测解码（speculative decoding），运行时还要显式实现候选生成和验证流程。

FP8 属于数值表示与计算内核路线。固定配置声明了 E4M3 格式、动态激活缩放和
`weight_block_size=[128, 128]`。

真正的收益还取决于硬件是否支持、哪些算子使用 FP8、累加精度、回退路径、量化比例、数值误差和额外工作区。

可以用下面的证据顺序判断一句“FP8 加速了模型”是否站得住：

```text
config 中有 FP8 字段
→ 权重文件确实采用对应格式
→ 目标运行时加载了这些权重
→ profiler 看到目标 FP8 kernel
→ 质量误差在预算内
→ 同一 workload 的端到端延迟或吞吐改善
```

前一步是后一步的必要条件，但不能替代后一步。

## 配置文件能回答到哪一步

读模型资料时，可以把证据分成五层：

| 层级 | 你手里的证据 | 可以回答的问题 |
|---|---|---|
| L1 | 固定技术报告或发布说明 | 发布者公开描述了什么 |
| L2 | 固定 config、tokenizer 或代码文件 | 文件中有哪些字段、模板和静态结构 |
| L3 | 权重清单与成功加载记录 | 指定权重是否完整读取 |
| L4 | 目标运行时的 trace | forward、cache、routing 和 kernel 实际怎样执行 |
| L5 | 目标任务与硬件报告 | 质量、容量、吞吐、延迟和 SLO |

本仓库当前的 DeepSeek-V3 证据到 L2。它足以阻止标准 K/V 公式被误用，也足以教你从配置辨认结构；
它不能给出真实 cache 大小、专家选择、FP8 kernel 覆盖、模型质量或 GPU 性能。

你可以在仓库根目录运行：

```bash
python projects/transformers-basics/verify_release_evidence.py
python projects/transformers-basics/inspect_config.py \
  projects/transformers-basics/release-evidence/deepseek-v3.config.json
```

第一条命令核对本地固定文件与预期字段，第二条命令会识别 MLA 并拒绝标准 K/V 容量计算。
它们都不会加载模型权重。

## 在 3070 Laptop 上怎样学习

完整 DeepSeek-V3 权重不适合作为单张消费级 GPU 的入门实验。

你现在使用的 Qwen3-0.6B + nano-vLLM 反而是理解运行时的好起点。它能让你亲手观察分词、预填充、
逐 token 解码、调度、分页式 KV Cache 和采样，而不会先被超大权重与多卡通信挡住。

把当前进度与 DeepSeek 页面对应起来：

| 你已经能在 Qwen3 + nano-vLLM 中观察 | DeepSeek-V3 带来的新问题 |
|---|---|
| 稠密 GQA 的 prefill 与 decode | MLA 改变了跨 decode 保存的状态 |
| Paged KV block 的分配和释放 | page 中究竟存潜在表示、位置分量还是展开后的 K/V |
| 单张 GPU 上的模型 forward | MoE 需要保存大量专家，多卡时还要 dispatch token |
| 普通 attention 与 MLP kernel | MLA、grouped GEMM 和 FP8 需要额外 kernel 支持 |
| 一次生成的 token 轨迹 | R1 评测还要固定推理预算、候选数和验证程序 |

推荐按三步推进：

1. **先做静态对比**：把 Qwen3 的 GQA config 与本页 DeepSeek-V3 config 并排阅读，指出标准 K/V 公式
   从哪个字段开始失效。
2. **再运行可承受的 Distill 学生**：选择显存允许的具体 Qwen/Llama 系 Distill checkpoint，用学生自己的
   config、tokenizer 和运行时做 prefill/decode；不要把学生结果标成 V3 架构结果。
3. **最后研究远程行为**：如果需要评测较大的 DeepSeek 模型，使用云 API 保存 model ID、请求参数、usage、
   finish reason 与核对日期。API 结果能回答行为和成本，不能揭示服务端权重或 kernel。

下载任何 Distill 权重前，先检查 weight shard 总大小、量化格式、目标运行时支持和预期峰值显存。
先跑短输入、并发 1 的冒烟测试，再逐步增加上下文与并发。

## Test-time compute 怎样公平比较

推理模型常通过更长生成、多次采样、self-consistency（自一致投票）、验证程序选择或工具调用增加
test-time compute。比较两个系统时，至少要固定或同时报告：

- prompt、chat template 与停止条件；
- sampled tokens、候选数 \(k\) 和最大轮数；
- temperature、top-p 与随机种子策略；
- 验证程序、工具、超时和重试；
- 所有尝试与成功样例各自的成本和延迟。

`pass@k` 根据较大的采样池估计“给 \(k\) 次机会时至少成功一次”的概率；`oracle@k` 直接检查本次
\(k\) 个候选中是否存在正确答案。它们回答候选集有没有覆盖正确解。Self-consistency 或验证程序选择还要
回答另一件事：系统最终选出的答案是否正确。

因此，候选覆盖率提高并不保证最终答案变好。验证程序很弱时，增加候选只会增加成本，甚至给错误候选更多
被选中的机会。仓库的候选选择指标会同时报告“正确答案是否出现”和“最终答案是否选对”，避免把两者混为一谈。

## 常见错误

- 看到 DeepSeek 名称就默认 checkpoint 同时使用 MLA 和 MoE。
- 用 Distill 学生的输出行为推断它复制了 teacher 的内部架构。
- 对含 MLA 字段的 checkpoint 套用标准 MHA/GQA K/V 公式。
- 从专家数量直接推断 active FLOPs、通信效率或服务吞吐。
- 把 MTP 或 FP8 配置字段写成已经执行的解码加速与显存收益。
- 用云 API 的兼容请求格式推断服务端采用某个开放权重 revision。
- 把通用 MoE、强化学习或 CPU 通信样例写成 DeepSeek-V3 复现。

## 面试时怎样回答

面对“介绍 DeepSeek 的关键技术”，可以沿着本页的 token 路径回答：

1. 先说明具体讨论技术报告、开放 checkpoint、Distill 学生还是云 API。
2. 从 attention 讲 MLA 怎样改变 K/V 表示与缓存，再讲 MoE 怎样替换部分前馈层。
3. 把 R1 放到训练轴上，说明后训练改变行为，但 Distill 不复制 teacher 架构。
4. 给出运行时后果：MLA 需要匹配的 cache layout，MoE 需要专家 dispatch，FP8 需要硬件与 kernel 支持。
5. 最后说明验证层级：config 只能证明静态字段，真实行为和性能需要目标运行时与 workload。

继续追问时，应能解释：

- total parameters、active parameters 与服务成本为何不同；
- 为什么 `hidden_size / num_attention_heads` 不能解释本页的 Q/K 维度；
- Distill 学生为什么按自己的 base config 选择 LoRA modules；
- `pass@k`、`oracle@k` 与最终选择准确率分别测什么；
- FP8 config、FP8 权重、FP8 kernel 与端到端加速怎样逐级验证。

## 自测

1. 一枚 token 在后半段 decoder layer 中，先经过 MLA 还是 MoE？两者分别改变什么？
2. 为什么 `num_key_value_heads=128` 仍不足以使用标准 K/V 容量公式？
3. MoE 每个 token 只选 8 个专家，为什么模型加载和分布式服务仍可能很难？
4. R1 Distill 学生能够继承哪些行为，又不能据此声称继承哪些结构？
5. 从 FP8 配置字段走到“端到端加速”还缺哪些证据？
6. 你当前的 Qwen3 + nano-vLLM 实验已经覆盖了这条学习路线的哪些部分？

## 一手资料入口

- DeepSeek-AI，[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)。
- DeepSeek-AI，[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)。
- DeepSeek-AI，[DeepSeek-V2](https://arxiv.org/abs/2405.04434)。
- 具体 config、revision 与本仓库运行过的检查见[DeepSeek 证据台账](../evidence/deepseek-controls.md)。
