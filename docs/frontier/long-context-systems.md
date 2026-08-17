# 长上下文系统：从“能放”到“有效”

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要处理长文档、代码库、多轮 memory 或长上下文服务的工程师。
- **先修**：[Attention](../core/transformer.md)、KV Cache、tokenizer 和基础 RAG。
- **首次阅读**：三层长度 → 位置与 attention → KV/服务 → RAG/memory → 评测。
- **完成信号**：能把 context claim 拆成 protocol、runtime 和 effective task evidence。
- **卡住时**：先做同一事实位于开头/中间/结尾的三 case 实验。

</div>

**专题导航**：[前沿总览](reasoning-long-context-moe.md) · [RAG](../applications/rag.md) · [推理优化](../systems/inference-optimization.md) · [证据台账](../evidence/frontier-controls.md)
{ .doc-nav }

Context window 是一个容量上限，不是能力分数。模型接受长输入、runtime 成功完成和任务在远距离仍可靠，是三种不同结论。

## 先分开三层长度

| 层级 | 问题 | 需要的证据 |
|---|---|---|
| Protocol acceptance | API/tokenizer/runtime 是否接受该长度 | request/response 或明确 error |
| Runtime completion | 是否在显存、时间和预算内结束 | terminal、usage、latency、memory |
| Effective context | 不同位置和任务是否可靠 | sliced task evaluation |

一个请求成功只能回答前两层的一部分。标称 128k、1M 或更长，也不能单独回答第三层。

### Token 长度不是字符长度

保存目标 tokenizer 的 rendered prompt 和 token IDs。中文、代码、JSON、空白和 Unicode normalization 都会改变 token count。

Chat template、tool schemas、system instructions 和输出预算也占 context。不要只计算用户可见正文。

## Position representation 决定怎样感知顺序

### Absolute position

为每个位置提供独立 embedding。训练范围外的位置没有自然定义，扩展表长也不会自动获得可靠外推。

### Relative bias

Attention score 加入相对距离或 bucket bias，直接表达 token 间距离。Bucket 设计决定远距离怎样压缩。

### RoPE

Rotary Position Embedding 对 Q/K 通道按位置旋转，让点积带有相对位置结构。

RoPE scaling、base frequency、使用维度和插值策略必须绑定具体 checkpoint 与实现。只修改 max position 或 scaling config，不能证明模型在新长度上理解正确。

## Attention 的二次项来自哪里

标准 dense attention 为：

\[
S=\frac{QK^\top}{\sqrt{d_h}},\qquad
A=\operatorname{softmax}(S),\qquad
O=AV.
\]

Score matrix 对 sequence length \(T\) 有 \(O(T^2)\) elements，QK/AV 计算也包含二次项。

FlashAttention 通过 tiling 和 online softmax 减少 HBM traffic 与完整中间矩阵物化，但没有普遍把 exact dense attention 的算术复杂度变成线性。

### 常见替代路线

| 方法 | 核心思路 | 主要代价 |
|---|---|---|
| Sliding/local attention | 每个 token 只看局部窗口 | 远距离信息需跨层传播 |
| Block sparse/global tokens | 只连接选定 blocks 或 anchors | Pattern 与任务不匹配会漏信息 |
| Linear attention | 用 kernel/状态重排避免显式 \(T^2\) | 数学与数值性质改变 |
| Recurrent/SSM memory | 把历史压入固定或分层状态 | 信息压缩与遗忘 |
| Retrieval/memory | 只取相关外部内容 | 检索、权限和更新错误 |

“线性”可能指 memory、time 或特定前提下的复杂度。阅读论文时写清对象，不要只记标签。

## KV Cache 是解码主账之一

标准 dense K/V layout 的理想 payload：

\[
M_{KV}
=2LBTH_{kv}d_hs.
\]

其中 \(L\) 为层数、\(B\) 为 batch、\(T\) 为 cached tokens、\(H_{kv}\) 为 KV heads、\(d_h\) 为 head dimension、\(s\) 为元素 bytes。

这个公式不包含：

- page/block 对齐和 fragmentation；
- allocator 与 metadata；
- prefix sharing/refcount；
- quantization scales；
- beam/best-of-N 的多序列状态；
- temporary/workspace；
- weights 和 activations。

因此 config formula 是预检，不是目标 runtime 峰值。

### 自定义 cache 必须实测

MLA、recurrent memory 或 provider opaque state 可能缓存不同对象。不要为了填容量表而套用标准 GQA 公式。

检查目标 runtime：

1. Cache tensors/blocks 的 shape 与 dtype。
2. Prefill 后初始占用。
3. 每增加一个 token 的增量。
4. Batch、beam、prefix sharing 和 cancellation 行为。
5. Peak/resident memory 与 allocator。

## Prefill 与 decode 是两种负载

长 prompt 主要增加 prefill 计算和 TTFT；长输出持续增长 KV，并影响 decode concurrency。

分别报告：

- input/output token distribution；
- prefill latency 和 TTFT；
- decode TPOT；
- tokens/s 与 completed requests/s；
- peak memory；
- queue、preemption 与 rejection；
- timeout/OOM/truncation 分母。

一个短输出长输入 benchmark 不能代表长对话持续生成；反过来也一样。

## 长上下文与 RAG 互补

把全部知识库塞进 context 的问题：

- latency 与费用增加；
- irrelevant evidence 干扰；
- 权限内容可能混入；
- 更新需要重建输入；
- citation 和来源难追踪；
- effective context 可能不足。

RAG 可以缩小输入、更新知识、执行 ACL 并提供来源；长 context 则减少切分损失，支持跨文档综合。

一种分层 memory：

~~~text
working context
├── recent conversation
├── current retrieved evidence
├── compact state / summary
└── tool results needed for this step

external memory
├── authorized documents
├── structured state
├── event log
└── user-approved long-term memory
~~~

Summary 是有损压缩，不是原始事实。保存 source identity，并让高风险决定重新读取受信证据。

## 有效上下文需要任务矩阵

### Retrieval

- 单 needle 位于开头/中间/结尾；
- 多 needle；
- distractors；
- 相似实体与否定；
- 不同语言与格式。

### Integration

- 跨段 multi-hop；
- 新旧版本冲突；
- 来源优先级；
- 顺序与因果；
- 全局计数和聚合。

### Generation

- 答案是否引用支持 span；
- 长输出后段是否维持约束；
- 无答案是否 abstain；
- 是否复制无关或敏感内容；
- 格式与 stop 是否正确。

### Systems

- TTFT、terminal latency 与 cost；
- OOM/timeout/truncation；
- Prefix cache hit/miss；
- cancellation 与资源释放；
- offered/admitted/completed 分母。

一个 needle accuracy 不能代表 integration、generation 或 systems。

## 位置切片怎样设计

构造同一 answer-bearing evidence 的位置变体：

~~~text
case-a: evidence at 10%
case-b: evidence at 50%
case-c: evidence at 90%
~~~

其他内容、token budget 和 question 保持一致。再改变 distractor density，形成 position × interference 矩阵。

保存 rendered token positions，而不是字符百分比。Chat template 与 tokenizer 会移动真实位置。

按 case 做 paired comparison，并把 timeout/truncation 纳入分母。不要只对成功完成的长输入计算准确率。

## Long-context benchmark 的污染风险

公开 needle 模板或问答可能出现在训练和反复 prompt tuning 中。降低风险：

- 程序生成新的实体、数字和关系；
- 保留独立 final templates；
- 使用业务领域的授权 held-out documents；
- 加入 counterfactual 和冲突版本；
- 记录每次查看 test 的历史。

程序生成也要检查是否过于简单，避免模型只识别模板。

## 一个渐进式实验

### Step 1：短 baseline

同一问题只提供 answer-bearing 段落，建立正确回答、引用和 token/latency baseline。

### Step 2：位置

把证据移动到 10%、50%、90%，其他内容不变。

### Step 3：干扰与冲突

加入相似实体、旧版本和高词面重叠 distractors，要求按来源/时间规则消解。

### Step 4：容量

逐步增加 input tokens，记录 protocol acceptance、runtime completion、quality、TTFT、peak memory 和 cost。

### Step 5：RAG 对照

同一 corpus 用授权 retrieval 只提供 top-k evidence，比较质量、引用、延迟和成本。

每一步先预测最可能失败的位置，再保存最差 case。

## 常见错误

- 用 model-card context window 代替 effective context。
- 用字符数而不是目标 tokenizer token 数。
- 只修改 max position 配置就声称扩展成功。
- 把 FlashAttention 说成普遍线性 attention。
- 用理想 KV payload 代替 runtime peak memory。
- 长输入只统计成功请求，删除 OOM/timeout/truncation。
- 一个 needle benchmark 代表所有长上下文能力。
- 把 context stuffing 当作 RAG、ACL 和来源管理的替代。

## 面试时怎样回答

面对“如何支持长上下文”，按五层回答：

1. 固定 tokenizer/template 和真实 input/output token budget。
2. 区分 protocol length、runtime completion 和 effective context。
3. 解释 attention/prefill 与 KV/decode 成本。
4. 用 position × task × interference 矩阵评测。
5. 与 RAG/memory hierarchy 比较质量、权限、延迟和成本。

继续追问时，应能说明 FlashAttention 改善的主要是 IO/物化，KV formula 为什么不是显存峰值，以及 provider 接受请求为何不证明中间位置可靠。

## 自测

1. Context window 的三层含义分别需要什么证据？
2. 长 prompt 主要影响 prefill 还是 decode？长 output 呢？
3. 为什么 MLA checkpoint 不能套标准 GQA KV 公式？
4. 怎样构造位置切片又不改变问题本身？
5. 哪些场景应优先使用 RAG，而不是扩大 context？
