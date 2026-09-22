# NLP 与语言建模

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要理解文本如何变成语言模型输入输出的开发者。
- **先修**：Python 基础和条件概率直觉。
- **首次阅读**：先运行两条样本命令，再沿 bytes → token IDs → labels → logits → loss 阅读。
- **完成信号**：能逐位置解释模型看见什么、预测什么，并从三个有效目标复算平均 NLL。
- **卡住时**：先做[新手知识地图](../guide/beginner-map.md)的四项自检。

</div>

自然语言处理（Natural Language Processing，NLP）研究怎样把语言变成可计算对象，又怎样把模型结果还原为对人
有用的语言行为。

这页先让 `你好🙂!` 经过 UTF-8 字节、BPE 子词和因果语言模型的训练目标。后面的概率、训练输入和生成过程，
都会回到这一个样本解释。

## 先运行一个完整样本

```powershell
python projects/transformers-basics/trace_language_model_sample.py
```

这条命令会训练仓库的微型 Byte BPE，然后构造 labels、因果可见范围和 loss mask。输出中的关键数字是：

| 阶段 | 固定样本的结果 |
|---|---|
| 人类输入 | `你好🙂!`，4 个 Unicode code points |
| UTF-8 | 11 bytes |
| BPE 文本 token | `[264, 33]`，分别展开为 `你好🙂` 和 `!` |
| 本实验特殊 ID | `BOS=265`、`EOS=266`、`PAD=267` |
| 模型输入 | `[265, 264, 33, 266]` |
| 预测目标 | `[264, 33, 266, 267]` |
| Loss mask | `[true, true, true, false]` |

程序实际训练了 BPE，并构造了特殊 token、因果 mask 和计分位置。运行到这里便停在模型前向计算之前，
所以输出中没有 logits、loss 或 perplexity。

ID `264` 只属于这次训练出来的微型词表。Qwen、Llama 等真实 checkpoint 必须使用各自配套的 [tokenizer](../reference/glossary.md#term-tokenizer)。

你可以先只记住三条边界：

```text
字符串不是 token IDs
因果 [attention](../reference/glossary.md#term-attention) mask 决定“能看哪些输入位置”
loss mask 决定“哪些预测目标参与计分”
```

### 再让这个样本真正经过模型

```powershell
python projects/transformers-basics/trace_minigpt_training_step.py
```

第二条命令复用同一组 IDs，把它们送入仓库从零实现的 MiniGPT。这个模型只有 1 层、16 维和 6496 个参数。
它会依次执行词嵌入、因果注意力、前馈网络和词表输出，再计算[交叉熵](../reference/glossary.md#term-cross-entropy)、反向传播并完成一步 SGD 更新。

模型会在四个位置都输出 268 个 logits，所以输出形状是 `[1, 4, 268]`。前三个目标参与 loss；最后一个位置
虽然也会得到 `<PAD>` 的概率，却被 `-100` 排除。固定 seed 下，逐位置结果如下：

| 输入 → 目标 | 目标概率 | 负对数概率（NLL） | 计入平均值 |
|---|---:|---:|---|
| `BOS → 你好🙂` | 0.003612 | 5.623527 | 是 |
| `你好🙂 → !` | 0.004073 | 5.503419 | 是 |
| `! → EOS` | 0.003587 | 5.630446 | 是 |
| `EOS → PAD` | 0.004083 | — | 否 |

前三项 NLL 的平均值是 5.585798，与模型返回的交叉熵一致；对它取指数，得到这三个目标上的 perplexity
约 266.6。随机初始化时，268 个词表项的概率接近均匀，因此这个数量级并不意外。

一步 SGD 后，同一小样本上的 NLL 降到 5.134247，12 个参数张量都发生了变化。这只说明训练接线工作正常：
模型刚刚朝记住这三个目标的方向走了一步。它没有因此学会中文，也没有在新样本上证明[泛化](../reference/glossary.md#term-generalization)。

## 1. 语言的层次

上面的程序只处理了文本表示和训练目标。真实语言问题还可以分成：

- **形态（morphology）**：词素、屈折和派生，例如时态、数、词缀；
- **句法（syntax）**：成分和依存关系，决定“谁修饰谁”；
- **语义（semantics）**：字面意义、指代、量词和组合意义；
- **语用（pragmatics）**：说话者意图、共同知识、含义和社会语境；
- **篇章（discourse）**：跨句指代、主题推进、全局一致性。

一个流畅句子可以句法正确但事实错误；一个事实正确的回答也可能违背用户意图或缺少必要前提。将所有失败统称为“语言理解不好”会妨碍定位。

## 2. 从文本到可计算单位

原始文本首先经过 Unicode、空白、大小写、规范化和分段策略，再被表示为字符、词、子词或字节。每一步都可能改变信息。

### 2.1 传统表示

**词袋（bag of words）**用词项计数表示文档，忽略顺序。TF-IDF 降低高频通用词的权重；BM25 进一步引入词频饱和与文档长度归一化。它们对精确术语、编号、专有名词和可解释检索仍很强。

**n-gram**保留长度为 \(n\) 的局部顺序。它能建模短程搭配，但状态空间随词表和 \(n\) 快速增长，未见组合需要平滑（smoothing）或回退（backoff）。

**静态词向量**让每个词项对应一个稠密向量。它能表达分布相似性，但同一个词在不同语境仍共享向量。

### 2.2 子词与字节 token

现代 LLM 常使用 BPE、Unigram、WordPiece 变体或字节级方案，将开放词表问题转成固定词表上的序列建模。权衡包括：

- 词表越大，常见片段可用更少 token 表示，但 embedding/output matrix 更大；
- 词表越小，序列通常更长，attention 和 KV cache 成本可能增加；
- 字节回退可避免真正的 unknown token，但罕见文本可能被拆得很长；
- 不同语言、代码和数字格式的 token 效率可能差异很大。

Tokenizer 是模型契约的一部分。添加 special token、修改正规化或改变 chat template 都可能改变 ID 序列。只加载权重而使用“名字相近”的 tokenizer 不保证兼容。

完整算法见 [Tokenization](../core/tokenization.md)。

### 2.3 同一个样本怎样 shift { #shift-and-mask }

程序先构造完整序列：

```text
[BOS, 你好🙂, !, EOS, PAD]
```

然后把输入和目标错开一位：

| 位置 | 当前位置输入 | 要预测的目标 | 可见输入位置 | 计入 loss |
|---:|---|---|---|---|
| 0 | `BOS` | `你好🙂` | `0` | 是 |
| 1 | `你好🙂` | `!` | `0, 1` | 是 |
| 2 | `!` | `EOS` | `0, 1, 2` | 是 |
| 3 | `EOS` | `PAD` | `0, 1, 2, 3` | 否 |

这张表同时出现两个 mask。因果 attention mask 是下三角形，让位置 1 看见位置 0 和 1，却看不见 2、3。
Loss mask 则保留前三个预测目标，排除最后的 padding 目标。改变其中一个，不会自动替你改变另一个。

不同库对 shift 的分工不同：有的模型在 forward 内部移动 labels，有的训练代码先构造错位序列。接入时应打印
一个 batch 的输入、目标和有效位置，确认 shift 只发生一次。

训练样本的 `EOS` 表示“这条已知文本到这里结束”，它是上表中位置 2 要预测的目标。
以同样的 `你好🙂!` 为例，若应用希望模型续写，这个教学例子的开放前缀可以是
`[BOS, 你好🙂, !]`。Prefill 的最后一行 logits 选出 `u_1`；把 `u_1` 追加后，下一次 forward 才用
`[BOS, 你好🙂, !, u_1]` 的最后一行 logits 选 `u_2`。本例在新选中的 ID 命中配置的 EOS，或应用停止规则命中时结束。
这不要求从真实 prompt 中删除所有 EOS：历史消息的结束标记可能是模型模板的一部分。
对话模型还会用模板标出角色及待续写的位置，应保留目标模板，不能把这个教学前缀照搬到任一 checkpoint。

## 3. 语言模型的概率定义

语言模型给 token 序列 \(x_{1:T}\) 分配概率。概率链式法则给出：

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

因果语言模型最小化负对数似然：

\[
\mathcal L(\theta)
=-
\sum_{t=1}^{T}m_t
\log p_\theta(x_t\mid x_{<t}),
\]

其中 \(m_t\in\{0,1\}\) 表示该 token 是否计入目标。固定样本的三个有效目标是 `你好🙂`、`!` 和 `EOS`，
所以 token mean 的分母是 3，而不是输入长度 4 或完整序列长度 5。只有模型给出每个目标的概率后，才能计算这项
loss；walkthrough 本身没有伪造概率。

这个分解来自概率规则，不要求模型一定使用 Transformer；n-gram、RNN、状态空间模型也可定义自回归概率。

### 3.1 交叉熵与困惑度

若平均 token negative log-likelihood 为 \(H\)（使用自然对数），困惑度（perplexity）是

\[
\operatorname{PPL}=\exp(H).
\]

它可直觉理解为模型在每一步面临的“有效分支数”，但只是比喻。PPL 有重要边界：

1. 必须说明 tokenizer、数据、正规化、上下文截断和 loss mask；
2. 不同 tokenizer 改变 token 单位，token-level PPL 通常不能直接横比；
3. PPL 低不保证事实性、指令遵循、安全或任务成功；
4. 长文本的平均值会掩盖某些位置或切片的严重失败。

跨 tokenizer 比较可考虑按 byte/character 归一化的负对数似然，但仍要确保文本与预处理一致，不能把单位转换当作消除所有差异。

## 4. 主要预训练范式

### 4.1 因果语言模型（causal LM）

每个位置只能使用左侧上下文预测下一个 token，适合开放式生成。训练时可并行计算整段所有位置的 loss；“自回归”描述的是条件依赖和生成过程，不意味着训练必须逐 token 顺序执行。

### 4.2 掩码语言模型（masked LM）

从输入中遮盖部分 token，再使用双向上下文恢复它们。它适合表示学习与理解任务。训练时模型看到特殊 mask 或替换噪声，和下游输入可能存在差异。

### 4.3 Encoder-decoder 与去噪目标

编码器（encoder）双向读取输入，解码器（decoder）根据编码结果逐步生成输出。

Span corruption 会遮盖一段连续文本，再要求模型恢复它。这是一类常见的序列到序列预训练目标，适合翻译和
摘要等条件生成任务。只有解码器的模型也能把输入与回答拼成一个序列来学习这些任务。

### 4.4 Prefix LM 与混合掩码

Prefix 区域内部可双向可见，生成区域只能看 prefix 和自己左侧。实现关键是 attention mask，而不是模型名称。若 mask 与训练任务不匹配，信息可能从未来泄漏。

## 5. Teacher forcing 与训练—推理差异

训练固定样本时，四个输入位置一次送入模型。因果 mask 保证每个位置只使用真实前缀，因此三个有效目标可以并行
计算。这个做法叫 teacher forcing。

生成时只有 `BOS` 是已知的。模型先生成第一个 token，再把自己的结果放回前缀；某一步偏离以后，后续条件也会
变化。这种训练与推理前缀的差异常称为 exposure bias。

Teacher forcing 仍然提供清晰、密集且可并行的最大似然监督。生产系统通常结合数据改进、后训练、解码约束、
检索和验证器来控制生成错误。Scheduled sampling 等替代方法会改变训练分布，需要单独验证其目标与偏差。

## 主线到这里结束：其余主题按问题进入

到这里，`你好🙂!` 已从文本变成 token IDs、labels 和 loss，因果语言模型的训练目标也已经闭合。后续问题各有自己的
权威页面，不在这里再展开一套概览：

| 接下来要回答的问题 | 继续阅读 | 这一页不重复的内容 |
|---|---|---|
| logits 怎样变成输出并停止 | [生成与解码](../core/generation.md) | temperature、top-k/top-p、约束解码和停止条件 |
| attention 怎样计算并复用前缀 | [Transformer](../core/transformer.md) | 张量形状、mask、RoPE 与 KV Cache |
| 语料怎样摄取、去重并响应删除 | [数据工程](../training/data.md) | lineage、污染、多语言切片和训练 artifact |
| 传统与稠密检索怎样组合 | [RAG 召回](../applications/rag-retrieval.md) | BM25、向量召回、融合、重排与权限边界 |
| 怎样定义任务指标和错误切片 | [评测方法](../quality/evaluation-methodology.md) | 构念、分母、失败分类与发布判断 |

“预测下一个 token”会迫使模型压缩对搭配、句法、篇章和任务格式有用的统计规律，但它只优化条件分布拟合。
事实更新、来源追溯、权限和工具副作用仍需要 RAG、工具运行时与验证器。传统规则、BM25 或小模型也仍可作为更便宜、
更确定的基线；是否替换它们，应由同一任务上的质量、延迟和成本对照决定。

本页开头的两条命令先构造训练目标，再用同一个样本执行一次 MiniGPT 参数更新。若想逐单元格观察张量，继续运行
`notebooks/02_minigpt_forward.ipynb`；若想比较 PyTorch 与 JAX 的训练语义，进入 `projects/jax-minigpt/`。

## 常见错误结论

- **“Tokenizer 只是预处理，可以随时替换”**：token ID 与 embedding/output 权重绑定，替换会破坏模型契约。
- **“训练自回归模型必须逐 token 运行”**：训练可利用 causal mask 并行计算所有位置；逐 token 是常见解码形态。
- **“PPL 更低就一定回答更好”**：PPL 只测指定数据和 tokenizer 下的 token likelihood。
- **“更长上下文等于可靠记忆”**：窗口容量不保证检索、整合或位置鲁棒性。
- **“向量相似就是事实相同”**：embedding 学到的是训练目标下的相似性，可能忽略否定、数字或时效。
- **“结构化输出合法就代表答案正确”**：grammar/schema 只能约束形式，不能验证内容。

## 自测与实践

1. 写出因果语言模型的概率分解，并解释为什么训练时仍可并行。
2. 为什么使用不同 tokenizer 的两个模型不能直接比较 token-level PPL？
3. 给一个 batch 画出 input、labels、causal mask 与 loss mask，区分它们的作用。
4. 为什么 next-token loss 下降不能单独证明事实性或工具安全？
5. 你应去哪个专题定义检索指标，为什么不在本页继续展开？
