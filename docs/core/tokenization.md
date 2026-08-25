# Tokenization：模型究竟看见了什么？

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：正在学习模型训练、推理、RAG、上下文预算或云 API 的开发者。
- **先修**：知道文本可以编码成 UTF-8；不要求提前掌握 BPE。
- **首次阅读**：一段文本的四种表示 → BPE 怎样学习 → 模板怎样加入角色 → token 数怎样影响训练和推理。
- **完成信号**：能解释同一段文字为什么在不同模型中 token 数不同，以及换 tokenizer 为什么可能直接破坏模型。
- **卡住时**：先运行[30 分钟 Byte-BPE 实验](../guide/beginner-map.md#30-minutes)，只观察 bytes、merge 和 IDs。

</div>

## `你好🙂` 到底有多长

从人的视角看，`你好🙂` 是三个字符。Python 的 `len()` 也返回 3，因为这里恰好有三个 Unicode code points。
编码成 UTF-8 以后，它占 10 bytes：

```text
你        好        🙂
e4 bd a0  e5 a5 bd  f0 9f 99 82
```

至于它有几个模型 token，目前还无法回答。必须先指定 tokenizer 及其版本。

仓库提供了一个可以故意“记住”这句话的微型 byte-level BPE。运行：

```powershell
python projects/transformers-basics/train_byte_bpe.py `
  --text "你好🙂你好🙂" `
  --sample "你好🙂" `
  --vocab-size 280 `
  --min-pair-frequency 2
```

这次运行会看到四种表示：

| 层次 | `你好🙂` 在本实验中的值 |
|---|---|
| Unicode code points | 3 个：`U+4F60 U+597D U+1F642` |
| UTF-8 bytes | 10 个：`e4 bd a0 e5 a5 bd f0 9f 99 82` |
| 初始 byte IDs | `[228, 189, 160, 229, 165, 189, 240, 159, 153, 130]` |
| 训练后的 token IDs | `[264]` |

为什么最后只有一个 token？训练语料把同一句话重复了两次，BPE 逐轮合并这些高频相邻 bytes，最终学到 ID `264`
代表完整的 `你好🙂`。

这个结果只展示机制。它不说明 Qwen、Llama 或任何真实模型也会把这句话切成一个 token。换一份语料、词表大小、合并顺序
或预切分规则，结果都会变化。

## Token ID 是 Embedding 的行号

模型不会直接接收字符串。Tokenizer 最终输出整数序列，例如 `[264]`。模型随后用每个 ID 查找 Embedding 矩阵中的一行：

```text
raw text
→ normalization（可选）
→ pre-tokenization（可选）
→ 子词或 byte 切分
→ token IDs
→ 加入 BOS/EOS 与聊天角色
→ Embedding rows
```

这条链中至少有五种容易混淆的对象：

| 对象 | 它描述什么 |
|---|---|
| Grapheme cluster | 用户视觉上感知的“一个字符” |
| Unicode code point | 文本标准中的码点；组合字符可能由多个码点显示成一个字形 |
| UTF-8 byte | 文件和网络常用的编码单位，一个码点占 1～4 bytes |
| Token piece | Tokenizer 选择的词、子词、空白或 byte 片段 |
| Token ID | Piece 在固定词表中的整数编号，也是模型权重的索引 |

Python 的 `len(text)` 通常数 code points；网页光标更接近 grapheme；网络限制常按 bytes；模型窗口和 API 用量按 tokens。
工程接口应写清 `utf8_bytes`、`token_count` 等单位，避免所有字段都叫 `length`。

## 为什么不用“一个字一个 token”

词级词表很难覆盖姓名、代码、拼写变化和不断出现的新词。字符或 byte 词表能覆盖长尾输入，却会把句子拉得很长。
子词 tokenizer 在两端之间折中：高频片段用一个 ID，罕见片段退回较小单位。

这个折中会同时改变模型和系统成本：

- 词表大小为 \(V\)、隐藏维度为 \(d\) 时，输入 Embedding 有 \(Vd\) 个参数；
- 没有 weight tying 时，输出 head 还有同量级参数；
- 较大的词表通常能缩短序列，但会增加 Embedding、输出层计算和稀有 token 行；
- 较小的词表会产生更多 tokens，增加 prefill、激活和 KV Cache 压力。

这不是“词表越大越快”或“越小越省参数”的单向选择。真实速度还取决于长度分布、batch、kernel 和硬件。

## BPE 先学习 merge，再用固定顺序编码

Byte Pair Encoding（BPE）的核心动作很简单：反复找出语料中最常见的相邻 pair，把它们合成新 token。

用仓库例子观察 `banana`：

```powershell
python projects/transformers-basics/train_byte_bpe.py `
  --text "banana bandana" `
  --text "banana" `
  --sample "bandana" `
  --vocab-size 280 `
  --min-pair-frequency 2
```

这份语料实际学到四条规则：

| Rank | 左 ID + 右 ID | 新 ID | 展开后的 bytes/text |
|---:|---|---:|---|
| 0 | `97 + 110` | 256 | `an` |
| 1 | `98 + 256` | 257 | `ban` |
| 2 | `256 + 97` | 258 | `ana` |
| 3 | `257 + 258` | 259 | `banana` |

编码 `bandana` 时，Tokenizer 从 bytes 开始，按已学习的 rank 应用规则，得到：

```text
[257, 100, 258] = [ban, d, ana]
```

### 训练与编码最容易混淆的地方

训练阶段需要统计语料频次并产生 merge list。编码新文本时不会重新统计“这个输入里哪个 pair 最多”，而是重放固定的
merge rank。否则，同一句话放进不同 batch 可能得到不同 ID，模型便无法用固定 Embedding 解释它。

一份可复现的 BPE 还要明确：

- pair 是否跨文档或换行统计；
- 同频 pair 怎样打破平局；
- `aaa` 中两个重叠的 `aa` 选择哪一个；
- 是否先按空白、正则或词边界预切分；
- 初始字母表、最低频次和目标词表大小。

所以“使用 BPE，词表 32k”远远不足以标识一个 tokenizer。

### 仓库实现刻意保持简单

`ByteBPETokenizer` 的基础 IDs 0～255 直接表示 bytes，新 ID 按 `256 + merge_rank` 分配。同频 pair 使用 ID pair 的
字典序打破平局，统计不会跨调用者传入的文档边界。

它每轮重新统计整个小语料，适合手算和测试。

GPT-2、Qwen 和 Llama 各自还有规范化、预切分、特殊 token、文件格式与高性能实现。仓库实验只验证固定字符串上的
合并顺序和 UTF-8 往返符合本实现。

## BPE、WordPiece、Unigram 和字节回退有什么差别

这些名字描述算法家族，不足以推断某个 checkpoint 的精确切分：

| 方法 | 怎样得到或选择 piece | 阅读实现时还要确认什么 |
|---|---|---|
| BPE | 从基础符号出发，反复合并相邻 pair | merge rank、预切分、平局与边界规则 |
| WordPiece | 构造子词词表，常带词内 continuation 标记 | trainer 的评分、未知词和 normalization |
| Unigram | 从较大候选集出发，为 piece 建模概率并逐步删减 | 动态规划、采样和候选剪枝规则 |
| Byte fallback | 未知片段退回 byte tokens | 是否覆盖全部 256 bytes，以及 fallback 何时触发 |

Unigram 中的“unigram”指 piece 概率模型，不表示后面的语言模型只能看一个 token。按分词分布采样可以用于 subword
regularization，但这会让编码具有随机性，需要在训练协议中明确。

字节回退（byte fallback）只是未知片段的处理方式。现代 tokenizer 可以同时使用常见子词与字节回退，并不因此变成
纯 byte-level 实现。

单个 byte token 或中间合并 token 可能不是合法 UTF-8。解码器需要先展开完整序列，再恢复原始 bytes。

## Normalization 会改变“原文”

屏幕上看起来相同的 `é` 可以有两种码点序列：

```text
预组字符：U+00E9
组合形式：U+0065 U+0301
```

NFC 可以把组合形式规范成预组形式。NFKC 还会折叠兼容字符，例如部分全角形式。这样的规则可能提高语料共享，也可能
改变用户名、产品名、代码、数学符号和安全匹配。

因此要分别保存：

1. 用户提交的原始文本；
2. Tokenizer 实际处理的规范化文本；
3. 二者之间经过验证的位置映射。

Offset mapping 必须标明坐标系：

- UTF-8 byte offset；
- Unicode code-point index；
- UTF-16 code unit；
- 规范化文本位置。

`decode(encode(text))` 成功只说明当前编码与解码路径相容，无法单独证明高亮位置仍能回到原文。

RAG 引用、PII 标注和网页高亮尤其依赖这条映射。索引阶段与查询阶段如果使用不同 normalization，检索结果也会漂移。

## Chat template 会在用户文字之外加入 tokens

聊天 API 接收的是结构化 messages，模型训练和推理看到的却是模板序列：

```text
[{role: system, ...}, {role: user, ...}]
→ chat template
→ BOS / role delimiters / message content / assistant prompt
→ token IDs
```

因此，“用户 Prompt 有多少 token”通常不能只编码消息正文。应该先用目标模型的模板渲染完整输入，再计数。

需要核对的边界包括：

- 模板是否已经添加 BOS/EOS，调用方是否又重复添加；
- generation prompt 是否打开了正确的 assistant turn；
- tool call、tool result、空内容和 JSON escaping 怎样表示；
- padding side、attention mask 与 position IDs 是否符合运行路径；
- SFT 中哪些 assistant tokens 参与 loss，哪些 Prompt 和 padding 位置被 mask；
- tokenizer、template 和模型权重是否固定到兼容 revision。

把新 token 追加到词表只创建了一个新 ID。模型还要扩展输入 Embedding 和输出 head，并用数据训练新行。重新排列旧 ID
或修改 merge ranks 更危险：同一个整数会指向另一段文本，原有权重的含义随之错位。

## 放到 Qwen3 与 nano-vLLM 中看

在 Qwen3-0.6B + nano-vLLM 的实验里，职责可以这样分：

| 部件 | 与 tokenization 有关的职责 |
|---|---|
| Qwen checkpoint | 提供 tokenizer 文件、special token 配置、chat template 和与词表对齐的模型权重 |
| Transformers | 加载 tokenizer/config，执行模板、encode 与 decode |
| nano-vLLM | 调度 token IDs，运行 Qwen 模型、维护 KV Cache，并把生成 ID 交回 tokenizer 解码 |
| CUDA / Triton kernels | 计算张量，不决定一段文字应该切成哪些 IDs |

模型运行时主要处理整数 ID。Tokenizer 若换成不兼容版本，即使 tensor shape 勉强一致，ID 也可能指向错误的 Embedding 行。

上下文长度同样要在应用 chat template 以后计算。Tokenizer 能告诉你当前输入有多少 IDs；模型 config 和推理 Runtime
共同决定最多能接受多少，以及超长时是报错、截断还是采用其他位置扩展策略。完整运行链路见
[实验 7B](../practice/labs/lab-7b-nano-vllm-qwen3.md)。

## Token 数会影响哪些训练和推理量

同一原始 workload 换 tokenizer 后，以下内容都会变化：

- SFT 的输入 IDs、assistant mask、labels 和有效训练 token 数；
- RAG 能装入多少证据，以及截断发生在哪个文档；
- Prefill 序列长度、激活、注意力工作与 KV Cache 分配；
- Decode 的停止 ID 与可见输出边界；
- 云 API 报告的 input/output usage 和费用估算。

比较 tokenizer 时，不要只报告平均 token/字符比。至少按中文、英文、代码、JSON、数字、emoji、OCR 噪声和长尾姓名分组，
观察：

- UTF-8 bytes/token；
- code points/token，并注明它不等于 grapheme；
- 每条样本的 token 数分位数与超窗率；
- `[UNK]` 或 byte fallback 占比；
- encode/decode 延迟和内存；
- round trip、offset、special token 与模板回归。

两个模型都声称支持 128k tokens，不代表它们能装入同样多的中文、代码或业务记录。Token 数也不是跨模型公平的语义
工作量单位。比较成本或吞吐时，应固定原始样本和模板，并同时报告各自的 token 分布。

序列变短通常会降低 prefill 和 KV 压力，但不能直接推出端到端延迟按平方下降。Decode、模型权重、kernel、batch 和调度
仍然参与最终时间。

## 常见故障怎样定位

| 现象 | 先检查什么 |
|---|---|
| 同一句话在本地与服务端 token 数不同 | tokenizer、revision、normalization 与 chat template |
| 训练 loss 异常地低 | labels 是否把 Prompt、padding 或控制 tokens 算入/排除错了 |
| RAG 引用高亮偏移 | offset 的坐标系，以及 normalization 前后的映射 |
| 批处理左侧出现奇怪输出 | padding side、attention mask、position IDs 与 PAD/EOS 配置 |
| 模型一直生成或过早停止 | EOS、stop IDs、generation prompt 与模板边界 |
| 用户文本伪装成角色 delimiter | 应用是否把普通字符串误当成 special token ID |
| 截断后 JSON 或中文损坏 | 是否按 byte/code point 截断，而不是在完整模板上按 token 边界处理 |
| 安全过滤器与模型看到的内容不同 | 两边是否采用相同 canonicalization 与 normalization |

零宽字符、双向控制符、同形异义字和异常组合字符还会影响审计显示与匹配。Tokenizer 无法解决 Prompt Injection；角色
delimiter 也只是模型行为提示。权限、工具参数和副作用必须继续由模型外控制面验证。

## 保存 tokenizer 时需要保存什么

算法名称和词表大小不够。一个可复现工件至少包括：

1. Normalizer、pre-tokenizer、trainer 和所有平局规则；
2. 基础字母表、vocabulary、merge ranks 或 piece probabilities；
3. Special token 的字符串、ID、匹配方式和自动添加规则；
4. Decoder、cleanup、chat template、padding 与 truncation policy；
5. 训练语料 snapshot、许可、混合比例和 trainer revision；
6. 规范文件的 hash、模型 revision 和兼容性测试集；
7. 多语言、长尾和安全样本的长度、round trip、offset 与模板回归。

无密钥 hash 可以标识所覆盖文件的 bytes，不能认证发布者、语料许可或远端服务实际部署的版本。

## 实验与自测

推荐先观察 0～255 byte baseline，再训练仓库的小 BPE，最后加载一个固定 revision 的真实 tokenizer。

三个阶段都使用同一组样本：中文、英文、代码、JSON、数学、emoji、组合字符和罕见姓名。这样差异才能归因到 tokenizer，
而不是输入变化。

1. `你好🙂` 为什么在本章 toy 中是一个 token，却不能据此推断 Qwen3 的 token 数？
2. BPE 编码为什么必须使用训练得到的 merge rank，不能对每个输入重新统计？
3. Byte fallback 消除 `[UNK]` 后，为什么某些语言仍可能产生更长序列？
4. Round trip 通过以后，normalization、offset、template 和模型兼容性还需要哪些测试？
5. 扩词表时为什么要同时修改并训练 Embedding/LM head？Weight tying 会改变哪部分参数？
6. 怎样验证截断没有切断完整 assistant turn，且用户文本不能注入真实控制 ID？
