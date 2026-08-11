# Tokenization：从 Unicode 到模型契约

## 一句话结论

Tokenizer 不是可以随意替换的文本工具，而是模型参数语义的一部分：它确定原始输入怎样变成整数 id、角色和边界怎样序列化、哪些位置参与 loss，以及“上下文长度”和计费中的 token 到底是什么。模型只看到 id；同一段可见文本经过不同 normalization、pre-tokenizer、merge ranks、special tokens 或 chat template，可能成为完全不同的序列。

## 先分清五层对象

以一个包含中文、组合音标和 emoji 的字符串为例，至少要区分：

1. **Unicode code point**：例如预组字符 `é` 与 `e` + combining acute 是不同码点序列；
2. **grapheme cluster**：用户感知的“一个字符”，可能由多个码点组成；
3. **UTF-8 byte**：模型 tokenizer 常用的可逆底层单位，一个码点可能占 1～4 byte；
4. **token piece**：一个词、子词、空白连同单词、单个 byte 或它们的组合；
5. **token id**：Embedding/LM head 的行号。

Python 的 `len(text)` 近似数 code point，不等于 grapheme 数、UTF-8 byte 数或 token 数。UI 截断、模型预算和网络 payload 因此不能共用一个“长度”。

典型流水线是：

```text
raw text
  -> optional normalization
  -> optional pre-tokenization
  -> subword/byte segmentation
  -> vocabulary ids
  -> special tokens / chat template
```

并非每个 tokenizer 都执行全部步骤，步骤顺序也属于版本化契约。若 normalization 改写了原文，`decode(encode(text))` 可能只等于规范化后的文本，而不等于原始 byte；若业务需要原文 span，必须另存原文并验证 offset mapping。

## 为什么需要子词

词级词表会遇到开放词汇、拼写变化和巨大长尾；字符或 byte 词表能覆盖所有输入，却往往产生更长序列。子词方法在两者之间折中：把高频片段做成单个 token，让罕见输入退化为更细单位。

若词表大小为 (V)、隐藏维度为 (d)，输入 Embedding 有 (Vd) 个参数；未与输入权重共享时，输出 head 还会有同量级参数。增大 (V) 通常缩短序列，却增加参数、softmax 工作和稀有行的数据稀疏性。减小 (V) 节省词表参数，但更长的序列会增加 prefill attention、激活和 KV Cache 成本。这里只是方向关系，实际速度还取决于 kernel、batch、序列分布与硬件。

## BPE：训练和编码不是同一件事

### 训练

一个透明的 byte-level BPE 变体可以写成：

1. 将每篇文档独立编码为 UTF-8 byte id；
2. 统计文档内部所有相邻 token pair 的频次，不跨文档边界；
3. 选择最高频 pair，给它分配新 id；
4. 从左到右合并所有非重叠出现；
5. 重复，直到达到词表上限或没有 pair 达到最低频次。

例如 `abab` 从 byte id 开始，先学到 `(a,b) -> 256`，序列变成 `[256,256]`；若继续训练，可学到 `(256,256) -> 257`。对 `aaa` 合并 `(a,a)` 时只能取非重叠 occurrence；确定采用左优先还是其他规则，否则训练工件不唯一。

频次相同的 pair 如何打破平局、是否跨行/文档、预切分是否允许跨空白合并、最低频次和初始字母表都会改变 merge list。因此“BPE、词表 32k”不足以标识 tokenizer。

### 编码

编码新文本时，不重新按新文本频次训练。它从基础符号开始，按训练得到的 **merge rank** 应用已有规则。把“当前样本里最高频的 pair”继续合并会让同一文本的 id 依赖所在 batch，无法与固定 Embedding 对齐。

仓库提供一个故意朴素但可测试的实现：

```python
from about_llm.from_scratch import ByteBPETokenizer

tokenizer = ByteBPETokenizer.train(
    ["low lower lowest", "newer wider lower"],
    vocab_size=280,
    min_pair_frequency=2,
)
ids = tokenizer.encode("lower")
assert tokenizer.decode(ids) == "lower"
```

可运行实验会输出 merge rank、byte expansion、token 数和 round-trip：

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/train_byte_bpe.py `
  --text "banana bandana" --text "banana" --sample "bandana"
~~~

`ByteBPETokenizer` 的基础 id 0～255 就是 raw byte，新 token id 为 `256 + merge_rank`；同频 pair 用 id pair 的字典序确定选择，训练不会跨传入的 document 边界。若没有 pair 达到 `min_pair_frequency`，实际词表会小于请求上限。朴素训练每轮重数 pair，适合小语料观察机制，不是大规模 tokenizer trainer。

这个 reference **不是 GPT-2 或任一 checkpoint tokenizer 的复刻**：它没有 regex pre-tokenizer、GPT-2 式 byte-to-Unicode 可见字符映射、normalizer、special token、offset map 或高性能索引。测试通过只能证明给定字符串上的确定性 merge 与 UTF-8 round-trip，不能证明目标语料压缩率、目标模型兼容性或生产吞吐。

## WordPiece 与 Unigram

### WordPiece

WordPiece 也构造子词词表，常配合词边界 pre-tokenization 和词内 continuation 标记。很多教学复现用与语料似然或 pair 相对关联度相关的分数，而不是只取原始 pair 频次；具体 trainer、normalization、未知词处理和 tie-break 以 tokenizer 工件为准。不能只凭算法名称推断某 checkpoint 的切分。

### Unigram Language Model

Unigram 从较大的候选 piece 集开始，为 piece 建模概率，再迭代移除对语料目标影响较小的候选。给定 piece 概率后，可用动态规划寻找高概率分词，也可按分词分布采样以做 subword regularization。它的“unigram”指 piece 概率模型，不表示语言模型本身只看一个 token。

### byte fallback

只有当 tokenizer 的可达基础词表确实覆盖全部 256 byte，并在未知片段时执行 fallback，才能说任意 UTF-8 输入无需 `[UNK]`。出现 byte token 不代表整个 tokenizer 是纯 byte-level；现代 tokenizer 常把常见子词和 byte fallback 组合使用。单个 byte 或 merge token 可能不是合法 UTF-8，只有完整 token 序列展开后才可解码。

## normalization、offset 与原文证据

常见 normalization 包括 NFC、NFKC、大小写或空白规则。它们可能改善统计共享，也可能改变：

- 产品名、用户名、代码、数学符号和全半角差异；
- 原文 byte、字符位置和高亮 span；
- exact match、去重、检索和安全规则的行为；
- 数字、换行、前导空格或组合字符的 token 数。

NFKC 会折叠 compatibility character，不应被笼统描述为“无损清洗”。若训练阶段规范化而线上没有，或检索索引与 query 使用不同规则，模型兼容和召回都会漂移。

Offset mapping 也要写明坐标系：byte offset、code-point index、UTF-16 code unit 还是 normalized text position。仅验证 `decode(encode(text))` 无损，不能证明 offset 能准确映射回原始文档。

## special token 与 chat template

常见控制 token 包括 BOS、EOS、PAD、UNK，以及 system/user/assistant/tool 边界。聊天接口最终通常把结构化 messages 通过 chat template 序列化；模型训练看到的不是抽象 role 字段，而是模板产生的 token 序列。

必须核对：

- 模板是否已经加入 BOS/EOS，避免调用方重复添加；
- generation prompt 是否正确打开 assistant turn；
- tool call/result 的边界、JSON escaping 和空内容怎样表示；
- padding side、attention mask 与 position id 是否符合模型/批处理路径；
- SFT 中哪些 assistant token 参与 loss，prompt、padding 和控制 token 怎样 mask；
- tokenizer revision、chat template revision 与模型权重 revision 是否共同固定。

把新 token append 到词表只会创建新的 id；还必须扩展输入 Embedding 和输出 head，并用数据训练这些行。即使 shape 能加载，随机或未训练的新行也不具备期望语义。重新排序旧词表或改变 merge ranks 更严重：原有 id 指向不同 piece，等同破坏权重契约。

## 长度、性能与公平比较

不要用一个聚合 token/字符比掩盖分布。至少按语言、代码/自然语言、数字、JSON、emoji、OCR 噪声和长尾实体分 slice 报告：

- UTF-8 bytes/token；
- code points/token，并明确它不是 grapheme；
- 每样本 token 数分位数与 context overflow rate；
- `[UNK]` 或 byte fallback 占比；
- encode/decode latency 与峰值内存；
- round-trip、special-token 和 offset 正确性。

相同“128k token context”不代表不同 tokenizer 或语言能承载相同信息量。token 数也不是跨模型公平的语义工作量单位；比较 API 成本或吞吐时，应同时固定原始 workload、模板、输入/输出 token 分布和 tokenizer revision。

注意力的理想 dense prefill 项随序列长度近似二次增长，但完整推理延迟不会只由 (T^2) 决定；decode、KV 读写、kernel 和调度也参与。不能从 token 数下降直接声称端到端延迟按平方下降。

## 安全与失败模式

- 可见字符相同但 Unicode 序列不同，可能绕过基于原始字符串的 allow/deny 规则；
- 零宽字符、双向控制符、同形异义字和异常组合字符会影响审计展示与匹配；
- special token 形似文本不一定被当控制 id，反之允许用户注入真实控制 id 会破坏角色边界；
- 直接按 code point 或 byte 截断可能切断 token、UTF-8 或 chat turn；应在完整模板上按目标 tokenizer 预算并重新验证；
- tokenizer/parser 与安全过滤器使用不同 normalization 会产生 canonicalization gap；
- `errors="replace"` 可用于诊断非法 byte，却会丢失原始身份，不能用于需要精确审计的 canonical artifact。

Tokenizer 不能解决 Prompt Injection；角色 delimiter 也只是模型行为提示。权限、工具参数和副作用仍必须在模型外执行验证。

## 一个可复现 tokenizer 工件应记录什么

最低限度包括：

1. normalizer、pre-tokenizer、trainer/算法和所有 tie-break；
2. 基础字母表、vocabulary、merge ranks 或 piece probabilities；
3. special token 的字符串、id、匹配策略和是否自动添加；
4. decoder、cleanup 行为、chat template 和 padding/truncation policy；
5. 训练语料 snapshot/许可/混合口径与 trainer revision；
6. canonical files 的 hash、模型 revision 和兼容性测试集；
7. 多语言/长尾/安全 slice 的长度、round-trip、offset 和模板回归。

无密钥 hash 只能标识所覆盖文件的 bytes，不能证明训练语料来源合法、文件由可信主体发布或与远端服务真实部署一致。

## 实验与自测

建议先运行 byte baseline，再训练小 BPE，最后加载一个固定 revision 的真实 tokenizer，对中文、英文、代码、JSON、数学、emoji、组合字符和罕见姓名比较。

1. 为什么 BPE 编码必须使用已学习的 merge rank，而不能在每个输入上重数频次？
2. 为什么 byte fallback 可以消除 `[UNK]`，却仍可能让某些语言付出更长序列？
3. `decode(encode(x)) == x` 通过后，还有哪些 normalization、offset、template 与模型兼容性问题未被证明？
4. 为什么扩词表需要同时修改并训练 Embedding/LM head？权重共享时有什么变化？
5. 如何构造测试，确认 merge 不跨文档、special token 不被用户文本伪造、截断不破坏完整 assistant turn？
