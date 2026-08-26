# Transformers Basics：从 bytes 到真实 checkpoint

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想把 tokenizer、attention、generation 和 checkpoint 检查串成可运行实验的开发者。
- **先修**：Python、NumPy/PyTorch 张量基础和 [Transformer](../../core/transformer.md)。
- **首次实践**：BPE → 因果语言模型训练目标 → 注意力与缓存 → 微型模型 → 生成 → 配置。
- **完成信号**：能预测一个中间结果、制造一个失败，并说明每个实验不能外推什么。
- **卡住时**：只运行当前阶段的一条命令，不要先跑完整测试集。

</div>

**项目导航**：[项目索引](../project-index.md) · [实验 1–3](../labs.md#lab-1) · [生成机制](../../core/generation.md) · [证据台账](../../evidence/transformers-controls.md)
{ .doc-nav }

这个项目不是教你记住 Transformers API，而是建立一条可验证链路：

~~~text
raw text
→ tokenizer tokens
→ labels / loss mask
→ attention logits
→ model logits
→ generation protocol
→ checkpoint files
→ runtime observations
~~~

每个阶段都先写预测，再运行，再对账。只有这样，框架输出才会变成你真正理解的知识。

如果你还不能清楚解释 train/test 泄漏、NLL、梯度下降和 accuracy 的区别，先运行一次
`python projects/transformers-basics/ticket_classification_walkthrough.py`。它用一批工单把这四件事串在一起，
只需 CPU 和 Python 标准库；逐步讲解见[机器学习最小闭环](../../foundations/ml-dl.md#ml-minimal-loop)。

## 先认识四级证据

| 层级 | 例子 | 能证明什么 |
|---|---|---|
| 手算与参考实现 | NumPy 实现的 BPE 和注意力 | 给定输入下的数学与状态 |
| 框架小实验 | 随机初始化的微型 PyTorch 模型 | 框架接口、自动求导和缓存路径能执行 |
| 发布材料 | 固定版本的配置和模型卡 | 指定发布物声明了哪些静态字段 |
| 目标 checkpoint | 固定权重和一次前向计算 | 特定环境和输入上的真实执行 |

这四级证据回答不同问题。例如，微型模型的损失下降，说明训练代码可以更新参数；
它没有运行公开模型的权重。配置文件中的注意力头数量只是静态声明；只有实际加载权重并完成前向计算，
才能继续检查目标 checkpoint 的执行路径。即便如此，单条生成结果也只代表这次输入，不是总体质量评测。

项目报告中，每个观察都标明属于哪一级。

## Phase 1：从零训练 byte-level BPE

一段文本会经过三个层次：人看到的是 Unicode 字符，文件中保存的是 UTF-8 字节，模型接收的是 token ID。
字节级分词器（byte-level tokenizer）先把字符串转换成 0–255 的字节，再学习哪些相邻字节经常一起出现，
并把这些字节对合并成新的 token。

若当前序列集合为 \(\mathcal D\)，一次训练选择：

\[
(a^\*,b^\*)=
\arg\max_{(a,b)}
\operatorname{count}_{\mathcal D}(a,b).
\]

关键不是公式，而是四个实现决定：

1. 是否允许相邻字节对跨越文档边界；
2. 多个字节对频数相同时，怎样确定唯一顺序；
3. 同一个字节对连续出现时，怎样执行不重叠合并；
4. 编码新文本时，是否严格按训练得到的合并优先级重放。

### 先预测

对 `banana bandana` 手算第一次最常见的相邻字节对。再加入一篇新文档，预测频数相同时的选择是否改变。

### 再运行

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --text "banana bandana" --text "banana" --sample "bandana"
~~~

检查输出中的合并顺序、token ID、每个 token 展开后的字节，以及编码后再解码能否还原原文。

### 故意破坏

把编码器改成每次重新选择当前最高频的字节对，而不是按照训练阶段固定的优先级合并。寻找一个输入，使两种结果不同。

编码后能够还原原文，只说明这个分词器内部自洽。它是否兼容某个现有 checkpoint，要继续核对词表和合并规则；
中文文本的 token 成本也必须用目标分词器实测。

## Phase 1.5：把同一个 token 序列变成训练目标

```powershell
python projects/transformers-basics/trace_language_model_sample.py
```

脚本让 `你好🙂!` 经过微型 Byte BPE，得到两个文本 token，再加入本实验自己的 `BOS`、`EOS` 和 `PAD`。
输出会逐位置列出模型输入、下一个目标、可见的输入位置和是否参与 loss：

```text
model input: [BOS, 你好🙂, !, EOS]
labels:      [你好🙂, !, EOS, PAD]
loss mask:  [true, true, true, false]
```

先回答两个问题：为什么位置 1 看不见位置 2？为什么最后一个位置可以参与因果计算，却不参与 loss？因果注意力
mask 回答第一个问题，loss mask 回答第二个问题。

脚本停在模型前向计算之前。下一阶段才让注意力和微型模型真正运行。逐位置解释见
[NLP 与语言建模](../../foundations/nlp.md#shift-and-mask)。

## Phase 1.75：换成 Qwen3 自己的 tokenizer

前两个实验中的字节级 BPE、`BOS/EOS/PAD` 和 token IDs 都是教学实现。

下面仍使用同一句中文问题，但把输入交给固定版本 Qwen3-0.6B 的 tokenizer 与 chat template：

~~~powershell
python projects/transformers-basics/trace_qwen3_tokenizer.py --local-files-only
~~~

如果模型文件位于单独目录，改用 `--model-snapshot <path>`。第一次尚未缓存时可以去掉 `--local-files-only`；
程序请求的是完整 commit `c1899de289a04d12100db370d81485cdf75e47ca`。

固定输入在当前模板下得到 29 个 ID。前 3 个是 `<|im_start|>`、`user` 和换行，随后才是消息正文；结尾还包括
`<|im_end|>`、assistant 起始标记，以及禁用 thinking 时模板补上的空 `<think>...</think>` 区间。
因此，不能只编码消息正文来估算 prefill 长度。

输出中的 tokenizer 类名是 `Qwen2TokenizerFast`。Qwen3 checkpoint 复用了这个 Transformers tokenizer 实现，
模型权重仍然属于 Qwen3。

`<think>` 和 `</think>` 展示了另一个边界：它们是 added tokens，却不在当前 `all_special_ids` 中。
因此，“模板控制词”和“解码时会跳过的 special token”需要分开判断。

这一步没有加载权重、计算 logits 或进入 nano-vLLM。它只把真实应用输入接到目标 tokenizer；模型执行和调度要在
后续实验中分别观察。

## Phase 2：手算 causal attention

先看单头缩放点积注意力（scaled dot-product attention）：

\[
S=\frac{QK^\top}{\sqrt{d_h}}+M,\qquad
P=\operatorname{softmax}(S),\qquad
O=PV.
\]

因果掩码（causal mask）\(M\) 让位置 \(t\) 只能看见 \(0\ldots t\)。为了避免指数溢出，
计算 softmax 前先让每行分数减去该行最大值。

### 先用两个 token

为形状 \([T,d_h]\) 的 Q/K/V 取 \(T=2\)：

1. 手算分数矩阵；
2. 写出加入因果掩码后的矩阵；
3. 算出每一行的注意力概率；
4. 验证第一个位置的输出不依赖第二个位置的 value 向量。

### 再验证 cache

自回归解码不需要每一步都重新计算所有历史 K/V。Prefill 阶段处理完整输入，并把历史 K/V 保存到缓存中；
之后每生成一个 token，只计算这个新位置的 Q/K/V。

正确性对照是：

~~~text
prefill + 使用缓存得到的下一 token logits
≈
在相同位置完整重算得到的 logits
~~~

两条路径必须使用相同的绝对位置、掩码、数据类型和 dropout 设置。
缓存中已有历史 token 时，新查询只有一行，却要看到全部历史键；这时不能直接复用从位置 0 开始的方形下三角掩码。

运行 NumPy 参考测试：

~~~powershell
python -m pytest tests/test_attention_numpy.py -q
~~~

### 故意破坏

故意让下一个 token 使用错误的位置编号，或者让掩码忘记已有缓存长度。
比较“使用缓存”和“完整重算”两条路径时，测试应该失败。

这一步检查的是 NumPy 参考实现中的数学语义。FlashAttention 的内核行为、GPU 显存和吞吐需要另外测量。

## Phase 3：理解在线 softmax

普通注意力会完整保存 \(QK^\top\) 分数矩阵。分块在线 softmax（blockwise online softmax）改为逐块读取 key，
并为每一行 query 保存三个中间量：

- 当前已经见过的最大分数 \(m\)；
- 指数和，也就是归一化因子 \(\ell\)；
- 加权 value 的累加值 \(o\)。

新 block 最大值为 \(m_b\) 时：

\[
m'=\max(m,m_b),
\]

\[
\ell'=e^{m-m'}\ell+\sum_j e^{s_j-m'},
\]

\[
o'=e^{m-m'}o+\sum_j e^{s_j-m'}v_j.
\]

新分块可能出现更大的分数。此时旧的累加值必须按照新的行最大值重新缩放，否则前后两个分块不在同一数值基准上。

运行：

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
~~~

先预测把分块大小改为 1 后，最终结果是否变化，以及一次需要保存的最大分数块会变成多大。
然后故意删除旧累加值的重新缩放，观察输出误差。

这个演示仍会计算完整注意力作为参考答案，所以进程本身并不节省全部内存。
它解释在线 softmax 的数学过程，不是在测量 FlashAttention 内核或进程峰值显存。

## Phase 3.5：跟一次 RMSNorm 走到 ATen 图

```powershell
python projects/transformers-basics/trace_rmsnorm_operator_stack.py
```

Transformer 主章已经给出 RMSNorm 数学定义，这一步继续追踪它在框架中的表达。程序先让同一个张量经过
`transpose` 和 `contiguous()`，观察 shape、stride 与 storage；再比较手写分解和
`torch.nn.functional.rms_norm`，最后打印 FX 与 `torch.export` 的 ATen 图。

加入 `--profile` 可以看到当前 PyTorch 的算子事件；在支持 CUDA 的 3070 Laptop 环境还可以加入
`--device cuda`。这些事件不能直接当作稳定的 GPU kernel 清单。完整的抽象层、支持审计和实验记录方法见
[算子计算栈](../../systems/operator-stack.md)与[实验 2D](../labs/lab-2d-operator-stack.md)。

## Phase 4：让同一个样本真正经过 MiniGPT

~~~powershell
python projects/transformers-basics/trace_minigpt_training_step.py
~~~

Phase 1.5 已经得到下面四个训练位置：

~~~text
input:  [BOS, 你好🙂, !, EOS]
target: [你好🙂, !, EOS, PAD]
score:  [yes, yes, yes, no]
~~~

现在，仓库从零实现的 PyTorch MiniGPT 会实际执行词嵌入、因果注意力、前馈网络和词表输出层，得到
`[1,4,268]` 的 logits。脚本逐位置取出目标 token 的对数概率，并复算：

\[
L=\frac{-\log p(\text{你好🙂})-\log p(!)-\log p(\mathrm{EOS})}{3}.
\]

这个值应与模型使用 `ignore_index=-100` 返回的交叉熵一致。`PAD` 位置仍有 logits，但它不进入分子，也不进入
分母。默认运行的两种算法应给出相同的六位小数：`5.585798`。

接下来，脚本执行反向传播和一步 SGD。固定 seed 下，同一小样本的 NLL 会降到 `5.134247`，12 个参数张量都发生
变化。这里观察的是训练接线，不是模型能力：随机模型只在刚见过的三个目标上更新了一次，还没有接受新样本检验。

### 故意破坏

先把第三个有效标签也改成 `-100`，预测平均 NLL 的分母怎样变化。再把 labels 错开两位，程序仍可能正常运行，
但模型学习的目标已经不再是“预测下一个 token”。这说明 finite loss 只能证明计算可执行，不能替你验证监督语义。

## Phase 4.5：再换成 Transformers 的 tiny GPT-2

~~~powershell
python projects/transformers-basics/smoke_tiny.py
~~~

上一步使用本仓库可直接阅读的 MiniGPT；这一步换成 Transformers 的 `GPT2LMHeadModel`，仍然只创建随机初始化
的微型模型，不下载预训练权重。运行时逐项检查：

1. 配置是否构造出预期的模型类；
2. 输入和标签的形状是否一致；
3. 标签向左错一位后，哪些 token 会产生损失；
4. 反向传播后，哪些参数获得了梯度；
5. 优化器更新是否真的改变参数；
6. 评测和生成路径是否都能执行。

### 不要只看 loss 下降

一个固定的小 batch 很容易被模型记住。损失下降的意义，是确认标签、反向传播和参数更新这条训练接线能够工作。
泛化能力、语言能力和真实 checkpoint 的微调效果，需要新的数据和目标模型实验。

可以制造两个反例：把所有标签都设成不参与损失的 `ignore index`，或者在优化器更新前清空梯度。
脚本应该明确报告“有效监督 token 为零”或“参数没有更新”，而不是只打印一个看似正常的损失数字。

## Phase 5：Generation 是多方协议

一次生成何时停止，至少涉及四层设置：

~~~text
分词器中特殊 token 的 ID
+ 模型配置引用的默认 ID
+ 生成配置中的停止设置
+ 本次调用传入的覆盖参数
~~~

分词器决定某个 ID 对应什么 token；模型配置和生成配置再引用这些 ID。
如果当前调用显式传入了停止参数，框架还会按自己的优先级规则合并或覆盖默认值。

先检查两个离线协议样例：

~~~powershell
python projects/transformers-basics/inspect_generation_protocol.py projects/transformers-basics/protocols/aligned-superset-eos.example.json
python projects/transformers-basics/inspect_generation_protocol.py projects/transformers-basics/protocols/drift-out-of-range.example.json
~~~

你要找出开头、结尾和填充 token 的 ID 是否一致，是否超出词表范围。
如果生成配置允许多个结束 token，还要判断这是有意扩充，还是不同配置已经发生漂移。

再运行真实框架小实验：

~~~powershell
python projects/transformers-basics/generation_runtime_control.py
~~~

脚本使用预先安排好的 token 序列，把“遇到结束 token”和“达到长度上限”分开。
运行前先预测：本次调用覆盖结束 token 后，配置中原来的结束 token 是否还会触发停止？

框架的 `generate` 返回一串 token，云 API 则可能另外返回结束原因。应用设置的停止字符串、模型生成的结束 token、
长度上限和网络传输被取消，是四种不同事件。只有分别记录，才能解释一次生成为什么结束。

## Phase 6：从 config 建立容量假设

只读取配置文件时，可以回答下面这些静态问题：

- 隐藏维度、层数、注意力头数、中间层宽度和词表大小；
- 标准多头注意力（MHA）或分组查询注意力（GQA）下，每个头可能有多宽；
- 理想情况下 KV cache 需要保存多少数据；
- 配置是否声明了 MoE、MLA 或自定义模型代码；
- 现有信息不足时，哪些公式不应继续套用。

标准 dense K/V 的理想 payload 为：

\[
M_{KV}=B\cdot L\cdot 2\cdot H_{kv}\cdot d_h
\cdot n_{\mathrm{layers}}\cdot b.
\]

运行正反例：

~~~powershell
python projects/transformers-basics/inspect_config.py projects/transformers-basics/configs/standard-gqa.example.json --tokens 4096 --batch-size 1 --element-bytes 2
python projects/transformers-basics/inspect_config.py projects/transformers-basics/configs/mla-moe.example.json --tokens 4096
~~~

标准 GQA 样例会给出容量估算。遇到已知的 MLA 配置字段时，程序会停止套用 GQA 公式，并说明缺少哪些信息。

这个公式只计算理想的 KV 数据量。模型权重、内存分配器开销、分页对齐、量化缩放因子、内核工作区和临时张量，
都会继续占用显存，因此计算结果不是运行时的显存峰值。

## Phase 7：检查真实 checkpoint，不先加载权重

选择具体的模型 ID，并使用完整提交哈希固定版本：

~~~powershell
python projects/transformers-basics/inspect_checkpoint.py <model-id> --revision <full-commit-hash>
~~~

检查程序会输出：

- 请求的版本和仓库实际解析到的版本；
- 模型类型、架构名、词表大小和上下文配置；
- 特殊 token ID 和模板渲染后的 Prompt；
- 标准 attention/GQA 配置是否足以推导 KV 布局；
- 可用 generation config 与 tokenizer/model config 的特殊 token 关系。

这一步可以发现 Base 模型没有对话模板、特殊 token 配置漂移，或者模型依赖自定义注意力实现。
程序使用 `trust_remote_code=False`，也不读取权重分片、模型卡或许可证。权重是否完整、前向计算是否正确和模型质量
都需要下一阶段验证。

## Phase 8：真实权重是选修

仓库提供固定 Qwen 小模型 CPU 运行：

~~~powershell
python projects/transformers-basics/run_target_checkpoint.py --local-files-only
~~~

只有模型快照已经完整缓存时，才能使用 `--local-files-only`。首次下载方式、文件总量和固定版本见
[证据台账](../../evidence/transformers-controls.md)。

运行前先预检磁盘、内存、网络和许可。运行后核对：

1. 加载器实际读取了哪些文件，它们的哈希是否匹配；
2. 参数分别使用什么数据类型，位于哪个设备；
3. Prefill 输出的 logits 形状是否正确，数值是否有限；
4. 使用 KV cache 与完整重算得到的下一 token logits 是否接近；
5. 固定的贪心生成结果，以及它为什么停止。

仓库使用一条英文算术 Prompt 作为冒烟输入。通过后，可以确认这条输入在当前环境中走通了加载和生成路径。
中文、长上下文、GPU 或 vLLM 路径需要各自的测试；总体质量和生产安全则需要系统评测。

## 推荐运行顺序 { #run }

### 第一次：90 分钟 CPU 路线

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/trace_language_model_sample.py
python projects/transformers-basics/online_softmax_demo.py
python projects/transformers-basics/trace_rmsnorm_operator_stack.py
python projects/transformers-basics/trace_minigpt_training_step.py
python projects/transformers-basics/smoke_tiny.py
python projects/transformers-basics/generation_runtime_control.py
~~~

每条命令前写一条预测，运行后保存一条失败或边界解释。

### 第二次：静态 checkpoint 路线

~~~powershell
python projects/transformers-basics/trace_qwen3_tokenizer.py --local-files-only
python projects/transformers-basics/verify_release_evidence.py
python projects/transformers-basics/inspect_config.py projects/transformers-basics/configs/standard-gqa.example.json --tokens 4096
~~~

离线发布检查只读取仓库中固定的清单和快照。它不会重新下载上游文件；上游地址、固定版本和文件哈希记录在证据页。

### 第三次：机制选修

在[实验目录](../labs.md#lab-2)中选择一个：

- 激活修补（activation patching）：设计干净输入、受扰动输入和正负对照；
- MoE 路由：观察 top-k 专家选择、容量限制和跨设备 all-to-all 通信；
- 算子计算栈：比较非连续布局、FX/ATen 图和当前 backend 的 profiler 事件；
- 局部 INT4：区分被量化权重的局部误差与整模型质量；
- 真实 checkpoint：对账 KV cache、生成结果和资源用量。

不要在第一次学习同时做四项。

## 你应保存哪些实验记录 { #artifact }

~~~text
experiment manifest
├── code revision and command
├── environment and hardware
├── input identity
├── prediction
├── raw output
├── one negative case
├── interpretation
└── claims not supported
~~~

截图可以辅助展示，但机器可读输出和失败样例更便于别人复核。

## 故障定位

### Loss 不下降

先打印真正参与监督的 token 数，并逐位置检查标签是否正确左移、填充位置是否被掩码。
然后确认优化器拿到了目标参数、梯度中没有 `NaN` 或无穷值，学习率也不是零。最后尝试让单个 batch 过拟合。

### Cached/full logits 不一致

先切换到评测模式并关闭 dropout，再核对位置 ID 和注意力掩码是否包含已有缓存长度。
最后确认两条路径的数据类型相同，而且比较的是同一个序列位置。

### Generation 提前或不停止

打印模板处理后的完整输入 ID、所有结束 token、填充 token 和最大新增 token 数。
同时保存本次调用覆盖了哪些默认参数，以及模型逐步生成的 token 轨迹。

### Config 数字看起来不合理

先确认每个配置字段是否采用标准架构的语义。遇到自定义模型代码、MLA 或无法识别的张量布局时，
停止套用标准公式，转而阅读目标模型实现。

### 保存的报告通过，但新运行结果不同 { #recorded-report }

保存的报告只记录历史运行。先比较两次运行使用的代码版本、环境、输入文件、具体输入和数值容差。
验证程序通过表示旧报告内部一致，不等于当前环境已经重新执行并得到同样结果。

## 项目完成标准

你能提交：

- BPE 合并顺序和“编码—解码”还原结果；
- 完整重算与缓存注意力的正负对照；
- 微型模型的标签、梯度和参数更新记录；
- 生成协议中结束 token 与长度覆盖实验；
- 一份配置字段“可以推导 / 信息不足”的对照表；
- 可选的真实 checkpoint 文件清单与冒烟结果；
- 每项结论的证据等级和不可外推边界。

## 面试与作品集

不要写“从零实现 Transformers 并部署大模型”，除非你真的做到了对应范围。

更诚实的表述是：

> 从原始字节实现 BPE 参考版本，并用 NumPy 对账因果注意力、KV cache 和在线 softmax；
> 使用微型 PyTorch 模型验证训练与生成路径，再为固定 checkpoint 分别保存配置、分词器、权重和运行结果证据。

面试时应能回答：

1. Byte、Unicode 字符与 token 有什么区别？
2. 使用 KV cache 解码时，为什么仍需要正确的绝对位置？
3. 在线 softmax 为什么要按照新的最大值重新缩放旧累加值？
4. 模型生成结束 token 与应用命中停止字符串有什么差别？
5. 根据配置估算的 KV 数据量与 GPU 峰值显存之间还缺哪些项？
6. 保存的历史报告为什么不等于本轮已经真实运行？

## 下一步

- 想深入机制：回到[实验 2](../labs.md#lab-2)做 activation patching 或 MoE。
- 想微调：[Single-GPU Finetuning](single-gpu-finetuning.md)。
- 想部署：[Inference Serving](inference-serving.md)。
- 想核对精确结果：[Transformers 证据台账](../../evidence/transformers-controls.md)。
