# Transformers Basics：从 bytes 到真实 checkpoint

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想把 tokenizer、attention、generation 和 checkpoint 检查串成可运行实验的开发者。
- **先修**：Python、NumPy/PyTorch 张量基础和 [Transformer](../../core/transformer.md)。
- **首次实践**：BPE → attention/cache → tiny model → generation → config；真实权重是选修。
- **完成信号**：能预测一个中间结果、制造一个失败，并说明每个实验不能外推什么。
- **卡住时**：只运行当前阶段的一条命令，不要先跑完整测试集。

</div>

**项目导航**：[项目索引](../project-index.md) · [实验 1–3](../labs.md#lab-1) · [生成机制](../../core/generation.md) · [证据台账](../../evidence/transformers-controls.md)
{ .doc-nav }

这个项目不是教你记住 Transformers API，而是建立一条可验证链路：

~~~text
raw text
→ tokenizer tokens
→ attention logits
→ model logits
→ generation protocol
→ checkpoint files
→ runtime observations
~~~

每个阶段都先写预测，再运行，再对账。只有这样，框架输出才会变成你真正理解的知识。

## 先认识四级证据

| 层级 | 例子 | 能证明什么 |
|---|---|---|
| 手算与参考实现 | NumPy BPE/attention | 给定输入下的数学与状态 |
| 框架小实验 | 随机 tiny PyTorch model | API、autograd、cache 路径能执行 |
| 发布证据 | 固定 config/model card bytes | 指定 artifact 声明与静态字段 |
| 目标 checkpoint | 固定 weights + forward | 特定环境和输入上的真实执行 |

一个 tiny model loss 下降，不能证明公开 checkpoint 可训练；config 的 head 数也不能证明 weights 已加载；一条真实 generation 不能证明总体质量。

项目报告中，每个观察都标明属于哪一级。

## Phase 1：从零训练 byte-level BPE

Unicode 字符、UTF-8 bytes 和 model tokens 是三个层次。Byte-level tokenizer 先把任意字符串表示成 0–255 的 bytes，再学习常见相邻 pair 的 merge。

若当前序列集合为 \(\mathcal D\)，一次训练选择：

\[
(a^\*,b^\*)=
\arg\max_{(a,b)}
\operatorname{count}_{\mathcal D}(a,b).
\]

关键不是公式，而是四个实现决定：

1. 是否跨 document 统计 pair；
2. 频数相同怎样 tie-break；
3. 同一 pair 的相邻出现怎样做非重叠 merge；
4. 编码时是否按已学习的 merge rank 重放。

### 先预测

对 banana bandana 手算第一次最常见 pair。再预测加入一个新 document 后，tie-break 是否改变。

### 再运行

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --text "banana bandana" --text "banana" --sample "bandana"
~~~

检查输出中的 merges、token IDs、byte expansion 和 round trip。

### 故意破坏

把 encoder 改成每次选择当前最高频 pair，而不是按训练得到的 rank。寻找一个输入，使两种结果不同。

你应能解释：round trip 只证明本 tokenizer 自洽，不证明它兼容任何现有 checkpoint，也不能据此推断中文 token 成本。

## Phase 2：手算 causal attention

单头 scaled dot-product attention：

\[
S=\frac{QK^\top}{\sqrt{d_h}}+M,\qquad
P=\operatorname{softmax}(S),\qquad
O=PV.
\]

Causal mask \(M\) 让位置 \(t\) 只能看见 \(0\ldots t\)。稳定 softmax 先减去每行最大值。

### 先用两个 token

为形状 \([T,d_h]\) 的 Q/K/V 取 \(T=2\)：

1. 手算 score matrix；
2. 写出 causal mask 后的矩阵；
3. 算每行 probability；
4. 验证第一位置不依赖第二个 value。

### 再验证 cache

Autoregressive decode 不需要每步重算所有历史 K/V。Prefill 保存历史 cache，下一步只计算新 token 的 Q/K/V。

正确性对照是：

~~~text
prefill + cached next-token logits
≈
full causal recompute at the same position
~~~

要对齐 absolute position、mask、dtype 和 dropout。Past length 存在时，不能直接复用从零开始的方阵下三角 mask。

运行 NumPy 参考测试：

~~~powershell
python -m pytest tests/test_attention_numpy.py -q
~~~

### 故意破坏

让下一 token 使用错误 position，或忘记把 past length 加入 mask。测试应在 cached/full comparison 中失败。

这一步证明 NumPy 语义，不证明 FlashAttention、GPU 内存或吞吐。

## Phase 3：理解在线 softmax

完整 attention 会物化 \(QK^\top\)。Blockwise online softmax 按 key blocks 处理，并为每个 query row 维护：

- running maximum \(m\)；
- normalizer \(\ell\)；
- value accumulator \(o\)。

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

旧 accumulator 必须按新的 row maximum 重新缩放。

运行：

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
~~~

先预测把 block size 改为 1 后，数值结果和最大 score tile 怎样变化。然后故意删除旧 accumulator rescale，观察误差。

这个 demo 为对照仍会计算 dense reference；它不是 FlashAttention kernel 或进程峰值显存测量。

## Phase 4：用 tiny model 验证框架接线

~~~powershell
python projects/transformers-basics/smoke_tiny.py
~~~

脚本从 config 创建随机 tiny causal LM，不下载权重。你要检查：

1. config 是否构造出预期 class；
2. input/labels shape 是否一致；
3. causal shift 后哪些 token 产生 loss；
4. backward 后哪些 parameters 有 gradient；
5. optimizer step 是否真的改变参数；
6. eval/generate 路径是否可执行。

### 不要只看 loss 下降

固定小 batch 可以被记住。Loss 下降证明 training plumbing 工作，不证明泛化、语言能力或真实 checkpoint 微调成功。

故意把 labels 全部 mask 为 ignore index，或在 optimizer 前清掉 gradient。脚本应暴露 zero supervised tokens 或参数未更新，而不是只输出一个数字。

## Phase 5：Generation 是多方协议

最终停止由至少四个来源共同决定：

~~~text
tokenizer special tokens
+ model config
+ generation config
+ call-level overrides
~~~

先检查两个离线协议样例：

~~~powershell
python projects/transformers-basics/inspect_generation_protocol.py projects/transformers-basics/protocols/aligned-superset-eos.example.json
python projects/transformers-basics/inspect_generation_protocol.py projects/transformers-basics/protocols/drift-out-of-range.example.json
~~~

你要找出 BOS/EOS/PAD 是否一致、是否越过 vocab，以及 generation EOS superset 是有意设计还是 drift。

再运行真实框架小实验：

~~~powershell
python projects/transformers-basics/generation_runtime_control.py
~~~

它用受控 token plan 隔离 EOS 与 length cap。先预测 call-level EOS override 后，原 EOS token 是否还会停止。

框架 generate 返回 token sequence，不等于云 API provider 的 finish reason。Application stop string、token EOS、长度上限和 transport cancellation 也必须分开。

## Phase 6：从 config 建立容量假设

Config-only 检查适合回答：

- hidden/layer/head/intermediate/vocab 等静态字段；
- standard MHA/GQA 下的候选 head dimension；
- 理想 KV payload；
- 是否出现 MoE、MLA 或 custom code markers；
- 哪些公式必须拒绝。

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

标准 GQA 样例应给出 estimate；出现已知 MLA markers 时，程序应停止套用 GQA 公式并说明原因。

公式不含 weights、allocator、page alignment、quantization scales、workspace 和 temporary tensors，因此不是显存峰值。

## Phase 7：检查真实 checkpoint，不先加载权重

选择具体 model ID 和 immutable revision：

~~~powershell
python projects/transformers-basics/inspect_checkpoint.py <model-id> --revision <full-commit-hash>
~~~

先保存：

- requested 与 resolved revision；
- config、tokenizer、template、generation config；
- special-token IDs 和 rendered prompt；
- weight shard inventory；
- runtime 与 trust_remote_code policy；
- license/model-card review。

这一步可以发现 Base 模型没有 chat template、special tokens 漂移或自定义 attention。它不能证明权重完整、forward 正确或模型质量。

## Phase 8：真实权重是选修

仓库提供固定 Qwen 小模型 CPU 运行：

~~~powershell
python projects/transformers-basics/run_target_checkpoint.py --local-files-only
~~~

只有 snapshot 已经完整缓存时使用 local-files-only；首次下载、文件总量和精确 revision 见[证据台账](../../evidence/transformers-controls.md)。

运行前先预检磁盘、内存、网络和许可。运行后核对：

1. Loader 实际读取的文件与 hash。
2. Parameter/dtype/device inventory。
3. Prefill logits shape 和 finite values。
4. Cached/full next-token logits。
5. Fixed greedy continuation 与 stop。

一次英文算术 prompt 只证明这个 input 的执行路径，不证明中文、长上下文、GPU/vLLM、总体质量或生产安全。

## 推荐运行顺序 { #run }

### 第一次：90 分钟 CPU 路线

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/online_softmax_demo.py
python projects/transformers-basics/smoke_tiny.py
python projects/transformers-basics/generation_runtime_control.py
~~~

每条命令前写一条预测，运行后保存一条失败或边界解释。

### 第二次：静态 checkpoint 路线

~~~powershell
python projects/transformers-basics/verify_release_evidence.py
python projects/transformers-basics/inspect_config.py projects/transformers-basics/configs/standard-gqa.example.json --tokens 4096
~~~

离线 release verification 只核对固定本地 manifest/snapshot；显式 upstream 下载与精确 hash 在证据页说明。

### 第三次：机制选修

在[实验目录](../labs.md#lab-2)中选择一个：

- activation patching：设计 clean/corrupt pair 和正负对照；
- MoE routing：从 top-k/capacity 到 all-to-all；
- 局部 INT4：区分 selected weight error 与整模型质量；
- 真实 checkpoint：对账 cache、generation 和资源。

不要在第一次学习同时做四项。

## 你应保存哪些 artifact

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

截图可以辅助展示，但机器可读输出和失败样例才便于复核。

## 故障定位

### Loss 不下降

检查 supervised token 数、shift、padding mask、optimizer 参数集合、gradient finite 和 learning rate。先尝试让一个 batch 过拟合。

### Cached/full logits 不一致

检查 eval mode、dropout、position IDs、attention mask、past length、dtype 与比较位置。

### Generation 提前或不停止

打印最终 input IDs、EOS set、PAD、max new tokens、call overrides 和生成 token trace。

### Config 数字看起来不合理

确认字段语义属于标准 architecture。遇到 custom code、MLA 或 unknown layout 时停止公式推导。

### Recorded report 通过但新运行不同

Recorded artifact 只证明历史运行报告内部一致。比较 code、environment、files、input 和 tolerance，不能把 verifier pass 当成重新执行。

## 项目完成标准

你能提交：

- BPE merge 与 round-trip 对账；
- dense/cached attention 的正负对照；
- tiny model 的 labels、gradient 与 parameter update；
- generation protocol 的 EOS/length override 实验；
- 一份 config 可推导/必须拒绝表；
- 可选的真实 checkpoint manifest 与 smoke；
- 每项结论的证据等级和不可外推边界。

## 面试与作品集

不要写“从零实现 Transformers 并部署大模型”，除非你真的做到了对应范围。

更诚实的表述是：

> 从 raw bytes 实现 BPE reference，以 NumPy 对账 causal/cache/online attention；用 tiny PyTorch 模型验证训练与
> generation 路径，并为固定 checkpoint 建立 config、tokenizer、权重和 runtime 分层证据。

面试时应能回答：

1. Byte、Unicode 字符与 token 有什么区别？
2. Cached decode 为什么需要 absolute position？
3. Online softmax 为什么要 rescale 旧 accumulator？
4. Generation 的 EOS 与 application stop 有什么差别？
5. Config estimate 与 GPU peak memory 之间缺哪些项？
6. Recorded report 为什么不等于本轮真实运行？

## 下一步

- 想深入机制：回到[实验 2](../labs.md#lab-2)做 activation patching 或 MoE。
- 想微调：[Single-GPU Finetuning](single-gpu-finetuning.md)。
- 想部署：[Inference Serving](inference-serving.md)。
- 想核对精确结果：[Transformers 证据台账](../../evidence/transformers-controls.md)。
