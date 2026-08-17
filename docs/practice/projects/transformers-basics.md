# Transformers Basics

**项目导航**：[返回项目索引](../project-index.md) · [Transformer 原理](../../core/transformer.md) · [生成机制](../../core/generation.md) · [实验 1–3](../labs.md#lab-1)
{ .doc-nav }

这不是一页“运行几个脚本”的索引，而是一条从 tokenizer、attention 和生成协议走到真实 checkpoint、因果干预与 MoE distributed controls 的可执行学习路径。所有数字都来自仓库当前 fixture 或固定 recorded artifact；先判断证据层级，再解释结果。

## 学习目标与先修

完成本项目后，你应该能：

- 从 raw bytes 推导 byte-level BPE 的训练与编码过程；
- 解释 causal self-attention、RMSNorm、RoPE、GQA、KV cache 与 online softmax 的核心不变量；
- 区分“随机 tiny 模型跑通”“config 被检查”“发布 artifact 被固定”“目标权重真实执行”；
- 审计 tokenizer / model config / generation config 的 special-token 与停止协议；
- 解释 activation patching 的干预量、metric、正负对照和不可外推结论；
- 从 top-k routing 逐步推到 capacity、expert ownership、all-to-all、反向传播和 optimizer step；
- 把上述机制组织成可复查的求职作品，而不是堆砌框架名。

建议先读 [Transformer](../../core/transformer.md) 中的张量形状与残差结构，并准备 Python 3.12、NumPy、PyTorch 和 Transformers。双进程 MoE controls 需要当前 PyTorch build 提供 Gloo；本页所有已录制运行均为 CPU，不包含 CUDA/NCCL。

## 先建立证据梯子

本项目故意同时保留四种强度不同的证据：

~~~mermaid
flowchart LR
    A["机制 oracle<br/>NumPy / authored math"] --> B["框架 control<br/>随机 tiny PyTorch"]
    B --> C["发布证据<br/>immutable config / model card"]
    C --> D["目标 checkpoint<br/>固定 Qwen 真实权重"]
~~~

四层证据不能合并：

| 层级 | 本项目实际执行 | 能证明 | 不能证明 |
|---|---|---|---|
| 机制 oracle | byte BPE、attention、routing | 给定输入与 authored 语义的数学/账本正确 | 真实模型或生产 kernel |
| 框架 control | tiny GPT-2、hooks、Gloo | 当前框架 API、autograd 或 collective 路径被调用 | 公开 checkpoint 的质量、GPU 行为 |
| 发布证据 | 固定 URL/revision/bytes/config 投影 | 指定 artifact 与审阅字段相符 | 权重已加载、发布者签名、有效上下文 |
| 目标权重 | 固定 Qwen CPU FP32 control | 审阅 snapshot 的特定 forward/cache/generate/hook 观察 | 总体质量、生产性能、其他 runtime 等价 |

报告里出现 `passed`、loss 下降或 hash 相同，只能在所在行的范围内解释。不同 control 不能拼接成 CUDA、完整 expert parallel、模型质量或生产安全已经成立。

## 1. 从零训练 byte-level BPE

### 1.1 为什么从 bytes 开始

Unicode 字符、UTF-8 bytes 和 model tokens 是三个不同层次。这个 reference 的基础词表固定为 256 个 raw byte IDs，因此任意 UTF-8 字符串都有可逆表示；训练只学习把高频相邻 token pair 合成新 token。

设当前 token 序列集合为 \(\mathcal D\)，一次 merge 选择：

\[
(a^\*,b^\*)=\arg\max_{(a,b)}\operatorname{count}_{\mathcal D}(a,b).
\]

本实现只在单篇 document 内统计相邻 pair，不跨 document 边界；频次相同按 ID pair 字典序稳定打破平局。编码时必须按学习到的 merge rank 重放，而不是重新按当前最高频 pair 贪心训练。

### 1.2 运行与读取输出

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/train_byte_bpe.py `
  --text "banana bandana" `
  --text "banana" `
  --sample "bandana"
~~~

仓库默认 corpus 的本次实跑结果是：

- `implementation=about-llm.byte-bpe-reference.v1`；
- 请求词表 280，实际 `actual_vocab_size=273`；
- 学到 17 次 merge；
- 四个样本 token 数为 `6/10/17/9`；
- 中文和英文样本的 UTF-8 round trip 全部为真。

实际词表小于请求值不是失败：达到 `min_pair_frequency=2` 后没有更多合法 pair。每条 merge 同时输出 byte expansion 和 UTF-8 preview；preview 中的替换字符只说明单个 token bytes 未必独立构成完整 Unicode 字符，不影响 byte-level decode 的可逆性。

### 1.3 这个 tokenizer 还缺什么

它没有 normalization、pre-tokenizer、special tokens、offset map、added-token policy、chat template 或 checkpoint compatibility。小型 authored corpus 的 token 数也不能推成中文压缩率、计费量或目标模型上下文长度。连接真实 checkpoint 时，必须加载该 revision 的 tokenizer 并保存实际 prompt token IDs。

## 2. Attention：先锁定数学，再谈 kernel

### 2.1 Dense causal attention

单头 scaled dot-product attention 为：

\[
S=\frac{QK^\top}{\sqrt{d_h}}+M,\qquad
P=\operatorname{softmax}(S),\qquad
O=PV.
\]

causal mask \(M\) 对不可见位置写 \(-\infty\)。带 past cache 的 decode 不能简单用方阵下三角 mask：当前 query 的绝对位置需要加上 past length。仓库 NumPy oracle 同时检查 prefill、逐 token cache 与 full causal recompute 的一致性。

稳定 softmax 先减行最大值：

\[
\operatorname{softmax}(x)_i=
\frac{\exp(x_i-\max_jx_j)}
{\sum_j\exp(x_j-\max_jx_j)}.
\]

完全被 mask 的行没有合法概率分布，reference 选择 fail closed，而不是悄悄返回 NaN 或均匀分布。

### 2.2 RMSNorm、RoPE 与 GQA

RMSNorm 不做 mean subtraction：

\[
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot w.
\]

RoPE 对每对通道施加二维旋转；同一 position shift 同时作用于 Q/K 时，注意力相关的相对位置结构保持。测试检查旋转后的范数以及共同平移后的 Q/K dot product。

GQA 令 query heads 数 \(H_q\) 大于 K/V heads 数 \(H_{kv}\)。解释性 oracle 显式 repeat K/V 来和普通 MHA 展开对账；生产 runtime 通常不会把这种物理复制当作理想实现。统一的 `AutoModel` API 不代表 Llama、Qwen、DeepSeek 或任意 checkpoint 的 head layout、RoPE、cache layout 相同。

### 2.3 Blockwise online softmax

完整 \(QK^\top\) 需要 \(O(L_qL_k)\) score elements。按 key blocks 处理时，对每一行维护 running maximum \(m\)、normalizer \(\ell\) 和未归一化 value accumulator \(o\)。新块最大值为 \(m_b\) 时：

\[
m'=\max(m,m_b),
\]

\[
\ell'=e^{m-m'}\ell+
\sum_j e^{s_j-m'},
\]

\[
o'=e^{m-m'}o+
\sum_j e^{s_j-m'}v_j.
\]

最终输出 \(o'/\ell'\)。重新缩放旧 accumulator 是关键；遗漏它会在 block 最大值上升时产生系统误差。

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
python -m pytest tests/test_attention_numpy.py -q
~~~

固定 `query_length=5`、`key_length=7`、`block_size=3` 的实跑结果：

| 观察项 | 值 |
|---|---:|
| key blocks | 3 |
| 最大 logical score tile | 15 elements |
| dense score | 35 elements |
| online vs dense max abs error | `2.220446049250313e-16` |
| online 输出 finite | true |

Demo 为比较而另外物化 dense reference，所以 `15 vs 35` 不是整个进程的峰值内存测量。Float64 NumPy recurrence 也不是 FlashAttention kernel；它没有测 HBM traffic、workspace、CUDA、吞吐或延迟。

## 3. Tiny Transformers：验证框架接线

`smoke_tiny.py` 从 `GPT2Config` 创建随机模型，不下载权重。固定 batch 训练 12 步后执行 greedy `generate()`：

~~~powershell
python projects/transformers-basics/smoke_tiny.py
~~~

当前实跑得到 27,008 个参数、纯参数存储 108,032 bytes，loss `3.4888949394226074→2.1879992485046387`，生成 shape 为 `[2,5]`。

这个结果适合回答五个工程问题：

1. config 是否真的构造出目标类；
2. `labels=input_ids` 是否进入 causal-LM shift/loss；
3. backward 与 optimizer 是否更新参数；
4. `train()/eval()` 和 generation 路径是否可调用；
5. 参数量账本是否与实际 model parameters 一致。

它不回答语言能力、泛化、公开权重身份或显存峰值。纯参数 bytes 不含 gradients、optimizer state、activations、KV cache、allocator 与 workspace；tiny loss 下降也不等于目标模型训练已跑通。

## 4. Generation 是三方协议，不只是 `generate()`

### 4.1 静态 special-token 对账

实际停止语义至少受 tokenizer、model config、generation config 和调用级 kwargs 共同影响。先检查 authored 快照：

~~~powershell
python projects/transformers-basics/inspect_generation_protocol.py `
  projects/transformers-basics/protocols/aligned-superset-eos.example.json
python projects/transformers-basics/inspect_generation_protocol.py `
  projects/transformers-basics/protocols/drift-out-of-range.example.json
~~~

第一份 fixture 中 tokenizer/model EOS 为 `{2}`，GenerationConfig EOS `{2,3}` 是前者的 superset；额外停止 token 可能是有意设计，因此只报告差异。第二份把 BOS/EOS/PAD 改成 disjoint IDs，并让 PAD=9 超出 8-token vocab，必须显式暴露。

`PAD=EOS` 不是天然错误，三方 ID 一致也不证明实际调用没有 override。检查器不猜 `max_length` 与 `max_new_tokens` 优先级，也不把 authored JSON 冒充真实 checkpoint snapshot。

### 4.2 真实 `GenerationMixin.generate()` 控制

~~~powershell
python projects/transformers-basics/generation_runtime_control.py
~~~

脚本在随机 3,824-parameter tiny GPT-2 上真实调用 `generate()`，再用 authored `LogitsProcessor` 强制每一步只有一个 token 可选，从而隔离停止控制流：

| Case | Config | 调用级覆盖 | 强制序列 | 结果 |
|---|---|---|---|---|
| EOS set | EOS `{2,3}` | 无 | `[4,3]` | token 3 后停止 |
| EOS override | EOS `{2,3}` | EOS=5 | `[3,5]` | 3 不停，5 停 |
| length cap | max new=5 | max new=2 | `[4,6]` | 无 EOS，恰好 2 token |

测试还确认 caller-owned `GenerationConfig` 没被 mutation。报告中的 finish reason 是根据受控 token plan 与 EOS/length **推断**的；当前 Transformers 返回对象没有 provider 风格 finish reason。

这证明当前安装版本的三条 API control flow，不证明自然 logits、真实 tokenizer/chat template、vLLM/provider precedence、stop-string tokenization、GPU 行为或生成质量。

## 5. Config、KV 账本与发布证据

### 5.1 Config-only 检查

`inspect_config.py` strict-load 本地 JSON，拒绝 duplicate key、`NaN`/`Infinity` 和非 object 根节点。只有标准 dense K/V 所需字段明确且没有已知 MLA markers 时，才估算理想 K/V tensor payload：

\[
\text{bytes}
=B\cdot L\cdot 2\cdot H_{kv}\cdot d_h
\cdot n_{\text{layers}}\cdot b.
\]

其中 2 表示 K 与 V，\(b\) 是 element bytes。运行：

~~~powershell
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/standard-gqa.example.json `
  --tokens 4096 --batch-size 1 --element-bytes 2
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/mla-moe.example.json `
  --tokens 4096
~~~

`configs/*.example.json` 的 model type 都带 `authored`，只是公式 fixture。标准 GQA 结果 536,870,912 bytes；MoE-GQA 在其参数下为 402,653,184 bytes。MoE marker 不自动改变标准 attention K/V 公式，也不能给出 total/active parameter 数。MLA marker 出现时 verifier 明确要求 `standard dense K/V formula must not be applied`。

公式不含 block 对齐、allocator metadata、量化 scales、workspace、临时张量或权重，因此不是 VRAM 峰值。`max_position_embeddings` 也不是有效上下文或质量证明。

### 5.2 Immutable release evidence

~~~powershell
# 完全离线：manifest、本地 snapshot、投影和公式
python projects/transformers-basics/verify_release_evidence.py

# 显式联网：重下 immutable artifacts 并核对 bytes
python projects/transformers-basics/verify_release_evidence.py --verify-upstream
~~~

`release-evidence/manifest.json` 分别绑定：

- Llama 3.2 text-only：固定 Meta 官方 model-card commit、byte hash 和六段 exact fragments；结论只标成 vendor-reported claim；
- Qwen2.5-0.5B-Instruct：固定官方组织 immutable config、完整 semantic snapshot 与标准 GQA KV 投影；
- DeepSeek-V3：固定 immutable config，并要求 MLA+MoE markers 触发标准 K/V 公式拒绝。

本次离线验证的 manifest fingerprint 为 `sha256:74166133…53b`，projection fingerprint 为 `sha256:40b3fe7b…e4638`，`upstream_verified=false`。这表示当前只复核本地审阅快照与 manifest；只有显式 `--verify-upstream` 的那次运行才重新下载 bytes。

Model card claim、config field、artifact identity 与 runtime observation 是四类证据。无密钥 SHA-256 不认证发布者；config hash 不证明权重与 config 匹配，也不证明模型类、许可、有效上下文、质量、显存、性能或安全。

### 5.3 检查任意真实 checkpoint，但不加载权重

~~~powershell
python projects/transformers-basics/inspect_checkpoint.py `
  Qwen/Qwen2.5-0.5B-Instruct `
  --revision <immutable-commit-hash>
~~~

该入口只取 config、tokenizer 和可用 generation config，输出 requested/resolved revision metadata、canonical config fingerprint、attention/MoE/MLA contract、chat-template text/token IDs 和 special-token 对账。默认 `trust_remote_code=False`。

Resolved commit metadata 不是签名；base tokenizer 没有 chat template、仓库没有独立 generation config 都可能是合法状态。`unavailable_or_load_error` 也可能来自网络、认证或 cache，不能武断解释成文件不存在。

## 6. 固定 Qwen 真实权重 control

Release verifier 没有加载权重；下面的入口才执行目标 checkpoint：

~~~powershell
# 首次允许匿名下载固定 revision 的 7 个文件，约 1 GB
python projects/transformers-basics/run_target_checkpoint.py

# cache 已完整时强制不联网
python projects/transformers-basics/run_target_checkpoint.py --local-files-only
~~~

Manifest 固定：

- model：`Qwen/Qwen2.5-0.5B-Instruct`；
- revision：`7ae557604adf67be50417f59c2c2f167def9a775`；
- 7 个 selected files：999,586,347 bytes；
- `model.safetensors`：988,097,824 bytes，SHA-256 `fdf756fa…fb7fe`；
- runtime：CPU / FP32 / eager / `trust_remote_code=False`。

Control 先逐文件重算 bytes/hash，再从已验证的本地 snapshot 加载 `Qwen2ForCausalLM`。它冻结参数并进入 inference mode，执行 prefill、带 `past_key_values` 的 cached step、同位置 full recompute 和 greedy `generate()`。

Recorded verifier 当前锁定：

| 观察项 | 值 |
|---|---:|
| parameters | 494,032,768 |
| FP32 parameter storage | 1,976,131,072 bytes |
| trainable parameters | 0 |
| prompt tokens | 31 |
| prefill logits | `[1,31,151936]` |
| continuation IDs | `[17,151645]` |
| decoded continuation | `2<\|im_end\|>` |
| cached/full argmax | 相同 |
| cached/full logits max error | `3.719329833984375e-05` |
| tolerance | `1e-4` |

该 recorded artifact 在普通 CI 中被 strict schema、内部一致性、manifest binding 和 self-fingerprint 重新校验，不会重下约 1 GB 权重。`123 passed` 的本轮专项结果包含这项 verifier，但不代表本轮再次执行了 1 GB 模型 forward。

Hash 使用一个 open handle，Transformers 随后按路径 reopen，因此仍存在 hash→loader reopen 的 TOCTOU 窗口。生产消费应增加不可变目录、ACL/lease、内容寻址句柄或等价控制。单个英文算术 prompt 不证明中文/总体质量、32k 有效上下文、许可适用性、峰值 RSS、CUDA/vLLM、吞吐、延迟或安全。

### 6.1 固定 Qwen 单矩阵 packed INT4

下面的 control 不把 generic toy 冒充目标模型，也不把局部量化冒充完整 checkpoint：

~~~powershell
python projects/transformers-basics/run_qwen_weight_quantization_control.py `
  --local-files-only
python projects/transformers-basics/run_qwen_weight_quantization_control.py `
  --verify projects/transformers-basics/target-checkpoints/qwen2.5-0.5b-instruct.weight-int4.recorded-report.json
~~~

它先重哈希同一 999,586,347-byte snapshot，加载 494,032,768 个 CPU FP32 参数，再选择第一层 bias-free `model.layers.0.self_attn.o_proj.weight`。矩阵为 `[896,896]`、802,816 参数，只占全模型 `0.0016250258120530175`；使用 contiguous-row group 128、每行 7 groups、FP32 absmax scale、4-bit 对称码 `[-7,7]`。

| 层级 | 固定观察 |
|---|---:|
| FP32 selected weight | 3,211,264 bytes |
| packed codes + scales | 401,408 + 25,088 bytes |
| strict bundle | 427,328 bytes，7.514752134192002× |
| weight relative-L2 | 0.1323337087062499 |
| selected output relative-L2 | 0.07000153078579582 |
| last logits relative-L2 / max-abs | 0.08513807180570929 / 1.6255179643630981 |
| last argmax | 17 → 17 |

真实 31-token forward 捕获 `[1,31,896]` activation；packed artifact 重载后的 selected-layer output 与内存 quantized output exact。Control 暂时把反量化 FP32 weight 写回该层执行第二次完整 forward，随后 byte-exact 恢复 source weight；artifact tamper 在 decode 前拒绝。Recorded report fingerprint 是 `sha256:df9ee045be4bf2e2ab4441bacfe24ffd1f903e9a0715bda0f35219ac3928f5cb`。

Argmax 相同只是一项单提示观察，不能抵消 logits 误差，更不等于质量无损。这个 artifact 只含一个 weight，不含其余 99.8375% 参数、config/tokenizer 或可执行低位 runtime；计算先反量化到 FP32。没有完整 low-bit Qwen checkpoint、fused kernel、NF4/GPTQ/AWQ/SmoothQuant、calibration、generation、GPU/CUDA/vLLM、resident/peak memory、速度或代表性质量证据。

## 7. Activation patching：从 hook 到因果对照

### 7.1 随机 MiniGPT 管线 control

`activation_patching.py` 固定 seed，在随机两层 MiniGPT 上缓存第 0 层 post-residual，把 clean activation 的指定位置写入 corrupted forward，并用：

\[
\text{recovery}=
\frac{m_{\text{patch}}-m_{\text{corrupt}}}
{m_{\text{clean}}-m_{\text{corrupt}}}
\]

报告不裁剪 normalized recovery。运行：

~~~powershell
python projects/transformers-basics/activation_patching.py
python -m pytest tests/test_activation_patching.py -q
~~~

本次实跑中，joint causal-prefix recovery=1，future-position 负对照 recovery=0；单独 source/readout position 的 recovery 分别约 0.4327/0.5760。测试还检查 hook 移除、clone/detach、shape/device/token 边界和过小 denominator 拒绝。

Token 27/19 是按这个随机 fixture 的 contrast 事后选择；模型没有训练，结果只能验证干预管线与 causal visibility，不能解释成自然语言 circuit。

### 7.2 固定 Qwen source-position control

~~~powershell
python projects/transformers-basics/run_qwen_activation_patching_control.py `
  --local-files-only
~~~

该协议复用同一 Qwen snapshot，固定 26-token France/Germany chat pair、source position 19、readout position 25、单-token `Paris−Berlin` metric，以及预先指定的 layer 0/11/23。它真实执行 10 次 CPU FP32 forward/hook。

Recorded clean/corrupt metric 为 `9.210311/-7.700302`；三层 source recovery 为 `1.000024/0.992244/0`。完整 layer-0 prefix 和 final-layer readout 是 recovery=1/1 的构造性正对照；future-position 负对照为 0，结束后 hooks 全部移除。

高恢复只说明这个 batch-1 pair 中，替换整个 896-d post-layer residual 对所定义 metric 的效果。它不定位 attention head、MLP 或 feature，不证明事实存储层、唯一自然 circuit、总体事实性或安全。Final-layer source recovery=0 符合该 hook 后不再跨 position 混合的结构，不能说“最后层没有作用”。

## 8. MoE：按六级证据递进

### 8.1 Top-k、capacity 与 combine

对每个 active token 的 router logits 做 stable softmax，选 top-\(k\) experts。仓库的 authored capacity 为：

\[
C=\left\lceil
\frac{\text{capacity factor}\cdot N_{\text{active}}\cdot k}
{E}
\right\rceil .
\]

需要分别记录：

- pre/post capacity expert counts；
- dropped assignments 与整 token 全丢两个分母；
- drop 后重新归一化，还是保留丢失 mixture mass；
- padding 是否排除；
- tie-break、routing group 与 overflow policy；
- balance diagnostic、z-loss 和 entropy 的确切版本。

~~~powershell
python projects/transformers-basics/moe_routing.py
python -m pytest tests/test_moe_routing.py -q
~~~

NumPy fixture 为 4 tokens、3 experts、top-2、capacity=2，counts `(3,4,1)→(2,2,1)`；8 个 assignments 丢 3 个，但没有整 token 全丢。Kept assignment 会真实经过 bias-free linear expert 和 weighted combine。这仍没有 trainable router/MLP、backward 或 collective。

### 8.2 可训练 router 与 experts

~~~powershell
python projects/transformers-basics/moe_training_control.py
python -m pytest tests/test_moe_training.py -q
~~~

PyTorch CPU Float64 fixture 真实执行 top-2 router 和三组 `Linear(3,4)→Tanh→Linear(4,2)` experts。Selected-only sparse path 与计算所有 token×expert 的 dense masked oracle 对齐：

- forward 最大差为 0；
- 所有参数 backward 最大差约 `6.94e-18`；
- 一步 SGD 同时改变 router 和三个 experts；
- authored MSE `0.0886473→0.0875580`。

Control 已把 score-priority capacity/drop 放入同一训练图。`capacity_factor=0.5` 时 capacity=2，counts `[4,3,3]→[2,2,2]`，10 个 assignments 丢 4 个；post-drop 重归一化与保留丢失 mass 是两种不同 contract。另一个 tie fixture 验证全丢 token 的 routed expert 输出为零。

Padding/group fixture 使用 `[T,T,T,T,F]` 与两个 2-token active groups，得到 per-group capacities `[1,1]`。分组与单组结果最大差约 `0.329387`，说明 group boundary 会改变竞争。Padding output/hidden gradient 为 0。这里的 CPU-local int64 label 不是 process group 或 distributed collective。

梯度因果控制显示：detach selected gate 后 experts 仍有 task gradient，而 router 主任务 gradient 消失；collapsed top-1 的 authored balance step 可在 hard assignment 未变时把诊断 `2.567724→2.552751`。

V3 从同一 top-1/capacity=2 初态比较 `drop`、deterministic full-ranking `reroute` 与 `dropless`。三者 post-policy counts 分别为 `[2,0,0]`、`[2,0,2]`、`[4,0,0]`；dropless 如实报告 nominal excess `[[2,0,0]]`。Reroute/dropless 的 sparse—dense forward 与 materialized-zero 全参数梯度差均为 0。这些策略是仓库自写 contract，不是 PyTorch、DeepSeek 或 Qwen 的默认语义。

### 8.3 六条 control 的边界矩阵

~~~mermaid
flowchart TD
    R["NumPy routing oracle"] --> T["单进程 trainable router/experts"]
    T --> C["两进程 global capacity collective"]
    T --> A["两进程 token-to-owner all-to-all"]
    A --> B["all-to-all backward + SGD"]
    C --> D["capacity-aware all-to-all backward + SGD"]
    A --> D
~~~

图中的箭头表示学习依赖，不表示代码或证据可以自动组合。四条 distributed scripts 是各自独立、范围不同的 authored fixtures。

| Control | 真正执行 | 明确没有 |
|---|---|---|
| `moe_routing.py` | NumPy top-k/capacity/drop/combine | trainable MLP、collective |
| `moe_training_control.py` | PyTorch forward/backward/SGD | 跨进程通信 |
| `moe_distributed_capacity_control.py` | Gloo all-gather/all-reduce global competition | expert ownership、backward |
| `moe_all_to_all_control.py` | owner-only expert + variable-split dispatch/return | capacity、backward |
| `moe_all_to_all_training_control.py` | reverse collective、router reduce、SGD | capacity |
| `moe_all_to_all_capacity_training_control.py` | global drop + kept-only dispatch/backward/SGD | reroute/dropless、DDP、CUDA |

### 8.4 Global capacity collective

~~~powershell
python projects/transformers-basics/moe_distributed_capacity_control.py
~~~

两个 spawn workers 用 CPU/Gloo 和 temporary FileStore。`all_gather` 建立 4-token replicated global batch，两个 `all_reduce` 得到 active=4、selected counts=`[4,0]`。Rank-local 独立 capacity 合计保留 2 个 assignment；global competition 只保留全局最高分 token，mask `[F,F,T,F]`、drop=3，rank-0 output counterfactual 差为 `0.9640275800758169`。

这是真 collective，但 router/experts 在两 rank 复制，仍无 expert `all_to_all`、distributed backward、CUDA/NCCL、多节点或性能证据。

### 8.5 Token-to-owner all-to-all

~~~powershell
python projects/transformers-basics/moe_all_to_all_control.py
~~~

Rank 0/1 各自只持有 expert 0/1 的 owner-only expert parameters。Source→owner counts 为 `[[1,2],[1,0]]`。每 rank 五次 `all_to_all_single` 分别交换 count、dispatch float/metadata、return output/gate/metadata。

Rank 0 return arrival 的 global token 顺序 `[1,0,2]` 不是 source-local 顺序；必须按 `(source_rank, source_local_index, global_token_id, expert_id)` scatter。正确路径与单进程 oracle 最大差为 0，忽略 metadata 的反事实差为 `0.8958737432590591`。

报告的 256+160=`416 logical tensor-payload bytes` 只是源码张量 `numel×element_size` 账本，不是 wire bytes，不含 Gloo/TCP/FileStore 协议、分包或对齐。

### 8.6 All-to-all backward 与 SGD

~~~powershell
python projects/transformers-basics/moe_all_to_all_training_control.py
~~~

Authored `autograd.Function` 在 backward 交换 input/output splits，执行 reverse-split backward。Owner expert 已收齐所有 source 发来的 tokens，因此 owner expert gradient 不再按 data-parallel 语义重复 reduce；replicated router 只看到 local source gate path，必须做 gradient SUM all-reduce。

Distributed forward、gradients、一步参数和 post-step forward 都与单进程 global-batch oracle 相同，global MSE `20.78017329703821→19.41091750734501`。这不是 `torch.distributed.nn.functional`、RPC distributed autograd 或 DDP，不等于 DDP 或生产 EP。

### 8.7 Capacity-aware all-to-all training

~~~powershell
python projects/transformers-basics/moe_all_to_all_capacity_training_control.py
~~~

四个 active tokens 的初选 counts 为 `[2,2]`；`capacity_factor=0.5` 得到 per-expert capacity=1、global keep mask `[F,T,T,F]` 和 kept `[1,1]`。Kept-only source→owner splits 为 `[[1,1],[0,0]]`。

Rank 1 是 zero-assignment source rank，但 zero-size graph edge 仍参加 reverse collective。Dropped token 0/3 的 routed output 与 task hidden gradient 为 0。Forward、hidden/router/expert gradients、一步参数和 post-step forward 都与单进程 capacity oracle 对齐；global MSE `15.253670387373656→14.530264380025987`。

该 control 仍不执行 reroute/dropless、shared/fine-grained experts、DDP/FSDP/ZeRO、mixed precision、optimizer resume、CUDA/NCCL、多节点或目标 MoE checkpoint。一步 toy loss 下降不是收敛、专门化或质量提升。

## 9. 推荐运行顺序 { #run }

### 9.1 五分钟机制路径

~~~powershell
python projects/transformers-basics/train_byte_bpe.py
python projects/transformers-basics/online_softmax_demo.py
python projects/transformers-basics/smoke_tiny.py
python projects/transformers-basics/generation_runtime_control.py
python projects/transformers-basics/activation_patching.py
~~~

每个命令都应保存环境、seed、输入 identity 和完整 JSON 输出。不要只截一行 `passed`。

### 9.2 离线 artifact 路径

~~~powershell
python projects/transformers-basics/verify_release_evidence.py
python -m pytest tests/test_model_release_evidence.py -q
~~~

这组回归只检查离线 manifest、快照投影和 verifier 负例，不重新运行约 1 GB 权重。需要新 runtime observation 时，才显式运行三个 target-Qwen scripts，并保存新环境、版本、报告和失败日志。

### 9.3 完整 MoE 路径

依次运行 8.4–8.7 的四个实验脚本；每次只比较该脚本输出与本节给出的预期，不把不同实验的证据合并。

## 10. 故障定位

### Loss 不下降

先固定 seed 与 batch，确认 `model.train()`、labels shift、ignored labels、`zero_grad()`、`backward()`、optimizer 参数集合和 learning rate。Tiny overfit 是 wiring probe；不要先换大模型掩盖基础错误。

### Cached/full logits 不一致

逐项检查 position IDs、attention mask 的 past 长度、cache position、dtype、eval mode 和 dropout。Argmax 相同不代表完整 logits 对齐，应同时保存 max error 与 tolerance。

### Generation 提前或不停止

保存 tokenizer/model/generation config 的 EOS 集合以及实际 call kwargs；检查 stop string 的 tokenization、prompt 是否已含 EOS、`max_new_tokens` 与 beam/sampling 路径。Provider 的 finish reason 不能从客户端截断自动推得。

### MoE rank hang

先核对所有 rank 的 collective 顺序、split sizes 和 zero-length edge。一个 rank 没 assignment 也不能擅自跳过其他 rank 正在等待的 collective。再检查 metadata 是否随 payload 往返，以及 router/expert gradient 的 reduce 语义是否混淆。

### Recorded report 通过但新运行不同

先判断是 artifact identity、runtime version、数值 tolerance 还是行为 contract 变化。不要直接重写 expected JSON；保留旧报告，记录依赖和硬件差异，解释为何新 observation 仍满足或需要升级 schema。

## 11. 求职与项目验收

### 能讲清的核心问题

1. 为什么 byte-level BPE 可逆，但不等于真实 tokenizer compatibility？
2. Online softmax 的 running max 为什么变化时必须重标度旧 accumulator？
3. GQA 减少的是哪部分 KV payload，为什么 MLA 不能套同一公式？
4. Config hash、weight hash、model load 和一次 forward 分别证明什么？
5. `generate()` 的停止语义由哪些配置层共同决定？
6. Activation patching 的 recovery>1 或等于 0 应如何解释？
7. MoE assignment drop 与 token drop 为什么必须用两个分母？
8. Router gradient 与 owner expert gradient 为什么有不同 collective 语义？
9. 为什么 `416 logical tensor-payload bytes` 不能写成网络流量？
10. 为什么六条 MoE controls 不能宣称复现了 DeepSeek/Qwen MoE？

### 作品集最小证据包

- 一张证据梯子图，逐项标注 authored、recorded 与 live；
- byte BPE merge/round-trip artifact；
- attention dense/online/cache parity 报告与 tolerance；
- tiny overfit 的 config、seed、loss trace 和参数账本；
- generation protocol diff 与三条 forced-token trace；
- immutable revision、selected file hash、runtime versions；
- target Qwen prefill/cache/generate report及其 verifier；
- activation patching clean/corrupt/patch metric、正负对照与 site 选择规则；
- MoE pre/post counts、drop 分母、split matrix、collective ledger、oracle parity；
- 至少一个故意失败 case，以及明确的证据边界。

可以准确写：

> 构建从 NumPy mechanism oracle、tiny Transformers control 到 immutable Qwen checkpoint 的分层验证链；对账 BPE round trip、dense/online/cache attention、generation EOS override、固定权重 prefill/cache/generate，并以两进程 Gloo fixtures 隔离 MoE global capacity、token-to-owner dispatch 与反向传播语义。

同一句必须注明 CPU/Gloo/authored/单 prompt 或 recorded 范围。除非另有目标 GPU 与 workload artifact，不要写“复现 FlashAttention”“完成 DeepSeek expert parallel”“高并发部署”或“性能提升 X%”。

### 完成定义

- [ ] 能从公式手算一个 BPE merge、causal mask 和 online-softmax 更新；
- [ ] 能区分参数存储、KV 理想 payload、进程峰值和 wire bytes；
- [ ] 能解释 release evidence 为什么没有执行权重；
- [ ] 能从 recorded report 追到 manifest、revision、file hashes 和 scope；
- [ ] 能区分单矩阵 artifact ratio、完整 checkpoint bytes、resident memory 与 fused-kernel execution；
- [ ] 能为 generation/patching/MoE 各设计一个负对照；
- [ ] 能说明每条 distributed control 调用了什么 collective；
- [ ] 能在简历中只写 artifact 支持的数字和分母。

## 12. 总证据边界

本页验证的是确定性 mechanism fixtures、随机 tiny 框架路径、不可变发布 artifact 投影，以及固定 Qwen checkpoint 的少量 CPU FP32 observations；其中 packed INT4 只覆盖一个真实 weight，并以反量化 FP32 执行。它没有建立完整 Qwen low-bit checkpoint/量化 runtime、发布者签名、许可结论、训练数据来源、模型总体质量、长上下文有效性、GPU kernel、FlashAttention、vLLM/provider 等价、目标 MoE checkpoint、CUDA/NCCL、多节点、吞吐、延迟、峰值显存或生产安全。

尤其要保持以下边界：

- CPU/JAX/Gloo/loopback 结果不得外推 GPU、NCCL、目标模型或生产性能；
- hash identity 不等于来源认证，verify 后 reopen 仍有 TOCTOU；
- 单 pair activation recovery 不等于唯一自然 circuit；
- router/experts/collective 的独立 fixtures 不等于完整 expert-parallel stack；
- 一步 loss 下降不等于收敛、泛化、专门化或质量提升；
- 通过 recorded verifier 不等于本次重新执行了模型/provider。

完整实现、脚本参数和机器报告字段见 [projects/transformers-basics](https://github.com/NightLemon/about-llm/tree/main/projects/transformers-basics)。
