# 架构谱系与运行时依赖

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要比较模型架构、核对 checkpoint 兼容性或规划运行时支持的工程师。
- **先修**：[Transformer](transformer.md)、attention、KV Cache 与基本张量形状。
- **首次阅读**：信息流分类 → Transformer 之外的序列模型 → MoE → 运行时依赖 → 公平比较。
- **完成信号**：能从具体架构特征推导所需状态、算子、运行时能力和公平比较口径。
- **卡住时**：先只画 causal mask、历史状态和每 token 激活路径，再回到具体实现。

</div>

模型家族名不能替代计算图。面对一个新 checkpoint，先追踪位置之间怎样交换信息、生成时保存什么状态，
再判断 tokenizer、processor、算子、缓存和服务运行时是否真正支持它。本页负责这张架构与依赖地图；
若要判断某个内部表示是否真的影响行为，请继续阅读[机制可解释性](mechanistic-interpretability.md)。

## 先按信息流分类，不按品牌名分类

遇到新 checkpoint，先回答五个问题：

1. 每个输入位置可以读取哪些位置？
2. 输入和输出由一个 stack 还是两个 stack 处理？
3. 历史保存为全部 token 的 K/V，还是受限 recurrent state？
4. 每个 token 激活全部参数，还是只路由到部分 experts？
5. 训练 objective、attention mask 与推理 loop 是什么？

“像 GPT”或“属于某模型家族”只能帮助我们猜测。真正运行模型时，仍要查看具体版本的配置、
tokenizer 或 processor、张量形状和模型实现。

先把几类主干架构放在一张图里。表中的“状态”是生成下一个 token 时需要从历史中保留的内容，
它直接影响显存、调度和推理框架的实现。

| 架构 | 一个位置能读取什么 | 生成时保留什么 | 常见用法 |
|---|---|---|---|
| Encoder-only | 通常能双向读取整段输入 | 标准接口不做逐 token 开放生成 | 分类、检索、抽取、reranking |
| Decoder-only | 只能读取当前位置左侧的历史 | 每层历史 token 的 K/V | 对话、代码生成、工具调用 |
| Encoder-decoder | Encoder 双向读来源；decoder 读来源和已生成前缀 | Encoder 表示与 decoder K/V | 翻译、摘要、条件生成 |
| RNN / SSM | 新输入通过递推状态读取历史 | 固定形状或受限大小的状态 | 流式处理、长序列建模 |
| Convolution | 读取卷积核覆盖的窗口 | 取决于卷积缓存和感受野 | 并行序列混合、混合架构 |

这张表只描述计算图，不直接决定模型能力。预训练数据、目标函数、参数规模、后训练和工具系统，
都可能让同类架构表现得很不一样。

## Encoder-only：每个输入位置双向理解整段

标准 encoder self-attention 允许每个非 padding token 读取整段输入：

\[
H'=\operatorname{Attention}(Q(H),K(H),V(H)).
\]

这种双向可见性很适合“读懂一段已经给定的文字”。模型可以为每个 token 输出结合上下文后的表示，
也可以把整段聚合成一个向量。分类、重排（reranking）、序列标注、向量检索和抽取式问答都常用这类表示。

训练时，BERT 一类模型常遮住部分 token，再让模型恢复它们。这叫掩码语言建模（masked language modeling）。
它训练的是“利用左右文补全被遮住位置”，不是“根据左侧历史不断续写”。

因此，标准 encoder-only 接口不会直接执行从左到右的自回归生成。可以给它外接 decoder，
也可以反复遮盖和预测；不过此时已经加入了新的生成机制，不能把结果归因于原始 encoder 接口。

## Decoder-only：输入和输出共享一条 causal stream

位置 \(t\) 只能读取左侧历史：

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

训练时，完整答案已经在样本中。因果掩码（causal mask）虽然阻止位置看到未来，却仍允许 GPU 一次计算所有位置。

推理时，未来 token 尚不存在。模型只能生成一个、追加一个，再继续生成。为了避免每一步重算全部历史，
推理框架通常保存各层历史 token 的 key 和 value，也就是 KV cache。

Prompt、工具参数格式、RAG 检索结果和模型回答都可以排进同一条 token 序列，许多任务因此能统一成
“继续写下去”。统一接口很方便，但不表示模型天然理解这些区域的权限边界。

代价也来自这条统一序列：

- 输入 token 在同层无法读取右侧位置；
- 长 Prompt 的 prefill 和 KV cache 消耗计算与显存；
- System、user、tool 与 assistant 的边界依赖 chat template 和特殊 token；
- 训练时哪些 token 计入 loss，必须与推理时提供给模型的前缀对齐。

“Decoder-only 更通用”是生态、数据与训练结果的经验总结，不是一条只由 causal mask 推导出的架构定理。

## Encoder-decoder：source 与 target 天然分开

这类模型把“读输入”和“写输出”交给两个 stack。Encoder 双向处理来源 \(x\)；decoder 只能读取
已经生成的 \(y_{<t}\)，同时通过交叉注意力（cross-attention）读取 encoder 的表示：

\[
p(y\mid x)=\prod_t p(y_t\mid y_{<t},H_{enc}(x)).
\]

来源和目标天然分开，因此翻译、摘要、改写和结构转换很容易表达。生成期间可以复用 encoder 输出，
但每个 decoder 层仍要读取与来源对应的 key/value。

“输入只编码一次”不足以证明端到端更便宜。实际成本还受 encoder/decoder 的 layers、width、
输入输出长度、cache 和 kernel 支持影响。

## Attention mask 也可以创造中间形态

Prefix LM 让 prefix 内双向可见，生成区域只能读取 prefix 和自己的左侧。
UniLM 类设计则通过不同 mask，让共享 parameters 表达多种任务。

常见 mask 各自改变一条信息流：

| Mask | 它限制什么 |
|---|---|
| Padding | 排除无效位置 |
| Causal | 排除未来位置 |
| Prefix / block | 定义不同区域之间的可见性 |
| Local / sliding window | 限制读取距离 |
| Global tokens | 允许少数位置跨区域连接 |

张量形状能够广播，只说明代码可以运行。要检查 mask 的语义，可以用短序列打印每个 query 实际能看见的
key 位置，再比较训练与带 cache 的 decode 是否遵循同一规则。

## RNN、卷积和 SSM 怎样处理历史

RNN 将历史压入状态：

\[
h_t=f(h_{t-1},x_t),
\qquad
y_t=g(h_t).
\]

RNN 像一边阅读、一边改写固定大小的笔记 \(h_t\)。推理时只需带着这份状态继续，因此很适合流式输入。

代价是过去细节必须被压进有限状态。由于 \(h_t\) 依赖 \(h_{t-1}\)，训练也不容易沿时间维完全并行。
LSTM 和 GRU 用门控缓解长程梯度消失或爆炸，但仍然要把历史压进状态。

一维卷积同时处理许多局部窗口。堆叠更多层或使用空洞卷积（dilation），可以扩大一个输出所能看到的
输入范围，也就是感受野。长卷积还可以借助 FFT 加速。标准卷积对不同输入使用同一组混合权重，
不像 attention 那样根据当前内容重新计算“该读哪里”；动态卷积等变体则会改变这个前提。

线性 state-space model（SSM）的一种离散形式是：

\[
h_t=\bar Ah_{t-1}+\bar Bx_t,
\qquad
y_t=Ch_t+Dx_t.
\]

在线性、时不变的条件下，同一个 SSM 既可以写成递推，也可以展开成卷积：训练时利用并行算法，
推理时只更新状态。Selective SSM 进一步让部分更新依赖当前输入，从而获得内容选择能力；
此时可用的并行算法和需要保存的状态也会随具体实现变化。

线性复杂度并不保证实际运行时间更短。融合算子是否成熟、状态大小、batch、序列长度和硬件，
共同决定它从哪个长度开始占优。固定大小的状态也可能难以保留精确复制和远距离检索需要的细节。

混合模型可以交替 attention、SSM、convolution 或 recurrent layers，
让 attention 负责精确寻址，让其他模块承担更便宜的长程 mixing。

## 同为 Transformer，内部仍有许多架构轴

| 设计轴 | 常见做法 | 主要影响 |
|---|---|---|
| 归一化的位置 | Pre、Post、sandwich | 优化稳定性和残差路径 |
| 归一化的方法 | LayerNorm、RMSNorm | 统计量、参数和数值行为 |
| 前馈层 | GELU、SwiGLU、其他门控变体 | 参数量、中间激活和算子实现 |
| Attention heads | MHA、MQA、GQA、压缩 KV | 模型质量、KV cache 大小和投影结构 |
| 位置信息 | Absolute、RoPE、relative bias、ALiBi-like | 长度外推和实现方式 |
| 可见范围 | Full、sliding window、block sparse、hybrid | 感受野、cache 和可用算子 |
| 前馈层路由 | Dense、MoE | 每 token 计算量、权重显存和通信 |
| 残差拓扑 | Serial、parallel、scaled | 计算图和权重转换 |

加载第三方权重时，要逐项核对配置、attention head 数、RoPE 缩放、输入输出 embedding 是否共享、
张量形状和模型代码。用“看起来合理”的默认值补缺失字段，可能仍能生成文字，却已经执行了另一个函数。

## MoE 同时改变权重、计算与通信

稀疏专家混合模型（Sparse Mixture-of-Experts，MoE）把普通前馈层换成多组“专家”网络。
对每个 token，路由器（router）选出得分最高的 top-k experts，再合并它们的输出：

\[
y(x)=\sum_{e\in\operatorname{TopK}(r(x))}g_e(x)E_e(x).
\]

一次路由可以按下面的顺序理解：

1. Router 为当前 token 给所有 experts 打分；
2. Top-k 选择决定它准备发给谁，同分时还需要明确的排序规则；
3. Capacity 限制每个 expert 本批最多接收多少份 assignment；
4. 超出容量的 assignment 可以被丢弃、改派，或在 dropless 模式下继续执行；
5. 分布式运行时把 token 发到 expert 所在设备，expert 完成前馈计算后再送回；
6. Router 权重把返回结果加权合并成这个 token 的输出。

讨论“模型有多大”时，至少要分三本账：

| 账本 | 回答的问题 | 主要系统影响 |
|---|---|---|
| 总参数量 | 一共要保存多少 expert 权重 | 权重显存、加载时间、跨卡放置 |
| 激活参数量 | 一个 token 实际经过多少参数 | 理想计算量和算子形状 |
| 路由与通信 | token 要发到哪些设备，负载是否均衡 | All-to-all 流量、等待时间和吞吐 |

所以“激活参数少”不等于单卡能装下全部权重，也不保证运行更快。实现还必须明确共享 experts、
router 精度、负载均衡损失，以及丢弃 assignment 后是否重新归一化权重。

Top-k 的 expert 编号是离散选择，编号本身没有普通导数。任务梯度可以通过已选 expert 的连续 gate
概率回到 router；负载均衡目标或其他估计方法则用来改善路由分布。两者不能混成“top-k 自动可导”。

可解释性分析还要分开两个问题：expert 的输出对哪些行为有作用，以及 router 为什么把这些 token 发给它。
观察到某个 expert 经常接收代码 token，可以形成假设；只有进一步干预和负对照，才能判断它是否承担
稳定、必要的“代码功能”。

仓库的 NumPy/PyTorch 小实验正是沿着这六步展开。第一步手算容量和加权合并，第二步比较稀疏与稠密实现的
前向和反向结果，第三步用两个 CPU/Gloo 进程观察全局容量竞争与 all-to-all 往返。

这些实验帮助读者看清机制，但没有运行 DeepSeek 或 Qwen 的目标 MoE 权重，也没有测量 GPU 性能。
通信过程见[分布式训练](../systems/distributed-training.md)。

## 从架构推导运行时依赖 { #architecture-runtime-dependencies }

下载到 `safetensors`，只代表拿到了权重张量。要让一个 checkpoint 正确运行，下面每一层都要匹配：

```text
config + tensor identity
-> tokenizer / processor / chat template
-> model graph + operator semantics
-> decode state / cache layout
-> kernels + quantization
-> batching / scheduling / cancellation / output parser
```

不同架构特征会把要求传给不同组件：

| 架构特征 | 生成时保存的状态 | 推理框架还要实现什么 |
|---|---|---|
| Causal MHA / GQA | 每层的 K/V | 模型类、RoPE、GQA 算子和 KV 分配器 |
| Sliding / sparse attention | 窗口化或分层 K/V | 理解 mask 的算子、淘汰规则和 prefix cache 语义 |
| Compressed / latent KV | 非标准 latent cache | 专用计算图、cache 对象、算子和容量公式 |
| SSM / recurrent | 递推与卷积状态 | Prefill 扫描、状态更新和 batch 内序列重排 |
| Attention-recurrent hybrid | K/V 与递推状态并存 | 不同层各自的 cache 和生命周期 |
| Sparse MoE | 路由、容量和 expert 放置 | 融合 MoE、expert 量化和多卡 all-to-all |
| Vision/audio encoder + LM | 媒体张量、encoder 输出和位置 | Processor、融合图和多模态 batching |
| Early-fusion multimodal | 联合序列和媒体 metadata | 匹配的 mask、位置编码和服务请求格式 |
| Extra prediction heads | 候选 token 和验证状态 | 权重加载、speculative scheduler 与回退路径 |

因此，“能加载”只是兼容性最低的一层。下面这些现象都不足以证明完整支持：

- `AutoModel` 能实例化，就宣称 serving runtime 已正确支持 cache 和 batching；
- Sliding attention 回退成 full attention 后能输出，仍沿用原本的性能与显存估算；
- 文本请求成功，就宣称多模态 processor 和媒体路径可用；
- Checkpoint 中存在额外预测 head，就宣称 speculative decoding 已经加速；
- MoE active parameters 小，就推断整模型权重一定能放入设备。

### 不同库分别负责哪一层

| 组件 | 它通常负责什么 | 它单独证明不了什么 |
|---|---|---|
| Transformers / JAX 模型实现 | 配置、计算图、权重加载和基础 generation | 高吞吐调度、分页状态和请求取消 |
| Tokenizer / processor | 对话模板、特殊 token 和媒体预处理 | 服务端是否使用同一版本，模型是否真的读取媒体 |
| vLLM / SGLang | 模型注册、状态管理、batching 和流式输出 | 新架构算子、输出 parser 和所有量化组合都正确 |
| Nano 教学 runtime | 精简的 engine、scheduler、cache 和 runner | 未实现架构、混合状态和生产环境完整语义 |
| FlashAttention / Triton / CUTLASS / fused MoE | 特定形状、dtype 和 mask 下的算子 | 完整模型可运行，或端到端一定加速 |
| PEFT / TRL / trainer | Adapter、数据整理、loss 和优化循环 | Target modules、loss mask、部署重载和最终质量都正确 |
| llama.cpp / GGUF | 已支持计算图的转换、量化和执行 | 任意 Hugging Face 权重、所有模态或无损转换 |

### 用能力阶梯记录“支持到哪”

1. **识别**：配置中的 `model_type`、架构名和权重名称能够对应；
2. **实例化**：模型能以目标 dtype 加载到目标设备，缺失或多余权重符合预期；
3. **模型执行**：Prefill 和带 cache 或递推状态的 decode 能与参考实现对账；
4. **额外路径**：改变媒体输入会影响结果，MoE routing 和额外 heads 确实执行；
5. **服务执行**：Batch 重排、prefix cache、取消、OOM 回退和输出解析都正确；
6. **目标设备验收**：在固定 workload 下测量质量、显存、TTFT、TPOT、吞吐和失败情况。

前一层通过，不代表后一层自动通过。拿到新模型时，可以先填一张兼容性卡片：

```yaml
checkpoint: repo@immutable-revision
frontend: tokenizer-or-processor@revision
model_class: exact-class-and-library-version
state: kv | recurrent | hybrid | encoder-plus-kv
backend: attention-scan-moe-kernel-and-fallback
serving: runtime-version-and-enabled-features
hardware: gpu-driver-cuda-dtype
verified: [instantiate, prefill, decode]
not_verified: [multimodal, extra-head, cancellation, throughput]
```

这样讨论的是架构与依赖，而不是追逐模型品牌列表。

## 架构比较先选择公平口径

三种常见比较各自回答不同问题：

1. **相同参数量**：比较模型容量，但训练 FLOPs、KV cache 和中间激活仍可能不同；
2. **相同训练 FLOPs/token**：比较计算效率，但权重显存可能不同；
3. **相同硬件和运行时间**：比较真实系统表现，算子成熟度也会进入结果。

无论选择哪一种，还要固定或报告训练数据、tokenizer、上下文长度、optimizer 和训练 token 数。
调参预算与服务 batch 也会影响比较。比起寻找“架构总冠军”，更有用的是分别画出质量与训练成本、
延迟、显存之间的 Pareto 前沿。

## 把架构结论交给下一层

架构只能限定“哪些信息流可能存在”，不能证明模型在某个行为中实际使用了哪条路径。
要从行为对照走向 probe、归因、activation patching、SAE 与模型编辑，请进入
[机制可解释性](mechanistic-interpretability.md)。要验证缓存、调度与服务容量，则进入
[一次推理请求](../systems/inference-request-lifecycle.md)和[推理优化](../systems/inference-optimization.md)。

## 自测

1. 为什么同为 decoder-only，两个 checkpoint 仍可能需要不同的缓存与算子实现？
2. MoE 的总参数量、激活参数量与通信账本分别回答什么问题？
3. “模型可以加载”位于兼容性阶梯的哪一层，还缺哪些验证？
4. 比较 dense 与 MoE 模型时，怎样分别固定参数、训练计算与目标硬件口径？
