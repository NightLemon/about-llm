# 架构谱系与机制可解释性

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要比较模型架构，或设计机制可解释性实验的工程师与研究者。
- **先修**：[Transformer](transformer.md)、残差流、attention 和基本因果实验直觉。
- **首次阅读**：先看信息怎样流动，再跟一组 France/Germany 反事实走到 activation patching。
- **完成信号**：能区分架构事实、行为相关、可解码信息和干预证据。
- **卡住时**：先只画 causal mask 与 residual stream，不必理解所有归因方法。

</div>

给模型两条几乎相同的 Prompt：

```text
clean:   The capital of France is
corrupt: The capital of Germany is
```

模型分别更偏向 `Paris` 和 `Berlin`。这里有两类容易混在一起的问题：

1. **架构问题**：`France` 的信息通过哪些 token、layers、attention 或 recurrent state 到达输出位置？
2. **机制问题**：在这个具体行为中，哪些表示或路径真的影响了 `Paris - Berlin` 的 logit difference？

架构告诉我们哪些信息流在计算图上可能发生；可解释性实验尝试缩小“实际用了哪条路径”的答案范围。
知道模型是 decoder-only，不会自动告诉我们国家信息存在哪里；观察到一个高相关 neuron，也不会自动证明输出依赖它。

本章先建立架构地图，再用同一 clean/corrupt pair 解释证据强度。

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

## 从行为到机制：先看证据阶梯

| 证据层 | 怎样做 | 可以回答 | 仍不能直接回答 |
|---|---|---|---|
| 行为 | 最小对照或反事实 Prompt | 改输入后，输出是否随之改变 | 内部哪一部分造成变化 |
| 可解码性 | 训练线性 probe 读取 hidden state | 某项信息能否从表示中读出 | 模型生成答案时是否使用它 |
| 归因 | Gradient、Integrated Gradients、遮挡、attention 分析 | 输出对哪些输入或组件敏感 | 该组件是否必要，归因是否唯一 |
| 组件干预 | Ablation、activation patching | 改动指定组件后，metric 是否改变 | 它是否是唯一机制或完整 circuit |
| 路径干预 | Path patching、causal scrubbing | 限制传播路径后，预先预测的行为是否成立 | 结论能否推广到未测行为和输入 |

这不是一张“做得越往下就一定越正确”的排行榜。每一层都需要自己的对照和假设，只是回答的问题不同。
Probe 准确率很高，只说明 probe 能取到信息；activation patching 恢复一个指标，也不等于发现了唯一 circuit。

## 行为实验先定义连续 metric

先用本章开头的 France/Germany 对照定义一个从头到尾不变的读数。两个 Prompt 只改变国家 token，
模板、长度和 tokenization 尽量保持一致。然后固定比较 `Paris` 与 `Berlin` 的 logit：

\[
m(x)=z_{\mathrm{Paris}}(x)-z_{\mathrm{Berlin}}(x).
\]

这个方向在所有运行中都不变：France 输入应得到较大的 \(m\)，Germany 输入应得到较小的 \(m\)。
如果每个输入都把“正确答案”放在减号前，两个指标的方向会同时为正，反而无法计算后面的恢复程度。

连续的 logit difference 比只看最终输出更敏感，可以在 top-1 翻转前观察变化。不过，它只衡量这两个
单 token 候选。答案包含多个 token 时，应预先定义整个序列的得分，不能看完结果再挑最有利的首 token。

负对照至少包括：

- 替换无关位置或未来位置；
- 随机打乱作为来源的 activation；
- 使用分布相近、但与任务无关的来源；
- 检查变化是否只来自 activation norm；
- 在多个模板、位置、语言和随机种子上复现。

## Probe 检查“能否读出”，不是“是否使用”

线性 probe 是一个额外训练的小模型。它从 frozen hidden state \(h\) 预测我们关心的属性：

\[
\hat y=Wh+b.
\]

如果同一实体或模板同时出现在训练集和测试集，probe 可能只记住词面。数据应按实体和模板分组切分，
同时控制类别、长度和 token 位置。还要拿简单的输入特征作基线，而不只是和随机猜测比较。

Probe 太强时，可能自己学会任务，而不是揭示原模型已有的简单表示。反复尝试 layer、feature 和
超参数后继续汇报同一个测试集，也会把选择偏差伪装成泛化能力。

更进一步，可以先预测“沿 probe 找到的方向做干预后，目标行为会怎样变化”，再检查预测是否成立，
以及无关能力是否保持。

## Attribution 描述 sensitivity，仍要小心输入分布

Gradient 描述当前点附近的局部敏感度：输入轻微变化时，输出分数朝哪个方向变化。遇到 saturation，
gradient 可能接近零，即使离开当前区域后该 feature 仍然重要。

Integrated Gradients 从 baseline \(x'\) 沿路径积分：

\[
IG_i(x)=(x_i-x_i')
\int_0^1
\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}
d\alpha.
\]

结果会随 baseline、积分路径和目标函数改变。特别是对 token embedding 做线性插值时，
路径中间点通常不对应任何自然语言输入。

遮挡法（occlusion）会删除、mask 或置零某项输入。这很直观，却可能制造训练分布之外的样本；
多个相关 feature 共同起作用时，贡献也不会自动公平分摊。

注意力权重（attention weight）只表示某层某个 head 怎样加权 value。它还没有包含 value 的内容、
输出投影、残差、MLP 和后续层。

因此，attention 图可以提示模型“正在读哪里”，却不是最终 token 重要性的完整答案。

## Activation patching 直接干预中间表示

Probe 问“这里能不能读出国家信息”，activation patching 问得更直接：如果把 France 运行中的某个
中间表示搬到 Germany 运行里，模型会不会重新偏向 `Paris`？基本实验分四步：

1. 运行 clean input `France`，缓存指定位置的 activation，并记录 \(m_{clean}\)；
2. 运行 corrupt input `Germany`，记录 \(m_{corrupt}\)；
3. 再运行 `Germany`，但在预先选定的位置换入 `France` 的 activation；
4. 记录干预后的 \(m_{patched}\)，其余计算保持不变。

常用恢复分数：

\[
R=\frac{m_{patched}-m_{corrupt}}{m_{clean}-m_{corrupt}}.
\]

当 \(R\) 接近 1，说明这次干预把选定指标拉回了 clean 水平；接近 0，说明它几乎没有缩小差距。
分母接近零时，比例会非常不稳定。此时应停止计算恢复率，只报告三个原始 metric。
\(R>1\) 和 \(R<0\) 都可能出现，不能为了画图方便而裁剪到 `[0,1]`。

这项干预可能制造模型训练时从未见过的 activation 组合。即使恢复率很高，严谨结论仍然只是：
“在这组输入、这个位置和这项替换下，指定 metric 发生了变化。”冗余组件、备用路径和纠缠在一起的
features，都可能让一次 ablation 或 patching 给出不完整答案。

### 先用随机 MiniGPT 检查 hook 和因果负例

第一次不要急着解释自然语言知识。仓库先在固定随机种子的两层 MiniGPT 上执行真实 PyTorch hook：

~~~powershell
python projects/transformers-basics/activation_patching.py
python -m pytest tests/test_activation_patching.py -q
~~~

输入 `[1,2,3,4]` 与 `[5,2,3,4]` 只改首 token。程序分别替换第 0 层残差输出中的来源位置、
读取位置和完整 causal prefix，确认 metric 会按构造改变；再替换未来位置，确认它不能影响过去的读取位置。
这个负对照同时检查 causal mask 和 hook 位置是否接对。

这一步只检查张量、hook 和因果方向。模型没有训练过；比较哪两个输出 token，也是在看过两次运行的差异后
才为这个样例选择的。完整 prefix 恢复 clean 输出更是特意构造的正例，所以这里不能声称发现了自然语言 circuit。

### 在固定 Qwen checkpoint 上做一次单事实干预

确认实验方法能工作后，再把同一思路用于真实权重。这里使用
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`。
套用 chat template 后，两条输入都是 26 tokens，只有位置 19 从单 token ` France` 变成 ` Germany`；
程序在位置 25 始终计算 `logit(Paris) - logit(Berlin)`。

~~~powershell
python projects/transformers-basics/run_qwen_activation_patching_control.py `
  --local-files-only
~~~

实验在执行前按“第一层、中间层、最后一层”的规则选定 layers 0、11、23，没有从全部层里挑最好结果。
Clean metric 是 `9.210311`，corrupt metric 是 `-7.700302`，两者相差 `16.910613`。录制结果如下：

| 替换的位置 | Recovery | 这一步实际说明什么 |
|---|---:|---|
| Layer 0，来源位置 19 | 约 1.000 | 替换整条 896 维 residual 后，这个指标回到 clean 附近 |
| Layer 11，来源位置 19 | 约 0.992 | 对这组输入和干预接近恢复，尚未定位具体 head、MLP 或 feature |
| Layer 23，来源位置 19 | 0 | 该 hook 之后没有让来源位置再次影响读取位置的跨 token mixing |
| Layer 0，全部 causal prefix | 1 | 完整搬入 clean prefix 的构造性正对照 |
| Layer 23，读取位置 25 | 1 | 直接换入最终读取 residual 的构造性正对照 |
| Layer 0，未来位置 26 | 0 | 未来位置不能改变过去读取位置的负对照 |

这些数字来自真实 Qwen 权重，但它们不表示“第 11 层存储了法国首都”。样本只有一个英文事实、一个模板，
batch size 为 1；一次替换 896 维 residual，可能同时搬运实体、词形、位置信息和其他 features。

要形成机制研究，还要加入随机或无关来源、改写、跨语言对照、多个事实、更细的组件或路径干预，
并在未参与选择方案的数据上复现。报告自带的无密钥 hash 能发现内容是否变化，却不能认证是谁执行了实验。

## Path patching 把问题缩到组件之间的边

普通 activation patching 替换一个节点的全部输出，无法区分它通过哪条下游路径起作用。Path patching
尝试只改变“来源组件 → 指定下游组件”这条边，同时控制来源组件通往其他位置的影响。

在动手之前，应把 circuit 假设写成可以失败的预测：

```text
节点：哪些 heads、MLPs、features 或 residual sites
边：哪个投影、从哪个 token 位置传到哪里
行为：研究哪组输入，用什么 metric
必要性：移除这条 circuit，目标行为应下降多少
充分性：只保留这条 circuit，目标行为应保留多少
忠实性：干预不应制造哪些无关错误
```

即使这些预测在几十个模板上成立，结论范围仍然是这些行为、组件和干预，不能直接升级成
“整个模型都使用这套算法”。

## Sparse Autoencoder 尝试拆解 superposition

一个 neuron 同时对城市名、代码符号和否定词有反应时，很难给它贴上唯一标签。

Superposition hypothesis 提供了一种解释：模型可能在有限维 residual space 中，用彼此不正交的方向
表示更多稀疏 features，于是单个 neuron 看起来具有多种语义。这是一种待检验的表示模型，
不是已经适用于所有网络的定理。

Sparse autoencoder（SAE）常学习：

\[
f=\operatorname{ReLU}(W_{enc}x+b_{enc}),
\qquad
\hat x=W_{dec}f+b_{dec},
\]

并优化：

\[
\lVert x-\hat x\rVert_2^2+\lambda\lVert f\rVert_1.
\]

实际 SAE 也可能只保留 top-k activations、约束 decoder 向量的范数，或采用不同的归一化。不能只展示几个
“看起来很会说话”的 feature 名称，至少要从下面几方面评价：

| 问题 | 可以观察的指标 |
|---|---|
| 原表示保留了多少 | Reconstruction error、下游 loss 变化 |
| 表示是否真的稀疏 | L0/L1、每条样本激活的 feature 数 |
| 字典是否健康 | Activation frequency、dead features、不同宽度下的稳定性 |
| 人类解释能否泛化 | Held-out 样本上的 precision/recall |
| Feature 是否影响行为 | 预先设计的 feature intervention |

一个 feature 容易命名，不代表它对应真实且唯一的概念。随机种子、字典宽度或 regularization 改变后，
feature 可能分裂、合并或消失；这些稳定性本身就是实验结果。

## Model editing 检查改变是否局部而持久

Model editing 的目标不是重新训练整套模型，而是有针对性地改变某项行为。做法可以是更新少量参数或层、
施加 low-rank update、编辑中间表示，也可以把新信息放进 adapter、外部 memory 或检索系统。

至少从七个方向评估：

| 维度 | 问题 |
|---|---|
| 生效（Efficacy） | 目标 Prompt 是否更新 |
| 改写（Paraphrase） | 换一种问法是否一致 |
| 迁移（Portability） | 相关推论和其他语言是否同步 |
| 局部性（Locality） | 无关事实与能力是否保持 |
| 特异性（Specificity） | 相似实体是否被误改 |
| 持久性（Persistence） | 保存、merge 或继续训练后是否仍存在 |
| 冲突（Conflict） | 参数记忆与新的 RAG 证据冲突时怎样处理 |

把一条问答改对，不代表相关知识已经全局一致。经常变化、需要删除或追溯来源的事实，通常更适合放在
数据库或 RAG 中，因为来源、回滚和访问权限更容易审计。

## 可解释性工具也需要自己的验收

可解释性代码也可能“图画得很漂亮，但 hook 根本接错了”。至少检查：

- 随机化权重或标签后，attribution/probe 是否仍几乎不变；
- 更换 baseline、metric 或 Prompt 后，结论是否稳定；
- Hook 拿到的是 norm 前后、投影前后，还是完整 module output；
- KV cache、tensor parallel 和 quantization 是否改变 site；
- Patch 后 forward 是否真正消费新 activation；
- 更换 batch、dtype 或重复运行后是否复现。

Module 名称不是可靠的概念边界。名为 `attention` 的 hook 可能拿到每个 head 的 value、合并后的结果，
也可能已经经过输出投影。要结合源码、张量形状和一个结果可预测的最小干预来确认。

## 从 checkpoint 开始的一次机制审计

1. 记录模型与 tokenizer 的具体 revision，以及代码、dtype 和执行后端；
2. 读取配置与张量形状，画出真实计算图、推理状态和可干预的 residual sites；
3. 建立行为数据，预先定义连续 metric 和 clean/corrupt 对照；
4. 先跑行为基线，再做 probe 或 attribution，避免直接从内部图猜故事；
5. 写下因果预测，然后用 ablation 或 patching 尝试推翻它；
6. 加入无关来源、未来位置和随机权重等负对照；
7. 在不同模板、语言、位置、事实和随机种子上报告结果分布；
8. 把结论限定在实际测过的行为、metric、site 和 intervention 内。

仓库目前先用随机 MiniGPT 检查 hook，再在上述 Qwen checkpoint 上完成一次单事实干预。
这说明实验流程和局部 intervention 可以运行。要研究目标模型的普遍机制，还需要多样本、事先写定的分析方案
和 held-out 验证。

## 安全结论不能从一个 circuit 外推

内部分析可以帮助发现记忆现象、router collapse、异常 features 或拒答路径，但覆盖范围有限。
Features 会随上下文变化，量化或分布式执行也可能改变 hook 对应的实际计算。

即使一条 circuit 在当前 Prompt 上非常稳定，也只解释了模型内部的一小部分行为。系统风险还可能来自
RAG、工具、权限、cache 和 runtime。

高风险部署仍需要行为评测、red teaming、访问控制、审批、监控和事件响应。

## 自测与实践

1. 分别画 encoder-only、decoder-only 和 encoder-decoder 的可见性矩阵。
2. 为什么同参数量、同 FLOPs 和同 wall time 可能得到不同架构排名？
3. 为一个 probe 设计按实体和模板分组的切分，怎样避免词面泄漏？
4. Activation patching 的 clean-corrupt 分母接近零时应怎样报告？
5. Layer 11 的 source patch 恢复 0.99，为什么仍不能说“事实存储在第 11 层”？
6. 为一个 SAE 报告设计重建、稀疏度、稳定性和因果干预四类指标。
7. 选择公开小模型，在运行 patching 前写下行为、metric、hook shape、负对照和结论边界。
