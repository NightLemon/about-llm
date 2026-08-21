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

“像 GPT”或“属于某模型家族”只能作为线索。运行契约来自具体 config、tokenizer/processor、
tensor shapes、model implementation 和 immutable revision。

## Encoder-only：每个输入位置双向理解整段

标准 encoder self-attention 允许每个非 padding token 读取整段输入：

\[
H'=\operatorname{Attention}(Q(H),K(H),V(H)).
\]

它常用于 masked language modeling、替换检测和对比学习，输出每个 token 的 contextual representation，
或聚合成 sequence representation。典型任务包括分类、reranking、span tagging、dense retrieval 和抽取式问答。

标准 encoder 没有左到右开放生成所需的 causal mask 与 autoregressive loop。
可以外接 decoder，或使用 iterative mask-predict 等机制；这样已经改变了原始推理协议。
准确表述是“标准 encoder-only 接口不直接执行 causal generation”，而不是给所有改造后的系统下绝对结论。

## Decoder-only：输入和输出共享一条 causal stream

位置 \(t\) 只能读取左侧历史：

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

训练时 causal mask 允许并行计算所有位置的 logits；推理时通常逐 token decode，并复用 KV cache。
Prompt、工具 schema、RAG evidence 和输出都在同一 token stream，很多任务因此可以统一成 continuation。

代价也来自这条统一序列：

- 输入 token 在同层无法读取右侧位置；
- 长 Prompt 的 prefill 和 KV cache 消耗计算与显存；
- System、user、tool 与 assistant 边界依赖 template 和 special tokens；
- Training loss mask 与 inference generation prompt 必须对齐。

“Decoder-only 更通用”是生态、数据与训练结果的经验总结，不是一条只由 causal mask 推导出的架构定理。

## Encoder-decoder：source 与 target 天然分开

Encoder 双向处理来源 \(x\)，decoder 用 causal self-attention 处理 \(y_{<t}\)，
并通过 cross-attention 读取 encoder states：

\[
p(y\mid x)=\prod_t p(y_t\mid y_{<t},H_{enc}(x)).
\]

这种分工适合翻译、摘要、改写、结构转换和条件生成。Encoder output 可以在 decode 时复用，
但 cross-attention 仍要读取 encoder K/V。

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

Tensor shape 能 broadcast，只说明代码可以运行。语义验证应在短序列上打印每个 query 可见的 key set，
并比较训练与 cached decode 使用的 mask 是否一致。

## Recurrent、convolution 和 SSM 压缩历史

RNN 将历史压入状态：

\[
h_t=f(h_{t-1},x_t),
\qquad
y_t=g(h_t).
\]

推理 state 可以保持固定大小，适合 streaming；时间依赖却会限制训练并行，长程 gradients 也可能消失或爆炸。
LSTM / GRU 用 gating 缓解这些问题，没有取消历史压缩这个基本约束。

一维 convolution 并行混合局部窗口，stack 或 dilation 扩大 receptive field。
长 convolution 还可以使用 FFT 等算法。它的 mixing kernel 通常比 attention 更固定，
与 input-dependent content addressing 的行为不同。

线性 state-space model（SSM）的一种离散形式是：

\[
h_t=\bar Ah_{t-1}+\bar Bx_t,
\qquad
y_t=Ch_t+Dx_t.
\]

线性时不变形式可以等价写成 convolution，训练时并行、推理时 recurrent。
Selective SSM 让部分更新参数依赖输入，提高 content sensitivity，也改变可用并行算法和 cache state。

线性复杂度不会自然变成更低 wall time。Kernel fusion、state size、batch、sequence length 和硬件决定 crossover；
固定 state 还可能成为精确复制或远距离检索的瓶颈。

混合模型可以交替 attention、SSM、convolution 或 recurrent layers，
让 attention 负责精确寻址，让其他模块承担更便宜的长程 mixing。

## 同为 Transformer，内部仍有许多架构轴

| 轴 | 常见变体 | 主要影响 |
|---|---|---|
| Norm placement | Pre、Post、sandwich | 优化稳定、residual path 与 checkpoint |
| Norm type | LayerNorm、RMSNorm | 统计量、parameters 与数值行为 |
| MLP | GELU、SwiGLU、gated variants | 参数量、activation 与 kernels |
| Attention heads | MHA、MQA、GQA、compressed KV | 质量、KV size 与 projections |
| Position | Absolute、RoPE、relative bias、ALiBi-like | 长度外推与实现 |
| Attention span | Full、sliding、block sparse、hybrid | Receptive field、cache 与 kernels |
| FFN routing | Dense、MoE | Active compute、weight memory 与通信 |
| Residual topology | Serial、parallel、scaled | 计算图和 weight conversion |

加载第三方权重时检查 config、head counts、RoPE scaling、tied embeddings、tensor shapes 和 model code。
用“看起来合理”的默认值补缺失字段，可能让模型能输出文字，却已经改变原 checkpoint 的函数。

## MoE 同时改变权重、计算与通信

Sparse Mixture-of-Experts（MoE）为 token 计算 router scores，选择 top-k experts 后加权合并：

\[
y(x)=\sum_{e\in\operatorname{TopK}(r(x))}g_e(x)E_e(x).
\]

总 parameters 决定 weight storage，active parameters 影响 token compute，
expert placement 和 all-to-all 决定通信。三者不能用一个“模型大小”数字替代。

实现还要定义 top-k tie-break、capacity、overflow/drop/reroute/dropless policy、shared experts、
router precision、load-balancing loss 和 post-drop gate normalization。

Router index 是离散选择。Hard index 本身不会自然把 task gradient 送回 router；
训练通常依赖 selected probabilities、auxiliary objective 或其他 estimator。

可解释性分析还要分开两个问题：expert 在处理什么，以及哪些 tokens 为什么被路由到它。
平均激活主题或人为 expert 名称，并不证明一个 expert 只有单一功能。

仓库提供了一组 NumPy/PyTorch 小实验，逐层检查 capacity、sparse-vs-dense forward/backward、
two-process global competition、owner-only all-to-all 和 reverse gradient。实验使用 CPU/Gloo 和本仓库准备的
固定输入，目的是把路由与通信过程拆开观察。DeepSeek、Qwen 的实际 routing policy 和目标 GPU 性能仍需另行验证。
完整通信主线见
[分布式训练](../systems/distributed-training.md)。

## 从架构推导运行时依赖 { #architecture-runtime-dependencies }

下载到 `safetensors` 只得到 tensors。真正运行 checkpoint，还要让下面几层同时匹配：

```text
config + tensor identity
-> tokenizer / processor / chat template
-> model graph + operator semantics
-> decode state / cache layout
-> kernels + quantization
-> batching / scheduling / cancellation / output parser
```

不同架构特征会把要求传给不同组件：

| 架构特征 | 推理 state | Runtime 额外需要支持 |
|---|---|---|
| Causal MHA / GQA | Per-layer K/V | Model class、RoPE、GQA kernels、KV allocator |
| Sliding / sparse attention | Windowed or hierarchical K/V | Mask-aware kernel、eviction 与 prefix semantics |
| Compressed / latent KV | 非标准 latent cache | 专用 graph、cache object、kernel 与容量公式 |
| SSM / recurrent | Recurrent + convolution state | Prefill scan、state update、sequence reorder |
| Attention-recurrent hybrid | K/V 和 recurrent state 并存 | Heterogeneous cache 与 layer lifecycle |
| Sparse MoE | Route / capacity / expert placement | Fused MoE、expert quantization、多卡 all-to-all |
| Vision/audio encoder + LM | Media tensor、encoder output、positions | Processor、fusion graph、multimodal batching |
| Early-fusion multimodal | Joint sequence + media metadata | Matching mask、position/RoPE 与 serving schema |
| Extra prediction heads | Candidates + verification state | Loader、speculative scheduler 与 fallback |

几种常见“假兼容”是：

- `AutoModel` 能 instantiate，便宣称 serving runtime 支持 cache 和 batching；
- Sliding attention fallback 到 full attention 后能输出，仍沿用原性能与显存估算；
- 文本请求成功，就宣称 multimodal processor 与 media path 可用；
- Checkpoint 中存在额外 head，就宣称 speculative decoding 已加速；
- MoE active parameters 小，就推断整模型权重一定能放入设备。

### 不同库分别负责哪一层

| 组件 | 主要职责 | 仍需其他层验证什么 |
|---|---|---|
| Transformers / JAX model | Config、graph、tensor loading、forward/generation | 高吞吐 scheduling、paged state、cancellation |
| Tokenizer / processor | Template、special tokens、media preprocessing | 服务端是否使用同一 revision，模型是否读取 media |
| vLLM / SGLang runtime | Model registration、state、batching、streaming | 新 graph 的 kernels、parser、quantization combinations |
| Nano teaching runtime | 精简 engine、scheduler、cache 和 runner | 未注册 graph、hybrid state 和 production semantics |
| FlashAttention / Triton / CUTLASS / fused MoE | 特定 shape/dtype/mask 上的 operators | 完整 model、state lifecycle 与端到端加速 |
| PEFT / TRL / trainer | Adapter、collator、loss、optimizer loop | Target modules、mask、serving reload 与质量 |
| llama.cpp / GGUF stack | 已实现 graph 的 conversion、quantization、execution | 任意 HF weights、所有 modalities 与无损转换 |

### 用能力阶梯记录“支持到哪”

1. **识别**：Config、`model_type`、architectures 和 tensor names 对应；
2. **实例化**：目标 dtype/device 可加载，missing/unexpected keys 符合预期；
3. **模型执行**：Prefill、cached/recurrent decode 与 reference 对账；
4. **额外路径**：Media counterfactual、MoE routing、extra heads 真正执行；
5. **服务执行**：Batch、reorder、prefix、cancel、OOM/fallback 与 parser 正确；
6. **目标设备验收**：固定 workload 下验证质量、memory、TTFT、TPOT、throughput 和 failures。

低层通过不能继承高层结论。拿到新模型时先填一张 compatibility card：

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

1. **相同参数量**：比较参数容量，training FLOPs、KV 和 activations 仍可能不同；
2. **相同训练 FLOPs/token**：比较计算效率，weight memory 可能不同；
3. **相同 hardware/wall time**：比较真实系统，kernel maturity 也进入结果。

数据、tokenizer、context、optimizer、training tokens、tuning budget 和 serving batch 还要固定或报告。
比起寻找“架构总冠军”，更诚实的是画 quality–training cost、quality–latency 和 quality–memory Pareto front。

## 从行为到机制：先看证据阶梯

| 证据层 | 常用方法 | 可以支持什么 |
|---|---|---|
| 行为 | Minimal pair、counterfactual Prompt | 输入变化与输出变化相关 |
| 可解码性 | Linear probe | 某信息可以从 representation 读出 |
| 归因 | Gradient、IG、occlusion、attention analysis | 局部 sensitivity 或 score 分配 |
| Component intervention | Ablation、activation patching | 指定干预会改变 metric |
| Path intervention | Path patching、causal scrubbing | 某条受限路径对该行为有因果作用 |

结论强度要停在证据所在层。Probe accuracy 很高，只说明 probe 能取得信息；
activation patching 恢复一个 logit metric，也不等于发现了唯一自然 circuit。

## 行为实验先定义连续 metric

France/Germany pair 只改变国家 token，并尽量固定 template、length、tokenization 和 surface frequency。
机制实验常用正确与错误 token 的 logit difference：

\[
m(x)=z_{correct}(x)-z_{incorrect}(x).
\]

它比 exact match 更敏感，可以观察最终 argmax 翻转之前的变化。
但 token pair 本身限定了研究问题；多-token 答案需要 sequence score，不能只挑有利的首 token。

负对照至少包括：

- Patch 无关位置或未来位置；
- 随机打乱 source activations；
- 使用同分布但任务无关的 source；
- 检查改变是否只来自 activation norm；
- 在多个 templates、positions、languages 和 seeds 上复现。

## Probe 检查“能否读出”，不是“是否使用”

Linear probe 从 hidden state \(h\) 预测属性：

\[
\hat y=Wh+b.
\]

设计 split 时防止同一词项、实体和 template 跨 train/test；控制类别、长度和 token position，
并与简单 input features 比较，而不只和 random baseline 比。

Probe 太强时会自己学会任务。选 layer、feature 和 hyperparameters 后继续用同一 test，也会产生 selection bias。

更强的下一步是干预 probe 识别的 direction，预测目标行为怎样变化，并检查无关能力是否保持。

## Attribution 描述 sensitivity，仍要小心输入分布

Gradient 说明当前点附近，输出分数对输入小变化的 sensitivity。Saturation 时 gradient 可能很小，
即使 feature 对离开当前区域仍然重要。

Integrated Gradients 从 baseline \(x'\) 沿路径积分：

\[
IG_i(x)=(x_i-x_i')
\int_0^1
\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}
d\alpha.
\]

结果依赖 baseline、path 和 target function。对 token embeddings 的线性插值，
中间点通常不对应自然语言输入。

Occlusion 删除、mask 或置零 feature，容易产生 out-of-distribution 输入；相关 features 之间的交互也不会自然分摊。

Attention weights 是某层某 head 对 values 的加权系数。它没有包含 value content、output projection、residual、MLP 和后续 layers，
因此可以作为分析对象，不能直接当作最终 token importance。

## Activation patching 直接干预中间表示

基本实验是：

1. 在 clean input 上缓存 activation 和 \(m_{clean}\)；
2. 在 corrupt input 上得到 \(m_{corrupt}\)；
3. 运行 corrupt input，同时用 clean activation 替换指定 site；
4. 得到 \(m_{patched}\)。

常用恢复分数：

\[
R=\frac{m_{patched}-m_{corrupt}}{m_{clean}-m_{corrupt}}.
\]

分母接近零时该比例不稳定，应该失败或只报告 raw metrics。\(R>1\) 和 \(R<0\) 都可能出现，
不应裁剪到 `[0,1]` 后再解释。

Patching 把另一个输入的 activation 放进当前 computation，可能制造训练分布之外的组合。
正确结论是“在这项 intervention 下，该 site 改变了指定 metric”。Redundancy、backup path 和 entangled features
仍可能让单组件 ablation 或 patching 给出不完整答案。

### 先用随机 MiniGPT 检查 hook 和因果负例

仓库的 `activation_patching.py` 在固定 seed 的两层 MiniGPT 上执行真实 PyTorch hook：

~~~powershell
python projects/transformers-basics/activation_patching.py
python -m pytest tests/test_activation_patching.py -q
~~~

Clean `[1,2,3,4]` 与 corrupt `[5,2,3,4]` 只改首 token。
Patch layer-0 post-residual 的 source、readout 和 joint causal prefix 会按预期改变预先选定的 logit metric；
patch future position 不能改变过去 readout，作为 causal mask 与 hook-site 负对照。

这一步只检查随机、未训练模型上的张量和 hook 是否按预期工作。Metric token pair 是看过 clean/corrupt 差异后
为这个样例选择的，joint prefix 恢复 clean output 也是特意构造的正例。因此它还不是自然语言 circuit discovery。

### 在固定 Qwen checkpoint 上做一次单事实干预

`run_qwen_activation_patching_control.py` 加载指定版本的 Qwen2.5-0.5B-Instruct。
Templated clean/corrupt inputs 都是 26 tokens，仅位置 19 从单 token ` France` 改成 ` Germany`；
readout position 25 使用 `Paris - Berlin` logit difference。

~~~powershell
python projects/transformers-basics/run_qwen_activation_patching_control.py `
  --local-files-only
~~~

实验开始前就选定 layers 0、11、23，不在看到结果后挑最好层。记录到的结果如下：

| Patch | Recovery | 应怎样解释 |
|---|---:|---|
| Source pos 19, layer 0 | 约 1.000 | 整个 source residual 替换后恢复这一个 metric |
| Source pos 19, layer 11 | 约 0.992 | 该 pair/intervention 上接近恢复，尚未定位 lookup 或 routing |
| Source pos 19, layer 23 | 0 | 最后一层 source site 后没有跨位置 mixing |
| All positions, layer 0 | 1 | 特意构造的正例 |
| Readout pos 25, layer 23 | 1 | 直接替换最终 readout residual 的正例 |
| Future pos 26, layer 0 | 0 | Future patch 不能改变过去 readout |

这次干预已经运行在真实 target weights 上，但样本仍然只有一个英文事实、一个 template，batch size 也是 1。
Source patch 一次替换 896 dimensions，可能同时搬运实体、词形、position 和其他 features。

要形成机制研究，还需要 random/unrelated sources、paraphrases、cross-language pairs、多个 facts、
component/path interventions 和 held-out replication。无密钥 report hash 也只绑定记录内容，不认证执行来源。

## Path patching 把问题缩到组件之间的边

Path patching 尝试只替换 source component 对指定 downstream component 的影响，同时阻断其他传播。
一个 circuit hypothesis 应明确：

```text
nodes: heads / MLPs / features / residual sites
edges: projection and token-position path
behavior + metric
necessity: remove circuit, behavior degrades
sufficiency: keep circuit, behavior remains
faithfulness: intervention does not create unrelated failures
```

即使在几十个 templates 上成立，结论范围仍是这组 behavior 和 interventions，
不能直接升级为“整个模型的算法”。

## Sparse Autoencoder 尝试拆解 superposition

Superposition hypothesis 提出：模型可以在有限维 residual space 中用非正交 directions 表示更多 sparse features，
从而产生 polysemantic neurons。它是解释表示现象的一种模型，不是对所有网络的完成证明。

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

实现还可能使用 top-k activations、decoder norm constraints 和不同 normalization。
评价要同时看 reconstruction error、downstream loss、L0/L1 sparsity、activation frequency、dead features、
dictionary size sensitivity、held-out explanation precision/recall 和 causal feature intervention。

一个 feature 容易命名，不代表它是真实、唯一的 ontology。Seed、width 和 regularization 改变后，
features 可能 split、merge 或消失。

## Model editing 检查改变是否局部而持久

Model editing 可以对少量 parameters/layers 更新、使用 low-rank update、编辑某个 representation，
或把新信息放在 adapter、external memory 与 retrieval 中。

至少从七个方向评估：

| 维度 | 问题 |
|---|---|
| Efficacy | 目标 Prompt 是否更新 |
| Paraphrase | 改写是否一致 |
| Portability | 相关推论和语言是否同步 |
| Locality | 无关事实与能力是否保持 |
| Specificity | 相似实体是否被误改 |
| Persistence | 保存、merge 或继续训练后是否存在 |
| Conflict | 参数记忆与新 RAG evidence 冲突时怎样处理 |

把一条问答改对，不代表内部知识关系已经全局一致。时效事实通常更适合数据库或 RAG，
因为 source、删除与 rollback 更容易审计。

## 可解释性工具也需要自己的验收

Sanity checks 包括：

- 随机化 weights 或 labels 后，attribution/probe 是否仍几乎不变；
- 更换 baseline、metric、Prompt 后，结论是否稳定；
- Hook 对应 pre/post norm、projection 还是完整 module output；
- KV cache、tensor parallel 和 quantization 是否改变 site；
- Patch 后 forward 是否真正消费新 activation；
- Batch、dtype 和重复运行是否复现。

Module 名称不是概念边界。名为 `attention` 的 hook 可能拿到 per-head values、合并输出，
也可能已经经过 output projection。必须结合源码、tensor shape 与最小干预确认。

## 从 checkpoint 开始的一次机制审计

1. 固定 model/tokenizer revision、代码、dtype 和 execution backend；
2. 读取 config 与 shapes，画真实 graph、state 和 residual sites；
3. 建 behavior dataset、continuous metric 和 clean/corrupt pairs；
4. 先做 behavioral baseline，再做 probe 或 attribution；
5. 用 ablation/patching 检验预先写下的因果预测；
6. 加入 unrelated source、future position、random weight 等负对照；
7. 在 templates、languages、positions、facts 和 seeds 上报告分布；
8. 将结论限定在 behavior、metric、site 与 intervention 范围内。

仓库目前用随机 MiniGPT 检查 hook，又在固定 Qwen checkpoint 上完成了一次单事实干预。
两者说明这套实验流程和局部 intervention 可以运行；要研究目标模型的普遍机制，还需要多样本、预注册方案和
held-out 验证。

## 安全结论不能从一个 circuit 外推

内部分析可以帮助发现 memorization、router collapse、异常 features 或 refusal path，
但覆盖有限，features 会随 context 变化，quantized/distributed execution 也可能改变 hook 位置。

即使一条 circuit 在当前 prompts 上很稳定，系统风险还来自 RAG、tools、permissions、cache 和 runtime。
高风险部署仍需要行为评测、red teaming、访问控制、审批、监控和 incident response。

## 自测与实践

1. 分别画 encoder-only、decoder-only 和 encoder-decoder 的可见性矩阵。
2. 为什么同参数量、同 FLOPs 和同 wall time 可能得到不同架构排名？
3. 为一个 probe 设计 entity/template group split，避免 lexical leakage。
4. Activation patching 的 clean-corrupt denominator 接近零时应怎样报告？
5. Layer 11 source patch 恢复 0.99，为什么还不能说“事实存储在第 11 层”？
6. 为一个 SAE 报告设计 reconstruction、sparsity、stability 和 causal intervention 四类指标。
7. 选择公开小模型，在运行 patching 前写下 behavior、metric、hook shape、负对照和结论边界。
