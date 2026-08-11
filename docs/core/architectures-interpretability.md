# 架构谱系与机制可解释性

架构决定信息在 token、层和状态之间怎样流动；可解释性研究模型内部表示与输出之间有什么证据关系。两者经常一起讨论，但必须区分：**知道一层的结构，不等于知道它学到了什么；观察到一个相关模式，也不等于证明模型依赖该模式。**

## 1. 用信息流而不是品牌名分类

判断一个模型的基本架构，先回答：

1. 每个输入位置能看见哪些位置？
2. 输入和输出是否由独立模块处理？
3. 历史信息存为全部 token 的 K/V，还是压缩为固定/受限状态？
4. 每个 token 激活所有参数还是部分专家？
5. 训练目标、attention mask 与推理循环是什么？

“某模型像 GPT/Llama”只是线索。权威运行契约来自具体 checkpoint 的 config、tokenizer、权重形状和实现版本。

## 2. Encoder-only

标准 encoder self-attention 允许每个非 padding token 双向访问整段输入：

\[
H'=\operatorname{Attention}(Q(H),K(H),V(H)).
\]

常见预训练目标是 masked language modeling、替换检测或对比学习。输出可以是每个 token 的 contextual representation，也可以聚合成 sequence representation。

### 2.1 擅长什么

- 文本分类与 reranking；
- token/span 标注；
- dense retrieval embedding；
- 成对文本相关性；
- 抽取式问答。

标准 encoder 没有按左到右开放生成所需的 causal mask 和 autoregressive loop。可以给它增加 decoder、迭代 mask-predict 或其他生成机制，但那已经改变了标准推理协议；不能简单说“encoder 永远不能生成”，也不能说它能直接等价替代 decoder-only generation。

## 3. Decoder-only

因果 self-attention 对位置 \(t\) 屏蔽未来位置：

\[
p(x_{1:T})=
\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

训练时可以用 causal mask 并行计算所有位置的 logits，推理时通常逐 token decode 并复用 KV Cache。Prompt、工具 schema、检索证据和输出处于同一 token stream，便于把很多任务统一成 continuation。

### 3.1 优势与代价

- 单一接口覆盖生成、in-context learning 和工具调用；
- 大量 serving kernel 和训练生态成熟；
- 输入 token 也使用因果表示，不能像 bidirectional encoder 一样在同层整合右侧输入；
- 长 prompt 的 prefill 和 KV Cache 会占用计算/显存；
- 输入、指令和输出边界依赖 special token、chat template 与 loss mask。

“decoder-only 更通用”是生态和训练结果的经验描述，不是架构定理。

## 4. Encoder-decoder

Encoder 双向处理来源序列 \(x\)，decoder 以 causal self-attention 处理目标前缀 \(y_{<t}\)，并通过 cross-attention 读取 encoder states：

\[
p(y\mid x)=
\prod_t p(y_t\mid y_{<t},H_{enc}(x)).
\]

它天然区分 source 与 target，适合翻译、摘要、改写、结构转换和条件生成。Encoder 输出可在整个 decode 过程中复用，但 decoder cross-attention 仍需读取 encoder K/V。

代价包括两套 stack、更多 shape/缓存协议，以及 serving 生态可能不如主流 decoder-only 路径统一。不能只凭“输入编码一次”断言总成本更低：还要比较层数、宽度、输入/输出长度和 kernel。

## 5. Prefix 与 attention mask 变体

Prefix LM 让 prefix 内部双向可见，生成区域只能看 prefix 与自己的左侧。UniLM 类方法可通过不同 mask 在共享参数上表达多种任务。重点不是名称，而是实际 attention mask：

- padding mask：排除无效位置；
- causal mask：排除未来位置；
- prefix/block mask：定义分区信息流；
- local/sliding-window mask：限制可见距离；
- global token：允许少数位置做全局连接。

Mask 写错会产生训练泄漏或推理不匹配。形状可广播不代表语义正确，应在小序列上打印每个 query 可见的 key 集合。

## 6. RNN、卷积与状态空间模型

### 6.1 递归状态

RNN 用状态更新概括历史：

\[
h_t=f(h_{t-1},x_t),
\qquad
y_t=g(h_t).
\]

推理状态可固定大小，天然支持 streaming；但训练的时间依赖通常限制并行，长程梯度也可能消失/爆炸。LSTM/GRU 用门控缓解而非彻底消除这些问题。

### 6.2 卷积与长卷积

一维卷积并行处理局部窗口，堆叠或 dilation 扩大 receptive field。长卷积可借 FFT 等方法提高某些序列长度下的效率。固定 kernel 擅长位置相对稳定的混合，但与 content-dependent attention 的行为不同。

### 6.3 State Space Model（SSM）

连续线性状态空间的一种离散形式为

\[
h_t=\bar A h_{t-1}+\bar Bx_t,
\qquad
y_t=Ch_t+Dx_t.
\]

线性、时不变形式可等价写成卷积，从而训练时并行、推理时递归。现代 selective SSM 让部分状态更新参数依赖输入，增加 content sensitivity；这也会改变可用的并行算法。

SSM 的潜在优势是长序列扫描和受限 recurrent state，但“线性复杂度”不自动等于真实 GPU 更快。常数、kernel fusion、state size、batch、sequence length 和硬件决定 crossover point。固定状态还会形成信息瓶颈，精确复制或远距离检索能力必须实测。

### 6.4 混合架构

混合模型可交替 attention、SSM、convolution 或 recurrent layers，用 attention 做精确 content addressing，用其他模块做廉价混合。评价应在相同质量、上下文、batch 与硬件上比较，而不是只对 Big-O。

## 7. Transformer 内部仍有多个架构轴

即使都叫 decoder-only Transformer，也可能在以下方面不同：

| 轴 | 变体 | 影响 |
| --- | --- | --- |
| Norm 位置 | Pre-Norm、Post-Norm、sandwich | 优化稳定、残差路径与 checkpoint 兼容 |
| Norm 类型 | LayerNorm、RMSNorm | 统计量、参数与数值行为 |
| MLP | GELU、SwiGLU、其他 gated MLP | 参数量、激活与 kernel |
| Attention heads | MHA、MQA、GQA、latent/compressed KV | 质量、KV Cache 与投影协议 |
| Position | learned absolute、RoPE、relative bias、ALiBi 类 | 长度外推与实现 |
| Attention span | full、sliding、block sparse、global/local mix | 感受野、成本与 kernel |
| FFN routing | dense、MoE | 激活计算、权重内存与通信 |
| Residual topology | serial、parallel、scaled residual | 计算图与权重转换 |

模型名称不能推断这些细节。加载第三方权重时应检查 config、tensor shapes、rope scaling、head counts、tie embeddings 和 remote code，而不是手写“看起来合理”的默认值。

## 8. MoE 是路由架构，不只是大参数量

典型 sparse MoE 层为每个 token 计算 router score，选择 top-k experts，再按权重合并：

\[
y(x)=\sum_{e\in\operatorname{TopK}(r(x))}
g_e(x)E_e(x).
\]

训练还需要考虑 load balancing、expert capacity、token dropping/padding、router precision 与 expert parallel all-to-all。总参数决定权重存储，active parameters 影响 token 计算，但通信和共享层不随 active count 简单缩放。

仓库 NumPy oracle 把一份具体协议钉死：padding 不进 capacity，top-k 同分按 expert id，expert 内按 probability/token/rank 竞争容量，分别报告 dropped assignment 与整 token drop，并显式选择 drop 后是否重归一化。它能执行 sparse linear expert combine，但不是训练过的 MoE、目标 checkpoint 或 expert-parallel kernel；不同模型的 capacity 与 auxiliary loss 仍须查对应实现。

Router 选择是输入依赖的，因而可解释性分析还要区分“专家本身做什么”和“哪些 token 被路由到它”。专家名称或平均激活主题不证明专家具有单一功能。

## 9. 架构比较的正确实验

公平比较至少有三种口径，各回答不同问题：

1. **相同参数量**：比较容量，但训练 FLOPs/KV/激活可能不同；
2. **相同训练 FLOPs/token**：比较计算效率，但权重内存可能不同；
3. **相同 wall-clock/hardware**：比较实际系统，但 kernel 成熟度成为结果的一部分。

还需固定或报告数据、tokenizer、上下文、optimizer、训练 token、调参预算和 serving batch。只比较论文中的最好数字会混入数据和工程差异。

建议画 Pareto frontier：质量—训练成本、质量—延迟、质量—显存，而不是寻找一个“架构总冠军”。

## 10. 可解释性的证据阶梯

可解释性方法回答的问题强度不同：

| 层级 | 例子 | 能支持的结论 | 不能自动支持 |
| --- | --- | --- | --- |
| 行为 | 最小对比、反事实 prompt | 输入变化与输出相关 | 内部哪个组件负责 |
| 可解码性 | linear probe | 信息可从表示读出 | 模型实际使用该信息 |
| 归因 | gradient、IG、occlusion | 局部敏感度/分数分配 | 稳定因果机制 |
| 组件干预 | ablation、activation patching | 干预组件会改变指标 | 唯一、自然或完整回路 |
| 路径干预 | path patching、causal scrubbing | 某路径对特定行为有因果作用 | 对所有输入都同样成立 |

结论应匹配证据层级。Probe accuracy 很高不能写成“模型靠这个特征推理”。

## 11. 行为实验与最小对比

先定义一个可复现行为：例如 subject–verb agreement、indirect object identification、copying 或拒答。构造只改变一个相关因素的 clean/corrupted pair，同时尽量控制 tokenization、长度和表面频率。

### 11.1 选择连续指标

Exact match 太离散。机制实验常用目标 token logit difference：

\[
m(x)=z_{correct}(x)-z_{incorrect}(x).
\]

它能观察尚未翻转最终 argmax 的变化。但选择哪两个 token 会限定结论；多 token 答案还要定义 sequence score，不能任意只看第一 token。

### 11.2 负对照

- 对无关位置做同样 patch；
- 随机打乱 patch source；
- 用同分布但任务无关样本；
- 检查 patch 是否只改变 activation norm；
- 在多个 prompt template、token 位置和 seed 上复现。

## 12. Probe：可读出不等于被使用

给定 hidden state \(h\)，linear probe 学习 \(Wh+b\) 预测属性。要避免：

- train/test 中出现同一词项或模板导致 lexical leakage；
- probe 容量太强，自己学会任务；
- 类别、长度或位置混淆；
- 只和 random baseline 比，而不和简单输入特征比；
- 在选层和调参后仍把同一测试集当无偏结果。

更强证据需要 intervention：修改 probe 识别的方向后，目标行为是否按预测改变，同时无关能力保持。

## 13. 归因方法

### 13.1 Gradient 与 Integrated Gradients

Gradient 回答输出分数对当前输入附近微小变化的敏感度；饱和区可能梯度小但特征仍重要。Integrated Gradients 沿 baseline 到输入的路径积分：

\[
IG_i(x)=(x_i-x_i')
\int_0^1
\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}
d\alpha.
\]

结果依 baseline、路径和目标函数。对离散 token embedding，插值路径未必对应自然语言输入。

### 13.2 Occlusion

遮蔽 token/组件并观察输出变化直观，但替换为 `[MASK]`、删除或零向量会构造不同的 out-of-distribution 输入。相邻特征相关时，单独遮蔽也不能分摊交互贡献。

### 13.3 Attention weights

Attention matrix描述某层某头 value 的加权系数，不包含 value 内容、output projection、残差、MLP 和后续层。高 attention weight 不等于对最终 logit 的高因果贡献；低权重路径也可能因 value magnitude 大而重要。Attention 可作为分析对象，不能天然当解释。

## 14. Activation patching

基本流程：

1. 在 clean input 上缓存激活和高目标指标 \(m_{clean}\)；
2. 在 corrupted input 上得到 \(m_{corrupt}\)；
3. 运行 corrupted input，但把某位置/层/组件激活替换为 clean activation；
4. 测量 \(m_{patched}\)。

常用归一化恢复分数：

\[
R=
\frac{m_{patched}-m_{corrupt}}
{m_{clean}-m_{corrupt}}.
\]

只有在分母非零且 metric 方向一致时该量才有意义；\(R>1\) 或 \(R<0\) 都可能发生，不能强行截到 `[0,1]` 后再解释。还要报告原始 metric，避免小分母放大噪声。

### 14.1 干预分布问题

把另一个 prompt 的激活直接塞入当前计算，可能产生模型训练时未见的组合。Patching 提供比观察相关性更强的因果证据，但结论是“在该干预定义下影响该指标”，不等于模型在自然运行中唯一依赖该组件。

### 14.2 Redundancy 与 backup behavior

消融一个头没有影响，可能因为它不重要，也可能因为冗余路径补偿；一次 patch 恢复很大，也可能同时带入多个纠缠特征。应结合单组件、联合消融、路径限制与跨样本验证。

### 14.3 可运行的 residual-stream patching reference

仓库提供 `projects/transformers-basics/activation_patching.py`。它建立固定 seed 的两层随机 MiniGPT，在 clean `[1,2,3,4]` 与只改首 token 的 corrupted `[5,2,3,4]` 上缓存第 0 层 block 的 **post-residual output**，再通过真实 PyTorch forward hook 替换对齐 batch 的指定 position。metric 是位置 1 上 token 27 与 token 19 的 logit difference；这对 token 是编写 fixture 时按 clean-corrupt contrast **事后选择**的，不具有语言语义，也不能作为无偏 hypothesis test。

~~~powershell
python projects/transformers-basics/activation_patching.py
python -m pytest tests/test_activation_patching.py -q
~~~

固定 CPU fixture 的原始结果为：

| patch site（layer 0 post-residual） | patched metric | recovery | 这条检查说明什么 |
|---|---:|---:|---|
| changed source position 0 | -0.00223266 | 0.432683 | 单位置干预可部分移动固定 metric |
| metric position 1 | 0.00695518 | 0.576049 | 当前位置 residual 也携带差异 |
| joint causal prefix 0+1 | 0.03412487 | 1.0 | 对 metric position 可见的 layer-0 prefix 全部恢复后，后续确定性计算精确复现 clean metric |
| future position 2 | -0.02996198 | 0.0 | 未来位置负对照不能改变过去位置 metric，回归 causal mask/hook site |

其中 (m_{clean}=0.03412487)，(m_{corrupt}=-0.02996198)。实现要求 `model.eval()`、`torch.long`、相同 batch/sequence shape、device、合法 layer/position/token；clean activation 会 detach+clone，hook 在 `finally` 中移除。恢复分数不裁剪，分母小于阈值会失败，而不是输出放大后的伪稳定数字。

这组数字只证明 seeded random model 上的 tensor contract、hook 确实进入 forward、causal negative control 与恢复分数实现。它没有训练语言行为，batch 只有 1，metric 还是 post-hoc 选择；完整 prefix patch 恢复 clean output 在计算图上近乎构造性成立，不能写成“发现了模型的自然 circuit”。对目标 checkpoint 的研究仍需预注册 behavior/metric、独立 clean-corrupt pairs、多个模板与 seed、逐样本分布、无关/随机 source 负对照，并确认 hook 对应真实实现中的 pre/post norm/projection site。

## 15. Path patching 与 circuit

Path patching 尝试只让 source component 对指定 downstream component 的影响被替换，同时阻断其他传播，从而定位组件间的因果路径。Circuit hypothesis 应明确：

- 节点：head、MLP、feature 或 residual stream site；
- 边：通过什么 projection/position 传递；
- 行为与 metric；
- sufficiency：只保留 circuit 是否仍完成行为；
- necessity：移除 circuit 是否破坏行为；
- faithfulness：保留/移除时是否引入不自然副作用。

一个在几十个模板上成立的 circuit 不应直接外推为“整个模型的算法”。

## 16. Superposition 与 Sparse Autoencoder

Superposition 假说认为模型可在有限维 residual space 中以非正交方向表示多于维度数的稀疏特征；这会产生 polysemantic neuron。它是解释某些表示现象的模型，不是所有网络内部都已被完全证明的事实。

Sparse autoencoder（SAE）常学习

\[
f=\operatorname{ReLU}(W_{enc}x+b_{enc}),
\qquad
\hat x=W_{dec}f+b_{dec},
\]

并优化类似

\[
\|x-\hat x\|_2^2+\lambda\|f\|_1.
\]

实现还可能使用 top-k activation、decoder norm 约束、不同 bias/normalization。评价不能只看“特征看起来可命名”：

- reconstruction error / explained variance；
- sparsity、L0/L1 与激活频率；
- dead feature 和高频 feature；
- feature splitting/merging 随 dictionary size 的变化；
- 自动解释在 held-out examples 上的 precision/recall；
- feature intervention 是否因果改变目标行为；
- 与原模型分布相比，重构后的 downstream loss 变化。

SAE feature 不是保证真实、唯一的 ontology。不同 seed、宽度和正则化可学到不同字典。

## 17. 模型编辑

模型编辑试图改变某个事实或行为，同时保持其他能力。方法包括：

- 对少量参数/层做梯度更新或低秩更新；
- 定位 MLP/representation 后做 closed-form 或 learned update；
- 使用 adapter、外部 memory 或 retrieval，而不永久改底座；
- 继续训练或构造 counterfactual data。

至少评价：

| 维度 | 问题 |
| --- | --- |
| Efficacy | 目标 prompt 是否更新 |
| Paraphrase generalization | 改写是否一致 |
| Portability | 推论、相关实体和多语言是否同步 |
| Locality | 无关事实与能力是否保持 |
| Specificity | 相似实体是否被误改 |
| Persistence | 保存/合并/继续训练后是否存在 |
| Conflict handling | 旧知识、检索证据和新编辑冲突时怎样 |

“把一条问答改对”不证明内部知识图谱全局一致。对时效事实，RAG/数据库通常比参数编辑更易审计和回滚。

## 18. 可解释性工具本身也要验证

常见 sanity checks：

- 随机化模型权重后归因图是否仍几乎不变；
- 随机化标签后 probe 是否仍高分；
- 换 baseline、metric 或 prompt 后结论是否稳定；
- hook 是否取到预期的 pre/post norm、pre/post projection 张量；
- KV Cache、tensor parallel 和 quantization 是否改变 hook 位置；
- patch 后 forward 是否真正使用了新 activation；
- 多次运行、batching 与精度是否复现。

框架模块名不等于概念边界。例如一个 `attention` module hook 的输出可能已经过 output projection，也可能只是 per-head value aggregation。必须结合源码和 tensor shape 确认。

## 19. 从 checkpoint 开始的审计流程

1. 固定 model revision、tokenizer、代码版本和 dtype。
2. 读取 config 与权重 shape，画真实计算图和 residual sites。
3. 建立 behavior dataset、连续 metric 和 clean/corrupt pairs。
4. 先做 behavioral baseline，再做 probe/attribution。
5. 用 ablation/patching 检验因果预测，并加入负对照。
6. 对多个模板、语言、位置、seed 和难度报告分布。
7. 明确 intervention、normalization 和失败案例。
8. 不把局部机制结论升级成总体安全证明。

仓库的 `projects/transformers-basics/inspect_checkpoint.py` 用真实 checkpoint config/shape 检查参数契约；六个模型家族章节则明确区分公开架构证据与闭源未知项。仓库已有 seeded random MiniGPT residual patching 的可执行控制实验，但当前仓库还没有对目标大模型执行 activation patching/SAE 实验，因此机制可解释性成熟度仍应标为“理论、实验协议与 hook correctness fixture 已成文，目标模型因果实证待补”。

## 20. 安全与治理边界

内部分析可帮助发现 memorization、router collapse、异常 feature 或拒答回路，但目前不能证明模型在所有输入上真实、安全或无偏。原因包括：

- 分析数据覆盖有限；
- 特征与回路随上下文变化；
- 工具可能漏掉分布式/量化执行中的行为；
- 人类命名会过度简化 feature；
- 已解释组件之外仍有冗余路径；
- 系统风险还来自 prompt、RAG、工具、权限和运行时。

高风险部署仍需要行为评测、red teaming、访问控制、人工审批、监控和事件响应。

## 21. 常见错误结论

- **“Encoder-only 完全不能生成”**：标准协议不直接做左到右开放生成，但可被接入其他生成机制。
- **“SSM 是线性复杂度，所以一定比 attention 快”**：真实 crossover 依 kernel、状态、长度和硬件。
- **“MoE 有 100B 参数，所以每 token 做了 100B dense compute”**：总参数、激活参数和通信是不同口径。
- **“Probe 能读出信息，所以模型使用了它”**：可解码性不是因果使用证据。
- **“Attention weight 就是 token 重要性”**：value、projection、residual 与后续层均未被计入。
- **“Activation patching 找到唯一负责层”**：冗余、纠缠和 OOD intervention 都限制结论。
- **“SAE feature 有名字，所以它是单一真实概念”**：字典依训练设置，自动命名也可能错。

## 自测与实践

1. 分别画 encoder-only、decoder-only 和 encoder-decoder 的 attention 可见性矩阵。
2. 为什么同 FLOPs 比较和同参数量比较会给架构排名不同的答案？
3. 设计一个 probe split，避免同一词项和模板跨 train/test 泄漏。
4. 给出 activation patching 恢复分数分母接近零时的失败例子。
5. 为一个 SAE 报告设计同时覆盖重构、稀疏、可解释与因果性的指标表。
6. 选择一个公开小模型，先固定 clean/corrupt 行为数据；在真正做 patching 前，写出 hook tensor shape、metric、负对照和证据边界。
