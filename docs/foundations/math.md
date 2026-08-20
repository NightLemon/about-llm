# 数学基础：从张量形状到可信实验

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：需要读懂 shape、概率、梯度和评测区间的工程师。
- **先修**：Python 基础；能理解变量、函数、指数和对数记号。
- **首次阅读**：先读 Shape、矩阵乘法、概率与 softmax；模型比较转到[评测统计](evaluation-statistics.md)。
- **完成信号**：能标注一次 attention 的 shape，并解释一个数值公式的成立前提。
- **卡住时**：先看[新手知识地图](../guide/beginner-map.md)，只回补当前公式需要的小节。

</div>

## 怎样使用本章

读 Attention 公式时，你可能同时卡在三个地方：`QKᵀ` 为什么能相乘，softmax 为什么放在这一轴，
以及代码里的 tensor shape 怎样对应纸面符号。本章就是为这类问题准备的数学工具箱。

不必先学完再去读模型章节。遇到 shape 问题就看前两节，遇到 loss 与 PPL 就看概率和信息论，
开始训练时再读梯度与优化。需要比较两个系统的区间和显著性时，转到[评测统计](evaluation-statistics.md)。

学完后，你应该能用小数字复算一次 Attention，追踪 loss 怎样传回参数，并说出公式依赖哪些假设。
仓库里的 NumPy Attention、PyTorch/JAX tiny GPT 和 KV Cache 计算器提供了可运行例子；大模型训练稳定性和
目标硬件性能仍要在相应环境里验证。

## 1. 先用 Shape 排除不可能

### 标量、向量、矩阵、张量

批量 token hidden states 常写为：

\[
X\in\mathbb R^{B\times T\times d}
\]

- \(B\)：batch；
- \(T\)：sequence length；
- \(d\)：hidden dimension。

线性投影 \(W\in\mathbb R^{d\times h}\)：

\[
Y=XW\in\mathbb R^{B\times T\times h}
\]

这次矩阵乘法收缩的是最后一个 \(d\) 维，batch 和 time 两个轴原样保留。权重参数量是 \(dh\)，
所以不会随 \(B,T\) 改变；中间 activation 和计算量会随着 batch 与序列变大。

遇到新的张量公式时，可以先写下面四列。它往往比盯着字母更快暴露问题：

| 量 | shape | dtype/device | 是否训练 |
|---|---|---|---|
| input ids | `[B,T]` | integer | 否 |
| embedding output | `[B,T,d]` | float | 中间量 |
| projection weight | `[d,h]` | float | 是 |
| output | `[B,T,h]` | float | 中间量 |

Shape 对得上只是第一关。把 Q/K 轴弄反后，矩阵乘法可能仍然合法，但每个分数所表示的 query-key 关系已经变了。

### Broadcasting

广播把 size 1 或缺失轴视为可重复。例如 bias \(b\in\mathbb R^h\) 加到 `[B,T,h]`，逻辑上在 B/T 维重复，但通常不物化副本。

危险案例：

- `[B,T] + [T]` 按最后维广播，可能是想要的 position bias；
- `[B,T,1] + [B,T]` 会广播成 `[B,T,T]`，常是灾难；
- mask `[T,T]` 可广播到 `[B,H,T,T]`，但 padding mask 还需 batch 维。

调试时不仅打印最终 shape，还断言每个语义轴。命名轴/Einsum 记法能降低“维度相等但语义不同”的错误。

### Reshape、transpose 与 contiguous

多头 attention 常做：

```text
[B,T,d] -> [B,T,H,D] -> [B,H,T,D]
```

其中 \(d=HD\)。`reshape` 改观察方式，`transpose` 改 strides/轴顺序；某些 kernel 需要 contiguous layout，会触发复制。数学 shape 相同不代表内存布局和性能相同。

## 2. 线性代数直觉

### 线性映射与基

矩阵 \(W\) 把输入坐标映到输出空间。神经网络线性层不只是“乘一个表”，它学习哪些输入方向要放大、衰减或组合。Bias 允许仿射平移。

Embedding lookup 等价于用 one-hot 向量选矩阵的一行，但实现不会真的构造巨大 one-hot。Tied embedding/lm head 共享参数：输入表 \(E\in\mathbb R^{V\times d}\)，输出 logits 常为 \(XE^T\)。

### 点积、范数与角度

点积：

\[
x^Ty=\lVert x\rVert_2\lVert y\rVert_2\cos\theta
\]

它同时受方向和模长影响。余弦相似度：

\[
\cos(x,y)=\frac{x^Ty}{\lVert x\rVert_2\lVert y\rVert_2}
\]

只比较方向，但零向量无定义。向量检索要核对 embedding 模型是否要求 L2 normalization；归一化后 dot product 等于 cosine，未归一化时二者排名可能不同。

常见范数：

- \(\lVert x\rVert_1=\sum_i|x_i|\)：稀疏性/绝对大小；
- \(\lVert x\rVert_2=(\sum_i x_i^2)^{1/2}\)：距离、gradient norm；
- \(\lVert x\rVert_\infty=\max_i|x_i|\)：最大分量；
- Frobenius norm：矩阵所有元素的 L2。

不同 norm 回答不同问题。参数 L2 小不等于模型输出变化小，输出还受输入、层间放大和非线性影响。

### Rank、SVD 与低秩更新

矩阵 SVD：

\[
W=U\Sigma V^T
\]

奇异值描述不同输入方向的放大程度。保留前 \(r\) 个奇异值给出 Frobenius/L2 意义下的最佳 rank-\(r\) 近似（在相应经典条件下）。

LoRA 不等于直接对原权重做 SVD 截断，而是学习低秩**增量**：

\[
W'=W_0+\frac{\alpha}{r}BA,
\quad A\in\mathbb R^{r\times d_{in}},
\quad B\in\mathbb R^{d_{out}\times r}
\]

可训练参数从 \(d_{in}d_{out}\) 变为 \(r(d_{in}+d_{out})\)。低 rank 是容量约束与归纳偏置，不保证所有任务都能用同一 rank 表达。

### Conditioning

若一个方向被矩阵极大放大、另一个极小，优化会呈狭长谷地；condition number 大时，统一学习率难以兼顾各方向。Normalization、初始化、自适应 optimizer 和 preconditioning 都在不同层面改善尺度，但不消除非凸性。

## 3. Attention 的矩阵推导

令：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

单头 scaled dot-product attention：

\[
S=\frac{QK^T}{\sqrt{d_h}}+M,
\quad A=\operatorname{softmax}(S),
\quad O=AV
\]

多头 shape：

```text
Q,K,V: [B,H,T,D]
K^T:   [B,H,D,T]
S,A:   [B,H,T,T]
O:     [B,H,T,D]
```

### 为什么除以 \(\sqrt{d_h}\)

若 Q/K 各分量近似独立、均值 0、方差约 1，点积 \(q^Tk\) 的方差约为 \(d_h\)。除以 \(\sqrt{d_h}\) 让 score 方差保持 \(O(1)\)，避免 softmax 随维度增大过度饱和。

这些独立/同方差是假设，不是训练后严格事实；缩放仍是稳定初始化和优化的有用设计。

### Mask 不是乘 0

Causal mask 应在 softmax 前把未来 score 设为足够负的值，使概率为 0。若 softmax 后再乘 0，行和不再为 1；若 mask 值在低精度下不够负，仍可能泄漏。

Padding mask、causal mask、packing block mask 和 loss mask 目的不同：前三者控制信息能否被读取，最后一个控制该位置是否贡献训练目标。

### 复杂度与存储 { #attention-storage-online-softmax }

朴素 score 矩阵有 \(O(T^2)\) 元素，QK/AV 计算也含二次项。FlashAttention 通过 tiled IO 与 online softmax 避免物化完整 score/probability，降低 HBM traffic 和内存；它没有普遍把精确 dense attention 的算术复杂度变成线性。

Online softmax 的关键是旧 block 的统计量可以在全局最大值变化后重标定。
对一个 query row，把第 \(b\) 个 key block 的 scaled、masked scores 写成 \(s_{b,j}\)。
维护 running maximum \(m_b\)、normalizer \(\ell_b\) 与未归一化 value accumulator \(o_b\)。初值为 \(m_0=-\infty,\ell_0=0,o_0=0\)：

\[
m_b=\max\left(m_{b-1},\max_j s_{b,j}\right),
\]

\[
\ell_b=e^{m_{b-1}-m_b}\ell_{b-1}+\sum_j e^{s_{b,j}-m_b},
\]

\[
o_b=e^{m_{b-1}-m_b}o_{b-1}+\sum_j e^{s_{b,j}-m_b}v_{b,j}.
\]

最后输出 \(o_B/\ell_B\)。第一项把旧 block 的分子和分母缩放到新 maximum 坐标系；第二项加入当前 block。
对不可见位置令 \(s=-\infty\)。
若此前还没有任何可见 key，旧 \(\ell/o\) 本来就是 0，实现应直接把旧贡献定义为 0，不能真的计算 \(-\infty-(-\infty)\)。
这样前几个 block 全被 mask、后续才出现可见 key 也不会产生 NaN。
若处理完全部 blocks 后整行仍不可见，则 \(\ell_B=0\)，实现必须拒绝。

在实数算术下，这和一次性计算 dense softmax 完全等价；有限精度下，block 划分与归约顺序可能造成微小误差，不能要求逐 bit 相同。
仓库的 blockwise online attention 使用 float64 累积，只构造当前 score tile 与每行状态，不返回完整 probability matrix。
报告的 logical peak score elements 只是最大逻辑 tile，不包含 Q/K/V、输出、NumPy temporary 或 allocator，因此不是进程峰值内存测量。
这个 CPU oracle 也不证明 CUDA kernel、FlashAttention backend、HBM traffic、速度或 vLLM 行为。

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
python -m pytest tests/test_attention_numpy.py -q
~~~

## 4. 概率：模型输出究竟是什么

### 随机变量与条件概率

离散分布满足 \(p(x)\ge0\)、\(\sum_xp(x)=1\)。条件概率：

\[
p(x\mid y)=\frac{p(x,y)}{p(y)}
\]

链式法则总成立：

\[
p(x_{1:T})=\prod_{t=1}^Tp(x_t\mid x_{<t})
\]

这不是 token 独立假设；每个条件分布都依赖前缀。

### 似然与概率不要混用

给定参数 \(\theta\)，数据概率是 \(p_\theta(D)\)；把同一表达式视为 \(\theta\) 的函数时叫 likelihood。最大似然选择：

\[
\hat\theta=\arg\max_\theta\sum_i\log p_\theta(x_i)
\]

取 log 把乘积变求和，也避免浮点下溢。Maximum likelihood 不保证因果解释、校准或分布外可靠。

### Token 概率不是事实置信度

`p(next_token="巴黎" | prompt)` 是生成分布下该 token 的概率，受到措辞、tokenizer、采样和训练分布影响。它不等于“巴黎是正确答案”的 Bayesian posterior。一个事实可能有多种 tokenization/表达；模型也可能对错误模板非常自信。

如果产品需要置信度，应定义事件和标签，在目标分布做 calibration、selective prediction/abstention 和切片评测，而不是直接展示首 token probability。

### Bayes 与 base rate

\[
p(H\mid E)=\frac{p(E\mid H)p(H)}{p(E)}
\]

低 base-rate 事件即使检测器 sensitivity 较高，阳性中仍可能有大量 false positive。安全分类、异常检测和成员推断都必须报告 precision/recall 与真实 prevalence，不能只报 accuracy。

## 5. Softmax、LogSumExp 与数值稳定

Softmax：

\[
p_i=\frac{e^{z_i}}{\sum_je^{z_j}}
\]

对任意常数 \(c\)，\(\operatorname{softmax}(z)=\operatorname{softmax}(z-c)\)。取 \(c=\max z\) 可避免 `exp` overflow：

\[
\log\sum_je^{z_j}=m+\log\sum_je^{z_j-m},\quad m=\max_jz_j
\]

Cross entropy 实现应使用 fused `log_softmax`/cross-entropy，而不是先算概率再 `log`，后者容易把极小概率舍入为 0。

Softmax Jacobian：

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
\]

对 one-hot label \(y\)，softmax + cross entropy 对 logits 的梯度简化为：

\[
\frac{\partial L}{\partial z}=p-y
\]

这给出直觉：正确类概率不足时得到负梯度提高 logit，错误类按当前概率被压低。

## 6. 信息论：Cross Entropy、KL 与 PPL

熵：

\[
H(p)=-\sum_xp(x)\log p(x)
\]

Cross entropy：

\[
H(p,q)=-\sum_xp(x)\log q(x)
\]

KL divergence：

\[
D_{KL}(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)}
\]

满足：

\[
H(p,q)=H(p)+D_{KL}(p\|q)
\]

当数据分布 \(p\) 固定，最小化 cross entropy 等价于减小 forward KL \(D_{KL}(p\|q)\)。KL 非负但不对称，也不满足三角不等式，不是距离。

### KL 方向的直觉

- \(D_{KL}(p\|q)\)：若 \(p(x)>0\) 而 \(q(x)\) 很小，惩罚很大，倾向覆盖 data modes；
- \(D_{KL}(q\|p)\)：对 q 放在 p 低密度区惩罚大，某些近似设置中呈 mode-seeking。

这是理想分布直觉；神经网络训练、有限样本和参数约束会改变实际行为。

### 困惑度

若平均 NLL 以自然 log 计：

\[
PPL=\exp(\overline{NLL})
\]

可直觉理解为平均“有效候选数”，但只在相同数据、tokenizer、normalization 和 mask 口径下比较。一个 tokenizer 把中文切得更细，会改变 token NLL 与 PPL，不能据此直接判断语言能力。

Bits-per-byte/character 可改善跨 tokenizer 比较，但仍要统一数据编码和归一化。

## 7. 微积分、Jacobian 与反向传播

### 导数是局部线性近似

对小扰动 \(\Delta x\)：

\[
f(x+\Delta x)\approx f(x)+J_f(x)\Delta x
\]

标量对向量梯度 \(\nabla_xL\) 指向局部上升最快方向。高维网络不显式构造巨大 Jacobian；reverse-mode autodiff 计算 vector-Jacobian product（VJP），对“多参数 → 一个标量 loss”特别高效。

### 链式法则

若 \(x\to y\to L\)：

\[
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
\]

反向传播是把上游 cotangent 沿计算图应用局部 VJP。它不是另一种学习规则；optimizer 才决定怎样用梯度更新参数。

### 线性层的梯度

对 \(Y=XW\)，上游梯度 \(G=\partial L/\partial Y\)：

\[
\frac{\partial L}{\partial X}=GW^T,
\quad
\frac{\partial L}{\partial W}=X^TG
\]

带 batch/time 时先把这些轴视为样本维做 contraction。这也解释为什么 weight gradient 的计算量与 forward 同量级。

### Gradient accumulation

若 global objective 是 \(A\) 个 micro-batch loss 的平均：

\[
\nabla\left(\frac1A\sum_{a=1}^AL_a\right)
=\frac1A\sum_{a=1}^A\nabla L_a
\]

每次 backward 若未除以 \(A\)，累积梯度是 sum，等价学习率会放大 \(A\) 倍。框架可能在 loss、distributed reducer 或 optimizer 中缩放，必须用单卡大 batch 对照验证。

变长序列时，应按有效 token 对 loss numerator/denominator 聚合，而不是对 micro-batch mean 等权平均。

### Gradient check

小模型可用有限差分：

\[
\frac{\partial L}{\partial\theta_i}\approx
\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}
\]

\(\epsilon\) 太大会有截断误差，太小会有浮点消减。Gradient check 适合少量参数和 FP64/FP32 debug，不适合大模型全量验证。

## 8. Optimization：梯度不等于更新

### SGD 与 Momentum

SGD：

\[
\theta_{t+1}=\theta_t-\eta_tg_t
\]

Mini-batch gradient 是随机估计，batch 大小影响方差和每 token 更新频率。Momentum 对梯度做指数平滑，减少狭长谷地中的来回震荡，但会引入 state 和超参数。

### Adam

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

偏差修正：

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\quad
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

Adam 按历史二阶矩自适应缩放方向。\(\epsilon\) 不只是防除 0，在低方差参数上也影响有效步长；放在平方根内/外是不同算法实现细节。

### AdamW

简化更新：

\[
\theta_{t+1}=\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta_t\lambda\theta_t
\]

Decoupled weight decay 不经过 Adam 的二阶缩放。对 SGD，L2 regularization 与 weight decay 在简单条件下可等价；对自适应 optimizer 通常不能混为一谈。

Norm scale、bias-like 参数常排除 decay，但这是参数 mask 配置，不是 AdamW 数学自动知道。

### Warmup 与 schedule

Warmup 降低初始化早期的大步风险；cosine/linear decay 控制后期更新。Schedule 的横轴应明确是 optimizer step 还是 consumed token；gradient accumulation 或 batch 变化会让二者关系改变。

### Gradient clipping

全局 norm clipping：

\[
g'=g\min\left(1,\frac{c}{\lVert g\rVert_2+\epsilon}\right)
\]

它保留全局方向、缩小长度。Per-value clipping 会逐元素截断，方向改变更大。记录 pre-clip norm 和触发率；clip 不是修复 NaN、错误数据或过大学习率的万能工具。

## 9. 数值精度与误差

浮点数只有有限指数和尾数。常见风险：

- overflow：`exp(large)`、低精度累加；
- underflow：极小概率/gradient 变 0；
- catastrophic cancellation：相近大数相减；
- non-associativity：\((a+b)+c\neq a+(b+c)\)；
- reduction order：多卡 collective 顺序改变末位。

BF16 指数范围接近 FP32、尾数更短；FP16 指数范围更小，训练常用 loss scaling；FP8 依赖格式和动态 scale。Mixed precision 必须逐项声明参数、matmul、accumulation、gradient、optimizer 和 collective dtype。

### Stable variance

直接用 \(E[x^2]-E[x]^2\) 计算低方差大数可能严重消减；在线算法/Welford 更稳定。Normalization 和统计指标实现应使用框架的稳定 primitive，不要只照搬代数式。

### 容差

`allclose` 需要绝对与相对容差：

\[
|a-b|\le atol+rtol|b|
\]

近 0 值由 `atol` 主导，大值由 `rtol` 主导。容差应由 dtype、运算长度和业务影响决定，不能为让测试绿而无限放宽。

## 10. 可执行小实验

### Attention 因果性

本仓库 NumPy/PyTorch/JAX 测试使用两条只在未来位置不同的序列，断言过去位置 logits 相同。这比“看代码里有 tril mask”更强，因为它验证 observable invariant。

### KV Cache 容量

理想化 dense K/V：

\[
M=2\times L\times B\times T\times H_{kv}\times d_h\times bytes(dtype)
\]

`2` 表示 K 和 V。它不含 allocator、block metadata、fragmentation、workspace，也不适用于 MLA 等不同 cache layout。本仓库测试验证特定配置精确为 1 GiB，只证明公式实现。

### Tiny-batch overfit

JAX/Optax 实验固定 632 参数模型和 tiny batch，验证 loss 大幅下降、参数变化、gradient norm 有限、JIT 同步计时。它能发现训练闭环错误，不证明 validation 泛化。

## 常见错误

- Shape 能乘就认为语义正确，忽略轴含义；
- 把广播意外扩成 `[B,T,T]`；
- 认为 dot product 就是 cosine；
- 把 token probability 当事实为真的概率；
- 先 softmax 再 log，制造 underflow；
- 用不同 tokenizer 的 PPL 排名模型；
- gradient accumulation 忘记平均或有效 token weighting；
- 把 AdamW 说成“Adam + 在 loss 加 L2”而不讲条件；
- 进行模型比较却没有先读[评测统计](evaluation-statistics.md)定义采样单位与 estimand；
- 用理想 FLOPs/bytes 公式冒充硬件实测。

## 面试追问

1. `[B,T,H,D] @ [B,H,D,T]` 为什么不能直接相乘，先要转哪个轴？
2. \(1/\sqrt{d_h}\) 缩放的方差直觉和假设是什么？
3. 为什么 causal mask 与 loss mask 不能互相替代？
4. Softmax + cross entropy 为什么得到 \(p-y\) 梯度？
5. Forward KL 与 reverse KL 的直觉差别是什么？
6. AdamW 为什么不等同于任意 Adam + L2 loss？
7. 可变长度 micro-batch 怎样得到正确 global token mean？
8. 如何证明一个数值优化没有改变模型语义，而不只是最终文本相似？

## 一手资料与继续学习

- Goodfellow、Bengio、Courville，《Deep Learning》：线性代数、概率、数值计算与优化。
- Murphy，《Probabilistic Machine Learning》：概率模型、估计与不确定性。
- Boyd、Vandenberghe，《Convex Optimization》：凸性、对偶与优化直觉；神经网络虽非凸，基础工具仍重要。
- JAX/PyTorch autodiff、numerical accuracy 与 distributed 官方文档；实现行为以固定版本为准。
- 本仓库的 attention、PyTorch/JAX GPT、KV Cache 实现与对应测试。
- 模型比较、置信区间与 judge 校准继续读[评测统计](evaluation-statistics.md)。
