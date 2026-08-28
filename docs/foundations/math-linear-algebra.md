# 线性代数：沿着 shape 看懂 LLM

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已经走完[一次模型预测](math.md)，还会被 shape、转置或广播卡住的读者。
- **先修**：知道向量和矩阵是数值数组，会手算一次 <code>[1,2] @ [2,3]</code>。
- **首次阅读**：读到 Attention shape 即可回主线；SVD、LoRA 和 conditioning 都是选读。
- **完成信号**：看到一行张量公式时，能标出每条轴的语义，并在运行前判断输出 shape。
- **卡住时**：回到[数学基础主线](math.md#objects)，先只区分向量、矩阵和 shape。

</div>

线性代数在 LLM 中主要回答三个问题：

1. 一组数字怎样重新组合成另一组数字？
2. 哪些轴参与计算，哪些轴只是并行保留？
3. 两个向量在当前度量下有多相似？

本页不从抽象向量空间开始，而是一直追踪 token 表示的 shape。

## 1. Shape 不只是尺寸，还包含语义

假设一批输入的 [hidden states](../reference/glossary.md#term-hidden-state) 为：

\[
X\in\mathbb R^{B\times T\times D}.
\]

- \(B\)：batch，一次处理的序列数；
- \(T\)：每条序列的 token 数；
- \(D\)：每个 token 的特征数。

若 \(X\) 的 shape 是 <code>[2,3,4]</code>，它共有 \(2\times3\times4=24\) 个数字。
但只知道 24 还不够：<code>[2,3,4]</code> 与 <code>[3,2,4]</code> 的元素数相同，batch 和 token 的含义却交换了。

调试[张量](../reference/glossary.md#term-tensor)时，建议同时写四件事：

| 量 | shape | 轴的含义 | 备注 |
|---|---|---|---|
| input IDs | <code>[B,T]</code> | batch, token | 整数索引 |
| hidden states | <code>[B,T,D]</code> | batch, token, feature | 浮点中间量 |
| projection weight | <code>[D,H]</code> | input feature, output feature | 可训练参数 |
| projection output | <code>[B,T,H]</code> | batch, token, new feature | 浮点中间量 |

Shape 能对上只说明计算在尺寸上合法，不保证轴的含义正确。

## 2. [矩阵](../reference/glossary.md#term-matrix)乘法是在做许多次[点积](../reference/glossary.md#term-dot-product)

对于：

\[
X=
\begin{bmatrix}
x_{11}&x_{12}\\
x_{21}&x_{22}
\end{bmatrix},
\qquad
W=
\begin{bmatrix}
w_{11}&w_{12}&w_{13}\\
w_{21}&w_{22}&w_{23}
\end{bmatrix},
\]

\(XW\) 的左上角元素是第一行与第一列的点积：

\[
(XW)_{11}=x_{11}w_{11}+x_{12}w_{21}.
\]

其余元素只是换一行或换一列，重复同样的逐项乘加。Shape 规则：

\[
[M,K]@[K,N]\rightarrow[M,N].
\]

中间的 \(K\) 表示每次点积要配对多少个数字，因此必须相等；\(M,N\) 表示要做多少行、多少列这样的点积。

### 带 batch 和 token 轴时

\[
[B,T,D]@[D,H]\rightarrow[B,T,H].
\]

矩阵 \(W\) 对每个 batch 中的每个 token 使用同一个映射。计算只收缩 \(D\) 轴，\(B,T\) 原样保留。
参数量是 \(D\times H\)，不会因为 batch 或序列变长而增加；中间 activation 和计算量会增加。

??? note "自测：<code>[4,8,16] @ [16,32]</code> 的输出 shape 是什么？"

    是 <code>[4,8,32]</code>。16 被收缩，batch 4 和 token 8 保留，最后一个特征轴变成 32。

## 3. [转置](../reference/glossary.md#term-transpose)：交换轴，不是改变数字

二维矩阵转置会交换行和列：

\[
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}^{T}
=
\begin{bmatrix}
1&4\\
2&5\\
3&6
\end{bmatrix}.
\]

原 shape 是 <code>[2,3]</code>，转置后是 <code>[3,2]</code>。数字没有被重新计算，只是索引方式改变。

Attention 需要让每个 query 与每个 key 做点积。若 \(Q,K\) 都是 <code>[T,D]</code>：

\[
QK^T:\quad[T,D]@[D,T]\rightarrow[T,T].
\]

结果的第 \(i,j\) 项表示第 \(i\) 个 query 与第 \(j\) 个 key 的分数。

在多维张量代码中，“转置”必须说明交换哪两条轴。把 <code>[B,T,H,D]</code> 变成
<code>[B,H,T,D]</code> 是交换 token 轴与 head 轴，不是把整个四维张量简单倒序。

## 4. Reshape 和 transpose 解决不同问题

多头 Attention 常把 hidden dimension 拆成 head 数与每头维度：

~~~text
[B,T,D_model]
→ reshape  [B,T,H,D_head]
→ transpose [B,H,T,D_head]
~~~

其中 \(D_{\text{model}}=H\times D_{\text{head}}\)。

- reshape 改变“怎样给同一批数字分组”，元素总数必须不变；
- transpose 改变轴的顺序；
- contiguous 描述底层存储是否与当前轴顺序连续。

数学 shape 一样，内存 layout 仍可能不同。某些 kernel 要求连续输入，框架可能插入复制；这会影响性能，
但不改变纸面公式的结果。

## 5. Broadcasting：看似没复制，逻辑上重复使用

广播（broadcasting）让较小张量在 size 为 1 或缺失的轴上重复使用。

最常见的例子是给 <code>[B,T,H]</code> 加 bias \(b\in\mathbb R^H\)。逻辑上，同一组 \(H\) 个 bias
被加到每个 batch、每个 token 上：

~~~text
[B,T,H]
    [H]
---------
[B,T,H]
~~~

框架通常不会真的复制 \(B\times T\) 份 bias。

危险在于：尺寸规则允许，不代表语义正确。

- <code>[B,T] + [T]</code>：最后一轴对齐，可能是给每个位置加 bias；
- <code>[B,T,1] + [B,T]</code>：从右对齐后可能变成 <code>[B,T,T]</code>；
- attention mask <code>[T,T]</code> 可以扩到 <code>[B,H,T,T]</code>，但 padding mask 还需要 batch 信息。

遇到意外的大张量，不要只看最后结果；把两个输入从右向左对齐，逐轴检查“相等、其中一个为 1、或缺失”。

??? note "自测：<code>[2,3,4] + [4]</code> 的输出 shape 是什么？"

    是 <code>[2,3,4]</code>。长度为 4 的向量沿 batch 和 token 两轴重复使用。

## 6. 点积：把两组特征合成一个分数

两个等长向量的点积：

\[
x^Ty=\sum_i x_iy_i.
\]

例如：

\[
[1,2]\cdot[3,4]=1\times3+2\times4=11.
\]

它在 LLM 中有多种角色：

- Attention 中，query 与 key 的点积形成相关性分数；
- LM head 中，hidden state 与某个 token 的输出向量点积形成 logit；
- 向量检索中，query embedding 与文档 embedding 的点积可以用于排序。

点积同时受方向和长度影响。把一个向量放大 10 倍，点积也会放大 10 倍。

## 7. 范数与余弦：长度和方向分开看

L2 范数可以理解成向量长度：

\[
\lVert x\rVert_2=\sqrt{\sum_i x_i^2}.
\]

例如 \([3,4]\) 的长度是 \(\sqrt{3^2+4^2}=5\)。

余弦相似度把点积除以双方长度：

\[
\cos(x,y)=\frac{x^Ty}{\lVert x\rVert_2\lVert y\rVert_2}.
\]

它主要比较方向。若 \(y=10x\)，二者方向相同，余弦仍为 1；点积却随长度增大。

常见范数回答不同问题：

- L1：\(\lVert x\rVert_1=\sum_i|x_i|\)；
- L2：常用于距离和整体 gradient norm；
- L∞：\(\lVert x\rVert_\infty=\max_i|x_i|\)，只看最大分量；
- Frobenius norm：把矩阵所有元素视为一组数后求 L2。

向量检索中要确认 embedding 是否已经 L2 normalization。归一化后，dot product 与 cosine 的排序相同；
未归一化时，向量长度可能改变排名。零向量的余弦相似度没有定义。

## 8. Attention 的完整 shape

令输入为 \(X\in\mathbb R^{B\times T\times D_{\text{model}}}\)，三个投影先产生 Q/K/V，
再拆成 \(H\) 个 head：

~~~text
Q, K, V: [B,H,T,D_head]
K 转置:  [B,H,D_head,T]
score:   [B,H,T,T]
output:  [B,H,T,D_head]
~~~

逐步检查：

1. <code>[T,D_head] @ [D_head,T] → [T,T]</code>，每个 token 给每个位置打分；
2. mask 和 softmax 不改变 score shape；
3. <code>[T,T] @ [T,D_head] → [T,D_head]</code>，每个 query 混合所有 value；
4. 所有 head 拼回 <code>[B,T,D_model]</code>。

为什么 score 除以 \(\sqrt{D_{\text{head}}}\)？先采用一个便于分析初始化的假设：Q/K 各分量大致独立，
均值为 0，方差约为 1。

在这个假设下，点积方差会随 \(D_{\text{head}}\) 增长。除以平方根可以稳定初始分数的尺度，减少 softmax
过早饱和。这个推导提供的是初始化直觉；训练后的分量无需继续满足独立假设。

??? note "自测：Q/K 为 <code>[B,H,T,64]</code> 时，单个 head 的 score shape 是什么？"

    是 <code>[T,T]</code>；保留 batch 与 head 后是 <code>[B,H,T,T]</code>。

## 9. 选读：SVD 与 LoRA 为什么都提到“低秩”

!!! info "本节不阻塞主线"

    只有在学习 LoRA、压缩或矩阵谱时再读。第一次理解 Transformer 可以跳过。

奇异值分解（SVD）把矩阵写成：

\[
W=U\Sigma V^T.
\]

可以把它想成三步：先转到一组特殊方向，按奇异值放大或缩小，再转到输出方向。若只有少数奇异值很大，
矩阵的主要作用集中在少数方向。保留前 \(r\) 个奇异值，可以得到经典范数意义下的最佳 rank-\(r\) 近似。

LoRA 不是直接把原矩阵做 SVD 截断。它保留 \(W_0\)，学习一个低秩增量：

\[
W'=W_0+\frac{\alpha}{r}BA,
\quad
A\in\mathbb R^{r\times d_{\text{in}}},
\quad
B\in\mathbb R^{d_{\text{out}}\times r}.
\]

可训练参数从 \(d_{\text{in}}d_{\text{out}}\) 变为
\(r(d_{\text{in}}+d_{\text{out}})\)。低秩是一种容量约束与归纳偏置，不保证所有任务都适合同一个 rank。

## 10. 选读：Conditioning 为什么影响优化难度

!!! info "本节不阻塞主线"

    只有在 loss 震荡、不同方向学习速度差异很大时再深入。

一个矩阵若把某个方向放大很多、另一个方向压缩很多，就会让优化地形像狭长山谷：同一 [learning rate](../reference/glossary.md#term-learning-rate)
在陡峭方向可能太大，在平缓方向又太小。Condition number（条件数）概括最大与最小尺度的差异。

Normalization、初始化、自适应 optimizer 和 preconditioning 会从不同层面改善尺度问题，
但它们不会把非凸训练变成一个总能轻松求解的问题。

## 常见误解

- “Shape 能相乘，公式就一定对”：尺寸合法不代表轴语义正确。
- “Transpose 和 reshape 都只是换 shape”：前者交换轴，后者重新分组。
- “Broadcast 不分配内存，所以没有风险”：意外输出仍可能变成巨大的逻辑 shape。
- “Dot product 就是 cosine”：未归一化时，点积还受向量长度影响。
- “LoRA 就是对原权重做 SVD”：LoRA 学的是低秩增量，两者不是同一操作。

## 换一种讲法

若你希望配合图示和可运行张量练习，可以阅读《动手学深度学习》的
[线性代数章节](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html)。
它提供另一套讲解与代码；本页仍是继续阅读本仓库所需内容的自包含入口。

下一步：回到[数学主线](math.md#two-token-attention)手算 Attention，或进入
[Transformer](../core/transformer.md)观察这些 shape 怎样贯穿完整层。
