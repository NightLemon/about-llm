# 概率与信息论：模型的概率究竟在说什么

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已经理解 logit、softmax 和单个 token NLL，想继续读语言模型目标与评测的读者。
- **先修**：[数学主线](math.md)中的三候选预测；不要求学过概率论。
- **首次阅读**：条件概率 → 链式分解 → NLL → PPL。Bayes、交叉熵和 KL 可以按需阅读。
- **完成信号**：能说清“概率是在什么条件下的”，并知道何时不能直接比较两个 PPL。
- **卡住时**：回到[数学基础主线](math.md#softmax)，只重算三个候选的 softmax 和 NLL。

</div>

模型读到“天空通常是”后给“蓝”0.2447 的概率。这个数字不是脱离上下文的“蓝有多真”，而是：

\[
p(\text{下一个 token 是“蓝”}\mid\text{前缀是“天空通常是”}).
\]

概率章节最重要的习惯，就是每次都问：候选空间是什么？已经给定了哪些条件？概率来自哪个模型或数据过程？

## 1. 概率分布是一组受约束的数字

对有限候选集合，概率必须满足：

\[
p(x)\ge0,\qquad\sum_xp(x)=1.
\]

例如三个 token 的分布 \([0.6652,0.2447,0.0900]\)：

- 每项都不小于 0；
- 三项相加约等于 1；
- 它描述的是一次选择中三个互斥候选的相对可能性。

“约等于”来自小数截断。程序计算时仍应在合理浮点误差内检查总和。

### 随机变量不等于“完全随机”

随机变量只是把可能结果映射成数值或类别。即使模型每次都用 greedy 选择最大概率 token，
它内部仍然产生一个概率分布；只是解码规则没有从中随机采样。

## 2. 条件概率：竖线右边是已知信息

\[
p(x\mid y)
\]

读作“在 \(y\) 已知的条件下，\(x\) 的概率”。定义为：

\[
p(x\mid y)=\frac{p(x,y)}{p(y)},\qquad p(y)>0.
\]

一个日常例子：某班 100 人中，40 人会 Python；其中 10 人也会 JAX。那么：

\[
p(\text{会 JAX}\mid\text{会 Python})=\frac{10}{40}=0.25.
\]

分母不再是全班 100 人，而是已经满足条件的 40 人。交换条件通常会得到不同结果：

\[
p(\text{会 Python}\mid\text{会 JAX})
\neq
p(\text{会 JAX}\mid\text{会 Python}).
\]

在语言模型中，改变 prompt、chat template、已生成 token 或工具返回值，就改变了条件，后续概率也会改变。

??? note "自测：80 个请求中有 20 个来自移动端，其中 5 个超时。移动端请求的超时概率是多少？"

    \(p(\text{超时}\mid\text{移动端})=5/20=0.25\)，不是 \(5/80\)。

## 3. 链式分解：整段文本怎样变成逐 token 预测

对三个 token \(x_1,x_2,x_3\)，联合概率总能写成：

\[
p(x_1,x_2,x_3)
=p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_1,x_2).
\]

推广到长度 \(T\)：

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

这就是自回归语言模型的训练与生成结构：每一步只预测下一个 token，整段序列概率是所有条件概率的乘积。
它没有假设 token 相互独立；恰恰相反，每一项都依赖前缀。

### 为什么训练时用 log

许多小概率相乘会很快小到浮点数无法表示。取对数后：

\[
\log p(x_{1:T})
=\sum_{t=1}^{T}\log p(x_t\mid x_{<t}).
\]

乘积变成求和，更容易计算和聚合。因为 log 是单调函数，最大化 log probability 与最大化原 probability
会选出同一个参数解。

## 4. Stable softmax：为什么先减最大 logit

Softmax 为：

\[
p_i=\frac{e^{z_i}}{\sum_je^{z_j}}.
\]

数学上，对所有 logits 减同一个常数不会改变概率。程序取 \(m=\max_jz_j\)：

\[
p_i=\frac{e^{z_i-m}}{\sum_je^{z_j-m}}.
\]

若 logits 是 \([10000,9999,9998]\)，直接计算 \(e^{10000}\) 会溢出；减最大值后变成
\([0,-1,-2]\)，与[数学主线](math.md#softmax)完全相同，结果保持有限。

计算 NLL 时，框架通常使用融合的 log-softmax 或 cross-entropy，而不是先产生低精度 probability 再取 log。
后者可能先把极小概率舍入成 0，继而得到无穷大 loss。

## 5. Probability 与 likelihood 只是观察角度不同

写作 \(p_\theta(D)\) 时：

- 固定参数 \(\theta\)，把 \(D\) 看成可能变化的结果，称为数据的 probability；
- 固定已经观察到的数据 \(D\)，把表达式看成参数 \(\theta\) 的函数，称为 likelihood。

Maximum likelihood estimation（最大似然估计）选择让训练数据最可能的参数：

\[
\hat\theta=\arg\max_\theta\sum_i\log p_\theta(x_i).
\]

等价地，也可以最小化 negative log-likelihood：

\[
\hat\theta=\arg\min_\theta-\sum_i\log p_\theta(x_i).
\]

最大似然只描述训练目标。它不自动保证事实正确、概率校准、公平性、因果解释或分布外可靠。

## 6. NLL、Cross-entropy 和语言模型 loss 的关系

若正确类别是 one-hot 分布 \(y\)，模型输出 \(q\)，cross-entropy 为：

\[
H(y,q)=-\sum_i y_i\log q_i.
\]

由于只有正确项的 \(y_i=1\)，其余为 0：

\[
H(y,q)=-\log q_{\text{正确项}}.
\]

所以单个 one-hot token 的 cross-entropy 就是该目标 token 的 NLL。

训练一批变长序列时，必须说清分母：

\[
\text{mean NLL}
=
\frac{\sum_{b,t}m_{b,t}\,[-\log q(x_{b,t})]}
{\sum_{b,t}m_{b,t}},
\]

其中 \(m_{b,t}\) 是 loss mask。按有效 token 平均与“先对每条序列平均，再对序列平均”不是同一个量；
长短样本的权重会不同。

??? note "自测：一条序列有 2 个有效 token，NLL 分别为 1 和 3。按 token 平均是多少？"

    是 \((1+3)/2=2\)。若与其他长度不同的序列合并，还要继续按事先定义的分母聚合。

## 7. Perplexity：把平均 NLL 指数化

若平均 NLL 使用自然对数：

\[
\operatorname{PPL}=\exp(\overline{\operatorname{NLL}}).
\]

平均 NLL 为 \(\ln 10\) 时，PPL 为 10。它可以粗略理解为模型在每一步面对的“有效候选数”，
但这个直觉不能替代严格定义。

比较 PPL 时必须保持以下条件一致：

- 相同文本数据和预处理；
- 相同 tokenizer；
- 相同 BOS/EOS、截断和 sliding-window 规则；
- 相同 loss mask 与平均分母；
- 相同 log 底数或正确换算。

不同 tokenizer 会改变 token 数与切分方式，因此以 token 为单位的 PPL 不能直接用于跨 tokenizer 排名。

Bits-per-byte（每字节比特数）或 bits-per-character（每字符比特数）可以改善某些比较，
但仍要统一文本编码和数据口径。

## 8. Bayes：新证据怎样更新原有判断

Bayes 定理：

\[
p(H\mid E)=\frac{p(E\mid H)p(H)}{p(E)}.
\]

- \(H\)：一个假设；
- \(E\)：观察到的证据；
- \(p(H)\)：看到证据前的 base rate（基准率）；
- \(p(H\mid E)\)：看到证据后的概率。

例如 10,000 个事件中只有 10 个真实异常。检测器找回其中 9 个，同时把 1% 的正常事件误报为异常，
会产生约 100 个误报。看到“检测器报警”后，真实异常比例约为：

\[
\frac{9}{9+100}\approx8.3\%.
\]

即使召回很高，极低的 base rate 仍会让阳性结果以误报为主。这对安全检测、内容审核、成员推断和异常告警都很重要。

## 9. 熵与交叉熵：平均要花多少信息

分布 \(p\) 的熵：

\[
H(p)=-\sum_xp(x)\log p(x).
\]

它衡量在 \(p\) 自己的结果上，平均有多少不确定性。若一个候选概率为 1，熵为 0；候选更均匀时熵更高。

用模型分布 \(q\) 为来自真实分布 \(p\) 的结果计分，得到 cross-entropy：

\[
H(p,q)=-\sum_xp(x)\log q(x).
\]

若 \(q\) 给真实常见结果的概率太低，平均编码代价会增大。训练数据中的 one-hot label 是这个定义的单样本估计。

## 10. 选读：KL divergence 与方向

!!! info "本节不阻塞主线"

    第一次理解语言模型 loss 和 PPL 不需要掌握 KL。学习蒸馏、变分方法或偏好优化时再回来。

KL divergence：

\[
D_{\mathrm{KL}}(p\|q)
=\sum_xp(x)\log\frac{p(x)}{q(x)}.
\]

它满足：

\[
H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q).
\]

当数据分布 \(p\) 固定时，最小化 cross-entropy 等价于减小
\(D_{\mathrm{KL}}(p\|q)\)。KL 非负，但不对称，也不满足三角不等式，因此不是数学意义上的距离。

方向很重要：

- \(D_{\mathrm{KL}}(p\|q)\)：从 \(p\) 采样并检查 \(q\)，若 \(p(x)>0\) 而 \(q(x)\) 很小，惩罚很大；
- \(D_{\mathrm{KL}}(q\|p)\)：从 \(q\) 采样并检查 \(p\)，对 \(q\) 放在 \(p\) 低密度区惩罚很大。

“覆盖 modes”与“寻找 modes”是常见直觉，只在相应近似设置下使用。有限样本、参数约束和优化过程会改变实际行为，
不能只凭这句口诀预测神经网络结果。

## 11. Token 概率不是事实置信度

\[
p(\text{token}=\text{“巴黎”}\mid\text{prompt})
\]

表示在当前 prompt、tokenizer、参数和解码前处理下，模型下一步生成该 token 的概率。它不等于
“巴黎是事实答案的概率”，原因包括：

- 同一事实有多种措辞和 tokenization；
- 模型学习的是训练分布中的统计关系；
- Prompt 和模板稍变，条件分布就会变化；
- 一个答案可能需要许多 token，首 token 概率不能代表完整事件；
- 模型可能对熟悉但错误的模式给出高概率。

产品若需要置信度，应先定义可验证事件与标签，再在目标分布上做 calibration、selective prediction、
abstention 和切片评测。

## 常见误解

- “链式分解假设 token 独立”：每一项都依赖前缀，没有做这种独立假设。
- “Likelihood 是另一种概率公式”：表达式相同，只是固定与变化的对象不同。
- “Loss 低说明答案都正确”：它只说明当前目标与数据口径下的平均 token 预测更好。
- “PPL 可以跨 tokenizer 比模型”：token 单位不同，数值不能直接比较。
- “KL 是距离”：它不对称，也不满足三角不等式。
- “Token probability 是事实概率”：生成事件与现实命题不是同一个样本空间。

## 换一种讲法

想配合随机变量、采样和代码图示学习，可以阅读《动手学深度学习》的
[概率章节](https://zh.d2l.ai/chapter_preliminaries/probability.html)。
它是另一条入门路线；继续本仓库主线所需的概率概念已经包含在本页。

下一步：想知道 \(p-y\) 怎样传回所有参数，进入[训练数学](math-training.md)；想理解 temperature、
top-k 和 top-p 怎样改动 token 分布，进入[生成入门](../core/generation-basics.md)。
