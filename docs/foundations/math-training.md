# 训练数学：梯度怎样变成一次参数更新

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：理解 loss 和梯度正负号，想继续看反向传播、梯度累积与 AdamW 的读者。
- **先修**：[数学主线](math.md)中的一次负梯度更新。
- **首次阅读**：导数 → 计算图 → 链式法则 → 反向传播 → gradient accumulation（§7）→ SGD。
  AdamW、gradient clipping、浮点误差与 VJP 可按需阅读。
- **完成信号**：能区分 backward 与 optimizer step，并解释 gradient accumulation 的分母。
- **卡住时**：回到[数学基础主线](math.md#one-update)，先只判断一个梯度的正负号。

</div>

训练不是“模型看见答案后自动记住”。程序先用 forward 计算预测与 [loss](../reference/glossary.md#term-loss)，再用 backward 求出每个参数附近的
变化方向，最后由 optimizer 更新参数：

~~~text
参数与样本
→ forward 得到 loss
→ backward 得到 gradient
→ optimizer 计算 update
→ 新参数
~~~

这一页用简单函数把四步拆开，再说明它们在 LLM 训练中怎样扩大。

## 1. 导数是局部斜率

设：

\[
L(w)=(w-3)^2.
\]

当 \(w=1\) 时，\(L=4\)。把 \(w\) 稍微改成 1.01，loss 会变小；改成 0.99，loss 会变大。
中心有限差分用两次扰动估计斜率：

\[
\frac{dL}{dw}
\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}.
\]

这个函数的精确导数是：

\[
\frac{dL}{dw}=2(w-3).
\]

代入 \(w=1\) 得 \(-4\)。负号表示：在当前位置增大 \(w\)，loss 会下降。梯度下降更新：

\[
w_{\text{new}}=w-\eta\frac{dL}{dw}.
\]

因为导数为负，减去负数会让 \(w\) 增大，向最小点 3 靠近。

有限差分能帮助检查小实现，但训练大模型不会为每个参数额外做两次 forward；参数太多，成本也太高。

??? note "自测：若梯度为 5，learning rate（[术语](../reference/glossary.md#term-learning-rate)）为 0.1，一次 SGD 更新量是多少？"

    \(w_{\text{new}}=w-0.1\times5=w-0.5\)，参数减小 0.5。

## 2. 偏导数与梯度：一次只看一个参数

若 loss 同时依赖 \(w_1,w_2\)：

\[
L(w_1,w_2),
\]

\(\frac{\partial L}{\partial w_1}\) 表示暂时固定 \(w_2\)，只让 \(w_1\) 变化时的斜率。
把所有偏导数排成向量：

\[
\nabla L=
\left[
\frac{\partial L}{\partial w_1},
\frac{\partial L}{\partial w_2}
\right],
\]

就是梯度。它指向当前位置局部上升最快的方向，负梯度指向局部下降最快的方向。

“局部”很重要：一步太大可能越过低点；复杂神经网络也可能有平坦区、鞍点和噪声。

## 3. 计算图：复杂公式拆成简单节点

考虑：

\[
a=wx,\qquad
b=a+c,\qquad
L=b^2.
\]

它可以画成：

~~~text
w ─┐
   × → a ─┐
x ─┘      + → b → square → L
       c ─┘
~~~

Forward 从左到右保存必要中间量。Backward 从 \(L\) 开始，沿反方向把梯度传回 \(b,a,w\)。

真实 Transformer 计算图很大，但每个节点仍是矩阵乘法、加法、归一化、激活或其他局部运算。
自动微分系统只需知道每种局部运算怎样把上游梯度传给输入。

## 4. [链式法则](../reference/glossary.md#term-chain-rule)：沿依赖关系相乘

若 \(w\rightarrow a\rightarrow b\rightarrow L\)，则：

\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial b}
\frac{\partial b}{\partial a}
\frac{\partial a}{\partial w}.
\]

给出 \(w=2,x=3,c=1\)：

\[
a=6,\quad b=7,\quad L=49.
\]

局部导数：

\[
\frac{\partial L}{\partial b}=2b=14,\qquad
\frac{\partial b}{\partial a}=1,\qquad
\frac{\partial a}{\partial w}=x=3.
\]

所以：

\[
\frac{\partial L}{\partial w}=14\times1\times3=42.
\]

用人话说：\(w\) 变化 1 单位让 \(a\) 约变化 3 单位；\(a\) 变化 1 单位让 \(b\) 变化 1 单位；
\(b\) 变化 1 单位让 loss 在当前位置约变化 14 单位。连乘得到总影响 42。

若一个变量沿多条路径影响 loss，各路径贡献需要相加。Residual connection 正是这种“分叉后再汇合”的常见情况。

??? note "自测：上例中 \(\partial L/\partial c\) 是多少？"

    \(c\rightarrow b\rightarrow L\)，所以
    \(\frac{\partial L}{\partial c}=\frac{\partial L}{\partial b}\frac{\partial b}{\partial c}=14\times1=14\)。

## 5. 反向传播计算梯度，optimizer 更新参数

Backpropagation（反向传播）是高效应用链式法则，得到：

\[
g_t=\nabla_\theta L_t.
\]

Optimizer 接收梯度和自己的状态，产生参数更新：

\[
\theta_{t+1}=\operatorname{update}(\theta_t,g_t,\text{state}_t).
\]

因此：

- 调用 backward：计算或累积 gradient，通常还没改参数；
- 调用 optimizer step：根据 gradient 真正改参数；
- 调用 zero grad：清除或置空旧 gradient，避免无意累积。

调试“loss 不下降”时，要分别检查 loss 是否连到参数、gradient 是否有限、参数是否交给 optimizer，
以及 step 是否真的执行。这样可以定位训练链路中具体出错的环节。

## 6. 线性层的梯度怎样对应 shape

对：

\[
Y=XW,
\]

若上游梯度 \(G=\frac{\partial L}{\partial Y}\)，则：

\[
\frac{\partial L}{\partial X}=GW^T,
\qquad
\frac{\partial L}{\partial W}=X^TG.
\]

检查 shape：

~~~text
X:       [N,D]
W:       [D,H]
Y, G:    [N,H]
dX:      [N,H] @ [H,D] → [N,D]
dW:      [D,N] @ [N,H] → [D,H]
~~~

\(dW\) 会把 \(N\) 个样本对同一共享权重的贡献加起来。LLM 中的 \(N\) 常把 batch 与 token 轴展平；
哪些 token 参与贡献还受 loss mask 影响。

## 7. Gradient accumulation：先累积，再更新

显存装不下 global batch 时，可以把它拆成 \(A\) 个 micro-batch。若目标是这些 loss 的平均：

\[
\nabla\left(\frac1A\sum_{a=1}^{A}L_a\right)
=
\frac1A\sum_{a=1}^{A}\nabla L_a.
\]

常见做法：

1. 清空 gradient；
2. 对每个 micro-batch 计算 loss；
3. 用除以 \(A\) 的 loss 做 backward，或在别处统一缩放；
4. 累积完 \(A\) 次后执行一次 optimizer step；
5. 再次清空 gradient。

若每个 loss 都没有除以 \(A\)，最终得到的是 sum，不是 mean，等效更新尺度会放大 \(A\) 倍。

### 变长序列不能简单平均 micro-batch mean

一个 micro-batch 有 100 个有效 token，另一个只有 10 个。若先各自求 mean 再等权平均，
第二组每个 token 的权重会更大。要得到 token mean，应累计：

\[
\frac{\sum \text{token loss}}{\sum \text{有效 token 数}}.
\]

框架、分布式 reducer 或训练器可能在不同位置缩放。最稳妥的检查是：用同一批样本比较“单次大 batch”
与“拆分 accumulation”的参数更新。

??? note "自测：4 个等长 micro-batch 的 loss 都未经缩放就 backward，累积梯度相对平均梯度大多少？"

    大 4 倍。可以让每个 loss 除以 4，或在更新前对累积梯度做等价缩放。

## 8. SGD、Momentum 与 learning-rate schedule

最简单的 SGD：

\[
\theta_{t+1}=\theta_t-\eta_tg_t.
\]

\(\eta_t\) 是 learning rate。Mini-batch gradient 只是完整数据梯度的随机估计，因此不同 batch 会带来噪声。

Momentum 对历史梯度做指数平滑，使更新在长期一致的方向积累，在来回震荡的方向部分抵消。它增加了 optimizer state，
也引入 momentum 系数。

Warmup 在训练初期逐步增大学习率，减少初始化阶段一步过大的风险。线性或余弦衰减则控制后期更新。

报告学习率曲线时，要说明横轴按“参数更新次数”还是“已消费 token 数”计算。改变梯度累积次数或 batch 大小后，
相同的参数更新次数会对应不同的数据量。

## 9. AdamW：为什么它不只是“高级 SGD”

Adam 维护梯度的一阶、二阶移动平均：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

偏差修正后：

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
\]

AdamW 的简化更新是：

\[
\theta_{t+1}
=\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta_t\lambda\theta_t.
\]

中间项根据历史梯度尺度调整每个参数的有效步长；最后一项是 decoupled weight decay，
直接让参数按比例衰减，不经过 Adam 的二阶矩缩放。

### 本节真正需要记住的只有三点

1. Gradient 是当前 loss 的局部信息；
2. AdamW 还维护历史状态，所以仅保存参数不能无缝继续训练；
3. Weight decay 的参数 mask 很重要，bias 与 normalization scale 常按训练配置排除，但不是算法自动识别。

!!! info "Adam 细节是选读"

    第一次学习只需理解 optimizer 会把 gradient、历史状态和 learning rate 合成 update。
    偏差修正、epsilon 位置和不同实现差异可在调试训练时再查。

## 10. Gradient clipping：限制一步有多大

Global norm clipping：

\[
g'=g\min\left(1,\frac{c}{\lVert g\rVert_2+\epsilon}\right).
\]

若整体 norm 不超过阈值 \(c\)，gradient 不变；超过时，所有分量按同一比例缩小，整体方向保持不变。
Per-value clipping 则逐元素截断，可能明显改变方向。

应记录裁剪前的梯度范数和触发比例。若几乎每一步都大幅裁剪，需要继续检查学习率、数据、loss scale 和数值问题。
裁剪只能限制有限梯度的大小，NaN 与错误标签仍要单独修复。

## 11. 浮点数为什么会让正确公式算坏 {#floating-point}

计算机浮点数能表示的范围和精度有限：

- overflow：大数取指数后超出范围；
- underflow：极小概率或 gradient 变成 0；
- cancellation：两个接近的大数相减，丢失有效位；
- non-associativity：浮点下 \((a+b)+c\) 可能不等于 \(a+(b+c)\)；
- reduction order：并行求和顺序改变，末位也可能改变。

常见格式：

- BF16：指数范围接近 FP32，但尾数较短；
- FP16：指数范围更小，训练中常配合 loss scaling；
- FP8：还要说明具体格式和 scaling 策略；
- FP32：更高精度和更大存储成本，仍不是实数的无限精确表示。

“使用 BF16 训练”仍不够具体。参数存储、矩阵乘法输入、累加、gradient、optimizer state 和通信归约
可能使用不同 dtype。

测试浮点结果时常用：

\[
|a-b|\le \text{atol}+\text{rtol}|b|.
\]

近 0 数值主要由绝对容差 atol 控制，大数主要由相对容差 rtol 控制。容差应来自 dtype、运算长度和用途，
不能只为让测试通过而不断放宽。

## 12. 选读：Jacobian 与 VJP

!!! info "本节不阻塞主线"

    若你只想训练和微调模型，知道自动微分能正确应用链式法则就够了。

向量函数 \(y=f(x)\) 的 Jacobian 收集每个输出对每个输入的偏导。若输入和输出都很大，
显式构造完整 Jacobian 会非常昂贵。

神经网络训练从一个标量 loss 出发。反向模式自动微分会直接计算反向传播所需的乘积。

这个乘积叫向量—Jacobian 乘积（vector–Jacobian product，VJP）；对应的自动微分方式英文写作
reverse-mode autodiff。

这个过程只传播当前上游梯度需要的乘积，无需创建完整 Jacobian。因此，它特别适合“很多参数 → 一个标量 loss”
的训练问题。

## 13. 用什么证据检查训练链路

按成本从低到高：

1. 手算一个极小例子的 forward 与 gradient 符号；
2. 用有限差分检查少量参数，优先用 FP64/FP32；
3. 确认 gradient 非零且有限，optimizer step 后参数确实变化；
4. 用极小 batch overfit，检查实现能否降低训练 loss；
5. 与独立实现或单卡大 batch 对齐；
6. 最后才看 held-out 数据、多个 seed 和真实任务指标。

每一层回答的问题不同。Tiny-batch overfit 能发现训练闭环错误，不能证明泛化；训练 loss 下降也不能替代
validation/test 评测。

## 常见误解

- “Backward 会自动修改参数”：它通常只计算或累积 gradient。
- “梯度就是更新量”：optimizer 还会使用 learning rate、历史状态和 weight decay。
- “Gradient accumulation 只要多 backward 几次”：还必须定义 sum/mean 和有效 token 分母。
- “AdamW 等于 Adam 加 L2 loss”：对自适应 optimizer，解耦 weight decay 与 L2 正则通常不等价。
- “Gradient clipping 能治 NaN”：NaN 的来源仍需单独定位。
- “混合精度只有一个 dtype”：不同运算和状态可能使用不同精度。

## 换一种讲法

想从代码直觉补充自动微分，可以阅读
[micrograd](https://github.com/karpathy/micrograd)；想跟着视频与 Notebook 从标量梯度走到语言模型，
可以看 [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)。
需要更传统的微积分图示时，可参考《动手学深度学习》的
[微积分章节](https://zh.d2l.ai/chapter_preliminaries/calculus.html)。

下一步：回到[数学主线](math.md#one-update)运行一次更新，或进入[机器学习与深度学习](ml-dl.md)
把训练 loss、validation 与 test 接成完整学习闭环。
