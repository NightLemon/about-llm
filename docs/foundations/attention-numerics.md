# Attention 数值计算：为什么分块后仍是同一个 Softmax

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已经会算 Attention，想继续理解 FlashAttention 与 online softmax 的读者。
- **先修**：[数学基础](math.md)中的 Attention 与 softmax；浮点误差可按需查[训练数学](math-training.md#floating-point)。
- **首次阅读**：完整矩阵的问题 → 一行四个分数 → 三项在线状态 → mask 边界 → 可运行对照。
- **完成信号**：能解释新最大值出现时旧块为何要重标定，并区分逻辑 tile 大小与真实显存峰值。
- **卡住时**：回到[两-token 手算例子](math.md#two-token-attention)；先理解普通 softmax，再看本页递推式。

</div>

普通实现先构造完整分数[矩阵](../reference/glossary.md#term-matrix)，再逐行 [softmax](../reference/glossary.md#term-softmax)：

\[
S=\frac{QK^T}{\sqrt{d_h}}+M,
\qquad
O=\operatorname{softmax}(S)V.
\]

序列长度为 \(T\) 时，\(S\) 有 \(T^2\) 个元素。[FlashAttention](../reference/glossary.md#term-flashattention)不改变这个数学目标；它把 key/value 分块读取，
算完当前块就更新少量逐行状态，避免把完整分数和概率矩阵写回高带宽显存（HBM）。

先记住结论：它主要减少中间存储和 HBM 读写。对精确的稠密 Attention，\(QK^T\) 与 \(AV\) 的算术复杂度通常仍是
\(O(T^2)\)。

## 用一行四个分数理解“重标定”

假设一行 Attention 分数分成两块：

```text
block 1: [2, 1]
block 2: [4, 0]
```

第一块的最大值是 2。为了避免直接计算很大的指数，先减去 2：

\[
\ell_1=e^{2-2}+e^{1-2}=1+e^{-1}\approx1.368.
\]

第二块出现了更大的最大值 4。最终 softmax 必须以 4 为共同基准，所以第一块已经累计的指数和要乘
\(e^{2-4}=e^{-2}\)：

\[
\ell_2=e^{-2}\ell_1+e^{4-4}+e^{0-4}
\approx1.203.
\]

这正好等于一次性计算
\(e^{2-4}+e^{1-4}+e^{4-4}+e^{0-4}\)。旧分数不必重新读取，只需把旧统计量换到新的最大值坐标系。

## 三项状态怎样同时更新

固定一行 query，把第 \(b\) 个 key block 中已经缩放并应用 mask 的分数记作 \(s_{b,j}\)。算法维护：

| 状态 | 含义 |
|---|---|
| \(m_b\) | 到当前块为止见过的最大分数 |
| \(\ell_b\) | 以 \(m_b\) 为基准的指数和 |
| \(o_b\) | 尚未除以 \(\ell_b\) 的 value 加权和 |

初值为 \(m_0=-\infty,\ell_0=0,o_0=0\)。读入第 \(b\) 个块后：

\[
m_b=\max\left(m_{b-1},\max_j s_{b,j}\right),
\]

\[
\ell_b=e^{m_{b-1}-m_b}\ell_{b-1}+
\sum_j e^{s_{b,j}-m_b},
\]

\[
o_b=e^{m_{b-1}-m_b}o_{b-1}+
\sum_j e^{s_{b,j}-m_b}v_{b,j}.
\]

处理完全部 \(B\) 个块后，输出 \(o_B/\ell_B\)。\(\ell\) 更新 softmax 的分母，\(o\) 用同样的缩放更新
带 value 的分子，因此两者始终位于同一个最大值坐标系。

## 全 mask 行要显式处理

不可见位置的分数设为 \(-\infty\)。如果此前没有可见 key，旧 \(\ell\) 和 \(o\) 都是 0；实现应直接把旧贡献定义为 0，
避免真的计算 \(-\infty-(-\infty)\)。

若前几个块全部不可见，后面的块第一次出现可见 key，算法仍可从零状态开始累计。若处理完所有块后整行都不可见，
则 \(\ell_B=0\)，softmax 没有定义，程序应拒绝这组 mask。

## 为什么实现结果允许有微小误差

在实数算术下，分块递推与完整 softmax 等价。真实计算使用有限精度；分块方式和归约顺序不同，末位可能发生变化。
验收时应根据 dtype 和运算规模设置误差容限，而不是要求逐 bit 相同。

仓库的 CPU 对照使用 float64 累积，只构造当前分数 tile 和逐行状态：

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
python -m pytest tests/test_attention_numpy.py -q
~~~

测试覆盖：

- 普通广播；
- causal prefill 与 decode；
- 稀疏 mask 和大 logits；
- 全 mask 行拒绝；
- 不同 block size。

每组结果都会与独立的完整 Attention 参考实现比较。

## 怎样读演示报告

报告里的 `logical_peak_score_elements` 表示算法一次最多保留多少个逻辑分数元素。
它没有包含 Q/K/V、输出、NumPy 临时量、Python 对象或 allocator，因此不是进程峰值内存。

CPU 对照能够验证递推公式、mask 边界和数值误差，但它没有执行 CUDA。

[Kernel](../reference/glossary.md#term-kernel)是否正确、FlashAttention backend 是否启用，以及 HBM 流量与速度是否改善，还要在目标 GPU 和固定版本的
运行时中测量。

继续阅读 [Transformer 的 kernel 边界](../core/transformer.md#10-kernel) 可以把这套数值算法放回完整模型；
[推理请求生命周期](../systems/inference-request-lifecycle.md)则展示 prefill 与 decode 怎样调用不同 Attention 路径。
