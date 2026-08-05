# 公式速查

## 概率与损失

| 名称 | 公式 | 说明 |
|---|---|---|
| 链式法则 | \(p(x_{1:T})=\prod_t p(x_t\mid x_{<t})\) | 自回归分解 |
| Softmax | \(p_i=e^{z_i}/\sum_j e^{z_j}\) | logits → 概率 |
| 交叉熵 | \(H(p,q)=-\sum_x p(x)\log q(x)\) | 真实分布对模型的编码代价 |
| KL 散度 | \(D_{KL}(p\Vert q)=\sum_xp(x)\log\frac{p(x)}{q(x)}\) | 非对称，不是距离 |
| 困惑度 | \(PPL=e^{\text{mean NLL}}\) | 只在同 tokenization/数据上比较 |
| 熵 | \(H(p)=-\sum_xp(x)\log p(x)\) | 分布不确定性 |

## Transformer

| 名称 | 公式 | 说明 |
|---|---|---|
| 注意力 | \(\text{softmax}(QK^T/\sqrt{d_k}+M)V\) | M 含因果/填充约束 |
| LayerNorm | \(\gamma(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta\) | 按特征归一化 |
| RMSNorm | \(\gamma x/\sqrt{\text{mean}(x^2)+\epsilon}\) | 不减均值 |
| SwiGLU | \((\text{SiLU}(xW_g)\odot xW_u)W_d\) | 门控 MLP |
| LoRA | \(W'=W+(\alpha/r)BA\) | 低秩可训练增量 |

## 优化与规模

| 名称 | 公式 | 说明 |
|---|---|---|
| 梯度下降 | θₜ₊₁ = θₜ - η∇L(θₜ) | η 为学习率 |
| 全局 batch | \(B_{micro}\times accumulation\times DP\) | 不乘 TP/PP |
| 粗略训练 FLOPs | \(C\approx6ND\) | dense decoder 的预算近似 |
| Scaling law | \(L\approx L_\infty+aN^{-\alpha}+bD^{-\beta}\) | 参数依实验而变 |
| KV 元素/序列 | \(2LTH_{kv}D\) | 再乘 batch 与元素字节 |

## 检索与评价

| 名称 | 公式 | 说明 |
|---|---|---|
| 余弦相似度 | \(x^Ty/(\lVert x\rVert\lVert y\rVert)\) | 向量方向相似 |
| RRF | \(\sum_i1/(k+\text{rank}_i(d))\) | 融合多个排序 |
| Precision | \(TP/(TP+FP)\) | 预测为正中有多少正确 |
| Recall | \(TP/(TP+FN)\) | 真实为正中找回多少 |
| F1 | \(2PR/(P+R)\) | P/R 调和平均 |

公式前先确认符号、shape、对数底、归一化单位与 mask。不同论文同一字母可能含义完全不同。
