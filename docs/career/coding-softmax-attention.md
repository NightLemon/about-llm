# 编码题 1：数值稳定 Softmax 与 Causal Attention

**练习导航**：[返回编码轮索引](coding-round.md) · [Transformer](../core/transformer.md) · [Transformers Basics](../practice/projects/transformers-basics.md)
{ .doc-nav }

本页是一道可以独立计时、实现和验收的练习。开始前先写清输入输出、错误契约和至少两个边界例。

**题面**：实现 `scaled_dot_product_attention(query, key, value, mask=None)`，
返回 `(output, probabilities)`。query 形状 `(..., query_length, head_dim)`，
key/value 形状 `(..., key_length, head_dim)`，前导维度需要广播。再实现一个
`causal_mask(query_length, key_length)`，`True` 表示该 key 对该 query 可见。

**必须先问清楚的三件事**：

- mask 的语义是 `True 可见` 还是 `True 屏蔽`？两种约定都常见，写反了结果完全错。
- `key_length` 可以大于 `query_length` 吗？（带 KV Cache 的 decode 就是这种情况。）
- 需要返回注意力概率吗？返回它才能做后面的不变性测试。

**最小实现**：

```python
import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def causal_mask(query_length, key_length=None):
    key_length = query_length if key_length is None else key_length
    past_length = key_length - query_length
    if past_length < 0:
        raise ValueError("key_length cannot be smaller than query_length")
    query_positions = np.arange(query_length)[:, None] + past_length
    key_positions = np.arange(key_length)[None, :]
    return key_positions <= query_positions


def scaled_dot_product_attention(query, key, value, *, mask=None):
    scale = float(query.shape[-1]) ** -0.5
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
    if mask is not None:
        mask = np.broadcast_to(mask, scores.shape)
        if np.any(np.all(~mask, axis=-1)):
            raise ValueError("every query row must have at least one visible key")
        scores = np.where(mask, scores, -np.inf)
    probabilities = softmax(scores, axis=-1)
    return np.matmul(probabilities, value), probabilities
```

`causal_mask` 里的 `past_length` 偏移是这题真正的考点：decode 时只送入一个新 query，
但 key 包含全部历史，`query_positions` 必须右移 `key_length - query_length` 才对齐。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 极大 logit | 不减最大值时 `exp` 溢出成 `inf`，再相除得 `nan` | 输入 `[1000., 1001.]`，断言结果有限且和为 1 |
| 整行被 mask | 全 `-inf` 行 softmax 得 `0/0 = nan`，且会静默传播 | 构造全 `False` 的一行，断言抛出异常而不是返回 `nan` |
| 未来 token 不变性 | mask 广播方向写反时仍能跑，但泄漏未来信息 | 只改 \(t\) 之后的 token，断言位置 \(0..t\) 的输出不变 |
| `key_length > query_length` | 偏移漏掉时 decode 路径全错 | 令 `query_length=1`、`key_length=5`，断言该行全部可见 |

**为什么全 mask 行应该抛异常而不是返回 0**：返回零向量会让上游把"无可见 key"当成一次正常注意力，
错误一直传到输出层才暴露。抛异常把失败点固定在构造 mask 的地方。

**面试官的下一个追问**：

- *为什么除以 \(\sqrt{d}\) 而不是 \(d\)？* —— 见[核心题第 1 题](interview-questions.md#attention-scaling)。
- *这个实现要保存完整的 `(query_length, key_length)` 概率矩阵，长序列怎么办？*
  这引向 online softmax 与分块计算，仓库的 `blockwise_online_attention` 是可对照的实现。

**仓库参考实现**：`src/about_llm/from_scratch/attention_numpy.py`，测试见
`tests/test_attention_numpy.py`。

```bash
python -m pytest tests/test_attention_numpy.py -q
```

## 完成信号

在不查看参考实现的情况下重新写一遍，并能解释复杂度、失败行为、一个反例以及测试为什么能推翻错误实现。
