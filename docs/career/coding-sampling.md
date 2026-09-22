# 编码题 2：Top-k 与 Top-p 采样

**练习导航**：[返回编码轮索引](coding-round.md) · [生成与解码](../core/generation.md) · [实验 0A](../practice/labs/lab-0a-sampling.md)
{ .doc-nav }

本页是一道可以独立计时、实现和验收的练习。开始前先写清输入输出、错误契约和至少两个边界例。

**题面**：给定 logits 向量和 `temperature`、`top_k`、`top_p`，返回下一个 token id。
为了可测试，把随机数作为参数 `uniform ∈ [0, 1)` 传入，不要在函数内部调用全局随机数。

**必须先问清楚的三件事**：

- **顺序**：temperature、top-k、top-p 按什么次序作用？归一化发生在截断之前还是之后？
  不同框架的实现并不一致，这是本题最大的歧义来源。
- **并列**：两个 token 分数完全相同且卡在 top-k 边界上，保留哪个？
- **top-p 的边界**：累计概率恰好等于阈值的那个 token，包含还是排除？

**最小实现**：

```python
import numpy as np


def sample_next_token(logits, *, temperature=1.0, top_k=None, top_p=None, uniform):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("logits must be a non-empty finite vector")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0 or top_k > values.size
    ):
        raise ValueError("top_k must be in [1, vocabulary_size]")
    if top_p is not None and (
        isinstance(top_p, bool) or not isinstance(top_p, (int, float))
        or not np.isfinite(top_p) or not 0 < top_p <= 1
    ):
        raise ValueError("top_p must be in (0, 1]")
    if isinstance(uniform, bool) or not isinstance(uniform, (int, float)) \
            or not np.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError("uniform must be finite and in [0, 1)")

    scaled = values / temperature
    # 并列时按 token id 升序打破，保证可复现。
    token_ids = np.arange(scaled.size)
    order = np.lexsort((token_ids, -scaled))

    keep = order[: top_k] if top_k is not None else order
    probabilities = _masked_softmax(scaled, keep)

    if top_p is not None and top_p < 1:
        cumulative = np.cumsum(probabilities[keep])
        # side="left" 保留第一个达到或越过阈值的 token。
        cutoff = min(int(np.searchsorted(cumulative, top_p, side="left")), len(keep) - 1)
        keep = keep[: cutoff + 1]
        probabilities = _masked_softmax(scaled, keep)

    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0  # 消除浮点累加误差导致的末尾小于 1
    return int(np.searchsorted(cumulative, uniform, side="right"))


def _masked_softmax(scaled, keep):
    probabilities = np.zeros_like(scaled)
    subset = scaled[keep]
    exponentials = np.exp(subset - subset.max())
    probabilities[keep] = exponentials / exponentials.sum()
    return probabilities
```

注意 top-p 之后**重新归一化**了一次。如果沿用 top-k 阶段的概率直接采样，
累计和小于 1，落在尾部的 `uniform` 会取不到任何 token。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 空或非有限 logits | `max`、排序或 softmax 产生无意义结果 | 传空数组、`nan` 和 `inf`，断言在采样前抛异常 |
| `top_k` 大于词表 | 切片静默成功，调用方误以为保留了精确的 k 个候选 | 断言抛异常；本题不把它悄悄改成“全部保留” |
| 分数并列 | 依赖排序稳定性，跨版本结果漂移 | 两个相同 logit，断言固定返回较小的 token id |
| `top_p` 卡在边界 | `side` 参数写反时少保留一个 token | 概率 `[0.6, 0.3, 0.1]`、`top_p=0.6`，断言只保留第一个 |
| `uniform` 接近 1 | 浮点累加使末尾小于 1，`searchsorted` 越界 | 传 `uniform=0.999999`，断言返回合法 id |
| `temperature=0` | 直接除零得 `inf`/`nan` | 断言抛异常，并说明 greedy 应走单独分支 |

**面试官的下一个追问**：

- *`temperature=0` 为什么不能靠"除以一个很小的数"实现？* 会先溢出成 `inf`，再在 softmax 里变成 `nan`；
  greedy 是独立的 `argmax` 分支，不是极限情形。
- *相同参数、相同 seed，两次请求一定输出相同 token 吗？* 不一定，见
  [核心题第 30 题](interview-questions.md#replay-identity)。

**仓库参考实现**：`src/about_llm/inference/sampling.py`。该实现还加入了 repetition penalty，
并把每一步的候选集合与概率完整记录下来便于复核。测试见 `tests/test_sampling.py`。

```bash
python -m pytest tests/test_sampling.py -q
```

## 完成信号

在不查看参考实现的情况下重新写一遍，并能解释复杂度、失败行为、一个反例以及测试为什么能推翻错误实现。
