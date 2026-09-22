# 编码题 4：Reciprocal Rank Fusion

**练习导航**：[返回编码轮索引](coding-round.md) · [RAG 检索](../applications/rag-retrieval.md) · [RAG Foundations](../practice/projects/rag-foundations.md)
{ .doc-nav }

本页是一道可以独立计时、实现和验收的练习。开始前先写清输入输出、错误契约和至少两个边界例。

**题面**：给定多个检索器各自的排序结果，融合成一个排序。不同检索器的分数不可比
（BM25 是无界正数，余弦相似度在 \([-1, 1]\)），因此只能使用排名。
本题固定输入的 `rank` 为从 1 开始的正整数，`rank_constant >= 0`、`top_k > 0`；
仓库的 `SearchResult` 在构造时检查这个排名契约。

**必须先问清楚的三件事**：

- 同一文档在多个列表中出现，怎样合并？（按 id 累加。）
- 某个文档在同一个列表里出现两次怎么办？
- 排名从 0 还是从 1 开始？直接影响 \(k=0\) 时是否除零。

**最小实现**：

```python
from collections import defaultdict


def reciprocal_rank_fusion(rankings, *, rank_constant=60, top_k=10):
    if rank_constant < 0 or top_k <= 0:
        raise ValueError("rank_constant must be non-negative and top_k positive")
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for result in ranking:
            document_id = result.document.document_id
            if document_id in seen:      # 同一列表内去重，防止自我加权
                continue
            seen.add(document_id)
            scores[document_id] += 1.0 / (rank_constant + result.rank)
    # 并列时按 document_id 升序，保证结果可复现
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return ordered[:top_k]
```

公式是 \(\mathrm{RRF}(d)=\sum_{r} \frac{1}{k + \mathrm{rank}_r(d)}\)。
常数 \(k\)（通常取 60）压低了头部名次之间的差距。排名从 1 开始时，第 1 名与第 2 名的贡献比是
\(\frac{k+2}{k+1}\)：\(k=60\) 时约为 \(1.016\)，两者几乎等价；而 \(k=0\) 时是 \(2\)，
第一名的权重是第二名的两倍。
所以 \(k\) 越大越接近"多个检索器投票"，越小越接近"只信第一名"。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 同列表重复文档 | 该文档被重复加分，等于自我投票 | 构造重复项，断言只计一次 |
| 只有一个列表 | 应退化为原排序 | 断言输出顺序与输入一致 |
| 融合分并列 | 排序不稳定 | 断言按 `document_id` 打破并列 |
| 非正排名 | `rank_constant=0` 时除零；其他值也失去排名含义 | 构造 `rank=0`，断言在构造结果对象时拒绝 |
| 空输入 | 崩溃 | 断言返回空列表 |

**面试官的下一个追问**：

- *为什么不把两个分数归一化后加权求和？* 归一化依赖当前批次的极值，
  一个离群分数就会改变全部结果；RRF 只用排名，对分数尺度免疫。代价是丢掉了
  "第一名比第二名好很多"这类信息。
- *融合之后还要重新检查权限吗？* 要，理由同上一题。

**仓库参考实现**：`src/about_llm/rag/rank_fusion.py` 的 `reciprocal_rank_fusion`，
测试见 `tests/test_rag.py`。它与上面的最小实现有两处差别值得注意：返回的是重新编号的
`SearchResult` 而不是裸 id，并且把每个文档**命中过哪些检索器**记录进 `source`
（形如 `rrf:bm25+dense`）。融合之后无法回答"这条是谁召回的"，
线上归因就断了——这是最小实现最容易丢掉的一环。

## 完成信号

在不查看参考实现的情况下重新写一遍，并能解释复杂度、失败行为、一个反例以及测试为什么能推翻错误实现。
