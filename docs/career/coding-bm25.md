# 编码题 3：带权限的 BM25 检索

**练习导航**：[返回编码轮索引](coding-round.md) · [RAG 检索](../applications/rag-retrieval.md) · [RAG Foundations](../practice/projects/rag-foundations.md)
{ .doc-nav }

本页是一道可以独立计时、实现和验收的练习。开始前先写清输入输出、错误契约和至少两个边界例。

**题面**：实现一个内存 BM25 索引，`search(query, tenant_id, principals, top_k)`
只返回该租户下调用方有权访问的文档。文档带 `tenant_id` 和 `acl` 字段，
空 `acl` 表示租户内公开。

**必须先问清楚的三件事**：

- 权限过滤发生在打分**之前**还是之后？（这是本题真正的考点，见下。）
- 空查询、查询词全部未登录，返回空列表还是抛异常？
- 需要支持文档更新和删除吗？会影响索引结构的选择。

**最小实现**（BM25 打分部分）：

```python
import math
from collections import Counter


def bm25_scores(query_terms, term_frequencies, lengths, *, k1=1.5, b=0.75):
    number_of_documents = len(term_frequencies)
    average_length = sum(lengths) / number_of_documents

    document_frequency = Counter()
    for frequencies in term_frequencies:
        document_frequency.update(frequencies.keys())

    idf = {
        term: math.log(1 + (number_of_documents - count + 0.5) / (count + 0.5))
        for term, count in document_frequency.items()
    }

    scores = []
    for frequencies, length in zip(term_frequencies, lengths):
        score = 0.0
        for term in query_terms:
            if term not in frequencies:
                continue
            frequency = frequencies[term]
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += idf[term] * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores
```

**权限必须先于打分**，这是最容易答错的一层。很多人先算完分数再过滤结果，
但 BM25 的 IDF 和平均文档长度是**语料级统计量**：如果它们由全部文档算出，
分数本身就编码了不可见文档的信息。攻击者可以通过观察分数变化、结果数量或响应时间
推断某份机密文档是否存在。正确做法是先算出可见文档集合，再在这个子集上计算 IDF、
平均长度和分数。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 跨租户 | 过滤写在打分之后，分数已泄漏存在性 | 同一查询在两个租户下断言结果集合不相交 |
| 可见集合为空 | 平均长度除零 | 断言返回空列表而不是崩溃 |
| 空查询 | 返回全部文档或崩溃 | 断言返回空列表 |
| 未登录词 | `idf[term]` 抛 `KeyError` | 断言跳过该词而不是报错 |
| 长文档 | `b=0` 时长文档因词频高而霸榜 | 对比 `b=0` 与 `b=0.75` 下长短文档的排序 |
| 分数并列 | 排序不稳定，翻页结果抖动 | 断言按 `document_id` 打破并列 |

**面试官的下一个追问**：

- *为什么不直接用向量检索替代 BM25？* 见[深挖题 16](../evidence/interview-controls.md)。
- *重排序阶段还需要再检查一次 ACL 吗？* 需要。重排器可能引入索引之外的候选，
  权限必须在每一层重新成立，而不是依赖上游。

**仓库参考实现**：`src/about_llm/rag/bm25.py` 的 `BM25Index`。它把"先授权再统计"
做成了默认行为，并保留一个 `_legacy_global_statistics` 开关专门用于复现旧工件——
这本身就是一个可以讲的工程决策。测试见 `tests/test_rag.py`。

```bash
python -m pytest tests/test_rag.py -q
```

## 完成信号

在不查看参考实现的情况下重新写一遍，并能解释复杂度、失败行为、一个反例以及测试为什么能推翻错误实现。
