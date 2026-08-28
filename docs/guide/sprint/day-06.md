# 第 6 天：一次 RAG 请求的完整生命周期

**今日目标**：让同一个问题依次经过授权、BM25、重排、上下文 packing、抽取、引用检查和最终决策，
并用一个无答案问题验证拒答。这是整条速成路线最核心的一天。

**导航**：[上一天](day-05.md) · [返回速成总览](index.md) · [下一天](day-07.md)
{ .doc-nav }

## 时间盒

| 时段 | 内容 | 产出 |
|---|---|---|
| 上午 1h | [RAG 总览](../../applications/rag.md) | 七个阶段的名字 |
| 上午 3h | [实验 5](../../practice/labs/lab-5-rag-request.md) 第一到第三步 | 授权与重排预测表 |
| 下午 2h | 实验 5 第四到第七步 | 三个负例的运行结果 |
| 下午 2h | [一次 RAG 请求的生命周期](../../applications/rag-request-lifecycle.md) | 时序图 |

## 今天要建立的第一条工程直觉

**ACL 必须先于排序。**

不是"排完序再过滤"，也不是"过滤和排序谁先都行"。实验 5 里有一个专门的测试：

```text
tests/test_rag.py::test_hidden_documents_cannot_change_visible_bm25_scores
```

它检查的不是"越权文档不返回"（那太弱了），而是**加入隐藏文档后可见文档的分数也不能变化**。
因为 BM25 的 IDF 依赖全局词频统计——如果隐藏文档参与了统计，
攻击者就能通过观察分数变化推断出他看不到的内容。这是一条真实的信息泄露路径。

## 第二条直觉：有结果不等于能回答

请求 B 会返回 3 个候选，最终仍然 `abstain`。原因是覆盖率：

```text
meaningful query tokens = 9
covered query tokens    = 2
coverage                = 2 / 9
```

先自己数一遍那 9 个 token。注意分词规则是**中文按单字切分**，
所以「灾备」是两个 token 不是一个。

然后回答一个更重要的问题：检索非空、主题相关、覆盖所需事实——这三个信号，
哪个能决定 answerability？

## 第三条直觉：引用语法 ≠ 引用正确

```text
月球由奶酪构成。[S1]
```

如果 `S1` 存在且已授权，citation **syntax** 会通过；但 citation **correctness** 必须失败。
读测试名里的 "valid citation" 时，翻译成"ID 与段落语法合法"，不要翻译成"答案真实"。

## 必答题

1. 为什么 `tenant-b-secret` 不能参与查询期统计，而不只是不能出现在结果里？
2. 两个 chunk 来自同一个 stable source，为什么仍要给不同的 `S1/S2`？
3. `0.55` 这个 coverage 阈值能直接用到生产吗？要定一个真实阈值需要什么数据？

## 今日交付

```text
授权预测表（visible/blocked + 理由）
rerank 预测 rank 与实际 rank
去掉 --principal 后消失的内容清单
「有结果仍拒答」的三信号表
证据边界：本实验没有证明 learned reranker / LLM 质量 / 生产安全
```

## 明天接什么

[第 7 天](day-07.md) 把今天的 BM25 和预录分数换成可独立评测的检索与重排。
