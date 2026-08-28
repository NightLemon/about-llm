# 实验 5：追踪一次 RAG 问答

这个实验不下载模型，也不需要向量数据库。你会让同一个问题依次经过授权、BM25、重排、
上下文 packing、逐字抽取、引用检查和最终决策，再用无答案问题验证拒答。

**相关教材**：[RAG 总览](../../applications/rag.md) ·
[一次 RAG 请求的生命周期](../../applications/rag-request-lifecycle.md) ·
[RAG Foundations](../projects/rag-foundations.md)
{ .doc-nav }

## 完成标准

完成后，你应该能不看输出回答：

1. 为什么 `tenant-b-secret` 不能参与请求 A 的查询期统计和排序。
2. BM25 top-3 经过 recorded reranker 后，哪两个 chunk 进入 top-2。
3. `S1` 是怎样绑定到 stable source 和 exact chunk 的。
4. 为什么请求 B 有三个结果，最终仍然 abstain。
5. 本实验为什么不能证明 learned reranker、LLM 质量或生产安全。

预计时间为 45–90 分钟。

## 准备环境

从仓库根目录安装基础依赖：

~~~powershell
python -m pip install -c constraints/ci.txt -e .
~~~

本实验使用三个固定文件：

```text
projects/rag-foundations/sample_corpus.jsonl
projects/rag-foundations/reranker-scores.example.jsonl
projects/rag-foundations/rag_request_walkthrough.py
```

先不要运行脚本。打开 corpus，找到下面四个 stable source：

```text
rag-security
rag-evaluation
finetuning-basics
tenant-b-secret
```

## 第一步：先预测授权结果

请求 A 是：

```text
query = RAG 为什么要先做 ACL 权限过滤
tenant = tenant-a
principals = [engineering]
```

先填写：

| Source | visible / blocked | 你的理由 |
|---|---|---|
| `rag-security` |  |  |
| `rag-evaluation` |  |  |
| `finetuning-basics` |  |  |
| `tenant-b-secret` |  |  |

再预测 BM25 top-3 中是否会出现：

- `tenant-b-secret`：它包含很多与 query 相同的关键词；
- `finetuning-basics`：它和 caller 同 tenant，但 ACL 是 `ml`；
- `rag-security`：它与 caller 同 tenant，且 ACL 命中 `engineering`。

关键词相似不能覆盖授权规则。

## 第二步：预测重排与上下文

打开 `reranker-scores.example.jsonl`。不要先看教材里的答案，只根据三条 score 填表：

| Chunk 摘要 | `document_id` | Recorded score | 预测 rerank rank |
|---|---|---:|---:|
| ACL 必须先于排序 | `chk_bd3e8a67…` | 0.95 |  |
| 一般引用与评测 | `chk_2371a63e…` | 0.10 |  |
| 引用不等于语义蕴含 | `chk_8d8a68a0…` | 0.70 |  |

fixture 是按 `document_id` 而不是按顺序匹配的：JSONL 里三行的顺序与最终 rank 无关，
score 才决定排序。这也是为什么每行都要带 `query_sha256` 和 `content_sha256`——
换了 query 或改了 chunk 内容，这份 score 就应当失效而不是被悄悄复用。

假设 top-k 为 2、预算充足，你预计 context 中有几个短 source ID？
两个 chunk 来自同一个 stable source，为什么仍要给它们不同的 `S1/S2`？

答案是：短 ID 指向本次 Prompt 中的具体 chunk；stable source 用于跨 chunk 聚合和版本管理。

## 第三步：运行完整 walkthrough

~~~powershell
python projects/rag-foundations/rag_request_walkthrough.py
~~~

输出包含两个请求。先只看 `request-a-answerable`，按顺序找到：

```text
trusted_security_context
retrieval.candidates
rerank.results
packing.source_map
answer
citation
final
```

关键观察应为：

```text
BM25 ranks:        rag-security, rag-evaluation, rag-security
rerank top-2:      rag-security, rag-security
source map:        S1 -> rag-security, S2 -> rag-security
answer coverage:   1.0
final action:      answer
```

输出里的两个 `rag-security` 不是重复行。它们是同一 source 下的不同 chunk：
第一段给出 ACL 顺序，第二段限制引用的证明能力。

### 解释最终答案

逐字抽取器发出：

```text
检索必须先执行租户隔离和 ACL 权限过滤，再进行排序与上下文构建。 [S1]
```

请把它拆成三条独立结论：

1. 字符 offset 与 source 原文一致，所以 exact-span provenance 通过。
2. `[S1]` 位于本次 canonical source map，所以 citation syntax 通过。
3. 来源真实性、答案完整性和一般语义蕴含没有因此被证明。

第三条是本实验最重要的边界。

## 第四步：去掉 principal，观察安全负例

第三步跑的是 walkthrough 脚本，它内部固定带上 `engineering`。这一步改用 CLI 手工发同一个查询，
并且**故意不传** `--principal`，看看少了 principal 之后可见集合会怎么缩小：

~~~powershell
python -m about_llm.rag.cli retrieve `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --top-k 3
~~~

预期只剩 `rag-evaluation` 的一个主题相关 chunk。`rag-security` 不应出现在：

- `retrieved`；
- rendered context；
- `S1` 映射；
- score 计算的可见集合。

不要把这个结果解释为“匿名用户的问题不相关”。问题没有变，改变的是授权主体。

运行对应的安全回归：

~~~powershell
python -m pytest `
  tests/test_rag.py::test_bm25_filters_principal_acl_before_ranking `
  tests/test_rag.py::test_hidden_documents_cannot_change_visible_bm25_scores `
  -q
~~~

第一个测试检查越权结果不返回；第二个测试更严格，检查加入隐藏文档后可见 score 也不变化。

## 第五步：解释“有结果仍拒答”

回到 walkthrough 的 `request-b-no-answer`：

```text
query = 引用的 Kubernetes 灾难恢复步骤是什么
```

先预测：由于 corpus 中出现“引用”和“拒答”，BM25 会不会返回结果？
再预测：这些结果是否包含 Kubernetes 灾备步骤？

实际输出应显示：

```text
retrieval candidate count = 3
meaningful query tokens    = 9
covered query tokens       = 2
coverage                   = 2 / 9
final action               = abstain
```

想自己数一遍 `meaningful query tokens = 9` 的话，先记住分词规则：中文按**单字**切分，不是按词。
所以「灾备」是两个 token，不是一个。这也是本实验只用 coverage 做粗粒度拒答信号、不用它衡量语义相关性的原因之一。

这个 case 刻意区分三件事：

| 信号 | 本例结果 | 它能否决定 answerability |
|---|---|---|
| 检索是否非空 | 是 | 不能 |
| 是否主题相关 | 部分相关 | 不能 |
| 是否覆盖所需事实 | 否 | 应拒答 |

`0.55` 只是这个固定样例使用的 lexical threshold，不是生产默认值。
真实阈值需要在独立 calibration split 上比较 coverage 与 accepted-answer risk。

## 第六步：破坏一个 rerank 绑定

Recorded score 同时绑定 query hash、document ID、content hash 和 scorer identity。
下面的现成负例会把 query 改掉，但继续使用旧 score：

~~~powershell
python -m pytest `
  tests/test_rag_cli.py::test_recorded_rerank_cli_rejects_stale_query_binding `
  -q
~~~

它应该在排序前失败。若系统悄悄复用旧分数，rerank artifact 已经不能说明本次 query 的任何事情。

再运行 scorer 前的授权负例：

~~~powershell
python -m pytest `
  tests/test_rag_reranking.py::test_reranker_filters_tenant_and_acl_before_scorer_call `
  -q
~~~

这个测试关心 scorer 实际收到的候选，不只关心最终返回的 ID。

## 第七步：比较语法门禁与语义判断

考虑下面的输出：

```text
月球由奶酪构成。[S1]
```

如果 `S1` 是已授权且存在的 source，citation syntax 可能通过；
但 source 并不支持这句话，citation correctness 必须失败。

本地引用测试只应声称自己检查的内容：

~~~powershell
python -m pytest tests/test_rag_citations.py -q
~~~

读测试名称时把 “valid citation” 翻译成“ID 与段落语法合法”，不要翻译成“答案真实”。

## 实验记录模板

提交以下内容即可：

```text
请求：query、tenant、principals
运行前预测：visible sources、BM25 top-3、rerank top-2、final action
运行观察：candidate、source map、coverage、citation 和 reason code
权限负例：去掉 engineering 后，哪些内容消失
拒答负例：为什么 non-empty retrieval 仍 abstain
绑定负例：旧 query score 是否在排序前被拒绝
证据边界：本实验没有证明哪些模型、质量、性能和生产结论
```

下一步进入[召回、混合检索与重排](../../applications/rag-retrieval.md)，
把本实验的 BM25 与预先记录的 score 替换成可独立评测的 dense retriever 和 learned reranker。
