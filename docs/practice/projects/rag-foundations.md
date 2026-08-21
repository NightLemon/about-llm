# RAG Foundations：从一次问答到可审计服务

**项目导航**：[返回项目索引](../project-index.md) ·
[RAG 总览](../../applications/rag.md) ·
[请求生命周期](../../applications/rag-request-lifecycle.md) ·
[实验 5](../labs/lab-5-rag-request.md) ·
[证据页](../../evidence/rag-answer-controls.md)
{ .doc-nav }

这个项目不是“用框架把 PDF 接到聊天模型”。它用小而透明的组件回答五个工程问题：

1. 当前调用者能看哪些资料？
2. 答案证据在哪一层进入或离开候选集？
3. Context 和 citation 怎样绑定到精确 source？
4. 有结果但没答案时，系统怎样拒答？
5. 一次结果能支持什么结论，不能支持什么结论？

## 你会构建什么

```mermaid
flowchart LR
  C["Versioned corpus"] --> I["Markdown split"]
  I --> R["Authorized BM25"]
  R --> K["Rerank"]
  K --> P["Context packing"]
  P --> A["Extractive / model answer"]
  A --> G["Citation + publication gate"]
  G --> E["Trace + evaluation"]
```

项目同时提供一条离线教材路径和三条进阶路径：

| 路径 | 先解决什么 | 是否需要模型/GPU |
|---|---|---|
| 请求 walkthrough | 权限、召回、重排、packing、引用、拒答 | 否 |
| 持久化与服务 | 版本、删除、备份、认证、deadline | 否 |
| 目标 tokenizer packing | 完整 Prompt token 预算 | 需要 tokenizer，可离线 |
| 固定 Qwen 运行 | 真实模型的引用与拒答失败 | 需要本地约 1 GB snapshot，不需要 GPU |

第一次只完成请求 walkthrough。其余路径在你能解释 A/B 两个请求后再进入。

## 二十分钟最小路径 { #run }

安装基础依赖：

~~~powershell
python -m pip install -c constraints/ci.txt -e .
~~~

运行同一 corpus 上的一次 answer 与一次 abstain：

~~~powershell
python projects/rag-foundations/rag_request_walkthrough.py
~~~

不要从 JSON 第一行读到最后一行。按下面顺序观察：

```text
requests[0].trusted_security_context
requests[0].retrieval
requests[0].rerank
requests[0].packing.source_map
requests[0].answer
requests[0].citation
requests[0].final
```

请求 A 的关键结果是：

```text
query                 RAG 为什么要先做 ACL 权限过滤
BM25 stable sources   rag-security, rag-evaluation, rag-security
rerank top-2          rag-security, rag-security
source map            S1/S2 -> 两个不同 security chunks
coverage              1.0
final                  answer
```

请求 B 问 Kubernetes 灾备步骤。它仍有三个主题相关结果，但有效 query token 覆盖只有 `2/9`，
所以 final action 是 `abstain`。

这条路径没有执行 Embedding、learned reranker 或 LLM。
它先证明固定小语料上的控制流和 exact-span provenance。

## 先读懂四个输入文件

| 文件 | 用途 | 不要误解为 |
|---|---|---|
| `sample_corpus.jsonl` | 四份 versioned source 与 ACL | 代表性生产语料 |
| `sample_eval.jsonl` | Answerable/no-answer、qrels 与 security context | 独立人工 benchmark |
| `reranker-scores.example.jsonl` | Query/chunk 绑定的人工 score | Learned model 输出 |
| `sample_answers.jsonl` | Recorded action、claim 与 supplied verdict | 自动语义判断结果 |

打开 `sample_corpus.jsonl`，先预测 `tenant-a / engineering` 能看到哪些 source。
`tenant-b-secret` 即使关键词最多，也不能进入在线请求的查询期统计和排序。

## 选修：用可手算输入理解检索表示学习 { #retriever-learning-control }

想理解 dense retriever 的训练目标时，再运行：

~~~powershell
python projects/rag-foundations/retriever_learning_toy.py
python -m pytest tests/test_retriever_learning.py -q
~~~

它对 supplied embedding 精确计算单/多正例 InfoNCE、解析梯度、hard/false negative、
ColBERT-style MaxSim 与 SPLADE-style pooling。这个例子没有执行 encoder、ANN 或 GPU；
本仓库准备的 vectors 也不代表真实检索质量。推导见
[检索表示学习](../../applications/retrieval-learning.md)。

## 路径一：拆开观察一次请求

### 只看 authorization-first BM25

~~~powershell
python -m about_llm.rag.cli retrieve `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --top-k 3
~~~

输出同时给出 chunk rank、stable source、version、heading path 和 canonical `S1..Sn` context。

去掉 `--principal engineering` 再运行。Query 没变，`rag-security` 应消失；
不要把安全上下文变化误诊为相关性变化。

### 单独看 recorded rerank

~~~powershell
python -m about_llm.rag.cli rerank-recorded `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --scores projects/rag-foundations/reranker-scores.example.jsonl `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --candidate-k 3 `
  --top-k 2
~~~

Reranker 前会再次授权。Recorded score 精确绑定 query、chunk bytes 与 scorer identity；
改 query、改内容、缺分、多分或返回非有限数都会失败。

这个固定样例用于检查 query、chunk 和 score 的绑定能否正常工作。要声称质量提升，必须在 held-out qrels 上
比较排序与延迟。

### 单独看 packing

~~~powershell
python -m about_llm.rag.cli pack `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "ACL 引用" `
  --tenant tenant-a `
  --principal engineering `
  --candidate-k 20 `
  --budget-bytes 500 `
  --max-chunks-per-source 1
~~~

每个候选会得到：

```text
selected
duplicate_document
source_quota
budget
```

`budget-bytes` 使用 UTF-8 bytes，只用来直观演示 packing 取舍，不能称为模型 token budget。

### 单独看逐字答案

~~~powershell
python -m about_llm.rag.cli answer-extractive `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query-id acl-before-ranking `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering
~~~

Artifact 保存 exact offset、source/hash、coverage、packing decision 和 answer/abstain action。
API 不接收 qrels 或 `answerable` 标签，避免 gold 直接控制在线答案。

## 路径二：分别评价召回与回答 { #retrieval-reranking-metrics }

### Source-level retrieval evaluation

~~~powershell
python -m about_llm.rag.cli evaluate `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --top-k 3
~~~

报告把 answerable 与 no-answer 分开：

- Answerable：Recall、MRR、nDCG、Precision 与 all-evidence recall。
- No-answer：zero-result 只是检索信号，不等于最终拒答正确。
- Gold source：区分 visible、ACL blocked 与 missing from tenant corpus。
- 未标注结果：称为 unjudged，不自动视为 false positive。

### End-to-end extractive evaluation

~~~powershell
python -m about_llm.rag.cli evaluate-extractive `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl
~~~

固定五条 case 产生三次 answer、两次 abstain。全绿只说明这组人工编写的小数据没有回归，
不说明阈值、来源与语义适合真实业务。

### Recorded answer gate

~~~powershell
python -m about_llm.rag.cli evaluate-answers `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --answers projects/rag-foundations/sample_answers.jsonl
~~~

这条 gate 检查 case/output exact join、授权 context、claim citation 与 supplied judgment provenance。
`supported` verdict 由样例文件提供，不是评测器自动执行 entailment 得出的结论。

## 路径三：把 Prompt identity 也纳入 packing

Byte budget 不能回答目标模型是否超窗。使用目标 tokenizer：

~~~powershell
python -m about_llm.rag.cli pack-tokenized `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --candidate-k 20 `
  --max-total-tokens 4096 `
  --reserved-output-tokens 512 `
  --tokenizer C:\path\to\deployed-model-or-tokenizer `
  --tokenizer-revision exact-revision `
  --local-files-only `
  --system-prompt-file projects/rag-foundations/system-prompt.example.txt `
  --user-prompt-template-file projects/rag-foundations/user-prompt-template.example.txt
~~~

它为每个 prospective context 重新渲染完整 chat，保存 tokenizer/template identity、
最终 token IDs 与输出预留。

Caller 提供的 revision 与无密钥 hash 不认证实际模型来源；最大 context 也不能从 tokenizer 自动猜出。

## 路径四：版本化摄取与 SQLite

创建全新数据库：

~~~powershell
New-Item -ItemType Directory -Force artifacts/rag | Out-Null
python -m about_llm.rag.cli store-upsert `
  --database artifacts/rag/rag-demo.db `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --tenant tenant-a `
  --source-id rag-security `
  --expect-absent
~~~

从持久化 store 检索：

~~~powershell
python -m about_llm.rag.cli store-retrieve `
  --database artifacts/rag/rag-demo.db `
  --query "ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --top-k 3
~~~

更新与删除需要 expected current version。空抓取或空 split 不会被推断为“删除所有来源”。

~~~powershell
python -m about_llm.rag.cli store-delete `
  --database artifacts/rag/rag-demo.db `
  --tenant tenant-a `
  --source-id rag-security `
  --expected-current-version 1
~~~

SQLite reference 证明单库事务、版本冲突与授权读取，不证明远端 ANN、多副本或跨存储原子性。

## 路径五：备份与恢复演练

如果上一节已经删除 source，请使用新数据库重新 upsert。输出路径必须不存在：

~~~powershell
New-Item -ItemType Directory -Force artifacts/rag/backups | Out-Null
python -m about_llm.rag.cli store-backup `
  --database artifacts/rag/rag-demo.db `
  --backup artifacts/rag/backups/rag-snapshot.db `
  --manifest artifacts/rag/backups/rag-snapshot.manifest.json

python -m about_llm.rag.cli store-verify-backup `
  --backup artifacts/rag/backups/rag-snapshot.db `
  --manifest artifacts/rag/backups/rag-snapshot.manifest.json

python -m about_llm.rag.cli store-restore `
  --backup artifacts/rag/backups/rag-snapshot.db `
  --manifest artifacts/rag/backups/rag-snapshot.manifest.json `
  --database artifacts/rag/rag-restored.db
~~~

Manifest 绑定文件 bytes、schema、row count 和 ordered logical fingerprint。
它无签名、未加密，也不包含远端 index、cache 和流量切换，所以不能声称完成灾备。

## 路径六：本地 Persistent ASGI service

先准备 SQLite，再为 loopback demo 注入 token：

~~~powershell
$env:ABOUT_LLM_RAG_DEMO_TOKEN = "replace-with-a-long-local-demo-token"
python projects/rag-foundations/serve_extractive.py `
  --database artifacts/rag/rag-demo.db `
  --tenant tenant-a `
  --subject local-demo-user `
  --principal engineering
~~~

Request body 不能自报 tenant/principal；`/health/live` 与 `/health/ready` 含义不同。
认证作为受保护 route 的 dependency 执行；Host 使用显式 allowlist，实际 ASGI body 默认限制为 64 KiB，
Response 使用 server request ID、`Cache-Control: no-store`、安全响应头和 public allowlist。
生产部署仍需在反向代理设置 body/rate limit、TLS 与真实 IAM；应用内上限不能替代 edge admission control。

运行一个不打开 TCP socket 的固定验证程序：

~~~powershell
python projects/rag-foundations/rag_service_control.py
~~~

它真实执行 FastAPI/Starlette/HTTPX ASGI dispatch 与 SQLite reopen。
它没有 TLS、JWT/OAuth、多 worker 全局容量或远端取消证据。

## 路径七：保留真实模型的第一次失败

本地已有固定 Qwen snapshot 时运行：

~~~powershell
python projects/rag-foundations/run_qwen_rag_control.py --local-files-only
python -m pytest tests/test_rag_transformers_control.py -q
~~~

固定结果不是漂亮 Demo：

- Answerable case 复述证据却漏引。
- Empty-context case 仍编造 Kubernetes 灾备步骤。
- 行为 gate 为 `0/2`。

接着比较两种不同证据：

~~~powershell
python projects/rag-foundations/replay_qwen_rag_publication_policy.py `
  --verify projects/rag-foundations/qwen2.5-0.5b-rag.publication-policy-replay.json

python projects/rag-foundations/run_qwen_guarded_rag_control.py --local-files-only
~~~

第一条是 counterfactual replay，没有观察 guard 当时包裹模型；
第二条真实包裹 `GenerationMixin.generate()`，但只运行两个本仓库编写的 case，也没有 GPU/vLLM 证据。

## Generation trace 与 citation audit

~~~powershell
python -m about_llm.rag.cli audit-traces `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --answers projects/rag-foundations/sample_answers.jsonl `
  --traces projects/rag-foundations/generation-traces.example.jsonl

python -m about_llm.rag.cli audit `
  --answer-file projects/rag-foundations/sample_answer.md `
  --source-id S1 `
  --source-id S2
~~~

Trace 绑定 query/security、chunk/version/content、rendered context、Prompt 和 raw output identity。
这里使用手写协议样例。Unsigned hash 可以发现内容变化，但不能认证模型执行者，也不会执行语义蕴含判断。

## 代码阅读顺序

| 读者问题 | 文件 |
|---|---|
| 文档怎样切成稳定 chunk？ | `src/about_llm/rag/ingestion.py` |
| ACL 怎样影响 BM25 统计？ | `src/about_llm/rag/bm25.py` |
| Scorer 前怎样二次授权？ | `src/about_llm/rag/reranking.py` |
| Budget 和 source quota 怎样决定 context？ | `src/about_llm/rag/context_packing.py` |
| Exact span 与 abstain 怎样形成？ | `src/about_llm/rag/extractive.py` |
| Citation syntax 检查什么？ | `src/about_llm/rag/citations.py` |
| Publish/abstain/reject 怎样分开？ | `src/about_llm/rag/generation_policy.py` |
| SQLite 版本与删除怎样实现？ | `src/about_llm/rag/sqlite_store.py` |

按请求流向读，不要先逐文件通读全部测试。

## 最小测试门禁

~~~powershell
python -m pytest `
  tests/test_rag.py `
  tests/test_rag_reranking.py `
  tests/test_rag_context_packing.py `
  tests/test_rag_citations.py `
  tests/test_rag_extractive.py `
  tests/test_rag_answer_eval.py `
  tests/test_rag_generation_policy.py `
  tests/test_rag_request_walkthrough.py `
  -q
~~~

完整专项还包括 ingestion、SQLite、backup、service、trace 与 fixed-Qwen controls。
测试按 claim 解释，不以总用例数代表教材或系统完全正确。

## 项目交付物

不要只提交终端截图。一个可评审的结果应包含：

```text
1. 请求 A/B 的阶段表和最终 action
2. 去掉 engineering principal 的权限负例
3. 一个 query/content binding 破坏实验
4. Retrieval 与 answer 指标的分层报告
5. Corpus/index/model/policy identity
6. 一条失败 trace 的第一个错误环节
7. 已证明、未证明和目标环境待验证清单
```

可以写进简历的是具体证据，例如：

> 构建 authorization-first RAG reference，串联 versioned corpus、BM25、rerank、
> token-aware packing、exact-span citation 与 typed refusal；用跨 tenant、non-empty no-answer
> 和 stale-score 负例验证控制边界。

不要写“彻底解决幻觉”或“生产级零泄漏”。这些本地固定样例无法支持这类结论。
