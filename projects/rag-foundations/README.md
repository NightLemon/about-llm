# RAG Foundations：从一次问答到可审计服务

这个项目用同一条问题串起 RAG 的完整控制流：可信身份进入请求，权限过滤先于检索，候选经过重排和上下文装箱，
生成结果必须带可核对引用；证据不足时，系统明确拒答。

第一次学习请从[项目教学页](../../docs/practice/projects/rag-foundations.md)开始。那里逐步解释一次回答和一次拒答；
本页只保留快速运行、脚本索引和排错信息。

## 二十分钟最小路径 { #run }

```powershell
python -m pip install -c constraints/ci.txt -e .
python projects/rag-foundations/rag_request_walkthrough.py
```

不要从输出 JSON 第一行读到最后一行。按这个顺序观察第一条请求：

```text
trusted security context
→ authorization-first retrieval
→ rerank
→ context packing / source map
→ exact-span answer
→ citation audit
→ final action
```

请求 A 问“RAG 为什么要先做 ACL 权限过滤”。系统只对授权文档评分，重排后把两个不同的 security chunk 映射为
`S1/S2`，最后给出带引用的回答。

请求 B 问知识库没有覆盖的 Kubernetes 灾备步骤。检索仍会返回主题相近文档，但有效 query token 覆盖只有 `2/9`，
因此最终动作为 `abstain`。这个反例说明“检索到了内容”和“证据足以回答”是两件事。

这条 walkthrough 不调用 embedding model、learned reranker 或 LLM。它先让权限、排序、装箱、引用和拒答逻辑变得
可手算。完整逐步解释见[二十分钟最小路径](../../docs/practice/projects/rag-foundations.md#run)。

## 一次请求中各层负责什么

| 层次 | 输入 | 关键问题 | 输出 |
|---|---|---|---|
| 身份与 ACL | Tenant、principal、policy revision | 哪些文档允许进入候选集？ | 授权候选 |
| Retrieval | Query 与授权候选 | 相关文档是否进入 top-k？ | Ranked chunks |
| Rerank | Query、候选和 scorer identity | 更贵的排序是否真的改善 held-out qrels？ | Reordered chunks |
| Packing | Prompt、token budget、来源约束 | 哪些证据真正传给模型？ | Context 与 source map |
| Generation | Prompt 与 context | 回答、拒答还是错误？ | Raw output |
| Citation/verification | Output、source map、业务规则 | 引用存在吗，claim 是否得到支持？ | Publish、reject 或 abstain |

权限过滤必须发生在评分之前，并在 rerank、packing 和引用阶段继续保持同一安全身份。引用 ID 合法只证明来源映射正确；
claim 是否得到语义支持，还需要独立评测或人工审查。

## 拆开运行一次问答

只看授权后的 BM25：

```powershell
python -m about_llm.rag.cli retrieve `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --top-k 3
```

去掉 `--principal engineering` 再运行。Query 没变，但受限的 `rag-security` 应当消失；这是安全上下文变化，
不是检索质量突然下降。

再把检索、装箱、逐字证据和引用串起来：

```powershell
python -m about_llm.rag.cli answer-extractive `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query-id acl-before-ranking `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering
```

这个 baseline 只从已经装入上下文的 chunk 复制原文片段，并附上来源 ID。它适合检查 provenance 和控制流，
不负责自然语言改写，也不能证明来源本身真实或权威。

## 分别评价检索和回答

Source-level 检索评测：

```powershell
python -m about_llm.rag.cli evaluate `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --top-k 3
```

报告把“有答案”和“知识库无答案”分开。对于有答案问题，几个指标回答不同问题：

| 指标 | 在这里检查什么 |
|---|---|
| Recall@k | 前 k 个结果找回了多少比例的标注相关来源 |
| MRR | 第一份相关来源排得有多靠前 |
| nDCG | 带不同相关性等级的来源是否排在合理位置 |
| Precision@k | 返回结果中有多少属于标注来源 |
| All-evidence recall | 有多少问题在前 k 个结果中找齐了全部必需来源 |

对于无答案问题，检索到零条结果只是一个信号。系统还要评价最终是否正确拒答，以及有主题相关文本时会不会越界作答。

端到端 exact-span 评测：

```powershell
python -m about_llm.rag.cli evaluate-extractive `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl
```

固定五条 case 产生三次回答和两次拒答。结果全绿表示这组教学数据没有回归，不表示阈值或语料适用于真实业务。

## 根据当前问题选择入口

| 你想理解什么 | 入口 |
|---|---|
| 一次授权问答怎样变成 answer/abstain | `rag_request_walkthrough.py` |
| 一份来源怎样切片、更新并按 ACL 读取 | `rag_ingestion_walkthrough.py` |
| BM25、ACL 与来源编号 | `rag.cli retrieve` |
| Dense retriever 的 InfoNCE、negative 与 MaxSim | `retriever_learning_toy.py` |
| Recorded reranker 怎样绑定 query 和 chunk | `rag.cli rerank-recorded` |
| Byte/token budget、来源配额与 prompt identity | `rag.cli pack`、`rag.cli pack-tokenized` |
| Exact-span 回答与引用 | `rag.cli answer-extractive`、`rag.cli audit` |
| Recall/nDCG 与拒答如何分开评测 | `rag.cli evaluate`、`rag.cli evaluate-extractive` |
| Recorded claims 与 supplied verdict 怎样进入 gate | `rag.cli evaluate-answers`、`rag.cli audit-traces` |
| 自己操作版本化摄取、删除与 SQLite | `rag.cli store-upsert/store-retrieve/store-delete` |
| 备份、校验与恢复 | `rag.cli store-backup/store-verify-backup/store-restore` |
| Localhost ASGI 服务、认证和背压 | `serve_extractive.py`、`rag_service_control.py` |
| 固定 Qwen 的真实生成失败 | `run_qwen_rag_control.py` |
| 无证据预拒答与有证据后引用检查 | `replay_qwen_rag_publication_policy.py`、`run_qwen_guarded_rag_control.py` |

每条入口的完整命令、预期现象和完成信号见[项目教学页](../../docs/practice/projects/rag-foundations.md)。精确运行结果与
适用范围保存在[RAG 回答证据页](../../docs/evidence/rag-answer-controls.md)，不要把固定样例分数写成线上质量结论。

## 版本化摄取与本地服务

创建一份新的 SQLite 数据库，并写入一个明确版本的来源：

```powershell
New-Item -ItemType Directory -Force artifacts/rag | Out-Null
python -m about_llm.rag.cli store-upsert `
  --database artifacts/rag/rag-demo.db `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --tenant tenant-a `
  --source-id rag-security `
  --expect-absent
```

更新和删除必须携带读取到的当前版本。空抓取或空切分不会被自动解释成“删除整个来源”。从数据库检索：

```powershell
python -m about_llm.rag.cli store-retrieve `
  --database artifacts/rag/rag-demo.db `
  --query "ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --top-k 3
```

要运行 localhost API，先安装 `.[api]`，再从环境变量提供 demo token：

```powershell
python -m pip install -e ".[api]"
$env:ABOUT_LLM_RAG_DEMO_TOKEN = "replace-with-a-long-local-demo-token"
python projects/rag-foundations/serve_extractive.py `
  --database artifacts/rag/rag-demo.db `
  --tenant tenant-a `
  --subject local-demo-user `
  --principal engineering
```

租户和权限主体来自已经认证的会话，客户端请求正文不能自行声明这些身份。这个示例服务只监听本机回环地址，适合
观察请求路径。

真实部署还要增加 TLS、JWT 或 IAM 身份验证、跨进程并发限制、可取消的下游调用，以及受访问控制的 trace 存储。

## 固定 Qwen 为什么第一次会失败

仓库保留了一次 Qwen2.5-0.5B-Instruct 的真实 CPU 运行。它忠实记录了两个问题：

- 有证据的回答复述了正确内容，却没有输出 `[S1]`，因此引用检查失败。
- 空证据请求仍生成 Kubernetes 步骤，没有按要求拒答。

离线核对发布策略回放；这条命令也会严格验证它所引用的原始 Qwen 报告：

```powershell
python projects/rag-foundations/replay_qwen_rag_publication_policy.py `
  --verify projects/rag-foundations/qwen2.5-0.5b-rag.publication-policy-replay.json
```

这条回放验证模型外发布策略：空 context 在生成前直接拒答；非空 context 生成后必须通过引用检查才能发布。失败记录比
一份手写成功样例更有学习价值，因为它说明 Prompt 指令不能替代控制流和输出验证。

本机已有固定 snapshot 时，可以用 `run_qwen_rag_control.py --local-files-only` 重放真实 CPU 生成。

## 主要输入与输出

| 文件 | 用途 |
|---|---|
| `sample_corpus.jsonl` | 四份带版本、tenant 和 ACL 的教学来源 |
| `sample_eval.jsonl` | Answerable/no-answer、qrels 与安全上下文 |
| `reranker-scores.example.jsonl` | 与 query/chunk 绑定的人工重排分数 |
| `sample_answers.jsonl` | Recorded action、claim、citation 与 supplied verdict |
| `generation-traces.example.jsonl` | Packing、prompt、raw output 与答案身份的固定轨迹 |
| `system-prompt.example.txt`、`user-prompt-template.example.txt` | Tokenized packing 所使用的 Prompt 身份 |
| `*.recorded-report.json` | 固定 Qwen 运行结果，可在没有权重时离线核对 |
| `artifacts/rag/` | 本机数据库、备份、trace 和评测报告 |

Corpus、trace 和数据库都可能包含授权正文与身份信息。真实项目应把它们放入受控 artifact store，而不是普通日志或
metrics label。

## 常见故障

| 现象 | 先检查 |
|---|---|
| 高相关私有文档出现在结果中 | ACL 是否在评分前执行，rerank/packing 是否再次校验安全身份 |
| 去掉 principal 后分数变化很大 | 候选集合已经变化，不要把它误诊为模型相关性漂移 |
| Recall 很高，回答仍然错误 | Required evidence 是否齐全，packing 是否丢弃关键来源，生成是否受支持 |
| 检索有结果却应该拒答 | 主题相关不等于证据足够；检查 coverage、required sources 和 no-answer policy |
| 引用 ID 合法，但 claim 不被支持 | Citation syntax 与 semantic entailment 是两项独立评测 |
| Token budget 经常超限 | 使用目标 tokenizer 渲染完整 Prompt，并预留输出 token |
| 更新后旧 chunk 仍可检索 | Source version、stale delete 和索引事务是否一起提交 |
| 备份能打开但内容不一致 | 同时检查文件 hash、schema、row fingerprint 和逻辑不变量 |
| Client 超时后线程仍占容量 | HTTP timeout 不会强制终止同步 worker；检查协作取消或进程隔离 |
| 模型没有证据仍被调用 | Pre-generation policy 是否在 generator callback 之前执行 |

## 运行检查

```powershell
python -m pytest `
  tests/test_rag.py `
  tests/test_rag_ingestion.py `
  tests/test_rag_sqlite_store.py `
  tests/test_rag_citations.py `
  tests/test_rag_reranking.py `
  tests/test_rag_cli.py `
  tests/test_rag_answer_eval.py `
  tests/test_rag_trace.py `
  tests/test_rag_generation_policy.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

默认测试使用固定小语料和本地实现，不调用 embedding provider、远程 vector database 或真实 LLM。目标模型质量、
GPU 性能、阈值校准和生产安全必须在固定业务数据、身份系统和 workload 上重新验证。
