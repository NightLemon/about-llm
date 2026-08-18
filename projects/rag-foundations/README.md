# RAG Foundations

这是 RAG 项目的开发者运行手册。第一次进入时，不要先读完整个文件，也不要先启动模型。

从仓库根目录运行：

~~~powershell
python -m pip install -c constraints/ci.txt -e .
python projects/rag-foundations/rag_request_walkthrough.py
~~~

输出会用同一套 corpus 展示两个请求：

- 请求 A 找到授权证据，经 BM25、recorded rerank 和 packing 后输出带 `[S1]` 的 exact span。
- 请求 B 有三个主题相关结果，却因为没有 Kubernetes 灾备步骤而 abstain。

先按 `security context → retrieval → rerank → packing → answer → citation → final` 阅读 JSON。
完整讲解见[请求生命周期](../../docs/applications/rag-request-lifecycle.md)，动手步骤见
[实验 5](../../docs/practice/labs/lab-5-rag-request.md)，精确 oracle 见
[RAG 证据页](../../docs/evidence/rag-answer-controls.md)。

## 能力地图：需要时再查

- 一条可运行 walkthrough，把 answerable/no-answer 请求串过授权、召回、重排、packing、引用和拒答；
- UTF-8/中英文透明 lexical tokenizer；
- BM25，包含长度归一化和稳定排序；
- 检索前 tenant ACL 过滤；
- Reciprocal Rank Fusion；
- 注入式 dense cosine index；
- 单/多正例 InfoNCE exact control、analytic embedding gradients、hard/false-negative 反事实、ColBERT-style MaxSim 与 SPLADE-style pooling；
- authorization-first rerank core、严格 recorded-score artifact 与复用该核心的 sentence-transformers cross-encoder adapter；
- source-level Recall@k、MRR@k、nDCG@k、Precision@k 与 all-evidence recall@k；
- graded relevance、必须同时出现的多证据集合、显式无答案 query 与零结果诊断；
- recorded answer/abstain/error 工件、atomic claim、外部 judgment provenance 与保守 grounding gate；
- Markdown 标题感知切分、超长段落兜底、稳定内容哈希与 chunk id；
- tenant/source/version/ACL 元数据和显式 upsert/delete 增量计划；
- SQLite schema-v1 持久 chunk store：`BEGIN IMMEDIATE`、expected-current-version optimistic concurrency、同版本内容复用拒绝、显式 source delete、事务回滚与 ACL-before-ranking 读取；
- 可注入完整 prompt cost 的 greedy context packer、去重、per-source quota 与逐候选决策账本；
- 目标 tokenizer/chat template 的完整 prospective prompt 重计数、输出 token 预留、模板 identity 与最终 token IDs；
- packing→raw output→recorded answer 的 generation trace：query/security binding、逐 chunk version/content hash、canonical context、prompt/output identity 与严格审计；
- 固定 Qwen2.5-0.5B-Instruct 真实权重的 retrieval→packing→greedy generation control：逐步 logits、GenerationMixin 对照，以及不修补的漏引/拒答失败记录；
- 端到端非 LLM extractive answer baseline：授权检索、byte-budget packing、逐字 span、短引用、lexical coverage 拒答和独立 artifact fingerprint；
- persistent extractive FastAPI/ASGI service：body 外身份解析、每请求 SQLite 重开、ACL-before-score、closed request schema、结构化错误、readiness、request id、并发/排队/执行 deadline 与完整 artifact response；
- 授权上下文的规范化 `[S1]` 来源编号、未知引用和漏引段落审计；
- 单元测试覆盖精确术语、租户隔离、稳定 ID、编辑/删除、重复结果和指标。

## 可运行 CLI

先安装仓库本体：

~~~powershell
python -m pip install -e .
~~~

`sample_corpus.jsonl` 含版本化 Markdown、tenant、principal ACL 和元数据。下面命令执行切分、BM25、检索前权限过滤、来源编号与引用上下文构建：

~~~powershell
python -m about_llm.rag.cli retrieve `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --top-k 3
~~~

输出 JSON 保留 rank、score、稳定 chunk id、source/version、heading path、授权后的 `<source>` 上下文和 `[S1]` 来源映射。`tenant-b-secret` 即使包含高相关词也不会进入 tenant-a 候选；没有 `--principal engineering` 时，受限的 `rag-security` 也不会被评分。空 ACL 表示租户内公开，非空 ACL 需要与调用者 principals 至少匹配一项。

### 检索表示学习 exact control

~~~powershell
python projects/rag-foundations/retriever_learning_toy.py
python -m pytest tests/test_retriever_learning.py -q
~~~

该 CPU toy 对 supplied embeddings 精确计算单/多正例 InfoNCE、query/document gradients，并比较 easy negative、hard negative、漏标相关文档与 multi-positive mask；同一脚本还隔离验证 ColBERT-style MaxSim 和 SPLADE-style max pooling。它没有运行 encoder、训练 checkpoint 或建立 ANN index，不能证明 authored vectors、labels 或结果代表真实检索质量。完整推导见[检索表示学习](../../docs/applications/retrieval-learning.md)。

### 端到端 exact-span 回答基线

`answer-extractive` 把授权 BM25、`pack_citation_context` 和答案动作真正串起来。它只从 packed chunk 中按字符偏移复制句子/分句，在原文后附 `[S1]`；artifact 会保存 query/security fingerprint、检索 rank/score、source version/content hash、每个 packing decision、原始 context、span offset、distinct lexical token coverage、最终动作和统一 `RecordedAnswer`。被 budget 丢弃或未授权的 chunk 不可能提供 span。

~~~powershell
python -m about_llm.rag.cli answer-extractive `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query-id metrics-and-entailment `
  --query "RAG 为什么既要评测 Recall nDCG，又不能把合法引用当成语义蕴含" `
  --tenant tenant-a `
  --principal engineering
~~~

默认策略从 query 的去重 lexical tokens 中移除一小组固定停用 token；每个候选 span 至少命中 2 个 token，greedy set coverage 达到 0.55 才回答，否则输出明确 abstain。这个阈值只是透明 fixture policy，不是从测试集校准出的生产阈值。`proposed_spans` 在拒答案例中仅记录 gate 前的最佳证据尝试，不会进入 `answer_text` 或 factual claims。

离线 harness 先生成全部 artifact，再使用 qrels/`answerable` 标签评测；生成 API 本身不接受 relevance、required source 或 expected action。当前 5 条 authored fixture 得到 3 次回答、2 次拒答，action accuracy、grounded-answer pass rate 和 recorded gate pass rate 均为 1.0；这只是固定小语料上的回归结果：

~~~powershell
python -m about_llm.rag.cli evaluate-extractive `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl
~~~

该 baseline 没有调用 LLM，也没有执行 semantic entailment。它能机械证明 claim 是某个已授权 packed chunk 的 exact substring，因此 `judgment_source` 标记为 `deterministic-exact-source-span-v1`；它不能证明来源真实/权威、query 与 span 语义相关、答案完整、拒答阈值校准或自然语言生成质量。UTF-8 byte budget 也不冒充目标模型 token budget；接真实生成器时仍应使用 `pack-tokenized` 和 generation trace。

### Recorded-score 重排控制实验

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

核心会在 scorer 前重新检查 tenant/ACL，拒绝重复 document id、非连续 candidate rank、非有限首阶段/重排分数，并以原 candidate rank 处理 score tie。Recorded artifact 精确绑定 query SHA-256、chunk id、content SHA-256 和 scorer identity；缺分、余分、旧 query 或旧内容都 fail closed。示例分数是作者构造的 plumbing fixture，没有执行 learned model、目标 tokenizer/truncation 或质量/延迟评测；`authored-reranker-fixture@v1` 只是未认证标签。真实 CrossEncoder 可通过 adapter 接入同一核心，但仍须固定 revision 并做 held-out qrels 消融。

### 持久化摄取、读取与删除

内存 CLI 之外，同一入口提供单机 SQLite reference workflow。首次创建必须显式声明 source 应当不存在；命令从 corpus 中按 tenant/source 精确选择一条记录，不会把整份 JSONL 隐式批量写入：

~~~powershell
python -m about_llm.rag.cli store-upsert `
  --database .\rag-demo.db `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --tenant tenant-a `
  --source-id rag-security `
  --expect-absent
~~~

更新同一 source 时换用新 version 的 corpus，并显式携带读取到的当前 version；陈旧 caller 会失败，而不会覆盖新内容：

~~~powershell
python -m about_llm.rag.cli store-upsert `
  --database .\rag-demo.db `
  --corpus C:\path\to\corpus-with-new-rag-security.jsonl `
  --tenant tenant-a `
  --source-id rag-security `
  --expected-current-version 1
~~~

从持久 store 查询时，SQLite 先按 tenant 取行，进程再按 principal ACL 缩小候选，然后才构建内存 BM25；输出的 `authorized_candidate_count` 是授权后 chunk 数，不是全库数量：

~~~powershell
python -m about_llm.rag.cli store-retrieve `
  --database .\rag-demo.db `
  --query "ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --top-k 3
~~~

删除不会由空抓取或空切分推断，只能显式执行并携带当前 version：

~~~powershell
python -m about_llm.rag.cli store-delete `
  --database .\rag-demo.db `
  --tenant tenant-a `
  --source-id rag-security `
  --expected-current-version 1
~~~

三类命令输出 machine-readable JSON；版本冲突、重复 JSON key、`NaN/Infinity`、浮点溢出、无匹配 source 或数据库错误返回 exit code 2。`store-retrieve` 仍只是授权后的 lexical BM25：它不做 embedding/ANN，也没有证明跨进程服务治理或远端 vector index 与 manifest 的原子性。示例数据库包含原文与 ACL，应按敏感数据管理，不要提交到版本库；本地 backup/restore 的独立证据与限制见下一节。

### 可验证备份与恢复演练

先创建一个权限受控的备份目录；backup 和 manifest 输出路径都必须不存在，命令绝不会覆盖旧工件：

~~~powershell
New-Item -ItemType Directory -Force .\rag-backups
python -m about_llm.rag.cli store-backup `
  --database .\rag-demo.db `
  --backup .\rag-backups\rag-20260806.db `
  --manifest .\rag-backups\rag-20260806.manifest.json
~~~

该命令使用 SQLite online backup API 生成一致数据库快照，随后检查 `quick_check`、foreign keys、schema-v1 的精确 table/index/trigger 集合，以及 source/chunk 的逻辑不变量。manifest 同时绑定物理文件 byte size/SHA-256、source/chunk 数和与 SQLite page layout 无关的有序 row fingerprint；原数据库在快照后继续更新，不会改变已发布 backup。

恢复前应单独验证：

~~~powershell
python -m about_llm.rag.cli store-verify-backup `
  --backup .\rag-backups\rag-20260806.db `
  --manifest .\rag-backups\rag-20260806.manifest.json
~~~

恢复只能写入一个**尚不存在**的新数据库路径，不会原地覆盖线上库：

~~~powershell
python -m about_llm.rag.cli store-restore `
  --backup .\rag-backups\rag-20260806.db `
  --manifest .\rag-backups\rag-20260806.manifest.json `
  --database .\rag-restored.db
~~~

自动测试验证快照完成后源库更新不改变 backup、恢复后 source/version/text、输出不覆盖、物理篡改、strict manifest、内容 hash 漂移和未版本化 trigger 注入。这个闭环仍不等于完整 disaster recovery：manifest fingerprint 没有签名或 MAC，攻击者若能同时改写 DB、manifest 和无密钥 hash，来源仍不可认证；文件没有由工具加密，`created_at_utc` 也是本机自报时间。`fsync` 单文件不证明断电后的目录项 durability；一次 tiny fixture 恢复不证明目标 RPO/RTO。它也不包含远端 vector index、object store、cache、trace 或删除传播，恢复后仍需授权抽样、真实 query、容量测试和显式流量切换。

### Persistent extractive ASGI service

安装显式固定兼容范围的 FastAPI/Starlette/Uvicorn stack：

~~~powershell
python -m pip install -e ".[api]"
~~~

先用前述 `store-upsert` 创建 SQLite 数据库。演示 server 只从环境变量取 bearer token，不允许把 token 写进命令行；默认只绑定 loopback：

~~~powershell
$env:ABOUT_LLM_RAG_DEMO_TOKEN = "replace-with-a-long-local-demo-token"
python projects/rag-foundations/serve_extractive.py `
  --database .\rag-demo.db `
  --tenant tenant-a `
  --subject local-demo-user `
  --principal engineering
~~~

请求 body 没有 `tenant_id` 或 `principals` 字段；它们只能由注入的 `AuthResolver` 生成。多余字段会被 closed Pydantic schema 以 422 拒绝：

~~~powershell
$headers = @{ Authorization = "Bearer $env:ABOUT_LLM_RAG_DEMO_TOKEN" }
$body = @{
  query_id = "demo-q1"
  query = "RAG 为什么要先做 ACL 权限过滤"
  top_k = 5
  budget_units = 12000
} | ConvertTo-Json
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/rag/query `
  -Headers $headers `
  -ContentType application/json `
  -Body $body
~~~

`/health/live` 只证明进程可应答；`/health/ready` 会重新打开并校验现有 SQLite schema。查询路径每次以解析后的 tenant/principals 调用 `visible_chunks`，只把授权行转成 `Document`，然后 BM25 再做一次 tenant/ACL 检查。响应带 server-generated `X-Request-ID`、answer/citations、ordered retrieved IDs、完整 extractive artifact/fingerprint 和 `Cache-Control: no-store`。完整 artifact 仍含授权正文与身份上下文，应进入受控 trace store，不能写入普通 metrics label。

服务有 `max_concurrency`、queue timeout 和 execution timeout。同步 SQLite/BM25 在 worker thread 中运行；HTTP 504 只表示 caller 不再等待，不能杀死 Python thread。实现因此在超时或 client cancellation 后继续占用 semaphore permit，直到后台 work 真正终止，避免表面上限 8 实际运行超过 8。这个策略保持容量账本诚实，但也意味着卡死线程会长期占位；生产系统需要 cooperative cancellation、进程隔离/回收和外部 deadline，而不能把 `wait_for` 当作强制终止。

`StaticBearerAuthResolver` 只适合 localhost demo/test。它没有验证 JWT signature、issuer/audience/expiry/revocation，也没有 TLS、reverse-proxy trust、集中 IAM 或 key rotation。脚本默认拒绝 non-loopback bind，除非显式承认风险；即使显式允许，也不会把 demo token 升级为生产认证。Uvicorn 固定单 worker，因为 semaphore 是进程内的；多个 workers/replicas 各有自己的上限，必须使用全局 admission control 才能声明服务总并发。

运行不打开 TCP socket 的可复现 ASGI control：

~~~powershell
python projects/rag-foundations/rag_service_control.py
~~~

它真实执行 FastAPI、Starlette、HTTPX ASGI dispatch 与 SQLite reopen；engineering/anonymous 分别只看到 2/1 个授权 source，body tenant injection 为 422，缺 credential 为 401。它不执行 TCP/TLS/reverse proxy/remote identity、learned retrieval/reranking、LLM、multi-process admission 或生产 SLO，因此项目仍不能仅凭这个 control 宣称完整 L3/L4。

在显式 UTF-8 byte 预算下演示 context packing：

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

CLI 输出每个候选的 `selected / duplicate_document / source_quota / budget` 决策，以及 prospective cost、最终 canonical `[S1]` 映射和剩余预算。它明确使用序列化 UTF-8 bytes，只用于依赖无关的 CPU 演示，**不是 model token 预算**。生产代码应调用 `pack_citation_context` 并注入目标 tokenizer + 完整 chat template 的 cost closure。

使用部署目标的 tokenizer 和 chat template 做真实 token 预算：

~~~powershell
python -m pip install -e ".[transformers]"
python -m about_llm.rag.cli pack-tokenized `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "RAG 为什么要先做 ACL 权限过滤" `
  --tenant tenant-a `
  --principal engineering `
  --candidate-k 20 `
  --max-total-tokens 4096 `
  --reserved-output-tokens 512 `
  --tokenizer C:\path\to\deployed-model-or-tokenizer `
  --tokenizer-revision exact-checkpoint-revision `
  --local-files-only `
  --system-prompt-file projects/rag-foundations/system-prompt.example.txt `
  --user-prompt-template-file projects/rag-foundations/user-prompt-template.example.txt `
  --max-chunks-per-source 2
~~~

命令对每个 prospective context 重新执行完整 system/user chat template，并把 `reserved_output_tokens` 加入总预算，而不是把分段 token 长度相加。报告绑定 Transformers 版本、tokenizer class/path/revision、词表大小、chat template/system/user-template SHA-256、逐候选总成本、最终 prompt token 数与 token IDs。若 checkpoint 没有 chat template，必须显式提供 `--chat-template-path`，工具不会猜格式。

`--tokenizer-revision` 对本地目录只是调用者提供的 identity label，hash 也没有密钥；二者都不认证来源。工具不会从 tokenizer 推断部署模型真正可用的 context window，因此 `max-total-tokens` 必须来自固定 model/runtime 配置并另行验证。最终 token IDs 和上下文本身可能泄露敏感内容，生产报告应进入受控 artifact store，不应作为普通 metrics label。

运行 source-level 检索评测：

~~~powershell
python -m about_llm.rag.cli evaluate `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --top-k 3
~~~

CLI 会先按 chunk 排序、再对 source 去重，避免同一长文档的多个 chunk 占满 source-level 指标。答案型 case 输出 Recall@k、MRR@k、nDCG@k、Precision@k 与 all-evidence recall@k；无答案 case 单独输出 zero-result accuracy，二者绝不共用分母。每条 query 还保留 retrieved、found/missing relevant、found/missing required、judged-nonrelevant、unjudged、tenant/principals 和 gold source 的 `visible / acl_blocked / missing_from_tenant_corpus` 状态。

这里的 Precision@k 分母是**实际返回并检查的前 k 个去重 source**，零结果定义为 0；不是固定除以 k。`all_evidence_recall_at_k` 是 query-level 完整集合命中率：只有全部 `required_source_ids` 都进入 top-k 才记 1。顶层旧字段 `recall_at_k` 等为兼容保留，`legacy_metric_scope` 明确它们只覆盖 answerable cases；新代码应读取 `answerable_metrics` 与 `no_answer_metrics` 中的分母。

样例故意包含两类无答案 query：完全无关 query 得到零结果，主题相近但知识库没有所问事实的 query 仍可能召回文档。因此 zero-result accuracy 只是检索层信号，不证明生成器正确拒答，也不覆盖“检索到主题相关但不能回答”的情况。无答案/拒答端到端评测仍需记录生成器的 `insufficient_evidence`、reference decision 和人工标签。

评测录制答案、拒答和 claim 工件：

~~~powershell
python -m about_llm.rag.cli evaluate-answers `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --answers projects/rag-foundations/sample_answers.jsonl
~~~

该命令对 case 与 output 做 exact join，把 `answer / abstain / error` 都保留在分母；从 corpus 重新检查每个 `context_source_id` 在 case tenant/principals 下是 `visible`、`acl_blocked` 还是 `missing_from_tenant_corpus`。答案 claim 的引用必须属于可见 context，且 supplied verdict 必须全部为 `supported`，answerable case 才通过 recorded gate；无答案 case 必须 abstain，并且其 context 也不能越权。输出分别报告 action accuracy、coverage、error count、citation coverage/validity、judgment coverage、supported-claim rate、grounded-answer pass rate 和 recorded-gate pass rate。

`sample_answers.jsonl` 是**手写离线 fixture**，没有调用 LLM，也没有做独立人工双标。`supported / contradicted / insufficient` 是 artifact 提供的标签；评测器只聚合并检查 provenance、引用和权限，不会从文本推断 entailment。即使 recorded gate 为 1，也不证明回答完整、来源权威、标签可靠、真实模型表现或生产安全。

审计 packing→output→evaluation 的不可变关联：

~~~powershell
python -m about_llm.rag.cli audit-traces `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --answers projects/rag-foundations/sample_answers.jsonl `
  --traces projects/rag-foundations/generation-traces.example.jsonl
~~~

`audit-traces` 对 case、answer、trace 做 exact join，重新计算 case query SHA-256，并核对 tenant/principals。它按 trace 顺序从当前 corpus 解析 chunk，检查 document/stable-source/version/content SHA-256 与 ACL，重建规范化 `<source>` context，再核对 recorded answer 的 canonical fingerprint。报告同时给出 context、prompt、raw-output、trace 和整体 manifest fingerprint；任一 finding 返回退出码 1。

样例 trace 是 authored non-execution fixture：其中 regex-hash token IDs 只用于验证 artifact plumbing，**不是任何部署模型的 tokenization**。审计不会重新 tokenize prompt、不会把 tokenizer/model revision 与可信 registry 比对，也不会判断 raw output 是否在语义上蕴含 parsed claims。无密钥 SHA-256 只能发现相对某个已信任 manifest 的字节变化；单独拿到一组可共同重写的 unsigned 文件，不能证明它们来自真实调用、未被协同篡改或具有生产 provenance。

### 固定 Qwen 的真实生成失败控制

下面的 control 复用 Transformers Basics 中固定 revision `7ae5576…9a775`、7 个文件和 999,586,347 bytes 的 Qwen2.5-0.5B-Instruct snapshot。它先逐文件重哈希，再在 CPU FP32 eager 下执行 authorization-first BM25、目标 tokenizer 完整 prompt packing、逐步 argmax/KV cache 和 `GenerationMixin.generate()`；没有 `LogitsProcessor` 强制 token，也不在生成后 repair：

~~~powershell
python projects/rag-foundations/run_qwen_rag_control.py --local-files-only
python -m pytest tests/test_rag_transformers_control.py -q
~~~

manifest fingerprint 为 `sha256:4ee16617…09dfd`，录制报告 fingerprint 为 `sha256:829663e2…e5b60`。attempt-1 冻结后不再为追求漂亮结果调 prompt。answerable case 检索并装入 `acl-order-v1`、`citation-boundary-v1`，209-token prompt 的 greedy 输出复述了正确证据并以 EOS 结束，但没有任何 `[S1]`，因此 citation syntax gate 失败。no-answer case 在 tenant/ACL 过滤后检索和 context 均为空，115-token prompt 却生成了无来源的 Kubernetes 步骤，64 token 后因上限停止，没有输出固定拒答句；abstention gate 同样失败。总体行为门禁是 **0/2**。

这组失败比手写“成功答案”更重要：它直接证明检索零结果不等于最终拒答、事实内容看似正确不等于引用合格，也说明 prompt 指令不能替代输出验证和 no-evidence policy。closed-schema verifier 会重算 report/case/raw-output hash、当前 corpus context、citation audit、exact abstention、token/packing 账本和 scope；攻击者即使协同重算无密钥 self-hash，嵌套字段或本地审计漂移仍会被拒绝。不过本地引用检查只证明语法，不证明 claim-evidence entailment；unsigned hash 不认证发布者/录制者，verify→loader reopen 仍有 TOCTOU。两个 authored case 也不能外推总体质量、许可、安全、GPU/vLLM、延迟或生产表现。

### 模型外发布策略的反事实回放

`generation_policy.py` 把 failure control 暴露出的两个问题变成 fail-closed 发布边界，而不修改 attempt-1 的 raw output：授权 context 为空时在 `pre_generation` 阶段确定性 abstain，generator 不得调用；context 非空时只调用一次 generator，随后用本地 citation audit 决定 `publish` 或 `reject`。`reject` 会保留 raw output 供受限审计，但对调用方只给固定拒绝文案。这里的 citation gate 仍只检查短 ID 和段落引用语法，`semantic_entailment_verified` 固定为 false。

代码还把两个序列化面分开：`decision.to_dict()` 是包含 raw output、未知 ID 与未引用段落的**审计投影**，不得直接作为 HTTP response；`decision.to_public_dict()` 是 allowlist 用户面，不包含 `raw_output` 或 finding text，只包含可发布的 `response_text`、typed action/stage 和边界标志。即使 action 是 reject 且模型输出含敏感/注入文本，public projection 也不会因调试字段把它重新带回用户面；受限审计存储仍要单独做访问控制、保留期与脱敏。

先严格验证原 Qwen report，再重建或验证 replay：

~~~powershell
python projects/rag-foundations/replay_qwen_rag_publication_policy.py `
  --verify projects/rag-foundations/qwen2.5-0.5b-rag.publication-policy-replay.json
python -m pytest tests/test_rag_generation_policy.py -q
~~~

冻结 replay fingerprint 为 `sha256:ed4d16ad…b13239`。answerable case 的既有 raw output 没有引用，因此策略回放为 `post_generation/reject`，generator call count 为 1；empty-evidence case 回放为 `pre_generation/abstain`，call count 为 0。两条原始 behavior gate 都是 false，但没有任何 baseline failure 被策略判为 publish。

必须把这份报告称为 **counterfactual policy replay**：它说明这套确定性代码若包裹相同已录制输入/输出会作出什么决策，不是观察到 guard 当时真实包裹了 Qwen runtime，也不是实际测得 provider/GPU 调用被省掉。报告先依赖原 report 的独立严格验证，再绑定其 self-fingerprint、case fingerprint、packed IDs 和 raw-output hash；unsigned SHA-256 仍不认证来源，语法 gate 仍不证明语义忠实、总体质量或生产集成。

### 真实运行时 guarded control

反事实回放之后还有一条独立证据链：`guarded_transformers_control.py` 让同一 fail-closed policy **真实包裹** `GenerationMixin.generate()` callback。它复用相同的固定 checkpoint 与 authored corpus，但换成不同的 case ID 和 query，避免把 attempt-1 输出伪装成新运行；这两个 query 仍共享语料与设计过程，**不是代表性质量集**。

~~~powershell
python projects/rag-foundations/run_qwen_guarded_rag_control.py --local-files-only
python -m pytest tests/test_rag_guarded_transformers_control.py -q
~~~

guarded manifest fingerprint 为 `sha256:9ead4c06…40778a`，真实报告 fingerprint 为 `sha256:00706d00…f29ede`。有授权证据的 case 完成 BM25 与 272-token 总账本，其中 prompt 为 208 tokens、输出预留为 64；guard callback 与 `GenerationMixin.generate` API 各调用 1 次。Qwen 实际生成“无权文档不得进行排序”并以 EOS 结束，但没有 `[S1]`，所以 `post_generation/reject`。空证据 case 的 retrieval、packing 与 source map 都为空，prompt 虽在本地为审计而计算出 116 tokens，却没有传给模型；`generator_callback_invocation_count=0`、`framework_generate_invocation_count=0`，策略直接给出 `pre_generation/abstain`。总计 publish=0。

这里的“1 次”只指 Python 层 `GenerationMixin.generate` API invocation，不是底层 `forward()` 次数、kernel 数、provider 请求或计费次数。空证据的“0 次”是该受控本地 callback 路径的观察，不证明远端取消语义。报告同时保存受限 audit projection 与 allowlist public projection；后者没有 `raw_output`、未知引用或未引用段落，因此 rejected text 不会因直接序列化 decision 再次暴露。

离线 verifier 重建 authorization-first BM25、source/context identity、packing 决策关系、policy decision、public projection、调用计数、summary、scope 和所有 canonical hash，并拒绝协同 rehash 后的 count/cost/reason/chat-template/raw-output 漂移。它不重放模型生成、token IDs 或 decode；unsigned hash 不认证录制者，verify→loader reopen TOCTOU 仍存在。这条 CPU FP32 控制也不证明 claim-evidence entailment、总体质量、GPU/vLLM、性能、provider billing/cancellation 或生产集成。

引用语法与段落覆盖审计：

~~~powershell
python -m about_llm.rag.cli audit `
  --answer-file projects/rag-foundations/sample_answer.md `
  --source-id S1 `
  --source-id S2
~~~

安装后也可使用 `about-llm-rag` 命令。完整回归测试：

~~~powershell
python -m pytest tests/test_rag.py tests/test_rag_ingestion.py tests/test_rag_sqlite_store.py tests/test_rag_citations.py tests/test_rag_reranking.py tests/test_rag_cli.py tests/test_rag_answer_eval.py tests/test_rag_trace.py tests/test_rag_transformers_control.py tests/test_rag_generation_policy.py tests/test_rag_guarded_transformers_control.py
~~~

### JSONL schema

corpus 每行至少包含：

```json
{"source_id":"...","tenant_id":"...","version":"...","text":"# Markdown","acl":["principal"],"metadata":{"uri":"..."}}
```

旧版 binary evaluation case 仍可写：

```json
{"query_id":"...","query":"...","tenant_id":"...","principals":["principal"],"relevant_source_ids":["source-id"]}
```

推荐 schema 使用 source-level graded qrels：

```json
{"query_id":"...","query":"...","tenant_id":"...","principals":["principal"],"answerable":true,"relevance":{"best-source":3,"topical-but-wrong":0},"required_source_ids":["best-source"]}
```

- `relevance` 的 key 是 source id，value 是有限非负数；大于 0 才计入 Recall/MRR/Precision，数值大小用于 nDCG。
- `required_source_ids` 必须是正相关 source 的子集；多跳问题把所有缺一不可的证据列入该集合。
- 无答案 case 必须显式写 `"answerable": false`，没有正相关或 required evidence；空 qrels 若不显式声明会被拒绝，避免标注遗漏被误当成负例。
- 同时提供旧 `relevant_source_ids` 和新 `relevance` 时，两者的正相关集合必须一致。

未标注的 top result 被报告为 `unjudged_retrieved_source_ids`，而不是直接叫 false positive。真实 qrels 常不完备，应先对多个系统的候选做 pooling 和补标，再解释 Precision/nDCG；样例中的 0 label 是明确人工判为不支持该 query，不代表该文档对所有问题都无用。

recorded answer 每行使用严格 schema；未知字段会失败，防止拼写错误被静默忽略：

```json
{"query_id":"...","action":"answer","context_source_ids":["stable-source-id"],"claims":[{"claim_id":"c1","text":"atomic claim","source_ids":["stable-source-id"],"verdict":"supported","judgment_source":"human-gold-v1"}],"missing_information":[]}
```

- `action` 只能是 `answer / abstain / error`；answer 至少一个 claim，abstain 不含 factual claim 且必须说明 `missing_information`，error 必须给 `error_type`。
- verdict 只能是 `supported / contradicted / insufficient / unjudged`。前三者必须给非空 `judgment_source`；unjudged 不得伪装成已有 judge provenance。
- 这里记录稳定 corpus source id。线上生成器若使用短 ID `S1`，写 artifact 前必须通过当次授权 context map 解析回稳定 ID；不能让模型直接声明任意内部 source id。
- `context_source_ids` 只说明 recorded answer 声称模型看到了哪些稳定来源；它自身不能重放 prompt。`generation-traces.example.jsonl` 另行保存逐 chunk snapshot identity、规范化 context、prompt/output identity，并用 answer fingerprint 做 join；生产系统还需把 manifest 放入受控或签名存储，并绑定 policy/packer/decoding/runtime revision。

CLI 是透明 CPU baseline，不调用 LLM，也不声称完成 claim-evidence entailment。它的价值是把生成前的语料、权限、检索、上下文和离线指标变成可审计证据。`gold_source_status=acl_blocked` 表示标注与当前 caller 权限不一致，不应通过放宽 ACL 来“修复召回”；应修正 case 的访问上下文或把它定义为无答案权限切片。

## 为什么先不用框架

RAG 的错误可能来自解析、chunk、召回、过滤、重排、上下文或生成。若第一版就把这些阶段封进链式框架，很难判断提升来自哪里。本项目的领域对象与指标保持框架无关，后续 adapter 只转换输入输出。

## 摄取与引用边界

`split_markdown` 的 chunk id 不包含顺序号，因此在同一标题下插入一个不同段落不会让后续 chunk 全部改名；修改内容、移动标题或相同内容的重复次数变化则会产生新 id。`plan_incremental_update` 明确返回写入和删除集合，调用方应在同一索引事务中应用，并拒绝把“空抓取结果”自动解释为删除全部来源。

`SQLiteChunkStore` 把这条约束落到单机持久层。首次写入要求 `expected_current_version=None`，更新/删除必须给出当前版本；读取与写入之间由 `BEGIN IMMEDIATE` 串行化 writer，陈旧版本 fail closed。同一个 version 若绑定不同 source fingerprint 会拒绝；fingerprint 同时绑定原文、ACL、metadata、`max_chars` 与显式 chunker revision，因而不能在不升级 source version 时悄悄改变切分配置。空切分也不会被解释为全删；删除只能调用显式 `delete_source`。chunks、source manifest 与 stale delete 位于同一事务，测试用 SQLite trigger 在删除后、插入前注入失败，验证旧 version/chunks 完整回滚。

store 重载 heading/ACL/metadata 时使用 strict JSON，拒绝 duplicate key、非有限 number 与类型漂移。`visible_chunks` 先以 tenant 限定 DB 行，再在 Python 中做 principal ACL 过滤，返回值才可交给 BM25/dense scorer。SQLite 文件仍含文档正文和 ACL，默认不加密；生产环境必须配置文件 ACL、加密、备份/删除和审计。

这只是单机 reference store，不是向量数据库：没有 embedding column/ANN、在线 snapshot alias、在线 schema migration、跨进程 lease、replication 或多副本一致性。仓库已有 SQLite backup/verify/restore 演练，但只覆盖本地 schema-v1 文件与 authored fixture；`BEGIN IMMEDIATE` 和 online backup 证明单 SQLite 数据库边界内的事务/一致快照，不证明分布式 vector/index 与 source manifest 原子提交或完整生产灾备。

`build_citation_context` 在渲染前再次检查 tenant，去重后分配短来源 id。`audit_citations` 只验证引用是否存在、id 是否已授权以及段落是否漏引；即使它返回成功，也不代表来源在语义上支持 claim。claim-evidence entailment 应使用人工标注集、NLI/LLM judge 和抽样审计，并报告误判率。

仍需在目标语料上完成 embedding/reranker 消融、真实向量库事务、受认证的在线 generation trace 采集与语义忠实度评测。SQLite 回滚证据不能外推到远端向量库。仓库的 authored trace 只验证离线 identity binding；新增固定 Qwen control 确实执行了真实权重，却在两条 case 上同时暴露漏引与拒答失败。后续 publication-policy replay 证明当前代码能对已录制 attempt 确定性地产生 reject/abstain，但它是反事实重放，不是 guard 与真实生成同时执行的观测。两者都不能证明历史 corpus 可用、claim-evidence entailment、总体质量或生产安全；这些仍依赖部署环境和代表性评测。

## 安全不变量

tenant 与 principal ACL 在评分前执行，不能先全局召回再让 LLM 忽略无权文档。生成器只收到已授权证据；缓存键、日志、评测样本和 trace 同样按租户与权限上下文隔离。
