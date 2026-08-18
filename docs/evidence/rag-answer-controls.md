# RAG 请求、回答与拒答证据账本

第一次学习 RAG 时，不要从本页开始。先读[RAG 总览](../applications/rag.md)，
再跟随[一次 RAG 请求的生命周期](../applications/rag-request-lifecycle.md)完成
[实验 5](../practice/labs/lab-5-rag-request.md)。

本页面向内容维护者和项目评审者，集中回答：教材里的关键结论由什么 oracle 支持，
以及这些证据没有证明什么。

**读者入口**：[召回与重排](../applications/rag-retrieval.md) ·
[上下文、引用与拒答](../applications/rag-generation.md) ·
[RAG Foundations](../practice/projects/rag-foundations.md)
{ .doc-nav }

## 证据层级

| 层级 | 本仓库实例 | 可以支持的结论 |
|---|---|---|
| 公式与状态机 oracle | BM25、RRF、指标、packing、citation audit | 固定输入下的分数、排序、预算和语法行为 |
| 本地组合 control | extractive walkthrough、SQLite、ASGI service | 指定组件在当前进程或本地服务路径中真实组合 |
| 固定模型 control | Qwen CPU FP32 原始失败与 guarded run | 指定 checkpoint、Prompt 和少量 case 的观察行为 |
| 目标环境证据 | 真实 corpus、IAM、向量库、模型与流量 | 目标系统的质量、安全、延迟、成本和运维结论 |

前三层可在仓库中复算。第四层必须由目标环境产生，不能由 authored fixture 或全绿测试代替。

## 教材结论与对应 oracle

| 教材结论 | 主要实现或测试 | Oracle 怎样独立 | 仍未证明 |
|---|---|---|---|
| 隐藏文档不能改变可见 BM25 score | `tests/test_rag.py` | 加入跨 tenant/ACL 文档前后逐项比较 ID 与 score | 索引进程未读隐藏正文、无时间侧信道 |
| Reranker 只能收到授权候选 | `tests/test_rag_reranking.py` | scorer spy 记录实际输入 | learned scorer 质量、模型来源或性能 |
| Recorded score 必须绑定 query 与 chunk | `tests/test_rag_cli.py` | 改 query/content 后重放必须失败 | 分数来自真实模型或正确标注 |
| 被预算丢弃的候选也必须先授权 | `tests/test_rag_context_packing.py` | 越权候选放在必丢位置仍触发拒绝 | 目标 tokenizer、最优 packing 或长上下文质量 |
| Exact answer span 来自 packed source | `tests/test_rag_extractive.py` | 独立检查 offset slice、hash 与 source map | 来源真实、语义相关或答案完整 |
| Citation audit 只检查 ID 与段落覆盖 | `tests/test_rag_citations.py` | 未知 ID、漏引和合法 ID 反例 | Claim–evidence entailment |
| No-answer case 可以在 non-empty retrieval 下拒答 | `tests/test_rag_extractive.py` | topical negative 有候选但 coverage 不足 | 0.55 阈值适用于目标业务 |
| Answer、abstain、error 必须同分母 join | `tests/test_rag_answer_eval.py` | case/output exact join 与终态聚合手算 | 人工 verdict 正确或回答完整 |
| 空授权 context 可以在生成前 abstain | `tests/test_rag_generation_policy.py` | generator spy 断言零调用 | 远端 provider 请求、计费或取消 |
| 被拒 raw output 不进入 public projection | `tests/test_rag_generation_policy.py` | public/audit 字段 allowlist 对比 | 日志、APM 和存储没有其他泄漏 |
| 请求体不能自报 tenant/principal | `tests/test_rag_service.py` | body 注入、缺 credential 与授权主体对照 | JWT/OAuth、TLS、可信代理或多副本 IAM |

测试名称只表达局部证据。`citation_valid` 不能读成“答案真实”，
`grounded_answer_pass` 也必须检查它在该实现中究竟聚合了什么 supplied verdict。

## 授权边界的精确表述

本仓库的 `BM25Index` 在构造时保存全部传入文档并预计算 term frequency。
查询时先找出 visible indices，再只用这些文档计算 IDF、平均长度与 per-document score。

因此可以声称：

```text
authorization-before-query-statistics-and-ranking
```

不能声称：

```text
unauthorized text was never read by the trusted indexing process
```

真实系统还要区分：

- 摄取与索引控制面的服务权限；
- 在线请求数据面的调用者权限；
- reranker、cache、trace、generator 与 UI 的下游权限；
- 策略更新后旧缓存和旧索引怎样失效。

## Walkthrough control 的边界

运行：

~~~powershell
python projects/rag-foundations/rag_request_walkthrough.py
python -m pytest tests/test_rag_request_walkthrough.py -q
~~~

请求 A 真实组合：

```text
Markdown split
-> authorization-first BM25
-> authorization-first recorded rerank
-> context packing
-> exact-span answer
-> citation syntax audit
```

固定结果为：

```text
retrieval stable sources = [rag-security, rag-evaluation, rag-security]
rerank top-2             = [rag-security, rag-security]
answer coverage          = 1.0
final action             = answer
```

请求 B 使用 BM25-score passthrough 走过同一 rerank 边界，packing 后 lexical coverage 为 `2/9`，
最终 `abstain`。

这条 control 没有执行 learned reranker、Embedding、ANN、目标 tokenizer 或 LLM。
它证明控制流和固定小语料的 oracle，不证明生产质量、延迟或安全。

## 引用证据的分层

| 层 | 本仓库当前可执行证据 | 缺口 |
|---|---|---|
| Source ID syntax | `audit_citations` | 不判断内容 |
| Source authorization | context 构建时 tenant/ACL 复查 | 不认证外部 IAM |
| Exact span identity | extractive offset 与 citation span metric | 不判断语义 |
| Claim verdict 聚合 | recorded answer gate | verdict 由工件提供 |
| Semantic correctness | 需要校准 judge 或人工标注 | 当前通用 gate 不推断 |
| Source truth/currentness | 需要来源治理与版本规则 | hash 只绑定 bytes |
| Answer completeness | 需要 reference claims/rubric | 输出一条真话仍可能漏答 |

一个刻意的反例是：把“The moon is cheese.”绑定到来源中的 exact `Earth` span。
Offset、quote 和 source membership 都可以通过，语义仍然不受支持。

## 固定 Qwen 的三层证据

### 原始 attempt：保留失败

~~~powershell
python projects/rag-foundations/run_qwen_rag_control.py --local-files-only
python -m pytest tests/test_rag_transformers_control.py -q
~~~

固定 Qwen2.5-0.5B-Instruct CPU FP32 control 有两个 authored case：

- Answerable case 复述了核心证据，但漏掉 `[S1]`。
- No-answer case 的授权 context 为空，模型仍生成 Kubernetes 灾备步骤。

行为 gate 是 `0/2`。它说明低温或 greedy、清晰 Prompt 与正确 ACL 仍不自动产生引用和拒答。
两个 case 不能代表模型总体质量。

### Counterfactual policy replay

~~~powershell
python projects/rag-foundations/replay_qwen_rag_publication_policy.py `
  --verify projects/rag-foundations/qwen2.5-0.5b-rag.publication-policy-replay.json
~~~

Replay 对已录制 output 应用 fail-closed policy：漏引 answer 被 reject，空 context case 在逻辑上 pre-generation abstain。
它没有观察 policy 当时真实包裹 Qwen，所以不能声称实际省掉了调用或费用。

### Guarded runtime control

~~~powershell
python projects/rag-foundations/run_qwen_guarded_rag_control.py --local-files-only
python -m pytest tests/test_rag_guarded_transformers_control.py -q
~~~

独立运行中，有证据 case 进入一次 `GenerationMixin.generate()`，因漏引被 reject；
空证据 case 的 callback 和 framework generate invocation 都为零，并在生成前 abstain。

计数对象是 Python API invocation，不是 forward、kernel、远端请求、取消或 billing。
Verifier 不重放 decode，也不证明语义蕴含、GPU/vLLM 或代表性质量。

## 核心 RAG 门禁

下面一组测试覆盖教材主线最容易写错的公式、授权、packing、回答和发布边界：

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

需要优先保留的失败路径是：

~~~powershell
python -m pytest `
  tests/test_rag.py::test_hidden_documents_cannot_change_visible_bm25_scores `
  tests/test_rag_reranking.py::test_reranker_filters_tenant_and_acl_before_scorer_call `
  tests/test_rag_context_packing.py::test_every_candidate_is_authorized_even_when_budget_would_drop_it `
  tests/test_rag_cli.py::test_recorded_rerank_cli_rejects_stale_query_binding `
  -q
~~~

这些测试通过只说明已选择的 oracle 没有回归。它们不能证明未知攻击不存在。

## 目标环境仍需验证

上线前至少补齐：

- 真实 IAM、policy revision、缓存失效和跨 tenant 对抗测试；
- 目标 parser、chunker、Embedding、ANN、reranker 与 tokenizer identity；
- 代表性 answerable/no-answer/conflict/injection qrels；
- Atomic claim 的人工或经过校准的 entailment 判断；
- Accepted-answer risk、拒答率、切片质量和置信区间；
- 真实并发下的 p95、队列、超时、取消、成本与容量；
- Corpus 更新、删除传播、备份恢复和回滚演练；
- Raw output、trace、审计库和 UI 的独立访问控制。

## 内容变更时怎样审查

修改 RAG 教材或实现时，按 claim 审查：

1. 写出正文声称的层级：召回、排序、provenance、语义、质量还是生产安全。
2. 找到对应 oracle，确认没有直接复制被测实现的同一逻辑。
3. 检查正常、边界和失败路径，尤其是权限、预算、无答案和 unknown citation。
4. 检查指标分母，以及 error/timeout 是否被悄悄丢掉。
5. 把模型总体质量、真实来源、目标性能和系统绝对安全留给目标环境。

完整实现和录制工件位于
[projects/rag-foundations](https://github.com/NightLemon/about-llm/tree/main/projects/rag-foundations)。
