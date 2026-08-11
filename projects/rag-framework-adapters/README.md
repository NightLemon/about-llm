# LangChain 与 LlamaIndex RAG Parity Control

目标：在不改变检索结果、ACL、Prompt 和评测口径的前提下比较框架，而不是重写三套无法公平对照的 RAG。项目既保留最小对象转换 demo，也提供实际调用两个框架 Retriever API 的离线端到端 control。

## Canonical-first

`about_llm.rag.Document` 和 `SearchResult` 是权威领域对象。原生 BM25 层先按 tenant 与 principal ACL 过滤，再评分和排名；adapter 只在这之后转换为：

- LangChain Document；
- LlamaIndex TextNode + NodeWithScore。

转换保留 `document_id`、`tenant_id`、`acl`、score、rank、retriever 和业务 metadata。业务 metadata 不得覆盖这些保护字段，输入结果还必须是无重复 document ID、连续 one-based rank 和有限 score。

LlamaIndex 的 `TextNode` 默认可能把 metadata 纳入 embedding/LLM content。本项目仍把保护字段保留在 node 上供审计，但同时写入 `excluded_embed_metadata_keys` 与 `excluded_llm_metadata_keys`，避免这些控制面字段在默认内容构造中影响 embedding 或进入模型上下文。这不是通用防泄漏保证：自定义 formatter、callback 或 prompt 仍可主动读取 metadata。

两个 round-trip validator 会把框架返回对象与同次 canonical retrieval 的 ID、正文、完整 metadata、rank/score 和 metadata-exclusion policy 逐项对照。它们检测本地管线中的丢字段、重排和意外 mutation；expected results 若来自不可信输入，或攻击者能同时改写 expected 与 framework object，就不提供来源认证。

~~~powershell
python -m pip install -e ".[langchain,llamaindex]"
python projects/rag-framework-adapters/demo.py
python projects/rag-framework-adapters/parity_control.py
~~~

`parity_control.py` 在固定四文档语料上真实执行 `LangChain BaseRetriever.invoke()` 与 `LlamaIndex BaseRetriever.retrieve()`：

- `engineering` 主体只能看到 `acl-before-ranking` 与 `citation-binding`；
- 匿名主体只能看到 public 的 `acl-before-ranking`；
- 高 lexical overlap 的 `finance-secret` 和跨租户 `other-tenant` 在 scorer 前被过滤；
- 两边返回顺序、分数和 metadata 与 canonical BM25 exact match；
- 两种 PromptTemplate 渲染结果 SHA-256 相同；
- 共同的 deterministic extractive non-LLM baseline 产生相同 answer artifact fingerprint；
- authored qrels 下 engineering case 的 Recall@4 与 nDCG@4 都是 1.0。

这些数字是小型 authored fixture 的协议回归，不是框架质量榜、learned retriever 证据或外推性能。

## 为什么不让框架接管 ACL

框架默认抽象不知道你的租户模型、资源归属与审计要求。ACL 必须进入数据访问和检索查询；把无权文档召回后再在 Prompt 里要求模型忽略，已经越过信任边界。本项目证明的是“canonical retriever 在框架 API 后面仍执行既有 ACL”，不是“LangChain/LlamaIndex 默认提供了正确 ACL”。

## 公平比较

用同一 corpus、chunk、query、top-k 和 golden relevance：

1. 原生 retriever 作为可解释基线；
2. LangChain 只做 orchestration 或 retriever adapter；
3. LlamaIndex 只做 index/query adapter；
4. 比较 Recall@k、nDCG、延迟、依赖复杂度和 trace；
5. 最终生成使用同一模型、Prompt、证据顺序和采样参数。

本 control 完成前四项中的离线协议闭环，并用同一个 extractive oracle 代替第 5 项的真实 LLM。若要比较生成模型，必须额外绑定 checkpoint/provider revision、chat template、generation config、raw output、usage、重试和费用；不能把当前 `answer_artifact_fingerprint` 当作模型执行证明。

## 当前证据边界

真实执行：安装在当前环境中的 `langchain-core`、`llama-index-core`，两个 Retriever API、两个 PromptTemplate、canonical BM25/ACL、严格 round trip、extractive answer、Recall@k/nDCG。

没有执行或证明：learned embedding、向量 index、learned reranker、provider/local LLM、框架默认 query engine、网络、并发、持久化、延迟/吞吐、规模扩展、模型忠实度或生产安全。框架版本由每次机器报告读取；仓库宽松版本范围不保证未来 major/minor 行为不变。

## 何时使用

LangChain 适合组合 Runnable、工具和 provider；LlamaIndex 对数据摄取、node/index/query abstraction 更集中。简单系统用原生代码更容易审计。选择依据是团队维护成本与所需能力，不是框架热度。
