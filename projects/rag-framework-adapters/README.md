# LangChain 与 LlamaIndex RAG Adapter

目标：在不改变检索结果、ACL 和评测口径的前提下比较框架，而不是重写三套无法公平对照的 RAG。

## Canonical-first

about_llm.rag.Document 和 SearchResult 是权威领域对象。原生 BM25/hybrid 层先完成 tenant 过滤和排名，adapter 再转换为：

- LangChain Document；
- LlamaIndex TextNode + NodeWithScore。

转换保留 document_id、tenant_id、score、rank、retriever 和业务 metadata。业务 metadata 不得覆盖这些保护字段。

~~~powershell
python -m pip install -e ".[langchain,llamaindex]"
python projects/rag-framework-adapters/demo.py
~~~

## 为什么不让框架接管 ACL

框架默认通常面向功能演示，不知道你的租户模型、资源归属与审计要求。ACL 必须进入数据访问和检索查询；把无权文档召回后再在 Prompt 里要求模型忽略，已经越过信任边界。

## 公平比较

用同一 corpus、chunk、query、top-k 和 golden relevance：

1. 原生 retriever 作为可解释基线；
2. LangChain 只做 orchestration 或 retriever adapter；
3. LlamaIndex 只做 index/query adapter；
4. 比较 Recall@k、nDCG、延迟、依赖复杂度和 trace；
5. 最终生成使用同一模型、Prompt、证据顺序和采样参数。

## 何时使用

LangChain 适合组合 Runnable、工具和 provider；LlamaIndex 对数据摄取、node/index/query abstraction 更集中。简单系统用原生代码更容易审计。选择依据是团队维护成本与所需能力，不是框架热度。
