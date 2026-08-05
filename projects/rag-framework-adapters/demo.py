"""Run one canonical retrieval and adapt it to both RAG frameworks."""

from __future__ import annotations

from about_llm.integrations.rag_frameworks import (
    to_langchain_documents,
    to_llamaindex_nodes,
)
from about_llm.rag import BM25Index, Document


def main() -> None:
    index = BM25Index(
        [
            Document("rag", "RAG 包含检索 重排 生成 引用", "demo"),
            Document("agent", "Agent 调用工具并观察结果", "demo"),
        ]
    )
    results = index.search("RAG 检索", tenant_id="demo")
    langchain_documents = to_langchain_documents(results)
    llamaindex_nodes = to_llamaindex_nodes(results)
    print("canonical ids:", [item.document.document_id for item in results])
    print("LangChain ids:", [item.id for item in langchain_documents])
    print("LlamaIndex ids:", [item.node.node_id for item in llamaindex_nodes])


if __name__ == "__main__":
    main()
