"""把同一组 RAG 检索结果转换为 LangChain Document 和 LlamaIndex Node。

检索和权限逻辑只在仓库的 canonical BM25 实现中执行一次，两个框架适配器只负责转换对象。
最后比较三组 document ID，直观看出接入框架后文档身份没有变化。
"""

from __future__ import annotations

import json

from about_llm.integrations.rag_frameworks import (
    to_langchain_documents,
    to_llamaindex_nodes,
)
from about_llm.rag import BM25Index, Document


def main() -> None:
    """建立最小索引，检索一次并打印两个框架中的文档 ID。"""

    # 两篇文档足以让查询命中 RAG，同时保留一个主题不同的对照。
    index = BM25Index(
        [
            Document("rag", "RAG 包含检索 重排 生成 引用", "demo"),
            Document("agent", "Agent 调用工具并观察结果", "demo"),
        ]
    )
    # ACL 与排序先在 canonical 层完成，框架不会重新检索或扩大可见集合。
    results = index.search("RAG 检索", tenant_id="demo")
    langchain_documents = to_langchain_documents(results)
    llamaindex_nodes = to_llamaindex_nodes(results)
    canonical_ids = [item.document.document_id for item in results]
    langchain_ids = [item.id for item in langchain_documents]
    llamaindex_ids = [item.node.node_id for item in llamaindex_nodes]
    report = {
        "scenario": "adapt one authorized canonical retrieval result into two framework types",
        "input": {
            "query": "RAG 检索",
            "tenant_id": "demo",
            "documents": [
                {
                    "document_id": item.document.document_id,
                    "text": item.document.text,
                }
                for item in results
            ],
        },
        "document_ids": {
            "canonical": canonical_ids,
            "langchain": langchain_ids,
            "llamaindex": llamaindex_ids,
        },
        "conclusion": {
            "langchain_preserved_document_ids": langchain_ids == canonical_ids,
            "llamaindex_preserved_document_ids": llamaindex_ids == canonical_ids,
        },
        "scope": {
            "retrieval_and_acl_executed_once_in_canonical_layer": True,
            "frameworks_retrieved_or_reauthorized_documents": False,
            "answer_generation_or_framework_quality_compared": False,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
