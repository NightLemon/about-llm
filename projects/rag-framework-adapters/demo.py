"""把同一组 RAG 检索结果转换为 LangChain Document 和 LlamaIndex Node。

检索和权限逻辑只在仓库的 canonical BM25 实现中执行一次，两个框架适配器只负责转换对象。
最后比较三组 document ID，直观看出接入框架后文档身份没有变化。
"""

from __future__ import annotations

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
    print("canonical ids:", [item.document.document_id for item in results])
    print("LangChain ids:", [item.id for item in langchain_documents])
    print("LlamaIndex ids:", [item.node.node_id for item in llamaindex_nodes])


if __name__ == "__main__":
    main()
