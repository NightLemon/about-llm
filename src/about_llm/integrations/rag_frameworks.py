"""Lossless adapters from canonical RAG results to framework objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from about_llm.rag import SearchResult


def _result_metadata(result: SearchResult) -> dict[str, Any]:
    metadata = dict(result.document.metadata)
    protected = {"document_id", "tenant_id", "retrieval_score", "retrieval_rank", "retriever"}
    collision = protected & set(metadata)
    if collision:
        raise ValueError(f"document metadata uses protected keys: {sorted(collision)}")
    metadata.update(
        {
            "document_id": result.document.document_id,
            "tenant_id": result.document.tenant_id,
            "retrieval_score": result.score,
            "retrieval_rank": result.rank,
            "retriever": result.source,
        }
    )
    return metadata


def to_langchain_documents(results: Sequence[SearchResult]) -> list[Any]:
    """Convert results while preserving ids, ACL context, rank, and score."""
    try:
        from langchain_core.documents import Document as LangChainDocument
    except ImportError as error:
        raise ImportError("LangChain adapter requires: pip install -e '.[langchain]'") from error
    return [
        LangChainDocument(
            id=result.document.document_id,
            page_content=result.document.text,
            metadata=_result_metadata(result),
        )
        for result in results
    ]


def to_llamaindex_nodes(results: Sequence[SearchResult]) -> list[Any]:
    """Convert results to NodeWithScore without rerunning retrieval."""
    try:
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError as error:
        raise ImportError("LlamaIndex adapter requires: pip install -e '.[llamaindex]'") from error
    return [
        NodeWithScore(
            node=TextNode(
                id_=result.document.document_id,
                text=result.document.text,
                metadata=_result_metadata(result),
            ),
            score=result.score,
        )
        for result in results
    ]
