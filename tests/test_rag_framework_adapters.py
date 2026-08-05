from __future__ import annotations

import pytest

from about_llm.integrations.rag_frameworks import (
    to_langchain_documents,
    to_llamaindex_nodes,
)
from about_llm.rag import Document, SearchResult


@pytest.fixture
def result() -> SearchResult:
    return SearchResult(
        document=Document(
            "doc-1",
            "RAG separates retrieval from generation.",
            "tenant-a",
            {"page": 3},
        ),
        score=0.75,
        rank=1,
        source="hybrid",
    )


def test_langchain_adapter_preserves_canonical_fields(result: SearchResult) -> None:
    pytest.importorskip("langchain_core.documents")
    document = to_langchain_documents([result])[0]
    assert document.id == "doc-1"
    assert document.page_content == result.document.text
    assert document.metadata == {
        "page": 3,
        "document_id": "doc-1",
        "tenant_id": "tenant-a",
        "retrieval_score": 0.75,
        "retrieval_rank": 1,
        "retriever": "hybrid",
    }


def test_llamaindex_adapter_preserves_canonical_fields(result: SearchResult) -> None:
    pytest.importorskip("llama_index.core.schema")
    node_with_score = to_llamaindex_nodes([result])[0]
    assert node_with_score.node.node_id == "doc-1"
    assert node_with_score.node.text == result.document.text
    assert node_with_score.score == pytest.approx(0.75)
    assert node_with_score.node.metadata["tenant_id"] == "tenant-a"


def test_adapter_rejects_metadata_that_overrides_security_fields() -> None:
    result = SearchResult(
        Document("doc", "text", "tenant-a", {"tenant_id": "forged"}),
        score=1,
        rank=1,
        source="test",
    )
    with pytest.raises(ValueError, match="protected keys"):
        to_langchain_documents([result])
