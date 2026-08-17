from __future__ import annotations

import math

import pytest

from about_llm.integrations.rag_frameworks import (
    build_langchain_retriever,
    build_llamaindex_retriever,
    to_langchain_documents,
    to_llamaindex_nodes,
    validate_langchain_round_trip,
    validate_llamaindex_round_trip,
)
from about_llm.rag import BM25Index, Document, SearchResult


@pytest.fixture
def result() -> SearchResult:
    return SearchResult(
        document=Document(
            "doc-1",
            "RAG separates retrieval from generation.",
            "tenant-a",
            {"page": 3},
            acl=("reader",),
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
        "acl": ["reader"],
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
    expected_exclusions = [
        "acl",
        "document_id",
        "retrieval_rank",
        "retrieval_score",
        "retriever",
        "tenant_id",
    ]
    assert node_with_score.node.excluded_embed_metadata_keys == expected_exclusions
    assert node_with_score.node.excluded_llm_metadata_keys == expected_exclusions


def test_adapter_rejects_metadata_that_overrides_security_fields() -> None:
    result = SearchResult(
        Document("doc", "text", "tenant-a", {"tenant_id": "forged"}),
        score=1,
        rank=1,
        source="test",
    )
    with pytest.raises(ValueError, match="protected keys"):
        to_langchain_documents([result])


def test_round_trip_validators_reject_framework_mutation(result: SearchResult) -> None:
    pytest.importorskip("langchain_core.documents")
    pytest.importorskip("llama_index.core.schema")
    langchain_documents = to_langchain_documents([result])
    llamaindex_nodes = to_llamaindex_nodes([result])

    assert validate_langchain_round_trip(langchain_documents, [result]) == (result,)
    assert validate_llamaindex_round_trip(llamaindex_nodes, [result]) == (result,)

    langchain_documents[0].metadata["retrieval_rank"] = 2
    with pytest.raises(ValueError, match="metadata drift"):
        validate_langchain_round_trip(langchain_documents, [result])

    llamaindex_nodes[0].node.text = "mutated"
    with pytest.raises(ValueError, match="text drift"):
        validate_llamaindex_round_trip(llamaindex_nodes, [result])

    llamaindex_nodes = to_llamaindex_nodes([result])
    llamaindex_nodes[0].node.excluded_llm_metadata_keys = []
    with pytest.raises(ValueError, match="LLM metadata exclusion drift"):
        validate_llamaindex_round_trip(llamaindex_nodes, [result])


def test_adapter_rejects_noncanonical_result_identity_and_rank(
    result: SearchResult,
) -> None:
    pytest.importorskip("langchain_core.documents")
    rank_gap = SearchResult(
        result.document,
        score=result.score,
        rank=2,
        source=result.source,
    )
    with pytest.raises(ValueError, match="contiguous one-based ranks"):
        to_langchain_documents([rank_gap])

    duplicate = SearchResult(
        result.document,
        score=0.5,
        rank=2,
        source=result.source,
    )
    with pytest.raises(ValueError, match="duplicate document ids"):
        to_langchain_documents([result, duplicate])


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_adapter_rejects_non_finite_scores(
    result: SearchResult,
    score: float,
) -> None:
    pytest.importorskip("langchain_core.documents")
    invalid = SearchResult(
        result.document,
        score=score,
        rank=1,
        source=result.source,
    )
    with pytest.raises(ValueError, match="scores must be finite"):
        to_langchain_documents([invalid])


def test_adapter_rejects_boolean_score(result: SearchResult) -> None:
    pytest.importorskip("langchain_core.documents")
    invalid = SearchResult(
        result.document,
        score=True,
        rank=1,
        source=result.source,
    )
    with pytest.raises(TypeError, match="scores must be real numbers"):
        to_langchain_documents([invalid])


def test_framework_retrievers_keep_acl_inside_canonical_search() -> None:
    pytest.importorskip("langchain_core.retrievers")
    pytest.importorskip("llama_index.core.retrievers")
    index = BM25Index(
        [
            Document("public", "RAG retrieval", "tenant-a"),
            Document("allowed", "RAG retrieval", "tenant-a", acl=("reader",)),
            Document("denied", "RAG retrieval", "tenant-a", acl=("finance",)),
            Document("other", "RAG retrieval", "tenant-b"),
        ]
    )
    canonical = index.search(
        "RAG retrieval",
        tenant_id="tenant-a",
        principals=("reader",),
    )

    langchain_documents = build_langchain_retriever(
        index,
        tenant_id="tenant-a",
        principals=("reader",),
    ).invoke("RAG retrieval")
    llamaindex_nodes = build_llamaindex_retriever(
        index,
        tenant_id="tenant-a",
        principals=("reader",),
    ).retrieve("RAG retrieval")

    assert [document.id for document in langchain_documents] == ["allowed", "public"]
    assert [item.node.node_id for item in llamaindex_nodes] == ["allowed", "public"]
    assert validate_langchain_round_trip(langchain_documents, canonical) == tuple(canonical)
    assert validate_llamaindex_round_trip(llamaindex_nodes, canonical) == tuple(canonical)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tenant_id": "", "principals": (), "top_k": 1}, "tenant_id"),
        ({"tenant_id": "t", "principals": ("p", "p"), "top_k": 1}, "duplicates"),
        ({"tenant_id": "t", "principals": (), "top_k": True}, "top_k"),
    ],
)
def test_framework_retriever_rejects_ambiguous_security_context(
    kwargs: dict[str, object],
    message: str,
) -> None:
    pytest.importorskip("langchain_core.retrievers")
    index = BM25Index([Document("doc", "RAG", "t")])

    with pytest.raises(ValueError, match=message):
        build_langchain_retriever(index, **kwargs)  # type: ignore[arg-type]
