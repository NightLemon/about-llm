from __future__ import annotations

import math

import pytest

from about_llm.evaluation import (
    all_evidence_recall_at_k,
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    precision_at_k,
    recall_at_k,
)
from about_llm.rag import BM25Index, Document, SearchResult, reciprocal_rank_fusion

pytestmark = [pytest.mark.formula, pytest.mark.security]


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document("a-llama", "Llama 使用 grouped query attention", "tenant-a"),
        Document("a-rag", "RAG 包含检索 重排 生成 引用", "tenant-a"),
        Document("b-secret", "RAG 项目的秘密代号 ORANGE", "tenant-b"),
    ]


def test_bm25_retrieves_exact_terms_and_enforces_tenant_before_ranking(
    documents: list[Document],
) -> None:
    index = BM25Index(documents)
    tenant_a = index.search("RAG ORANGE", tenant_id="tenant-a")
    tenant_b = index.search("RAG ORANGE", tenant_id="tenant-b")

    assert [result.document.document_id for result in tenant_a] == ["a-rag"]
    assert [result.document.document_id for result in tenant_b] == ["b-secret"]


def test_bm25_filters_principal_acl_before_ranking() -> None:
    index = BM25Index(
        [
            Document("public", "RAG ACL baseline", "tenant-a"),
            Document("restricted", "RAG ACL private details", "tenant-a", acl=("eng",)),
        ]
    )

    anonymous = index.search("RAG ACL private", tenant_id="tenant-a")
    engineer = index.search("RAG ACL private", tenant_id="tenant-a", principals=("eng",))

    assert [result.document.document_id for result in anonymous] == ["public"]
    assert [result.document.document_id for result in engineer] == ["restricted", "public"]


def test_hidden_documents_cannot_change_visible_bm25_scores() -> None:
    visible = [
        Document("visible-a", "RAG ACL baseline", "tenant-a"),
        Document("visible-b", "RAG retrieval baseline", "tenant-a"),
    ]
    hidden = [
        Document("other-tenant", "RAG RAG RAG secret", "tenant-b"),
        Document(
            "restricted",
            "RAG private private private private details",
            "tenant-a",
            acl=("finance",),
        ),
    ]

    baseline = BM25Index(visible).search("RAG", tenant_id="tenant-a")
    with_hidden = BM25Index([*visible, *hidden]).search("RAG", tenant_id="tenant-a")

    assert [item.document.document_id for item in with_hidden] == [
        item.document.document_id for item in baseline
    ]
    assert [item.score for item in with_hidden] == pytest.approx(
        [item.score for item in baseline], rel=0, abs=0
    )


def test_rrf_ignores_duplicate_document_within_one_ranking(
    documents: list[Document],
) -> None:
    first = SearchResult(documents[0], score=99, rank=1, source="dense")
    duplicate = SearchResult(documents[0], score=98, rank=2, source="dense")
    lexical = SearchResult(documents[1], score=1, rank=1, source="bm25")
    fused = reciprocal_rank_fusion([[first, duplicate], [lexical]], rank_constant=0)

    assert [result.document.document_id for result in fused] == ["a-llama", "a-rag"]
    assert fused[0].score == pytest.approx(1.0)


def test_retrieval_metrics() -> None:
    retrieved = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
    relevant = {"q1": {"d2"}, "q2": {"d3", "d5"}}

    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.75)
    assert mean_reciprocal_rank(retrieved, relevant, k=2) == pytest.approx(0.75)


def test_ndcg_supports_graded_relevance_and_ignores_duplicate_results() -> None:
    retrieved = {"q1": ["partial", "partial", "best"]}
    relevance = {"q1": {"best": 3.0, "partial": 1.0}}

    score = normalized_discounted_cumulative_gain(retrieved, relevance, k=2)
    expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert score == pytest.approx(expected)


def test_precision_uses_returned_slots_and_duplicates_receive_no_extra_credit() -> None:
    retrieved = {"q1": ["gold", "gold", "noise"], "q2": []}
    relevant = {"q1": {"gold"}, "q2": {"other"}}

    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 6)


def test_all_evidence_recall_is_query_level_complete_set_rate() -> None:
    retrieved = {"q1": ["a", "b"], "q2": ["a", "noise"]}
    required = {"q1": {"a", "b"}, "q2": {"a", "b"}}

    assert all_evidence_recall_at_k(retrieved, required, k=2) == pytest.approx(0.5)
