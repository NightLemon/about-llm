from __future__ import annotations

import math

import pytest

from about_llm.evaluation import (
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    recall_at_k,
)
from about_llm.rag import BM25Index, Document, SearchResult, reciprocal_rank_fusion


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
