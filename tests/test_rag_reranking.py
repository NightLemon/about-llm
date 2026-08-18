from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from about_llm.integrations.sentence_transformers import CrossEncoderReranker
from about_llm.rag import (
    Document,
    RecordedRerankScore,
    RecordedRerankScorer,
    SearchResult,
    rerank_authorized_candidates,
)

pytestmark = pytest.mark.security


class RecordingScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> Sequence[float]:
        document_ids = tuple(candidate.document.document_id for candidate in candidates)
        self.calls.append((query, document_ids))
        return [self.scores[document_id] for document_id in document_ids]


def _result(
    document_id: str,
    *,
    rank: int,
    tenant_id: str = "tenant-a",
    acl: tuple[str, ...] = (),
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        document=Document(
            document_id=document_id,
            text=f"content for {document_id}",
            tenant_id=tenant_id,
            acl=acl,
        ),
        score=score,
        rank=rank,
        source="bm25",
    )


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def test_reranker_filters_tenant_and_acl_before_scorer_call() -> None:
    scorer = RecordingScorer({"public": 0.2, "engineering": 0.9})
    report = rerank_authorized_candidates(
        "query",
        (
            _result("other-tenant", rank=1, tenant_id="tenant-b", score=99),
            _result("finance", rank=2, acl=("finance",), score=98),
            _result("public", rank=3, score=0.8),
            _result("engineering", rank=4, acl=("engineering",), score=0.7),
        ),
        scorer,
        tenant_id="tenant-a",
        principals=("engineering",),
        top_k=2,
        scorer_identity="fixture-reranker@v1",
    )

    assert scorer.calls == [("query", ("public", "engineering"))]
    assert [result.document.document_id for result in report.results] == [
        "engineering",
        "public",
    ]
    assert [result.rank for result in report.results] == [1, 2]
    assert [trace.authorization for trace in report.candidates] == [
        "tenant_mismatch",
        "acl_blocked",
        "visible",
        "visible",
    ]
    assert report.candidates[0].candidate_score is None
    assert report.candidates[0].tenant_id == "tenant-b"
    assert report.candidates[1].acl == ("finance",)
    assert report.candidates[1].rerank_score is None
    assert report.authorized_candidate_count == 2
    payload = report.to_dict()
    assert str(payload["query_sha256"]).startswith("sha256:")
    assert payload["scope"] == {
        "authorization_rechecked_before_scorer": True,
        "candidate_content_identity_recorded": True,
        "scorer_output_shape_and_finiteness_verified": True,
        "scorer_or_model_provenance_authenticated": False,
        "target_tokenizer_or_truncation_verified": False,
        "relevance_quality_verified": False,
        "latency_or_cost_measured": False,
    }


def test_reranker_uses_candidate_rank_as_deterministic_score_tie_break() -> None:
    scorer = RecordingScorer({"first": 0.5, "second": 0.5})
    report = rerank_authorized_candidates(
        "query",
        (_result("first", rank=1), _result("second", rank=2)),
        scorer,
        tenant_id="tenant-a",
        top_k=2,
        scorer_identity="fixture@v1",
    )

    assert [result.document.document_id for result in report.results] == [
        "first",
        "second",
    ]


def test_reranker_does_not_call_scorer_without_authorized_candidates() -> None:
    scorer = RecordingScorer({})
    report = rerank_authorized_candidates(
        "query",
        (_result("blocked", rank=1, acl=("finance",)),),
        scorer,
        tenant_id="tenant-a",
        principals=("engineering",),
        top_k=1,
        scorer_identity="fixture@v1",
    )

    assert scorer.calls == []
    assert report.scorer_called is False
    assert report.results == ()


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), True, "0.5"])
def test_reranker_rejects_invalid_candidate_scores(invalid_score: object) -> None:
    scorer = RecordingScorer({"candidate": 0.5})
    candidate = _result("candidate", rank=1, score=invalid_score)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidate score"):
        rerank_authorized_candidates(
            "query",
            (candidate,),
            scorer,
            tenant_id="tenant-a",
            top_k=1,
            scorer_identity="fixture@v1",
        )


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), True, "0.5"])
def test_reranker_rejects_invalid_scorer_outputs(invalid_score: object) -> None:
    class InvalidScorer:
        def score(
            self,
            query: str,
            candidates: Sequence[SearchResult],
        ) -> Sequence[float]:
            return [invalid_score]  # type: ignore[list-item]

    with pytest.raises(ValueError, match="reranker score"):
        rerank_authorized_candidates(
            "query",
            (_result("candidate", rank=1),),
            InvalidScorer(),
            tenant_id="tenant-a",
            top_k=1,
            scorer_identity="fixture@v1",
        )


def test_reranker_rejects_score_count_and_candidate_identity_drift() -> None:
    class EmptyScorer:
        def score(
            self,
            query: str,
            candidates: Sequence[SearchResult],
        ) -> Sequence[float]:
            return []

    with pytest.raises(ValueError, match="returned 0 scores"):
        rerank_authorized_candidates(
            "query",
            (_result("candidate", rank=1),),
            EmptyScorer(),
            tenant_id="tenant-a",
            top_k=1,
            scorer_identity="fixture@v1",
        )
    with pytest.raises(ValueError, match="document_id values must be unique"):
        rerank_authorized_candidates(
            "query",
            (_result("duplicate", rank=1), _result("duplicate", rank=2)),
            RecordingScorer({"duplicate": 0.5}),
            tenant_id="tenant-a",
            top_k=1,
            scorer_identity="fixture@v1",
        )
    with pytest.raises(ValueError, match="contiguous one-based ranks"):
        rerank_authorized_candidates(
            "query",
            (_result("candidate", rank=2),),
            RecordingScorer({"candidate": 0.5}),
            tenant_id="tenant-a",
            top_k=1,
            scorer_identity="fixture@v1",
        )


def test_cross_encoder_adapter_uses_authorization_first_core() -> None:
    class FakeCrossEncoder:
        def __init__(self) -> None:
            self.pairs: list[tuple[str, str]] = []

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            self.pairs = pairs
            return [0.25]

    adapter = CrossEncoderReranker.__new__(CrossEncoderReranker)
    adapter.model = FakeCrossEncoder()
    adapter.scorer_identity = "cross-encoder-fixture@v1"
    report = adapter.rerank_with_report(
        "query",
        (
            _result("blocked", rank=1, acl=("finance",)),
            _result("visible", rank=2),
        ),
        tenant_id="tenant-a",
        principals=("engineering",),
        top_k=1,
    )

    assert adapter.model.pairs == [("query", "content for visible")]
    assert [result.document.document_id for result in report.results] == ["visible"]
    assert report.scorer_identity == "cross-encoder-fixture@v1"

    class BooleanCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[bool]:
            return [True]

    adapter.model = BooleanCrossEncoder()
    with pytest.raises(ValueError, match="not booleans"):
        adapter.rerank_with_report(
            "query",
            (_result("visible", rank=1),),
            tenant_id="tenant-a",
            top_k=1,
        )


def test_recorded_scorer_binds_query_candidate_content_and_identity() -> None:
    candidates = (_result("first", rank=1), _result("second", rank=2))
    scorer = RecordedRerankScorer(
        tuple(
            RecordedRerankScore(
                query_sha256=_hash("query"),
                document_id=candidate.document.document_id,
                content_sha256=_hash(candidate.document.text),
                score=score,
                scorer_identity="authored-fixture@v1",
            )
            for candidate, score in zip(candidates, (0.1, 0.9), strict=True)
        )
    )

    report = rerank_authorized_candidates(
        "query",
        candidates,
        scorer,
        tenant_id="tenant-a",
        top_k=2,
        scorer_identity=scorer.scorer_identity,
    )

    assert [result.document.document_id for result in report.results] == [
        "second",
        "first",
    ]


def test_recorded_scorer_rejects_stale_query_content_and_candidate_set() -> None:
    candidate = _result("candidate", rank=1)
    record = RecordedRerankScore(
        query_sha256=_hash("query"),
        document_id="candidate",
        content_sha256=_hash(candidate.document.text),
        score=0.5,
        scorer_identity="fixture@v1",
    )
    scorer = RecordedRerankScorer((record,))

    with pytest.raises(ValueError, match="query hash"):
        scorer.score("changed query", (candidate,))
    changed = SearchResult(
        document=Document("candidate", "changed content", "tenant-a"),
        score=1.0,
        rank=1,
        source="bm25",
    )
    with pytest.raises(ValueError, match="content hash mismatch"):
        scorer.score("query", (changed,))
    with pytest.raises(ValueError, match="candidate set mismatch"):
        scorer.score("query", (_result("other", rank=1),))


def test_recorded_scorer_rejects_invalid_or_mixed_artifact_identity() -> None:
    with pytest.raises(ValueError, match="lowercase sha256"):
        RecordedRerankScore("sha256:bad", "doc", _hash("content"), 0.5, "fixture")
    first = RecordedRerankScore(
        _hash("query"), "first", _hash("first"), 0.5, "fixture@v1"
    )
    duplicate = RecordedRerankScore(
        _hash("query"), "first", _hash("first"), 0.6, "fixture@v1"
    )
    with pytest.raises(ValueError, match="document_id values must be unique"):
        RecordedRerankScorer((first, duplicate))
    mixed = RecordedRerankScore(
        _hash("other-query"), "second", _hash("second"), 0.7, "fixture@v1"
    )
    with pytest.raises(ValueError, match="one query hash"):
        RecordedRerankScorer((first, mixed))
