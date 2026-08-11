"""Authorization-first reranking with auditable candidate identity and score scope."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from about_llm.rag.models import SearchResult


class RerankScorer(Protocol):
    """Minimal pair-scoring contract; candidates have already passed authorization."""

    def score(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class RecordedRerankScore:
    query_sha256: str
    document_id: str
    content_sha256: str
    score: float
    scorer_identity: str

    def __post_init__(self) -> None:
        _validate_sha256(self.query_sha256, "query_sha256")
        _validate_sha256(self.content_sha256, "content_sha256")
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not self.scorer_identity.strip():
            raise ValueError("scorer_identity cannot be empty")
        object.__setattr__(self, "score", _finite_score(self.score, "recorded score"))


class RecordedRerankScorer:
    """Replay exact externally recorded scores without claiming model execution."""

    def __init__(self, records: Sequence[RecordedRerankScore]) -> None:
        self.records = tuple(records)
        if not self.records:
            raise ValueError("at least one recorded rerank score is required")
        document_ids = [record.document_id for record in self.records]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("recorded rerank document_id values must be unique")
        query_hashes = {record.query_sha256 for record in self.records}
        if len(query_hashes) != 1:
            raise ValueError("recorded rerank scores must bind one query hash")
        scorer_identities = {record.scorer_identity for record in self.records}
        if len(scorer_identities) != 1:
            raise ValueError("recorded rerank scores must bind one scorer identity")
        self.query_sha256 = next(iter(query_hashes))
        self.scorer_identity = next(iter(scorer_identities))
        self._by_document_id = {record.document_id: record for record in self.records}

    def score(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> Sequence[float]:
        if _text_sha256(query) != self.query_sha256:
            raise ValueError("recorded rerank query hash does not match current query")
        candidate_ids = [candidate.document.document_id for candidate in candidates]
        if set(candidate_ids) != set(self._by_document_id):
            missing = sorted(set(candidate_ids) - set(self._by_document_id))
            extra = sorted(set(self._by_document_id) - set(candidate_ids))
            raise ValueError(
                "recorded rerank candidate set mismatch: "
                f"missing_scores={missing}, extra_scores={extra}"
            )
        scores: list[float] = []
        for candidate in candidates:
            record = self._by_document_id[candidate.document.document_id]
            if _text_sha256(candidate.document.text) != record.content_sha256:
                raise ValueError(
                    "recorded rerank content hash mismatch for "
                    f"{candidate.document.document_id!r}"
                )
            scores.append(record.score)
        return scores


@dataclass(frozen=True)
class RerankCandidateTrace:
    document_id: str
    content_sha256: str
    tenant_id: str
    acl: tuple[str, ...]
    candidate_rank: int
    candidate_source: str
    candidate_score: float | None
    authorization: str
    rerank_score: float | None = None
    rerank_rank: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "content_sha256": self.content_sha256,
            "tenant_id": self.tenant_id,
            "acl": list(self.acl),
            "candidate_rank": self.candidate_rank,
            "candidate_source": self.candidate_source,
            "candidate_score": self.candidate_score,
            "authorization": self.authorization,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
        }


@dataclass(frozen=True)
class RerankReport:
    query_sha256: str
    tenant_id: str
    principals: tuple[str, ...]
    scorer_identity: str
    requested_top_k: int
    input_candidate_count: int
    authorized_candidate_count: int
    scorer_called: bool
    results: tuple[SearchResult, ...]
    candidates: tuple[RerankCandidateTrace, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "query_sha256": self.query_sha256,
            "tenant_id": self.tenant_id,
            "principals": list(self.principals),
            "scorer_identity": self.scorer_identity,
            "requested_top_k": self.requested_top_k,
            "input_candidate_count": self.input_candidate_count,
            "authorized_candidate_count": self.authorized_candidate_count,
            "scorer_called": self.scorer_called,
            "results": [
                {
                    "document_id": result.document.document_id,
                    "score": result.score,
                    "rank": result.rank,
                    "source": result.source,
                }
                for result in self.results
            ],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "scope": {
                "authorization_rechecked_before_scorer": True,
                "candidate_content_identity_recorded": True,
                "scorer_output_shape_and_finiteness_verified": True,
                "scorer_or_model_provenance_authenticated": False,
                "target_tokenizer_or_truncation_verified": False,
                "relevance_quality_verified": False,
                "latency_or_cost_measured": False,
            },
        }


def rerank_authorized_candidates(
    query: str,
    candidates: Sequence[SearchResult],
    scorer: RerankScorer,
    *,
    tenant_id: str,
    principals: Iterable[str] = (),
    top_k: int,
    scorer_identity: str,
) -> RerankReport:
    """Filter by tenant/ACL before scoring, then rerank with deterministic ties."""

    if not query.strip():
        raise ValueError("query cannot be empty")
    if not tenant_id.strip():
        raise ValueError("tenant_id cannot be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not scorer_identity.strip():
        raise ValueError("scorer_identity cannot be empty")
    principal_tuple = tuple(sorted(set(principals)))
    if any(not principal.strip() for principal in principal_tuple):
        raise ValueError("principals cannot contain an empty value")

    candidate_tuple = tuple(candidates)
    document_ids = [candidate.document.document_id for candidate in candidate_tuple]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("candidate document_id values must be unique")
    expected_ranks = list(range(1, len(candidate_tuple) + 1))
    if [candidate.rank for candidate in candidate_tuple] != expected_ranks:
        raise ValueError("candidates must be ordered with contiguous one-based ranks")

    principal_set = set(principal_tuple)
    authorized: list[SearchResult] = []
    trace_by_id: dict[str, RerankCandidateTrace] = {}
    for candidate in candidate_tuple:
        document = candidate.document
        authorization = "visible"
        candidate_score: float | None = None
        if document.tenant_id != tenant_id:
            authorization = "tenant_mismatch"
        elif document.acl and principal_set.isdisjoint(document.acl):
            authorization = "acl_blocked"
        else:
            candidate_score = _finite_score(candidate.score, "candidate score")
            authorized.append(candidate)
        trace_by_id[document.document_id] = RerankCandidateTrace(
            document_id=document.document_id,
            content_sha256=_text_sha256(document.text),
            tenant_id=document.tenant_id,
            acl=document.acl,
            candidate_rank=candidate.rank,
            candidate_source=candidate.source,
            candidate_score=candidate_score,
            authorization=authorization,
        )

    if not authorized:
        return RerankReport(
            query_sha256=_text_sha256(query),
            tenant_id=tenant_id,
            principals=principal_tuple,
            scorer_identity=scorer_identity,
            requested_top_k=top_k,
            input_candidate_count=len(candidate_tuple),
            authorized_candidate_count=0,
            scorer_called=False,
            results=(),
            candidates=tuple(trace_by_id[document_id] for document_id in document_ids),
        )

    raw_scores = tuple(scorer.score(query, authorized))
    if len(raw_scores) != len(authorized):
        raise ValueError(
            f"reranker returned {len(raw_scores)} scores for {len(authorized)} candidates"
        )
    scored = [
        (_finite_score(score, "reranker score"), candidate)
        for score, candidate in zip(raw_scores, authorized, strict=True)
    ]
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].rank,
            item[1].document.document_id,
        )
    )
    ranked_results: list[SearchResult] = []
    for rerank_rank, (score, candidate) in enumerate(scored, start=1):
        previous = trace_by_id[candidate.document.document_id]
        trace_by_id[candidate.document.document_id] = RerankCandidateTrace(
            document_id=previous.document_id,
            content_sha256=previous.content_sha256,
            tenant_id=previous.tenant_id,
            acl=previous.acl,
            candidate_rank=previous.candidate_rank,
            candidate_source=previous.candidate_source,
            candidate_score=previous.candidate_score,
            authorization=previous.authorization,
            rerank_score=score,
            rerank_rank=rerank_rank,
        )
        if rerank_rank <= top_k:
            ranked_results.append(
                SearchResult(
                    document=candidate.document,
                    score=score,
                    rank=rerank_rank,
                    source=f"rerank:{candidate.source}",
                )
            )

    return RerankReport(
        query_sha256=_text_sha256(query),
        tenant_id=tenant_id,
        principals=principal_tuple,
        scorer_identity=scorer_identity,
        requested_top_k=top_k,
        input_candidate_count=len(candidate_tuple),
        authorized_candidate_count=len(authorized),
        scorer_called=True,
        results=tuple(ranked_results),
        candidates=tuple(trace_by_id[document_id] for document_id in document_ids),
    )


def _finite_score(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    prefix = "sha256:"
    digest = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} must be a lowercase sha256: digest")
