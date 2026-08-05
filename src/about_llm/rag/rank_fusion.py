"""Rank fusion independent of incomparable retriever score scales."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from about_llm.rag.models import SearchResult


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[SearchResult]],
    *,
    rank_constant: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """Fuse rankings by document id using Reciprocal Rank Fusion."""
    if rank_constant < 0:
        raise ValueError("rank_constant cannot be negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scores: defaultdict[str, float] = defaultdict(float)
    documents = {}
    sources: defaultdict[str, set[str]] = defaultdict(set)
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for result in ranking:
            document_id = result.document.document_id
            if document_id in seen_in_ranking:
                continue
            seen_in_ranking.add(document_id)
            scores[document_id] += 1.0 / (rank_constant + result.rank)
            documents[document_id] = result.document
            sources[document_id].add(result.source)

    ordered = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return [
        SearchResult(
            document=documents[document_id],
            score=scores[document_id],
            rank=rank,
            source="rrf:" + "+".join(sorted(sources[document_id])),
        )
        for rank, document_id in enumerate(ordered[:top_k], start=1)
    ]
