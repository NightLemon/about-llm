"""Retrieval metrics with explicit query-level inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set


def recall_at_k(
    retrieved: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Set[str]],
    *,
    k: int,
) -> float:
    """Macro-average the fraction of relevant ids retrieved in the first k."""
    _validate_queries(retrieved, relevant, k)
    values = []
    for query_id, relevant_ids in relevant.items():
        if not relevant_ids:
            raise ValueError(f"query {query_id!r} has no relevant documents")
        found = set(retrieved[query_id][:k]) & relevant_ids
        values.append(len(found) / len(relevant_ids))
    return sum(values) / len(values)


def mean_reciprocal_rank(
    retrieved: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Set[str]],
    *,
    k: int,
) -> float:
    """Average reciprocal rank of the first relevant result up to k."""
    _validate_queries(retrieved, relevant, k)
    values = []
    for query_id, relevant_ids in relevant.items():
        reciprocal_rank = 0.0
        for rank, document_id in enumerate(retrieved[query_id][:k], start=1):
            if document_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        values.append(reciprocal_rank)
    return sum(values) / len(values)


def normalized_discounted_cumulative_gain(
    retrieved: Mapping[str, Sequence[str]],
    relevance: Mapping[str, Mapping[str, float]],
    *,
    k: int,
) -> float:
    """Macro-average nDCG@k with graded, non-negative relevance labels."""
    _validate_queries(retrieved, relevance, k)
    values: list[float] = []
    for query_id, labels in relevance.items():
        if not labels or any(value < 0 for value in labels.values()):
            raise ValueError(f"query {query_id!r} needs non-negative relevance labels")
        ideal = sorted(labels.values(), reverse=True)[:k]
        ideal_dcg = _discounted_cumulative_gain(ideal)
        if ideal_dcg == 0:
            raise ValueError(f"query {query_id!r} has no positive relevance label")
        seen: set[str] = set()
        gains: list[float] = []
        for document_id in retrieved[query_id]:
            if document_id in seen:
                continue
            seen.add(document_id)
            gains.append(labels.get(document_id, 0.0))
            if len(gains) == k:
                break
        values.append(_discounted_cumulative_gain(gains) / ideal_dcg)
    return sum(values) / len(values)


def _discounted_cumulative_gain(relevance: Sequence[float]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(relevance, 1))


def _validate_queries(
    retrieved: Mapping[str, Sequence[str]],
    relevant: Mapping[str, object],
    k: int,
) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("at least one query is required")
    missing = set(relevant) - set(retrieved)
    if missing:
        raise ValueError(f"retrieved results missing queries: {sorted(missing)}")
