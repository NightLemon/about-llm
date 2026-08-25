"""Code-generation metrics with explicit sample-count semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSelectionSummary:
    """Task-level coverage and selection rates for one fixed candidate set."""

    task_count: int
    k: int
    oracle_success_count: int
    selected_success_count: int

    @property
    def oracle_at_k(self) -> float:
        """Fraction of tasks with at least one correct candidate."""

        return self.oracle_success_count / self.task_count

    @property
    def selected_at_k(self) -> float:
        """Fraction of tasks whose selected candidate is correct."""

        return self.selected_success_count / self.task_count

    @property
    def selector_recall_on_oracle_positive(self) -> float | None:
        """Selection success among tasks that contain a correct candidate."""

        if self.oracle_success_count == 0:
            return None
        return self.selected_success_count / self.oracle_success_count


def pass_at_k(*, num_samples: int, num_correct: int, k: int) -> float:
    """Estimate the probability that at least one of ``k`` samples is correct.

    This is ``1 - C(n-c, k) / C(n, k)`` for ``n`` evaluated generations with
    ``c`` correct. Under the usual i.i.d. generation assumption it is the common
    unbiased pass@k estimator. It is not a single-attempt production success rate.
    """

    values = {"num_samples": num_samples, "num_correct": num_correct, "k": k}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be in [0, num_samples]")
    if not 1 <= k <= num_samples:
        raise ValueError("k must be in [1, num_samples]")
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def summarize_candidate_selection(
    *,
    candidate_correctness: Sequence[Sequence[bool]],
    selected_indices: Sequence[int],
) -> CandidateSelectionSummary:
    """Summarize one-candidate selection from fixed equal-sized candidate sets.

    ``oracle_at_k`` asks whether any of the ``k`` supplied candidates is correct.
    ``selected_at_k`` asks whether the supplied selector index points to a correct
    candidate.  The function does not generate candidates, run a verifier, or
    estimate production success.
    """

    rows = tuple(tuple(row) for row in candidate_correctness)
    selected = tuple(selected_indices)
    if not rows:
        raise ValueError("candidate_correctness must contain at least one task")
    k = len(rows[0])
    if k == 0:
        raise ValueError("each task must contain at least one candidate")
    if any(len(row) != k for row in rows):
        raise ValueError("all tasks must contain the same number of candidates")
    if any(type(value) is not bool for row in rows for value in row):
        raise TypeError("candidate correctness values must be booleans")
    if len(selected) != len(rows):
        raise ValueError("selected_indices must contain one index per task")
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("selected indices must be integers")
        if not 0 <= index < k:
            raise ValueError("selected index is outside the candidate set")

    oracle_success_count = sum(any(row) for row in rows)
    selected_success_count = sum(
        row[index] for row, index in zip(rows, selected, strict=True)
    )
    return CandidateSelectionSummary(
        task_count=len(rows),
        k=k,
        oracle_success_count=oracle_success_count,
        selected_success_count=selected_success_count,
    )
