"""Exact finite-support oracle for verifier-guided best-of-N selection."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

MAX_CANDIDATES = 256
MAX_SAMPLE_COUNT = 10_000
MAX_SAMPLING_WEIGHT = 1_000_000
MAX_ABS_VERIFIER_SCORE = 1_000_000
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _fraction_payload(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


@dataclass(frozen=True, slots=True)
class VerifierCandidate:
    """One authored outcome class in a fixed i.i.d. sampling distribution."""

    candidate_id: str
    sampling_weight: int
    verifier_score: int
    target_success: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or _CANDIDATE_ID.fullmatch(
            self.candidate_id
        ) is None:
            raise ValueError(
                "candidate_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        if (
            type(self.sampling_weight) is not int
            or self.sampling_weight <= 0
            or self.sampling_weight > MAX_SAMPLING_WEIGHT
        ):
            raise ValueError(
                f"sampling_weight must be an integer in [1, {MAX_SAMPLING_WEIGHT}]"
            )
        if (
            type(self.verifier_score) is not int
            or abs(self.verifier_score) > MAX_ABS_VERIFIER_SCORE
        ):
            raise ValueError(
                "verifier_score must be an integer with absolute value at most "
                f"{MAX_ABS_VERIFIER_SCORE}"
            )
        if type(self.target_success) is not bool:
            raise ValueError("target_success must be boolean")


@dataclass(frozen=True, slots=True)
class CandidateSelectionProbability:
    """Exact probability that best-of-N selects one outcome class."""

    candidate: VerifierCandidate
    selection_rank: int
    sampling_probability: Fraction
    selection_probability: Fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "selection_rank": self.selection_rank,
            "sampling_weight": self.candidate.sampling_weight,
            "sampling_probability": _fraction_payload(self.sampling_probability),
            "verifier_score": self.candidate.verifier_score,
            "target_success": self.candidate.target_success,
            "selection_probability": _fraction_payload(self.selection_probability),
        }


@dataclass(frozen=True, slots=True)
class BestOfNAnalysis:
    """Exact selection, oracle, and proxy expectations for fixed best-of-N."""

    sample_count: int
    total_sampling_weight: int
    logical_candidate_sequences: int
    selections: tuple[CandidateSelectionProbability, ...]
    oracle_success_probability: Fraction
    selected_success_probability: Fraction
    oracle_selection_gap: Fraction
    expected_selected_verifier_score: Fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "logical_model_samples": self.sample_count,
            "logical_verifier_scores": self.sample_count,
            "logical_candidate_sequences": self.logical_candidate_sequences,
            "total_sampling_weight": self.total_sampling_weight,
            "tie_break": "maximum (verifier_score, candidate_id)",
            "oracle_success_probability": _fraction_payload(
                self.oracle_success_probability
            ),
            "selected_success_probability": _fraction_payload(
                self.selected_success_probability
            ),
            "oracle_selection_gap": _fraction_payload(self.oracle_selection_gap),
            "expected_selected_verifier_score": _fraction_payload(
                self.expected_selected_verifier_score
            ),
            "selections": [selection.to_dict() for selection in self.selections],
        }


def analyze_verifier_guided_best_of_n(
    candidates: Iterable[VerifierCandidate],
    *,
    sample_count: int,
) -> BestOfNAnalysis:
    """Analyze i.i.d. best-of-N without enumerating candidate sequences.

    Every draw comes from the same finite authored distribution. All sampled
    candidates are scored, and the maximum ``(verifier_score, candidate_id)``
    wins. The lexicographically larger canonical ID therefore wins score ties.
    ``oracle_success_probability`` asks whether at least one target-successful
    candidate was sampled; ``selected_success_probability`` asks whether the
    verifier-selected candidate is target-successful.
    """

    if (
        type(sample_count) is not int
        or sample_count <= 0
        or sample_count > MAX_SAMPLE_COUNT
    ):
        raise ValueError(
            f"sample_count must be an integer in [1, {MAX_SAMPLE_COUNT}]"
        )
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("at least one candidate is required")
    if len(candidate_tuple) > MAX_CANDIDATES:
        raise ValueError(f"candidate count cannot exceed {MAX_CANDIDATES}")
    if any(not isinstance(candidate, VerifierCandidate) for candidate in candidate_tuple):
        raise ValueError("all candidates must be VerifierCandidate instances")
    candidate_ids = [candidate.candidate_id for candidate in candidate_tuple]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_id values must be unique")

    total_weight = sum(candidate.sampling_weight for candidate in candidate_tuple)
    ordered_worst_to_best = sorted(
        candidate_tuple,
        key=lambda candidate: (candidate.verifier_score, candidate.candidate_id),
    )
    cumulative_weight = 0
    probability_by_id: dict[str, Fraction] = {}
    for candidate in ordered_worst_to_best:
        before = Fraction(cumulative_weight, total_weight) ** sample_count
        cumulative_weight += candidate.sampling_weight
        through = Fraction(cumulative_weight, total_weight) ** sample_count
        probability_by_id[candidate.candidate_id] = through - before

    selections = tuple(
        CandidateSelectionProbability(
            candidate=candidate,
            selection_rank=rank,
            sampling_probability=Fraction(candidate.sampling_weight, total_weight),
            selection_probability=probability_by_id[candidate.candidate_id],
        )
        for rank, candidate in enumerate(reversed(ordered_worst_to_best), start=1)
    )
    if sum(
        (selection.selection_probability for selection in selections),
        start=Fraction(0, 1),
    ) != Fraction(1, 1):
        raise AssertionError("selection probabilities must sum exactly to one")

    success_weight = sum(
        candidate.sampling_weight
        for candidate in candidate_tuple
        if candidate.target_success
    )
    oracle_success = Fraction(1, 1) - Fraction(
        total_weight - success_weight, total_weight
    ) ** sample_count
    selected_success = sum(
        (
            selection.selection_probability
            for selection in selections
            if selection.candidate.target_success
        ),
        start=Fraction(0, 1),
    )
    expected_verifier_score = sum(
        (
            selection.selection_probability * selection.candidate.verifier_score
            for selection in selections
        ),
        start=Fraction(0, 1),
    )
    if selected_success > oracle_success:
        raise AssertionError("selected success cannot exceed oracle candidate success")
    return BestOfNAnalysis(
        sample_count=sample_count,
        total_sampling_weight=total_weight,
        logical_candidate_sequences=len(candidate_tuple) ** sample_count,
        selections=selections,
        oracle_success_probability=oracle_success,
        selected_success_probability=selected_success,
        oracle_selection_gap=oracle_success - selected_success,
        expected_selected_verifier_score=expected_verifier_score,
    )
