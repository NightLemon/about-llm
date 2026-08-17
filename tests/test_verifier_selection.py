from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

from about_llm.inference import (
    VerifierCandidate,
    analyze_verifier_guided_best_of_n,
)

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "projects" / "inference-serving" / "verifier_best_of_n_toy.py"


@pytest.fixture
def candidates() -> tuple[VerifierCandidate, ...]:
    return (
        VerifierCandidate("wrong", 5, 20, False),
        VerifierCandidate("correct", 4, 80, True),
        VerifierCandidate("verifier_hack", 1, 99, False),
    )


def test_single_sample_matches_authored_distribution(
    candidates: tuple[VerifierCandidate, ...],
) -> None:
    analysis = analyze_verifier_guided_best_of_n(candidates, sample_count=1)
    probabilities = {
        selection.candidate.candidate_id: selection.selection_probability
        for selection in analysis.selections
    }

    assert probabilities == {
        "wrong": Fraction(1, 2),
        "correct": Fraction(2, 5),
        "verifier_hack": Fraction(1, 10),
    }
    assert analysis.oracle_success_probability == Fraction(2, 5)
    assert analysis.selected_success_probability == Fraction(2, 5)
    assert analysis.oracle_selection_gap == 0
    assert analysis.expected_selected_verifier_score == Fraction(519, 10)


def test_best_of_four_exactly_separates_oracle_and_selection(
    candidates: tuple[VerifierCandidate, ...],
) -> None:
    analysis = analyze_verifier_guided_best_of_n(candidates, sample_count=4)
    probabilities = {
        selection.candidate.candidate_id: selection.selection_probability
        for selection in analysis.selections
    }

    assert probabilities == {
        "wrong": Fraction(1, 16),
        "correct": Fraction(371, 625),
        "verifier_hack": Fraction(3439, 10_000),
    }
    assert analysis.oracle_success_probability == Fraction(544, 625)
    assert analysis.selected_success_probability == Fraction(371, 625)
    assert analysis.oracle_selection_gap == Fraction(173, 625)
    assert analysis.expected_selected_verifier_score == Fraction(827_841, 10_000)
    assert analysis.logical_candidate_sequences == 81


def test_larger_n_can_raise_proxy_score_while_target_success_collapses(
    candidates: tuple[VerifierCandidate, ...],
) -> None:
    one = analyze_verifier_guided_best_of_n(candidates, sample_count=1)
    four = analyze_verifier_guided_best_of_n(candidates, sample_count=4)
    sixteen = analyze_verifier_guided_best_of_n(candidates, sample_count=16)

    assert (
        one.expected_selected_verifier_score
        < four.expected_selected_verifier_score
        < sixteen.expected_selected_verifier_score
    )
    assert four.selected_success_probability > one.selected_success_probability
    assert sixteen.selected_success_probability < one.selected_success_probability
    assert (
        one.oracle_success_probability
        < four.oracle_success_probability
        < sixteen.oracle_success_probability
    )
    assert sixteen.selected_success_probability == (
        Fraction(9, 10) ** 16 - Fraction(1, 2) ** 16
    )
    assert sixteen.oracle_success_probability == 1 - Fraction(3, 5) ** 16
    assert sixteen.logical_candidate_sequences == 3**16


def test_equal_scores_use_explicit_candidate_id_tie_break() -> None:
    analysis = analyze_verifier_guided_best_of_n(
        (
            VerifierCandidate("C", 1, 0, False),
            VerifierCandidate("A", 1, 5, True),
            VerifierCandidate("B", 1, 5, False),
        ),
        sample_count=2,
    )

    assert [selection.candidate.candidate_id for selection in analysis.selections] == [
        "B",
        "A",
        "C",
    ]
    assert [selection.selection_probability for selection in analysis.selections] == [
        Fraction(5, 9),
        Fraction(1, 3),
        Fraction(1, 9),
    ]
    assert analysis.selected_success_probability == Fraction(1, 3)


def test_closed_form_matches_explicit_small_sequence_enumeration() -> None:
    candidates = (
        VerifierCandidate("low", 2, 1, False),
        VerifierCandidate("tie_a", 3, 7, True),
        VerifierCandidate("tie_b", 1, 7, False),
    )
    analysis = analyze_verifier_guided_best_of_n(candidates, sample_count=3)
    enumerated = {candidate.candidate_id: Fraction(0, 1) for candidate in candidates}
    total_weight = sum(candidate.sampling_weight for candidate in candidates)

    for sequence in product(candidates, repeat=3):
        winner = max(
            sequence,
            key=lambda candidate: (candidate.verifier_score, candidate.candidate_id),
        )
        sequence_weight = 1
        for candidate in sequence:
            sequence_weight *= candidate.sampling_weight
        enumerated[winner.candidate_id] += Fraction(
            sequence_weight, total_weight**3
        )

    assert {
        selection.candidate.candidate_id: selection.selection_probability
        for selection in analysis.selections
    } == enumerated


@pytest.mark.parametrize("target_success", [False, True])
def test_all_same_target_label_has_exact_boundary(target_success: bool) -> None:
    analysis = analyze_verifier_guided_best_of_n(
        (
            VerifierCandidate("low", 2, 1, target_success),
            VerifierCandidate("high", 1, 2, target_success),
        ),
        sample_count=7,
    )
    expected = Fraction(int(target_success), 1)

    assert analysis.oracle_success_probability == expected
    assert analysis.selected_success_probability == expected
    assert analysis.oracle_selection_gap == 0


@pytest.mark.parametrize("sample_count", [0, -1, True, 10_001])
def test_analysis_rejects_invalid_sample_count(sample_count: int) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        analyze_verifier_guided_best_of_n(
            (VerifierCandidate("only", 1, 1, True),),
            sample_count=sample_count,
        )


def test_analysis_rejects_empty_duplicate_and_wrong_candidate_types() -> None:
    with pytest.raises(ValueError, match="at least one"):
        analyze_verifier_guided_best_of_n((), sample_count=1)
    with pytest.raises(ValueError, match="unique"):
        analyze_verifier_guided_best_of_n(
            (
                VerifierCandidate("same", 1, 1, True),
                VerifierCandidate("same", 2, 2, False),
            ),
            sample_count=1,
        )
    with pytest.raises(ValueError, match="VerifierCandidate"):
        analyze_verifier_guided_best_of_n(
            ("not-a-candidate",),  # type: ignore[arg-type]
            sample_count=1,
        )


def test_analysis_rejects_candidate_count_above_resource_cap() -> None:
    candidates = tuple(
        VerifierCandidate(f"c{index:03d}", 1, index, False)
        for index in range(257)
    )

    with pytest.raises(ValueError, match="cannot exceed 256"):
        analyze_verifier_guided_best_of_n(candidates, sample_count=1)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("", 1, 1, True), "candidate_id"),
        (("bad id", 1, 1, True), "candidate_id"),
        (("x", 0, 1, True), "sampling_weight"),
        (("x", True, 1, True), "sampling_weight"),
        (("x", 1_000_001, 1, True), "sampling_weight"),
        (("x", 1, True, True), "verifier_score"),
        (("x", 1, 1_000_001, True), "verifier_score"),
        (("x", 1, 1, 1), "target_success"),
    ],
)
def test_candidate_rejects_noncanonical_or_unbounded_fields(
    args: tuple[object, object, object, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VerifierCandidate(*args)  # type: ignore[arg-type]


def test_toy_cli_reports_exact_counterexample_and_bounded_scope() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOY)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert completed.stderr == ""
    assert report["implementation"] == "about-llm.verifier-best-of-n-toy.v1"
    assert [analysis["sample_count"] for analysis in report["analyses"]] == [
        1,
        4,
        16,
    ]
    assert report["observations"] == {
        "expected_verifier_score_strictly_increases": True,
        "selected_success_n4_above_n1": True,
        "selected_success_n16_below_n1": True,
        "oracle_success_strictly_increases": True,
    }
    assert report["scope"] == {
        "authored_finite_candidate_distribution": True,
        "iid_fixed_distribution_assumed": True,
        "closed_form_exact_fraction_analysis_executed": True,
        "candidate_sequence_enumeration_executed": False,
        "oracle_target_labels_authored": True,
        "model_tokenizer_or_prm_executed": False,
        "verifier_calibration_or_semantic_correctness_proved": False,
        "latency_cost_parallelism_or_target_quality_measured": False,
        "target_model_provider_or_gpu_behavior_proved": False,
    }
