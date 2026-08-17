from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

from about_llm.evaluation import (
    analyze_repeated_two_sided_sign_tests,
    two_sided_sign_test_p_value,
)

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "projects" / "evaluation-gate" / "sequential_peeking_toy.py"


@pytest.mark.parametrize(
    ("positive_count", "expected"),
    [
        (0, Fraction(1, 512)),
        (1, Fraction(11, 512)),
        (2, Fraction(7, 64)),
        (5, Fraction(1, 1)),
        (8, Fraction(7, 64)),
        (9, Fraction(11, 512)),
        (10, Fraction(1, 512)),
    ],
)
def test_two_sided_sign_p_value_is_exact_and_symmetric(
    positive_count: int,
    expected: Fraction,
) -> None:
    assert (
        two_sided_sign_test_p_value(
            positive_count=positive_count,
            sample_count=10,
        )
        == expected
    )


def test_naive_repeated_testing_has_exact_inflated_type_one_error() -> None:
    result = analyze_repeated_two_sided_sign_tests(
        (10, 20, 30, 40, 50),
        per_look_alpha=Fraction(1, 20),
    )

    assert result.familywise_null_rejection_probability == Fraction(
        7_109_832_616_777,
        70_368_744_177_664,
    )
    assert result.familywise_null_rejection_probability > Fraction(1, 20)
    assert result.union_bound == Fraction(1, 4)
    assert result.logical_binary_sign_sequences == 2**50
    assert result.dynamic_programming_state_cells_evaluated < 2**50
    assert [look.sample_count for look in result.looks] == [10, 20, 30, 40, 50]
    assert [look.first_rejection_probability for look in result.looks] == [
        Fraction(11, 512),
        Fraction(8_601, 262_144),
        Fraction(12_097_545, 536_870_912),
        Fraction(8_034_089_379, 549_755_813_888),
        Fraction(675_177_912_777, 70_368_744_177_664),
    ]
    assert result.looks[-1].fixed_look_null_rejection_probability == Fraction(
        18_486_790_962_201,
        562_949_953_421_312,
    )


def test_prespecified_bonferroni_split_controls_union_bound() -> None:
    result = analyze_repeated_two_sided_sign_tests(
        (10, 20, 30, 40, 50),
        per_look_alpha=Fraction(1, 100),
    )

    assert result.familywise_null_rejection_probability == Fraction(
        2_142_139_082_367,
        140_737_488_355_328,
    )
    assert result.familywise_null_rejection_probability < Fraction(1, 20)
    assert result.union_bound == Fraction(1, 20)
    assert result.looks[0].lower_rejection_max_positive_count == 0
    assert result.looks[0].upper_rejection_min_positive_count == 10
    assert result.looks[-1].alive_probability_after_look == (
        1 - result.familywise_null_rejection_probability
    )


def test_dynamic_program_matches_explicit_small_sequence_enumeration() -> None:
    looks = (2, 4, 6)
    alpha = Fraction(1, 2)
    result = analyze_repeated_two_sided_sign_tests(
        looks,
        per_look_alpha=alpha,
    )
    rejected_sequences = 0
    for sequence in product((0, 1), repeat=looks[-1]):
        for sample_count in looks:
            positive_count = sum(sequence[:sample_count])
            if (
                two_sided_sign_test_p_value(
                    positive_count=positive_count,
                    sample_count=sample_count,
                )
                <= alpha
            ):
                rejected_sequences += 1
                break

    assert result.familywise_null_rejection_probability == Fraction(
        rejected_sequences,
        2 ** looks[-1],
    )


def test_no_rejection_region_is_represented_explicitly() -> None:
    result = analyze_repeated_two_sided_sign_tests(
        (1,),
        per_look_alpha=Fraction(1, 20),
    )
    look = result.looks[0]

    assert look.lower_rejection_max_positive_count is None
    assert look.upper_rejection_min_positive_count is None
    assert look.fixed_look_null_rejection_probability == 0
    assert result.familywise_null_rejection_probability == 0


@pytest.mark.parametrize(
    "look_sample_counts",
    [
        (),
        (0,),
        (True,),
        (513,),
        (10, 10),
        (20, 10),
        tuple(range(1, 66)),
    ],
)
def test_analysis_rejects_invalid_or_unbounded_look_schedules(
    look_sample_counts: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        analyze_repeated_two_sided_sign_tests(
            look_sample_counts,
            per_look_alpha=Fraction(1, 20),
        )


@pytest.mark.parametrize(
    "alpha",
    [0.05, 0, 1, True, Fraction(0, 1), Fraction(1, 1), Fraction(-1, 20)],
)
def test_analysis_requires_exact_fraction_alpha_strictly_between_zero_and_one(
    alpha: object,
) -> None:
    with pytest.raises(ValueError, match="Fraction"):
        analyze_repeated_two_sided_sign_tests(
            (10,),
            per_look_alpha=alpha,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("positive_count", "sample_count"),
    [(-1, 10), (11, 10), (True, 10), (0, 0), (0, True), (0, 513)],
)
def test_sign_test_rejects_invalid_counts(
    positive_count: int,
    sample_count: int,
) -> None:
    with pytest.raises(ValueError):
        two_sided_sign_test_p_value(
            positive_count=positive_count,
            sample_count=sample_count,
        )


def test_toy_reports_exact_peeking_counterexample_and_scope() -> None:
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
    assert report["implementation"] == "about-llm.sequential-peeking-toy.v1"
    assert report["observations"] == {
        "bonferroni_familywise_error_at_most_five_percent": True,
        "naive_familywise_error_exceeds_five_percent": True,
        "same_look_schedule": True,
    }
    naive = report["scenarios"]["naive_alpha_at_every_look"]
    bonferroni = report["scenarios"]["prespecified_bonferroni_alpha_split"]
    assert naive["familywise_null_rejection_probability"] == {
        "numerator": 7_109_832_616_777,
        "denominator": 70_368_744_177_664,
        "decimal": pytest.approx(0.10103679836642243),
    }
    assert bonferroni["familywise_null_rejection_probability"] == {
        "numerator": 2_142_139_082_367,
        "denominator": 140_737_488_355_328,
        "decimal": pytest.approx(0.015220813639636788),
    }
    assert naive["logical_binary_sign_sequences"] == 2**50
    assert report["scope"] == {
        "case_sampling_labels_clusters_or_exchangeability_validated": False,
        "confidence_sequence_or_always_valid_p_value_implemented": False,
        "effect_size_power_or_sample_size_estimated": False,
        "exact_fraction_dynamic_program_executed": True,
        "logical_sign_sequence_enumeration_executed": False,
        "look_schedule_and_thresholds_prespecified": True,
        "model_judge_provider_or_online_ab_test_executed": False,
    }
