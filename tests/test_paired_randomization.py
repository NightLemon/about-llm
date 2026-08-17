from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from about_llm.evaluation import paired_randomization_test

ROOT = Path(__file__).resolve().parents[1]


def test_exact_one_sided_and_two_sided_p_values_include_observed_assignment() -> None:
    baseline = [0, 0, 0, 0]
    candidate = [1, 1, 1, 1]

    greater = paired_randomization_test(
        baseline, candidate, alternative="greater"
    )
    two_sided = paired_randomization_test(
        baseline, candidate, alternative="two-sided"
    )

    assert greater.method == "exact"
    assert greater.assignments_evaluated == 16
    assert greater.extreme_assignments == 1
    assert greater.p_value == pytest.approx(1 / 16)
    assert greater.p_value_resolution == pytest.approx(1 / 16)
    assert two_sided.extreme_assignments == 2
    assert two_sided.p_value == pytest.approx(2 / 16)


def test_zero_differences_stay_in_mean_but_do_not_duplicate_sign_assignments() -> None:
    result = paired_randomization_test(
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        alternative="greater",
    )

    assert result.pair_count == 5
    assert result.nonzero_pair_count == 4
    assert result.zero_difference_count == 1
    assert result.mean_difference == pytest.approx(0.8)
    assert result.assignments_evaluated == 16
    assert result.p_value == pytest.approx(1 / 16)


def test_all_zero_differences_have_one_unique_assignment_and_p_value_one() -> None:
    result = paired_randomization_test([1, 0, 1], [1, 0, 1])

    assert result.nonzero_pair_count == 0
    assert result.assignments_evaluated == 1
    assert result.extreme_assignments == 1
    assert result.p_value == 1


def test_exact_result_matches_small_brute_force_distribution() -> None:
    differences = np.array([2.0, -1.0, 0.5])
    observed = float(differences.mean())
    statistics = [
        float(np.mean(differences * np.array(signs)))
        for signs in itertools.product((-1, 1), repeat=3)
    ]
    expected = sum(abs(value) >= abs(observed) for value in statistics) / len(
        statistics
    )

    result = paired_randomization_test(
        [0, 0, 0], differences, alternative="two-sided"
    )

    assert result.p_value == pytest.approx(expected)


def test_swapping_systems_preserves_two_sided_and_swaps_one_sided_direction() -> None:
    baseline = [0.1, 0.8, 0.2, 0.4]
    candidate = [0.4, 0.7, 0.9, 0.5]

    forward_two = paired_randomization_test(baseline, candidate)
    reverse_two = paired_randomization_test(candidate, baseline)
    forward_greater = paired_randomization_test(
        baseline, candidate, alternative="greater"
    )
    reverse_less = paired_randomization_test(
        candidate, baseline, alternative="less"
    )

    assert forward_two.mean_difference == pytest.approx(-reverse_two.mean_difference)
    assert forward_two.p_value == reverse_two.p_value
    assert forward_greater.p_value == reverse_less.p_value


def test_monte_carlo_path_is_seeded_and_uses_plus_one_correction() -> None:
    arguments = {
        "alternative": "greater",
        "exact_max_nonzero_pairs": 2,
        "monte_carlo_samples": 1_000,
        "seed": 17,
    }
    first = paired_randomization_test([0] * 4, [1] * 4, **arguments)
    second = paired_randomization_test([0] * 4, [1] * 4, **arguments)

    assert first == second
    assert first.method == "monte_carlo"
    assert first.assignments_evaluated == 1_000
    assert first.p_value == pytest.approx(
        (first.extreme_assignments + 1) / 1_001
    )
    assert first.p_value_resolution == pytest.approx(1 / 1_001)
    assert first.seed == 17
    assert first.p_value > 0


def test_additive_score_shift_does_not_change_paired_differences() -> None:
    baseline = np.array([0.1, 0.5, 0.2])
    candidate = np.array([0.4, 0.4, 0.8])

    original = paired_randomization_test(baseline, candidate)
    shifted = paired_randomization_test(baseline + 1000, candidate + 1000)

    assert original.mean_difference == pytest.approx(shifted.mean_difference)
    assert original.p_value == shifted.p_value


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (
            lambda: paired_randomization_test([], []),
            ValueError,
            "same non-zero length",
        ),
        (
            lambda: paired_randomization_test([0], [0, 1]),
            ValueError,
            "same non-zero length",
        ),
        (
            lambda: paired_randomization_test([[0]], [[1]]),
            ValueError,
            "one-dimensional",
        ),
        (
            lambda: paired_randomization_test([0], [float("nan")]),
            ValueError,
            "finite",
        ),
        (
            lambda: paired_randomization_test([0], [1], alternative="up"),
            ValueError,
            "alternative",
        ),
        (
            lambda: paired_randomization_test(
                [0], [1], exact_max_nonzero_pairs=-1
            ),
            ValueError,
            "exact_max_nonzero_pairs",
        ),
        (
            lambda: paired_randomization_test(
                [0], [1], exact_max_nonzero_pairs=25
            ),
            ValueError,
            "exact_max_nonzero_pairs",
        ),
        (
            lambda: paired_randomization_test([0], [1], monte_carlo_samples=0),
            ValueError,
            "monte_carlo_samples",
        ),
        (
            lambda: paired_randomization_test([0], [1], seed=True),
            ValueError,
            "seed",
        ),
    ],
)
def test_invalid_randomization_contracts_fail_closed(
    operation: Callable[[], object], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        operation()


