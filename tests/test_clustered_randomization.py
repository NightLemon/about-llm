from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from about_llm.evaluation import (
    clustered_paired_randomization_test,
    paired_randomization_test,
)

ROOT = Path(__file__).resolve().parents[1]


def test_joint_cluster_flip_avoids_treating_repeated_cases_as_six_units() -> None:
    baseline = [0, 0, 0, 0, 0, 0]
    candidate = [1, 1, 1, 1, 1, -1]
    clusters = ["user-a"] * 5 + ["user-b"]

    naive = paired_randomization_test(
        baseline, candidate, alternative="greater"
    )
    clustered = clustered_paired_randomization_test(
        baseline,
        candidate,
        clusters,
        cluster_weighting="case",
        alternative="greater",
    )

    assert naive.assignments_evaluated == 64
    assert naive.p_value == pytest.approx(7 / 64)
    assert clustered.case_count == 6
    assert clustered.cluster_count == 2
    assert clustered.cluster_sizes == (5, 1)
    assert clustered.nonzero_cluster_count == 2
    assert clustered.mean_difference == pytest.approx(4 / 6)
    assert clustered.assignments_evaluated == 4
    assert clustered.extreme_assignments == 2
    assert clustered.p_value == pytest.approx(2 / 4)


def test_case_and_equal_cluster_weighting_target_different_estimands() -> None:
    baseline = [0, 0, 0, 0, 0, 0]
    candidate = [1, 1, 1, 1, 1, -1]
    clusters = ["large"] * 5 + ["small"]

    case_weighted = clustered_paired_randomization_test(
        baseline, candidate, clusters, cluster_weighting="case"
    )
    equal_cluster = clustered_paired_randomization_test(
        baseline, candidate, clusters, cluster_weighting="equal"
    )

    assert case_weighted.candidate_estimand == pytest.approx(4 / 6)
    assert case_weighted.mean_difference == pytest.approx(4 / 6)
    assert equal_cluster.candidate_estimand == 0
    assert equal_cluster.mean_difference == 0
    assert equal_cluster.p_value == 1


def test_zero_cluster_contribution_stays_in_estimand_but_not_enumeration() -> None:
    result = clustered_paired_randomization_test(
        [0, 0, 0],
        [1, -1, 1],
        ["cancelled", "cancelled", "positive"],
        cluster_weighting="case",
        alternative="greater",
    )

    assert result.cluster_count == 2
    assert result.nonzero_cluster_count == 1
    assert result.zero_contribution_cluster_count == 1
    assert result.mean_difference == pytest.approx(1 / 3)
    assert result.assignments_evaluated == 2
    assert result.p_value == pytest.approx(1 / 2)


def test_case_order_within_clusters_does_not_change_inference() -> None:
    first = clustered_paired_randomization_test(
        [0, 0, 0, 0],
        [2, -1, 0.5, 1],
        ["a", "a", "b", "b"],
    )
    reordered = clustered_paired_randomization_test(
        [0, 0, 0, 0],
        [1, 2, 0.5, -1],
        ["b", "a", "b", "a"],
    )

    assert first.mean_difference == pytest.approx(reordered.mean_difference)
    assert first.p_value == reordered.p_value
    assert first.assignments_evaluated == reordered.assignments_evaluated


def test_clustered_monte_carlo_is_seeded_and_uses_plus_one_correction() -> None:
    kwargs = {
        "alternative": "greater",
        "exact_max_nonzero_clusters": 1,
        "monte_carlo_samples": 1_000,
        "seed": 13,
    }
    first = clustered_paired_randomization_test(
        [0, 0], [1, 1], ["a", "b"], **kwargs
    )
    second = clustered_paired_randomization_test(
        [0, 0], [1, 1], ["a", "b"], **kwargs
    )

    assert first == second
    assert first.method == "monte_carlo"
    assert first.p_value == pytest.approx((first.extreme_assignments + 1) / 1_001)
    assert first.p_value_resolution == pytest.approx(1 / 1_001)
    assert first.seed == 13


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: clustered_paired_randomization_test([0], [1], []),
            "one value per paired case",
        ),
        (
            lambda: clustered_paired_randomization_test([0], [1], [""]),
            "non-empty strings",
        ),
        (
            lambda: clustered_paired_randomization_test([0], [1], "cluster"),
            "sequence",
        ),
        (
            lambda: clustered_paired_randomization_test(
                [0], [1], ["a"], cluster_weighting="size"
            ),
            "cluster_weighting",
        ),
        (
            lambda: clustered_paired_randomization_test(
                [0], [1], ["a"], exact_max_nonzero_clusters=25
            ),
            "exact_max_nonzero_clusters",
        ),
        (
            lambda: clustered_paired_randomization_test(
                [0], [1], ["a"], monte_carlo_samples=0
            ),
            "monte_carlo_samples",
        ),
        (
            lambda: clustered_paired_randomization_test(
                [0], [1], ["a"], seed=True
            ),
            "seed",
        ),
    ],
)
def test_invalid_clustered_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


