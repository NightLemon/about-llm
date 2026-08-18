from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from about_llm.evaluation import clustered_paired_bootstrap, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.formula


def test_exact_case_weighted_cluster_bootstrap_uses_ratio_per_resample() -> None:
    result = clustered_paired_bootstrap(
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, -1],
        ["large"] * 5 + ["small"],
        cluster_weighting="case",
    )

    assert result.cluster_sizes == (5, 1)
    assert result.mean_difference == pytest.approx(4 / 6)
    assert result.method == "exact"
    assert result.resamples_evaluated == 4
    assert result.confidence_low == pytest.approx(-0.875)
    assert result.confidence_high == pytest.approx(0.975)
    assert result.probability_of_improvement == pytest.approx(3 / 4)
    assert result.quantile_method == "linear"
    assert result.seed is None


def test_equal_cluster_bootstrap_targets_mean_of_cluster_means() -> None:
    result = clustered_paired_bootstrap(
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, -1],
        ["large"] * 5 + ["small"],
        cluster_weighting="equal",
    )

    assert result.mean_difference == 0
    assert result.confidence_low == pytest.approx(-0.925)
    assert result.confidence_high == pytest.approx(0.925)
    assert result.probability_of_improvement == pytest.approx(1 / 4)


def test_exact_ordered_resamples_match_independent_brute_force() -> None:
    differences = np.asarray([2.0, -1.0, 0.5])
    expected = np.asarray(
        [
            differences[list(indices)].mean()
            for indices in itertools.product(range(3), repeat=3)
        ]
    )
    result = clustered_paired_bootstrap(
        [0, 0, 0], differences, ["a", "b", "c"]
    )
    low, high = np.quantile(expected, [0.025, 0.975], method="linear")

    assert result.resamples_evaluated == 27
    assert result.confidence_low == pytest.approx(low)
    assert result.confidence_high == pytest.approx(high)
    assert result.probability_of_improvement == pytest.approx(np.mean(expected > 0))


def test_case_reordering_does_not_change_cluster_bootstrap_result() -> None:
    first = clustered_paired_bootstrap(
        [0, 0, 0, 0],
        [2, -1, 0.5, 1],
        ["a", "a", "b", "b"],
    )
    reordered = clustered_paired_bootstrap(
        [0, 0, 0, 0],
        [1, 2, 0.5, -1],
        ["b", "a", "b", "a"],
    )

    assert first.mean_difference == pytest.approx(reordered.mean_difference)
    assert first.confidence_low == pytest.approx(reordered.confidence_low)
    assert first.confidence_high == pytest.approx(reordered.confidence_high)
    assert first.probability_of_improvement == pytest.approx(
        reordered.probability_of_improvement
    )


def test_monte_carlo_cluster_bootstrap_is_seeded_and_records_sampling_budget() -> None:
    kwargs = {
        "exact_max_clusters": 1,
        "monte_carlo_samples": 2_000,
        "seed": 17,
    }
    first = clustered_paired_bootstrap(
        [0, 0, 0], [1, -1, 2], ["a", "b", "c"], **kwargs
    )
    second = clustered_paired_bootstrap(
        [0, 0, 0], [1, -1, 2], ["a", "b", "c"], **kwargs
    )

    assert first == second
    assert first.method == "monte_carlo"
    assert first.resamples_evaluated == 2_000
    assert first.seed == 17


def test_one_observed_cluster_has_degenerate_empirical_bootstrap() -> None:
    result = clustered_paired_bootstrap(
        [0, 0], [1, 3], ["only", "only"], cluster_weighting="equal"
    )

    assert result.resamples_evaluated == 1
    assert result.mean_difference == 2
    assert result.confidence_low == 2
    assert result.confidence_high == 2
    assert result.probability_of_improvement == 1


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: clustered_paired_bootstrap([0], [1], []),
            "one value per paired case",
        ),
        (
            lambda: clustered_paired_bootstrap([0], [1], [""]),
            "non-empty strings",
        ),
        (
            lambda: clustered_paired_bootstrap(
                [0], [1], ["a"], cluster_weighting="size"
            ),
            "cluster_weighting",
        ),
        (
            lambda: clustered_paired_bootstrap(
                [0], [1], ["a"], confidence=float("nan")
            ),
            "confidence",
        ),
        (
            lambda: clustered_paired_bootstrap(
                [0], [1], ["a"], exact_max_clusters=8
            ),
            "exact_max_clusters",
        ),
        (
            lambda: clustered_paired_bootstrap(
                [0], [1], ["a"], monte_carlo_samples=True
            ),
            "monte_carlo_samples",
        ),
        (
            lambda: clustered_paired_bootstrap([0], [1], ["a"], seed=True),
            "seed",
        ),
        (lambda: paired_bootstrap([0], [1], samples=True), "samples"),
        (lambda: paired_bootstrap([0], [1], confidence=True), "confidence"),
        (lambda: paired_bootstrap([0], [1], seed=True), "seed"),
    ],
)
def test_invalid_bootstrap_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()

