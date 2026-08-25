from __future__ import annotations

import numpy as np
import pytest

from about_llm.evaluation import ReleaseGate, paired_bootstrap

pytestmark = pytest.mark.formula


def test_paired_bootstrap_constant_difference_has_exact_degenerate_oracle() -> None:
    result = paired_bootstrap(
        baseline=[-2.0, 0.0, 7.0],
        candidate=[-1.75, 0.25, 7.25],
        samples=17,
        seed=91,
    )

    assert result.baseline_mean == pytest.approx(5 / 3)
    assert result.candidate_mean == pytest.approx(23 / 12)
    assert result.mean_difference == pytest.approx(0.25)
    assert result.confidence_low == pytest.approx(0.25)
    assert result.confidence_high == pytest.approx(0.25)
    assert result.probability_of_improvement == 1


def test_paired_bootstrap_counts_zero_difference_as_not_strictly_improved() -> None:
    result = paired_bootstrap([1.0, 2.0], [1.0, 2.0], samples=11, seed=3)

    assert result.mean_difference == 0
    assert result.confidence_low == 0
    assert result.confidence_high == 0
    assert result.probability_of_improvement == 0


def test_paired_bootstrap_is_reproducible_and_detects_improvement() -> None:
    baseline = np.array([0, 0, 1, 0, 1, 0, 0, 1], dtype=float)
    candidate = np.ones_like(baseline)
    first = paired_bootstrap(baseline, candidate, samples=2_000, seed=7)
    second = paired_bootstrap(baseline, candidate, samples=2_000, seed=7)

    assert first == second
    assert first.mean_difference == pytest.approx(0.625)
    assert first.confidence_low > 0
    assert first.probability_of_improvement > 0.99


def test_documented_five_case_bootstrap_trace_stays_reproducible() -> None:
    result = paired_bootstrap(
        baseline=[0, 0, 0, 0, 1],
        candidate=[1, 1, 1, 1, 1],
        samples=10_000,
        seed=7,
    )

    assert result.mean_difference == pytest.approx(0.8)
    assert result.confidence_low == pytest.approx(0.4)
    assert result.confidence_high == pytest.approx(1.0)
    assert result.probability_of_improvement == pytest.approx(0.9995)


def test_release_gate_reports_every_failed_constraint() -> None:
    quality = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], samples=500)
    gate = ReleaseGate(
        minimum_quality_difference=0.01,
        maximum_safety_regression=0.01,
        maximum_latency_increase_fraction=0.1,
    )
    passed, reasons = gate.evaluate(
        quality=quality,
        safety_difference=-0.02,
        baseline_latency=1.0,
        candidate_latency=1.2,
    )
    assert not passed
    assert len(reasons) == 3


@pytest.mark.parametrize(
    "gate",
    [
        {"minimum_quality_difference": float("nan")},
        {"maximum_safety_regression": -0.01},
        {"maximum_latency_increase_fraction": -1.0},
    ],
)
def test_release_gate_rejects_semantically_invalid_configuration(
    gate: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        ReleaseGate(**gate)  # type: ignore[arg-type]
