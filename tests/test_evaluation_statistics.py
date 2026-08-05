from __future__ import annotations

import numpy as np
import pytest

from about_llm.evaluation import ReleaseGate, paired_bootstrap


def test_paired_bootstrap_is_reproducible_and_detects_improvement() -> None:
    baseline = np.array([0, 0, 1, 0, 1, 0, 0, 1], dtype=float)
    candidate = np.ones_like(baseline)
    first = paired_bootstrap(baseline, candidate, samples=2_000, seed=7)
    second = paired_bootstrap(baseline, candidate, samples=2_000, seed=7)

    assert first == second
    assert first.mean_difference == pytest.approx(0.625)
    assert first.confidence_low > 0
    assert first.probability_of_improvement > 0.99


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
