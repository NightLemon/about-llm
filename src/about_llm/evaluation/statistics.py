"""Statistical comparisons for paired model-evaluation outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PairedBootstrapResult:
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    confidence_low: float
    confidence_high: float
    probability_of_improvement: float


def paired_bootstrap(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Bootstrap paired case-level metric differences.

    Pairing preserves the fact that both systems answered the same cases and
    usually gives a tighter, more relevant comparison than independent means.
    """
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if baseline_array.ndim != 1 or candidate_array.ndim != 1:
        raise ValueError("baseline and candidate must be one-dimensional")
    if baseline_array.shape != candidate_array.shape or baseline_array.size == 0:
        raise ValueError("baseline and candidate must have the same non-zero length")
    if not np.all(np.isfinite(baseline_array)) or not np.all(np.isfinite(candidate_array)):
        raise ValueError("scores must be finite")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if samples <= 0:
        raise ValueError("samples must be positive")

    differences = candidate_array - baseline_array
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    bootstrapped_means = differences[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(bootstrapped_means, [alpha, 1 - alpha])
    return PairedBootstrapResult(
        baseline_mean=float(baseline_array.mean()),
        candidate_mean=float(candidate_array.mean()),
        mean_difference=float(differences.mean()),
        confidence_low=float(low),
        confidence_high=float(high),
        probability_of_improvement=float(np.mean(bootstrapped_means > 0)),
    )


@dataclass(frozen=True)
class ReleaseGate:
    """A transparent gate over quality, safety, and latency deltas."""

    minimum_quality_difference: float = 0.0
    maximum_safety_regression: float = 0.0
    maximum_latency_increase_fraction: float = 0.10

    def evaluate(
        self,
        *,
        quality: PairedBootstrapResult,
        safety_difference: float,
        baseline_latency: float,
        candidate_latency: float,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if quality.confidence_low < self.minimum_quality_difference:
            reasons.append(
                "quality confidence lower bound "
                f"{quality.confidence_low:.4f} < {self.minimum_quality_difference:.4f}"
            )
        if safety_difference < -self.maximum_safety_regression:
            reasons.append(f"safety difference {safety_difference:.4f} exceeds allowed regression")
        if baseline_latency <= 0 or candidate_latency <= 0:
            raise ValueError("latencies must be positive")
        latency_increase = candidate_latency / baseline_latency - 1
        if latency_increase > self.maximum_latency_increase_fraction:
            reasons.append(
                f"latency increase {latency_increase:.1%} exceeds "
                f"{self.maximum_latency_increase_fraction:.1%}"
            )
        return not reasons, reasons
