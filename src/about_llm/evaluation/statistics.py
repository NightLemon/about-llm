"""Statistical comparisons for paired model-evaluation outcomes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

RandomizationAlternative = Literal["two-sided", "greater", "less"]
ClusterWeighting = Literal["case", "equal"]
_MAX_EXACT_SIGN_FLIP_UNITS = 24
_MAX_SIGN_MATRIX_ELEMENTS = 1_000_000
_MAX_EXACT_BOOTSTRAP_CLUSTERS = 7
_MAX_BOOTSTRAP_MATRIX_ELEMENTS = 1_000_000


def _paired_score_arrays(
    baseline: ArrayLike, candidate: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if baseline_array.ndim != 1 or candidate_array.ndim != 1:
        raise ValueError("baseline and candidate must be one-dimensional")
    if baseline_array.shape != candidate_array.shape or baseline_array.size == 0:
        raise ValueError("baseline and candidate must have the same non-zero length")
    if not np.all(np.isfinite(baseline_array)) or not np.all(
        np.isfinite(candidate_array)
    ):
        raise ValueError("scores must be finite")
    return baseline_array, candidate_array


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
    baseline_array, candidate_array = _paired_score_arrays(baseline, candidate)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be a finite number in (0, 1)")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    differences = candidate_array - baseline_array
    rng = np.random.default_rng(seed)
    bootstrapped_means = np.empty(samples, dtype=np.float64)
    block_size = max(
        1,
        min(8192, _MAX_BOOTSTRAP_MATRIX_ELEMENTS // differences.size),
    )
    for start in range(0, samples, block_size):
        stop = min(start + block_size, samples)
        indices = rng.integers(
            0,
            differences.size,
            size=(stop - start, differences.size),
        )
        bootstrapped_means[start:stop] = differences[indices].mean(axis=1)
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
class ClusteredPairedBootstrapResult:
    """Percentile cluster-bootstrap interval for one explicit estimand."""

    case_count: int
    cluster_count: int
    cluster_sizes: tuple[int, ...]
    cluster_weighting: ClusterWeighting
    baseline_estimand: float
    candidate_estimand: float
    mean_difference: float
    confidence: float
    confidence_low: float
    confidence_high: float
    probability_of_improvement: float
    method: Literal["exact", "monte_carlo"]
    resamples_evaluated: int
    quantile_method: Literal["linear"]
    seed: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "cluster_count": self.cluster_count,
            "cluster_sizes": list(self.cluster_sizes),
            "cluster_weighting": self.cluster_weighting,
            "baseline_estimand": self.baseline_estimand,
            "candidate_estimand": self.candidate_estimand,
            "mean_difference": self.mean_difference,
            "confidence": self.confidence,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "probability_of_improvement": self.probability_of_improvement,
            "method": self.method,
            "resamples_evaluated": self.resamples_evaluated,
            "quantile_method": self.quantile_method,
            "seed": self.seed,
        }


def _validated_cluster_positions(
    cluster_ids: Sequence[str], case_count: int
) -> dict[str, list[int]]:
    if isinstance(cluster_ids, (str, bytes)):
        raise ValueError("cluster_ids must be a sequence of non-empty strings")
    cluster_id_values = tuple(cluster_ids)
    if len(cluster_id_values) != case_count:
        raise ValueError("cluster_ids must have one value per paired case")
    if any(not isinstance(value, str) or not value for value in cluster_id_values):
        raise ValueError("cluster_ids must be non-empty strings")

    cluster_positions: dict[str, list[int]] = {}
    for case_index, cluster_id in enumerate(cluster_id_values):
        cluster_positions.setdefault(cluster_id, []).append(case_index)
    return cluster_positions


def _cluster_bootstrap_statistics(
    difference_sums: NDArray[np.float64],
    cluster_sizes: NDArray[np.int64],
    difference_means: NDArray[np.float64],
    sampled_clusters: NDArray[np.int64],
    cluster_weighting: ClusterWeighting,
) -> NDArray[np.float64]:
    if cluster_weighting == "case":
        numerators = difference_sums[sampled_clusters].sum(axis=1)
        denominators = cluster_sizes[sampled_clusters].sum(axis=1)
        return np.asarray(numerators / denominators, dtype=np.float64)
    return np.asarray(
        difference_means[sampled_clusters].mean(axis=1), dtype=np.float64
    )


def clustered_paired_bootstrap(
    baseline: ArrayLike,
    candidate: ArrayLike,
    cluster_ids: Sequence[str],
    *,
    cluster_weighting: ClusterWeighting = "case",
    confidence: float = 0.95,
    exact_max_clusters: int = 6,
    monte_carlo_samples: int = 10_000,
    seed: int = 0,
) -> ClusteredPairedBootstrapResult:
    """Resample whole clusters for a paired percentile-bootstrap interval.

    Each resample draws ``G`` clusters with replacement from the ``G`` observed
    clusters and keeps every case in a selected cluster together. For
    ``cluster_weighting='case'``, every resample is a ratio of sampled cluster
    difference sums to sampled cluster sizes. For ``'equal'``, it is the mean
    of sampled cluster mean differences. These are different estimands.

    Up to ``exact_max_clusters`` the function enumerates all ``G**G`` ordered
    resamples, which gives multinomial bootstrap multiplicities exactly. Larger
    inputs use seeded Monte Carlo. The returned interval is the percentile
    interval with NumPy's linear quantile interpolation; it is not BCa or a
    small-cluster guarantee and does not establish representative sampling,
    cluster independence, a valid cluster definition, or causal improvement.
    """

    baseline_array, candidate_array = _paired_score_arrays(baseline, candidate)
    cluster_positions = _validated_cluster_positions(
        cluster_ids, int(baseline_array.size)
    )
    if cluster_weighting not in ("case", "equal"):
        raise ValueError("cluster_weighting must be 'case' or 'equal'")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be a finite number in (0, 1)")
    if (
        isinstance(exact_max_clusters, bool)
        or not isinstance(exact_max_clusters, int)
        or not 0 <= exact_max_clusters <= _MAX_EXACT_BOOTSTRAP_CLUSTERS
    ):
        raise ValueError(
            "exact_max_clusters must be an integer in "
            f"[0, {_MAX_EXACT_BOOTSTRAP_CLUSTERS}]"
        )
    if (
        isinstance(monte_carlo_samples, bool)
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples <= 0
    ):
        raise ValueError("monte_carlo_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    differences = candidate_array - baseline_array
    positions = tuple(cluster_positions.values())
    cluster_sizes = np.asarray([len(indices) for indices in positions], dtype=np.int64)
    difference_sums = np.asarray(
        [float(differences[indices].sum()) for indices in positions],
        dtype=np.float64,
    )
    difference_means = difference_sums / cluster_sizes
    cluster_count = int(cluster_sizes.size)

    if cluster_weighting == "case":
        baseline_estimand = float(baseline_array.mean())
        candidate_estimand = float(candidate_array.mean())
        observed = float(differences.mean())
    else:
        baseline_cluster_means = np.asarray(
            [float(baseline_array[indices].mean()) for indices in positions],
            dtype=np.float64,
        )
        candidate_cluster_means = np.asarray(
            [float(candidate_array[indices].mean()) for indices in positions],
            dtype=np.float64,
        )
        baseline_estimand = float(baseline_cluster_means.mean())
        candidate_estimand = float(candidate_cluster_means.mean())
        observed = float(difference_means.mean())

    if cluster_count <= exact_max_clusters:
        resamples = cluster_count**cluster_count
        statistics = np.empty(resamples, dtype=np.float64)
        block_size = 8192
        for start in range(0, resamples, block_size):
            stop = min(start + block_size, resamples)
            codes = np.arange(start, stop, dtype=np.int64)
            sampled_clusters = np.empty(
                (stop - start, cluster_count), dtype=np.int64
            )
            working = codes.copy()
            for column in range(cluster_count):
                sampled_clusters[:, column] = working % cluster_count
                working //= cluster_count
            statistics[start:stop] = _cluster_bootstrap_statistics(
                difference_sums,
                cluster_sizes,
                difference_means,
                sampled_clusters,
                cluster_weighting,
            )
        method: Literal["exact", "monte_carlo"] = "exact"
        result_seed: int | None = None
    else:
        resamples = monte_carlo_samples
        statistics = np.empty(resamples, dtype=np.float64)
        rng = np.random.default_rng(seed)
        block_size = max(
            1,
            min(8192, _MAX_BOOTSTRAP_MATRIX_ELEMENTS // cluster_count),
        )
        for start in range(0, resamples, block_size):
            stop = min(start + block_size, resamples)
            sampled_clusters = rng.integers(
                0,
                cluster_count,
                size=(stop - start, cluster_count),
            )
            statistics[start:stop] = _cluster_bootstrap_statistics(
                difference_sums,
                cluster_sizes,
                difference_means,
                sampled_clusters,
                cluster_weighting,
            )
        method = "monte_carlo"
        result_seed = seed

    alpha = (1 - confidence) / 2
    low, high = np.quantile(statistics, [alpha, 1 - alpha], method="linear")
    return ClusteredPairedBootstrapResult(
        case_count=int(differences.size),
        cluster_count=cluster_count,
        cluster_sizes=tuple(int(value) for value in cluster_sizes),
        cluster_weighting=cluster_weighting,
        baseline_estimand=baseline_estimand,
        candidate_estimand=candidate_estimand,
        mean_difference=observed,
        confidence=float(confidence),
        confidence_low=float(low),
        confidence_high=float(high),
        probability_of_improvement=float(np.mean(statistics > 0)),
        method=method,
        resamples_evaluated=resamples,
        quantile_method="linear",
        seed=result_seed,
    )


@dataclass(frozen=True)
class PairedRandomizationResult:
    """Case-level sign-flip randomization result for one prespecified contrast."""

    pair_count: int
    nonzero_pair_count: int
    zero_difference_count: int
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    alternative: RandomizationAlternative
    method: Literal["exact", "monte_carlo"]
    assignments_evaluated: int
    extreme_assignments: int
    p_value: float
    p_value_resolution: float
    seed: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "nonzero_pair_count": self.nonzero_pair_count,
            "zero_difference_count": self.zero_difference_count,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "mean_difference": self.mean_difference,
            "alternative": self.alternative,
            "method": self.method,
            "assignments_evaluated": self.assignments_evaluated,
            "extreme_assignments": self.extreme_assignments,
            "p_value": self.p_value,
            "p_value_resolution": self.p_value_resolution,
            "seed": self.seed,
        }


def _extreme_count(
    statistics: NDArray[np.float64],
    observed: float,
    alternative: RandomizationAlternative,
) -> int:
    tolerance = 1e-15 * max(1.0, abs(observed))
    if alternative == "greater":
        extreme = statistics >= observed - tolerance
    elif alternative == "less":
        extreme = statistics <= observed + tolerance
    else:
        extreme = np.abs(statistics) >= abs(observed) - tolerance
    return int(np.count_nonzero(extreme))


@dataclass(frozen=True)
class _SignFlipOutcome:
    method: Literal["exact", "monte_carlo"]
    assignments_evaluated: int
    extreme_assignments: int
    p_value: float
    p_value_resolution: float
    seed: int | None


def _sign_flip_outcome(
    nonzero_contributions: NDArray[np.float64],
    *,
    denominator: float,
    observed: float,
    alternative: RandomizationAlternative,
    exact_max_nonzero_units: int,
    monte_carlo_samples: int,
    seed: int,
) -> _SignFlipOutcome:
    nonzero_count = int(nonzero_contributions.size)
    if nonzero_count <= exact_max_nonzero_units:
        assignments = 1 << nonzero_count
        extreme_assignments = 0
        if nonzero_count == 0:
            extreme_assignments = 1
        else:
            bit_positions = np.arange(nonzero_count, dtype=np.uint64)
            block_size = 8192
            for start in range(0, assignments, block_size):
                stop = min(start + block_size, assignments)
                codes = np.arange(start, stop, dtype=np.uint64)[:, None]
                signs = np.where((codes >> bit_positions) & 1, 1.0, -1.0)
                statistics = (signs @ nonzero_contributions) / denominator
                extreme_assignments += _extreme_count(
                    statistics, observed, alternative
                )
        return _SignFlipOutcome(
            method="exact",
            assignments_evaluated=assignments,
            extreme_assignments=extreme_assignments,
            p_value=float(extreme_assignments / assignments),
            p_value_resolution=float(1 / assignments),
            seed=None,
        )

    rng = np.random.default_rng(seed)
    extreme_assignments = 0
    remaining = monte_carlo_samples
    block_size = max(
        1,
        min(8192, _MAX_SIGN_MATRIX_ELEMENTS // nonzero_count),
    )
    while remaining:
        current = min(block_size, remaining)
        signs = rng.integers(
            0,
            2,
            size=(current, nonzero_count),
            dtype=np.int8,
        ).astype(np.float64)
        signs = signs * 2 - 1
        statistics = (signs @ nonzero_contributions) / denominator
        extreme_assignments += _extreme_count(statistics, observed, alternative)
        remaining -= current
    return _SignFlipOutcome(
        method="monte_carlo",
        assignments_evaluated=monte_carlo_samples,
        extreme_assignments=extreme_assignments,
        p_value=float((extreme_assignments + 1) / (monte_carlo_samples + 1)),
        p_value_resolution=float(1 / (monte_carlo_samples + 1)),
        seed=seed,
    )


def paired_randomization_test(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    alternative: RandomizationAlternative = "two-sided",
    exact_max_nonzero_pairs: int = 20,
    monte_carlo_samples: int = 100_000,
    seed: int = 0,
) -> PairedRandomizationResult:
    """Run a paired sign-flip test over case-level score differences.

    Under the sharp null and within-pair label exchangeability, swapping the
    baseline/candidate sign independently for every non-zero pair yields the
    null distribution of the mean difference. Up to
    ``exact_max_nonzero_pairs`` this function enumerates all assignments;
    otherwise it uses seeded Monte Carlo and the ``(extreme + 1)/(draws + 1)``
    correction. Exact zero differences are removed from sign enumeration but
    remain in the reported pair count and mean denominator.

    This is not a causal guarantee, effect-size threshold, cluster-aware test,
    multiple-comparison correction, or probability that the null is true.
    """

    baseline_array, candidate_array = _paired_score_arrays(baseline, candidate)
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if (
        isinstance(exact_max_nonzero_pairs, bool)
        or not isinstance(exact_max_nonzero_pairs, int)
        or not 0 <= exact_max_nonzero_pairs <= _MAX_EXACT_SIGN_FLIP_UNITS
    ):
        raise ValueError(
            "exact_max_nonzero_pairs must be an integer in "
            f"[0, {_MAX_EXACT_SIGN_FLIP_UNITS}]"
        )
    if (
        isinstance(monte_carlo_samples, bool)
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples <= 0
    ):
        raise ValueError("monte_carlo_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    differences = candidate_array - baseline_array
    nonzero = differences[differences != 0]
    pair_count = int(differences.size)
    nonzero_count = int(nonzero.size)
    observed = float(differences.mean())
    denominator = float(pair_count)

    outcome = _sign_flip_outcome(
        nonzero,
        denominator=denominator,
        observed=observed,
        alternative=alternative,
        exact_max_nonzero_units=exact_max_nonzero_pairs,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
    )

    return PairedRandomizationResult(
        pair_count=pair_count,
        nonzero_pair_count=nonzero_count,
        zero_difference_count=pair_count - nonzero_count,
        baseline_mean=float(baseline_array.mean()),
        candidate_mean=float(candidate_array.mean()),
        mean_difference=observed,
        alternative=alternative,
        method=outcome.method,
        assignments_evaluated=outcome.assignments_evaluated,
        extreme_assignments=outcome.extreme_assignments,
        p_value=outcome.p_value,
        p_value_resolution=outcome.p_value_resolution,
        seed=outcome.seed,
    )


@dataclass(frozen=True)
class ClusteredPairedRandomizationResult:
    """Joint cluster sign-flip result for one prespecified contrast/estimand."""

    case_count: int
    cluster_count: int
    nonzero_cluster_count: int
    zero_contribution_cluster_count: int
    cluster_sizes: tuple[int, ...]
    cluster_weighting: ClusterWeighting
    baseline_estimand: float
    candidate_estimand: float
    mean_difference: float
    alternative: RandomizationAlternative
    method: Literal["exact", "monte_carlo"]
    assignments_evaluated: int
    extreme_assignments: int
    p_value: float
    p_value_resolution: float
    seed: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "cluster_count": self.cluster_count,
            "nonzero_cluster_count": self.nonzero_cluster_count,
            "zero_contribution_cluster_count": self.zero_contribution_cluster_count,
            "cluster_sizes": list(self.cluster_sizes),
            "cluster_weighting": self.cluster_weighting,
            "baseline_estimand": self.baseline_estimand,
            "candidate_estimand": self.candidate_estimand,
            "mean_difference": self.mean_difference,
            "alternative": self.alternative,
            "method": self.method,
            "assignments_evaluated": self.assignments_evaluated,
            "extreme_assignments": self.extreme_assignments,
            "p_value": self.p_value,
            "p_value_resolution": self.p_value_resolution,
            "seed": self.seed,
        }


def clustered_paired_randomization_test(
    baseline: ArrayLike,
    candidate: ArrayLike,
    cluster_ids: Sequence[str],
    *,
    cluster_weighting: ClusterWeighting = "case",
    alternative: RandomizationAlternative = "two-sided",
    exact_max_nonzero_clusters: int = 20,
    monte_carlo_samples: int = 100_000,
    seed: int = 0,
) -> ClusteredPairedRandomizationResult:
    """Flip all case differences in the same cluster with one joint sign.

    ``cluster_weighting='case'`` targets the mean over cases: each cluster's
    contribution is its difference sum and large clusters receive more weight.
    ``cluster_weighting='equal'`` targets the mean of cluster means: every
    cluster receives equal weight. The estimand and cluster unit must be chosen
    before inspecting outcomes.

    This permits arbitrary dependence among cases inside a cluster, but valid
    inference still requires cluster-level label exchangeability and independent
    sign assignments across clusters. The function does not establish those
    assumptions, repair data-dependent cluster/family selection, or prove a
    causal, practically important, or general model improvement.
    """

    baseline_array, candidate_array = _paired_score_arrays(baseline, candidate)
    cluster_positions = _validated_cluster_positions(
        cluster_ids, int(baseline_array.size)
    )
    if cluster_weighting not in ("case", "equal"):
        raise ValueError("cluster_weighting must be 'case' or 'equal'")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if (
        isinstance(exact_max_nonzero_clusters, bool)
        or not isinstance(exact_max_nonzero_clusters, int)
        or not 0
        <= exact_max_nonzero_clusters
        <= _MAX_EXACT_SIGN_FLIP_UNITS
    ):
        raise ValueError(
            "exact_max_nonzero_clusters must be an integer in "
            f"[0, {_MAX_EXACT_SIGN_FLIP_UNITS}]"
        )
    if (
        isinstance(monte_carlo_samples, bool)
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples <= 0
    ):
        raise ValueError("monte_carlo_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    differences = candidate_array - baseline_array
    cluster_sizes = tuple(len(indices) for indices in cluster_positions.values())
    if cluster_weighting == "case":
        contributions = np.asarray(
            [float(differences[indices].sum()) for indices in cluster_positions.values()],
            dtype=np.float64,
        )
        denominator = float(differences.size)
        baseline_estimand = float(baseline_array.mean())
        candidate_estimand = float(candidate_array.mean())
    else:
        baseline_cluster_means = np.asarray(
            [float(baseline_array[indices].mean()) for indices in cluster_positions.values()],
            dtype=np.float64,
        )
        candidate_cluster_means = np.asarray(
            [float(candidate_array[indices].mean()) for indices in cluster_positions.values()],
            dtype=np.float64,
        )
        contributions = candidate_cluster_means - baseline_cluster_means
        denominator = float(len(cluster_positions))
        baseline_estimand = float(baseline_cluster_means.mean())
        candidate_estimand = float(candidate_cluster_means.mean())

    nonzero = contributions[contributions != 0]
    observed = float(contributions.sum() / denominator)
    outcome = _sign_flip_outcome(
        nonzero,
        denominator=denominator,
        observed=observed,
        alternative=alternative,
        exact_max_nonzero_units=exact_max_nonzero_clusters,
        monte_carlo_samples=monte_carlo_samples,
        seed=seed,
    )
    cluster_count = len(cluster_positions)
    nonzero_count = int(nonzero.size)
    return ClusteredPairedRandomizationResult(
        case_count=int(differences.size),
        cluster_count=cluster_count,
        nonzero_cluster_count=nonzero_count,
        zero_contribution_cluster_count=cluster_count - nonzero_count,
        cluster_sizes=cluster_sizes,
        cluster_weighting=cluster_weighting,
        baseline_estimand=baseline_estimand,
        candidate_estimand=candidate_estimand,
        mean_difference=observed,
        alternative=alternative,
        method=outcome.method,
        assignments_evaluated=outcome.assignments_evaluated,
        extreme_assignments=outcome.extreme_assignments,
        p_value=outcome.p_value,
        p_value_resolution=outcome.p_value_resolution,
        seed=outcome.seed,
    )


@dataclass(frozen=True)
class HolmHypothesisResult:
    """One hypothesis in rank order after Holm step-down adjustment."""

    original_index: int
    rank: int
    p_value: float
    multiplier: int
    scaled_p_value: float
    adjusted_p_value: float
    rejected: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "original_index": self.original_index,
            "rank": self.rank,
            "p_value": self.p_value,
            "multiplier": self.multiplier,
            "scaled_p_value": self.scaled_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "rejected": self.rejected,
        }


@dataclass(frozen=True)
class HolmCorrectionResult:
    """Holm family-wise-error correction with both rank and input-order views."""

    alpha: float
    family_size: int
    ordered_hypotheses: tuple[HolmHypothesisResult, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "family_size": self.family_size,
            "ordered_hypotheses": [
                hypothesis.to_dict() for hypothesis in self.ordered_hypotheses
            ],
            "adjusted_p_values": list(self.adjusted_p_values),
            "rejected": list(self.rejected),
        }


def holm_bonferroni_correction(
    p_values: ArrayLike,
    *,
    alpha: float = 0.05,
) -> HolmCorrectionResult:
    """Adjust one prespecified family of valid p-values with Holm's procedure.

    P-values are sorted ascending with original input order as the deterministic
    tie-break. At sorted rank ``i`` (one based), the scaled value is
    ``(m - i + 1) * p_(i)``. Holm adjusted p-values are the running maximum of
    those scaled values, capped at one, then mapped back to input order.

    Under valid input p-values, Holm controls family-wise error under arbitrary
    dependence. It does not make post-hoc family selection valid, repair invalid
    component tests, handle repeated peeking, estimate effect size, or establish
    that any rejection is practically important or causal.
    """

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a finite number in (0, 1)")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0 < alpha_value < 1:
        raise ValueError("alpha must be a finite number in (0, 1)")

    raw = np.asarray(p_values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional array")
    if raw.dtype.kind not in "iuf":
        raise ValueError("p_values must contain real numbers, not booleans")
    values = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("p_values must be finite")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("p_values must be in [0, 1]")

    family_size = int(values.size)
    order = np.argsort(values, kind="stable")
    adjusted_in_input_order = np.empty(family_size, dtype=np.float64)
    rejected_in_input_order = np.empty(family_size, dtype=np.bool_)
    ordered_results: list[HolmHypothesisResult] = []
    running_adjusted = 0.0

    for zero_based_rank, original_index_value in enumerate(order):
        original_index = int(original_index_value)
        rank = zero_based_rank + 1
        multiplier = family_size - zero_based_rank
        p_value = float(values[original_index])
        scaled = multiplier * p_value
        running_adjusted = min(1.0, max(running_adjusted, scaled))
        rejected = running_adjusted <= alpha_value
        adjusted_in_input_order[original_index] = running_adjusted
        rejected_in_input_order[original_index] = rejected
        ordered_results.append(
            HolmHypothesisResult(
                original_index=original_index,
                rank=rank,
                p_value=p_value,
                multiplier=multiplier,
                scaled_p_value=float(scaled),
                adjusted_p_value=float(running_adjusted),
                rejected=bool(rejected),
            )
        )

    return HolmCorrectionResult(
        alpha=alpha_value,
        family_size=family_size,
        ordered_hypotheses=tuple(ordered_results),
        adjusted_p_values=tuple(float(value) for value in adjusted_in_input_order),
        rejected=tuple(bool(value) for value in rejected_in_input_order),
    )


@dataclass(frozen=True)
class ReleaseGate:
    """A transparent gate over quality, safety, and latency deltas."""

    minimum_quality_difference: float = 0.0
    maximum_safety_regression: float = 0.0
    maximum_latency_increase_fraction: float = 0.10

    def __post_init__(self) -> None:
        values = {
            "minimum_quality_difference": self.minimum_quality_difference,
            "maximum_safety_regression": self.maximum_safety_regression,
            "maximum_latency_increase_fraction": self.maximum_latency_increase_fraction,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.maximum_safety_regression < 0:
            raise ValueError("maximum_safety_regression must be non-negative")
        if self.maximum_latency_increase_fraction <= -1:
            raise ValueError(
                "maximum_latency_increase_fraction must be greater than -1"
            )

    def evaluate(
        self,
        *,
        quality: PairedBootstrapResult | ClusteredPairedBootstrapResult,
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
