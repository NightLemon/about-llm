"""Finite categorical controls for score-function policy gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CategoricalPolicyGradientReport:
    """Exact objective and enumerated REINFORCE estimator distribution."""

    probabilities: NDArray[np.float64]
    rewards: NDArray[np.float64]
    expected_reward: float
    baseline: float
    exact_gradient: NDArray[np.float64]
    expected_score_gradient: NDArray[np.float64]
    per_action_score_gradients: NDArray[np.float64]
    per_action_estimators: NDArray[np.float64]
    estimator_total_variance: float


@dataclass(frozen=True)
class GroupRelativeAdvantageReport:
    """One-group reward normalization with an explicit tied-reward branch."""

    rewards: NDArray[np.float64]
    reward_mean: float
    reward_standard_deviation: float
    advantages: NDArray[np.float64]
    degenerate: bool


def _finite_vector(values: ArrayLike, label: str, *, minimum_size: int = 1) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain numeric values, not booleans")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum_size:
        raise ValueError(
            f"{label} must be a one-dimensional array with at least {minimum_size} values"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _finite_scalar(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _categorical_probabilities(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits)
    unnormalized = np.exp(shifted)
    return np.asarray(unnormalized / np.sum(unnormalized), dtype=np.float64)


def categorical_policy_gradient(
    logits: ArrayLike,
    rewards: ArrayLike,
    *,
    baseline: float = 0.0,
) -> CategoricalPolicyGradientReport:
    """Enumerate a finite categorical policy-gradient estimator exactly.

    For action ``a``, the score-function estimator is
    ``(reward[a] - baseline) * grad(log probability[a])``. ``baseline`` is a
    scalar independent of the sampled action, so it can change variance but not
    the estimator expectation. This function enumerates all actions; it does not
    use Monte Carlo sampling or execute an environment or language model.
    """

    logit_array = _finite_vector(logits, "logits", minimum_size=2)
    reward_array = _finite_vector(rewards, "rewards", minimum_size=2)
    if reward_array.shape != logit_array.shape:
        raise ValueError("logits and rewards must have equal shape")
    baseline_value = _finite_scalar(baseline, "baseline")

    probabilities = _categorical_probabilities(logit_array)
    expected_reward = float(probabilities @ reward_array)
    exact_gradient = probabilities * (reward_array - expected_reward)
    score_gradients = np.eye(logit_array.size, dtype=np.float64) - probabilities
    estimators = (reward_array - baseline_value)[:, None] * score_gradients
    expected_estimator = probabilities @ estimators
    centered = estimators - exact_gradient
    total_variance = float(probabilities @ np.sum(centered * centered, axis=1))

    return CategoricalPolicyGradientReport(
        probabilities=probabilities,
        rewards=reward_array,
        expected_reward=expected_reward,
        baseline=baseline_value,
        exact_gradient=exact_gradient,
        expected_score_gradient=np.asarray(expected_estimator, dtype=np.float64),
        per_action_score_gradients=score_gradients,
        per_action_estimators=estimators,
        estimator_total_variance=total_variance,
    )


def variance_minimizing_score_baseline(logits: ArrayLike, rewards: ArrayLike) -> float:
    """Return the scalar baseline minimizing total gradient-estimator variance.

    The criterion is the expected squared Euclidean error of the full logit
    gradient estimator. It need not equal the expected reward because different
    actions can have different score-gradient norms.
    """

    logit_array = _finite_vector(logits, "logits", minimum_size=2)
    reward_array = _finite_vector(rewards, "rewards", minimum_size=2)
    if reward_array.shape != logit_array.shape:
        raise ValueError("logits and rewards must have equal shape")
    probabilities = _categorical_probabilities(logit_array)
    score_gradients = np.eye(logit_array.size, dtype=np.float64) - probabilities
    squared_norms = np.sum(score_gradients * score_gradients, axis=1)
    denominator = float(probabilities @ squared_norms)
    if denominator <= 0:
        raise ValueError("policy must have non-zero score-gradient variance")
    return float(probabilities @ (reward_array * squared_norms) / denominator)


def group_relative_advantages(
    rewards: ArrayLike,
    *,
    epsilon: float = 1e-8,
) -> GroupRelativeAdvantageReport:
    """Standardize rewards within one authored group.

    This is a small GRPO-style normalization reference, not a complete GRPO
    objective. Real implementations differ in group construction, standard
    deviation convention, token reduction, clipping, KL treatment, and masks.
    """

    reward_array = _finite_vector(rewards, "rewards")
    epsilon_value = _finite_scalar(epsilon, "epsilon")
    if epsilon_value <= 0:
        raise ValueError("epsilon must be positive")
    mean = float(np.mean(reward_array))
    standard_deviation = float(np.std(reward_array, ddof=0))
    degenerate = standard_deviation <= epsilon_value
    advantages = (
        np.zeros_like(reward_array)
        if degenerate
        else (reward_array - mean) / standard_deviation
    )
    return GroupRelativeAdvantageReport(
        rewards=reward_array,
        reward_mean=mean,
        reward_standard_deviation=standard_deviation,
        advantages=np.asarray(advantages, dtype=np.float64),
        degenerate=degenerate,
    )
