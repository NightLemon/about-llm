"""From-scratch linear Bradley-Terry reward-model optimization reference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PairwiseRewardMetrics:
    pair_count: int
    mean_loss: float
    strict_pair_accuracy: float
    tie_count: int
    mean_margin: float
    minimum_margin: float
    mean_preference_probability: float


@dataclass(frozen=True)
class LinearRewardTrainingReport:
    pair_count: int
    feature_count: int
    steps: int
    learning_rate: float
    l2_penalty: float
    initial_objective: float
    final_objective: float
    initial_metrics: PairwiseRewardMetrics
    final_metrics: PairwiseRewardMetrics
    weights: tuple[float, ...]


def _feature_matrix(values: ArrayLike, label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain numeric features, not booleans")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0 or 0 in matrix.shape:
        raise ValueError(f"{label} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain only finite values")
    return matrix


def _weight_vector(
    values: ArrayLike,
    *,
    feature_count: int,
) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError("weights must contain numeric values, not booleans")
    weights = np.asarray(raw, dtype=np.float64)
    if weights.shape != (feature_count,) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be a finite vector matching feature count")
    return weights


def _paired_features(
    chosen_features: ArrayLike,
    rejected_features: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    chosen = _feature_matrix(chosen_features, "chosen_features")
    rejected = _feature_matrix(rejected_features, "rejected_features")
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected feature matrices must have equal shape")
    return chosen, rejected


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    probabilities[~positive] = exponent / (1 + exponent)
    return probabilities


def pairwise_reward_metrics(
    chosen_features: ArrayLike,
    rejected_features: ArrayLike,
    weights: ArrayLike,
) -> PairwiseRewardMetrics:
    chosen, rejected = _paired_features(chosen_features, rejected_features)
    vector = _weight_vector(weights, feature_count=chosen.shape[1])
    margins = (chosen - rejected) @ vector
    losses = np.logaddexp(0, -margins)
    probabilities = _sigmoid(margins)
    return PairwiseRewardMetrics(
        pair_count=int(chosen.shape[0]),
        mean_loss=float(np.mean(losses)),
        strict_pair_accuracy=float(np.mean(margins > 0)),
        tie_count=int(np.sum(margins == 0)),
        mean_margin=float(np.mean(margins)),
        minimum_margin=float(np.min(margins)),
        mean_preference_probability=float(np.mean(probabilities)),
    )


def fit_linear_pairwise_reward_model(
    chosen_features: ArrayLike,
    rejected_features: ArrayLike,
    *,
    steps: int = 300,
    learning_rate: float = 0.1,
    l2_penalty: float = 0.0,
    initial_weights: ArrayLike | None = None,
) -> LinearRewardTrainingReport:
    """Fit a full-batch linear scorer with Bradley-Terry negative log-likelihood."""

    chosen, rejected = _paired_features(chosen_features, rejected_features)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if (
        isinstance(l2_penalty, bool)
        or not isinstance(l2_penalty, Real)
        or not math.isfinite(float(l2_penalty))
        or l2_penalty < 0
    ):
        raise ValueError("l2_penalty must be finite and non-negative")

    learning_rate = float(learning_rate)
    l2_penalty = float(l2_penalty)
    deltas = chosen - rejected
    weights = (
        np.zeros(chosen.shape[1], dtype=np.float64)
        if initial_weights is None
        else _weight_vector(initial_weights, feature_count=chosen.shape[1]).copy()
    )

    def objective(vector: NDArray[np.float64]) -> float:
        margins = deltas @ vector
        return float(
            np.mean(np.logaddexp(0, -margins))
            + 0.5 * l2_penalty * np.dot(vector, vector)
        )

    initial_metrics = pairwise_reward_metrics(chosen, rejected, weights)
    initial_objective = objective(weights)
    for _ in range(steps):
        margins = deltas @ weights
        probability_chosen_loses = _sigmoid(-margins)
        gradient = -np.mean(
            probability_chosen_loses[:, None] * deltas,
            axis=0,
        ) + l2_penalty * weights
        weights -= learning_rate * gradient

    final_metrics = pairwise_reward_metrics(chosen, rejected, weights)
    final_objective = objective(weights)
    return LinearRewardTrainingReport(
        pair_count=int(chosen.shape[0]),
        feature_count=int(chosen.shape[1]),
        steps=steps,
        learning_rate=float(learning_rate),
        l2_penalty=float(l2_penalty),
        initial_objective=initial_objective,
        final_objective=final_objective,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        weights=tuple(float(value) for value in weights),
    )
