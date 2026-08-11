"""Binary calibration and selective-prediction metrics with explicit semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CalibrationBin:
    """One non-empty equal-width probability bin."""

    lower: float
    upper: float
    count: int
    mean_probability: float
    positive_rate: float
    absolute_gap: float


@dataclass(frozen=True)
class BinaryCalibrationResult:
    """Brier score and equal-width expected calibration error."""

    count: int
    brier_score: float
    expected_calibration_error: float
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class SelectiveRiskPoint:
    """Risk when accepting every example at or above one confidence threshold."""

    threshold: float
    accepted_count: int
    coverage: float
    risk: float


def _one_dimensional_finite(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_labels(labels: ArrayLike) -> NDArray[np.float64]:
    array = _one_dimensional_finite("labels", labels)
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("labels must contain only 0 and 1")
    return array


def _probabilities(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = _one_dimensional_finite(name, values)
    if not np.all((array >= 0) & (array <= 1)):
        raise ValueError(f"{name} must be in [0, 1]")
    return array


def _validate_pair(
    labels: ArrayLike, probabilities: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    label_array = _binary_labels(labels)
    probability_array = _probabilities("probabilities", probabilities)
    if label_array.shape != probability_array.shape:
        raise ValueError("labels and probabilities must have the same length")
    return label_array, probability_array


def binary_calibration(
    labels: ArrayLike, probabilities: ArrayLike, *, bins: int = 10
) -> BinaryCalibrationResult:
    """Compute binary Brier score and equal-width ECE.

    Bins are ``[lower, upper)`` except that probability 1 belongs to the final
    bin. Empty bins are omitted from the returned details and contribute zero to
    ECE. ECE is descriptive and depends on the requested binning scheme.
    """

    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an integer")
    if bins <= 0:
        raise ValueError("bins must be positive")
    label_array, probability_array = _validate_pair(labels, probabilities)
    bin_ids = np.minimum((probability_array * bins).astype(np.int64), bins - 1)
    details: list[CalibrationBin] = []
    expected_calibration_error = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        count = int(selected.sum())
        if count == 0:
            continue
        mean_probability = float(probability_array[selected].mean())
        positive_rate = float(label_array[selected].mean())
        absolute_gap = abs(mean_probability - positive_rate)
        expected_calibration_error += count / label_array.size * absolute_gap
        details.append(
            CalibrationBin(
                lower=bin_id / bins,
                upper=(bin_id + 1) / bins,
                count=count,
                mean_probability=mean_probability,
                positive_rate=positive_rate,
                absolute_gap=absolute_gap,
            )
        )
    return BinaryCalibrationResult(
        count=int(label_array.size),
        brier_score=float(np.mean((probability_array - label_array) ** 2)),
        expected_calibration_error=float(expected_calibration_error),
        bins=tuple(details),
    )


def risk_coverage_curve(
    correctness: ArrayLike, confidence: ArrayLike
) -> tuple[SelectiveRiskPoint, ...]:
    """Return a tie-aware selective risk curve over unique confidence values.

    Higher confidence is assumed to indicate greater expected correctness. For
    each unique threshold, all examples with ``confidence >= threshold`` are
    accepted together, so equal-confidence examples are never split by order.
    Risk is the error rate among accepted examples.
    """

    correct_array, confidence_array = _validate_pair(correctness, confidence)
    thresholds = np.unique(confidence_array)[::-1]
    points: list[SelectiveRiskPoint] = []
    for threshold in thresholds:
        selected = confidence_array >= threshold
        accepted_count = int(selected.sum())
        points.append(
            SelectiveRiskPoint(
                threshold=float(threshold),
                accepted_count=accepted_count,
                coverage=accepted_count / correct_array.size,
                risk=float(1 - correct_array[selected].mean()),
            )
        )
    return tuple(points)


__all__ = [
    "BinaryCalibrationResult",
    "CalibrationBin",
    "SelectiveRiskPoint",
    "binary_calibration",
    "risk_coverage_curve",
]
