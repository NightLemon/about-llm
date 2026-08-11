"""Explicit accuracy-matrix metrics for sequential continual learning."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContinualLearningReport:
    """Metrics for a square matrix evaluated after every sequential task stage."""

    accuracy_matrix: tuple[tuple[float, ...], ...]
    pretraining_baseline: tuple[float, ...]
    diagonal_accuracy: tuple[float, ...]
    final_average_accuracy: float
    backward_transfer: float
    per_task_forgetting: tuple[float, ...]
    average_forgetting_old_tasks: float
    forward_transfer: float

    def to_dict(self) -> dict[str, object]:
        return {
            "definitions": {
                "accuracy_matrix": (
                    "R[i][j] is task-j accuracy after sequential training through task i"
                ),
                "final_average_accuracy": "mean_j R[T-1][j]",
                "backward_transfer": (
                    "mean_{j<T-1}(R[T-1][j] - R[j][j])"
                ),
                "forgetting": (
                    "max_{i in [j,T-1]} R[i][j] - R[T-1][j]"
                ),
                "forward_transfer": (
                    "mean_{j>0}(R[j-1][j] - pretraining_baseline[j])"
                ),
            },
            "accuracy_matrix": [list(row) for row in self.accuracy_matrix],
            "pretraining_baseline": list(self.pretraining_baseline),
            "diagonal_accuracy": list(self.diagonal_accuracy),
            "final_average_accuracy": self.final_average_accuracy,
            "backward_transfer": self.backward_transfer,
            "per_task_forgetting": list(self.per_task_forgetting),
            "average_forgetting_old_tasks": self.average_forgetting_old_tasks,
            "forward_transfer": self.forward_transfer,
            "scope": {
                "metric_direction": "higher_accuracy_is_better",
                "future_tasks_evaluated_before_training": True,
                "last_task_forgetting_defined_as_zero": True,
                "confidence_intervals_computed": False,
            },
        }


def summarize_accuracy_matrix(
    accuracy_matrix: Sequence[Sequence[float]],
    *,
    pretraining_baseline: Sequence[float],
) -> ContinualLearningReport:
    """Compute ACC/BWT/forgetting/FWT under the definitions in ``to_dict``."""

    matrix = tuple(
        tuple(_accuracy(value, "accuracy_matrix") for value in row)
        for row in accuracy_matrix
    )
    task_count = len(matrix)
    if task_count < 2:
        raise ValueError("accuracy_matrix must contain at least two task stages")
    if any(len(row) != task_count for row in matrix):
        raise ValueError("accuracy_matrix must be square with one row per task stage")
    baseline = tuple(
        _accuracy(value, "pretraining_baseline") for value in pretraining_baseline
    )
    if len(baseline) != task_count:
        raise ValueError("pretraining_baseline must have one value per task")

    final_row = matrix[-1]
    final_average = sum(final_row) / task_count
    diagonal = tuple(matrix[index][index] for index in range(task_count))
    backward_transfer = sum(
        final_row[task] - diagonal[task] for task in range(task_count - 1)
    ) / (task_count - 1)
    forgetting = tuple(
        max(matrix[stage][task] for stage in range(task, task_count))
        - final_row[task]
        for task in range(task_count)
    )
    average_forgetting = sum(forgetting[:-1]) / (task_count - 1)
    forward_transfer = sum(
        matrix[task - 1][task] - baseline[task]
        for task in range(1, task_count)
    ) / (task_count - 1)
    return ContinualLearningReport(
        accuracy_matrix=matrix,
        pretraining_baseline=baseline,
        diagonal_accuracy=diagonal,
        final_average_accuracy=final_average,
        backward_transfer=backward_transfer,
        per_task_forgetting=forgetting,
        average_forgetting_old_tasks=average_forgetting,
        forward_transfer=forward_transfer,
    )


def reservoir_sample_indices(
    stream_length: int,
    capacity: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    """Return a uniform reservoir sample from a single pass over stream indices.

    The returned indices are sorted only to make downstream replay ordering stable;
    sorting does not change which examples the reservoir selected.
    """

    for value, label in ((stream_length, "stream_length"), (capacity, "capacity")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if capacity > stream_length:
        raise ValueError("capacity cannot exceed stream_length")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    generator = random.Random(seed)
    reservoir = list(range(capacity))
    for stream_index in range(capacity, stream_length):
        replacement = generator.randrange(stream_index + 1)
        if replacement < capacity:
            reservoir[replacement] = stream_index
    return tuple(sorted(reservoir))


def _accuracy(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label} values must be finite accuracies in [0, 1]")
    return float(value)
