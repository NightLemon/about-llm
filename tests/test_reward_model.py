from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from about_llm.finetuning import (
    fit_linear_pairwise_reward_model,
    pairwise_reward_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def test_zero_weight_reward_model_starts_at_log_two_with_explicit_ties() -> None:
    chosen = np.array([[1, 0], [0, 1]], dtype=np.float64)
    rejected = np.zeros_like(chosen)
    metrics = pairwise_reward_metrics(chosen, rejected, [0, 0])

    assert metrics.pair_count == 2
    assert metrics.mean_loss == pytest.approx(math.log(2))
    assert metrics.strict_pair_accuracy == 0
    assert metrics.tie_count == 2
    assert metrics.mean_margin == 0
    assert metrics.mean_preference_probability == 0.5


def test_full_batch_optimizer_reduces_bradley_terry_objective() -> None:
    chosen = np.array([[1, 0], [2, 0], [1, 1], [2, -1]], dtype=np.float64)
    rejected = np.zeros_like(chosen)
    report = fit_linear_pairwise_reward_model(
        chosen,
        rejected,
        steps=300,
        learning_rate=0.1,
        l2_penalty=0.01,
    )

    assert report.initial_objective == pytest.approx(math.log(2))
    assert report.final_objective < report.initial_objective
    assert report.final_metrics.mean_loss < report.initial_metrics.mean_loss
    assert report.final_metrics.strict_pair_accuracy == 1
    assert report.final_metrics.minimum_margin > 0


def test_counterfactual_pairs_remove_the_authored_length_shortcut() -> None:
    rejected = np.zeros((4, 2), dtype=np.float64)
    confounded = np.array([[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.float64)
    counterfactual = np.array(
        [[1, -1], [2, -2], [1, -1], [2, -2]], dtype=np.float64
    )
    held_out = np.array([[1, -2], [2, -3]], dtype=np.float64)
    held_out_rejected = np.zeros_like(held_out)

    confounded_report = fit_linear_pairwise_reward_model(confounded, rejected)
    balanced = np.concatenate((confounded, counterfactual), axis=0)
    balanced_report = fit_linear_pairwise_reward_model(
        balanced, np.zeros_like(balanced)
    )
    confounded_held_out = pairwise_reward_metrics(
        held_out, held_out_rejected, confounded_report.weights
    )
    balanced_held_out = pairwise_reward_metrics(
        held_out, held_out_rejected, balanced_report.weights
    )

    assert confounded_report.final_metrics.strict_pair_accuracy == 1
    assert confounded_report.weights[0] == pytest.approx(confounded_report.weights[1])
    assert confounded_held_out.strict_pair_accuracy == 0
    assert balanced_report.final_metrics.strict_pair_accuracy == 1
    assert balanced_report.weights[1] == pytest.approx(0, abs=1e-12)
    assert balanced_held_out.strict_pair_accuracy == 1


def test_pairwise_metrics_are_stable_for_extreme_margins() -> None:
    metrics = pairwise_reward_metrics([[1.0], [-1.0]], [[0.0], [0.0]], [1000])
    assert math.isfinite(metrics.mean_loss)
    assert metrics.mean_loss == pytest.approx(500)
    assert metrics.mean_preference_probability == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("chosen", "rejected", "weights"),
    [
        ([[1.0]], [[1.0, 2.0]], [1.0]),
        ([[float("nan")]], [[0.0]], [1.0]),
        ([[1.0]], [[0.0]], [float("inf")]),
        ([[True]], [[False]], [1.0]),
        ([[1.0]], [[0.0]], [True]),
        ([], [], [1.0]),
    ],
)
def test_reward_metrics_reject_invalid_shapes_and_values(
    chosen: object, rejected: object, weights: object
) -> None:
    with pytest.raises(ValueError):
        pairwise_reward_metrics(chosen, rejected, weights)


@pytest.mark.parametrize(
    ("steps", "learning_rate", "l2_penalty"),
    [(0, 0.1, 0), (1, 0, 0), (1, float("nan"), 0), (1, 0.1, -1)],
)
def test_reward_training_rejects_invalid_optimizer_contract(
    steps: int, learning_rate: float, l2_penalty: float
) -> None:
    with pytest.raises(ValueError):
        fit_linear_pairwise_reward_model(
            [[1.0]],
            [[0.0]],
            steps=steps,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
        )


