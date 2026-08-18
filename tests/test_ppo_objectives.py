from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from about_llm.finetuning import (
    generalized_advantage_estimation,
    masked_mean,
    ppo_clipped_surrogate,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.formula


def test_gae_matches_two_step_analytic_result_and_ignores_padding() -> None:
    result = generalized_advantage_estimation(
        rewards=[0.0, 1.0, 999.0],
        values=[0.5, 0.25, 999.0],
        next_values=[0.25, 10.0, 999.0],
        valid_mask=[True, True, False],
        terminated=[False, True, False],
        truncated=[False, False, False],
        gamma=0.9,
        gae_lambda=0.8,
        bootstrap_truncated=True,
    )

    assert result.td_residuals == pytest.approx([-0.275, 0.75, 0.0])
    assert result.advantages == pytest.approx([0.265, 0.75, 0.0])
    assert result.returns == pytest.approx([0.765, 1.0, 0.0])
    assert result.bootstrap_mask.tolist() == [True, False, False]
    assert result.continuation_mask.tolist() == [True, False, False]


def test_truncation_bootstrap_is_explicit_but_never_crosses_episode_boundary() -> None:
    common = {
        "rewards": [1.0, 100.0],
        "values": [0.5, 0.0],
        "next_values": [2.0, 0.0],
        "valid_mask": [True, True],
        "terminated": [False, True],
        "truncated": [True, False],
        "gamma": 0.9,
        "gae_lambda": 0.8,
    }
    bootstrapped = generalized_advantage_estimation(
        **common, bootstrap_truncated=True
    )
    unbootstrapped = generalized_advantage_estimation(
        **common, bootstrap_truncated=False
    )

    assert bootstrapped.advantages[0] == pytest.approx(2.3)
    assert unbootstrapped.advantages[0] == pytest.approx(0.5)
    assert bootstrapped.continuation_mask.tolist() == [False, False]
    assert bootstrapped.advantages[0] != pytest.approx(2.3 + 0.9 * 0.8 * 100)


def test_padding_gap_resets_gae_recursion() -> None:
    result = generalized_advantage_estimation(
        rewards=[[1.0, 999.0, 2.0]],
        values=[[0.0, 999.0, 0.0]],
        next_values=[[0.0, 999.0, 0.0]],
        valid_mask=[[True, False, True]],
        terminated=[[False, False, True]],
        truncated=[[False, False, False]],
        gamma=1.0,
        gae_lambda=1.0,
        bootstrap_truncated=True,
    )
    np.testing.assert_allclose(result.advantages, [[1.0, 0.0, 2.0]])


def test_ppo_clips_positive_upper_and_negative_lower_ratios() -> None:
    ratios = np.array([1.5, 0.5, 1.0])
    report = ppo_clipped_surrogate(
        np.log(ratios),
        np.zeros(3),
        [1.0, -1.0, 1.0],
        valid_mask=[True, True, True],
        clip_epsilon=0.2,
    )

    assert report.probability_ratios == pytest.approx(ratios)
    assert report.clipped_probability_ratios == pytest.approx([1.2, 0.8, 1.0])
    assert report.per_action_surrogate == pytest.approx([1.2, -0.8, 1.0])
    assert report.clipped_actions.tolist() == [True, True, False]
    assert report.sampled_action_count == 3
    assert report.surrogate_objective == pytest.approx((1.2 - 0.8 + 1.0) / 3)
    assert report.policy_loss == pytest.approx(-report.surrogate_objective)
    assert report.mean_unclipped_objective == pytest.approx((1.5 - 0.5 + 1) / 3)
    assert report.clip_fraction == pytest.approx(2 / 3)


def test_ppo_mask_excludes_padding_from_all_aggregates() -> None:
    report = ppo_clipped_surrogate(
        [math.log(1.1), 700.0],
        [0.0, -700.0],
        [2.0, 999.0],
        valid_mask=[True, False],
        clip_epsilon=0.2,
    )
    assert report.sampled_action_count == 1
    assert report.mean_probability_ratio == pytest.approx(1.1)
    assert report.surrogate_objective == pytest.approx(2.2)
    assert report.clip_fraction == 0
    assert report.probability_ratios[1] == 1
    assert report.per_action_surrogate[1] == 0


def test_same_sampled_ratio_does_not_bound_full_distribution_kl() -> None:
    old = np.array([0.1, 0.45, 0.45])
    new = np.array([0.1, 0.9 - 1e-12, 1e-12])
    report = ppo_clipped_surrogate(
        [math.log(new[0])],
        [math.log(old[0])],
        [1.0],
        valid_mask=[True],
        clip_epsilon=0.2,
    )
    full_forward_kl = float(np.sum(old * np.log(old / new)))

    assert report.mean_probability_ratio == pytest.approx(1)
    assert report.clip_fraction == 0
    assert report.approximate_sampled_kl == pytest.approx(0)
    assert full_forward_kl > 10


def test_masked_mean_and_objectives_reject_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        masked_mean([1.0], [False])
    with pytest.raises(ValueError, match="both terminated and truncated"):
        generalized_advantage_estimation(
            [1.0],
            [0.0],
            [0.0],
            valid_mask=[True],
            terminated=[True],
            truncated=[True],
            bootstrap_truncated=True,
        )
    with pytest.raises(ValueError, match="padding transitions"):
        generalized_advantage_estimation(
            [1.0],
            [0.0],
            [0.0],
            valid_mask=[False],
            terminated=[True],
            truncated=[False],
            bootstrap_truncated=True,
        )
    with pytest.raises(ValueError, match="non-finite ratios"):
        ppo_clipped_surrogate(
            [1000.0], [0.0], [1.0], valid_mask=[True], clip_epsilon=0.2
        )

