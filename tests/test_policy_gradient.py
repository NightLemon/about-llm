from __future__ import annotations

import numpy as np
import pytest

from about_llm.finetuning import (
    categorical_policy_gradient,
    group_relative_advantages,
    variance_minimizing_score_baseline,
)


def test_categorical_policy_gradient_matches_finite_difference() -> None:
    logits = np.array([-0.4, 0.1, 0.3], dtype=np.float64)
    rewards = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    report = categorical_policy_gradient(logits, rewards, baseline=1.7)
    epsilon = 1e-6
    finite_difference = np.empty_like(logits)

    for index in range(logits.size):
        positive = logits.copy()
        negative = logits.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        positive_return = categorical_policy_gradient(positive, rewards).expected_reward
        negative_return = categorical_policy_gradient(negative, rewards).expected_reward
        finite_difference[index] = (positive_return - negative_return) / (2 * epsilon)

    np.testing.assert_allclose(report.exact_gradient, finite_difference, atol=1e-9)
    np.testing.assert_allclose(
        report.expected_score_gradient, report.exact_gradient, atol=1e-15
    )
    assert np.sum(report.exact_gradient) == pytest.approx(0)


def test_action_independent_baseline_preserves_mean_and_optimal_reduces_variance() -> None:
    logits = [-0.4, 0.1, 0.3]
    rewards = [0.0, 1.0, 4.0]
    zero = categorical_policy_gradient(logits, rewards, baseline=0)
    value = categorical_policy_gradient(logits, rewards, baseline=zero.expected_reward)
    optimal_baseline = variance_minimizing_score_baseline(logits, rewards)
    optimal = categorical_policy_gradient(logits, rewards, baseline=optimal_baseline)

    np.testing.assert_allclose(zero.expected_score_gradient, value.expected_score_gradient)
    np.testing.assert_allclose(zero.expected_score_gradient, optimal.expected_score_gradient)
    assert optimal.estimator_total_variance <= value.estimator_total_variance
    assert optimal.estimator_total_variance < zero.estimator_total_variance
    assert optimal_baseline != pytest.approx(zero.expected_reward)


def test_group_relative_advantages_standardize_and_expose_tied_rewards() -> None:
    varied = group_relative_advantages([0.0, 1.0, 4.0, 4.0])
    tied = group_relative_advantages([2.0, 2.0, 2.0, 2.0])

    assert np.mean(varied.advantages) == pytest.approx(0, abs=1e-15)
    assert np.std(varied.advantages) == pytest.approx(1)
    assert not varied.degenerate
    np.testing.assert_array_equal(tied.advantages, np.zeros(4))
    assert tied.degenerate


def test_policy_gradient_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        categorical_policy_gradient([0.0], [1.0])
    with pytest.raises(ValueError, match="equal shape"):
        categorical_policy_gradient([0.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not booleans"):
        group_relative_advantages([True, False])
    with pytest.raises(ValueError, match="positive"):
        group_relative_advantages([0.0, 1.0], epsilon=0)
