"""Masked NumPy references for GAE and the PPO clipped policy surrogate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GeneralizedAdvantageEstimate:
    """Per-action GAE values under an explicit transition-boundary contract."""

    advantages: NDArray[np.float64]
    returns: NDArray[np.float64]
    td_residuals: NDArray[np.float64]
    bootstrap_mask: NDArray[np.bool_]
    continuation_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class PPOClippedSurrogateReport:
    """Per-action PPO terms plus mask-aware aggregate diagnostics."""

    sampled_action_count: int
    clip_epsilon: float
    surrogate_objective: float
    policy_loss: float
    mean_unclipped_objective: float
    mean_clipped_objective: float
    clip_fraction: float
    mean_probability_ratio: float
    mean_log_probability_ratio: float
    approximate_sampled_kl: float
    probability_ratios: NDArray[np.float64]
    clipped_probability_ratios: NDArray[np.float64]
    per_action_surrogate: NDArray[np.float64]
    clipped_actions: NDArray[np.bool_]


def _numeric_array(values: ArrayLike, label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain numeric values, not booleans")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim not in (1, 2) or array.size == 0 or 0 in array.shape:
        raise ValueError(f"{label} must be a non-empty one- or two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _boolean_array(
    values: ArrayLike,
    label: str,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    raw = np.asarray(values)
    if raw.dtype.kind != "b" or raw.shape != shape:
        raise ValueError(f"{label} must be a boolean array matching value shape")
    return np.asarray(raw, dtype=np.bool_)


def _unit_interval(value: object, label: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real scalar")
    result = float(value)
    lower_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not lower_ok or result > 1:
        qualifier = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{label} must be finite and in {qualifier}")
    return result


def masked_mean(values: ArrayLike, valid_mask: ArrayLike) -> float:
    """Return the arithmetic mean over explicitly valid actions only."""

    array = _numeric_array(values, "values")
    mask = _boolean_array(valid_mask, "valid_mask", shape=array.shape)
    if not np.any(mask):
        raise ValueError("valid_mask must select at least one action")
    return float(np.mean(array[mask]))


def generalized_advantage_estimation(
    rewards: ArrayLike,
    values: ArrayLike,
    next_values: ArrayLike,
    *,
    valid_mask: ArrayLike,
    terminated: ArrayLike,
    truncated: ArrayLike,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_truncated: bool,
) -> GeneralizedAdvantageEstimate:
    """Compute GAE along the final axis without crossing episode or padding boundaries.

    ``terminated[t]`` means the environment reached an absorbing terminal state, so
    ``next_values[t]`` is never bootstrapped. ``truncated[t]`` means collection ended
    for an external reason. Its value bootstrap is controlled explicitly by
    ``bootstrap_truncated``; either way, GAE recursion never crosses that boundary.
    Callers must supply ``next_values`` for each transition rather than relying on the
    next padded/concatenated array position.
    """

    reward_array = _numeric_array(rewards, "rewards")
    value_array = _numeric_array(values, "values")
    next_value_array = _numeric_array(next_values, "next_values")
    if value_array.shape != reward_array.shape or next_value_array.shape != reward_array.shape:
        raise ValueError("rewards, values, and next_values must have equal shape")
    mask = _boolean_array(valid_mask, "valid_mask", shape=reward_array.shape)
    terminated_array = _boolean_array(
        terminated, "terminated", shape=reward_array.shape
    )
    truncated_array = _boolean_array(truncated, "truncated", shape=reward_array.shape)
    if not isinstance(bootstrap_truncated, bool):
        raise ValueError("bootstrap_truncated must be a boolean")
    if np.any(terminated_array & truncated_array):
        raise ValueError("a transition cannot be both terminated and truncated")
    if np.any((terminated_array | truncated_array) & ~mask):
        raise ValueError("padding transitions cannot be terminated or truncated")
    gamma_value = _unit_interval(gamma, "gamma")
    lambda_value = _unit_interval(gae_lambda, "gae_lambda")

    bootstrap_mask = mask & ~terminated_array
    if not bootstrap_truncated:
        bootstrap_mask &= ~truncated_array
    continuation_mask = mask & ~(terminated_array | truncated_array)

    advantages = np.zeros_like(reward_array)
    td_residuals = np.zeros_like(reward_array)
    row_rewards = reward_array.reshape(-1, reward_array.shape[-1])
    row_values = value_array.reshape(-1, value_array.shape[-1])
    row_next_values = next_value_array.reshape(-1, next_value_array.shape[-1])
    row_mask = mask.reshape(-1, mask.shape[-1])
    row_bootstrap = bootstrap_mask.reshape(-1, bootstrap_mask.shape[-1])
    row_continuation = continuation_mask.reshape(-1, continuation_mask.shape[-1])
    row_advantages = advantages.reshape(-1, advantages.shape[-1])
    row_residuals = td_residuals.reshape(-1, td_residuals.shape[-1])

    for row in range(row_rewards.shape[0]):
        next_advantage = 0.0
        for step in range(row_rewards.shape[1] - 1, -1, -1):
            if not row_mask[row, step]:
                next_advantage = 0.0
                continue
            residual = (
                row_rewards[row, step]
                + gamma_value
                * float(row_bootstrap[row, step])
                * row_next_values[row, step]
                - row_values[row, step]
            )
            advantage = (
                residual
                + gamma_value
                * lambda_value
                * float(row_continuation[row, step])
                * next_advantage
            )
            row_residuals[row, step] = residual
            row_advantages[row, step] = advantage
            next_advantage = advantage

    returns = np.zeros_like(value_array)
    returns[mask] = advantages[mask] + value_array[mask]
    return GeneralizedAdvantageEstimate(
        advantages=advantages,
        returns=returns,
        td_residuals=td_residuals,
        bootstrap_mask=bootstrap_mask,
        continuation_mask=continuation_mask,
    )


def ppo_clipped_surrogate(
    new_log_probabilities: ArrayLike,
    old_log_probabilities: ArrayLike,
    advantages: ArrayLike,
    *,
    valid_mask: ArrayLike,
    clip_epsilon: float = 0.2,
) -> PPOClippedSurrogateReport:
    """Evaluate the PPO clipped sampled-action policy objective.

    ``approximate_sampled_kl`` is the mask mean of
    ``exp(log_ratio) - 1 - log_ratio``. It is a sampled diagnostic, not an exact
    categorical or sequence-distribution KL and not a hard constraint.
    """

    new_log_probs = _numeric_array(new_log_probabilities, "new_log_probabilities")
    old_log_probs = _numeric_array(old_log_probabilities, "old_log_probabilities")
    advantage_array = _numeric_array(advantages, "advantages")
    if old_log_probs.shape != new_log_probs.shape or advantage_array.shape != new_log_probs.shape:
        raise ValueError("new/old log probabilities and advantages must have equal shape")
    mask = _boolean_array(valid_mask, "valid_mask", shape=new_log_probs.shape)
    if not np.any(mask):
        raise ValueError("valid_mask must select at least one action")
    epsilon = _unit_interval(clip_epsilon, "clip_epsilon", allow_zero=False)

    log_ratios = np.zeros_like(new_log_probs)
    log_ratios[mask] = new_log_probs[mask] - old_log_probs[mask]
    probability_ratios = np.ones_like(log_ratios)
    with np.errstate(over="ignore", invalid="ignore"):
        probability_ratios[mask] = np.exp(log_ratios[mask])
    if not np.all(np.isfinite(probability_ratios[mask])):
        raise ValueError("active log-probability differences produce non-finite ratios")

    clipped_ratios = np.ones_like(probability_ratios)
    clipped_ratios[mask] = np.clip(
        probability_ratios[mask], 1 - epsilon, 1 + epsilon
    )
    unclipped_objective = np.zeros_like(advantage_array)
    clipped_objective = np.zeros_like(advantage_array)
    unclipped_objective[mask] = probability_ratios[mask] * advantage_array[mask]
    clipped_objective[mask] = clipped_ratios[mask] * advantage_array[mask]
    if not (
        np.all(np.isfinite(unclipped_objective[mask]))
        and np.all(np.isfinite(clipped_objective[mask]))
    ):
        raise ValueError("active PPO objective terms must remain finite")
    surrogate = np.zeros_like(advantage_array)
    surrogate[mask] = np.minimum(
        unclipped_objective[mask], clipped_objective[mask]
    )
    clipped_actions = mask & (
        (probability_ratios < 1 - epsilon) | (probability_ratios > 1 + epsilon)
    )
    approximate_kl_terms = np.zeros_like(log_ratios)
    approximate_kl_terms[mask] = np.expm1(log_ratios[mask]) - log_ratios[mask]

    objective = masked_mean(surrogate, mask)
    return PPOClippedSurrogateReport(
        sampled_action_count=int(np.sum(mask)),
        clip_epsilon=epsilon,
        surrogate_objective=objective,
        policy_loss=-objective,
        mean_unclipped_objective=masked_mean(unclipped_objective, mask),
        mean_clipped_objective=masked_mean(clipped_objective, mask),
        clip_fraction=float(np.mean(clipped_actions[mask])),
        mean_probability_ratio=masked_mean(probability_ratios, mask),
        mean_log_probability_ratio=masked_mean(log_ratios, mask),
        approximate_sampled_kl=masked_mean(approximate_kl_terms, mask),
        probability_ratios=probability_ratios,
        clipped_probability_ratios=clipped_ratios,
        per_action_surrogate=surrogate,
        clipped_actions=clipped_actions,
    )
