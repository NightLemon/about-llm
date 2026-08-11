"""Numerically stable preference-objective primitives for teaching and tests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

LogProbabilityReduction = Literal["sum", "mean"]


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _positive_finite(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _softplus(value: float) -> float:
    """Compute log(1 + exp(value)) without overflowing."""

    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def sequence_log_probability(
    token_log_probabilities: Iterable[float],
    *,
    reduction: LogProbabilityReduction = "sum",
) -> float:
    """Aggregate response-token log probabilities under an explicit convention.

    Prompt, padding, and other masked positions must be removed by the caller.
    DPO is normally defined with the response sequence log-probability sum. Using
    a mean changes the objective and is exposed only to make that choice auditable.
    """

    values = [
        _finite(f"token_log_probabilities[{index}]", value)
        for index, value in enumerate(token_log_probabilities)
    ]
    if not values:
        raise ValueError("token_log_probabilities must not be empty")
    if reduction == "sum":
        return math.fsum(values)
    if reduction == "mean":
        return math.fsum(values) / len(values)
    raise ValueError("reduction must be 'sum' or 'mean'")


def bradley_terry_loss(chosen_reward: float, rejected_reward: float) -> float:
    """Return ``-log(sigmoid(r_chosen - r_rejected))``."""

    chosen = _finite("chosen_reward", chosen_reward)
    rejected = _finite("rejected_reward", rejected_reward)
    return _softplus(-(chosen - rejected))


def dpo_logit(
    *,
    chosen_policy_logp: float,
    rejected_policy_logp: float,
    chosen_reference_logp: float,
    rejected_reference_logp: float,
    beta: float,
) -> float:
    """Return the reference-relative DPO classification logit.

    Each log-probability must use the same response-token mask and aggregation
    convention. The function does not tokenize text or run a policy model.
    """

    policy_margin = _finite("chosen_policy_logp", chosen_policy_logp) - _finite(
        "rejected_policy_logp", rejected_policy_logp
    )
    reference_margin = _finite(
        "chosen_reference_logp", chosen_reference_logp
    ) - _finite("rejected_reference_logp", rejected_reference_logp)
    return _positive_finite("beta", beta) * (policy_margin - reference_margin)


def dpo_loss(
    *,
    chosen_policy_logp: float,
    rejected_policy_logp: float,
    chosen_reference_logp: float,
    rejected_reference_logp: float,
    beta: float,
) -> float:
    """Return the per-pair negative log-sigmoid DPO loss."""

    logit = dpo_logit(
        chosen_policy_logp=chosen_policy_logp,
        rejected_policy_logp=rejected_policy_logp,
        chosen_reference_logp=chosen_reference_logp,
        rejected_reference_logp=rejected_reference_logp,
        beta=beta,
    )
    return _softplus(-logit)
