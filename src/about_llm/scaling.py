"""Small, assumption-explicit utilities for scaling-law exercises."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _non_negative_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def estimate_dense_training_flops(
    num_parameters: float,
    training_tokens: float,
    *,
    flops_per_parameter_token: float = 6.0,
) -> float:
    """Return a budgeting approximation, not measured hardware FLOPs.

    The default ``6ND`` convention approximates forward and backward model FLOPs
    for a dense Transformer. It intentionally does not model attention-length
    corrections, rematerialization, optimizer kernels, padding, communication,
    sparsity, or achieved hardware utilization.
    """

    parameters = _positive_finite("num_parameters", num_parameters)
    tokens = _positive_finite("training_tokens", training_tokens)
    multiplier = _positive_finite(
        "flops_per_parameter_token", flops_per_parameter_token
    )
    return multiplier * parameters * tokens


@dataclass(frozen=True)
class ComputeOptimalEstimate:
    """Analytic optimum under one fitted separable power-law model."""

    num_parameters: float
    training_tokens: float
    compute_flops: float
    modeled_loss: float


def compute_optimal_under_power_law(
    compute_flops: float,
    *,
    parameter_coefficient: float,
    data_coefficient: float,
    parameter_exponent: float,
    data_exponent: float,
    irreducible_loss: float = 0.0,
    flops_per_parameter_token: float = 6.0,
) -> ComputeOptimalEstimate:
    """Minimize ``L_inf + a*N^-alpha + b*D^-beta`` under ``C=k*N*D``.

    All coefficients and exponents must come from a compatible empirical fit.
    The result is not a universal prescription and may extrapolate poorly beyond
    the model sizes, token budgets, architecture, and data mixture used in that fit.
    """

    compute = _positive_finite("compute_flops", compute_flops)
    a = _positive_finite("parameter_coefficient", parameter_coefficient)
    b = _positive_finite("data_coefficient", data_coefficient)
    alpha = _positive_finite("parameter_exponent", parameter_exponent)
    beta = _positive_finite("data_exponent", data_exponent)
    loss_floor = _non_negative_finite("irreducible_loss", irreducible_loss)
    k = _positive_finite("flops_per_parameter_token", flops_per_parameter_token)

    # At the stationary point:
    # beta*b*(k/C)^beta*N^(alpha+beta) = alpha*a.
    num_parameters = (
        (alpha * a) / (beta * b) * (compute / k) ** beta
    ) ** (1.0 / (alpha + beta))
    training_tokens = compute / (k * num_parameters)
    modeled_loss = (
        loss_floor
        + a * num_parameters ** (-alpha)
        + b * training_tokens ** (-beta)
    )
    return ComputeOptimalEstimate(
        num_parameters=num_parameters,
        training_tokens=training_tokens,
        compute_flops=compute,
        modeled_loss=modeled_loss,
    )
