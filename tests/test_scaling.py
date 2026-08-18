from __future__ import annotations

import math

import pytest

from about_llm.scaling import (
    compute_optimal_under_power_law,
    estimate_dense_training_flops,
)

pytestmark = pytest.mark.formula


def test_dense_training_flops_uses_explicit_budgeting_convention() -> None:
    assert estimate_dense_training_flops(1e9, 20e9) == pytest.approx(1.2e20)
    assert estimate_dense_training_flops(
        1e9, 20e9, flops_per_parameter_token=8
    ) == pytest.approx(1.6e20)


def test_symmetric_power_law_has_symmetric_optimum() -> None:
    estimate = compute_optimal_under_power_law(
        100,
        parameter_coefficient=1,
        data_coefficient=1,
        parameter_exponent=1,
        data_exponent=1,
        flops_per_parameter_token=1,
    )

    assert estimate.num_parameters == pytest.approx(10)
    assert estimate.training_tokens == pytest.approx(10)
    assert estimate.modeled_loss == pytest.approx(0.2)
    assert estimate.num_parameters * estimate.training_tokens == pytest.approx(100)


def test_optimum_respects_budget_and_scaling_exponents() -> None:
    kwargs = {
        "parameter_coefficient": 2.0,
        "data_coefficient": 3.0,
        "parameter_exponent": 0.4,
        "data_exponent": 0.3,
        "flops_per_parameter_token": 6.0,
    }
    small = compute_optimal_under_power_law(1e18, **kwargs)
    large = compute_optimal_under_power_law(16e18, **kwargs)

    expected_parameter_ratio = 16 ** (0.3 / (0.4 + 0.3))
    expected_token_ratio = 16 ** (0.4 / (0.4 + 0.3))
    assert large.num_parameters / small.num_parameters == pytest.approx(
        expected_parameter_ratio
    )
    assert large.training_tokens / small.training_tokens == pytest.approx(
        expected_token_ratio
    )
    assert 6 * large.num_parameters * large.training_tokens == pytest.approx(16e18)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True])
def test_scaling_inputs_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        estimate_dense_training_flops(bad, 100)

    with pytest.raises(ValueError, match="positive finite"):
        compute_optimal_under_power_law(
            100,
            parameter_coefficient=1,
            data_coefficient=1,
            parameter_exponent=bad,
            data_exponent=1,
        )
