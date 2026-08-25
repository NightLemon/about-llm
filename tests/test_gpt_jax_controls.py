from __future__ import annotations

import pytest

from about_llm.from_scratch.gpt_cross_framework import (
    run_gpt_cross_framework_parity_control,
)
from about_llm.from_scratch.gpt_cross_framework_training import (
    run_gpt_cross_framework_training_parity_control,
)
from about_llm.from_scratch.jax_training_resume import (
    run_jax_training_resume_control,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.extended,
    pytest.mark.slow,
]


@pytest.mark.formula
def test_pytorch_jax_forward_gradient_and_sgd_control() -> None:
    report = run_gpt_cross_framework_parity_control()

    assert all(report["assertions"].values())
    scope = report["scope"]
    assert scope["same_initial_parameter_values_compared"] is True
    assert scope["every_unique_parameter_gradient_compared"] is True
    assert scope["native_rmsnorm_architecture_counterfactual_executed"] is True
    assert scope["framework_rng_equivalence_claimed"] is False


@pytest.mark.formula
def test_pytorch_jax_three_step_adamw_control() -> None:
    report = run_gpt_cross_framework_training_parity_control()

    assert all(report["assertions"].values())
    scope = report["scope"]
    assert scope["adamw_first_second_moments_and_count_compared"] is True
    assert scope["learning_rate_schedule_compared"] is True
    assert scope["wrong_materialized_mask_counterfactual_executed"] is True
    assert scope["framework_native_rng_equivalence_claimed"] is False


@pytest.mark.contract
def test_cross_process_jax_checkpoint_resume_control() -> None:
    report = run_jax_training_resume_control()

    assert all(report["assertions"].values())
    scope = report["scope"]
    assert scope["cross_process_split_resume_executed"] is True
    assert scope["bit_exact_full_state_and_trace_compared"] is True
    assert scope["wrong_prng_and_cursor_counterfactuals_executed"] is True
    assert scope["artifact_origin_authentication_or_confidentiality_proved"] is False
