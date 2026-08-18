from __future__ import annotations

import math

import pytest

from about_llm.finetuning.preference import (
    bradley_terry_loss,
    dpo_logit,
    dpo_loss,
    sequence_log_probability,
)

pytestmark = pytest.mark.formula


def test_sequence_log_probability_makes_length_convention_explicit() -> None:
    token_logps = [-0.1, -0.2, -0.3]
    assert sequence_log_probability(token_logps) == pytest.approx(-0.6)
    assert sequence_log_probability(token_logps, reduction="mean") == pytest.approx(-0.2)


def test_equal_rewards_or_reference_equivalent_policy_have_log_two_loss() -> None:
    assert bradley_terry_loss(3.0, 3.0) == pytest.approx(math.log(2))
    assert dpo_loss(
        chosen_policy_logp=-2.0,
        rejected_policy_logp=-4.0,
        chosen_reference_logp=-3.0,
        rejected_reference_logp=-5.0,
        beta=0.1,
    ) == pytest.approx(math.log(2))


def test_dpo_uses_reference_relative_not_raw_policy_margin() -> None:
    logit = dpo_logit(
        chosen_policy_logp=-2.0,
        rejected_policy_logp=-5.0,
        chosen_reference_logp=-2.5,
        rejected_reference_logp=-3.5,
        beta=0.2,
    )
    assert logit == pytest.approx(0.4)

    improved = dpo_loss(
        chosen_policy_logp=-2.0,
        rejected_policy_logp=-5.0,
        chosen_reference_logp=-2.5,
        rejected_reference_logp=-3.5,
        beta=0.2,
    )
    regressed = dpo_loss(
        chosen_policy_logp=-4.0,
        rejected_policy_logp=-5.0,
        chosen_reference_logp=-2.5,
        rejected_reference_logp=-3.5,
        beta=0.2,
    )
    assert improved < regressed


def test_preference_losses_are_stable_for_large_margins() -> None:
    assert bradley_terry_loss(1_000.0, -1_000.0) == pytest.approx(0.0)
    assert bradley_terry_loss(-1_000.0, 1_000.0) == pytest.approx(2_000.0)
    assert dpo_loss(
        chosen_policy_logp=-1_000.0,
        rejected_policy_logp=0.0,
        chosen_reference_logp=0.0,
        rejected_reference_logp=-1_000.0,
        beta=10.0,
    ) == pytest.approx(
        20_000.0
    )


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, True])
def test_preference_objectives_reject_non_finite_values(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        bradley_terry_loss(bad, 0.0)
    with pytest.raises(ValueError, match="finite"):
        sequence_log_probability([bad])


@pytest.mark.parametrize("beta", [0.0, -1.0, math.inf, math.nan, True])
def test_dpo_beta_must_be_positive_and_finite(beta: float) -> None:
    with pytest.raises(ValueError):
        dpo_loss(
            chosen_policy_logp=-1,
            rejected_policy_logp=-2,
            chosen_reference_logp=-1,
            rejected_reference_logp=-2,
            beta=beta,
        )


def test_empty_sequence_and_unknown_reduction_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sequence_log_probability([])
    with pytest.raises(ValueError, match="reduction"):
        sequence_log_probability([-1.0], reduction="median")  # type: ignore[arg-type]
