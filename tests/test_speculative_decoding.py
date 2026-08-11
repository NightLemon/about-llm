from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from about_llm.inference import (
    audit_speculative_distribution,
    speculative_sample_step,
    verify_speculative_block,
)

ROOT = Path(__file__).resolve().parents[1]


def test_one_step_marginal_equals_target_and_acceptance_is_one_minus_tv() -> None:
    audit = audit_speculative_distribution(
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
    )

    assert audit.theoretical_output_probabilities == pytest.approx(
        audit.target_probabilities
    )
    assert audit.acceptance_probability == pytest.approx(0.6)
    assert audit.rejection_probability == pytest.approx(0.4)
    assert audit.total_variation_distance == pytest.approx(0.4)
    assert audit.acceptance_probability == pytest.approx(
        1 - audit.total_variation_distance
    )
    assert audit.maximum_target_difference < 1e-15


def test_equal_distributions_are_always_accepted() -> None:
    audit = audit_speculative_distribution([0.25, 0.75], [0.25, 0.75])
    result = speculative_sample_step(
        [0.25, 0.75],
        [0.25, 0.75],
        draft_uniform=0.9,
        acceptance_uniform=np.nextafter(1.0, 0.0),
        correction_uniform=0.0,
    )

    assert audit.acceptance_probability == 1
    assert audit.rejection_probability == 0
    assert result.accepted
    assert result.proposed_token == result.output_token == 1
    assert result.correction_probabilities is None


def test_rejection_samples_positive_target_minus_draft_residual() -> None:
    rejected = speculative_sample_step(
        [0.6, 0.4],
        [0.2, 0.8],
        draft_uniform=0.1,
        acceptance_uniform=0.5,
        correction_uniform=0.0,
    )
    accepted = speculative_sample_step(
        [0.6, 0.4],
        [0.2, 0.8],
        draft_uniform=0.1,
        acceptance_uniform=0.2,
        correction_uniform=0.0,
    )

    assert rejected.proposed_token == 0
    assert not rejected.accepted
    assert rejected.acceptance_probability == pytest.approx(1 / 3)
    assert rejected.correction_probabilities == pytest.approx((0, 1))
    assert rejected.output_token == 1
    assert accepted.accepted and accepted.output_token == 0


def test_block_stops_at_first_rejection_and_does_not_emit_bonus() -> None:
    result = verify_speculative_block(
        draft_tokens=(0, 0),
        draft_probabilities=((0.5, 0.5), (0.8, 0.2)),
        target_probabilities=((0.5, 0.5), (0.2, 0.8)),
        acceptance_uniforms=(0.9, 0.5),
        correction_uniforms=(0.0, 0.0),
        bonus_target_probabilities=(0.1, 0.9),
        bonus_uniform=0.9,
    )

    assert result.emitted_tokens == (0, 1)
    assert result.accepted_draft_tokens == 1
    assert result.first_rejection_index == 1
    assert not result.used_bonus_target_token
    assert result.acceptance_probabilities == pytest.approx((1, 0.25))


def test_block_emits_one_bonus_token_when_all_proposals_are_accepted() -> None:
    result = verify_speculative_block(
        draft_tokens=(0, 1),
        draft_probabilities=((0.5, 0.5), (0.5, 0.5)),
        target_probabilities=((0.5, 0.5), (0.5, 0.5)),
        acceptance_uniforms=(0.9, 0.9),
        correction_uniforms=(0.0, 0.0),
        bonus_target_probabilities=(0.1, 0.9),
        bonus_uniform=0.5,
    )

    assert result.emitted_tokens == (0, 1, 1)
    assert result.accepted_draft_tokens == 2
    assert result.first_rejection_index is None
    assert result.used_bonus_target_token


@pytest.mark.parametrize(
    ("draft", "target"),
    [
        ([0.5, 0.4], [0.5, 0.5]),
        ([0.5, -0.5], [0.5, 0.5]),
        ([0.5, float("nan")], [0.5, 0.5]),
        ([True, False], [0.5, 0.5]),
        ([1.0], [0.5, 0.5]),
        ([], []),
    ],
)
def test_distribution_contract_rejects_invalid_probabilities(
    draft: list[float], target: list[float]
) -> None:
    with pytest.raises(ValueError):
        audit_speculative_distribution(draft, target)


@pytest.mark.parametrize("uniform", [-0.1, 1.0, float("nan"), True])
def test_step_rejects_invalid_uniforms(uniform: float) -> None:
    with pytest.raises(ValueError):
        speculative_sample_step(
            [0.5, 0.5],
            [0.5, 0.5],
            draft_uniform=uniform,
            acceptance_uniform=0.0,
            correction_uniform=0.0,
        )


def test_block_rejects_zero_mass_proposal() -> None:
    with pytest.raises(ValueError, match="positive draft mass"):
        verify_speculative_block(
            draft_tokens=(1,),
            draft_probabilities=([1.0, 0.0],),
            target_probabilities=([0.5, 0.5],),
            acceptance_uniforms=(0.0,),
            correction_uniforms=(0.0,),
            bonus_target_probabilities=[0.5, 0.5],
            bonus_uniform=0.0,
        )


def test_block_validates_tail_even_when_an_earlier_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        verify_speculative_block(
            draft_tokens=(0, 0),
            draft_probabilities=([1.0, 0.0], [0.5, 0.4]),
            target_probabilities=([0.0, 1.0], [0.5, 0.5]),
            acceptance_uniforms=(0.0, 0.0),
            correction_uniforms=(0.0, 0.0),
            bonus_target_probabilities=[0.5, 0.5],
            bonus_uniform=0.0,
        )


def test_block_rejects_mismatched_input_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        verify_speculative_block(
            draft_tokens=(0,),
            draft_probabilities=([1.0],),
            target_probabilities=([1.0],),
            acceptance_uniforms=(),
            correction_uniforms=(0.0,),
            bonus_target_probabilities=[1.0],
            bonus_uniform=0.0,
        )


def test_speculative_toy_reports_analytic_identity_and_explicit_scope() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "projects"
                / "inference-serving"
                / "speculative_decoding_toy.py"
            ),
            "--seed",
            "23",
            "--trials",
            "5000",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    analytic = artifact["analytic_one_step"]
    assert analytic["acceptance_probability"] == pytest.approx(0.6)
    assert analytic["rejection_probability"] == pytest.approx(0.4)
    assert analytic["theoretical_output_probabilities"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert artifact["monte_carlo"]["maximum_target_difference"] < 0.03
    assert artifact["forced_block_rejection"]["emitted_tokens"] == [0, 1]
    assert artifact["scope"] == {
        "authored_probability_vectors": True,
        "analytic_identity_checked": True,
        "monte_carlo_is_demonstration_not_proof": True,
        "model_forward_or_tokenizer_executed": False,
        "gpu_verification_kernel_executed": False,
        "latency_or_speedup_proved": False,
    }
