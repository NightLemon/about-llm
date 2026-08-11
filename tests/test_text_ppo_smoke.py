from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "single-gpu-finetuning" / "smoke_text_ppo.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_text_ppo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load text PPO smoke script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_text_ppo_executes_chat_rollout_boundaries_and_optimizer() -> None:
    report = _load_script().run_smoke()

    assert report["model_class"] == "separate GPT2 policy/value backbones+heads"
    assert report["rendered_prompt"] == (
        "<|system|> Return one word. </s> "
        "<|user|> Say good. </s> <|assistant|> "
    )
    assert report["prompt_token_count"] == 10
    assert report["vocab_size"] == 13
    assert report["initial_exact_expected_task_reward"] == pytest.approx(25 / 169)
    assert report["initial_exact_good_then_eos_probability"] == pytest.approx(1 / 169)
    assert report["final_exact_expected_task_reward"] > 1.9
    assert report["final_exact_good_then_eos_probability"] > 0.95
    assert report["last_sampled_task_reward_mean"] > report[
        "first_sampled_task_reward_mean"
    ]
    assert report["total_optimizer_steps"] == 96
    assert report["bootstrap_truncated_in_optimizer"] is False
    assert report["optimizer_matches_reported_finite_horizon_objective"] is True
    assert report["reference_parameters_unchanged"] is True
    assert report["backbone_parameters_changed"] is True
    assert report["value_backbone_parameters_changed"] is True
    assert report["policy_head_parameters_changed"] is True
    assert report["value_head_parameters_changed"] is True
    assert report["all_stored_old_log_probabilities_unchanged"] is True
    assert report["maximum_snapshot_log_probability_error"] <= 1e-7

    initial = report["initial_exact_objectives"]
    assert initial == pytest.approx(
        {
            "expected_task_reward": 25 / 169,
            "good_first_probability": 1 / 13,
            "eos_after_good_probability": 1 / 13,
            "good_then_eos_probability": 1 / 169,
        }
    )
    final = report["final_exact_objectives"]
    assert final["good_first_probability"] > 0.95
    assert final["eos_after_good_probability"] > 0.95

    iterations = report["iterations"]
    assert len(iterations) == 8
    first = iterations[0]
    assert first["rollout_episode_count"] == 128
    assert first["terminated_transition_count"] > 0
    assert first["truncated_transition_count"] > 0
    assert first["padding_transition_count"] > 0
    assert first["truncated_post_action_value_count"] == first[
        "truncated_transition_count"
    ]
    assert first["exact_categorical_kl_at_valid_states_mean"] == pytest.approx(0)
    assert first["sampled_reference_log_ratio_mean"] == pytest.approx(0)
    assert first["post_update_maximum_ratio"] > 1.2
    assert first["post_update_minimum_ratio"] < 0.8
    for iteration in iterations:
        assert iteration["old_log_probabilities_require_grad"] is False
        assert iteration["stored_old_log_probabilities_unchanged"] is True
        assert iteration["snapshot_log_probability_max_error"] <= 1e-7
        assert iteration["truncated_post_action_value_count"] == iteration[
            "truncated_transition_count"
        ]
        assert iteration["optimizer_steps"] == 12
        assert iteration["exact_categorical_kl_at_valid_states_mean"] >= -1e-7

    assert report["scope"] == {
        "device": "CPU",
        "local_wordlevel_tokenizer_executed": True,
        "chat_template_and_natural_language_prompt_executed": True,
        "random_tiny_gpt2_backbone_executed": True,
        "autoregressive_text_token_sampling_executed": True,
        "eos_termination_executed": True,
        "max_new_tokens_truncation_executed": True,
        "padding_mask_executed": True,
        "truncated_post_action_values_computed": True,
        "truncated_transition_value_bootstrap_executed": False,
        "finite_horizon_task_return_stops_at_generation_cap": True,
        "optimizer_matches_reported_finite_horizon_objective": True,
        "exact_short_horizon_objective_enumerated": True,
        "learned_reward_model_executed": False,
        "human_preference_or_natural_language_quality_proved": False,
        "target_checkpoint_executed": False,
        "checkpoint_or_resume_executed": False,
        "cuda_or_distributed_execution": False,
    }


def test_text_ppo_small_control_is_reproducible() -> None:
    module = _load_script()
    kwargs = {
        "iterations": 2,
        "episodes_per_iteration": 16,
        "epochs": 2,
        "minibatch_size": 16,
    }
    first = module.run_smoke(**kwargs)
    second = module.run_smoke(**kwargs)
    assert first == second


def test_text_ppo_reports_counterfactual_truncation_bootstrap_scope() -> None:
    report = _load_script().run_smoke(
        iterations=1,
        episodes_per_iteration=16,
        epochs=1,
        minibatch_size=16,
        bootstrap_truncated=True,
    )
    assert report["bootstrap_truncated_in_optimizer"] is True
    assert report["optimizer_matches_reported_finite_horizon_objective"] is False
    assert report["scope"]["truncated_transition_value_bootstrap_executed"] is True


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("iterations", 0),
        ("episodes_per_iteration", True),
        ("epochs", -1),
        ("minibatch_size", 0),
        ("learning_rate", float("nan")),
        ("learning_rate", 0),
        ("kl_coefficient", -1),
        ("value_coefficient", float("inf")),
        ("entropy_coefficient", -1),
        ("bootstrap_truncated", 1),
    ],
)
def test_text_ppo_rejects_invalid_configuration(
    parameter: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _load_script().run_smoke(**{parameter: value})
