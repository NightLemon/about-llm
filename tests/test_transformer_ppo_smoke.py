from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "single-gpu-finetuning" / "smoke_transformer_ppo.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_transformer_ppo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Transformer PPO smoke script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_transformer_ppo_executes_token_rollout_reference_kl_and_optimizer() -> None:
    report = _load_script().run_smoke()

    assert report["model_class"] == "GPT2Model+policy_head+value_head"
    assert report["initial_exact_expected_task_reward"] == pytest.approx(1 / 3)
    assert report["final_exact_expected_task_reward"] > 1.8
    assert report["last_sampled_task_reward_mean"] > report[
        "first_sampled_task_reward_mean"
    ]
    assert report["total_optimizer_steps"] == 36
    assert report["reference_parameters_unchanged"] is True
    assert report["backbone_parameters_changed"] is True
    assert report["policy_head_parameters_changed"] is True
    assert report["value_head_parameters_changed"] is True
    assert report["all_stored_old_log_probabilities_unchanged"] is True
    assert report["maximum_snapshot_log_probability_error"] <= 1e-7

    iterations = report["iterations"]
    assert len(iterations) == 6
    assert iterations[0]["exact_categorical_kl_at_sampled_states_mean"] == pytest.approx(0)
    assert iterations[0]["sampled_reference_log_ratio_mean"] == pytest.approx(0)
    assert iterations[-1]["exact_categorical_kl_at_sampled_states_mean"] > 0
    for iteration in iterations:
        assert iteration["rollout_episode_count"] == 64
        assert iteration["rollout_action_count"] == 128
        assert iteration["terminated_transition_count"] == 64
        assert iteration["truncated_transition_count"] == 0
        assert iteration["old_log_probabilities_require_grad"] is False
        assert iteration["stored_old_log_probabilities_unchanged"] is True
        assert iteration["snapshot_log_probability_max_error"] <= 1e-7
        assert iteration["optimizer_steps"] == 6
        assert iteration["exact_categorical_kl_at_sampled_states_mean"] >= -1e-7

    assert iterations[0]["post_update_maximum_ratio"] > 1.2
    assert iterations[0]["post_update_minimum_ratio"] < 0.8
    assert report["scope"] == {
        "device": "CPU",
        "integer_token_ids_without_tokenizer": True,
        "random_tiny_gpt2_backbone_executed": True,
        "autoregressive_token_sampling_executed": True,
        "frozen_reference_forward_executed": True,
        "sampled_reference_log_ratio_reward_executed": True,
        "exact_two_step_task_reward_enumerated": True,
        "gae_and_transformer_optimizer_executed": True,
        "learned_reward_model_executed": False,
        "natural_language_quality_proved": False,
        "time_limit_truncation_executed": False,
        "checkpoint_or_resume_executed": False,
        "cuda_or_distributed_execution": False,
        "target_llm_ppo_quality_or_safety_proved": False,
    }


def test_transformer_ppo_small_control_is_reproducible() -> None:
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
    ],
)
def test_transformer_ppo_rejects_invalid_configuration(
    parameter: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _load_script().run_smoke(**{parameter: value})
