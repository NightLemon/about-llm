from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "single-gpu-finetuning" / "smoke_torch_ppo.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_torch_ppo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load torch PPO smoke script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_torch_ppo_smoke_executes_rollout_gae_and_optimizer() -> None:
    report = _load_script().run_smoke()

    assert report["initial_exact_expected_return"] == pytest.approx(1.0)
    assert report["final_exact_expected_return"] > 1.8
    assert report["last_rollout_reward_mean"] > report["first_rollout_reward_mean"]
    assert report["policy_parameters_changed"] is True
    assert report["value_parameters_changed"] is True
    assert report["all_stored_old_log_probabilities_unchanged"] is True
    assert report["all_snapshot_log_probability_errors_zero"] is True
    assert report["total_optimizer_steps"] == 96
    assert len(report["iterations"]) == 6
    for iteration in report["iterations"]:
        assert iteration["rollout_episode_count"] == 128
        assert iteration["rollout_action_count"] == 256
        assert iteration["terminated_transition_count"] == 128
        assert iteration["truncated_transition_count"] == 0
        assert iteration["old_log_probabilities_require_grad"] is False
        assert iteration["optimizer_steps"] == 16
        assert iteration["snapshot_log_probability_max_error"] == 0
        assert iteration["post_update_objective"]["clip_fraction"] >= 0
    first_post_update = report["iterations"][0]["post_update_objective"]
    assert first_post_update["maximum_probability_ratio"] > 1.2
    assert first_post_update["minimum_probability_ratio"] < 0.8

    assert report["scope"] == {
        "device": "CPU",
        "authored_two_state_environment": True,
        "on_policy_categorical_sampling_executed": True,
        "torch_policy_and_value_forward_executed": True,
        "gae_and_minibatch_optimizer_executed": True,
        "time_limit_truncation_executed": False,
        "language_model_or_tokenizer_executed": False,
        "reward_model_executed": False,
        "reference_policy_kl_controller_executed": False,
        "gpu_or_distributed_execution": False,
        "target_model_quality_or_safety_proved": False,
        "production_ppo_stability_proved": False,
    }


def test_torch_ppo_smoke_is_reproducible() -> None:
    module = _load_script()
    first = module.run_smoke(iterations=2, episodes_per_iteration=32)
    second = module.run_smoke(iterations=2, episodes_per_iteration=32)
    assert first == second


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("iterations", 0),
        ("episodes_per_iteration", True),
        ("epochs", -1),
        ("minibatch_size", 0),
        ("learning_rate", float("nan")),
    ],
)
def test_torch_ppo_smoke_rejects_invalid_configuration(
    parameter: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _load_script().run_smoke(**{parameter: value})
