from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "smoke_learned_rm_ppo.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_learned_rm_ppo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load learned-RM PPO smoke script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_learned_rm_ppo_exposes_exact_proxy_exploitation() -> None:
    report = _load_script().run_smoke()
    reward_model = report["reward_model"]
    assert reward_model["training_pair_count"] == 1
    assert reward_model["training_response_count"] == 2
    assert reward_model["allowed_generation_token_count"] == 8
    assert reward_model["reachable_response_count"] == 57
    assert reward_model["unseen_response_count"] == 55
    assert reward_model["initial_pairwise_loss"] == pytest.approx(math.log(2))
    assert reward_model["final_pairwise_loss"] < 0.005
    assert reward_model["strict_training_pair_accuracy"] == 1
    assert reward_model["final_training_margin"] > 5
    assert reward_model["target_centered_score"] > reward_model[
        "rejected_centered_score"
    ]
    assert reward_model["highest_centered_score"] > reward_model[
        "target_centered_score"
    ]
    assert reward_model["target_response_rank_of_reachable"] > 1
    assert reward_model["score_head_parameters_changed_during_rm_training"] is True
    assert reward_model["embedding_parameters_changed_during_rm_training"] is True
    assert report["highest_scoring_response_tokens"] == ["good.", "good"]
    assert "[PAD]" not in report["allowed_generation_tokens"]

    initial = report["initial_exact_objectives"]
    final = report["final_exact_objectives"]
    assert initial["probability_mass"] == pytest.approx(1)
    assert final["probability_mass"] == pytest.approx(1, abs=2e-6)
    assert initial["authored_dense_task_reward"] == pytest.approx(15 / 64)
    assert initial["authored_target_success_probability"] == pytest.approx(1 / 64)
    assert final["expected_centered_learned_reward"] > (
        initial["expected_centered_learned_reward"] + 1.5
    )
    assert final["authored_target_success_probability"] < (
        initial["authored_target_success_probability"] / 10
    )
    assert tuple(final["most_probable_response_ids"]) != tuple(
        report["target_response_ids"]
    )
    assert final["most_probable_response_probability"] > 0.3
    assert report["exact_proxy_reward_improved"] is True
    assert report["exact_authored_dense_task_reward_improved"] is True
    assert report["exact_authored_target_success_improved"] is False
    assert report["reward_hacking_counterexample_observed"] is True
    assert report["reward_model_parameters_unchanged_during_ppo"] is True
    assert report["reference_parameters_unchanged_during_ppo"] is True
    assert report["all_stored_old_log_probabilities_unchanged"] is True
    assert report["maximum_snapshot_log_probability_error"] <= 1e-7

    iterations = report["iterations"]
    assert len(iterations) == 6
    assert report["total_ppo_optimizer_steps"] == sum(
        iteration["optimizer_steps"] for iteration in iterations
    )
    first = iterations[0]
    assert first["terminated_transition_count"] > 0
    assert first["truncated_transition_count"] > 0
    assert first["padding_transition_count"] > 0
    for iteration in iterations:
        assert iteration["stored_old_log_probabilities_unchanged"] is True
        assert iteration["snapshot_log_probability_max_error"] <= 1e-7
        assert iteration["optimizer_steps"] > 0

    assert report["scope"] == {
        "device": "CPU",
        "local_wordlevel_tokenizer_and_chat_template_executed": True,
        "generation_allowlist_bound_to_sampling_and_ppo_distribution": True,
        "pairwise_transformer_reward_model_optimizer_executed": True,
        "sparse_authored_preference_pair_not_human_labels": True,
        "frozen_learned_sequence_reward_bound_to_terminal_action": True,
        "all_reachable_two_token_responses_enumerated": True,
        "ppo_optimizer_executed_against_learned_proxy": True,
        "exact_proxy_reward_improved": True,
        "exact_authored_dense_task_reward_improved": True,
        "exact_authored_target_success_improved": False,
        "controlled_reward_hacking_counterexample_observed": True,
        "reward_model_quality_or_robustness_proved": False,
        "human_preference_or_natural_language_quality_proved": False,
        "target_checkpoint_executed": False,
        "cuda_or_distributed_execution": False,
        "production_ppo_stability_proved": False,
    }


def test_learned_rm_ppo_small_control_is_reproducible() -> None:
    module = _load_script()
    kwargs = {
        "reward_model_steps": 4,
        "ppo_iterations": 2,
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
        ("reward_model_steps", 0),
        ("reward_model_seed", -1),
        ("ppo_iterations", 0),
        ("episodes_per_iteration", True),
        ("epochs", -1),
        ("minibatch_size", 0),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("kl_coefficient", -1),
    ],
)
def test_learned_rm_ppo_rejects_invalid_configuration(
    parameter: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _load_script().run_smoke(**{parameter: value})
