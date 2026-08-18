"""Run exact finite controls for REINFORCE and group-relative advantages."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np

from about_llm.finetuning import (
    categorical_policy_gradient,
    group_relative_advantages,
    variance_minimizing_score_baseline,
)


def _jsonable_report(report: object) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in asdict(report).items()
    }


def run_experiment() -> dict[str, Any]:
    logits = [-0.4, 0.1, 0.3]
    rewards = [0.0, 1.0, 4.0]
    zero = categorical_policy_gradient(logits, rewards, baseline=0)
    value = categorical_policy_gradient(logits, rewards, baseline=zero.expected_reward)
    optimal_baseline = variance_minimizing_score_baseline(logits, rewards)
    optimal = categorical_policy_gradient(
        logits, rewards, baseline=optimal_baseline
    )
    varied_group = group_relative_advantages([0.0, 1.0, 4.0, 4.0])
    tied_group = group_relative_advantages([2.0, 2.0, 2.0, 2.0])

    return {
        "categorical_bandit": {
            "logits": logits,
            "zero_baseline": _jsonable_report(zero),
            "expected_reward_baseline": _jsonable_report(value),
            "variance_minimizing_baseline": _jsonable_report(optimal),
        },
        "group_relative_advantages": {
            "varied_rewards": _jsonable_report(varied_group),
            "tied_rewards": _jsonable_report(tied_group),
        },
        "scope": {
            "device": "CPU",
            "all_categorical_actions_enumerated": True,
            "monte_carlo_sampling_executed": False,
            "environment_or_language_model_executed": False,
            "complete_grpo_or_ppo_training_proved": False,
            "reward_validity_or_policy_quality_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
