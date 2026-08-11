"""Train confounded and counterfactually balanced linear reward-model controls."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np

from about_llm.finetuning import (
    fit_linear_pairwise_reward_model,
    pairwise_reward_metrics,
)


def run_experiment() -> dict[str, Any]:
    rejected = np.zeros((4, 2), dtype=np.float64)
    confounded_chosen = np.array(
        [[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.float64
    )
    counterfactual_chosen = np.array(
        [[1, -1], [2, -2], [1, -1], [2, -2]], dtype=np.float64
    )
    held_out_chosen = np.array([[1, -2], [2, -3]], dtype=np.float64)
    held_out_rejected = np.zeros_like(held_out_chosen)

    confounded = fit_linear_pairwise_reward_model(
        confounded_chosen,
        rejected,
        steps=300,
        learning_rate=0.1,
    )
    balanced_chosen = np.concatenate(
        (confounded_chosen, counterfactual_chosen), axis=0
    )
    balanced_rejected = np.zeros_like(balanced_chosen)
    balanced = fit_linear_pairwise_reward_model(
        balanced_chosen,
        balanced_rejected,
        steps=300,
        learning_rate=0.1,
    )
    return {
        "schema_version": 1,
        "feature_semantics": [
            "authored_quality_signal",
            "authored_length_proxy",
        ],
        "confounded_training": asdict(confounded),
        "confounded_held_out": asdict(
            pairwise_reward_metrics(
                held_out_chosen,
                held_out_rejected,
                confounded.weights,
            )
        ),
        "counterfactually_balanced_training": asdict(balanced),
        "counterfactually_balanced_held_out": asdict(
            pairwise_reward_metrics(
                held_out_chosen,
                held_out_rejected,
                balanced.weights,
            )
        ),
        "scope": {
            "device": "CPU",
            "authored_numeric_features_and_preferences": True,
            "text_tokenizer_or_transformer_executed": False,
            "human_preference_quality_proved": False,
            "target_reward_model_quality_proved": False,
            "reward_hacking_or_policy_optimization_evaluated": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
