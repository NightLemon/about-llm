"""手算 GAE、PPO clipped surrogate 与 sampled action ratio 的边界。

实验分别覆盖 terminal 与 truncation 的 bootstrap 差异、padding mask、正负 advantage 下 clipping
方向，以及“只约束采到的 action ratio”并不限制完整策略 KL 的反例。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np

from about_llm.finetuning import (
    generalized_advantage_estimation,
    ppo_clipped_surrogate,
)


def run_experiment() -> dict[str, Any]:
    """运行 GAE、截断 bootstrap、PPO clip 与 ratio 反例。"""

    # 第三行填入巨大数但 valid_mask=False，验证 padding 不参与递推或 loss。
    gae = generalized_advantage_estimation(
        rewards=[0.0, 1.0, 999.0],
        values=[0.5, 0.25, 999.0],
        next_values=[0.25, 10.0, 999.0],
        valid_mask=[True, True, False],
        terminated=[False, True, False],
        truncated=[False, False, False],
        gamma=0.9,
        gae_lambda=0.8,
        bootstrap_truncated=True,
    )
    # 时间上限 truncation 可从 next value bootstrap；真正 terminal 则不能。
    truncated_with_bootstrap = generalized_advantage_estimation(
        rewards=[1.0],
        values=[0.5],
        next_values=[2.0],
        valid_mask=[True],
        terminated=[False],
        truncated=[True],
        gamma=0.9,
        gae_lambda=0.8,
        bootstrap_truncated=True,
    )
    truncated_without_bootstrap = generalized_advantage_estimation(
        rewards=[1.0],
        values=[0.5],
        next_values=[2.0],
        valid_mask=[True],
        terminated=[False],
        truncated=[True],
        gamma=0.9,
        gae_lambda=0.8,
        bootstrap_truncated=False,
    )

    ratios = np.array([1.5, 0.5, 1.0], dtype=np.float64)
    ppo = ppo_clipped_surrogate(
        np.log(ratios),
        np.zeros_like(ratios),
        [1.0, -1.0, 1.0],
        valid_mask=[True, True, True],
        clip_epsilon=0.2,
    )

    tail_probability = 1e-12
    old_distribution = np.array([0.1, 0.45, 0.45], dtype=np.float64)
    new_distribution = np.array(
        [0.1, 0.9 - tail_probability, tail_probability], dtype=np.float64
    )
    sampled_action = 0
    sampled_ratio = (
        new_distribution[sampled_action] / old_distribution[sampled_action]
    )
    full_forward_kl = float(
        np.sum(old_distribution * np.log(old_distribution / new_distribution))
    )
    sampled_report = ppo_clipped_surrogate(
        [math.log(new_distribution[sampled_action])],
        [math.log(old_distribution[sampled_action])],
        [1.0],
        valid_mask=[True],
        clip_epsilon=0.2,
    )

    ppo_scalars = {
        key: value
        for key, value in asdict(ppo).items()
        if not isinstance(value, np.ndarray)
    }
    return {
        "gae": {
            "advantages": gae.advantages.tolist(),
            "returns": gae.returns.tolist(),
            "td_residuals": gae.td_residuals.tolist(),
            "bootstrap_mask": gae.bootstrap_mask.tolist(),
            "continuation_mask": gae.continuation_mask.tolist(),
        },
        "truncation_bootstrap_control": {
            "with_bootstrap_advantage": float(
                truncated_with_bootstrap.advantages[0]
            ),
            "without_bootstrap_advantage": float(
                truncated_without_bootstrap.advantages[0]
            ),
            "recursion_continues_across_truncation": bool(
                truncated_with_bootstrap.continuation_mask[0]
            ),
        },
        "ppo": {
            **ppo_scalars,
            "probability_ratios": ppo.probability_ratios.tolist(),
            "clipped_probability_ratios": (
                ppo.clipped_probability_ratios.tolist()
            ),
            "per_action_surrogate": ppo.per_action_surrogate.tolist(),
            "clipped_actions": ppo.clipped_actions.tolist(),
        },
        "sampled_ratio_counterexample": {
            "old_distribution": old_distribution.tolist(),
            "new_distribution": new_distribution.tolist(),
            "sampled_action": sampled_action,
            "sampled_probability_ratio": float(sampled_ratio),
            "sampled_clip_fraction": sampled_report.clip_fraction,
            "sampled_approximate_kl": sampled_report.approximate_sampled_kl,
            "full_distribution_forward_kl": full_forward_kl,
        },
        "scope": {
            "device": "CPU",
            "authored_rewards_values_and_distributions": True,
            "numpy_objectives_executed": True,
            "rollout_engine_or_language_model_executed": False,
            "reward_or_value_model_quality_proved": False,
            "full_distribution_kl_constrained": False,
            "stable_ppo_training_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
