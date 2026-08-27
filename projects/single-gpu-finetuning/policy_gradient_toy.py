"""用三动作策略精确演示 REINFORCE baseline 与 group-relative advantage。

实验枚举全部动作，因此能直接计算期望梯度和梯度方差；它比较零 baseline、value baseline
与最小方差 baseline，并展示 group 内奖励全相同时归一化 advantage 应为零。
"""

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
    """把包含 NumPy 数组的报告转换为 JSON 基础类型。"""

    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in asdict(report).items()
    }


def run_experiment() -> dict[str, Any]:
    """计算三种 baseline 的精确策略梯度和两组 group advantage。"""

    # 三个 logit 定义当前策略，rewards 是每个离散动作的完整回报表。
    logits = [-0.4, 0.1, 0.3]
    rewards = [0.0, 1.0, 4.0]
    zero = categorical_policy_gradient(logits, rewards, baseline=0)
    value = categorical_policy_gradient(logits, rewards, baseline=zero.expected_reward)
    # 最小方差 baseline 考虑 score-function 范数，不一定等于普通期望回报。
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
