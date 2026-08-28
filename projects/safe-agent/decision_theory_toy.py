"""用有限状态例子串起 belief update、信息价值、硬约束和 Agent 终止性。

前半段让 Agent 在两种故障间更新概率，并比较“先观察再行动”是否值得成本；后半段枚举
有限状态图，区分终态可达、一定终止、可进入禁区和可能无限循环四种性质。
"""

from __future__ import annotations

import json
from typing import Any

from about_llm.agents import (
    TransitionSystemReport,
    ValueOfInformationReport,
    analyze_transition_system,
    select_expected_utility_action,
    update_belief,
    value_of_information,
)


def _information_summary(report: ValueOfInformationReport) -> dict[str, Any]:
    """把观察前后决策与净信息价值整理成可读字段。"""

    return {
        "prior_expected_utilities": report.prior_decision.expected_utilities.tolist(),
        "prior_best_action_index": report.prior_decision.best_action_index,
        "prior_best_expected_utility": report.prior_decision.best_expected_utility,
        "observation_probabilities": report.observation_probabilities.tolist(),
        "posterior_beliefs": report.posterior_beliefs.tolist(),
        "posterior_best_action_indices": report.posterior_best_action_indices.tolist(),
        "expected_utility_with_information": report.expected_utility_with_information,
        "expected_value_of_sample_information": (
            report.expected_value_of_sample_information
        ),
        "observation_cost": report.observation_cost,
        "net_value_of_information": report.net_value_of_information,
        "information_is_worth_cost": report.information_is_worth_cost,
    }


def _transition_summary(report: TransitionSystemReport) -> dict[str, Any]:
    """提取状态图的可达性、安全性与终止性结论。"""

    return {
        "reachable_states": list(report.reachable_states),
        "reachable_terminal_states": list(report.reachable_terminal_states),
        "reachable_forbidden_states": list(report.reachable_forbidden_states),
        "nonterminal_dead_end_states": list(report.nonterminal_dead_end_states),
        "nonterminal_cycle_detected": report.nonterminal_cycle_detected,
        "safety_holds": report.safety_holds,
        "terminal_is_reachable": report.terminal_is_reachable,
        "guaranteed_termination": report.guaranteed_termination,
    }


def run_experiment() -> dict[str, Any]:
    """计算一次 Bayes 更新、两种信号价值和三张状态图。"""

    # prior 表示两类故障的当前信念；identity transition 假设观察期间状态不变。
    prior = [0.6, 0.4]
    identity_transition = [[1.0, 0.0], [0.0, 1.0]]
    strong_signal = [[0.85, 0.15], [0.15, 0.85]]
    weak_signal = [[0.51, 0.49], [0.49, 0.51]]
    # 第四个 action 的效用故意最高，但 allowed_actions=False 表示策略层禁止它。
    utilities = [
        [10.0, -14.0],
        [-14.0, 10.0],
        [0.0, 0.0],
        [100.0, 100.0],
    ]
    allowed_actions = [True, True, True, False]

    # 先看到 signal-a 后按 Bayes 规则更新 belief，再做受约束的期望效用决策。
    belief = update_belief(prior, identity_transition, strong_signal[0])
    constrained = select_expected_utility_action(
        prior,
        utilities,
        allowed_actions=allowed_actions,
    )
    # 强信号能改变动作且可能值回成本；弱信号通常不足以改变最优动作。
    strong_information = value_of_information(
        prior,
        strong_signal,
        utilities,
        observation_cost=1.0,
        allowed_actions=allowed_actions,
    )
    weak_information = value_of_information(
        prior,
        weak_signal,
        utilities,
        observation_cost=1.0,
        allowed_actions=allowed_actions,
    )

    # 第一张图所有路径最终到终态；第二张图结构相同但把一个可达状态标成 forbidden。
    terminating_graph = [
        [False, True, True, False],
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, True],
    ]
    # cycling_graph 的状态 2 有自环，所以终态虽可达，却不保证每条路径都会终止。
    cycling_graph = [
        [False, True, True, False],
        [False, False, False, True],
        [False, False, True, True],
        [False, False, False, True],
    ]
    initial = [True, False, False, False]
    terminal = [False, False, False, True]

    return {
        "fixture": {
            "prior_belief": prior,
            "state_transition_during_observation": identity_transition,
            "strong_signal_likelihoods": strong_signal,
            "weak_signal_likelihoods": weak_signal,
            "action_utilities_by_state": utilities,
            "allowed_actions": allowed_actions,
            "observation_cost": 1.0,
            "transition_graph_state_labels": [
                "start",
                "branch_a",
                "branch_b",
                "terminal",
            ],
            "terminating_graph": terminating_graph,
            "cycling_graph": cycling_graph,
            "initial_state_mask": initial,
            "terminal_state_mask": terminal,
        },
        "state_labels": ["fault_a", "fault_b"],
        "action_labels": ["repair_a", "repair_b", "escalate", "forbidden_shortcut"],
        "belief_update_after_observation_a": {
            "prior": belief.prior_belief.tolist(),
            "observation_probability": belief.observation_probability,
            "posterior": belief.posterior_belief.tolist(),
        },
        "hard_constraint": {
            "expected_utilities": constrained.expected_utilities.tolist(),
            "allowed_actions": constrained.allowed_actions.tolist(),
            "best_action_index": constrained.best_action_index,
            "forbidden_action_had_highest_unconstrained_utility": True,
        },
        "strong_signal": _information_summary(strong_information),
        "weak_signal": _information_summary(weak_information),
        "transition_systems": {
            "safe_and_terminating": _transition_summary(
                analyze_transition_system(
                    terminating_graph,
                    initial,
                    terminal,
                    [False, False, False, False],
                )
            ),
            "reachable_forbidden_state": _transition_summary(
                analyze_transition_system(
                    terminating_graph,
                    initial,
                    terminal,
                    [False, False, True, False],
                )
            ),
            "terminal_reachable_but_cycle_possible": _transition_summary(
                analyze_transition_system(
                    cycling_graph,
                    initial,
                    terminal,
                    [False, False, False, False],
                )
            ),
        },
        "scope": {
            "device": "CPU",
            "finite_probabilities_enumerated_exactly": True,
            "language_model_or_tool_executed": False,
            "transition_or_observation_model_learned": False,
            "utility_values_empirically_validated": False,
            "open_world_agent_safety_or_task_success_proved": False,
        },
    }


def main() -> None:
    """打印决策论数值与状态图审计结果。"""

    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
