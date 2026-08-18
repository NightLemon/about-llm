"""Run finite controls for belief, information value, and agent termination."""

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
    prior = [0.6, 0.4]
    identity_transition = [[1.0, 0.0], [0.0, 1.0]]
    strong_signal = [[0.85, 0.15], [0.15, 0.85]]
    weak_signal = [[0.51, 0.49], [0.49, 0.51]]
    utilities = [
        [10.0, -14.0],
        [-14.0, 10.0],
        [0.0, 0.0],
        [100.0, 100.0],
    ]
    allowed_actions = [True, True, True, False]

    belief = update_belief(prior, identity_transition, strong_signal[0])
    constrained = select_expected_utility_action(
        prior,
        utilities,
        allowed_actions=allowed_actions,
    )
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

    terminating_graph = [
        [False, True, True, False],
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, True],
    ]
    cycling_graph = [
        [False, True, True, False],
        [False, False, False, True],
        [False, False, True, True],
        [False, False, False, True],
    ]
    initial = [True, False, False, False]
    terminal = [False, False, False, True]

    return {
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
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
