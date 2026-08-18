from __future__ import annotations

import numpy as np
import pytest

from about_llm.agents import (
    analyze_transition_system,
    select_expected_utility_action,
    update_belief,
    value_of_information,
)

pytestmark = [pytest.mark.formula, pytest.mark.contract]


def test_belief_update_matches_binary_bayes_rule() -> None:
    report = update_belief(
        [0.6, 0.4],
        [[1.0, 0.0], [0.0, 1.0]],
        [0.85, 0.15],
    )

    assert report.observation_probability == pytest.approx(0.57)
    np.testing.assert_allclose(report.predicted_belief, [0.6, 0.4])
    np.testing.assert_allclose(
        report.posterior_belief,
        [0.51 / 0.57, 0.06 / 0.57],
    )
    assert np.sum(report.posterior_belief) == pytest.approx(1.0)


def test_belief_update_applies_transition_before_observation() -> None:
    report = update_belief(
        [0.75, 0.25],
        [[0.8, 0.2], [0.2, 0.8]],
        [0.1, 0.9],
    )

    np.testing.assert_allclose(report.predicted_belief, [0.65, 0.35])
    assert report.observation_probability == pytest.approx(0.38)
    np.testing.assert_allclose(report.posterior_belief, [0.065 / 0.38, 0.315 / 0.38])


def test_expected_utility_never_overrides_hard_action_constraint() -> None:
    decision = select_expected_utility_action(
        [0.6, 0.4],
        [
            [10.0, -14.0],
            [-14.0, 10.0],
            [0.0, 0.0],
            [100.0, 100.0],
        ],
        allowed_actions=[True, True, True, False],
    )

    np.testing.assert_allclose(decision.expected_utilities, [0.4, -4.4, 0.0, 100.0])
    assert decision.best_action_index == 0
    assert decision.best_expected_utility == pytest.approx(0.4)
    assert not decision.allowed_actions[3]


def test_value_of_information_accounts_for_signal_and_cost() -> None:
    report = value_of_information(
        [0.6, 0.4],
        [[0.85, 0.15], [0.15, 0.85]],
        [[10.0, -14.0], [-14.0, 10.0], [0.0, 0.0]],
        observation_cost=1.0,
    )

    assert report.prior_decision.best_action_index == 0
    assert report.prior_decision.best_expected_utility == pytest.approx(0.4)
    np.testing.assert_allclose(report.observation_probabilities, [0.57, 0.43])
    np.testing.assert_array_equal(report.posterior_best_action_indices, [0, 1])
    assert report.expected_utility_with_information == pytest.approx(6.4)
    assert report.expected_value_of_sample_information == pytest.approx(6.0)
    assert report.net_value_of_information == pytest.approx(5.0)
    assert report.information_is_worth_cost


def test_weak_signal_has_no_decision_value_and_is_not_worth_cost() -> None:
    report = value_of_information(
        [0.6, 0.4],
        [[0.51, 0.49], [0.49, 0.51]],
        [[10.0, -14.0], [-14.0, 10.0], [0.0, 0.0]],
        observation_cost=1.0,
    )

    np.testing.assert_array_equal(report.posterior_best_action_indices, [0, 0])
    assert report.expected_value_of_sample_information == pytest.approx(0.0)
    assert report.net_value_of_information == pytest.approx(-1.0)
    assert not report.information_is_worth_cost


def test_zero_probability_observation_is_explicit_and_does_not_add_utility() -> None:
    report = value_of_information(
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, -2.0], [-2.0, 2.0]],
    )

    np.testing.assert_array_equal(report.possible_observations, [True, False])
    np.testing.assert_array_equal(report.posterior_best_action_indices, [0, -1])
    np.testing.assert_allclose(report.posterior_beliefs[1], [0.0, 0.0])
    assert report.expected_value_of_sample_information == pytest.approx(0.0)


def test_transition_system_distinguishes_reachability_safety_and_termination() -> None:
    safe_terminating = analyze_transition_system(
        [
            [False, True, True, False],
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, True],
        ],
        [True, False, False, False],
        [False, False, False, True],
        [False, False, True, False],
    )
    cycle = analyze_transition_system(
        [
            [False, True, True, False],
            [False, False, False, True],
            [False, False, True, True],
            [False, False, False, True],
        ],
        [True, False, False, False],
        [False, False, False, True],
        [False, False, False, False],
    )

    assert safe_terminating.terminal_is_reachable
    assert safe_terminating.guaranteed_termination
    assert not safe_terminating.safety_holds
    assert safe_terminating.reachable_forbidden_states == (2,)
    assert cycle.terminal_is_reachable
    assert cycle.nonterminal_cycle_detected
    assert not cycle.guaranteed_termination
    assert cycle.safety_holds


def test_nonterminal_dead_end_prevents_guaranteed_termination() -> None:
    report = analyze_transition_system(
        [[False, True, True], [False, False, False], [False, False, True]],
        [True, False, False],
        [False, False, True],
        [False, False, False],
    )

    assert report.terminal_is_reachable
    assert report.nonterminal_dead_end_states == (1,)
    assert not report.guaranteed_termination


def test_decision_theory_contracts_reject_invalid_models() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        update_belief([0.6, 0.6], [[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5])
    with pytest.raises(ValueError, match="columns must each sum to 1"):
        update_belief([0.5, 0.5], [[1.0, 0.0], [0.0, 0.5]], [0.5, 0.5])
    with pytest.raises(ValueError, match="zero probability"):
        update_belief([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    with pytest.raises(ValueError, match="at least one action"):
        select_expected_utility_action(
            [0.5, 0.5], [[1.0, 0.0]], allowed_actions=[False]
        )
    with pytest.raises(ValueError, match="non-negative"):
        value_of_information(
            [0.5, 0.5],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0]],
            observation_cost=-1.0,
        )
    with pytest.raises(ValueError, match="boolean"):
        analyze_transition_system([[0, 1], [0, 0]], [True, False], [False, True], [False, False])
