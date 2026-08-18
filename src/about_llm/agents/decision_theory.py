"""Finite exact controls for partially observable agent decisions."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BeliefUpdateReport:
    """One predict-observe Bayesian belief update."""

    prior_belief: NDArray[np.float64]
    predicted_belief: NDArray[np.float64]
    observation_likelihood: NDArray[np.float64]
    observation_probability: float
    posterior_belief: NDArray[np.float64]


@dataclass(frozen=True)
class ExpectedUtilityDecision:
    """Expected utility for each action after a hard allow-mask."""

    belief: NDArray[np.float64]
    action_utilities: NDArray[np.float64]
    expected_utilities: NDArray[np.float64]
    allowed_actions: NDArray[np.bool_]
    best_action_index: int
    best_expected_utility: float


@dataclass(frozen=True)
class ValueOfInformationReport:
    """Exact one-step expected value of a noisy observation."""

    prior_decision: ExpectedUtilityDecision
    observation_likelihoods: NDArray[np.float64]
    observation_probabilities: NDArray[np.float64]
    possible_observations: NDArray[np.bool_]
    posterior_beliefs: NDArray[np.float64]
    posterior_action_utilities: NDArray[np.float64]
    posterior_best_action_indices: NDArray[np.int64]
    posterior_best_utilities: NDArray[np.float64]
    expected_utility_with_information: float
    expected_value_of_sample_information: float
    observation_cost: float
    net_value_of_information: float
    information_is_worth_cost: bool


@dataclass(frozen=True)
class TransitionSystemReport:
    """Reachability, safety, and universal finite-path termination checks."""

    reachable_states: tuple[int, ...]
    reachable_terminal_states: tuple[int, ...]
    reachable_forbidden_states: tuple[int, ...]
    nonterminal_dead_end_states: tuple[int, ...]
    nonterminal_cycle_detected: bool
    safety_holds: bool
    terminal_is_reachable: bool
    guaranteed_termination: bool


def _finite_vector(values: ArrayLike, label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values, not booleans")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _finite_matrix(values: ArrayLike, label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values, not booleans")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{label} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _probability_vector(values: ArrayLike, label: str) -> NDArray[np.float64]:
    probabilities = _finite_vector(values, label)
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"{label} values must be probabilities in [0, 1]")
    if not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} probabilities must sum to 1")
    return probabilities


def _column_stochastic_matrix(values: ArrayLike, label: str) -> NDArray[np.float64]:
    probabilities = _finite_matrix(values, label)
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"{label} values must be probabilities in [0, 1]")
    if not np.allclose(
        np.sum(probabilities, axis=0),
        np.ones(probabilities.shape[1]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{label} columns must each sum to 1")
    return probabilities


def _boolean_vector(
    values: ArrayLike,
    *,
    size: int,
    label: str,
) -> NDArray[np.bool_]:
    raw = np.asarray(values)
    if raw.dtype.kind != "b":
        raise ValueError(f"{label} must contain boolean values")
    array = np.asarray(raw, dtype=np.bool_)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},)")
    return array


def _nonnegative_scalar(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def update_belief(
    prior_belief: ArrayLike,
    transition: ArrayLike,
    observation_likelihood: ArrayLike,
) -> BeliefUpdateReport:
    r"""Apply a POMDP-style prediction and Bayes observation update.

    ``transition[next_state, current_state]`` is ``P(s' | s, a)`` for one
    already-selected action. ``observation_likelihood[next_state]`` is
    ``P(o | s', a)`` for the observation that was actually received.
    """

    prior = _probability_vector(prior_belief, "prior_belief")
    transition_matrix = _column_stochastic_matrix(transition, "transition")
    if transition_matrix.shape != (prior.size, prior.size):
        raise ValueError("transition must have one current and next row per state")
    likelihood = _finite_vector(observation_likelihood, "observation_likelihood")
    if likelihood.shape != prior.shape:
        raise ValueError("observation_likelihood must contain one value per state")
    if np.any(likelihood < 0) or np.any(likelihood > 1):
        raise ValueError("observation_likelihood values must be probabilities in [0, 1]")

    predicted = np.asarray(transition_matrix @ prior, dtype=np.float64)
    observation_probability = float(likelihood @ predicted)
    if observation_probability <= 0:
        raise ValueError("received observation has zero probability under the model")
    posterior = np.asarray(
        likelihood * predicted / observation_probability,
        dtype=np.float64,
    )
    return BeliefUpdateReport(
        prior_belief=prior,
        predicted_belief=predicted,
        observation_likelihood=likelihood,
        observation_probability=observation_probability,
        posterior_belief=posterior,
    )


def select_expected_utility_action(
    belief: ArrayLike,
    action_utilities: ArrayLike,
    *,
    allowed_actions: ArrayLike | None = None,
) -> ExpectedUtilityDecision:
    """Choose the highest expected-utility action among hard-allowed actions."""

    belief_array = _probability_vector(belief, "belief")
    utilities = _finite_matrix(action_utilities, "action_utilities")
    if utilities.shape[1] != belief_array.size:
        raise ValueError("action_utilities must contain one column per state")
    allowed = (
        np.ones(utilities.shape[0], dtype=np.bool_)
        if allowed_actions is None
        else _boolean_vector(
            allowed_actions,
            size=utilities.shape[0],
            label="allowed_actions",
        )
    )
    if not np.any(allowed):
        raise ValueError("at least one action must be allowed")
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.asarray(utilities @ belief_array, dtype=np.float64)
    if not np.all(np.isfinite(expected)):
        raise ValueError("expected action utilities must remain finite")
    allowed_expected = np.where(allowed, expected, -np.inf)
    best_index = int(np.argmax(allowed_expected))
    return ExpectedUtilityDecision(
        belief=belief_array,
        action_utilities=utilities,
        expected_utilities=expected,
        allowed_actions=allowed,
        best_action_index=best_index,
        best_expected_utility=float(expected[best_index]),
    )


def value_of_information(
    prior_belief: ArrayLike,
    observation_likelihoods: ArrayLike,
    action_utilities: ArrayLike,
    *,
    observation_cost: float = 0.0,
    allowed_actions: ArrayLike | None = None,
) -> ValueOfInformationReport:
    r"""Compute one-step expected value of sample information exactly.

    ``observation_likelihoods[observation, state]`` is ``P(o | s)`` and each
    state column must sum to one. The observation is assumed not to change the
    hidden state. A diagnostic tool that changes the world needs a transition
    model and is not a pure value-of-information query.
    """

    prior_decision = select_expected_utility_action(
        prior_belief,
        action_utilities,
        allowed_actions=allowed_actions,
    )
    likelihoods = _column_stochastic_matrix(
        observation_likelihoods, "observation_likelihoods"
    )
    if likelihoods.shape[1] != prior_decision.belief.size:
        raise ValueError("observation_likelihoods must contain one column per state")
    cost = _nonnegative_scalar(observation_cost, "observation_cost")

    observation_probabilities = np.asarray(
        likelihoods @ prior_decision.belief,
        dtype=np.float64,
    )
    possible = observation_probabilities > 0
    observation_count = likelihoods.shape[0]
    state_count = likelihoods.shape[1]
    action_count = prior_decision.action_utilities.shape[0]
    posteriors = np.zeros((observation_count, state_count), dtype=np.float64)
    posterior_action_utilities = np.zeros(
        (observation_count, action_count), dtype=np.float64
    )
    best_indices = np.full(observation_count, -1, dtype=np.int64)
    best_utilities = np.zeros(observation_count, dtype=np.float64)

    for observation_index in range(observation_count):
        probability = float(observation_probabilities[observation_index])
        if probability <= 0:
            continue
        posterior = (
            likelihoods[observation_index]
            * prior_decision.belief
            / probability
        )
        decision = select_expected_utility_action(
            posterior,
            prior_decision.action_utilities,
            allowed_actions=prior_decision.allowed_actions,
        )
        posteriors[observation_index] = decision.belief
        posterior_action_utilities[observation_index] = decision.expected_utilities
        best_indices[observation_index] = decision.best_action_index
        best_utilities[observation_index] = decision.best_expected_utility

    with np.errstate(over="ignore", invalid="ignore"):
        expected_with_information = float(observation_probabilities @ best_utilities)
    if not math.isfinite(expected_with_information):
        raise ValueError("expected utility with information must remain finite")
    sample_information_value = (
        expected_with_information - prior_decision.best_expected_utility
    )
    if sample_information_value < -1e-12:
        raise ArithmeticError("sample information value violated non-negativity")
    if abs(sample_information_value) < 1e-12:
        sample_information_value = 0.0
    net_value = sample_information_value - cost
    if not math.isfinite(net_value):
        raise ValueError("net value of information must remain finite")
    return ValueOfInformationReport(
        prior_decision=prior_decision,
        observation_likelihoods=likelihoods,
        observation_probabilities=observation_probabilities,
        possible_observations=possible,
        posterior_beliefs=posteriors,
        posterior_action_utilities=posterior_action_utilities,
        posterior_best_action_indices=best_indices,
        posterior_best_utilities=best_utilities,
        expected_utility_with_information=expected_with_information,
        expected_value_of_sample_information=sample_information_value,
        observation_cost=cost,
        net_value_of_information=net_value,
        information_is_worth_cost=net_value > 0,
    )


def analyze_transition_system(
    adjacency: ArrayLike,
    initial_states: ArrayLike,
    terminal_states: ArrayLike,
    forbidden_states: ArrayLike,
) -> TransitionSystemReport:
    """Analyze all paths in a finite nondeterministic transition graph.

    Reaching a terminal ends the run even if the supplied graph has outgoing
    terminal edges. Guaranteed termination is false when a reachable
    nonterminal dead end or nonterminal cycle permits an infinite/non-goal path.
    No fairness assumption forces a branch to leave a cycle.
    """

    raw_adjacency = np.asarray(adjacency)
    if raw_adjacency.dtype.kind != "b":
        raise ValueError("adjacency must contain boolean values")
    graph = np.asarray(raw_adjacency, dtype=np.bool_)
    if graph.ndim != 2 or graph.shape[0] == 0 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be a non-empty square matrix")
    state_count = graph.shape[0]
    initial = _boolean_vector(initial_states, size=state_count, label="initial_states")
    terminal = _boolean_vector(
        terminal_states, size=state_count, label="terminal_states"
    )
    forbidden = _boolean_vector(
        forbidden_states, size=state_count, label="forbidden_states"
    )
    if not np.any(initial):
        raise ValueError("at least one initial state is required")

    reachable = np.zeros(state_count, dtype=np.bool_)
    pending: deque[int] = deque(int(index) for index in np.flatnonzero(initial))
    while pending:
        state = pending.popleft()
        if reachable[state]:
            continue
        reachable[state] = True
        if terminal[state]:
            continue
        for successor in np.flatnonzero(graph[state]):
            successor_index = int(successor)
            if not reachable[successor_index]:
                pending.append(successor_index)

    nonterminal = reachable & ~terminal
    dead_ends = nonterminal & ~np.any(graph, axis=1)
    nonterminal_indices = [int(index) for index in np.flatnonzero(nonterminal)]
    indegree = {state: 0 for state in nonterminal_indices}
    for state in nonterminal_indices:
        for successor in np.flatnonzero(graph[state] & nonterminal):
            indegree[int(successor)] += 1
    zero_indegree: deque[int] = deque(
        state for state, degree in indegree.items() if degree == 0
    )
    removed = 0
    while zero_indegree:
        state = zero_indegree.popleft()
        removed += 1
        for successor in np.flatnonzero(graph[state] & nonterminal):
            successor_index = int(successor)
            indegree[successor_index] -= 1
            if indegree[successor_index] == 0:
                zero_indegree.append(successor_index)
    cycle_detected = removed != len(nonterminal_indices)

    reachable_terminals = tuple(int(index) for index in np.flatnonzero(reachable & terminal))
    reachable_forbidden = tuple(
        int(index) for index in np.flatnonzero(reachable & forbidden)
    )
    dead_end_states = tuple(int(index) for index in np.flatnonzero(dead_ends))
    terminal_is_reachable = bool(reachable_terminals)
    return TransitionSystemReport(
        reachable_states=tuple(int(index) for index in np.flatnonzero(reachable)),
        reachable_terminal_states=reachable_terminals,
        reachable_forbidden_states=reachable_forbidden,
        nonterminal_dead_end_states=dead_end_states,
        nonterminal_cycle_detected=cycle_detected,
        safety_holds=not reachable_forbidden,
        terminal_is_reachable=terminal_is_reachable,
        guaranteed_termination=(
            terminal_is_reachable and not dead_end_states and not cycle_detected
        ),
    )
