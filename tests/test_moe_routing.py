from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from about_llm.from_scratch import (
    MoERoutingResult,
    route_topk_capacity,
    routed_linear_expert_forward,
)

ROOT = Path(__file__).resolve().parents[1]


def _capacity_fixture() -> MoERoutingResult:
    logits = np.array(
        [[4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0], [0.0, 4.0, 3.0]]
    )
    return route_topk_capacity(logits, top_k=2, capacity_factor=0.75)


def test_exact_topk_capacity_drop_and_score_priority() -> None:
    routing = _capacity_fixture()

    np.testing.assert_array_equal(
        routing.selected_expert_indices,
        [[0, 1], [0, 1], [0, 1], [1, 2]],
    )
    np.testing.assert_array_equal(
        routing.kept_mask,
        [[True, False], [True, False], [False, True], [True, True]],
    )
    assert routing.expert_capacity == 2
    assert routing.expert_counts_before_capacity == (3, 4, 1)
    assert routing.expert_counts_after_capacity == (2, 2, 1)
    assert routing.assignments_before_capacity == 8
    assert routing.kept_assignments == 5
    assert routing.dropped_assignments == 3
    assert routing.dropped_assignment_fraction == pytest.approx(3 / 8)
    assert routing.tokens_with_all_assignments_dropped == 0


def test_combine_weights_renormalize_only_kept_assignments() -> None:
    routing = _capacity_fixture()

    np.testing.assert_allclose(routing.combine_weights[:3], [[1, 0], [1, 0], [0, 1]])
    np.testing.assert_allclose(
        routing.combine_weights[3],
        [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))],
    )
    np.testing.assert_allclose(routing.combine_weights.sum(axis=1), 1)


def test_ties_prefer_expert_then_token_and_can_drop_an_entire_token() -> None:
    routing = route_topk_capacity(
        np.zeros((3, 3)), top_k=2, capacity_factor=0.5
    )

    np.testing.assert_array_equal(routing.selected_expert_indices, [[0, 1]] * 3)
    np.testing.assert_array_equal(
        routing.kept_mask,
        [[True, True], [False, False], [False, False]],
    )
    assert routing.expert_capacity == 1
    assert routing.tokens_with_all_assignments_dropped == 2
    np.testing.assert_array_equal(routing.combine_weights[1:], 0)


def test_padding_mask_is_excluded_from_capacity_and_diagnostics() -> None:
    routing = route_topk_capacity(
        np.zeros((3, 2)),
        top_k=1,
        capacity_factor=1,
        token_mask=np.array([True, True, False]),
    )

    assert routing.active_token_count == 2
    assert routing.expert_capacity == 1
    np.testing.assert_array_equal(routing.selected_expert_indices[:, 0], [0, 0, -1])
    np.testing.assert_array_equal(routing.kept_mask[:, 0], [True, False, False])
    assert routing.assignments_before_capacity == 2
    assert routing.tokens_with_all_assignments_dropped == 1


def test_nonrenormalized_capacity_preserves_lost_topk_mass() -> None:
    routing = route_topk_capacity(
        np.array([[4.0, 1.0], [4.0, 2.0]]),
        top_k=2,
        capacity_factor=0.5,
        renormalize_after_capacity=False,
    )

    assert routing.renormalized_after_capacity is False
    assert routing.combine_weights[0].sum() > 0
    assert routing.combine_weights[0].sum() < 1
    assert routing.combine_weights[1].sum() > 0
    assert routing.combine_weights[1].sum() < 1


def test_load_balance_and_z_loss_follow_documented_formula() -> None:
    logits = np.zeros((2, 2))
    routing = route_topk_capacity(logits, top_k=2)

    assert routing.selection_fractions == (0.5, 0.5)
    assert routing.mean_router_probabilities == (0.5, 0.5)
    assert routing.load_balance_diagnostic == pytest.approx(1.0)
    assert routing.router_z_loss == pytest.approx(np.log(2) ** 2)
    assert routing.mean_router_entropy == pytest.approx(np.log(2))


def test_softmax_and_routing_are_stable_for_large_logits_and_row_shifts() -> None:
    logits = np.array([[1000.0, -1000.0, 0.0], [1000.0, 1000.0, 999.0]])
    shifted = logits + np.array([[123.0], [-456.0]])

    baseline = route_topk_capacity(logits, top_k=2)
    translated = route_topk_capacity(shifted, top_k=2)

    assert np.all(np.isfinite(baseline.probabilities))
    np.testing.assert_allclose(baseline.probabilities, translated.probabilities)
    np.testing.assert_array_equal(
        baseline.selected_expert_indices, translated.selected_expert_indices
    )
    np.testing.assert_array_equal(baseline.kept_mask, translated.kept_mask)
    # z-loss intentionally changes under a row-wise logit translation.
    assert baseline.router_z_loss != translated.router_z_loss


def test_randomized_capacity_and_weight_ledgers_hold() -> None:
    rng = np.random.default_rng(19)
    routing = route_topk_capacity(
        rng.normal(size=(11, 5)),
        top_k=3,
        capacity_factor=0.8,
        token_mask=np.array([True] * 9 + [False, False]),
    )

    assert routing.assignments_before_capacity == 27
    assert sum(routing.expert_counts_before_capacity) == 27
    assert sum(routing.expert_counts_after_capacity) == routing.kept_assignments
    assert max(routing.expert_counts_after_capacity) <= routing.expert_capacity
    kept_per_active_token = np.any(
        routing.kept_mask[routing.active_token_mask], axis=1
    )
    np.testing.assert_allclose(
        routing.combine_weights[routing.active_token_mask].sum(axis=1)[
            kept_per_active_token
        ],
        1,
    )
    np.testing.assert_array_equal(routing.combine_weights[~routing.active_token_mask], 0)


def test_sparse_linear_expert_dispatch_matches_manual_combination() -> None:
    routing = route_topk_capacity(np.array([[2.0, 1.0]]), top_k=2)
    hidden = np.array([[1.0, 2.0]])
    weights = np.stack([np.eye(2), 2 * np.eye(2)])

    output = routed_linear_expert_forward(hidden, weights, routing)
    expected = (
        routing.combine_weights[0, 0] * hidden
        + routing.combine_weights[0, 1] * (2 * hidden)
    )
    np.testing.assert_allclose(output, expected)


def test_all_dropped_token_has_zero_sparse_expert_output() -> None:
    routing = route_topk_capacity(
        np.zeros((2, 2)), top_k=1, capacity_factor=0.5
    )
    output = routed_linear_expert_forward(
        np.ones((2, 1)), np.ones((2, 1, 1)), routing
    )

    np.testing.assert_array_equal(output[:, 0], [1, 0])


def test_routing_arrays_are_immutable_and_input_changes_do_not_alias() -> None:
    logits = np.array([[1.0, 0.0]])
    routing = route_topk_capacity(logits, top_k=1)
    logits[0, 0] = -100

    assert routing.selected_expert_indices[0, 0] == 0
    assert not routing.probabilities.flags.writeable
    with pytest.raises(ValueError):
        routing.combine_weights[0, 0] = 0


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: route_topk_capacity(np.ones(2), top_k=1), "shape"),
        (lambda: route_topk_capacity(np.array([[np.nan, 0.0]]), top_k=1), "finite"),
        (lambda: route_topk_capacity(np.ones((1, 2)), top_k=0), "top_k"),
        (lambda: route_topk_capacity(np.ones((1, 2)), top_k=3), "top_k"),
        (
            lambda: route_topk_capacity(
                np.ones((1, 2)), top_k=1, capacity_factor=0
            ),
            "capacity_factor",
        ),
        (
            lambda: route_topk_capacity(
                np.ones((2, 2)), top_k=1, token_mask=np.array([False, False])
            ),
            "at least one",
        ),
        (
            lambda: route_topk_capacity(
                np.ones((2, 2)), top_k=1, token_mask=np.array([1, 0])
            ),
            "boolean vector",
        ),
    ],
)
def test_invalid_routing_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        operation()


@pytest.mark.parametrize(
    ("hidden", "weights", "message"),
    [
        (np.ones(2), np.ones((2, 2, 1)), "hidden_states"),
        (np.ones((1, 2)), np.ones((2, 3, 1)), "expert_weights"),
        (np.array([[np.inf, 0.0]]), np.ones((2, 2, 1)), "finite"),
    ],
)
def test_sparse_expert_forward_rejects_invalid_shapes(
    hidden: np.ndarray, weights: np.ndarray, message: str
) -> None:
    routing = route_topk_capacity(np.ones((1, 2)), top_k=1)
    with pytest.raises(ValueError, match=message):
        routed_linear_expert_forward(hidden, weights, routing)


