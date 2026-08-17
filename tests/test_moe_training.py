from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import torch

from about_llm.from_scratch import TrainableTopKMoE as ExportedTrainableTopKMoE
from about_llm.from_scratch.moe_training import (
    TRAINABLE_MOE_CONTROL_VERSION,
    TrainableTopKMoE,
    _authored_model,
    _fixture,
    run_trainable_moe_control,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "moe_training_control.py"


def test_trainable_moe_is_available_through_lazy_public_export() -> None:
    assert ExportedTrainableTopKMoE is TrainableTopKMoE


def _reject_nonfinite(value: str) -> NoReturn:
    raise AssertionError(f"non-standard JSON number: {value}")


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _materialized_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in model.named_parameters()
    }


def test_stable_topk_tie_break_prefers_lower_expert_id() -> None:
    model = TrainableTopKMoE(2, 3, 1, 3, 2, dtype=torch.float64)
    with torch.no_grad():
        model.router.weight.zero_()

    forward = model(torch.tensor([[1.0, -1.0]], dtype=torch.float64))

    assert forward.selected_expert_indices.tolist() == [[0, 1]]
    torch.testing.assert_close(
        forward.combine_weights,
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    assert forward.expert_capacity is None
    assert forward.expert_capacities_by_group is None
    assert forward.routing_group_labels == (0,)
    assert forward.active_tokens_per_group == (1,)
    assert forward.active_token_mask.tolist() == [True]
    assert forward.renormalized_after_capacity is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 2, 1, 2, 1), "d_model"),
        ((2, 0, 1, 2, 1), "hidden_dim"),
        ((2, 2, 0, 2, 1), "output_dim"),
        ((2, 2, 1, 0, 1), "expert_count"),
        ((2, 2, 1, 2, 0), "top_k"),
        ((2, 2, 1, 2, 3), "cannot exceed"),
    ],
)
def test_constructor_rejects_invalid_dimensions(
    arguments: tuple[int, int, int, int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainableTopKMoE(*arguments)


def test_constructor_rejects_nonfloating_dtype() -> None:
    with pytest.raises(ValueError, match="floating point"):
        TrainableTopKMoE(2, 2, 1, 2, 1, dtype=torch.int64)


@pytest.mark.parametrize(
    ("hidden", "kwargs", "exception", "message"),
    [
        (torch.ones(3, dtype=torch.float64), {}, ValueError, "shape"),
        (torch.ones((0, 3), dtype=torch.float64), {}, ValueError, "at least one"),
        (torch.ones((2, 3), dtype=torch.float32), {}, ValueError, "dtype"),
        (
            torch.tensor([[float("inf"), 0.0, 0.0]], dtype=torch.float64),
            {},
            ValueError,
            "finite",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"detach_combine_weights": 1},
            TypeError,
            "boolean",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"dense_oracle": 1},
            TypeError,
            "boolean",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"capacity_factor": 0},
            ValueError,
            "capacity_factor",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"capacity_factor": float("nan")},
            ValueError,
            "capacity_factor",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"capacity_factor": 10**400},
            ValueError,
            "capacity_factor",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"renormalize_after_capacity": 1},
            TypeError,
            "boolean",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"overflow_policy": 1},
            TypeError,
            "must be a string",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"overflow_policy": "spill"},
            ValueError,
            "drop, reroute, or dropless",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"overflow_policy": "reroute"},
            ValueError,
            "require capacity_factor",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"overflow_policy": "dropless"},
            ValueError,
            "require capacity_factor",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"token_mask": [True, False]},
            TypeError,
            "token_mask",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"token_mask": torch.ones(2, dtype=torch.float64)},
            ValueError,
            "boolean shape",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"token_mask": torch.zeros(2, dtype=torch.bool)},
            ValueError,
            "at least one active",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"routing_group_ids": [0, 1]},
            TypeError,
            "routing_group_ids",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"routing_group_ids": torch.tensor([0, 1], dtype=torch.int32)},
            ValueError,
            "int64 shape",
        ),
        (
            torch.ones((2, 3), dtype=torch.float64),
            {"routing_group_ids": torch.tensor([[0], [1]], dtype=torch.int64)},
            ValueError,
            "int64 shape",
        ),
    ],
)
def test_forward_contract_fails_closed(
    hidden: torch.Tensor,
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    model = _authored_model()
    with pytest.raises(exception, match=message):
        model(hidden, **kwargs)


@pytest.mark.parametrize("capacity_factor", [None, 0.5])
def test_sparse_dispatch_matches_dense_oracle_forward_and_backward(
    capacity_factor: float | None,
) -> None:
    hidden, target = _fixture()
    sparse_model = _authored_model()
    dense_model = copy.deepcopy(sparse_model)
    sparse = sparse_model(hidden, capacity_factor=capacity_factor)
    dense = dense_model(
        hidden,
        capacity_factor=capacity_factor,
        dense_oracle=True,
    )
    sparse_loss = (
        torch.nn.functional.mse_loss(sparse.output, target)
        + 0.05 * sparse.load_balance_loss
        + 0.001 * sparse.router_z_loss
    )
    dense_loss = (
        torch.nn.functional.mse_loss(dense.output, target)
        + 0.05 * dense.load_balance_loss
        + 0.001 * dense.router_z_loss
    )
    sparse_loss.backward()
    dense_loss.backward()

    torch.testing.assert_close(sparse.output, dense.output, rtol=0, atol=0)
    assert torch.equal(
        sparse.selected_expert_indices,
        dense.selected_expert_indices,
    )
    for (sparse_name, sparse_parameter), (dense_name, dense_parameter) in zip(
        sparse_model.named_parameters(),
        dense_model.named_parameters(),
        strict=True,
    ):
        assert sparse_name == dense_name
        assert sparse_parameter.grad is not None
        assert dense_parameter.grad is not None
        torch.testing.assert_close(
            sparse_parameter.grad,
            dense_parameter.grad,
            rtol=0,
            atol=1e-17,
        )


def test_capacity_score_priority_counts_and_post_drop_renormalization() -> None:
    hidden, _ = _fixture()
    forward = _authored_model()(hidden, capacity_factor=0.5)

    assert forward.expert_capacity == 2
    assert forward.expert_capacities_by_group == (2,)
    assert forward.routing_group_labels == (0,)
    assert forward.active_tokens_per_group == (5,)
    assert forward.renormalized_after_capacity is True
    assert forward.expert_counts_before_capacity.tolist() == [4, 3, 3]
    assert forward.expert_counts_after_capacity.tolist() == [2, 2, 2]
    assert forward.expert_counts_before_capacity_by_group.tolist() == [[4, 3, 3]]
    assert forward.expert_counts_after_capacity_by_group.tolist() == [[2, 2, 2]]
    assert forward.kept_mask.tolist() == [
        [True, False],
        [True, False],
        [True, True],
        [True, False],
        [True, False],
    ]
    assert forward.dropped_assignments == 4
    assert forward.tokens_with_all_assignments_dropped == 0
    torch.testing.assert_close(
        forward.combine_weights.sum(dim=1),
        torch.ones(5, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_preserving_dropped_mass_is_distinct_from_post_drop_renormalization() -> None:
    hidden, _ = _fixture()
    renormalized = _authored_model()(hidden, capacity_factor=0.5)
    preserved = _authored_model()(
        hidden,
        capacity_factor=0.5,
        renormalize_after_capacity=False,
    )

    assert torch.equal(renormalized.kept_mask, preserved.kept_mask)
    assert preserved.renormalized_after_capacity is False
    torch.testing.assert_close(
        preserved.combine_weights.sum(dim=1),
        torch.tensor(
            [
                0.8053384164084222,
                0.6681877721681662,
                1.0,
                0.638763175148842,
                0.5498339973124778,
            ],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=1e-15,
    )
    assert torch.max(torch.abs(renormalized.output - preserved.output)).item() == pytest.approx(
        0.1255417263895207
    )


def test_overflow_policies_pin_dispatch_drop_and_capacity_excess() -> None:
    hidden = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float64)

    dropped = _authored_model(top_k=1)(
        hidden,
        capacity_factor=1.0,
        overflow_policy="drop",
    )
    rerouted = _authored_model(top_k=1)(
        hidden,
        capacity_factor=1.0,
        overflow_policy="reroute",
    )
    dropless = _authored_model(top_k=1)(
        hidden,
        capacity_factor=1.0,
        overflow_policy="dropless",
    )

    expected_ranking = [[0, 2, 1]] * 4
    expected_selected = [[0]] * 4
    assert dropped.ranked_expert_indices.tolist() == expected_ranking
    assert dropped.selected_expert_indices.tolist() == expected_selected
    assert dropped.expert_capacity == 2
    assert dropped.overflow_policy == "drop"
    assert dropped.dispatched_expert_indices.tolist() == expected_selected
    assert dropped.expert_counts_before_capacity.tolist() == [4, 0, 0]
    assert dropped.expert_counts_after_capacity.tolist() == [2, 0, 0]
    assert dropped.pre_policy_capacity_excess_by_group.tolist() == [[2, 0, 0]]
    assert dropped.post_policy_capacity_excess_by_group.tolist() == [[0, 0, 0]]
    assert dropped.assignments_over_capacity_before_policy == 2
    assert dropped.assignments_over_capacity_after_policy == 0
    assert dropped.rerouted_assignments == 0
    assert dropped.dropped_assignments == 2

    assert rerouted.overflow_policy == "reroute"
    assert rerouted.dispatched_expert_indices.tolist() == [[0], [0], [2], [2]]
    assert rerouted.expert_counts_before_capacity.tolist() == [4, 0, 0]
    assert rerouted.expert_counts_after_capacity.tolist() == [2, 0, 2]
    assert rerouted.pre_policy_capacity_excess_by_group.tolist() == [[2, 0, 0]]
    assert rerouted.post_policy_capacity_excess_by_group.tolist() == [[0, 0, 0]]
    assert rerouted.assignments_over_capacity_before_policy == 2
    assert rerouted.assignments_over_capacity_after_policy == 0
    assert rerouted.rerouted_assignments == 2
    assert rerouted.dropped_assignments == 0
    torch.testing.assert_close(
        rerouted.dispatched_probabilities[2:],
        torch.full((2, 1), 0.2649461021163392, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )

    assert dropless.overflow_policy == "dropless"
    assert dropless.dispatched_expert_indices.tolist() == expected_selected
    assert dropless.expert_counts_before_capacity.tolist() == [4, 0, 0]
    assert dropless.expert_counts_after_capacity.tolist() == [4, 0, 0]
    assert dropless.pre_policy_capacity_excess_by_group.tolist() == [[2, 0, 0]]
    assert dropless.post_policy_capacity_excess_by_group.tolist() == [[2, 0, 0]]
    assert dropless.assignments_over_capacity_before_policy == 2
    assert dropless.assignments_over_capacity_after_policy == 2
    assert dropless.rerouted_assignments == 0
    assert dropless.dropped_assignments == 0

    assert torch.max(torch.abs(dropped.output - rerouted.output)).item() == pytest.approx(
        0.11622178688336826
    )
    assert torch.max(torch.abs(rerouted.output - dropless.output)).item() == pytest.approx(
        0.10698215447093767
    )


def test_reroute_scans_full_ranking_without_duplicate_experts_per_token() -> None:
    hidden = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)

    forward = _authored_model(top_k=2)(
        hidden,
        capacity_factor=1.0,
        overflow_policy="reroute",
    )

    assert forward.ranked_expert_indices.tolist() == [[0, 2, 1]] * 3
    assert forward.selected_expert_indices.tolist() == [[0, 2]] * 3
    assert forward.dispatched_expert_indices.tolist() == [
        [0, 2],
        [0, 2],
        [1, 2],
    ]
    assert forward.kept_mask.tolist() == [
        [True, True],
        [True, True],
        [True, False],
    ]
    assert forward.expert_counts_before_capacity.tolist() == [3, 0, 3]
    assert forward.expert_counts_after_capacity.tolist() == [2, 1, 2]
    assert forward.rerouted_assignments == 1
    assert forward.dropped_assignments == 1
    assert forward.assignments_over_capacity_after_policy == 0
    for indices, kept in zip(
        forward.dispatched_expert_indices.tolist(),
        forward.kept_mask.tolist(),
        strict=True,
    ):
        actual_experts = [index for index, is_kept in zip(indices, kept, strict=True) if is_kept]
        assert len(actual_experts) == len(set(actual_experts))


def test_reroute_capacity_is_scoped_to_each_routing_group() -> None:
    hidden = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float64)
    group_ids = torch.tensor([10, 10, 20, 20], dtype=torch.int64)

    forward = _authored_model(top_k=1)(
        hidden,
        capacity_factor=1.0,
        routing_group_ids=group_ids,
        overflow_policy="reroute",
    )

    assert forward.routing_group_labels == (10, 20)
    assert forward.active_tokens_per_group == (2, 2)
    assert forward.expert_capacities_by_group == (1, 1)
    assert forward.expert_counts_before_capacity_by_group.tolist() == [
        [2, 0, 0],
        [2, 0, 0],
    ]
    assert forward.expert_counts_after_capacity_by_group.tolist() == [
        [1, 0, 1],
        [1, 0, 1],
    ]
    assert forward.dispatched_expert_indices.tolist() == [[0], [2], [0], [2]]
    assert forward.post_policy_capacity_excess_by_group.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert forward.rerouted_assignments == 2
    assert forward.dropped_assignments == 0


def test_reroute_can_preserve_selected_mass_or_renormalize_after_dispatch() -> None:
    hidden = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float64)
    renormalized = _authored_model(top_k=1)(
        hidden,
        capacity_factor=1.0,
        overflow_policy="reroute",
    )
    preserved = _authored_model(top_k=1)(
        hidden,
        capacity_factor=1.0,
        overflow_policy="reroute",
        renormalize_after_capacity=False,
    )

    assert torch.equal(
        renormalized.dispatched_expert_indices,
        preserved.dispatched_expert_indices,
    )
    torch.testing.assert_close(
        renormalized.combine_weights.sum(dim=1),
        torch.ones(4, dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        preserved.combine_weights.sum(dim=1),
        torch.tensor(
            [1.0, 1.0, 0.44932896411722156, 0.44932896411722156],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=1e-15,
    )
    assert torch.max(
        torch.abs(renormalized.output - preserved.output)
    ).item() == pytest.approx(0.06399997177521191)


@pytest.mark.parametrize("overflow_policy", ["reroute", "dropless"])
def test_overflow_policy_sparse_dense_forward_and_backward_match_with_zero_fill(
    overflow_policy: str,
) -> None:
    hidden = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float64)
    target = _fixture()[1][:4]
    sparse_model = _authored_model(top_k=1)
    dense_model = copy.deepcopy(sparse_model)
    sparse = sparse_model(
        hidden,
        capacity_factor=1.0,
        overflow_policy=overflow_policy,
    )
    dense = dense_model(
        hidden,
        capacity_factor=1.0,
        overflow_policy=overflow_policy,
        dense_oracle=True,
    )
    sparse_loss = (
        torch.nn.functional.mse_loss(sparse.output, target)
        + 0.05 * sparse.load_balance_loss
        + 0.001 * sparse.router_z_loss
    )
    dense_loss = (
        torch.nn.functional.mse_loss(dense.output, target)
        + 0.05 * dense.load_balance_loss
        + 0.001 * dense.router_z_loss
    )
    sparse_loss.backward()
    dense_loss.backward()

    torch.testing.assert_close(sparse.output, dense.output, rtol=0, atol=0)
    assert torch.equal(
        sparse.dispatched_expert_indices,
        dense.dispatched_expert_indices,
    )
    sparse_gradients = _materialized_gradients(sparse_model)
    dense_gradients = _materialized_gradients(dense_model)
    assert sparse_gradients.keys() == dense_gradients.keys()
    for name in sparse_gradients:
        torch.testing.assert_close(
            sparse_gradients[name],
            dense_gradients[name],
            rtol=0,
            atol=0,
        )
    sparse_missing = [
        name
        for name, parameter in sparse_model.named_parameters()
        if parameter.grad is None
    ]
    assert sparse_missing
    dense_parameters = dict(dense_model.named_parameters())
    for name in sparse_missing:
        dense_gradient = dense_parameters[name].grad
        assert dense_gradient is not None
        assert torch.count_nonzero(dense_gradient) == 0


def test_all_dropped_tokens_have_zero_routed_expert_output() -> None:
    model = _authored_model()
    with torch.no_grad():
        model.router.weight.zero_()
    forward = model(torch.eye(3, dtype=torch.float64), capacity_factor=0.5)

    assert forward.expert_capacity == 1
    assert forward.selected_expert_indices.tolist() == [[0, 1], [0, 1], [0, 1]]
    assert forward.kept_mask.tolist() == [
        [True, True],
        [False, False],
        [False, False],
    ]
    assert forward.expert_counts_before_capacity.tolist() == [3, 3, 0]
    assert forward.expert_counts_after_capacity.tolist() == [1, 1, 0]
    assert forward.tokens_with_all_assignments_dropped == 2
    torch.testing.assert_close(
        forward.output[1:],
        torch.zeros((2, 2), dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_padding_and_routing_groups_scope_capacity_competition() -> None:
    hidden, _ = _fixture()
    token_mask = torch.tensor([True, True, True, True, False])
    group_ids = torch.tensor([10, 10, 20, 20, 999], dtype=torch.int64)
    grouped = _authored_model()(
        hidden,
        capacity_factor=0.5,
        token_mask=token_mask,
        routing_group_ids=group_ids,
    )
    single_group = _authored_model()(
        hidden,
        capacity_factor=0.5,
        token_mask=token_mask,
    )

    assert grouped.active_token_mask.tolist() == [True, True, True, True, False]
    assert grouped.routing_group_ids.tolist() == [10, 10, 20, 20, 999]
    assert grouped.routing_group_labels == (10, 20)
    assert grouped.active_tokens_per_group == (2, 2)
    assert grouped.expert_capacity is None
    assert grouped.expert_capacities_by_group == (1, 1)
    assert grouped.expert_counts_before_capacity.tolist() == [3, 2, 3]
    assert grouped.expert_counts_after_capacity.tolist() == [2, 2, 2]
    assert grouped.expert_counts_before_capacity_by_group.tolist() == [
        [2, 1, 1],
        [1, 1, 2],
    ]
    assert grouped.expert_counts_after_capacity_by_group.tolist() == [
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert grouped.kept_mask.tolist() == [
        [True, True],
        [True, False],
        [False, True],
        [True, True],
        [False, False],
    ]
    assert grouped.dropped_assignments == 2
    assert grouped.tokens_with_all_assignments_dropped == 0
    assert grouped.selection_fractions_by_group.tolist() == [
        [0.5, 0.25, 0.25],
        [0.25, 0.25, 0.5],
    ]
    assert single_group.expert_capacity == 2
    assert single_group.expert_capacities_by_group == (2,)
    assert single_group.kept_mask.tolist() == [
        [True, False],
        [True, False],
        [True, True],
        [True, True],
        [False, False],
    ]
    assert torch.max(torch.abs(grouped.output - single_group.output)).item() == pytest.approx(
        0.3293871976258794
    )


def test_grouped_padding_sparse_dense_forward_and_backward_match() -> None:
    hidden, target = _fixture()
    token_mask = torch.tensor([True, True, True, True, False])
    group_ids = torch.tensor([10, 10, 20, 20, 999], dtype=torch.int64)
    sparse_model = _authored_model()
    dense_model = copy.deepcopy(sparse_model)
    sparse = sparse_model(
        hidden,
        capacity_factor=0.5,
        token_mask=token_mask,
        routing_group_ids=group_ids,
    )
    dense = dense_model(
        hidden,
        capacity_factor=0.5,
        token_mask=token_mask,
        routing_group_ids=group_ids,
        dense_oracle=True,
    )
    sparse_loss = (
        torch.nn.functional.mse_loss(sparse.output[token_mask], target[token_mask])
        + 0.05 * sparse.load_balance_loss
        + 0.001 * sparse.router_z_loss
    )
    dense_loss = (
        torch.nn.functional.mse_loss(dense.output[token_mask], target[token_mask])
        + 0.05 * dense.load_balance_loss
        + 0.001 * dense.router_z_loss
    )
    sparse_loss.backward()
    dense_loss.backward()

    torch.testing.assert_close(sparse.output, dense.output, rtol=0, atol=0)
    torch.testing.assert_close(
        sparse.load_balance_loss_by_group,
        dense.load_balance_loss_by_group,
        rtol=0,
        atol=0,
    )
    for sparse_parameter, dense_parameter in zip(
        sparse_model.parameters(),
        dense_model.parameters(),
        strict=True,
    ):
        assert sparse_parameter.grad is not None
        assert dense_parameter.grad is not None
        torch.testing.assert_close(
            sparse_parameter.grad,
            dense_parameter.grad,
            rtol=0,
            atol=0,
        )


def test_padding_values_and_group_ids_do_not_affect_active_path_or_gradient() -> None:
    hidden, _ = _fixture()
    token_mask = torch.tensor([True, True, True, True, False])
    group_ids = torch.tensor([10, 10, 20, 20, 999], dtype=torch.int64)
    baseline = _authored_model()(
        hidden,
        capacity_factor=0.5,
        token_mask=token_mask,
        routing_group_ids=group_ids,
    )
    mutated_hidden = hidden.clone()
    mutated_hidden[-1] = torch.tensor([100.0, -100.0, 50.0], dtype=torch.float64)
    mutated_group_ids = group_ids.clone()
    mutated_group_ids[-1] = -999
    mutated = _authored_model()(
        mutated_hidden,
        capacity_factor=0.5,
        token_mask=token_mask,
        routing_group_ids=mutated_group_ids,
    )

    assert baseline.routing_group_labels == mutated.routing_group_labels == (10, 20)
    torch.testing.assert_close(
        baseline.output[token_mask],
        mutated.output[token_mask],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline.output[~token_mask],
        torch.zeros((1, 2), dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline.load_balance_loss,
        mutated.load_balance_loss,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        baseline.router_z_loss,
        mutated.router_z_loss,
        rtol=0,
        atol=0,
    )

    probe_hidden = hidden.clone().requires_grad_(True)
    probe = _authored_model()(
        probe_hidden,
        capacity_factor=0.5,
        token_mask=token_mask,
        routing_group_ids=group_ids,
    )
    (
        probe.output.sum()
        + 0.05 * probe.load_balance_loss
        + 0.001 * probe.router_z_loss
    ).backward()
    assert probe_hidden.grad is not None
    torch.testing.assert_close(
        probe_hidden.grad[~token_mask],
        torch.zeros((1, 3), dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_detaching_selected_gate_blocks_only_main_task_router_gradient() -> None:
    hidden, target = _fixture()
    model = _authored_model()
    forward = model(hidden, detach_combine_weights=True)

    torch.autograd.backward(torch.nn.functional.mse_loss(forward.output, target))

    assert model.router.weight.grad is None
    for expert in model.experts:
        assert all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.any(parameter.grad != 0))
            for parameter in expert.parameters()
        )


def test_control_report_pins_gradient_semantics_and_scope() -> None:
    report = run_trainable_moe_control()

    assert report["schema_version"] == TRAINABLE_MOE_CONTROL_VERSION
    assert report["fixture"] == {
        "assignment_counts": [4, 3, 3],
        "balance_formula": (
            "per group E * sum(stop_gradient(f_e) * mean_probability_e), "
            "active-token-weighted across groups"
        ),
        "selected_expert_indices": [[0, 2], [1, 0], [2, 1], [2, 0], [0, 1]],
        "task_loss": "mean squared error over 2D authored targets",
        "token_count": 5,
        "total_loss": "task + 0.05 * balance + 0.001 * z",
        "z_loss_formula": (
            "per group mean(square(logsumexp(router_logits))), "
            "active-token-weighted across groups"
        ),
    }
    oracle = report["sparse_dense_oracle"]
    assert isinstance(oracle, dict)
    assert oracle["output_max_abs_difference"] == 0.0
    assert oracle["all_parameter_gradient_max_abs_difference"] == pytest.approx(
        6.938893903907228e-18
    )
    capacity = report["capacity_training_control"]
    assert isinstance(capacity, dict)
    assert capacity["expert_capacity"] == 2
    assert capacity["expert_counts_before_capacity"] == [4, 3, 3]
    assert capacity["expert_counts_after_capacity"] == [2, 2, 2]
    assert capacity["dropped_assignments"] == 4
    assert capacity["tokens_with_all_assignments_dropped"] == 0
    assert capacity["sparse_dense_output_max_abs_difference"] == 0.0
    assert capacity[
        "sparse_dense_all_parameter_gradient_max_abs_difference"
    ] == 0.0
    assert capacity["renormalize_vs_preserve_output_max_abs_difference"] == pytest.approx(
        0.1255417263895207
    )
    all_drop = capacity["all_drop_fixture"]
    assert isinstance(all_drop, dict)
    assert all_drop["tokens_with_all_assignments_dropped"] == 2
    assert all_drop["routed_output"][1:] == [[0.0, 0.0], [0.0, 0.0]]
    grouped = report["routing_group_padding_control"]
    assert isinstance(grouped, dict)
    assert grouped["token_mask"] == [True, True, True, True, False]
    assert grouped["routing_group_ids"] == [10, 10, 20, 20, 999]
    assert grouped["routing_group_labels"] == [10, 20]
    assert grouped["active_tokens_per_group"] == [2, 2]
    assert grouped["expert_capacity"] is None
    assert grouped["expert_capacities_by_group"] == [1, 1]
    assert grouped["expert_counts_before_capacity"] == [3, 2, 3]
    assert grouped["expert_counts_after_capacity"] == [2, 2, 2]
    assert grouped["expert_counts_before_capacity_by_group"] == [
        [2, 1, 1],
        [1, 1, 2],
    ]
    assert grouped["expert_counts_after_capacity_by_group"] == [
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert grouped["kept_mask"] == [
        [True, True],
        [True, False],
        [False, True],
        [True, True],
        [False, False],
    ]
    assert grouped["dropped_assignments"] == 2
    assert grouped["sparse_dense_output_max_abs_difference"] == 0.0
    assert grouped[
        "sparse_dense_all_parameter_gradient_max_abs_difference"
    ] == 0.0
    assert grouped["grouped_vs_single_group_output_max_abs_difference"] == pytest.approx(
        0.3293871976258794
    )
    assert grouped["padding_routed_output"] == [[0.0, 0.0]]
    assert grouped["padding_hidden_gradient_max_abs"] == 0.0
    assert grouped[
        "padding_value_and_group_id_mutation_active_output_max_abs_difference"
    ] == 0.0
    assert grouped[
        "padding_value_and_group_id_mutation_balance_abs_difference"
    ] == 0.0
    assert grouped["padding_value_and_group_id_mutation_z_abs_difference"] == 0.0
    overflow = report["overflow_policy_control"]
    assert isinstance(overflow, dict)
    assert overflow["capacity_factor"] == 1.0
    assert overflow["top_k"] == 1
    assert overflow["expert_capacity"] == 2
    assert overflow["ranked_expert_indices"] == [[0, 2, 1]] * 4
    assert overflow["selected_expert_indices"] == [[0]] * 4
    drop = overflow["drop"]
    assert isinstance(drop, dict)
    assert drop["expert_counts_before_capacity"] == [4, 0, 0]
    assert drop["expert_counts_after_capacity"] == [2, 0, 0]
    assert drop["pre_policy_capacity_excess_by_group"] == [[2, 0, 0]]
    assert drop["post_policy_capacity_excess_by_group"] == [[0, 0, 0]]
    assert drop["rerouted_assignments"] == 0
    assert drop["dropped_assignments"] == 2
    reroute = overflow["reroute"]
    assert isinstance(reroute, dict)
    assert reroute["dispatched_expert_indices"] == [[0], [0], [2], [2]]
    assert reroute["expert_counts_after_capacity"] == [2, 0, 2]
    assert reroute["post_policy_capacity_excess_by_group"] == [[0, 0, 0]]
    assert reroute["rerouted_assignments"] == 2
    assert reroute["dropped_assignments"] == 0
    assert reroute["renormalized_weight_sums"] == [1.0] * 4
    assert reroute["preserved_mass_weight_sums"] == pytest.approx(
        [1.0, 1.0, 0.44932896411722156, 0.44932896411722156]
    )
    assert reroute[
        "renormalize_vs_preserve_output_max_abs_difference"
    ] == pytest.approx(0.06399997177521191)
    assert reroute["sparse_dense_output_max_abs_difference"] == 0.0
    assert reroute[
        "sparse_dense_materialized_zero_gradient_max_abs_difference"
    ] == 0.0
    assert reroute["sparse_parameters_with_missing_zero_gradient"] == [
        "experts.1.0.weight",
        "experts.1.2.weight",
    ]
    assert reroute["dense_corresponding_gradients_are_zero"] is True
    dropless = overflow["dropless"]
    assert isinstance(dropless, dict)
    assert dropless["dispatched_expert_indices"] == [[0]] * 4
    assert dropless["expert_counts_after_capacity"] == [4, 0, 0]
    assert dropless["post_policy_capacity_excess_by_group"] == [[2, 0, 0]]
    assert dropless["rerouted_assignments"] == 0
    assert dropless["dropped_assignments"] == 0
    assert dropless["sparse_dense_output_max_abs_difference"] == 0.0
    assert dropless[
        "sparse_dense_materialized_zero_gradient_max_abs_difference"
    ] == 0.0
    assert dropless["dense_corresponding_gradients_are_zero"] is True
    assert overflow["drop_vs_reroute_output_max_abs_difference"] == pytest.approx(
        0.11622178688336826
    )
    assert overflow[
        "reroute_vs_dropless_output_max_abs_difference"
    ] == pytest.approx(0.10698215447093767)
    optimizer = report["optimizer_step"]
    assert isinstance(optimizer, dict)
    assert optimizer["task_loss_before"] == pytest.approx(0.08864729306070791)
    assert optimizer["task_loss_after"] == pytest.approx(0.08755795603512319)
    negative = report["detached_gate_negative_control"]
    assert isinstance(negative, dict)
    assert negative["router_task_gradient_is_missing"] is True
    balance = report["balance_gradient_control"]
    assert isinstance(balance, dict)
    assert balance["selected_expert_indices_before"] == [[0], [0], [0], [0]]
    assert balance["selected_expert_indices_after"] == [[0], [0], [0], [0]]
    assert balance["load_balance_loss_after"] < balance["load_balance_loss_before"]
    assertions = report["assertions"]
    assert isinstance(assertions, dict)
    assert all(assertions.values())
    assert report["scope"] == {
        "authored_balance_and_z_loss_gradients_executed": True,
        "all_assignments_dropped_zero_routed_output_executed": True,
        "convergence_quality_throughput_memory_or_scaling_proved": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "detached_gate_missing_router_task_gradient_negative_control_executed": True,
        "distributed_capacity_group_collective_executed": False,
        "expert_parallel_all_to_all_grouped_gemm_or_gpu_executed": False,
        "hard_topk_indices_treated_as_nondifferentiable": True,
        "padding_aware_capacity_aux_and_gradient_executed": True,
        "post_drop_renormalize_and_preserve_mass_policies_executed": True,
        "routing_group_scoped_capacity_and_aux_executed": True,
        "score_priority_capacity_drop_in_training_graph_executed": True,
        "selected_probability_task_gradient_to_router_executed": True,
        "shared_or_fine_grained_experts_executed": False,
        "sparse_dispatch_dense_oracle_forward_backward_compared": True,
        "trainable_router_and_expert_mlp_forward_backward_executed": True,
        "deterministic_full_ranking_reroute_policy_executed": True,
        "dropless_nominal_capacity_excess_policy_executed": True,
    }
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()


def test_project_control_emits_strict_finite_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    report = json.loads(
        completed.stdout,
        parse_constant=_reject_nonfinite,
    )

    assert completed.stderr == ""
    assert report["schema_version"] == TRAINABLE_MOE_CONTROL_VERSION
    assert report["report_fingerprint"].startswith("sha256:")
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
