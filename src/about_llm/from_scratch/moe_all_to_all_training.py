"""Two-process CPU/Gloo MoE all-to-all forward/backward control."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.multiprocessing.spawn import spawn

MOE_ALL_TO_ALL_TRAINING_CONTROL_VERSION = (
    "about-llm.moe-all-to-all-training-control.v1"
)
MOE_ALL_TO_ALL_CAPACITY_TRAINING_CONTROL_VERSION = (
    "about-llm.moe-all-to-all-capacity-training-control.v1"
)
WORLD_SIZE = 2
GLOBAL_TOKEN_COUNT = 4
LEARNING_RATE = 0.01
CAPACITY_FACTOR = 0.5
EXPERT_CAPACITY = 1
MAX_LOCAL_TOKENS = 3
FLOAT_COLUMNS = 2
METADATA_COLUMNS = 4

_CALL_COUNTS = {
    "autograd_payload_forward_all_to_all_single": 0,
    "autograd_payload_backward_all_to_all_single": 0,
    "nondifferentiable_count_or_metadata_all_to_all_single": 0,
    "router_gradient_all_reduce": 0,
}
_CAPACITY_ROUTE_ALL_GATHER_CALLS = 0


class WorkerReport(TypedDict):
    rank: int
    process_id: int
    owned_expert_id: int
    local_hidden_states: list[list[float]]
    local_targets: list[list[float]]
    local_global_token_ids: list[int]
    selected_expert_indices: list[int]
    selected_probabilities: list[float]
    source_to_owner_counts: list[int]
    owner_from_source_counts: list[int]
    owner_received_metadata: list[list[int]]
    return_arrival_metadata: list[list[int]]
    training_outputs: list[list[float]]
    evaluation_outputs_after_step: list[list[float]]
    local_loss_contribution_before_step: float
    local_loss_contribution_after_step: float
    local_hidden_gradients: list[list[float]]
    router_gradient_before_all_reduce: list[list[float]]
    router_gradient_after_all_reduce: list[list[float]]
    owned_expert_weight_gradient: list[list[float]]
    owned_expert_bias_gradient: list[float]
    router_weight_before_step: list[list[float]]
    router_weight_after_step: list[list[float]]
    owned_expert_weight_before_step: list[list[float]]
    owned_expert_weight_after_step: list[list[float]]
    owned_expert_bias_before_step: list[float]
    owned_expert_bias_after_step: list[float]
    authored_collective_call_counts: dict[str, int]


class PublishedWorkerReport(TypedDict):
    rank: int
    owned_expert_id: int
    local_hidden_states: list[list[float]]
    local_targets: list[list[float]]
    local_global_token_ids: list[int]
    selected_expert_indices: list[int]
    selected_probabilities: list[float]
    source_to_owner_counts: list[int]
    owner_from_source_counts: list[int]
    owner_received_metadata: list[list[int]]
    return_arrival_metadata: list[list[int]]
    training_outputs: list[list[float]]
    evaluation_outputs_after_step: list[list[float]]
    local_loss_contribution_before_step: float
    local_loss_contribution_after_step: float
    local_hidden_gradients: list[list[float]]
    router_gradient_before_all_reduce: list[list[float]]
    router_gradient_after_all_reduce: list[list[float]]
    owned_expert_weight_gradient: list[list[float]]
    owned_expert_bias_gradient: list[float]
    router_weight_before_step: list[list[float]]
    router_weight_after_step: list[list[float]]
    owned_expert_weight_before_step: list[list[float]]
    owned_expert_weight_after_step: list[list[float]]
    owned_expert_bias_before_step: list[float]
    owned_expert_bias_after_step: list[float]
    authored_collective_call_counts: dict[str, int]


class CapacityWorkerReport(TypedDict):
    rank: int
    process_id: int
    owned_expert_id: int
    local_hidden_states: list[list[float]]
    local_targets: list[list[float]]
    local_global_token_ids: list[int]
    selected_expert_indices: list[int]
    selected_probabilities: list[float]
    global_keep_mask: list[bool]
    local_keep_mask: list[bool]
    selected_counts_by_expert: list[int]
    kept_counts_by_expert: list[int]
    dropped_assignments: int
    source_to_owner_counts: list[int]
    owner_from_source_counts: list[int]
    owner_received_metadata: list[list[int]]
    return_arrival_metadata: list[list[int]]
    training_outputs: list[list[float]]
    evaluation_outputs_after_step: list[list[float]]
    local_loss_contribution_before_step: float
    local_loss_contribution_after_step: float
    local_hidden_gradients: list[list[float]]
    router_gradient_before_all_reduce: list[list[float]]
    router_gradient_after_all_reduce: list[list[float]]
    owned_expert_weight_gradient: list[list[float]]
    owned_expert_bias_gradient: list[float]
    router_weight_after_step: list[list[float]]
    owned_expert_weight_after_step: list[list[float]]
    owned_expert_bias_after_step: list[float]
    authored_collective_call_counts: dict[str, int]


class PublishedCapacityWorkerReport(TypedDict):
    rank: int
    owned_expert_id: int
    local_hidden_states: list[list[float]]
    local_targets: list[list[float]]
    local_global_token_ids: list[int]
    selected_expert_indices: list[int]
    selected_probabilities: list[float]
    global_keep_mask: list[bool]
    local_keep_mask: list[bool]
    selected_counts_by_expert: list[int]
    kept_counts_by_expert: list[int]
    dropped_assignments: int
    source_to_owner_counts: list[int]
    owner_from_source_counts: list[int]
    owner_received_metadata: list[list[int]]
    return_arrival_metadata: list[list[int]]
    training_outputs: list[list[float]]
    evaluation_outputs_after_step: list[list[float]]
    local_loss_contribution_before_step: float
    local_loss_contribution_after_step: float
    local_hidden_gradients: list[list[float]]
    router_gradient_before_all_reduce: list[list[float]]
    router_gradient_after_all_reduce: list[list[float]]
    owned_expert_weight_gradient: list[list[float]]
    owned_expert_bias_gradient: list[float]
    router_weight_after_step: list[list[float]]
    owned_expert_weight_after_step: list[list[float]]
    owned_expert_bias_after_step: list[float]
    authored_collective_call_counts: dict[str, int]


@dataclass(frozen=True)
class RouterDecision:
    selected_expert_indices: Tensor
    selected_probabilities: Tensor


@dataclass(frozen=True)
class DistributedForward:
    decision: RouterDecision
    send_counts: Tensor
    received_counts: Tensor
    received_metadata: Tensor
    returned_metadata: Tensor
    final_output: Tensor


@dataclass(frozen=True)
class CapacityDistributedForward:
    decision: RouterDecision
    global_keep_mask: list[bool]
    local_keep_mask: Tensor
    selected_counts_by_expert: list[int]
    kept_counts_by_expert: list[int]
    send_counts: Tensor
    received_counts: Tensor
    received_metadata: Tensor
    returned_metadata: Tensor
    final_output: Tensor


class _VariableSplitAllToAll(torch.autograd.Function):
    """Authored autograd binding whose backward reverses variable splits."""

    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: Tensor,
        input_splits: tuple[int, ...],
        output_splits: tuple[int, ...],
    ) -> Tensor:
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        output = torch.empty(
            (sum(output_splits), *input_tensor.shape[1:]),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        dist.all_to_all_single(
            output,
            input_tensor.contiguous(),
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
        )
        _CALL_COUNTS["autograd_payload_forward_all_to_all_single"] += 1
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None]:
        input_splits = cast(tuple[int, ...], ctx.input_splits)
        output_splits = cast(tuple[int, ...], ctx.output_splits)
        grad_input = torch.empty(
            (sum(input_splits), *grad_output.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=list(input_splits),
            input_split_sizes=list(output_splits),
        )
        _CALL_COUNTS["autograd_payload_backward_all_to_all_single"] += 1
        return grad_input, None, None


class OwnedExpert(nn.Module):
    """A one-dimensional expert instantiated only by its owner rank."""

    def __init__(self, expert_id: int) -> None:
        super().__init__()
        if expert_id not in {0, 1}:
            raise ValueError("expert_id must be 0 or 1")
        self.expert_id = expert_id
        self.linear = nn.Linear(1, 1, bias=True, dtype=torch.float64)
        with torch.no_grad():
            if expert_id == 0:
                self.linear.weight.fill_(2.0)
                self.linear.bias.fill_(0.5)
            else:
                self.linear.weight.fill_(-3.0)
                self.linear.bias.fill_(1.0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return cast(Tensor, self.linear(hidden_states))


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


def _reset_call_counts() -> None:
    for key in _CALL_COUNTS:
        _CALL_COUNTS[key] = 0


def _local_hidden(rank: int) -> Tensor:
    if rank == 0:
        values = [[-1.0], [2.0], [-2.0]]
    elif rank == 1:
        values = [[1.0]]
    else:
        raise ValueError(f"rank must be in [0, {WORLD_SIZE})")
    return torch.tensor(values, dtype=torch.float64)


def _local_targets(rank: int) -> Tensor:
    if rank == 0:
        values = [[0.25], [-0.5], [1.0]]
    elif rank == 1:
        values = [[-1.5]]
    else:
        raise ValueError(f"rank must be in [0, {WORLD_SIZE})")
    return torch.tensor(values, dtype=torch.float64)


def _global_token_offset(rank: int) -> int:
    return sum(_local_hidden(lower).shape[0] for lower in range(rank))


def _new_router() -> nn.Linear:
    router = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[1.0], [-1.0]], dtype=torch.float64))
    return router


def _route(router: nn.Linear, hidden_states: Tensor) -> RouterDecision:
    logits = router(hidden_states)
    probabilities = torch.softmax(logits, dim=-1)
    ranking = torch.argsort(
        probabilities,
        dim=-1,
        descending=True,
        stable=True,
    )
    selected = ranking[:, 0]
    selected_probabilities = torch.gather(
        probabilities,
        dim=1,
        index=selected.unsqueeze(1),
    ).squeeze(1)
    return RouterDecision(selected, selected_probabilities)


def _pack_dispatch(
    rank: int,
    hidden_states: Tensor,
    decision: RouterDecision,
) -> tuple[Tensor, Tensor, Tensor]:
    local_count = hidden_states.shape[0]
    local_indices = torch.arange(local_count, dtype=torch.int64)
    global_ids = local_indices + _global_token_offset(rank)
    order = torch.argsort(decision.selected_expert_indices, stable=True)
    send_counts = torch.bincount(
        decision.selected_expert_indices,
        minlength=WORLD_SIZE,
    ).to(dtype=torch.int64)
    float_payload = torch.cat(
        [
            hidden_states[order],
            decision.selected_probabilities[order].unsqueeze(1),
        ],
        dim=1,
    ).contiguous()
    metadata = torch.stack(
        [
            torch.full_like(local_indices, rank)[order],
            local_indices[order],
            global_ids[order],
            decision.selected_expert_indices[order],
        ],
        dim=1,
    ).contiguous()
    return send_counts, float_payload, metadata


def _split_tuple(counts: Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in counts.tolist())


def _autograd_all_to_all_rows(
    input_tensor: Tensor,
    *,
    input_counts: Tensor,
    output_counts: Tensor,
) -> Tensor:
    return cast(
        Tensor,
        _VariableSplitAllToAll.apply(  # type: ignore[no-untyped-call]
            input_tensor,
            _split_tuple(input_counts),
            _split_tuple(output_counts),
        ),
    )


def _raw_all_to_all_rows(
    input_tensor: Tensor,
    *,
    input_counts: Tensor,
    output_counts: Tensor,
) -> Tensor:
    output = torch.empty(
        (int(output_counts.sum().item()), *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=list(_split_tuple(output_counts)),
        input_split_sizes=list(_split_tuple(input_counts)),
    )
    _CALL_COUNTS["nondifferentiable_count_or_metadata_all_to_all_single"] += 1
    return output


def _exchange_counts(send_counts: Tensor) -> Tensor:
    received_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(received_counts, send_counts)
    _CALL_COUNTS["nondifferentiable_count_or_metadata_all_to_all_single"] += 1
    return received_counts


def _restore_source_order(
    rank: int,
    local_count: int,
    returned_float: Tensor,
    returned_metadata: Tensor,
) -> Tensor:
    if returned_float.shape != (local_count, FLOAT_COLUMNS):
        raise AssertionError("returned float payload cardinality drifted")
    if returned_metadata.shape != (local_count, METADATA_COLUMNS):
        raise AssertionError("returned metadata cardinality drifted")
    arrival_row_by_local_index = torch.empty(local_count, dtype=torch.int64)
    seen: set[int] = set()
    for arrival_row in range(local_count):
        source_rank = int(returned_metadata[arrival_row, 0].item())
        local_index = int(returned_metadata[arrival_row, 1].item())
        if source_rank != rank or local_index in seen:
            raise AssertionError("invalid returned source metadata")
        if local_index < 0 or local_index >= local_count:
            raise AssertionError("returned local index is out of range")
        arrival_row_by_local_index[local_index] = arrival_row
        seen.add(local_index)
    if len(seen) != local_count:
        raise AssertionError("not every source token returned")
    reordered = returned_float[arrival_row_by_local_index]
    return reordered[:, :1] * reordered[:, 1:2]


def _distributed_forward(
    rank: int,
    hidden_states: Tensor,
    router: nn.Linear,
    owned_expert: OwnedExpert,
) -> DistributedForward:
    decision = _route(router, hidden_states)
    send_counts, send_float, send_metadata = _pack_dispatch(
        rank,
        hidden_states,
        decision,
    )
    received_counts = _exchange_counts(send_counts)
    received_float = _autograd_all_to_all_rows(
        send_float,
        input_counts=send_counts,
        output_counts=received_counts,
    )
    received_metadata = _raw_all_to_all_rows(
        send_metadata,
        input_counts=send_counts,
        output_counts=received_counts,
    )
    if not bool(torch.all(received_metadata[:, 3] == rank).item()):
        raise AssertionError("owner received a token for another expert")
    raw_output = owned_expert(received_float[:, :1])
    return_float = torch.cat(
        [raw_output, received_float[:, 1:2]],
        dim=1,
    ).contiguous()
    returned_float = _autograd_all_to_all_rows(
        return_float,
        input_counts=received_counts,
        output_counts=send_counts,
    )
    returned_metadata = _raw_all_to_all_rows(
        received_metadata,
        input_counts=received_counts,
        output_counts=send_counts,
    )
    final_output = _restore_source_order(
        rank,
        hidden_states.shape[0],
        returned_float,
        returned_metadata,
    )
    return DistributedForward(
        decision=decision,
        send_counts=send_counts,
        received_counts=received_counts,
        received_metadata=received_metadata,
        returned_metadata=returned_metadata,
        final_output=final_output,
    )


def _score_priority_capacity_mask(
    global_ids: list[int],
    expert_ids: list[int],
    probabilities: list[float],
) -> tuple[list[bool], list[int], list[int]]:
    if not (
        len(global_ids) == len(expert_ids) == len(probabilities) == GLOBAL_TOKEN_COUNT
    ):
        raise AssertionError("capacity inputs must contain every global token")
    if sorted(global_ids) != list(range(GLOBAL_TOKEN_COUNT)):
        raise AssertionError("global token ids must be unique and contiguous")
    selected_counts = [0] * WORLD_SIZE
    kept_global_ids: set[int] = set()
    for expert_id in range(WORLD_SIZE):
        candidates = [
            (probability, global_id)
            for global_id, selected, probability in zip(
                global_ids,
                expert_ids,
                probabilities,
                strict=True,
            )
            if selected == expert_id
        ]
        selected_counts[expert_id] = len(candidates)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        kept_global_ids.update(
            global_id for _, global_id in candidates[:EXPERT_CAPACITY]
        )
    keep_mask = [
        global_id in kept_global_ids for global_id in range(GLOBAL_TOKEN_COUNT)
    ]
    kept_counts = [0] * WORLD_SIZE
    for global_id, expert_id in zip(global_ids, expert_ids, strict=True):
        if keep_mask[global_id]:
            kept_counts[expert_id] += 1
    return keep_mask, selected_counts, kept_counts


def _collective_capacity_mask(
    rank: int,
    decision: RouterDecision,
) -> tuple[list[bool], Tensor, list[int], list[int]]:
    global _CAPACITY_ROUTE_ALL_GATHER_CALLS

    local_count = decision.selected_expert_indices.shape[0]
    padded_probabilities = torch.zeros(MAX_LOCAL_TOKENS, dtype=torch.float64)
    padded_metadata = torch.zeros((MAX_LOCAL_TOKENS, 3), dtype=torch.int64)
    padded_probabilities[:local_count] = decision.selected_probabilities.detach()
    local_indices = torch.arange(local_count, dtype=torch.int64)
    padded_metadata[:local_count, 0] = 1
    padded_metadata[:local_count, 1] = local_indices + _global_token_offset(rank)
    padded_metadata[:local_count, 2] = decision.selected_expert_indices.detach()

    gathered_probabilities = [
        torch.empty_like(padded_probabilities) for _ in range(WORLD_SIZE)
    ]
    gathered_metadata = [
        torch.empty_like(padded_metadata) for _ in range(WORLD_SIZE)
    ]
    dist.all_gather(gathered_probabilities, padded_probabilities)
    _CAPACITY_ROUTE_ALL_GATHER_CALLS += 1
    dist.all_gather(gathered_metadata, padded_metadata)
    _CAPACITY_ROUTE_ALL_GATHER_CALLS += 1

    global_ids: list[int] = []
    expert_ids: list[int] = []
    probabilities: list[float] = []
    for probability_rows, metadata_rows in zip(
        gathered_probabilities,
        gathered_metadata,
        strict=True,
    ):
        for row in range(MAX_LOCAL_TOKENS):
            if int(metadata_rows[row, 0].item()) == 0:
                continue
            global_ids.append(int(metadata_rows[row, 1].item()))
            expert_ids.append(int(metadata_rows[row, 2].item()))
            probabilities.append(float(probability_rows[row].item()))
    global_keep, selected_counts, kept_counts = _score_priority_capacity_mask(
        global_ids,
        expert_ids,
        probabilities,
    )
    offset = _global_token_offset(rank)
    local_keep = torch.tensor(
        [global_keep[offset + index] for index in range(local_count)],
        dtype=torch.bool,
    )
    return global_keep, local_keep, selected_counts, kept_counts


def _pack_kept_dispatch(
    rank: int,
    hidden_states: Tensor,
    decision: RouterDecision,
    local_keep_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    local_count = hidden_states.shape[0]
    kept_local_indices = torch.nonzero(
        local_keep_mask,
        as_tuple=False,
    ).squeeze(1)
    kept_experts = decision.selected_expert_indices[kept_local_indices]
    kept_order = torch.argsort(kept_experts, stable=True)
    ordered_local_indices = kept_local_indices[kept_order]
    ordered_experts = kept_experts[kept_order]
    send_counts = torch.bincount(
        ordered_experts,
        minlength=WORLD_SIZE,
    ).to(dtype=torch.int64)
    float_payload = torch.cat(
        [
            hidden_states[ordered_local_indices],
            decision.selected_probabilities[ordered_local_indices].unsqueeze(1),
        ],
        dim=1,
    ).contiguous()
    global_ids = ordered_local_indices + _global_token_offset(rank)
    metadata = torch.stack(
        [
            torch.full_like(ordered_local_indices, rank),
            ordered_local_indices,
            global_ids,
            ordered_experts,
        ],
        dim=1,
    ).contiguous()
    if local_keep_mask.shape != (local_count,):
        raise AssertionError("local keep mask shape drifted")
    return send_counts, float_payload, metadata


def _restore_kept_source_order(
    rank: int,
    local_count: int,
    returned_float: Tensor,
    returned_metadata: Tensor,
    decision: RouterDecision,
    local_keep_mask: Tensor,
) -> Tensor:
    expected_kept = int(local_keep_mask.sum().item())
    if returned_float.shape != (expected_kept, FLOAT_COLUMNS):
        raise AssertionError("returned kept float cardinality drifted")
    if returned_metadata.shape != (expected_kept, METADATA_COLUMNS):
        raise AssertionError("returned kept metadata cardinality drifted")
    output = torch.zeros((local_count, 1), dtype=torch.float64)
    kept_local_indices: list[int] = []
    weighted_rows: list[Tensor] = []
    for arrival_row in range(expected_kept):
        source_rank = int(returned_metadata[arrival_row, 0].item())
        local_index = int(returned_metadata[arrival_row, 1].item())
        if source_rank != rank or not bool(local_keep_mask[local_index].item()):
            raise AssertionError("returned assignment is not locally kept")
        if local_index in kept_local_indices:
            raise AssertionError("returned kept assignment is duplicated")
        kept_local_indices.append(local_index)
        weighted_rows.append(
            returned_float[arrival_row, :1]
            * returned_float[arrival_row, 1:2]
        )
    if weighted_rows:
        output = output.index_copy(
            0,
            torch.tensor(kept_local_indices, dtype=torch.int64),
            torch.stack(weighted_rows, dim=0),
        )
    graph_zero = (
        returned_float.sum() * 0.0
        + decision.selected_probabilities.sum() * 0.0
    )
    return output + graph_zero


def _capacity_distributed_forward(
    rank: int,
    hidden_states: Tensor,
    router: nn.Linear,
    owned_expert: OwnedExpert,
) -> CapacityDistributedForward:
    decision = _route(router, hidden_states)
    global_keep, local_keep, selected_counts, kept_counts = (
        _collective_capacity_mask(rank, decision)
    )
    send_counts, send_float, send_metadata = _pack_kept_dispatch(
        rank,
        hidden_states,
        decision,
        local_keep,
    )
    received_counts = _exchange_counts(send_counts)
    received_float = _autograd_all_to_all_rows(
        send_float,
        input_counts=send_counts,
        output_counts=received_counts,
    )
    received_metadata = _raw_all_to_all_rows(
        send_metadata,
        input_counts=send_counts,
        output_counts=received_counts,
    )
    if not bool(torch.all(received_metadata[:, 3] == rank).item()):
        raise AssertionError("owner received a kept token for another expert")
    raw_output = owned_expert(received_float[:, :1])
    return_float = torch.cat(
        [raw_output, received_float[:, 1:2]],
        dim=1,
    ).contiguous()
    returned_float = _autograd_all_to_all_rows(
        return_float,
        input_counts=received_counts,
        output_counts=send_counts,
    )
    returned_metadata = _raw_all_to_all_rows(
        received_metadata,
        input_counts=received_counts,
        output_counts=send_counts,
    )
    final_output = _restore_kept_source_order(
        rank,
        hidden_states.shape[0],
        returned_float,
        returned_metadata,
        decision,
        local_keep,
    )
    return CapacityDistributedForward(
        decision=decision,
        global_keep_mask=global_keep,
        local_keep_mask=local_keep,
        selected_counts_by_expert=selected_counts,
        kept_counts_by_expert=kept_counts,
        send_counts=send_counts,
        received_counts=received_counts,
        received_metadata=received_metadata,
        returned_metadata=returned_metadata,
        final_output=final_output,
    )


def _require_gradient(parameter: Tensor, name: str) -> Tensor:
    gradient = parameter.grad
    if gradient is None or not bool(torch.isfinite(gradient).all().item()):
        raise AssertionError(f"{name} gradient must be finite and materialized")
    return gradient


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_directory: str,
) -> None:
    if world_size != WORLD_SIZE:
        raise ValueError(f"world_size must be {WORLD_SIZE}")
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        _reset_call_counts()
        hidden_states = _local_hidden(rank).requires_grad_(True)
        targets = _local_targets(rank)
        router = _new_router()
        owned_expert = OwnedExpert(rank)
        optimizer = torch.optim.SGD(
            [*router.parameters(), *owned_expert.parameters()],
            lr=LEARNING_RATE,
        )
        router_before = router.weight.detach().clone()
        expert_weight_before = owned_expert.linear.weight.detach().clone()
        expert_bias_before = owned_expert.linear.bias.detach().clone()

        training = _distributed_forward(
            rank,
            hidden_states,
            router,
            owned_expert,
        )
        local_loss = functional.mse_loss(
            training.final_output,
            targets,
            reduction="sum",
        ) / GLOBAL_TOKEN_COUNT
        local_loss.backward()  # type: ignore[no-untyped-call]
        router_gradient = _require_gradient(router.weight, "router.weight")
        router_gradient_before_reduce = router_gradient.detach().clone()
        dist.all_reduce(router_gradient, op=dist.ReduceOp.SUM)
        _CALL_COUNTS["router_gradient_all_reduce"] += 1
        router_gradient_after_reduce = router_gradient.detach().clone()
        hidden_gradient = _require_gradient(hidden_states, "hidden_states")
        expert_weight_gradient = _require_gradient(
            owned_expert.linear.weight,
            "owned_expert.weight",
        )
        expert_bias_gradient = _require_gradient(
            owned_expert.linear.bias,
            "owned_expert.bias",
        )
        optimizer.step()

        with torch.no_grad():
            evaluation = _distributed_forward(
                rank,
                hidden_states.detach(),
                router,
                owned_expert,
            )
            evaluation_loss = functional.mse_loss(
                evaluation.final_output,
                targets,
                reduction="sum",
            ) / GLOBAL_TOKEN_COUNT

        offset = _global_token_offset(rank)
        payload: WorkerReport = {
            "rank": rank,
            "process_id": os.getpid(),
            "owned_expert_id": rank,
            "local_hidden_states": hidden_states.detach().tolist(),
            "local_targets": targets.tolist(),
            "local_global_token_ids": list(
                range(offset, offset + hidden_states.shape[0])
            ),
            "selected_expert_indices": (
                training.decision.selected_expert_indices.tolist()
            ),
            "selected_probabilities": (
                training.decision.selected_probabilities.detach().tolist()
            ),
            "source_to_owner_counts": training.send_counts.tolist(),
            "owner_from_source_counts": training.received_counts.tolist(),
            "owner_received_metadata": training.received_metadata.tolist(),
            "return_arrival_metadata": training.returned_metadata.tolist(),
            "training_outputs": training.final_output.detach().tolist(),
            "evaluation_outputs_after_step": (
                evaluation.final_output.detach().tolist()
            ),
            "local_loss_contribution_before_step": float(local_loss.item()),
            "local_loss_contribution_after_step": float(evaluation_loss.item()),
            "local_hidden_gradients": hidden_gradient.detach().tolist(),
            "router_gradient_before_all_reduce": (
                router_gradient_before_reduce.tolist()
            ),
            "router_gradient_after_all_reduce": (
                router_gradient_after_reduce.tolist()
            ),
            "owned_expert_weight_gradient": (
                expert_weight_gradient.detach().tolist()
            ),
            "owned_expert_bias_gradient": (
                expert_bias_gradient.detach().tolist()
            ),
            "router_weight_before_step": router_before.tolist(),
            "router_weight_after_step": router.weight.detach().tolist(),
            "owned_expert_weight_before_step": expert_weight_before.tolist(),
            "owned_expert_weight_after_step": (
                owned_expert.linear.weight.detach().tolist()
            ),
            "owned_expert_bias_before_step": expert_bias_before.tolist(),
            "owned_expert_bias_after_step": (
                owned_expert.linear.bias.detach().tolist()
            ),
            "authored_collective_call_counts": dict(_CALL_COUNTS),
        }
        (Path(output_directory) / f"rank-{rank}.json").write_bytes(
            _canonical_bytes(payload)
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _load_worker_report(path: Path) -> WorkerReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("worker report must be a JSON object")
    return cast(WorkerReport, payload)


def _publish_worker_report(report: WorkerReport) -> PublishedWorkerReport:
    return {key: value for key, value in report.items() if key != "process_id"}  # type: ignore[return-value]


def _single_process_oracle() -> dict[str, object]:
    hidden_states = torch.cat(
        [_local_hidden(rank) for rank in range(WORLD_SIZE)],
        dim=0,
    ).requires_grad_(True)
    targets = torch.cat(
        [_local_targets(rank) for rank in range(WORLD_SIZE)],
        dim=0,
    )
    router = _new_router()
    experts = [OwnedExpert(0), OwnedExpert(1)]
    expert_parameters = [
        parameter for expert in experts for parameter in expert.parameters()
    ]
    optimizer = torch.optim.SGD(
        [*router.parameters(), *expert_parameters],
        lr=LEARNING_RATE,
    )
    router_before = router.weight.detach().clone()
    expert_weights_before = [
        expert.linear.weight.detach().clone() for expert in experts
    ]
    expert_biases_before = [
        expert.linear.bias.detach().clone() for expert in experts
    ]

    decision = _route(router, hidden_states)
    raw_outputs = torch.empty_like(hidden_states)
    for expert_id, expert in enumerate(experts):
        indices = torch.nonzero(
            decision.selected_expert_indices == expert_id,
            as_tuple=False,
        ).squeeze(1)
        raw_outputs[indices] = expert(hidden_states[indices])
    outputs = raw_outputs * decision.selected_probabilities.unsqueeze(1)
    loss = functional.mse_loss(outputs, targets)
    loss.backward()  # type: ignore[no-untyped-call]
    router_gradient = _require_gradient(router.weight, "oracle.router.weight").detach()
    expert_weight_gradients = [
        _require_gradient(expert.linear.weight, f"oracle.expert_{index}.weight")
        .detach()
        .clone()
        for index, expert in enumerate(experts)
    ]
    expert_bias_gradients = [
        _require_gradient(expert.linear.bias, f"oracle.expert_{index}.bias")
        .detach()
        .clone()
        for index, expert in enumerate(experts)
    ]
    hidden_gradient = _require_gradient(
        hidden_states,
        "oracle.hidden_states",
    ).detach()
    optimizer.step()

    with torch.no_grad():
        decision_after = _route(router, hidden_states.detach())
        raw_outputs_after = torch.empty_like(hidden_states)
        for expert_id, expert in enumerate(experts):
            indices = torch.nonzero(
                decision_after.selected_expert_indices == expert_id,
                as_tuple=False,
            ).squeeze(1)
            raw_outputs_after[indices] = expert(hidden_states.detach()[indices])
        outputs_after = (
            raw_outputs_after
            * decision_after.selected_probabilities.unsqueeze(1)
        )
        loss_after = functional.mse_loss(outputs_after, targets)

    return {
        "selected_expert_indices": decision.selected_expert_indices.tolist(),
        "selected_probabilities": decision.selected_probabilities.detach().tolist(),
        "outputs_before_step": outputs.detach().tolist(),
        "outputs_after_step": outputs_after.tolist(),
        "loss_before_step": float(loss.item()),
        "loss_after_step": float(loss_after.item()),
        "hidden_gradients": hidden_gradient.tolist(),
        "router_gradient": router_gradient.tolist(),
        "expert_weight_gradients": [
            gradient.tolist() for gradient in expert_weight_gradients
        ],
        "expert_bias_gradients": [
            gradient.tolist() for gradient in expert_bias_gradients
        ],
        "router_weight_before_step": router_before.tolist(),
        "router_weight_after_step": router.weight.detach().tolist(),
        "expert_weights_before_step": [
            parameter.tolist() for parameter in expert_weights_before
        ],
        "expert_weights_after_step": [
            expert.linear.weight.detach().tolist() for expert in experts
        ],
        "expert_biases_before_step": [
            parameter.tolist() for parameter in expert_biases_before
        ],
        "expert_biases_after_step": [
            expert.linear.bias.detach().tolist() for expert in experts
        ],
    }


def _max_abs_difference(left: object, right: object) -> float:
    left_tensor = torch.tensor(left, dtype=torch.float64)
    right_tensor = torch.tensor(right, dtype=torch.float64)
    if left_tensor.shape != right_tensor.shape:
        raise AssertionError(
            f"comparison shape mismatch: {left_tensor.shape} != {right_tensor.shape}"
        )
    if left_tensor.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left_tensor - right_tensor)).item())


def _ordered_values(
    worker_reports: list[WorkerReport],
    field: str,
) -> list[list[float]]:
    by_global_id: dict[int, list[float]] = {}
    for report in worker_reports:
        values = cast(list[list[float]], report[field])  # type: ignore[literal-required]
        for global_id, value in zip(
            report["local_global_token_ids"],
            values,
            strict=True,
        ):
            by_global_id[global_id] = value
    return [by_global_id[index] for index in range(GLOBAL_TOKEN_COUNT)]


def run_moe_all_to_all_training_control() -> dict[str, object]:
    """Run forward, reverse all-to-all gradients, synchronization, and SGD."""

    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed with the Gloo backend is required")
    with tempfile.TemporaryDirectory(
        prefix="about-llm-moe-all-to-all-training-"
    ) as temporary:
        root = Path(temporary)
        init_method = (root / "rendezvous").resolve().as_uri()
        output_directory = root / "worker-results"
        output_directory.mkdir()
        spawn(  # type: ignore[no-untyped-call]
            _worker,
            args=(WORLD_SIZE, init_method, str(output_directory)),
            nprocs=WORLD_SIZE,
            join=True,
        )
        worker_reports = [
            _load_worker_report(output_directory / f"rank-{rank}.json")
            for rank in range(WORLD_SIZE)
        ]

    oracle = _single_process_oracle()
    outputs_before = _ordered_values(worker_reports, "training_outputs")
    outputs_after = _ordered_values(
        worker_reports,
        "evaluation_outputs_after_step",
    )
    hidden_gradients = _ordered_values(
        worker_reports,
        "local_hidden_gradients",
    )
    loss_before = sum(
        report["local_loss_contribution_before_step"]
        for report in worker_reports
    )
    loss_after = sum(
        report["local_loss_contribution_after_step"]
        for report in worker_reports
    )
    router_gradient_differences = [
        _max_abs_difference(
            report["router_gradient_after_all_reduce"],
            oracle["router_gradient"],
        )
        for report in worker_reports
    ]
    expert_gradient_differences = [
        max(
            _max_abs_difference(
                report["owned_expert_weight_gradient"],
                cast(list[object], oracle["expert_weight_gradients"])[rank],
            ),
            _max_abs_difference(
                report["owned_expert_bias_gradient"],
                cast(list[object], oracle["expert_bias_gradients"])[rank],
            ),
        )
        for rank, report in enumerate(worker_reports)
    ]
    parameter_differences = [
        max(
            _max_abs_difference(
                report["router_weight_after_step"],
                oracle["router_weight_after_step"],
            ),
            _max_abs_difference(
                report["owned_expert_weight_after_step"],
                cast(list[object], oracle["expert_weights_after_step"])[rank],
            ),
            _max_abs_difference(
                report["owned_expert_bias_after_step"],
                cast(list[object], oracle["expert_biases_after_step"])[rank],
            ),
        )
        for rank, report in enumerate(worker_reports)
    ]
    expected_call_counts = {
        "autograd_payload_forward_all_to_all_single": 4,
        "autograd_payload_backward_all_to_all_single": 2,
        "nondifferentiable_count_or_metadata_all_to_all_single": 6,
        "router_gradient_all_reduce": 1,
    }
    assertions = {
        "two_distinct_worker_processes_executed": (
            len({report["process_id"] for report in worker_reports}) == WORLD_SIZE
        ),
        "owner_only_expert_placement_and_routes_match": (
            [report["owned_expert_id"] for report in worker_reports] == [0, 1]
            and [report["selected_expert_indices"] for report in worker_reports]
            == [[1, 0, 1], [0]]
            and [report["source_to_owner_counts"] for report in worker_reports]
            == [[1, 2], [1, 0]]
        ),
        "forward_and_post_step_evaluation_match_single_process_oracle": (
            _max_abs_difference(outputs_before, oracle["outputs_before_step"])
            <= 1e-15
            and _max_abs_difference(outputs_after, oracle["outputs_after_step"])
            <= 1e-15
        ),
        "reverse_all_to_all_hidden_gradients_match_oracle": (
            _max_abs_difference(hidden_gradients, oracle["hidden_gradients"])
            <= 1e-15
        ),
        "router_gradient_all_reduce_matches_global_oracle": (
            max(router_gradient_differences) <= 1e-15
            and worker_reports[0]["router_gradient_after_all_reduce"]
            == worker_reports[1]["router_gradient_after_all_reduce"]
        ),
        "owner_expert_gradients_match_global_oracle": (
            max(expert_gradient_differences) <= 1e-15
        ),
        "one_step_parameters_match_global_oracle": (
            max(parameter_differences) <= 1e-15
        ),
        "global_mean_loss_matches_and_decreases_after_step": (
            math.isclose(
                loss_before,
                cast(float, oracle["loss_before_step"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            and math.isclose(
                loss_after,
                cast(float, oracle["loss_after_step"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            and loss_after < loss_before
        ),
        "authored_collective_call_ledger_matches_execution": all(
            report["authored_collective_call_counts"] == expected_call_counts
            for report in worker_reports
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"MoE all-to-all training assertion failed: {assertions}")

    report: dict[str, object] = {
        "schema_version": MOE_ALL_TO_ALL_TRAINING_CONTROL_VERSION,
        "runtime": {
            "torch_version": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
            "dtype": "torch.float64",
            "world_size": WORLD_SIZE,
            "process_start_method": "spawn",
            "rendezvous": "temporary-file-store",
        },
        "fixture": {
            "local_hidden_states_by_rank": [
                _local_hidden(rank).tolist() for rank in range(WORLD_SIZE)
            ],
            "local_targets_by_rank": [
                _local_targets(rank).tolist() for rank in range(WORLD_SIZE)
            ],
            "router_weight": [[1.0], [-1.0]],
            "expert_ownership": {"expert_0": 0, "expert_1": 1},
            "expert_parameters": {
                "expert_0": {"weight": 2.0, "bias": 0.5},
                "expert_1": {"weight": -3.0, "bias": 1.0},
            },
            "top_k": 1,
            "combine_weight_policy": "preserve selected softmax probability",
            "loss": "global mean squared error over four scalar targets",
            "learning_rate": LEARNING_RATE,
            "optimizer": "SGD without momentum or weight decay",
            "capacity_or_drop_policy": "none; every assignment is dispatched",
            "metadata_columns": [
                "source_rank",
                "source_local_index",
                "global_token_id",
                "expert_id",
            ],
        },
        "process_observation": {
            "distinct_worker_process_count": len(
                {report["process_id"] for report in worker_reports}
            ),
            "raw_process_ids_published": False,
        },
        "worker_reports": [
            _publish_worker_report(report) for report in worker_reports
        ],
        "single_process_oracle": oracle,
        "comparison": {
            "distributed_outputs_before_step_by_global_token_id": outputs_before,
            "distributed_outputs_after_step_by_global_token_id": outputs_after,
            "distributed_hidden_gradients_by_global_token_id": hidden_gradients,
            "distributed_global_mean_loss_before_step": loss_before,
            "distributed_global_mean_loss_after_step": loss_after,
            "output_before_step_max_abs_difference": _max_abs_difference(
                outputs_before,
                oracle["outputs_before_step"],
            ),
            "output_after_step_max_abs_difference": _max_abs_difference(
                outputs_after,
                oracle["outputs_after_step"],
            ),
            "hidden_gradient_max_abs_difference": _max_abs_difference(
                hidden_gradients,
                oracle["hidden_gradients"],
            ),
            "router_gradient_max_abs_difference_by_rank": (
                router_gradient_differences
            ),
            "owned_expert_gradient_max_abs_difference_by_rank": (
                expert_gradient_differences
            ),
            "post_step_parameter_max_abs_difference_by_rank": (
                parameter_differences
            ),
        },
        "assertions": assertions,
        "scope": {
            "real_two_process_same_host_gloo_process_group_executed": True,
            "owner_only_expert_parameter_placement_executed": True,
            "variable_split_token_and_metadata_dispatch_return_executed": True,
            "authored_autograd_all_to_all_forward_backward_executed": True,
            "reverse_split_hidden_and_gate_gradient_communication_executed": True,
            "replicated_router_gradient_sum_all_reduce_executed": True,
            "owner_local_expert_parameter_gradients_executed": True,
            "one_sgd_optimizer_step_executed": True,
            "post_step_distributed_forward_evaluation_executed": True,
            "single_process_global_mean_mse_oracle_compared": True,
            "capacity_drop_reroute_or_dropless_executed": False,
            "pytorch_distributed_nn_functional_wrapper_executed": False,
            "torch_distributed_autograd_rpc_context_executed": False,
            "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
            "optimizer_momentum_weight_decay_or_state_resume_executed": False,
            "cuda_nccl_multi_node_or_remote_host_executed": False,
            "wire_bytes_protocol_overhead_or_collective_profiler_measured": False,
            "deepseek_qwen_or_other_checkpoint_reproduced": False,
            "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


def _capacity_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_directory: str,
) -> None:
    global _CAPACITY_ROUTE_ALL_GATHER_CALLS

    if world_size != WORLD_SIZE:
        raise ValueError(f"world_size must be {WORLD_SIZE}")
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        _reset_call_counts()
        _CAPACITY_ROUTE_ALL_GATHER_CALLS = 0
        hidden_states = _local_hidden(rank).requires_grad_(True)
        targets = _local_targets(rank)
        router = _new_router()
        owned_expert = OwnedExpert(rank)
        optimizer = torch.optim.SGD(
            [*router.parameters(), *owned_expert.parameters()],
            lr=LEARNING_RATE,
        )

        training = _capacity_distributed_forward(
            rank,
            hidden_states,
            router,
            owned_expert,
        )
        local_loss = functional.mse_loss(
            training.final_output,
            targets,
            reduction="sum",
        ) / GLOBAL_TOKEN_COUNT
        local_loss.backward()  # type: ignore[no-untyped-call]
        router_gradient = _require_gradient(router.weight, "router.weight")
        router_gradient_before_reduce = router_gradient.detach().clone()
        dist.all_reduce(router_gradient, op=dist.ReduceOp.SUM)
        _CALL_COUNTS["router_gradient_all_reduce"] += 1
        router_gradient_after_reduce = router_gradient.detach().clone()
        hidden_gradient = _require_gradient(hidden_states, "hidden_states")
        expert_weight_gradient = _require_gradient(
            owned_expert.linear.weight,
            "owned_expert.weight",
        )
        expert_bias_gradient = _require_gradient(
            owned_expert.linear.bias,
            "owned_expert.bias",
        )
        optimizer.step()

        with torch.no_grad():
            evaluation = _capacity_distributed_forward(
                rank,
                hidden_states.detach(),
                router,
                owned_expert,
            )
            evaluation_loss = functional.mse_loss(
                evaluation.final_output,
                targets,
                reduction="sum",
            ) / GLOBAL_TOKEN_COUNT
        if evaluation.global_keep_mask != training.global_keep_mask:
            raise AssertionError("one SGD step changed the authored capacity mask")

        offset = _global_token_offset(rank)
        collective_counts = dict(_CALL_COUNTS)
        collective_counts["capacity_route_all_gather"] = (
            _CAPACITY_ROUTE_ALL_GATHER_CALLS
        )
        payload: CapacityWorkerReport = {
            "rank": rank,
            "process_id": os.getpid(),
            "owned_expert_id": rank,
            "local_hidden_states": hidden_states.detach().tolist(),
            "local_targets": targets.tolist(),
            "local_global_token_ids": list(
                range(offset, offset + hidden_states.shape[0])
            ),
            "selected_expert_indices": (
                training.decision.selected_expert_indices.tolist()
            ),
            "selected_probabilities": (
                training.decision.selected_probabilities.detach().tolist()
            ),
            "global_keep_mask": training.global_keep_mask,
            "local_keep_mask": training.local_keep_mask.tolist(),
            "selected_counts_by_expert": training.selected_counts_by_expert,
            "kept_counts_by_expert": training.kept_counts_by_expert,
            "dropped_assignments": (
                GLOBAL_TOKEN_COUNT - sum(training.kept_counts_by_expert)
            ),
            "source_to_owner_counts": training.send_counts.tolist(),
            "owner_from_source_counts": training.received_counts.tolist(),
            "owner_received_metadata": training.received_metadata.tolist(),
            "return_arrival_metadata": training.returned_metadata.tolist(),
            "training_outputs": training.final_output.detach().tolist(),
            "evaluation_outputs_after_step": (
                evaluation.final_output.detach().tolist()
            ),
            "local_loss_contribution_before_step": float(local_loss.item()),
            "local_loss_contribution_after_step": float(evaluation_loss.item()),
            "local_hidden_gradients": hidden_gradient.detach().tolist(),
            "router_gradient_before_all_reduce": (
                router_gradient_before_reduce.tolist()
            ),
            "router_gradient_after_all_reduce": (
                router_gradient_after_reduce.tolist()
            ),
            "owned_expert_weight_gradient": (
                expert_weight_gradient.detach().tolist()
            ),
            "owned_expert_bias_gradient": (
                expert_bias_gradient.detach().tolist()
            ),
            "router_weight_after_step": router.weight.detach().tolist(),
            "owned_expert_weight_after_step": (
                owned_expert.linear.weight.detach().tolist()
            ),
            "owned_expert_bias_after_step": (
                owned_expert.linear.bias.detach().tolist()
            ),
            "authored_collective_call_counts": collective_counts,
        }
        (Path(output_directory) / f"rank-{rank}.json").write_bytes(
            _canonical_bytes(payload)
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _load_capacity_worker_report(path: Path) -> CapacityWorkerReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("capacity worker report must be a JSON object")
    return cast(CapacityWorkerReport, payload)


def _publish_capacity_worker_report(
    report: CapacityWorkerReport,
) -> PublishedCapacityWorkerReport:
    return {key: value for key, value in report.items() if key != "process_id"}  # type: ignore[return-value]


def _oracle_capacity_forward(
    router: nn.Linear,
    experts: list[OwnedExpert],
    hidden_states: Tensor,
) -> tuple[RouterDecision, list[bool], list[int], list[int], Tensor]:
    decision = _route(router, hidden_states)
    global_ids = list(range(GLOBAL_TOKEN_COUNT))
    expert_ids = decision.selected_expert_indices.detach().tolist()
    probabilities = decision.selected_probabilities.detach().tolist()
    keep_mask, selected_counts, kept_counts = _score_priority_capacity_mask(
        global_ids,
        expert_ids,
        probabilities,
    )
    keep_tensor = torch.tensor(keep_mask, dtype=torch.bool)
    output = torch.zeros_like(hidden_states)
    for expert_id, expert in enumerate(experts):
        indices = torch.nonzero(
            keep_tensor & (decision.selected_expert_indices == expert_id),
            as_tuple=False,
        ).squeeze(1)
        weighted = (
            expert(hidden_states[indices])
            * decision.selected_probabilities[indices].unsqueeze(1)
        )
        output = output.index_copy(0, indices, weighted)
    output = output + decision.selected_probabilities.sum() * 0.0
    return decision, keep_mask, selected_counts, kept_counts, output


def _single_process_capacity_oracle() -> dict[str, object]:
    hidden_states = torch.cat(
        [_local_hidden(rank) for rank in range(WORLD_SIZE)],
        dim=0,
    ).requires_grad_(True)
    targets = torch.cat(
        [_local_targets(rank) for rank in range(WORLD_SIZE)],
        dim=0,
    )
    router = _new_router()
    experts = [OwnedExpert(0), OwnedExpert(1)]
    expert_parameters = [
        parameter for expert in experts for parameter in expert.parameters()
    ]
    optimizer = torch.optim.SGD(
        [*router.parameters(), *expert_parameters],
        lr=LEARNING_RATE,
    )

    decision, keep_mask, selected_counts, kept_counts, outputs = (
        _oracle_capacity_forward(router, experts, hidden_states)
    )
    loss = functional.mse_loss(outputs, targets)
    loss.backward()  # type: ignore[no-untyped-call]
    router_gradient = _require_gradient(router.weight, "oracle.router.weight").detach()
    expert_weight_gradients = [
        _require_gradient(expert.linear.weight, f"oracle.expert_{index}.weight")
        .detach()
        .clone()
        for index, expert in enumerate(experts)
    ]
    expert_bias_gradients = [
        _require_gradient(expert.linear.bias, f"oracle.expert_{index}.bias")
        .detach()
        .clone()
        for index, expert in enumerate(experts)
    ]
    hidden_gradient = _require_gradient(
        hidden_states,
        "oracle.hidden_states",
    ).detach()
    optimizer.step()

    with torch.no_grad():
        (
            decision_after,
            keep_mask_after,
            selected_counts_after,
            kept_counts_after,
            outputs_after,
        ) = _oracle_capacity_forward(router, experts, hidden_states.detach())
        loss_after = functional.mse_loss(outputs_after, targets)
    if (
        keep_mask_after != keep_mask
        or selected_counts_after != selected_counts
        or kept_counts_after != kept_counts
    ):
        raise AssertionError("oracle capacity route changed after one step")

    return {
        "selected_expert_indices": decision.selected_expert_indices.tolist(),
        "selected_probabilities": decision.selected_probabilities.detach().tolist(),
        "global_keep_mask": keep_mask,
        "selected_counts_by_expert": selected_counts,
        "kept_counts_by_expert": kept_counts,
        "outputs_before_step": outputs.detach().tolist(),
        "outputs_after_step": outputs_after.tolist(),
        "loss_before_step": float(loss.item()),
        "loss_after_step": float(loss_after.item()),
        "hidden_gradients": hidden_gradient.tolist(),
        "router_gradient": router_gradient.tolist(),
        "expert_weight_gradients": [
            gradient.tolist() for gradient in expert_weight_gradients
        ],
        "expert_bias_gradients": [
            gradient.tolist() for gradient in expert_bias_gradients
        ],
        "router_weight_after_step": router.weight.detach().tolist(),
        "expert_weights_after_step": [
            expert.linear.weight.detach().tolist() for expert in experts
        ],
        "expert_biases_after_step": [
            expert.linear.bias.detach().tolist() for expert in experts
        ],
        "selected_expert_indices_after_step": (
            decision_after.selected_expert_indices.tolist()
        ),
    }


def _ordered_capacity_values(
    worker_reports: list[CapacityWorkerReport],
    field: str,
) -> list[list[float]]:
    by_global_id: dict[int, list[float]] = {}
    for report in worker_reports:
        values = cast(list[list[float]], report[field])  # type: ignore[literal-required]
        for global_id, value in zip(
            report["local_global_token_ids"],
            values,
            strict=True,
        ):
            by_global_id[global_id] = value
    return [by_global_id[index] for index in range(GLOBAL_TOKEN_COUNT)]


def run_moe_all_to_all_capacity_training_control() -> dict[str, object]:
    """Run global drop capacity with kept-only all-to-all backward and SGD."""

    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed with the Gloo backend is required")
    with tempfile.TemporaryDirectory(
        prefix="about-llm-moe-all-to-all-capacity-training-"
    ) as temporary:
        root = Path(temporary)
        init_method = (root / "rendezvous").resolve().as_uri()
        output_directory = root / "worker-results"
        output_directory.mkdir()
        spawn(  # type: ignore[no-untyped-call]
            _capacity_worker,
            args=(WORLD_SIZE, init_method, str(output_directory)),
            nprocs=WORLD_SIZE,
            join=True,
        )
        worker_reports = [
            _load_capacity_worker_report(output_directory / f"rank-{rank}.json")
            for rank in range(WORLD_SIZE)
        ]

    oracle = _single_process_capacity_oracle()
    outputs_before = _ordered_capacity_values(worker_reports, "training_outputs")
    outputs_after = _ordered_capacity_values(
        worker_reports,
        "evaluation_outputs_after_step",
    )
    hidden_gradients = _ordered_capacity_values(
        worker_reports,
        "local_hidden_gradients",
    )
    loss_before = sum(
        report["local_loss_contribution_before_step"]
        for report in worker_reports
    )
    loss_after = sum(
        report["local_loss_contribution_after_step"]
        for report in worker_reports
    )
    router_gradient_differences = [
        _max_abs_difference(
            report["router_gradient_after_all_reduce"],
            oracle["router_gradient"],
        )
        for report in worker_reports
    ]
    expert_gradient_differences = [
        max(
            _max_abs_difference(
                report["owned_expert_weight_gradient"],
                cast(list[object], oracle["expert_weight_gradients"])[rank],
            ),
            _max_abs_difference(
                report["owned_expert_bias_gradient"],
                cast(list[object], oracle["expert_bias_gradients"])[rank],
            ),
        )
        for rank, report in enumerate(worker_reports)
    ]
    parameter_differences = [
        max(
            _max_abs_difference(
                report["router_weight_after_step"],
                oracle["router_weight_after_step"],
            ),
            _max_abs_difference(
                report["owned_expert_weight_after_step"],
                cast(list[object], oracle["expert_weights_after_step"])[rank],
            ),
            _max_abs_difference(
                report["owned_expert_bias_after_step"],
                cast(list[object], oracle["expert_biases_after_step"])[rank],
            ),
        )
        for rank, report in enumerate(worker_reports)
    ]
    expected_call_counts = {
        "autograd_payload_forward_all_to_all_single": 4,
        "autograd_payload_backward_all_to_all_single": 2,
        "nondifferentiable_count_or_metadata_all_to_all_single": 6,
        "router_gradient_all_reduce": 1,
        "capacity_route_all_gather": 4,
    }
    assertions = {
        "two_distinct_worker_processes_executed": (
            len({report["process_id"] for report in worker_reports}) == WORLD_SIZE
        ),
        "global_score_priority_capacity_mask_matches": all(
            report["global_keep_mask"] == [False, True, True, False]
            and report["selected_counts_by_expert"] == [2, 2]
            and report["kept_counts_by_expert"] == [1, 1]
            and report["dropped_assignments"] == 2
            for report in worker_reports
        ),
        "kept_only_variable_splits_include_zero_token_source": (
            [report["source_to_owner_counts"] for report in worker_reports]
            == [[1, 1], [0, 0]]
            and [report["owner_from_source_counts"] for report in worker_reports]
            == [[1, 0], [1, 0]]
            and worker_reports[1]["return_arrival_metadata"] == []
        ),
        "distributed_forward_and_post_step_match_capacity_oracle": (
            _max_abs_difference(outputs_before, oracle["outputs_before_step"])
            <= 1e-15
            and _max_abs_difference(outputs_after, oracle["outputs_after_step"])
            <= 1e-15
        ),
        "dropped_task_gradients_are_zero_and_kept_gradients_match_oracle": (
            _max_abs_difference(hidden_gradients, oracle["hidden_gradients"])
            <= 1e-15
            and hidden_gradients[0] == [0.0]
            and hidden_gradients[3] == [0.0]
            and hidden_gradients[1] != [0.0]
            and hidden_gradients[2] != [0.0]
        ),
        "router_gradient_sum_and_owner_expert_gradients_match_oracle": (
            max(router_gradient_differences) <= 1e-15
            and max(expert_gradient_differences) <= 1e-15
            and worker_reports[1]["router_gradient_before_all_reduce"]
            == [[0.0], [0.0]]
        ),
        "one_step_parameters_and_global_mean_loss_match_oracle": (
            max(parameter_differences) <= 1e-15
            and math.isclose(
                loss_before,
                cast(float, oracle["loss_before_step"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            and math.isclose(
                loss_after,
                cast(float, oracle["loss_after_step"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            and loss_after < loss_before
        ),
        "authored_collective_call_ledger_matches_execution": all(
            report["authored_collective_call_counts"] == expected_call_counts
            for report in worker_reports
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(
            f"MoE capacity all-to-all training assertion failed: {assertions}"
        )

    report: dict[str, object] = {
        "schema_version": MOE_ALL_TO_ALL_CAPACITY_TRAINING_CONTROL_VERSION,
        "runtime": {
            "torch_version": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
            "dtype": "torch.float64",
            "world_size": WORLD_SIZE,
            "process_start_method": "spawn",
            "rendezvous": "temporary-file-store",
        },
        "fixture": {
            "local_hidden_states_by_rank": [
                _local_hidden(rank).tolist() for rank in range(WORLD_SIZE)
            ],
            "local_targets_by_rank": [
                _local_targets(rank).tolist() for rank in range(WORLD_SIZE)
            ],
            "router_weight": [[1.0], [-1.0]],
            "expert_ownership": {"expert_0": 0, "expert_1": 1},
            "expert_parameters": {
                "expert_0": {"weight": 2.0, "bias": 0.5},
                "expert_1": {"weight": -3.0, "bias": 1.0},
            },
            "top_k": 1,
            "capacity_factor": CAPACITY_FACTOR,
            "expert_capacity": EXPERT_CAPACITY,
            "capacity_group": "all four active tokens across both ranks",
            "overflow_policy": "drop by selected probability, then global token id",
            "combine_weight_policy": "preserve selected softmax probability",
            "loss": "global mean squared error over kept and dropped token outputs",
            "learning_rate": LEARNING_RATE,
            "optimizer": "SGD without momentum or weight decay",
        },
        "process_observation": {
            "distinct_worker_process_count": len(
                {report["process_id"] for report in worker_reports}
            ),
            "raw_process_ids_published": False,
        },
        "worker_reports": [
            _publish_capacity_worker_report(report) for report in worker_reports
        ],
        "single_process_oracle": oracle,
        "comparison": {
            "distributed_outputs_before_step_by_global_token_id": outputs_before,
            "distributed_outputs_after_step_by_global_token_id": outputs_after,
            "distributed_hidden_gradients_by_global_token_id": hidden_gradients,
            "distributed_global_mean_loss_before_step": loss_before,
            "distributed_global_mean_loss_after_step": loss_after,
            "output_before_step_max_abs_difference": _max_abs_difference(
                outputs_before,
                oracle["outputs_before_step"],
            ),
            "output_after_step_max_abs_difference": _max_abs_difference(
                outputs_after,
                oracle["outputs_after_step"],
            ),
            "hidden_gradient_max_abs_difference": _max_abs_difference(
                hidden_gradients,
                oracle["hidden_gradients"],
            ),
            "router_gradient_max_abs_difference_by_rank": (
                router_gradient_differences
            ),
            "owned_expert_gradient_max_abs_difference_by_rank": (
                expert_gradient_differences
            ),
            "post_step_parameter_max_abs_difference_by_rank": (
                parameter_differences
            ),
        },
        "assertions": assertions,
        "scope": {
            "real_two_process_same_host_gloo_process_group_executed": True,
            "global_score_priority_drop_capacity_collective_executed": True,
            "owner_only_expert_parameter_placement_executed": True,
            "kept_only_variable_split_dispatch_return_executed": True,
            "zero_assignment_source_rank_forward_backward_executed": True,
            "authored_autograd_reverse_all_to_all_backward_executed": True,
            "dropped_token_zero_output_and_task_gradient_executed": True,
            "replicated_router_gradient_sum_all_reduce_executed": True,
            "owner_local_expert_parameter_gradients_executed": True,
            "one_sgd_optimizer_step_executed": True,
            "post_step_distributed_capacity_forward_executed": True,
            "single_process_capacity_training_oracle_compared": True,
            "reroute_dropless_shared_or_fine_grained_experts_executed": False,
            "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
            "optimizer_momentum_weight_decay_or_state_resume_executed": False,
            "cuda_nccl_multi_node_or_remote_host_executed": False,
            "wire_bytes_protocol_overhead_or_collective_profiler_measured": False,
            "deepseek_qwen_or_other_checkpoint_reproduced": False,
            "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


__all__ = [
    "MOE_ALL_TO_ALL_CAPACITY_TRAINING_CONTROL_VERSION",
    "MOE_ALL_TO_ALL_TRAINING_CONTROL_VERSION",
    "run_moe_all_to_all_capacity_training_control",
    "run_moe_all_to_all_training_control",
]
