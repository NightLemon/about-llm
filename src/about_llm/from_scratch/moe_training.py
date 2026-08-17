"""Trainable top-k MoE reference with explicit router-gradient semantics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn

TRAINABLE_MOE_CONTROL_VERSION = "about-llm.trainable-moe-control.v3"
OverflowPolicy: TypeAlias = Literal["drop", "reroute", "dropless"]


@dataclass(frozen=True)
class TrainableMoEForward:
    """Observable tensors from one trainable top-k MoE forward pass."""

    output: Tensor
    router_logits: Tensor
    router_probabilities: Tensor
    ranked_expert_indices: Tensor
    selected_expert_indices: Tensor
    selected_probabilities: Tensor
    dispatched_expert_indices: Tensor
    dispatched_probabilities: Tensor
    pre_capacity_combine_weights: Tensor
    active_token_mask: Tensor
    routing_group_ids: Tensor
    routing_group_labels: tuple[int, ...]
    active_tokens_per_group: tuple[int, ...]
    kept_mask: Tensor
    combine_weights: Tensor
    expert_capacity: int | None
    expert_capacities_by_group: tuple[int, ...] | None
    expert_counts_before_capacity: Tensor
    expert_counts_after_capacity: Tensor
    expert_counts_before_capacity_by_group: Tensor
    expert_counts_after_capacity_by_group: Tensor
    pre_policy_capacity_excess_by_group: Tensor
    post_policy_capacity_excess_by_group: Tensor
    assignments_over_capacity_before_policy: int
    assignments_over_capacity_after_policy: int
    overflow_policy: OverflowPolicy
    rerouted_assignments: int
    dropped_assignments: int
    tokens_with_all_assignments_dropped: int
    renormalized_after_capacity: bool
    selection_fractions: Tensor
    mean_router_probabilities: Tensor
    selection_fractions_by_group: Tensor
    mean_router_probabilities_by_group: Tensor
    load_balance_loss: Tensor
    load_balance_loss_by_group: Tensor
    router_z_loss: Tensor
    router_z_loss_by_group: Tensor


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _score_priority_keep_mask(
    selected_expert_indices: Tensor,
    selected_probabilities: Tensor,
    *,
    active_token_mask: Tensor,
    routing_group_ids: Tensor,
    routing_group_labels: tuple[int, ...],
    expert_count: int,
    expert_capacities_by_group: tuple[int, ...] | None,
) -> Tensor:
    if expert_capacities_by_group is None:
        return (
            active_token_mask.unsqueeze(1)
            .expand_as(selected_expert_indices)
            .clone()
        )
    if len(routing_group_labels) != len(expert_capacities_by_group):
        raise AssertionError("routing group/capacity cardinality mismatch")
    kept_mask = torch.zeros_like(selected_expert_indices, dtype=torch.bool)
    detached_probabilities = selected_probabilities.detach()
    with torch.no_grad():
        for group_label, expert_capacity in zip(
            routing_group_labels,
            expert_capacities_by_group,
            strict=True,
        ):
            group_mask = active_token_mask & (routing_group_ids == group_label)
            for expert_id in range(expert_count):
                positions = torch.nonzero(
                    (selected_expert_indices == expert_id)
                    & group_mask.unsqueeze(1),
                    as_tuple=False,
                )
                assignments = [
                    (
                        float(detached_probabilities[token_index, rank].item()),
                        token_index,
                        rank,
                    )
                    for token_index, rank in (
                        (int(position[0]), int(position[1]))
                        for position in positions.cpu().tolist()
                    )
                ]
                assignments.sort(key=lambda item: (-item[0], item[1], item[2]))
                for _, token_index, rank in assignments[:expert_capacity]:
                    kept_mask[token_index, rank] = True
    return kept_mask


def _reroute_dropped_assignments(
    selected_expert_indices: Tensor,
    selected_probabilities: Tensor,
    ranked_expert_indices: Tensor,
    initial_kept_mask: Tensor,
    *,
    active_token_mask: Tensor,
    routing_group_ids: Tensor,
    routing_group_labels: tuple[int, ...],
    expert_count: int,
    expert_capacities_by_group: tuple[int, ...],
) -> tuple[Tensor, Tensor, int]:
    if len(routing_group_labels) != len(expert_capacities_by_group):
        raise AssertionError("routing group/capacity cardinality mismatch")
    dispatched_expert_indices = selected_expert_indices.clone()
    kept_mask = initial_kept_mask.clone()
    rerouted_assignments = 0
    detached_probabilities = selected_probabilities.detach()
    with torch.no_grad():
        for group_label, expert_capacity in zip(
            routing_group_labels,
            expert_capacities_by_group,
            strict=True,
        ):
            group_mask = active_token_mask & (routing_group_ids == group_label)
            current_counts = [0] * expert_count
            for token_index, rank in torch.nonzero(
                group_mask.unsqueeze(1) & kept_mask,
                as_tuple=False,
            ).cpu().tolist():
                expert_id = int(dispatched_expert_indices[token_index, rank].item())
                current_counts[expert_id] += 1
            dropped_positions = torch.nonzero(
                group_mask.unsqueeze(1) & ~kept_mask,
                as_tuple=False,
            )
            dropped_assignments = [
                (
                    float(detached_probabilities[token_index, rank].item()),
                    int(token_index),
                    int(rank),
                )
                for token_index, rank in dropped_positions.cpu().tolist()
            ]
            dropped_assignments.sort(
                key=lambda item: (-item[0], item[1], item[2])
            )
            for _, token_index, rank in dropped_assignments:
                occupied_experts = {
                    int(dispatched_expert_indices[token_index, other_rank].item())
                    for other_rank in range(selected_expert_indices.shape[1])
                    if bool(kept_mask[token_index, other_rank])
                }
                for candidate in ranked_expert_indices[token_index].cpu().tolist():
                    candidate_id = int(candidate)
                    if candidate_id in occupied_experts:
                        continue
                    if current_counts[candidate_id] >= expert_capacity:
                        continue
                    dispatched_expert_indices[token_index, rank] = candidate_id
                    kept_mask[token_index, rank] = True
                    current_counts[candidate_id] += 1
                    rerouted_assignments += 1
                    break
    return dispatched_expert_indices, kept_mask, rerouted_assignments


class TrainableTopKMoE(nn.Module):
    """Small top-k MoE with a differentiable selected-gate path.

    Expert indices are discrete. Gradients reach the router through the selected
    softmax probabilities, and through the optional balance/z-loss terms. The
    load-balance formula is the authored per-routing-group diagnostic
    ``E * sum(f_e * p_e)``: hard selection fractions ``f_e`` are detached while
    mean probabilities ``p_e`` remain differentiable. Multiple groups are
    aggregated by active-token count. It is not a universal MoE training
    contract. ``expert_capacity`` is populated only for a single active group;
    multi-group callers must inspect ``expert_capacities_by_group``.

    Overflow policies are explicit teaching contracts. ``drop`` preserves the
    initial top-k slots that win score-priority capacity. ``reroute`` processes
    dropped slots by original gate score/token/rank, scans the token's complete
    stable expert ranking, forbids duplicate experts for one token, and respects
    the same group capacity. ``dropless`` admits every initial top-k assignment
    and reports capacity excess instead of pretending the nominal limit held.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        output_dim: int,
        expert_count: int,
        top_k: int,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.d_model = _positive_int(d_model, name="d_model")
        self.hidden_dim = _positive_int(hidden_dim, name="hidden_dim")
        self.output_dim = _positive_int(output_dim, name="output_dim")
        self.expert_count = _positive_int(expert_count, name="expert_count")
        self.top_k = _positive_int(top_k, name="top_k")
        if self.top_k > self.expert_count:
            raise ValueError("top_k cannot exceed expert_count")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")

        self.router = nn.Linear(
            self.d_model,
            self.expert_count,
            bias=False,
            dtype=dtype,
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.d_model,
                        self.hidden_dim,
                        bias=False,
                        dtype=dtype,
                    ),
                    nn.Tanh(),
                    nn.Linear(
                        self.hidden_dim,
                        self.output_dim,
                        bias=False,
                        dtype=dtype,
                    ),
                )
                for _ in range(self.expert_count)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        detach_combine_weights: bool = False,
        dense_oracle: bool = False,
        capacity_factor: float | None = None,
        renormalize_after_capacity: bool = True,
        token_mask: Tensor | None = None,
        routing_group_ids: Tensor | None = None,
        overflow_policy: OverflowPolicy = "drop",
    ) -> TrainableMoEForward:
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.d_model:
            raise ValueError("hidden_states must have shape [tokens, d_model]")
        if hidden_states.shape[0] == 0:
            raise ValueError("hidden_states must contain at least one token")
        if not hidden_states.is_floating_point() or hidden_states.dtype != self.router.weight.dtype:
            raise ValueError("hidden_states dtype must match the router dtype")
        if hidden_states.device != self.router.weight.device:
            raise ValueError("hidden_states device must match the router device")
        if not bool(torch.isfinite(hidden_states).all()):
            raise ValueError("hidden_states must contain only finite values")
        if not isinstance(detach_combine_weights, bool):
            raise TypeError("detach_combine_weights must be boolean")
        if not isinstance(dense_oracle, bool):
            raise TypeError("dense_oracle must be boolean")
        capacity_factor_value: float | None = None
        if capacity_factor is not None:
            if isinstance(capacity_factor, bool) or not isinstance(
                capacity_factor,
                (int, float),
            ):
                raise ValueError("capacity_factor must be finite and positive")
            try:
                capacity_factor_value = float(capacity_factor)
            except OverflowError as error:
                raise ValueError(
                    "capacity_factor must be finite and positive"
                ) from error
            if not math.isfinite(capacity_factor_value) or capacity_factor_value <= 0:
                raise ValueError("capacity_factor must be finite and positive")
        if not isinstance(renormalize_after_capacity, bool):
            raise TypeError("renormalize_after_capacity must be boolean")
        if not isinstance(overflow_policy, str):
            raise TypeError("overflow_policy must be a string")
        if overflow_policy not in {"drop", "reroute", "dropless"}:
            raise ValueError("overflow_policy must be drop, reroute, or dropless")
        if capacity_factor_value is None and overflow_policy != "drop":
            raise ValueError("reroute and dropless policies require capacity_factor")

        token_count = hidden_states.shape[0]
        if token_mask is None:
            active_token_mask = torch.ones(
                token_count,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            if not isinstance(token_mask, Tensor):
                raise TypeError("token_mask must be a torch.Tensor or None")
            if token_mask.shape != (token_count,) or token_mask.dtype != torch.bool:
                raise ValueError("token_mask must have boolean shape [tokens]")
            if token_mask.device != hidden_states.device:
                raise ValueError("token_mask device must match hidden_states")
            active_token_mask = token_mask.clone()
        active_token_count = int(torch.count_nonzero(active_token_mask).item())
        if active_token_count == 0:
            raise ValueError("token_mask must select at least one active token")

        if routing_group_ids is None:
            effective_routing_group_ids = torch.zeros(
                token_count,
                dtype=torch.int64,
                device=hidden_states.device,
            )
        else:
            if not isinstance(routing_group_ids, Tensor):
                raise TypeError("routing_group_ids must be a torch.Tensor or None")
            if (
                routing_group_ids.shape != (token_count,)
                or routing_group_ids.dtype != torch.int64
            ):
                raise ValueError("routing_group_ids must have int64 shape [tokens]")
            if routing_group_ids.device != hidden_states.device:
                raise ValueError("routing_group_ids device must match hidden_states")
            effective_routing_group_ids = routing_group_ids.clone()
        routing_group_labels = tuple(
            int(value)
            for value in torch.unique(
                effective_routing_group_ids[active_token_mask],
                sorted=True,
            )
            .detach()
            .cpu()
            .tolist()
        )
        active_tokens_per_group = tuple(
            int(
                torch.count_nonzero(
                    active_token_mask & (effective_routing_group_ids == label)
                ).item()
            )
            for label in routing_group_labels
        )
        expert_capacities_by_group: tuple[int, ...] | None = None
        if capacity_factor_value is not None:
            raw_capacities = tuple(
                capacity_factor_value
                * active_tokens
                * self.top_k
                / self.expert_count
                for active_tokens in active_tokens_per_group
            )
            if not all(math.isfinite(value) for value in raw_capacities):
                raise ValueError("capacity_factor produces non-finite expert capacity")
            expert_capacities_by_group = tuple(
                math.ceil(value) for value in raw_capacities
            )
        expert_capacity = (
            expert_capacities_by_group[0]
            if expert_capacities_by_group is not None
            and len(expert_capacities_by_group) == 1
            else None
        )

        router_logits = self.router(hidden_states)
        router_probabilities = torch.softmax(router_logits, dim=-1)
        ranked_expert_indices = torch.argsort(
            router_probabilities,
            dim=-1,
            descending=True,
            stable=True,
        )
        selected_expert_indices = ranked_expert_indices[:, : self.top_k]
        selected_probabilities = torch.gather(
            router_probabilities,
            dim=1,
            index=selected_expert_indices,
        )
        pre_capacity_combine_weights = (
            selected_probabilities / selected_probabilities.sum(
                dim=1,
                keepdim=True,
            )
        )

        initial_kept_mask = _score_priority_keep_mask(
            selected_expert_indices,
            selected_probabilities,
            active_token_mask=active_token_mask,
            routing_group_ids=effective_routing_group_ids,
            routing_group_labels=routing_group_labels,
            expert_count=self.expert_count,
            expert_capacities_by_group=expert_capacities_by_group,
        )
        dispatched_expert_indices = selected_expert_indices.clone()
        rerouted_assignments = 0
        if overflow_policy == "reroute":
            if expert_capacities_by_group is None:
                raise AssertionError("reroute requires computed capacities")
            (
                dispatched_expert_indices,
                kept_mask,
                rerouted_assignments,
            ) = _reroute_dropped_assignments(
                selected_expert_indices,
                selected_probabilities,
                ranked_expert_indices,
                initial_kept_mask,
                active_token_mask=active_token_mask,
                routing_group_ids=effective_routing_group_ids,
                routing_group_labels=routing_group_labels,
                expert_count=self.expert_count,
                expert_capacities_by_group=expert_capacities_by_group,
            )
        elif overflow_policy == "dropless":
            kept_mask = (
                active_token_mask.unsqueeze(1)
                .expand_as(selected_expert_indices)
                .clone()
            )
        else:
            kept_mask = initial_kept_mask
        dispatched_probabilities = torch.gather(
            router_probabilities,
            dim=1,
            index=dispatched_expert_indices,
        )
        pre_policy_combine_weights = dispatched_probabilities / selected_probabilities.sum(
            dim=1,
            keepdim=True,
        )
        combine_weights = torch.where(
            kept_mask,
            pre_policy_combine_weights,
            torch.zeros_like(pre_policy_combine_weights),
        )
        if capacity_factor_value is not None and renormalize_after_capacity:
            kept_denominator = combine_weights.sum(dim=1, keepdim=True)
            combine_weights = combine_weights / torch.clamp_min(
                kept_denominator,
                torch.finfo(combine_weights.dtype).tiny,
            )
            combine_weights = torch.where(
                kept_denominator > 0,
                combine_weights,
                torch.zeros_like(combine_weights),
            )
        effective_weights = (
            combine_weights.detach() if detach_combine_weights else combine_weights
        )

        assignment_indicators = torch.nn.functional.one_hot(
            selected_expert_indices,
            num_classes=self.expert_count,
        )
        active_assignment_indicators = (
            assignment_indicators * active_token_mask[:, None, None]
        )
        assignment_counts = active_assignment_indicators.sum(dim=(0, 1))
        dispatched_assignment_indicators = torch.nn.functional.one_hot(
            dispatched_expert_indices,
            num_classes=self.expert_count,
        )
        kept_assignment_indicators = (
            dispatched_assignment_indicators * kept_mask.unsqueeze(-1)
        )
        kept_assignment_counts = kept_assignment_indicators.sum(dim=(0, 1))
        assignment_counts_by_group = torch.stack(
            [
                (
                    active_assignment_indicators
                    * (effective_routing_group_ids == label)[:, None, None]
                ).sum(dim=(0, 1))
                for label in routing_group_labels
            ]
        )
        kept_assignment_counts_by_group = torch.stack(
            [
                (
                    kept_assignment_indicators
                    * (effective_routing_group_ids == label)[:, None, None]
                ).sum(dim=(0, 1))
                for label in routing_group_labels
            ]
        )
        if expert_capacities_by_group is None:
            pre_policy_capacity_excess_by_group = torch.zeros_like(
                assignment_counts_by_group
            )
            post_policy_capacity_excess_by_group = torch.zeros_like(
                kept_assignment_counts_by_group
            )
        else:
            capacity_tensor = torch.tensor(
                expert_capacities_by_group,
                dtype=assignment_counts_by_group.dtype,
                device=assignment_counts_by_group.device,
            ).unsqueeze(1)
            pre_policy_capacity_excess_by_group = torch.clamp_min(
                assignment_counts_by_group - capacity_tensor,
                0,
            )
            post_policy_capacity_excess_by_group = torch.clamp_min(
                kept_assignment_counts_by_group - capacity_tensor,
                0,
            )
        assignments_over_capacity_before_policy = int(
            pre_policy_capacity_excess_by_group.sum().item()
        )
        assignments_over_capacity_after_policy = int(
            post_policy_capacity_excess_by_group.sum().item()
        )
        dropped_assignments = int(
            torch.count_nonzero(active_token_mask.unsqueeze(1) & ~kept_mask).item()
        )
        tokens_with_all_assignments_dropped = int(
            torch.count_nonzero(
                active_token_mask & ~torch.any(kept_mask, dim=1)
            ).item()
        )

        if dense_oracle:
            all_expert_outputs = torch.stack(
                [expert(hidden_states) for expert in self.experts],
                dim=1,
            )
            dense_gates = torch.zeros_like(router_probabilities).scatter_add(
                1,
                dispatched_expert_indices,
                effective_weights,
            )
            output = torch.sum(
                all_expert_outputs * dense_gates.unsqueeze(-1),
                dim=1,
            )
        else:
            output = hidden_states.new_zeros((token_count, self.output_dim))
            for expert_id, expert in enumerate(self.experts):
                token_indices, ranks = torch.where(
                    (dispatched_expert_indices == expert_id) & kept_mask
                )
                if token_indices.numel() == 0:
                    continue
                expert_output = expert(hidden_states[token_indices])
                weighted = expert_output * effective_weights[
                    token_indices,
                    ranks,
                ].unsqueeze(-1)
                output = output.index_add(0, token_indices, weighted)

        selection_fractions = assignment_counts.to(router_probabilities.dtype) / (
            active_token_count * self.top_k
        )
        selection_fractions = selection_fractions.detach()
        mean_router_probabilities = router_probabilities[active_token_mask].mean(dim=0)
        selection_fractions_by_group = torch.stack(
            [
                assignment_counts_by_group[index].to(router_probabilities.dtype)
                / (active_tokens * self.top_k)
                for index, active_tokens in enumerate(active_tokens_per_group)
            ]
        ).detach()
        mean_router_probabilities_by_group = torch.stack(
            [
                router_probabilities[
                    active_token_mask & (effective_routing_group_ids == label)
                ].mean(dim=0)
                for label in routing_group_labels
            ]
        )
        load_balance_loss_by_group = self.expert_count * torch.sum(
            selection_fractions_by_group * mean_router_probabilities_by_group,
            dim=1,
        )
        group_weights = torch.tensor(
            active_tokens_per_group,
            dtype=router_probabilities.dtype,
            device=router_probabilities.device,
        ) / active_token_count
        load_balance_loss = torch.sum(load_balance_loss_by_group * group_weights)
        router_z_values = torch.logsumexp(router_logits, dim=-1).square()
        router_z_loss_by_group = torch.stack(
            [
                router_z_values[
                    active_token_mask & (effective_routing_group_ids == label)
                ].mean()
                for label in routing_group_labels
            ]
        )
        router_z_loss = torch.sum(router_z_loss_by_group * group_weights)

        return TrainableMoEForward(
            output=output,
            router_logits=router_logits,
            router_probabilities=router_probabilities,
            ranked_expert_indices=ranked_expert_indices,
            selected_expert_indices=selected_expert_indices,
            selected_probabilities=selected_probabilities,
            dispatched_expert_indices=dispatched_expert_indices,
            dispatched_probabilities=dispatched_probabilities,
            pre_capacity_combine_weights=pre_capacity_combine_weights,
            active_token_mask=active_token_mask,
            routing_group_ids=effective_routing_group_ids,
            routing_group_labels=routing_group_labels,
            active_tokens_per_group=active_tokens_per_group,
            kept_mask=kept_mask,
            combine_weights=combine_weights,
            expert_capacity=expert_capacity,
            expert_capacities_by_group=expert_capacities_by_group,
            expert_counts_before_capacity=assignment_counts,
            expert_counts_after_capacity=kept_assignment_counts,
            expert_counts_before_capacity_by_group=assignment_counts_by_group,
            expert_counts_after_capacity_by_group=kept_assignment_counts_by_group,
            pre_policy_capacity_excess_by_group=pre_policy_capacity_excess_by_group,
            post_policy_capacity_excess_by_group=post_policy_capacity_excess_by_group,
            assignments_over_capacity_before_policy=(
                assignments_over_capacity_before_policy
            ),
            assignments_over_capacity_after_policy=(
                assignments_over_capacity_after_policy
            ),
            overflow_policy=overflow_policy,
            rerouted_assignments=rerouted_assignments,
            dropped_assignments=dropped_assignments,
            tokens_with_all_assignments_dropped=tokens_with_all_assignments_dropped,
            renormalized_after_capacity=(
                capacity_factor_value is not None and renormalize_after_capacity
            ),
            selection_fractions=selection_fractions,
            mean_router_probabilities=mean_router_probabilities,
            selection_fractions_by_group=selection_fractions_by_group,
            mean_router_probabilities_by_group=mean_router_probabilities_by_group,
            load_balance_loss=load_balance_loss,
            load_balance_loss_by_group=load_balance_loss_by_group,
            router_z_loss=router_z_loss,
            router_z_loss_by_group=router_z_loss_by_group,
        )


def _authored_model(*, top_k: int = 2) -> TrainableTopKMoE:
    model = TrainableTopKMoE(3, 4, 2, 3, top_k, dtype=torch.float64)
    with torch.no_grad():
        model.router.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.2, -0.3],
                    [-0.4, 0.8, 0.1],
                    [0.2, -0.5, 0.9],
                ],
                dtype=torch.float64,
            )
        )
        first_layers = (
            [[0.4, -0.2, 0.1], [0.1, 0.3, -0.4], [-0.3, 0.2, 0.5], [0.2, 0.1, 0.3]],
            [[-0.2, 0.5, 0.3], [0.6, -0.1, 0.2], [0.1, 0.4, -0.5], [-0.3, 0.2, 0.4]],
            [[0.3, 0.1, 0.6], [-0.4, 0.5, 0.2], [0.2, -0.3, 0.4], [0.5, 0.2, -0.1]],
        )
        second_layers = (
            [[0.5, -0.2, 0.3, 0.1], [-0.1, 0.4, 0.2, -0.3]],
            [[-0.3, 0.2, 0.5, 0.1], [0.4, 0.1, -0.2, 0.3]],
            [[0.2, 0.6, -0.1, 0.4], [-0.5, 0.2, 0.3, 0.1]],
        )
        for expert_id, expert in enumerate(model.experts):
            if not isinstance(expert, nn.Sequential):
                raise AssertionError("authored expert topology drifted")
            first = expert[0]
            second = expert[2]
            if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
                raise AssertionError("authored expert topology drifted")
            first.weight.copy_(
                torch.tensor(first_layers[expert_id], dtype=torch.float64)
            )
            second.weight.copy_(
                torch.tensor(second_layers[expert_id], dtype=torch.float64)
            )
    return model


def _fixture() -> tuple[Tensor, Tensor]:
    hidden = torch.tensor(
        [
            [1.0, 0.2, -0.4],
            [0.1, 1.2, 0.3],
            [-0.5, 0.4, 1.1],
            [0.8, -0.7, 0.6],
            [0.3, 0.9, -0.8],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.4, -0.2], [-0.1, 0.5], [0.6, 0.1], [0.2, -0.4], [-0.3, 0.3]],
        dtype=torch.float64,
    )
    return hidden, target


def _gradient_snapshot(model: nn.Module) -> dict[str, Tensor]:
    gradients: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"expected gradient for {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"gradient for {name} must be finite")
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def _gradient_snapshot_with_materialized_zeros(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in model.named_parameters()
    }


def _named_gradients_are_materialized_zero(
    parameters: dict[str, nn.Parameter],
    names: list[str],
) -> bool:
    for name in names:
        gradient = parameters[name].grad
        if gradient is None or bool(torch.count_nonzero(gradient) != 0):
            return False
    return True


def _max_tensor_difference(left: Tensor, right: Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def _max_mapping_difference(
    left: dict[str, Tensor],
    right: dict[str, Tensor],
) -> float:
    if left.keys() != right.keys():
        raise AssertionError("gradient mappings have different parameter names")
    return max(_max_tensor_difference(left[name], right[name]) for name in left)


def _parameter_group_delta(
    before: dict[str, Tensor],
    model: nn.Module,
    *,
    prefix: str,
) -> float:
    deltas = [
        _max_tensor_difference(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    ]
    if not deltas:
        raise AssertionError(f"parameter prefix {prefix!r} matched nothing")
    return max(deltas)


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


def run_trainable_moe_control() -> dict[str, object]:
    """Execute trainable routing, capacity, padding, and gradient controls."""

    hidden, target = _fixture()
    sparse_model = _authored_model()
    dense_model = copy.deepcopy(sparse_model)
    sparse = sparse_model(hidden)
    dense = dense_model(hidden, dense_oracle=True)
    sparse_task_loss = torch.nn.functional.mse_loss(sparse.output, target)
    dense_task_loss = torch.nn.functional.mse_loss(dense.output, target)
    sparse_total = (
        sparse_task_loss + 0.05 * sparse.load_balance_loss + 0.001 * sparse.router_z_loss
    )
    dense_total = (
        dense_task_loss + 0.05 * dense.load_balance_loss + 0.001 * dense.router_z_loss
    )
    sparse_total.backward()
    dense_total.backward()
    sparse_gradients = _gradient_snapshot(sparse_model)
    dense_gradients = _gradient_snapshot(dense_model)
    output_difference = _max_tensor_difference(sparse.output, dense.output)
    gradient_difference = _max_mapping_difference(
        sparse_gradients,
        dense_gradients,
    )

    capacity_sparse_model = _authored_model()
    capacity_dense_model = copy.deepcopy(capacity_sparse_model)
    capacity_sparse = capacity_sparse_model(hidden, capacity_factor=0.5)
    capacity_dense = capacity_dense_model(
        hidden,
        capacity_factor=0.5,
        dense_oracle=True,
    )
    capacity_sparse_loss = (
        torch.nn.functional.mse_loss(capacity_sparse.output, target)
        + 0.05 * capacity_sparse.load_balance_loss
        + 0.001 * capacity_sparse.router_z_loss
    )
    capacity_dense_loss = (
        torch.nn.functional.mse_loss(capacity_dense.output, target)
        + 0.05 * capacity_dense.load_balance_loss
        + 0.001 * capacity_dense.router_z_loss
    )
    capacity_sparse_loss.backward()
    capacity_dense_loss.backward()
    capacity_output_difference = _max_tensor_difference(
        capacity_sparse.output,
        capacity_dense.output,
    )
    capacity_gradient_difference = _max_mapping_difference(
        _gradient_snapshot(capacity_sparse_model),
        _gradient_snapshot(capacity_dense_model),
    )
    preserve_mass = _authored_model()(
        hidden,
        capacity_factor=0.5,
        renormalize_after_capacity=False,
    )
    capacity_policy_output_difference = _max_tensor_difference(
        capacity_sparse.output,
        preserve_mass.output,
    )

    all_drop_model = _authored_model()
    with torch.no_grad():
        all_drop_model.router.weight.zero_()
    all_drop_hidden = torch.eye(3, dtype=torch.float64)
    all_drop = all_drop_model(all_drop_hidden, capacity_factor=0.5)

    padding_mask = torch.tensor([True, True, True, True, False])
    routing_group_ids = torch.tensor([10, 10, 20, 20, 999], dtype=torch.int64)
    grouped_sparse_model = _authored_model()
    grouped_dense_model = copy.deepcopy(grouped_sparse_model)
    grouped_sparse = grouped_sparse_model(
        hidden,
        capacity_factor=0.5,
        token_mask=padding_mask,
        routing_group_ids=routing_group_ids,
    )
    grouped_dense = grouped_dense_model(
        hidden,
        capacity_factor=0.5,
        token_mask=padding_mask,
        routing_group_ids=routing_group_ids,
        dense_oracle=True,
    )
    grouped_sparse_loss = (
        torch.nn.functional.mse_loss(
            grouped_sparse.output[padding_mask],
            target[padding_mask],
        )
        + 0.05 * grouped_sparse.load_balance_loss
        + 0.001 * grouped_sparse.router_z_loss
    )
    grouped_dense_loss = (
        torch.nn.functional.mse_loss(
            grouped_dense.output[padding_mask],
            target[padding_mask],
        )
        + 0.05 * grouped_dense.load_balance_loss
        + 0.001 * grouped_dense.router_z_loss
    )
    grouped_sparse_loss.backward()
    grouped_dense_loss.backward()
    grouped_output_difference = _max_tensor_difference(
        grouped_sparse.output,
        grouped_dense.output,
    )
    grouped_gradient_difference = _max_mapping_difference(
        _gradient_snapshot(grouped_sparse_model),
        _gradient_snapshot(grouped_dense_model),
    )
    single_group_capacity = _authored_model()(
        hidden,
        capacity_factor=0.5,
        token_mask=padding_mask,
    )
    grouped_vs_single_group_output_difference = _max_tensor_difference(
        grouped_sparse.output,
        single_group_capacity.output,
    )

    padding_probe_hidden = hidden.clone().requires_grad_(True)
    padding_probe = _authored_model()(
        padding_probe_hidden,
        capacity_factor=0.5,
        token_mask=padding_mask,
        routing_group_ids=routing_group_ids,
    )
    (
        padding_probe.output.sum()
        + 0.05 * padding_probe.load_balance_loss
        + 0.001 * padding_probe.router_z_loss
    ).backward()
    if padding_probe_hidden.grad is None:
        raise AssertionError("padding gradient control did not populate hidden gradient")
    padding_hidden_gradient_max_abs = float(
        torch.max(torch.abs(padding_probe_hidden.grad[~padding_mask])).item()
    )

    mutated_padding_hidden = hidden.clone()
    mutated_padding_hidden[-1] = torch.tensor(
        [100.0, -100.0, 50.0],
        dtype=torch.float64,
    )
    mutated_routing_group_ids = routing_group_ids.clone()
    mutated_routing_group_ids[-1] = -999
    mutated_padding = _authored_model()(
        mutated_padding_hidden,
        capacity_factor=0.5,
        token_mask=padding_mask,
        routing_group_ids=mutated_routing_group_ids,
    )
    padding_mutation_active_output_difference = _max_tensor_difference(
        grouped_sparse.output[padding_mask],
        mutated_padding.output[padding_mask],
    )
    padding_mutation_balance_difference = abs(
        float(grouped_sparse.load_balance_loss.item())
        - float(mutated_padding.load_balance_loss.item())
    )
    padding_mutation_z_difference = abs(
        float(grouped_sparse.router_z_loss.item())
        - float(mutated_padding.router_z_loss.item())
    )

    overflow_hidden = torch.tensor(
        [[1.0, 0.0, 0.0]] * 4,
        dtype=torch.float64,
    )
    overflow_target = target[:4]
    overflow_drop = _authored_model(top_k=1)(
        overflow_hidden,
        capacity_factor=1.0,
        overflow_policy="drop",
    )
    reroute_sparse_model = _authored_model(top_k=1)
    reroute_dense_model = copy.deepcopy(reroute_sparse_model)
    overflow_reroute = reroute_sparse_model(
        overflow_hidden,
        capacity_factor=1.0,
        overflow_policy="reroute",
    )
    overflow_reroute_dense = reroute_dense_model(
        overflow_hidden,
        capacity_factor=1.0,
        overflow_policy="reroute",
        dense_oracle=True,
    )
    reroute_sparse_loss = (
        torch.nn.functional.mse_loss(overflow_reroute.output, overflow_target)
        + 0.05 * overflow_reroute.load_balance_loss
        + 0.001 * overflow_reroute.router_z_loss
    )
    reroute_dense_loss = (
        torch.nn.functional.mse_loss(overflow_reroute_dense.output, overflow_target)
        + 0.05 * overflow_reroute_dense.load_balance_loss
        + 0.001 * overflow_reroute_dense.router_z_loss
    )
    reroute_sparse_loss.backward()
    reroute_dense_loss.backward()
    reroute_output_difference = _max_tensor_difference(
        overflow_reroute.output,
        overflow_reroute_dense.output,
    )
    reroute_gradient_difference = _max_mapping_difference(
        _gradient_snapshot_with_materialized_zeros(reroute_sparse_model),
        _gradient_snapshot_with_materialized_zeros(reroute_dense_model),
    )
    reroute_missing_sparse_gradients = [
        name
        for name, parameter in reroute_sparse_model.named_parameters()
        if parameter.grad is None
    ]
    reroute_dense_parameters = dict(reroute_dense_model.named_parameters())
    reroute_dense_missing_correspond_to_zero = _named_gradients_are_materialized_zero(
        reroute_dense_parameters,
        reroute_missing_sparse_gradients,
    )

    dropless_sparse_model = _authored_model(top_k=1)
    dropless_dense_model = copy.deepcopy(dropless_sparse_model)
    overflow_dropless = dropless_sparse_model(
        overflow_hidden,
        capacity_factor=1.0,
        overflow_policy="dropless",
    )
    overflow_dropless_dense = dropless_dense_model(
        overflow_hidden,
        capacity_factor=1.0,
        overflow_policy="dropless",
        dense_oracle=True,
    )
    dropless_sparse_loss = (
        torch.nn.functional.mse_loss(overflow_dropless.output, overflow_target)
        + 0.05 * overflow_dropless.load_balance_loss
        + 0.001 * overflow_dropless.router_z_loss
    )
    dropless_dense_loss = (
        torch.nn.functional.mse_loss(overflow_dropless_dense.output, overflow_target)
        + 0.05 * overflow_dropless_dense.load_balance_loss
        + 0.001 * overflow_dropless_dense.router_z_loss
    )
    dropless_sparse_loss.backward()
    dropless_dense_loss.backward()
    dropless_output_difference = _max_tensor_difference(
        overflow_dropless.output,
        overflow_dropless_dense.output,
    )
    dropless_gradient_difference = _max_mapping_difference(
        _gradient_snapshot_with_materialized_zeros(dropless_sparse_model),
        _gradient_snapshot_with_materialized_zeros(dropless_dense_model),
    )
    dropless_missing_sparse_gradients = [
        name
        for name, parameter in dropless_sparse_model.named_parameters()
        if parameter.grad is None
    ]
    dropless_dense_parameters = dict(dropless_dense_model.named_parameters())
    dropless_dense_missing_correspond_to_zero = _named_gradients_are_materialized_zero(
        dropless_dense_parameters,
        dropless_missing_sparse_gradients,
    )

    overflow_reroute_preserved_mass = _authored_model(top_k=1)(
        overflow_hidden,
        capacity_factor=1.0,
        overflow_policy="reroute",
        renormalize_after_capacity=False,
    )
    reroute_renormalize_vs_preserve_output_difference = _max_tensor_difference(
        overflow_reroute.output,
        overflow_reroute_preserved_mass.output,
    )
    drop_vs_reroute_output_difference = _max_tensor_difference(
        overflow_drop.output,
        overflow_reroute.output,
    )
    reroute_vs_dropless_output_difference = _max_tensor_difference(
        overflow_reroute.output,
        overflow_dropless.output,
    )

    trained_model = _authored_model()
    before = {
        name: parameter.detach().clone()
        for name, parameter in trained_model.named_parameters()
    }
    optimizer = torch.optim.SGD(trained_model.parameters(), lr=0.05)
    trained_before = trained_model(hidden)
    train_loss_before = torch.nn.functional.mse_loss(trained_before.output, target)
    train_total = (
        train_loss_before
        + 0.05 * trained_before.load_balance_loss
        + 0.001 * trained_before.router_z_loss
    )
    train_total.backward()
    train_gradients = _gradient_snapshot(trained_model)
    optimizer.step()
    trained_after = trained_model(hidden)
    train_loss_after = torch.nn.functional.mse_loss(trained_after.output, target)
    router_delta = _parameter_group_delta(before, trained_model, prefix="router.")
    expert_deltas = [
        _parameter_group_delta(before, trained_model, prefix=f"experts.{expert_id}.")
        for expert_id in range(trained_model.expert_count)
    ]

    detached_model = _authored_model()
    detached = detached_model(hidden, detach_combine_weights=True)
    detached_loss: Tensor = torch.nn.functional.mse_loss(detached.output, target)
    torch.autograd.backward(detached_loss)
    detached_router_gradient_missing = detached_model.router.weight.grad is None
    detached_expert_gradient_norms = [
        math.sqrt(
            sum(
                float(parameter.grad.detach().square().sum().item())
                for parameter in expert.parameters()
                if parameter.grad is not None
            )
        )
        for expert in detached_model.experts
    ]

    balance_model = _authored_model(top_k=1)
    with torch.no_grad():
        balance_model.router.weight.copy_(
            torch.tensor(
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
                dtype=torch.float64,
            )
        )
    positive_hidden = torch.tensor(
        [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [1.0, 1.0, 0.25], [0.25, 0.5, 1.0]],
        dtype=torch.float64,
    )
    balance_optimizer = torch.optim.SGD(balance_model.router.parameters(), lr=0.05)
    balance_before = balance_model(positive_hidden)
    balance_optimizer.zero_grad(set_to_none=True)
    balance_before.load_balance_loss.backward()
    balance_gradient = balance_model.router.weight.grad
    if balance_gradient is None or not bool(torch.isfinite(balance_gradient).all()):
        raise AssertionError("balance loss must create a finite router gradient")
    balance_gradient_norm = float(torch.linalg.vector_norm(balance_gradient).item())
    balance_optimizer.step()
    balance_after = balance_model(positive_hidden)

    assignment_counts = torch.nn.functional.one_hot(
        sparse.selected_expert_indices,
        num_classes=sparse_model.expert_count,
    ).sum(dim=(0, 1))
    router_task_gradient_norm = float(
        torch.linalg.vector_norm(train_gradients["router.weight"]).item()
    )
    expert_gradient_norms = [
        math.sqrt(
            sum(
                float(gradient.square().sum().item())
                for name, gradient in train_gradients.items()
                if name.startswith(f"experts.{expert_id}.")
            )
        )
        for expert_id in range(trained_model.expert_count)
    ]
    assertions = {
        "sparse_dense_selected_experts_match": bool(
            torch.equal(
                sparse.selected_expert_indices,
                dense.selected_expert_indices,
            )
        ),
        "sparse_dense_forward_matches": output_difference <= 1e-15,
        "sparse_dense_backward_matches": gradient_difference <= 1e-15,
        "capacity_sparse_dense_forward_matches": (
            capacity_output_difference <= 1e-15
        ),
        "capacity_sparse_dense_backward_matches": (
            capacity_gradient_difference <= 1e-15
        ),
        "score_priority_capacity_counts_and_drop_match_contract": (
            capacity_sparse.expert_capacity == 2
            and capacity_sparse.expert_counts_before_capacity.tolist() == [4, 3, 3]
            and capacity_sparse.expert_counts_after_capacity.tolist() == [2, 2, 2]
            and capacity_sparse.dropped_assignments == 4
            and capacity_sparse.tokens_with_all_assignments_dropped == 0
        ),
        "post_drop_renormalization_policy_changes_mass_and_output": (
            torch.equal(capacity_sparse.kept_mask, preserve_mass.kept_mask)
            and torch.allclose(
                capacity_sparse.combine_weights.sum(dim=1),
                torch.ones(hidden.shape[0], dtype=torch.float64),
                rtol=0,
                atol=0,
            )
            and bool(torch.all(preserve_mass.combine_weights.sum(dim=1) <= 1))
            and bool(torch.any(preserve_mass.combine_weights.sum(dim=1) < 1))
            and capacity_policy_output_difference > 0
        ),
        "all_assignments_dropped_tokens_have_zero_routed_output": (
            all_drop.expert_capacity == 1
            and all_drop.kept_mask.tolist()
            == [[True, True], [False, False], [False, False]]
            and all_drop.tokens_with_all_assignments_dropped == 2
            and bool(torch.count_nonzero(all_drop.output[1:]) == 0)
        ),
        "grouped_padding_sparse_dense_forward_matches": (
            grouped_output_difference <= 1e-15
        ),
        "grouped_padding_sparse_dense_backward_matches": (
            grouped_gradient_difference <= 1e-15
        ),
        "routing_group_scoped_capacity_matches_contract": (
            grouped_sparse.routing_group_labels == (10, 20)
            and grouped_sparse.active_tokens_per_group == (2, 2)
            and grouped_sparse.expert_capacity is None
            and grouped_sparse.expert_capacities_by_group == (1, 1)
            and grouped_sparse.expert_counts_before_capacity.tolist() == [3, 2, 3]
            and grouped_sparse.expert_counts_after_capacity.tolist() == [2, 2, 2]
            and grouped_sparse.expert_counts_before_capacity_by_group.tolist()
            == [[2, 1, 1], [1, 1, 2]]
            and grouped_sparse.expert_counts_after_capacity_by_group.tolist()
            == [[1, 1, 1], [1, 1, 1]]
            and grouped_sparse.kept_mask.tolist()
            == [
                [True, True],
                [True, False],
                [False, True],
                [True, True],
                [False, False],
            ]
            and grouped_sparse.dropped_assignments == 2
            and grouped_sparse.tokens_with_all_assignments_dropped == 0
        ),
        "routing_group_changes_competition_relative_to_single_group": (
            single_group_capacity.expert_capacity == 2
            and single_group_capacity.kept_mask.tolist()
            == [
                [True, False],
                [True, False],
                [True, True],
                [True, True],
                [False, False],
            ]
            and grouped_vs_single_group_output_difference > 0
        ),
        "padding_is_excluded_from_routing_aux_and_gradient": (
            bool(torch.count_nonzero(grouped_sparse.output[~padding_mask]) == 0)
            and padding_hidden_gradient_max_abs == 0.0
            and padding_mutation_active_output_difference == 0.0
            and padding_mutation_balance_difference == 0.0
            and padding_mutation_z_difference == 0.0
            and mutated_padding.routing_group_labels == (10, 20)
        ),
        "deterministic_reroute_reassigns_overflow_within_capacity": (
            overflow_reroute.ranked_expert_indices.tolist()
            == [[0, 2, 1], [0, 2, 1], [0, 2, 1], [0, 2, 1]]
            and overflow_reroute.selected_expert_indices.tolist()
            == [[0], [0], [0], [0]]
            and overflow_reroute.dispatched_expert_indices.tolist()
            == [[0], [0], [2], [2]]
            and overflow_reroute.expert_counts_before_capacity.tolist() == [4, 0, 0]
            and overflow_reroute.expert_counts_after_capacity.tolist() == [2, 0, 2]
            and overflow_reroute.rerouted_assignments == 2
            and overflow_reroute.dropped_assignments == 0
            and overflow_reroute.assignments_over_capacity_before_policy == 2
            and overflow_reroute.assignments_over_capacity_after_policy == 0
        ),
        "dropless_admits_overflow_and_reports_capacity_excess": (
            overflow_dropless.dispatched_expert_indices.tolist()
            == [[0], [0], [0], [0]]
            and overflow_dropless.expert_counts_after_capacity.tolist() == [4, 0, 0]
            and overflow_dropless.rerouted_assignments == 0
            and overflow_dropless.dropped_assignments == 0
            and overflow_dropless.post_policy_capacity_excess_by_group.tolist()
            == [[2, 0, 0]]
            and overflow_dropless.assignments_over_capacity_after_policy == 2
        ),
        "overflow_policy_sparse_dense_forward_backward_matches": (
            reroute_output_difference <= 1e-15
            and reroute_gradient_difference <= 1e-15
            and dropless_output_difference <= 1e-15
            and dropless_gradient_difference <= 1e-15
            and reroute_dense_missing_correspond_to_zero
            and dropless_dense_missing_correspond_to_zero
        ),
        "reroute_post_policy_weight_contract_changes_mass_and_output": (
            overflow_reroute.combine_weights.sum(dim=1).tolist()
            == [1.0, 1.0, 1.0, 1.0]
            and torch.allclose(
                overflow_reroute_preserved_mass.combine_weights.sum(dim=1),
                torch.tensor(
                    [1.0, 1.0, 0.44932896411722156, 0.44932896411722156],
                    dtype=torch.float64,
                ),
                rtol=0,
                atol=1e-15,
            )
            and reroute_renormalize_vs_preserve_output_difference > 0
        ),
        "router_and_all_experts_receive_finite_gradients": (
            router_task_gradient_norm > 0
            and all(value > 0 for value in expert_gradient_norms)
        ),
        "real_optimizer_step_updates_router_and_all_experts": (
            router_delta > 0 and all(value > 0 for value in expert_deltas)
        ),
        "detached_gate_blocks_task_gradient_to_router_only": (
            detached_router_gradient_missing
            and all(value > 0 for value in detached_expert_gradient_norms)
        ),
        "balance_loss_pushes_down_collapsed_diagnostic_before_route_changes": (
            torch.equal(
                balance_before.selected_expert_indices,
                balance_after.selected_expert_indices,
            )
            and float(balance_after.load_balance_loss.item())
            < float(balance_before.load_balance_loss.item())
            and balance_gradient_norm > 0
        ),
        "one_step_authored_task_loss_decreases": (
            float(train_loss_after.item()) < float(train_loss_before.item())
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"trainable MoE control assertion failed: {assertions}")

    report: dict[str, object] = {
        "schema_version": TRAINABLE_MOE_CONTROL_VERSION,
        "runtime": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "torch.float64",
            "router": "bias-free Linear(3,3)",
            "experts": "3 x bias-free Linear(3,4)-Tanh-Linear(4,2)",
            "top_k": 2,
            "tie_break": "stable probability descending, expert id ascending",
            "combine": "selected probabilities renormalized across top-k",
            "capacity": (
                "optional per-routing-group ceil(factor * active_tokens * top_k "
                "/ experts), probability/token/rank priority"
            ),
            "padding": "excluded from routing, capacity, auxiliary losses, and output",
            "routing_groups": "explicit CPU-local int64 labels; no distributed collective",
            "overflow_policies": (
                "drop; deterministic full-ranking reroute without per-token duplicate; "
                "dropless admission with explicit nominal-capacity excess"
            ),
            "post_drop_policy": "explicit renormalize or preserve lost top-k mass",
        },
        "fixture": {
            "token_count": int(hidden.shape[0]),
            "assignment_counts": [int(value) for value in assignment_counts.tolist()],
            "selected_expert_indices": sparse.selected_expert_indices.tolist(),
            "balance_formula": (
                "per group E * sum(stop_gradient(f_e) * mean_probability_e), "
                "active-token-weighted across groups"
            ),
            "z_loss_formula": (
                "per group mean(square(logsumexp(router_logits))), "
                "active-token-weighted across groups"
            ),
            "task_loss": "mean squared error over 2D authored targets",
            "total_loss": "task + 0.05 * balance + 0.001 * z",
        },
        "sparse_dense_oracle": {
            "output_max_abs_difference": output_difference,
            "all_parameter_gradient_max_abs_difference": gradient_difference,
            "task_loss": float(sparse_task_loss.item()),
            "load_balance_loss": float(sparse.load_balance_loss.item()),
            "router_z_loss": float(sparse.router_z_loss.item()),
        },
        "capacity_training_control": {
            "capacity_factor": 0.5,
            "expert_capacity": capacity_sparse.expert_capacity,
            "expert_counts_before_capacity": (
                capacity_sparse.expert_counts_before_capacity.tolist()
            ),
            "expert_counts_after_capacity": (
                capacity_sparse.expert_counts_after_capacity.tolist()
            ),
            "kept_mask": capacity_sparse.kept_mask.tolist(),
            "dropped_assignments": capacity_sparse.dropped_assignments,
            "tokens_with_all_assignments_dropped": (
                capacity_sparse.tokens_with_all_assignments_dropped
            ),
            "sparse_dense_output_max_abs_difference": (
                capacity_output_difference
            ),
            "sparse_dense_all_parameter_gradient_max_abs_difference": (
                capacity_gradient_difference
            ),
            "renormalized_weight_sums": (
                capacity_sparse.combine_weights.sum(dim=1).tolist()
            ),
            "preserved_mass_weight_sums": (
                preserve_mass.combine_weights.sum(dim=1).tolist()
            ),
            "renormalize_vs_preserve_output_max_abs_difference": (
                capacity_policy_output_difference
            ),
            "all_drop_fixture": {
                "expert_capacity": all_drop.expert_capacity,
                "selected_expert_indices": all_drop.selected_expert_indices.tolist(),
                "kept_mask": all_drop.kept_mask.tolist(),
                "expert_counts_before_capacity": (
                    all_drop.expert_counts_before_capacity.tolist()
                ),
                "expert_counts_after_capacity": (
                    all_drop.expert_counts_after_capacity.tolist()
                ),
                "tokens_with_all_assignments_dropped": (
                    all_drop.tokens_with_all_assignments_dropped
                ),
                "routed_output": all_drop.output.detach().tolist(),
            },
        },
        "routing_group_padding_control": {
            "token_mask": grouped_sparse.active_token_mask.tolist(),
            "routing_group_ids": grouped_sparse.routing_group_ids.tolist(),
            "routing_group_labels": list(grouped_sparse.routing_group_labels),
            "active_tokens_per_group": list(grouped_sparse.active_tokens_per_group),
            "capacity_factor": 0.5,
            "expert_capacity": grouped_sparse.expert_capacity,
            "expert_capacities_by_group": list(
                grouped_sparse.expert_capacities_by_group or ()
            ),
            "expert_counts_before_capacity": (
                grouped_sparse.expert_counts_before_capacity.tolist()
            ),
            "expert_counts_after_capacity": (
                grouped_sparse.expert_counts_after_capacity.tolist()
            ),
            "expert_counts_before_capacity_by_group": (
                grouped_sparse.expert_counts_before_capacity_by_group.tolist()
            ),
            "expert_counts_after_capacity_by_group": (
                grouped_sparse.expert_counts_after_capacity_by_group.tolist()
            ),
            "kept_mask": grouped_sparse.kept_mask.tolist(),
            "dropped_assignments": grouped_sparse.dropped_assignments,
            "tokens_with_all_assignments_dropped": (
                grouped_sparse.tokens_with_all_assignments_dropped
            ),
            "selection_fractions_by_group": (
                grouped_sparse.selection_fractions_by_group.tolist()
            ),
            "mean_router_probabilities_by_group": (
                grouped_sparse.mean_router_probabilities_by_group.tolist()
            ),
            "load_balance_loss_by_group": (
                grouped_sparse.load_balance_loss_by_group.tolist()
            ),
            "router_z_loss_by_group": grouped_sparse.router_z_loss_by_group.tolist(),
            "active_token_weighted_load_balance_loss": float(
                grouped_sparse.load_balance_loss.item()
            ),
            "active_token_weighted_router_z_loss": float(
                grouped_sparse.router_z_loss.item()
            ),
            "sparse_dense_output_max_abs_difference": grouped_output_difference,
            "sparse_dense_all_parameter_gradient_max_abs_difference": (
                grouped_gradient_difference
            ),
            "single_group_expert_capacity": single_group_capacity.expert_capacity,
            "single_group_kept_mask": single_group_capacity.kept_mask.tolist(),
            "grouped_vs_single_group_output_max_abs_difference": (
                grouped_vs_single_group_output_difference
            ),
            "padding_routed_output": grouped_sparse.output[~padding_mask].detach().tolist(),
            "padding_hidden_gradient_max_abs": padding_hidden_gradient_max_abs,
            "padding_value_and_group_id_mutation_active_output_max_abs_difference": (
                padding_mutation_active_output_difference
            ),
            "padding_value_and_group_id_mutation_balance_abs_difference": (
                padding_mutation_balance_difference
            ),
            "padding_value_and_group_id_mutation_z_abs_difference": (
                padding_mutation_z_difference
            ),
        },
        "overflow_policy_control": {
            "hidden_states": overflow_hidden.tolist(),
            "capacity_factor": 1.0,
            "top_k": 1,
            "expert_capacity": overflow_drop.expert_capacity,
            "ranked_expert_indices": overflow_drop.ranked_expert_indices.tolist(),
            "selected_expert_indices": overflow_drop.selected_expert_indices.tolist(),
            "selected_probabilities": overflow_drop.selected_probabilities.tolist(),
            "drop": {
                "dispatched_expert_indices": (
                    overflow_drop.dispatched_expert_indices.tolist()
                ),
                "expert_counts_before_capacity": (
                    overflow_drop.expert_counts_before_capacity.tolist()
                ),
                "expert_counts_after_capacity": (
                    overflow_drop.expert_counts_after_capacity.tolist()
                ),
                "pre_policy_capacity_excess_by_group": (
                    overflow_drop.pre_policy_capacity_excess_by_group.tolist()
                ),
                "post_policy_capacity_excess_by_group": (
                    overflow_drop.post_policy_capacity_excess_by_group.tolist()
                ),
                "rerouted_assignments": overflow_drop.rerouted_assignments,
                "dropped_assignments": overflow_drop.dropped_assignments,
                "combine_weight_sums": overflow_drop.combine_weights.sum(dim=1).tolist(),
                "output": overflow_drop.output.detach().tolist(),
            },
            "reroute": {
                "dispatched_expert_indices": (
                    overflow_reroute.dispatched_expert_indices.tolist()
                ),
                "dispatched_probabilities": (
                    overflow_reroute.dispatched_probabilities.tolist()
                ),
                "expert_counts_after_capacity": (
                    overflow_reroute.expert_counts_after_capacity.tolist()
                ),
                "post_policy_capacity_excess_by_group": (
                    overflow_reroute.post_policy_capacity_excess_by_group.tolist()
                ),
                "rerouted_assignments": overflow_reroute.rerouted_assignments,
                "dropped_assignments": overflow_reroute.dropped_assignments,
                "renormalized_weight_sums": (
                    overflow_reroute.combine_weights.sum(dim=1).tolist()
                ),
                "preserved_mass_weight_sums": (
                    overflow_reroute_preserved_mass.combine_weights.sum(dim=1).tolist()
                ),
                "renormalize_vs_preserve_output_max_abs_difference": (
                    reroute_renormalize_vs_preserve_output_difference
                ),
                "sparse_dense_output_max_abs_difference": reroute_output_difference,
                "sparse_dense_materialized_zero_gradient_max_abs_difference": (
                    reroute_gradient_difference
                ),
                "sparse_parameters_with_missing_zero_gradient": (
                    reroute_missing_sparse_gradients
                ),
                "dense_corresponding_gradients_are_zero": (
                    reroute_dense_missing_correspond_to_zero
                ),
                "output": overflow_reroute.output.detach().tolist(),
            },
            "dropless": {
                "dispatched_expert_indices": (
                    overflow_dropless.dispatched_expert_indices.tolist()
                ),
                "expert_counts_after_capacity": (
                    overflow_dropless.expert_counts_after_capacity.tolist()
                ),
                "post_policy_capacity_excess_by_group": (
                    overflow_dropless.post_policy_capacity_excess_by_group.tolist()
                ),
                "rerouted_assignments": overflow_dropless.rerouted_assignments,
                "dropped_assignments": overflow_dropless.dropped_assignments,
                "sparse_dense_output_max_abs_difference": dropless_output_difference,
                "sparse_dense_materialized_zero_gradient_max_abs_difference": (
                    dropless_gradient_difference
                ),
                "sparse_parameters_with_missing_zero_gradient": (
                    dropless_missing_sparse_gradients
                ),
                "dense_corresponding_gradients_are_zero": (
                    dropless_dense_missing_correspond_to_zero
                ),
                "output": overflow_dropless.output.detach().tolist(),
            },
            "drop_vs_reroute_output_max_abs_difference": (
                drop_vs_reroute_output_difference
            ),
            "reroute_vs_dropless_output_max_abs_difference": (
                reroute_vs_dropless_output_difference
            ),
        },
        "optimizer_step": {
            "task_loss_before": float(train_loss_before.item()),
            "task_loss_after": float(train_loss_after.item()),
            "router_task_plus_aux_gradient_norm": router_task_gradient_norm,
            "expert_task_plus_aux_gradient_norms": expert_gradient_norms,
            "router_parameter_max_abs_delta": router_delta,
            "expert_parameter_max_abs_deltas": expert_deltas,
        },
        "detached_gate_negative_control": {
            "router_task_gradient_is_missing": detached_router_gradient_missing,
            "expert_task_gradient_norms": detached_expert_gradient_norms,
            "selected_expert_indices": detached.selected_expert_indices.tolist(),
        },
        "balance_gradient_control": {
            "selected_expert_indices_before": (
                balance_before.selected_expert_indices.tolist()
            ),
            "selected_expert_indices_after": balance_after.selected_expert_indices.tolist(),
            "load_balance_loss_before": float(balance_before.load_balance_loss.item()),
            "load_balance_loss_after": float(balance_after.load_balance_loss.item()),
            "router_gradient_norm": balance_gradient_norm,
        },
        "assertions": assertions,
        "scope": {
            "trainable_router_and_expert_mlp_forward_backward_executed": True,
            "sparse_dispatch_dense_oracle_forward_backward_compared": True,
            "hard_topk_indices_treated_as_nondifferentiable": True,
            "selected_probability_task_gradient_to_router_executed": True,
            "detached_gate_missing_router_task_gradient_negative_control_executed": True,
            "authored_balance_and_z_loss_gradients_executed": True,
            "score_priority_capacity_drop_in_training_graph_executed": True,
            "post_drop_renormalize_and_preserve_mass_policies_executed": True,
            "all_assignments_dropped_zero_routed_output_executed": True,
            "padding_aware_capacity_aux_and_gradient_executed": True,
            "routing_group_scoped_capacity_and_aux_executed": True,
            "deterministic_full_ranking_reroute_policy_executed": True,
            "dropless_nominal_capacity_excess_policy_executed": True,
            "distributed_capacity_group_collective_executed": False,
            "shared_or_fine_grained_experts_executed": False,
            "expert_parallel_all_to_all_grouped_gemm_or_gpu_executed": False,
            "deepseek_qwen_or_other_checkpoint_reproduced": False,
            "convergence_quality_throughput_memory_or_scaling_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


__all__ = [
    "TRAINABLE_MOE_CONTROL_VERSION",
    "TrainableMoEForward",
    "TrainableTopKMoE",
    "run_trainable_moe_control",
]
