"""Transparent top-k MoE routing, capacity, and sparse linear dispatch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np


def _frozen_copy(value: np.ndarray, *, dtype: np.dtype, name: str) -> np.ndarray:
    array = cast(np.ndarray, np.asarray(value, dtype=dtype).copy())
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MoERoutingResult:
    """Materialized routing decisions for a single token group.

    ``selected_expert_indices`` uses ``-1`` for inactive/padding tokens. The
    capacity policy is score-priority within each expert, with token index and
    top-k rank as deterministic tie-breakers.
    """

    probabilities: np.ndarray
    active_token_mask: np.ndarray
    selected_expert_indices: np.ndarray
    selected_probabilities: np.ndarray
    kept_mask: np.ndarray
    combine_weights: np.ndarray
    expert_capacity: int
    assignments_before_capacity: int
    kept_assignments: int
    dropped_assignments: int
    tokens_with_all_assignments_dropped: int
    expert_counts_before_capacity: tuple[int, ...]
    expert_counts_after_capacity: tuple[int, ...]
    mean_router_probabilities: tuple[float, ...]
    selection_fractions: tuple[float, ...]
    load_balance_diagnostic: float
    router_z_loss: float
    mean_router_entropy: float
    renormalized_after_capacity: bool

    def __post_init__(self) -> None:
        probabilities = _frozen_copy(
            self.probabilities, dtype=np.dtype(np.float64), name="probabilities"
        )
        active = _frozen_copy(
            self.active_token_mask,
            dtype=np.dtype(np.bool_),
            name="active_token_mask",
        )
        selected = _frozen_copy(
            self.selected_expert_indices,
            dtype=np.dtype(np.int64),
            name="selected_expert_indices",
        )
        selected_probabilities = _frozen_copy(
            self.selected_probabilities,
            dtype=np.dtype(np.float64),
            name="selected_probabilities",
        )
        kept = _frozen_copy(
            self.kept_mask, dtype=np.dtype(np.bool_), name="kept_mask"
        )
        weights = _frozen_copy(
            self.combine_weights,
            dtype=np.dtype(np.float64),
            name="combine_weights",
        )
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "active_token_mask", active)
        object.__setattr__(self, "selected_expert_indices", selected)
        object.__setattr__(self, "selected_probabilities", selected_probabilities)
        object.__setattr__(self, "kept_mask", kept)
        object.__setattr__(self, "combine_weights", weights)

        if probabilities.ndim != 2 or not all(probabilities.shape):
            raise ValueError("probabilities must have shape [tokens, experts]")
        tokens, experts = probabilities.shape
        if active.shape != (tokens,) or active.dtype != np.bool_ or not np.any(active):
            raise ValueError("active_token_mask must select at least one token")
        if selected.ndim != 2 or selected.shape[0] != tokens or not selected.shape[1]:
            raise ValueError("selected_expert_indices must have shape [tokens, top_k]")
        if any(
            array.shape != selected.shape
            for array in (selected_probabilities, kept, weights)
        ):
            raise ValueError("routing assignment arrays must have matching shapes")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0):
            raise ValueError("router probability rows must sum to one")
        if np.any(probabilities < 0):
            raise ValueError("router probabilities cannot be negative")
        if np.any(selected[~active] != -1):
            raise ValueError("inactive tokens must use expert index -1")
        if np.any(selected[active] < 0) or np.any(selected[active] >= experts):
            raise ValueError("active selected expert index is out of range")
        if any(len(set(row.tolist())) != selected.shape[1] for row in selected[active]):
            raise ValueError("an active token cannot select the same expert twice")
        if np.any(selected_probabilities[~active] != 0) or np.any(kept[~active]):
            raise ValueError("inactive tokens cannot carry routing assignments")
        if np.any(weights[~kept] != 0) or np.any(weights < 0):
            raise ValueError("only kept assignments may have non-negative combine weight")
        active_rows = np.flatnonzero(active)
        expected_selected = probabilities[
            active_rows[:, None], selected[active_rows]
        ]
        if not np.allclose(
            selected_probabilities[active], expected_selected, atol=0, rtol=0
        ):
            raise ValueError("selected probabilities do not match router probabilities")
        expected_before = int(np.count_nonzero(active)) * selected.shape[1]
        if self.assignments_before_capacity != expected_before:
            raise ValueError("assignments_before_capacity is inconsistent")
        if self.kept_assignments != int(np.count_nonzero(kept)):
            raise ValueError("kept_assignments is inconsistent")
        if self.dropped_assignments != expected_before - self.kept_assignments:
            raise ValueError("dropped_assignments is inconsistent")
        if isinstance(self.expert_capacity, bool) or self.expert_capacity <= 0:
            raise ValueError("expert_capacity must be a positive integer")
        if len(self.expert_counts_before_capacity) != experts or len(
            self.expert_counts_after_capacity
        ) != experts:
            raise ValueError("expert count vectors must match the expert dimension")
        if sum(self.expert_counts_before_capacity) != expected_before or sum(
            self.expert_counts_after_capacity
        ) != self.kept_assignments:
            raise ValueError("expert counts do not match assignment totals")
        if any(count > self.expert_capacity for count in self.expert_counts_after_capacity):
            raise ValueError("an expert exceeds its capacity")
        all_dropped = int(np.count_nonzero(~np.any(kept[active], axis=1)))
        if self.tokens_with_all_assignments_dropped != all_dropped:
            raise ValueError("all-dropped token count is inconsistent")
        weight_sums = weights[active].sum(axis=1)
        has_kept = np.any(kept[active], axis=1)
        if self.renormalized_after_capacity:
            if not np.allclose(weight_sums[has_kept], 1.0, atol=1e-12, rtol=0):
                raise ValueError("renormalized kept weights must sum to one")
        elif np.any(weight_sums > 1 + 1e-12):
            raise ValueError("non-renormalized kept weights cannot sum above one")
        if np.any(weight_sums[~has_kept] != 0):
            raise ValueError("all-dropped tokens must have zero combine weight")
        if len(self.mean_router_probabilities) != experts or len(
            self.selection_fractions
        ) != experts:
            raise ValueError("router diagnostic vectors must match experts")
        if not math.isclose(sum(self.selection_fractions), 1.0, abs_tol=1e-12):
            raise ValueError("selection fractions must sum to one")
        for value in (
            *self.mean_router_probabilities,
            *self.selection_fractions,
            self.load_balance_diagnostic,
            self.router_z_loss,
            self.mean_router_entropy,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("router diagnostics must be finite and non-negative")

    @property
    def token_count(self) -> int:
        return int(self.probabilities.shape[0])

    @property
    def active_token_count(self) -> int:
        return int(np.count_nonzero(self.active_token_mask))

    @property
    def expert_count(self) -> int:
        return int(self.probabilities.shape[1])

    @property
    def top_k(self) -> int:
        return int(self.selected_expert_indices.shape[1])

    @property
    def dropped_assignment_fraction(self) -> float:
        return self.dropped_assignments / self.assignments_before_capacity

    def to_dict(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "active_token_count": self.active_token_count,
            "expert_count": self.expert_count,
            "top_k": self.top_k,
            "expert_capacity": self.expert_capacity,
            "assignments_before_capacity": self.assignments_before_capacity,
            "kept_assignments": self.kept_assignments,
            "dropped_assignments": self.dropped_assignments,
            "dropped_assignment_fraction": self.dropped_assignment_fraction,
            "tokens_with_all_assignments_dropped": (
                self.tokens_with_all_assignments_dropped
            ),
            "expert_counts_before_capacity": self.expert_counts_before_capacity,
            "expert_counts_after_capacity": self.expert_counts_after_capacity,
            "mean_router_probabilities": self.mean_router_probabilities,
            "selection_fractions": self.selection_fractions,
            "load_balance_diagnostic": self.load_balance_diagnostic,
            "router_z_loss": self.router_z_loss,
            "mean_router_entropy": self.mean_router_entropy,
            "renormalized_after_capacity": self.renormalized_after_capacity,
            "active_token_mask": self.active_token_mask.tolist(),
            "probabilities": self.probabilities.tolist(),
            "selected_expert_indices": self.selected_expert_indices.tolist(),
            "selected_probabilities": self.selected_probabilities.tolist(),
            "kept_mask": self.kept_mask.tolist(),
            "combine_weights": self.combine_weights.tolist(),
        }


def route_topk_capacity(
    router_logits: np.ndarray,
    *,
    top_k: int,
    capacity_factor: float = 1.0,
    token_mask: np.ndarray | None = None,
    renormalize_after_capacity: bool = True,
) -> MoERoutingResult:
    """Route active tokens with an explicit score-priority capacity policy.

    Per-expert capacity is ``ceil(capacity_factor * active_tokens * top_k /
    experts)``. Top-k ties prefer the lower expert id. Within an expert, higher
    router probability wins capacity; exact ties prefer lower token index and
    then lower top-k rank. The load-balance value is this module's generalized
    diagnostic ``E * sum(f_e * p_e)`` where ``f_e`` is the pre-capacity top-k
    assignment fraction and ``p_e`` the active-token mean router probability.
    It is not asserted to be the loss used by every MoE implementation.
    """

    logits = np.asarray(router_logits, dtype=np.float64)
    if logits.ndim != 2 or not all(logits.shape):
        raise ValueError("router_logits must have shape [tokens, experts]")
    if not np.all(np.isfinite(logits)):
        raise ValueError("router_logits must contain only finite values")
    tokens, experts = logits.shape
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= experts:
        raise ValueError("top_k must be an integer in [1, experts]")
    if (
        isinstance(capacity_factor, bool)
        or not isinstance(capacity_factor, (int, float))
        or not math.isfinite(capacity_factor)
        or capacity_factor <= 0
    ):
        raise ValueError("capacity_factor must be finite and positive")
    if not isinstance(renormalize_after_capacity, bool):
        raise TypeError("renormalize_after_capacity must be boolean")
    if token_mask is None:
        active = np.ones(tokens, dtype=np.bool_)
    else:
        raw_mask = np.asarray(token_mask)
        if raw_mask.dtype != np.bool_ or raw_mask.shape != (tokens,):
            raise ValueError("token_mask must be a boolean vector with one value per token")
        active = raw_mask.copy()
    if not np.any(active):
        raise ValueError("token_mask must select at least one active token")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    selected = np.full((tokens, top_k), -1, dtype=np.int64)
    selected_probabilities = np.zeros((tokens, top_k), dtype=np.float64)
    active_indices = np.flatnonzero(active)
    for token_index in active_indices:
        order = np.argsort(-probabilities[token_index], kind="stable")[:top_k]
        selected[token_index] = order
        selected_probabilities[token_index] = probabilities[token_index, order]

    active_count = int(active_indices.size)
    capacity = math.ceil(float(capacity_factor) * active_count * top_k / experts)
    before_counts = np.zeros(experts, dtype=np.int64)
    kept = np.zeros((tokens, top_k), dtype=np.bool_)
    for expert_id in range(experts):
        assignments = [
            (int(token_index), rank)
            for token_index in active_indices
            for rank in range(top_k)
            if selected[token_index, rank] == expert_id
        ]
        before_counts[expert_id] = len(assignments)
        assignments.sort(
            key=lambda item: (
                -selected_probabilities[item[0], item[1]],
                item[0],
                item[1],
            )
        )
        for assignment_token_index, rank in assignments[:capacity]:
            kept[assignment_token_index, rank] = True

    topk_denominator = selected_probabilities.sum(axis=1, keepdims=True)
    normalized_topk = np.divide(
        selected_probabilities,
        topk_denominator,
        out=np.zeros_like(selected_probabilities),
        where=topk_denominator > 0,
    )
    combine_weights = np.where(kept, normalized_topk, 0.0)
    if renormalize_after_capacity:
        kept_denominator = combine_weights.sum(axis=1, keepdims=True)
        combine_weights = np.divide(
            combine_weights,
            kept_denominator,
            out=np.zeros_like(combine_weights),
            where=kept_denominator > 0,
        )

    after_counts = np.asarray(
        [np.count_nonzero(kept & (selected == expert_id)) for expert_id in range(experts)],
        dtype=np.int64,
    )
    assignments_before = active_count * top_k
    kept_assignments = int(np.count_nonzero(kept))
    selection_fractions = before_counts.astype(np.float64) / assignments_before
    mean_probabilities = probabilities[active].mean(axis=0)
    load_balance = float(experts * np.sum(selection_fractions * mean_probabilities))
    row_maxima = np.max(logits[active], axis=1)
    log_partitions = row_maxima + np.log(
        np.exp(logits[active] - row_maxima[:, None]).sum(axis=1)
    )
    with np.errstate(over="ignore"):
        z_loss = float(np.mean(np.square(log_partitions)))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(
            probabilities[active] > 0,
            probabilities[active] * np.log(probabilities[active]),
            0.0,
        )
    entropy = float(-np.mean(np.sum(entropy_terms, axis=1)))
    if not all(math.isfinite(value) for value in (load_balance, z_loss, entropy)):
        raise ValueError("router logits make diagnostics non-finite")

    return MoERoutingResult(
        probabilities=probabilities,
        active_token_mask=active,
        selected_expert_indices=selected,
        selected_probabilities=selected_probabilities,
        kept_mask=kept,
        combine_weights=combine_weights,
        expert_capacity=capacity,
        assignments_before_capacity=assignments_before,
        kept_assignments=kept_assignments,
        dropped_assignments=assignments_before - kept_assignments,
        tokens_with_all_assignments_dropped=int(
            np.count_nonzero(~np.any(kept[active], axis=1))
        ),
        expert_counts_before_capacity=tuple(int(value) for value in before_counts),
        expert_counts_after_capacity=tuple(int(value) for value in after_counts),
        mean_router_probabilities=tuple(float(value) for value in mean_probabilities),
        selection_fractions=tuple(float(value) for value in selection_fractions),
        load_balance_diagnostic=load_balance,
        router_z_loss=z_loss,
        mean_router_entropy=entropy,
        renormalized_after_capacity=renormalize_after_capacity,
    )


def routed_linear_expert_forward(
    hidden_states: np.ndarray,
    expert_weights: np.ndarray,
    routing: MoERoutingResult,
) -> np.ndarray:
    """Execute kept sparse assignments for bias-free linear toy experts."""

    hidden = np.asarray(hidden_states, dtype=np.float64)
    weights = np.asarray(expert_weights, dtype=np.float64)
    if hidden.ndim != 2 or hidden.shape[0] != routing.token_count:
        raise ValueError("hidden_states must have shape [routing tokens, input_dim]")
    if weights.ndim != 3 or weights.shape[:2] != (
        routing.expert_count,
        hidden.shape[1],
    ):
        raise ValueError(
            "expert_weights must have shape [routing experts, input_dim, output_dim]"
        )
    if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(weights)):
        raise ValueError("hidden_states and expert_weights must be finite")
    output = np.zeros((routing.token_count, weights.shape[2]), dtype=np.float64)
    for token_index in range(routing.token_count):
        for rank in range(routing.top_k):
            if not routing.kept_mask[token_index, rank]:
                continue
            expert_id = routing.selected_expert_indices[token_index, rank]
            expert_output = hidden[token_index] @ weights[expert_id]
            output[token_index] += (
                routing.combine_weights[token_index, rank] * expert_output
            )
    return output
