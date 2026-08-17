"""Two-process CPU/Gloo control for a collective MoE capacity group."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TypedDict, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.multiprocessing.spawn import spawn

from about_llm.from_scratch.moe_training import (
    TrainableMoEForward,
    TrainableTopKMoE,
)

DISTRIBUTED_MOE_CAPACITY_CONTROL_VERSION = (
    "about-llm.distributed-moe-capacity-control.v1"
)
WORLD_SIZE = 2
TOKENS_PER_RANK = 2
CAPACITY_FACTOR = 0.5


class RoutePayload(TypedDict):
    expert_capacity: int | None
    selected_expert_indices: list[list[int]]
    selected_probabilities: list[list[float]]
    kept_mask: list[list[bool]]
    expert_counts_before_capacity: list[int]
    expert_counts_after_capacity: list[int]
    pre_policy_capacity_excess_by_group: list[list[int]]
    post_policy_capacity_excess_by_group: list[list[int]]
    dropped_assignments: int
    routed_output: list[list[float]]


class LocalSlicePayload(TypedDict):
    global_token_indices: list[int]
    kept_mask: list[list[bool]]
    routed_output: list[list[float]]
    vs_local_independent_output_max_abs_difference: float


class RankReport(TypedDict):
    rank: int
    process_id: int
    local_hidden_states: list[list[float]]
    gathered_hidden_states: list[list[float]]
    global_active_token_count_after_all_reduce: int
    global_selected_counts_after_all_reduce: list[int]
    local_independent_route: RoutePayload
    collective_global_route: RoutePayload
    collective_global_route_fingerprint: str
    collective_local_slice: LocalSlicePayload
    collective_call_counts: dict[str, int]


class PublishedRankReport(TypedDict):
    rank: int
    local_hidden_states: list[list[float]]
    gathered_hidden_states: list[list[float]]
    global_active_token_count_after_all_reduce: int
    global_selected_counts_after_all_reduce: list[int]
    local_independent_route: RoutePayload
    collective_global_route: RoutePayload
    collective_global_route_fingerprint: str
    collective_local_slice: LocalSlicePayload
    collective_call_counts: dict[str, int]


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


def _authored_distributed_model() -> TrainableTopKMoE:
    model = TrainableTopKMoE(
        d_model=1,
        hidden_dim=1,
        output_dim=1,
        expert_count=2,
        top_k=1,
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.router.weight.copy_(
            torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        )
        for expert_id, expert in enumerate(model.experts):
            if not isinstance(expert, nn.Sequential):
                raise AssertionError("authored expert topology drifted")
            first = expert[0]
            second = expert[2]
            if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
                raise AssertionError("authored expert topology drifted")
            first.weight.fill_(1.0 if expert_id == 0 else -1.0)
            second.weight.fill_(1.0)
    return model


def _local_hidden(rank: int) -> Tensor:
    if rank == 0:
        values = [[2.0], [1.0]]
    elif rank == 1:
        values = [[3.0], [0.5]]
    else:
        raise ValueError(f"rank must be in [0, {WORLD_SIZE})")
    return torch.tensor(values, dtype=torch.float64)


def _route_payload(forward: TrainableMoEForward) -> RoutePayload:
    selected_expert_indices = forward.selected_expert_indices
    return {
        "expert_capacity": forward.expert_capacity,
        "selected_expert_indices": selected_expert_indices.tolist(),
        "selected_probabilities": forward.selected_probabilities.detach().tolist(),
        "kept_mask": forward.kept_mask.tolist(),
        "expert_counts_before_capacity": (
            forward.expert_counts_before_capacity.tolist()
        ),
        "expert_counts_after_capacity": (
            forward.expert_counts_after_capacity.tolist()
        ),
        "pre_policy_capacity_excess_by_group": (
            forward.pre_policy_capacity_excess_by_group.tolist()
        ),
        "post_policy_capacity_excess_by_group": (
            forward.post_policy_capacity_excess_by_group.tolist()
        ),
        "dropped_assignments": forward.dropped_assignments,
        "routed_output": forward.output.detach().tolist(),
    }


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
        local_hidden = _local_hidden(rank)
        model = _authored_distributed_model()
        local_uncapped = model(local_hidden)

        gathered_hidden = [torch.zeros_like(local_hidden) for _ in range(world_size)]
        dist.all_gather(gathered_hidden, local_hidden)
        global_hidden = torch.cat(gathered_hidden, dim=0)

        active_token_count = torch.tensor(
            [local_hidden.shape[0]],
            dtype=torch.int64,
        )
        dist.all_reduce(active_token_count, op=dist.ReduceOp.SUM)

        selected_counts = local_uncapped.expert_counts_before_capacity.to(
            dtype=torch.int64
        )
        dist.all_reduce(selected_counts, op=dist.ReduceOp.SUM)

        local_independent = model(
            local_hidden,
            capacity_factor=CAPACITY_FACTOR,
            overflow_policy="drop",
        )
        collective_global = model(
            global_hidden,
            capacity_factor=CAPACITY_FACTOR,
            overflow_policy="drop",
        )
        start = rank * TOKENS_PER_RANK
        stop = start + TOKENS_PER_RANK
        global_route = _route_payload(collective_global)
        global_route_fingerprint = "sha256:" + hashlib.sha256(
            _canonical_bytes(global_route)
        ).hexdigest()
        local_output = local_independent.output.detach()
        collective_local_output = collective_global.output[start:stop].detach()
        payload: RankReport = {
            "rank": rank,
            "process_id": os.getpid(),
            "local_hidden_states": local_hidden.tolist(),
            "gathered_hidden_states": global_hidden.tolist(),
            "global_active_token_count_after_all_reduce": int(
                active_token_count.item()
            ),
            "global_selected_counts_after_all_reduce": selected_counts.tolist(),
            "local_independent_route": _route_payload(local_independent),
            "collective_global_route": global_route,
            "collective_global_route_fingerprint": global_route_fingerprint,
            "collective_local_slice": {
                "global_token_indices": list(range(start, stop)),
                "kept_mask": collective_global.kept_mask[start:stop].tolist(),
                "routed_output": collective_local_output.tolist(),
                "vs_local_independent_output_max_abs_difference": float(
                    torch.max(
                        torch.abs(collective_local_output - local_output)
                    ).item()
                ),
            },
            "collective_call_counts": {
                "all_gather": 1,
                "all_reduce": 2,
                "barrier": 1,
            },
        }
        output_path = Path(output_directory) / f"rank-{rank}.json"
        output_path.write_bytes(_canonical_bytes(payload))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _load_rank_report(path: Path) -> RankReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("rank report must be a JSON object")
    return cast(RankReport, payload)


def _publish_rank_report(report: RankReport) -> PublishedRankReport:
    return {
        "rank": report["rank"],
        "local_hidden_states": report["local_hidden_states"],
        "gathered_hidden_states": report["gathered_hidden_states"],
        "global_active_token_count_after_all_reduce": (
            report["global_active_token_count_after_all_reduce"]
        ),
        "global_selected_counts_after_all_reduce": (
            report["global_selected_counts_after_all_reduce"]
        ),
        "local_independent_route": report["local_independent_route"],
        "collective_global_route": report["collective_global_route"],
        "collective_global_route_fingerprint": (
            report["collective_global_route_fingerprint"]
        ),
        "collective_local_slice": report["collective_local_slice"],
        "collective_call_counts": report["collective_call_counts"],
    }


def run_distributed_moe_capacity_control() -> dict[str, object]:
    """Execute collective and rank-local MoE capacity competition controls."""

    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed with the Gloo backend is required")
    with tempfile.TemporaryDirectory(
        prefix="about-llm-distributed-moe-capacity-"
    ) as temporary:
        root = Path(temporary)
        init_method = (root / "rendezvous").resolve().as_uri()
        output_directory = root / "rank-results"
        output_directory.mkdir()
        spawn(  # type: ignore[no-untyped-call]
            _worker,
            args=(WORLD_SIZE, init_method, str(output_directory)),
            nprocs=WORLD_SIZE,
            join=True,
        )
        rank_reports = [
            _load_rank_report(output_directory / f"rank-{rank}.json")
            for rank in range(WORLD_SIZE)
        ]

    collective_routes = [
        report["collective_global_route"] for report in rank_reports
    ]
    local_routes = [report["local_independent_route"] for report in rank_reports]
    local_slices = [report["collective_local_slice"] for report in rank_reports]
    process_ids = [report["process_id"] for report in rank_reports]
    global_fingerprints = [
        report["collective_global_route_fingerprint"] for report in rank_reports
    ]
    first_global_route = collective_routes[0]
    independent_kept_assignments = sum(
        sum(route["expert_counts_after_capacity"])
        for route in local_routes
    )
    collective_kept_assignments = sum(
        first_global_route["expert_counts_after_capacity"]
    )
    assertions = {
        "two_distinct_worker_processes_executed": (
            len(set(process_ids)) == WORLD_SIZE
        ),
        "all_ranks_observe_same_collectively_gathered_hidden_states": all(
            report["gathered_hidden_states"]
            == [[2.0], [1.0], [3.0], [0.5]]
            for report in rank_reports
        ),
        "all_reduce_observes_global_tokens_and_selected_counts": all(
            report["global_active_token_count_after_all_reduce"] == 4
            and report["global_selected_counts_after_all_reduce"] == [4, 0]
            for report in rank_reports
        ),
        "replicated_global_routing_is_identical_across_ranks": (
            len(set(global_fingerprints)) == 1
            and all(route == first_global_route for route in collective_routes)
        ),
        "rank_local_independent_capacity_keeps_two_assignments": (
            independent_kept_assignments == 2
            and all(route["expert_capacity"] == 1 for route in local_routes)
            and all(
                route["kept_mask"] == [[True], [False]]
                for route in local_routes
            )
        ),
        "collective_capacity_group_keeps_only_global_score_winner": (
            first_global_route["expert_capacity"] == 1
            and first_global_route["selected_expert_indices"] == [[0]] * 4
            and first_global_route["kept_mask"]
            == [[False], [False], [True], [False]]
            and first_global_route["expert_counts_before_capacity"] == [4, 0]
            and first_global_route["expert_counts_after_capacity"] == [1, 0]
            and first_global_route["pre_policy_capacity_excess_by_group"]
            == [[3, 0]]
            and first_global_route["post_policy_capacity_excess_by_group"]
            == [[0, 0]]
            and first_global_route["dropped_assignments"] == 3
            and collective_kept_assignments == 1
        ),
        "collective_competition_changes_rank_zero_output": (
            local_slices[0]["kept_mask"] == [[False], [False]]
            and local_slices[0][
                "vs_local_independent_output_max_abs_difference"
            ]
            > 0
        ),
        "global_winner_on_rank_one_is_preserved": (
            local_slices[1]["kept_mask"] == [[True], [False]]
            and local_slices[1][
                "vs_local_independent_output_max_abs_difference"
            ]
            == 0.0
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(
            f"distributed MoE capacity control assertion failed: {assertions}"
        )

    published_rank_reports = [
        _publish_rank_report(rank_report) for rank_report in rank_reports
    ]

    report: dict[str, object] = {
        "schema_version": DISTRIBUTED_MOE_CAPACITY_CONTROL_VERSION,
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
            "router_weight": [[1.0], [0.0]],
            "expert_count": 2,
            "top_k": 1,
            "capacity_factor": CAPACITY_FACTOR,
            "overflow_policy": "drop",
            "capacity_formula": (
                "ceil(capacity_factor * active_tokens * top_k / experts)"
            ),
        },
        "process_observation": {
            "distinct_worker_process_count": len(set(process_ids)),
            "raw_process_ids_published": False,
        },
        "rank_reports": published_rank_reports,
        "comparison": {
            "independent_rank_local_kept_assignments": (
                independent_kept_assignments
            ),
            "collective_global_kept_assignments": collective_kept_assignments,
            "collective_minus_independent_kept_assignments": (
                collective_kept_assignments - independent_kept_assignments
            ),
            "rank_zero_collective_vs_local_output_max_abs_difference": (
                local_slices[0][
                    "vs_local_independent_output_max_abs_difference"
                ]
            ),
            "rank_one_collective_vs_local_output_max_abs_difference": (
                local_slices[1][
                    "vs_local_independent_output_max_abs_difference"
                ]
            ),
        },
        "assertions": assertions,
        "scope": {
            "real_two_process_same_host_gloo_process_group_executed": True,
            "temporary_file_store_rendezvous_executed": True,
            "hidden_state_all_gather_for_replicated_global_routing_executed": True,
            "global_active_token_count_all_reduce_executed": True,
            "global_selected_assignment_count_all_reduce_executed": True,
            "collective_capacity_group_competition_executed": True,
            "replicated_router_and_experts_used": True,
            "distributed_autograd_or_ddp_backward_executed": False,
            "expert_parallel_all_to_all_or_reduce_scatter_executed": False,
            "cuda_nccl_multi_node_or_remote_host_executed": False,
            "deepseek_qwen_or_other_checkpoint_reproduced": False,
            "throughput_memory_scaling_convergence_or_quality_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


__all__ = [
    "DISTRIBUTED_MOE_CAPACITY_CONTROL_VERSION",
    "run_distributed_moe_capacity_control",
]
