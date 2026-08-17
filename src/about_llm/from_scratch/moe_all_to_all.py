"""Two-process CPU/Gloo MoE token dispatch and return control."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.multiprocessing.spawn import spawn

MOE_ALL_TO_ALL_CONTROL_VERSION = "about-llm.moe-all-to-all-control.v1"
WORLD_SIZE = 2
EXPERT_COUNT = 2
FLOAT_COLUMNS = 2
METADATA_COLUMNS = 4


class DispatchTrace(TypedDict):
    send_counts_by_owner: list[int]
    received_counts_by_source: list[int]
    dispatch_send_float_payload: list[list[float]]
    dispatch_send_metadata: list[list[int]]
    owner_received_float_payload: list[list[float]]
    owner_received_metadata: list[list[int]]
    owner_raw_expert_outputs: list[list[float]]
    return_arrival_float_payload: list[list[float]]
    return_arrival_metadata: list[list[int]]
    final_local_outputs: list[list[float]]
    metadata_free_arrival_weighted_outputs: list[list[float]]
    metadata_free_vs_correct_max_abs_difference: float


class WorkerReport(TypedDict):
    rank: int
    process_id: int
    owned_expert_id: int
    local_hidden_states: list[list[float]]
    local_global_token_ids: list[int]
    selected_expert_indices: list[int]
    selected_probabilities: list[float]
    combine_weights: list[float]
    trace: DispatchTrace
    all_to_all_single_call_count: int
    logical_tensor_payload_bytes_sent: int


class PublishedWorkerReport(TypedDict):
    rank: int
    owned_expert_id: int
    local_hidden_states: list[list[float]]
    local_global_token_ids: list[int]
    selected_expert_indices: list[int]
    selected_probabilities: list[float]
    combine_weights: list[float]
    trace: DispatchTrace
    all_to_all_single_call_count: int
    logical_tensor_payload_bytes_sent: int


class OracleItem(TypedDict):
    global_token_id: int
    source_rank: int
    source_local_index: int
    hidden_state: float
    expert_id: int
    selected_probability: float
    combine_weight: float
    raw_expert_output: float
    combined_output: float


@dataclass(frozen=True)
class RouterDecision:
    selected_expert_indices: Tensor
    selected_probabilities: Tensor
    combine_weights: Tensor


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


def _local_hidden(rank: int) -> Tensor:
    if rank == 0:
        values = [[-1.0], [2.0], [-2.0]]
    elif rank == 1:
        values = [[1.0]]
    else:
        raise ValueError(f"rank must be in [0, {WORLD_SIZE})")
    return torch.tensor(values, dtype=torch.float64)


def _global_token_offset(rank: int) -> int:
    return sum(_local_hidden(lower_rank).shape[0] for lower_rank in range(rank))


def _route(hidden_states: Tensor) -> RouterDecision:
    router_weight = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    logits = hidden_states @ router_weight.T
    probabilities = torch.softmax(logits, dim=-1)
    ranked = torch.argsort(
        probabilities,
        dim=-1,
        descending=True,
        stable=True,
    )
    selected = ranked[:, 0]
    selected_probabilities = torch.gather(
        probabilities,
        dim=1,
        index=selected.unsqueeze(1),
    ).squeeze(1)
    return RouterDecision(
        selected_expert_indices=selected,
        selected_probabilities=selected_probabilities,
        combine_weights=selected_probabilities.clone(),
    )


class OwnedExpert(nn.Module):
    """One expert whose parameter set exists only on its owner rank."""

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


def _pack_dispatch(
    rank: int,
    hidden_states: Tensor,
    decision: RouterDecision,
) -> tuple[Tensor, Tensor, Tensor]:
    local_token_count = hidden_states.shape[0]
    local_indices = torch.arange(local_token_count, dtype=torch.int64)
    global_ids = local_indices + _global_token_offset(rank)
    order = torch.argsort(
        decision.selected_expert_indices,
        stable=True,
    )
    send_counts = torch.bincount(
        decision.selected_expert_indices,
        minlength=WORLD_SIZE,
    ).to(dtype=torch.int64)
    float_payload = torch.cat(
        [
            hidden_states[order],
            decision.combine_weights[order].unsqueeze(1),
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


def _all_to_all_rows(
    input_tensor: Tensor,
    *,
    input_counts: Tensor,
    output_counts: Tensor,
) -> Tensor:
    output_shape = (int(output_counts.sum().item()), *input_tensor.shape[1:])
    output_tensor = torch.empty(
        output_shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=[int(value) for value in output_counts.tolist()],
        input_split_sizes=[int(value) for value in input_counts.tolist()],
    )
    return output_tensor


def _scatter_returned_outputs(
    rank: int,
    local_token_count: int,
    returned_float_payload: Tensor,
    returned_metadata: Tensor,
) -> tuple[Tensor, Tensor, float]:
    if returned_float_payload.shape != (local_token_count, FLOAT_COLUMNS):
        raise AssertionError("returned float payload cardinality drifted")
    if returned_metadata.shape != (local_token_count, METADATA_COLUMNS):
        raise AssertionError("returned metadata cardinality drifted")
    final_output = torch.zeros((local_token_count, 1), dtype=torch.float64)
    seen_local_indices: set[int] = set()
    for row_index in range(local_token_count):
        source_rank, local_index, _, _ = (
            int(value) for value in returned_metadata[row_index].tolist()
        )
        if source_rank != rank:
            raise AssertionError("returned token reached the wrong source rank")
        if local_index in seen_local_indices:
            raise AssertionError("returned token local index is duplicated")
        if local_index < 0 or local_index >= local_token_count:
            raise AssertionError("returned token local index is out of range")
        seen_local_indices.add(local_index)
        raw_output = returned_float_payload[row_index, 0]
        combine_weight = returned_float_payload[row_index, 1]
        final_output[local_index, 0] = raw_output * combine_weight
    if len(seen_local_indices) != local_token_count:
        raise AssertionError("not every local token returned from its owner expert")

    metadata_free = (
        returned_float_payload[:, :1] * returned_float_payload[:, 1:2]
    )
    metadata_free_difference = float(
        torch.max(torch.abs(metadata_free - final_output)).item()
    )
    return final_output, metadata_free, metadata_free_difference


def _logical_payload_bytes_sent(
    send_float_payload: Tensor,
    send_metadata: Tensor,
    owner_received_count: int,
) -> int:
    count_exchange_bytes = WORLD_SIZE * torch.tensor([], dtype=torch.int64).element_size()
    dispatch_bytes = (
        send_float_payload.numel() * send_float_payload.element_size()
        + send_metadata.numel() * send_metadata.element_size()
    )
    return_bytes = owner_received_count * (
        FLOAT_COLUMNS * torch.tensor([], dtype=torch.float64).element_size()
        + METADATA_COLUMNS * torch.tensor([], dtype=torch.int64).element_size()
    )
    return count_exchange_bytes + dispatch_bytes + return_bytes


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
        local_token_count = local_hidden.shape[0]
        decision = _route(local_hidden)
        send_counts, send_float_payload, send_metadata = _pack_dispatch(
            rank,
            local_hidden,
            decision,
        )

        received_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(received_counts, send_counts)
        received_float_payload = _all_to_all_rows(
            send_float_payload,
            input_counts=send_counts,
            output_counts=received_counts,
        )
        received_metadata = _all_to_all_rows(
            send_metadata,
            input_counts=send_counts,
            output_counts=received_counts,
        )
        if not bool(torch.all(received_metadata[:, 3] == rank).item()):
            raise AssertionError("owner rank received a token for another expert")

        owned_expert = OwnedExpert(rank)
        with torch.no_grad():
            owner_raw_outputs = owned_expert(received_float_payload[:, :1])
        return_send_float = torch.cat(
            [owner_raw_outputs, received_float_payload[:, 1:2]],
            dim=1,
        ).contiguous()
        return_send_metadata = received_metadata.contiguous()
        returned_float_payload = _all_to_all_rows(
            return_send_float,
            input_counts=received_counts,
            output_counts=send_counts,
        )
        returned_metadata = _all_to_all_rows(
            return_send_metadata,
            input_counts=received_counts,
            output_counts=send_counts,
        )
        final_output, metadata_free, metadata_free_difference = (
            _scatter_returned_outputs(
                rank,
                local_token_count,
                returned_float_payload,
                returned_metadata,
            )
        )
        global_offset = _global_token_offset(rank)
        payload: WorkerReport = {
            "rank": rank,
            "process_id": os.getpid(),
            "owned_expert_id": rank,
            "local_hidden_states": local_hidden.tolist(),
            "local_global_token_ids": list(
                range(global_offset, global_offset + local_token_count)
            ),
            "selected_expert_indices": (
                decision.selected_expert_indices.tolist()
            ),
            "selected_probabilities": decision.selected_probabilities.tolist(),
            "combine_weights": decision.combine_weights.tolist(),
            "trace": {
                "send_counts_by_owner": send_counts.tolist(),
                "received_counts_by_source": received_counts.tolist(),
                "dispatch_send_float_payload": send_float_payload.tolist(),
                "dispatch_send_metadata": send_metadata.tolist(),
                "owner_received_float_payload": received_float_payload.tolist(),
                "owner_received_metadata": received_metadata.tolist(),
                "owner_raw_expert_outputs": owner_raw_outputs.tolist(),
                "return_arrival_float_payload": returned_float_payload.tolist(),
                "return_arrival_metadata": returned_metadata.tolist(),
                "final_local_outputs": final_output.tolist(),
                "metadata_free_arrival_weighted_outputs": metadata_free.tolist(),
                "metadata_free_vs_correct_max_abs_difference": (
                    metadata_free_difference
                ),
            },
            "all_to_all_single_call_count": 5,
            "logical_tensor_payload_bytes_sent": _logical_payload_bytes_sent(
                send_float_payload,
                send_metadata,
                int(received_counts.sum().item()),
            ),
        }
        output_path = Path(output_directory) / f"rank-{rank}.json"
        output_path.write_bytes(_canonical_bytes(payload))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _load_worker_report(path: Path) -> WorkerReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("worker report must be a JSON object")
    return cast(WorkerReport, payload)


def _publish_worker_report(report: WorkerReport) -> PublishedWorkerReport:
    return {
        "rank": report["rank"],
        "owned_expert_id": report["owned_expert_id"],
        "local_hidden_states": report["local_hidden_states"],
        "local_global_token_ids": report["local_global_token_ids"],
        "selected_expert_indices": report["selected_expert_indices"],
        "selected_probabilities": report["selected_probabilities"],
        "combine_weights": report["combine_weights"],
        "trace": report["trace"],
        "all_to_all_single_call_count": report[
            "all_to_all_single_call_count"
        ],
        "logical_tensor_payload_bytes_sent": report[
            "logical_tensor_payload_bytes_sent"
        ],
    }


def _single_process_oracle() -> list[OracleItem]:
    oracle: list[OracleItem] = []
    for rank in range(WORLD_SIZE):
        hidden_states = _local_hidden(rank)
        decision = _route(hidden_states)
        global_offset = _global_token_offset(rank)
        for local_index in range(hidden_states.shape[0]):
            hidden = float(hidden_states[local_index, 0].item())
            expert_id = int(
                decision.selected_expert_indices[local_index].item()
            )
            probability = float(
                decision.selected_probabilities[local_index].item()
            )
            expert = OwnedExpert(expert_id)
            with torch.no_grad():
                raw_output = float(
                    expert(hidden_states[local_index : local_index + 1])[
                        0, 0
                    ].item()
                )
            oracle.append(
                {
                    "global_token_id": global_offset + local_index,
                    "source_rank": rank,
                    "source_local_index": local_index,
                    "hidden_state": hidden,
                    "expert_id": expert_id,
                    "selected_probability": probability,
                    "combine_weight": probability,
                    "raw_expert_output": raw_output,
                    "combined_output": raw_output * probability,
                }
            )
    return oracle


def run_moe_all_to_all_control() -> dict[str, object]:
    """Execute variable-split token dispatch, expert compute, and return."""

    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed with the Gloo backend is required")
    with tempfile.TemporaryDirectory(
        prefix="about-llm-moe-all-to-all-"
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

    process_ids = [report["process_id"] for report in worker_reports]
    traces = [report["trace"] for report in worker_reports]
    oracle = _single_process_oracle()
    distributed_outputs_by_global_id: dict[int, float] = {}
    for worker_report in worker_reports:
        for global_id, output in zip(
            worker_report["local_global_token_ids"],
            worker_report["trace"]["final_local_outputs"],
            strict=True,
        ):
            distributed_outputs_by_global_id[global_id] = output[0]
    distributed_outputs = [
        distributed_outputs_by_global_id[global_id]
        for global_id in range(len(oracle))
    ]
    oracle_outputs = [item["combined_output"] for item in oracle]
    oracle_difference = max(
        abs(distributed - expected)
        for distributed, expected in zip(
            distributed_outputs,
            oracle_outputs,
            strict=True,
        )
    )
    source_to_owner_counts = [
        trace["send_counts_by_owner"] for trace in traces
    ]
    owner_from_source_counts = [
        trace["received_counts_by_source"] for trace in traces
    ]
    total_logical_payload_bytes = sum(
        report["logical_tensor_payload_bytes_sent"]
        for report in worker_reports
    )
    assertions = {
        "two_distinct_worker_processes_executed": (
            len(set(process_ids)) == WORLD_SIZE
        ),
        "variable_split_source_to_owner_matrix_matches": (
            source_to_owner_counts == [[1, 2], [1, 0]]
            and owner_from_source_counts == [[1, 1], [2, 0]]
        ),
        "each_rank_owns_exactly_one_expert": (
            [report["owned_expert_id"] for report in worker_reports] == [0, 1]
            and traces[0]["owner_received_metadata"]
            == [[0, 1, 1, 0], [1, 0, 3, 0]]
            and traces[1]["owner_received_metadata"]
            == [[0, 0, 0, 1], [0, 2, 2, 1]]
        ),
        "owner_expert_outputs_return_to_source_and_match_oracle": (
            oracle_difference <= 1e-15
            and len(distributed_outputs_by_global_id) == len(oracle)
        ),
        "return_metadata_restores_original_rank_zero_token_order": (
            traces[0]["return_arrival_metadata"]
            == [[0, 1, 1, 0], [0, 0, 0, 1], [0, 2, 2, 1]]
            and traces[0]["metadata_free_vs_correct_max_abs_difference"] > 0
        ),
        "rank_one_arrival_order_already_matches_local_order": (
            traces[1]["return_arrival_metadata"] == [[1, 0, 3, 0]]
            and traces[1]["metadata_free_vs_correct_max_abs_difference"]
            == 0.0
        ),
        "five_all_to_all_single_calls_execute_per_rank": all(
            report["all_to_all_single_call_count"] == 5
            for report in worker_reports
        ),
        "logical_payload_accounting_matches_authored_tensors": (
            [
                report["logical_tensor_payload_bytes_sent"]
                for report in worker_reports
            ]
            == [256, 160]
            and total_logical_payload_bytes == 416
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"MoE all-to-all assertion failed: {assertions}")

    published_workers = [
        _publish_worker_report(report) for report in worker_reports
    ]
    report: dict[str, object] = {
        "schema_version": MOE_ALL_TO_ALL_CONTROL_VERSION,
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
            "router_weight": [[1.0], [-1.0]],
            "expert_ownership": {"expert_0": 0, "expert_1": 1},
            "expert_parameters": {
                "expert_0": {"weight": 2.0, "bias": 0.5},
                "expert_1": {"weight": -3.0, "bias": 1.0},
            },
            "top_k": 1,
            "combine_weight_policy": "preserve selected softmax probability",
            "capacity_or_drop_policy": "none; every assignment is dispatched",
            "dispatch_float_columns": ["hidden_state", "combine_weight"],
            "metadata_columns": [
                "source_rank",
                "source_local_index",
                "global_token_id",
                "expert_id",
            ],
        },
        "process_observation": {
            "distinct_worker_process_count": len(set(process_ids)),
            "raw_process_ids_published": False,
        },
        "worker_reports": published_workers,
        "single_process_oracle": oracle,
        "comparison": {
            "source_to_owner_token_counts": source_to_owner_counts,
            "owner_from_source_token_counts": owner_from_source_counts,
            "distributed_outputs_by_global_token_id": distributed_outputs,
            "single_process_oracle_outputs_by_global_token_id": oracle_outputs,
            "distributed_vs_oracle_max_abs_difference": oracle_difference,
            "rank_zero_metadata_free_vs_correct_max_abs_difference": (
                traces[0]["metadata_free_vs_correct_max_abs_difference"]
            ),
            "rank_one_metadata_free_vs_correct_max_abs_difference": (
                traces[1]["metadata_free_vs_correct_max_abs_difference"]
            ),
            "logical_tensor_payload_bytes_sent_by_rank": [
                report["logical_tensor_payload_bytes_sent"]
                for report in worker_reports
            ],
            "logical_tensor_payload_bytes_sent_total": (
                total_logical_payload_bytes
            ),
        },
        "assertions": assertions,
        "scope": {
            "real_two_process_same_host_gloo_process_group_executed": True,
            "variable_split_all_to_all_single_count_exchange_executed": True,
            "token_to_owner_float_and_metadata_dispatch_executed": True,
            "owner_only_expert_parameter_placement_executed": True,
            "owner_to_source_output_and_metadata_return_executed": True,
            "source_metadata_scatter_and_gate_combine_executed": True,
            "single_process_forward_oracle_compared": True,
            "replicated_router_executed": True,
            "capacity_drop_reroute_or_dropless_executed": False,
            "distributed_autograd_backward_or_optimizer_executed": False,
            "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
            "cuda_nccl_multi_node_or_remote_host_executed": False,
            "wire_bytes_protocol_overhead_or_packet_capture_measured": False,
            "deepseek_qwen_or_other_checkpoint_reproduced": False,
            "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


__all__ = [
    "MOE_ALL_TO_ALL_CONTROL_VERSION",
    "run_moe_all_to_all_control",
]
