from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import torch.distributed as dist

from about_llm.from_scratch.moe_all_to_all import (
    MOE_ALL_TO_ALL_CONTROL_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT / "projects" / "transformers-basics" / "moe_all_to_all_control.py"
)


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


@pytest.mark.slow
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_process_gloo_variable_split_moe_all_to_all_control() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    report = json.loads(
        completed.stdout,
        parse_constant=_reject_nonfinite,
    )

    assert completed.stderr == ""
    assert report["schema_version"] == MOE_ALL_TO_ALL_CONTROL_VERSION
    assert report["runtime"] == {
        "backend": "gloo",
        "device": "cpu",
        "dtype": "torch.float64",
        "process_start_method": "spawn",
        "rendezvous": "temporary-file-store",
        "torch_version": report["runtime"]["torch_version"],
        "world_size": 2,
    }
    assert report["fixture"] == {
        "capacity_or_drop_policy": "none; every assignment is dispatched",
        "combine_weight_policy": "preserve selected softmax probability",
        "dispatch_float_columns": ["hidden_state", "combine_weight"],
        "expert_ownership": {"expert_0": 0, "expert_1": 1},
        "expert_parameters": {
            "expert_0": {"bias": 0.5, "weight": 2.0},
            "expert_1": {"bias": 1.0, "weight": -3.0},
        },
        "local_hidden_states_by_rank": [
            [[-1.0], [2.0], [-2.0]],
            [[1.0]],
        ],
        "metadata_columns": [
            "source_rank",
            "source_local_index",
            "global_token_id",
            "expert_id",
        ],
        "router_weight": [[1.0], [-1.0]],
        "top_k": 1,
    }
    assert report["process_observation"] == {
        "distinct_worker_process_count": 2,
        "raw_process_ids_published": False,
    }
    workers = report["worker_reports"]
    assert [worker["rank"] for worker in workers] == [0, 1]
    assert all("process_id" not in worker for worker in workers)
    assert [worker["owned_expert_id"] for worker in workers] == [0, 1]
    assert workers[0]["selected_expert_indices"] == [1, 0, 1]
    assert workers[1]["selected_expert_indices"] == [0]
    assert workers[0]["selected_probabilities"] == pytest.approx(
        [
            0.8807970779778823,
            0.9820137900379085,
            0.9820137900379085,
        ]
    )
    assert workers[1]["selected_probabilities"] == pytest.approx(
        [0.8807970779778823]
    )
    assert workers[0]["trace"]["send_counts_by_owner"] == [1, 2]
    assert workers[1]["trace"]["send_counts_by_owner"] == [1, 0]
    assert workers[0]["trace"]["received_counts_by_source"] == [1, 1]
    assert workers[1]["trace"]["received_counts_by_source"] == [2, 0]
    assert workers[0]["trace"]["owner_received_metadata"] == [
        [0, 1, 1, 0],
        [1, 0, 3, 0],
    ]
    assert workers[1]["trace"]["owner_received_metadata"] == [
        [0, 0, 0, 1],
        [0, 2, 2, 1],
    ]
    assert workers[0]["trace"]["owner_raw_expert_outputs"] == [
        [4.5],
        [2.5],
    ]
    assert workers[1]["trace"]["owner_raw_expert_outputs"] == [
        [4.0],
        [7.0],
    ]
    assert workers[0]["trace"]["return_arrival_metadata"] == [
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 2, 2, 1],
    ]
    assert workers[1]["trace"]["return_arrival_metadata"] == [
        [1, 0, 3, 0]
    ]
    comparison = report["comparison"]
    assert comparison["source_to_owner_token_counts"] == [[1, 2], [1, 0]]
    assert comparison["owner_from_source_token_counts"] == [[1, 1], [2, 0]]
    assert comparison["distributed_outputs_by_global_token_id"] == pytest.approx(
        [
            3.5231883119115293,
            4.419062055170588,
            6.874096530265359,
            2.201992694944706,
        ],
        abs=1e-15,
    )
    assert comparison[
        "single_process_oracle_outputs_by_global_token_id"
    ] == pytest.approx(
        comparison["distributed_outputs_by_global_token_id"],
        abs=1e-15,
    )
    assert comparison["distributed_vs_oracle_max_abs_difference"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert comparison[
        "rank_zero_metadata_free_vs_correct_max_abs_difference"
    ] == pytest.approx(0.8958737432590591, abs=1e-15)
    assert comparison[
        "rank_one_metadata_free_vs_correct_max_abs_difference"
    ] == 0.0
    assert comparison["logical_tensor_payload_bytes_sent_by_rank"] == [256, 160]
    assert comparison["logical_tensor_payload_bytes_sent_total"] == 416
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "capacity_drop_reroute_or_dropless_executed": False,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "distributed_autograd_backward_or_optimizer_executed": False,
        "owner_only_expert_parameter_placement_executed": True,
        "owner_to_source_output_and_metadata_return_executed": True,
        "real_two_process_same_host_gloo_process_group_executed": True,
        "replicated_router_executed": True,
        "single_process_forward_oracle_compared": True,
        "source_metadata_scatter_and_gate_combine_executed": True,
        "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
        "token_to_owner_float_and_metadata_dispatch_executed": True,
        "variable_split_all_to_all_single_count_exchange_executed": True,
        "wire_bytes_protocol_overhead_or_packet_capture_measured": False,
    }
    assert report["report_fingerprint"] == (
        "sha256:51c77e2499d84d5cf5500a5f5c1143b2979f3d70f1755114c34900b55299a61c"
    )
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
