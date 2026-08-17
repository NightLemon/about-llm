from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import torch.distributed as dist

from about_llm.from_scratch.moe_distributed_capacity import (
    DISTRIBUTED_MOE_CAPACITY_CONTROL_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "moe_distributed_capacity_control.py"
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
def test_two_process_gloo_moe_capacity_group_control() -> None:
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
    assert report["schema_version"] == DISTRIBUTED_MOE_CAPACITY_CONTROL_VERSION
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
        "capacity_factor": 0.5,
        "capacity_formula": (
            "ceil(capacity_factor * active_tokens * top_k / experts)"
        ),
        "expert_count": 2,
        "local_hidden_states_by_rank": [
            [[2.0], [1.0]],
            [[3.0], [0.5]],
        ],
        "overflow_policy": "drop",
        "router_weight": [[1.0], [0.0]],
        "top_k": 1,
    }
    assert report["process_observation"] == {
        "distinct_worker_process_count": 2,
        "raw_process_ids_published": False,
    }
    rank_reports = report["rank_reports"]
    assert [item["rank"] for item in rank_reports] == [0, 1]
    assert all("process_id" not in item for item in rank_reports)
    assert all(
        item["gathered_hidden_states"]
        == [[2.0], [1.0], [3.0], [0.5]]
        for item in rank_reports
    )
    assert all(
        item["global_active_token_count_after_all_reduce"] == 4
        and item["global_selected_counts_after_all_reduce"] == [4, 0]
        and item["collective_call_counts"]
        == {"all_gather": 1, "all_reduce": 2, "barrier": 1}
        for item in rank_reports
    )
    assert all(
        item["local_independent_route"]["expert_capacity"] == 1
        and item["local_independent_route"]["kept_mask"]
        == [[True], [False]]
        and item["local_independent_route"]["expert_counts_after_capacity"]
        == [1, 0]
        for item in rank_reports
    )
    global_route = rank_reports[0]["collective_global_route"]
    assert all(
        item["collective_global_route"] == global_route for item in rank_reports
    )
    assert global_route["expert_capacity"] == 1
    assert global_route["selected_expert_indices"] == [[0]] * 4
    assert global_route["kept_mask"] == [
        [False],
        [False],
        [True],
        [False],
    ]
    assert global_route["expert_counts_before_capacity"] == [4, 0]
    assert global_route["expert_counts_after_capacity"] == [1, 0]
    assert global_route["pre_policy_capacity_excess_by_group"] == [[3, 0]]
    assert global_route["post_policy_capacity_excess_by_group"] == [[0, 0]]
    assert global_route["dropped_assignments"] == 3
    assert report["comparison"] == {
        "collective_global_kept_assignments": 1,
        "collective_minus_independent_kept_assignments": -1,
        "independent_rank_local_kept_assignments": 2,
        "rank_one_collective_vs_local_output_max_abs_difference": 0.0,
        "rank_zero_collective_vs_local_output_max_abs_difference": pytest.approx(
            0.9640275800758169
        ),
    }
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "collective_capacity_group_competition_executed": True,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "distributed_autograd_or_ddp_backward_executed": False,
        "expert_parallel_all_to_all_or_reduce_scatter_executed": False,
        "global_active_token_count_all_reduce_executed": True,
        "global_selected_assignment_count_all_reduce_executed": True,
        "hidden_state_all_gather_for_replicated_global_routing_executed": True,
        "real_two_process_same_host_gloo_process_group_executed": True,
        "replicated_router_and_experts_used": True,
        "temporary_file_store_rendezvous_executed": True,
        "throughput_memory_scaling_convergence_or_quality_proved": False,
    }
    assert report["report_fingerprint"] == (
        "sha256:9e342b0ba87b0e11ebf43eb41eaca0be165a3ee365cc9285bbe5a2f2923887be"
    )
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
