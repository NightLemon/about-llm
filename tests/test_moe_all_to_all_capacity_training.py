from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import torch.distributed as dist

from about_llm.from_scratch.moe_all_to_all_training import (
    MOE_ALL_TO_ALL_CAPACITY_TRAINING_CONTROL_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "moe_all_to_all_capacity_training_control.py"
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
def test_capacity_drop_kept_only_all_to_all_backward_and_sgd() -> None:
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
    assert report["schema_version"] == (
        MOE_ALL_TO_ALL_CAPACITY_TRAINING_CONTROL_VERSION
    )
    assert report["runtime"] == {
        "backend": "gloo",
        "device": "cpu",
        "dtype": "torch.float64",
        "process_start_method": "spawn",
        "rendezvous": "temporary-file-store",
        "torch_version": report["runtime"]["torch_version"],
        "world_size": 2,
    }
    assert report["process_observation"] == {
        "distinct_worker_process_count": 2,
        "raw_process_ids_published": False,
    }
    workers = report["worker_reports"]
    assert [worker["rank"] for worker in workers] == [0, 1]
    assert all("process_id" not in worker for worker in workers)
    assert [worker["global_keep_mask"] for worker in workers] == [
        [False, True, True, False],
        [False, True, True, False],
    ]
    assert [worker["local_keep_mask"] for worker in workers] == [
        [False, True, True],
        [False],
    ]
    assert [worker["selected_counts_by_expert"] for worker in workers] == [
        [2, 2],
        [2, 2],
    ]
    assert [worker["kept_counts_by_expert"] for worker in workers] == [
        [1, 1],
        [1, 1],
    ]
    assert [worker["source_to_owner_counts"] for worker in workers] == [
        [1, 1],
        [0, 0],
    ]
    assert [worker["owner_from_source_counts"] for worker in workers] == [
        [1, 0],
        [1, 0],
    ]
    assert workers[1]["return_arrival_metadata"] == []
    expected_counts = {
        "autograd_payload_backward_all_to_all_single": 2,
        "autograd_payload_forward_all_to_all_single": 4,
        "capacity_route_all_gather": 4,
        "nondifferentiable_count_or_metadata_all_to_all_single": 6,
        "router_gradient_all_reduce": 1,
    }
    assert all(
        worker["authored_collective_call_counts"] == expected_counts
        for worker in workers
    )
    comparison = report["comparison"]
    assert comparison["distributed_outputs_before_step_by_global_token_id"] == [
        [0.0],
        [4.419062055170588],
        [6.874096530265359],
        [0.0],
    ]
    assert comparison["distributed_outputs_after_step_by_global_token_id"] == [
        [0.0],
        [4.29693726711294],
        [6.726949482533174],
        [0.0],
    ]
    assert comparison["distributed_hidden_gradients_by_global_token_id"] == [
        [0.0],
        [5.2215645378941495],
        [-9.378932784079748],
        [0.0],
    ]
    assert comparison["distributed_global_mean_loss_before_step"] == (
        15.253670387373656
    )
    assert comparison["distributed_global_mean_loss_after_step"] == (
        14.530264380025987
    )
    assert comparison["output_before_step_max_abs_difference"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert comparison["output_after_step_max_abs_difference"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert comparison["hidden_gradient_max_abs_difference"] == pytest.approx(
        0.0,
        abs=1e-15,
    )
    assert comparison["router_gradient_max_abs_difference_by_rank"] == pytest.approx(
        [0.0, 0.0],
        abs=1e-15,
    )
    assert comparison[
        "owned_expert_gradient_max_abs_difference_by_rank"
    ] == pytest.approx([0.0, 0.0], abs=1e-15)
    assert comparison[
        "post_step_parameter_max_abs_difference_by_rank"
    ] == pytest.approx([0.0, 0.0], abs=1e-15)
    assert comparison["distributed_hidden_gradients_by_global_token_id"][0] == [0.0]
    assert comparison["distributed_hidden_gradients_by_global_token_id"][3] == [0.0]
    assert comparison["distributed_global_mean_loss_after_step"] < comparison[
        "distributed_global_mean_loss_before_step"
    ]
    oracle = report["single_process_oracle"]
    assert oracle["router_gradient"] == [
        [1.1172448546425442],
        [-1.1172448546425469],
    ]
    assert oracle["expert_weight_gradients"] == [
        [[4.830586772229733]],
        [[-5.768443796734413]],
    ]
    assert oracle["expert_bias_gradients"] == [
        [2.4152933861148664],
        [2.8842218983672065],
    ]
    assert oracle["router_weight_after_step"] == [
        [0.9888275514535746],
        [-0.9888275514535745],
    ]
    assert oracle["expert_weights_after_step"] == [
        [[1.9516941322777026]],
        [[-2.9423155620326558]],
    ]
    assert oracle["expert_biases_after_step"] == [
        [0.47584706613885136],
        [0.9711577810163279],
    ]
    assert workers[0]["router_gradient_before_all_reduce"] == [
        [1.1172448546425442],
        [-1.1172448546425469],
    ]
    assert workers[1]["router_gradient_before_all_reduce"] == [
        [0.0],
        [0.0],
    ]
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "authored_autograd_reverse_all_to_all_backward_executed": True,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "dropped_token_zero_output_and_task_gradient_executed": True,
        "global_score_priority_drop_capacity_collective_executed": True,
        "kept_only_variable_split_dispatch_return_executed": True,
        "one_sgd_optimizer_step_executed": True,
        "optimizer_momentum_weight_decay_or_state_resume_executed": False,
        "owner_local_expert_parameter_gradients_executed": True,
        "owner_only_expert_parameter_placement_executed": True,
        "post_step_distributed_capacity_forward_executed": True,
        "real_two_process_same_host_gloo_process_group_executed": True,
        "replicated_router_gradient_sum_all_reduce_executed": True,
        "reroute_dropless_shared_or_fine_grained_experts_executed": False,
        "single_process_capacity_training_oracle_compared": True,
        "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
        "wire_bytes_protocol_overhead_or_collective_profiler_measured": False,
        "zero_assignment_source_rank_forward_backward_executed": True,
    }
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == (
        "sha256:33f11f199b9668c3600ce870cd8369c965cf9daad4bb716fe57fdb751373042e"
    )
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
