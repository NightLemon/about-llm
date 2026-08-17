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
    MOE_ALL_TO_ALL_TRAINING_CONTROL_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "moe_all_to_all_training_control.py"
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
def test_two_process_gloo_moe_all_to_all_forward_backward_and_sgd() -> None:
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
    assert report["schema_version"] == MOE_ALL_TO_ALL_TRAINING_CONTROL_VERSION
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
    assert report["fixture"] == {
        "capacity_or_drop_policy": "none; every assignment is dispatched",
        "combine_weight_policy": "preserve selected softmax probability",
        "expert_ownership": {"expert_0": 0, "expert_1": 1},
        "expert_parameters": {
            "expert_0": {"bias": 0.5, "weight": 2.0},
            "expert_1": {"bias": 1.0, "weight": -3.0},
        },
        "learning_rate": 0.01,
        "local_hidden_states_by_rank": [
            [[-1.0], [2.0], [-2.0]],
            [[1.0]],
        ],
        "local_targets_by_rank": [
            [[0.25], [-0.5], [1.0]],
            [[-1.5]],
        ],
        "loss": "global mean squared error over four scalar targets",
        "metadata_columns": [
            "source_rank",
            "source_local_index",
            "global_token_id",
            "expert_id",
        ],
        "optimizer": "SGD without momentum or weight decay",
        "router_weight": [[1.0], [-1.0]],
        "top_k": 1,
    }
    workers = report["worker_reports"]
    assert [worker["rank"] for worker in workers] == [0, 1]
    assert all("process_id" not in worker for worker in workers)
    assert [worker["owned_expert_id"] for worker in workers] == [0, 1]
    assert [worker["selected_expert_indices"] for worker in workers] == [
        [1, 0, 1],
        [0],
    ]
    assert [worker["source_to_owner_counts"] for worker in workers] == [
        [1, 2],
        [1, 0],
    ]
    assert [worker["owner_from_source_counts"] for worker in workers] == [
        [1, 1],
        [2, 0],
    ]
    assert workers[0]["router_gradient_before_all_reduce"] == [
        [1.8045724077794292],
        [-1.8045724077794323],
    ]
    assert workers[1]["router_gradient_before_all_reduce"] == [
        [0.48585685772479353],
        [-0.48585685772479265],
    ]
    assert workers[0]["router_gradient_after_all_reduce"] == [
        [2.2904292655042227],
        [-2.290429265504225],
    ]
    assert workers[1]["router_gradient_after_all_reduce"] == workers[0][
        "router_gradient_after_all_reduce"
    ]
    expected_counts = {
        "autograd_payload_backward_all_to_all_single": 2,
        "autograd_payload_forward_all_to_all_single": 4,
        "nondifferentiable_count_or_metadata_all_to_all_single": 6,
        "router_gradient_all_reduce": 1,
    }
    assert all(
        worker["authored_collective_call_counts"] == expected_counts
        for worker in workers
    )
    comparison = report["comparison"]
    assert comparison["distributed_outputs_before_step_by_global_token_id"] == [
        [3.5231883119115293],
        [4.419062055170588],
        [6.874096530265359],
        [2.201992694944706],
    ]
    assert comparison["distributed_outputs_after_step_by_global_token_id"] == [
        [3.4025704512978336],
        [4.245112885397256],
        [6.678486844293563],
        [2.097729901341357],
    ]
    assert comparison["distributed_hidden_gradients_by_global_token_id"] == [
        [-5.699177157478319],
        [5.2215645378941495],
        [-9.378932784079748],
        [4.232418063852349],
    ]
    assert comparison["distributed_global_mean_loss_before_step"] == pytest.approx(
        20.78017329703821,
        abs=1e-15,
    )
    assert comparison["distributed_global_mean_loss_after_step"] == pytest.approx(
        19.41091750734501,
        abs=1e-15,
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
    assert comparison["distributed_global_mean_loss_after_step"] < comparison[
        "distributed_global_mean_loss_before_step"
    ]
    oracle = report["single_process_oracle"]
    assert oracle["router_gradient"] == [
        [2.2904292655042227],
        [-2.290429265504225],
    ]
    assert oracle["expert_weight_gradients"] == [
        [[6.460938946431114]],
        [[-7.209951147135929]],
    ]
    assert oracle["expert_bias_gradients"] == [
        [4.045645560316248],
        [4.325729248768723],
    ]
    assert oracle["router_weight_after_step"] == [
        [0.9770957073449578],
        [-0.9770957073449578],
    ]
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "authored_autograd_all_to_all_forward_backward_executed": True,
        "capacity_drop_reroute_or_dropless_executed": False,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "one_sgd_optimizer_step_executed": True,
        "optimizer_momentum_weight_decay_or_state_resume_executed": False,
        "owner_local_expert_parameter_gradients_executed": True,
        "owner_only_expert_parameter_placement_executed": True,
        "post_step_distributed_forward_evaluation_executed": True,
        "pytorch_distributed_nn_functional_wrapper_executed": False,
        "real_two_process_same_host_gloo_process_group_executed": True,
        "replicated_router_gradient_sum_all_reduce_executed": True,
        "reverse_split_hidden_and_gate_gradient_communication_executed": True,
        "single_process_global_mean_mse_oracle_compared": True,
        "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
        "torch_distributed_autograd_rpc_context_executed": False,
        "variable_split_token_and_metadata_dispatch_return_executed": True,
        "wire_bytes_protocol_overhead_or_collective_profiler_measured": False,
    }
    assert report["report_fingerprint"] == (
        "sha256:f577b29dd9e1ccc6def8c1fa156a7aba40a352d883646911a603c06f5adca67c"
    )
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
