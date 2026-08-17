from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "ddp_amp_overflow_consensus_control.py"
)


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise AssertionError(f"non-standard JSON number: {value}")


@pytest.mark.slow
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_process_ddp_amp_overflow_consensus_control() -> None:
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
        parse_constant=_reject_nonfinite_json,
    )

    assert completed.stderr == ""
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
    assert report["implementation"] == (
        "about-llm.ddp-amp-overflow-consensus-control.v1"
    )
    assert report["runtime"] == {
        "autocast_dtype": "torch.float16",
        "backend": "gloo",
        "backoff_factor": 0.5,
        "device": "cpu",
        "growth_interval": 1000,
        "initial_learning_rate": 0.01,
        "initial_scale": 8.0,
        "optimizer": "torch.optim.AdamW",
        "parameter_dtype": "torch.float32",
        "process_start_method": "spawn",
        "rendezvous": "temporary-file-store",
        "scheduler": "torch.optim.lr_scheduler.StepLR",
        "torch_version": report["runtime"]["torch_version"],
        "world_size": 2,
    }
    assert all(report["assertions"].values())

    ranks = report["rank_reports"]
    assert [item["rank"] for item in ranks] == [0, 1]
    pre = [
        item["scenarios"]["pre_reduction_rank_local_overflow"]
        for item in ranks
    ]
    local = [
        item["scenarios"]["post_reduction_rank0_fault_without_consensus"]
        for item in ranks
    ]
    gated = [
        item["scenarios"]["post_reduction_rank0_fault_with_global_gate"]
        for item in ranks
    ]

    expected_warmup = {
        "grad_scaler": {
            "backoff_factor": 0.5,
            "growth_factor": 2.0,
            "growth_interval": 1000,
            "growth_tracker": 1,
            "scale": 8.0,
        },
        "learning_rate": 0.005,
        "optimizer": {
            "exp_avg": pytest.approx(0.1),
            "exp_avg_sq": pytest.approx(0.001),
            "step": 1,
        },
        "parameter": pytest.approx(0.99),
        "scheduler": {"last_epoch": 1, "step_count": 2},
    }
    assert all(
        scenario["training_state_before"] == expected_warmup
        for scenario in (*pre, *local, *gated)
    )

    assert [
        item["rank_local_scaled_gradient_after_no_sync"] for item in pre
    ] == [
        {"finite": False, "value": None},
        {"finite": True, "value": 8.0},
    ]
    assert [item["scaled_gradient_after_ddp_reduction"] for item in pre] == [
        {"finite": False, "value": None},
        {"finite": False, "value": None},
    ]
    assert [item["local_nonfinite_before_consensus"] for item in pre] == [
        True,
        True,
    ]
    assert [item["optimizer_step_executed"] for item in pre] == [False, False]
    assert pre[0]["training_state_after"] == pre[1]["training_state_after"]
    assert pre[0]["training_state_after"]["grad_scaler"] == {
        "backoff_factor": 0.5,
        "growth_factor": 2.0,
        "growth_interval": 1000,
        "growth_tracker": 0,
        "scale": 4.0,
    }
    assert pre[0]["training_state_after"]["optimizer"]["step"] == 1
    assert pre[0]["training_state_after"]["scheduler"]["last_epoch"] == 1

    assert [item["scaled_gradient_after_ddp_reduction"] for item in local] == [
        {"finite": True, "value": 8.0},
        {"finite": True, "value": 8.0},
    ]
    assert [item["gradient_after_unscale_before_step"] for item in local] == [
        {"finite": False, "value": None},
        {"finite": True, "value": 1.0},
    ]
    assert [item["global_nonfinite_after_max_all_reduce"] for item in local] == [
        None,
        None,
    ]
    assert [item["optimizer_step_executed"] for item in local] == [False, True]
    assert [item["scheduler_step_called"] for item in local] == [False, True]
    assert [item["training_state_after"]["grad_scaler"] for item in local] == [
        {
            "backoff_factor": 0.5,
            "growth_factor": 2.0,
            "growth_interval": 1000,
            "growth_tracker": 0,
            "scale": 4.0,
        },
        {
            "backoff_factor": 0.5,
            "growth_factor": 2.0,
            "growth_interval": 1000,
            "growth_tracker": 2,
            "scale": 8.0,
        },
    ]
    assert [item["training_state_after"]["optimizer"]["step"] for item in local] == [
        1,
        2,
    ]
    assert [
        item["training_state_after"]["scheduler"]["last_epoch"] for item in local
    ] == [1, 2]
    assert [item["training_state_after"]["learning_rate"] for item in local] == [
        0.005,
        0.0025,
    ]
    assert local[0]["training_state_after"]["parameter"] == pytest.approx(0.99)
    assert local[1]["training_state_after"]["parameter"] == pytest.approx(0.985)

    assert [item["local_nonfinite_before_consensus"] for item in gated] == [
        True,
        False,
    ]
    assert [item["global_nonfinite_after_max_all_reduce"] for item in gated] == [
        True,
        True,
    ]
    assert [item["scaler_step_called"] for item in gated] == [False, False]
    assert [item["optimizer_step_executed"] for item in gated] == [False, False]
    assert gated[0]["training_state_after"] == gated[1]["training_state_after"]
    assert gated[0]["training_state_after"]["grad_scaler"] == {
        "backoff_factor": 0.5,
        "growth_factor": 2.0,
        "growth_interval": 1000,
        "growth_tracker": 1,
        "scale": 4.0,
    }
    for item in gated:
        for field in ("parameter", "optimizer", "scheduler", "learning_rate"):
            assert item["training_state_after"][field] == item[
                "training_state_before"
            ][field]

    assert report["scope"] == {
        "builtin_default_ddp_reducer_executed": True,
        "builtin_reducer_collective_count_directly_instrumented": False,
        "checkpoint_resume_crash_recovery_or_elastic_restart_executed": False,
        "common_manual_scaler_backoff_after_global_skip_executed": True,
        "convergence_quality_throughput_memory_or_fault_rate_proved": False,
        "cpu_float16_autocast_and_cpu_grad_scaler_executed": True,
        "cuda_nccl_gpu_multi_node_or_remote_host_executed": False,
        "custom_ddp_comm_hook_or_conditional_parameter_graph_executed": False,
        "finite_adamw_and_steplr_warmup_executed": True,
        "fsdp_zero_tensor_pipeline_or_expert_parallel_executed": False,
        "global_nonfinite_max_all_reduce_gate_executed": True,
        "multiple_parameters_or_multiple_gradient_buckets_executed": False,
        "native_grad_scaler_found_inf_state_synchronized": False,
        "natural_model_or_data_induced_overflow_executed": False,
        "one_no_sync_microbatch_then_one_sync_microbatch_executed": True,
        "post_reduction_pre_unscale_rank0_fault_injection_executed": True,
        "post_unscale_gradient_corruption_executed": False,
        "pre_reduction_rank_local_nonfinite_fault_executed": True,
        "rank_local_scaler_divergence_negative_control_executed": True,
        "real_two_process_same_host_gloo_process_group_executed": True,
        "target_model_tokenizer_dataset_or_trainer_executed": False,
    }
