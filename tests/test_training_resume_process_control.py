from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from about_llm.finetuning.training_resume import (
    TRAINING_RESUME_CHECKPOINT_VERSION,
    TRAINING_RESUME_CONTROL_VERSION,
    run_training_resume_process_control,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "checkpoint_resume_control.py"
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=150,
    )
    assert completed.stderr == b""
    assert b"Infinity" not in completed.stdout
    assert b"NaN" not in completed.stdout
    decoded = json.loads(
        completed.stdout.decode("utf-8"),
        parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"),
    )
    assert isinstance(decoded, dict)
    return decoded


def test_finetuning_package_lazy_exports_process_control() -> None:
    from about_llm import finetuning

    assert finetuning.TRAINING_RESUME_CONTROL_VERSION == TRAINING_RESUME_CONTROL_VERSION
    assert finetuning.TRAINING_RESUME_CHECKPOINT_VERSION == (
        TRAINING_RESUME_CHECKPOINT_VERSION
    )
    assert (
        finetuning.run_training_resume_process_control
        is run_training_resume_process_control
    )


def test_real_process_exit_file_checkpoint_and_exact_resume(
    report: dict[str, Any],
) -> None:
    assert report["implementation"] == TRAINING_RESUME_CONTROL_VERSION
    processes = report["processes"]
    assert processes["phase1_process_exited_before_resume_launch"] is True
    assert processes["phase1_and_resume_pids_are_distinct"] is True
    assert processes["phase1_pid"] != processes["resumed_pid"]

    checkpoint = report["checkpoint"]
    assert checkpoint["schema_version"] == TRAINING_RESUME_CHECKPOINT_VERSION
    assert 1 <= checkpoint["bytes"] <= 16 * 1024 * 1024
    assert checkpoint["sha256"].startswith("sha256:")
    assert len(checkpoint["sha256"]) == 71
    assert checkpoint["loaded_with_torch_weights_only"] is True
    assert checkpoint["components"] == [
        "model",
        "optimizer",
        "scheduler",
        "grad_scaler",
        "torch_cpu_rng_state",
        "python_rng_state",
        "data_stream.permutation",
        "data_stream.cursor",
        "data_stream.epoch",
        "data_stream.generator_state",
        "progress.next_attempt_index",
        "progress.successful_updates",
        "dataset_sha256",
    ]

    uninterrupted = report["uninterrupted"]
    split = report["split_resume"]
    assert split["phase1_trace"] == uninterrupted["trace"][:4]
    assert split["resumed_trace"] == uninterrupted["trace"][4:]
    assert split["terminal"] == uninterrupted["terminal"]
    assert uninterrupted["terminal"]["progress"] == {
        "next_attempt_index": 8,
        "successful_updates": 5,
        "optimizer_step": 5,
        "scheduler_last_epoch": 5,
        "scheduler_step_count": 6,
        "learning_rate": 0.005,
        "grad_scaler_scale": 1.0,
        "data_epoch": 1,
        "data_cursor": 8,
    }


def test_overflow_skips_optimizer_and_scheduler_together(
    report: dict[str, Any],
) -> None:
    trace = report["uninterrupted"]["trace"]
    assert [item["scale_before"] for item in trace[:5]] == [8.0, 8.0, 4.0, 2.0, 1.0]
    assert [item["scale_after"] for item in trace[:5]] == [8.0, 4.0, 2.0, 1.0, 1.0]
    assert trace[0]["optimizer_step_executed"] is True
    for item in trace[1:4]:
        assert item["loss"] is None
        assert item["gradients_finite_after_unscale"] is False
        assert item["optimizer_step_before"] == item["optimizer_step_after"] == 1
        assert item["scheduler_last_epoch_before"] == 1
        assert item["scheduler_last_epoch_after"] == 1
        assert item["scheduler_step_count_before"] == 2
        assert item["scheduler_step_count_after"] == 2
        assert item["model_fingerprint_before"] == item["model_fingerprint_after"]
    assert trace[4]["optimizer_step_executed"] is True
    assert trace[4]["scheduler_last_epoch_after"] == 2
    assert trace[4]["learning_rate_after"] == 0.01


def test_each_omission_control_has_a_specific_counterfactual(
    report: dict[str, Any],
) -> None:
    correct = report["split_resume"]["resumed_trace"]
    controls = report["negative_controls"]

    wrong_schedule = controls["advance_scheduler_on_overflow"]["trace"]
    assert [item["scheduler_last_epoch_after"] for item in wrong_schedule[1:4]] == [
        2,
        3,
        4,
    ]
    assert all(item["optimizer_step_executed"] is False for item in wrong_schedule[1:4])

    omitted_scheduler = controls["omit_scheduler_state"]["trace"]
    assert omitted_scheduler[0]["scheduler_last_epoch_before"] == 0
    assert omitted_scheduler[0]["learning_rate_after"] == 0.02
    assert correct[0]["scheduler_last_epoch_before"] == 1
    assert correct[0]["learning_rate_after"] == 0.01

    omitted_scaler = controls["omit_grad_scaler_state"]["trace"]
    assert correct[0]["scale_before"] == 1.0
    assert correct[0]["optimizer_step_executed"] is True
    assert omitted_scaler[0]["scale_before"] == 8.0
    assert omitted_scaler[0]["scale_after"] == 4.0
    assert omitted_scaler[0]["optimizer_step_executed"] is False

    omitted_rng = controls["omit_rng_state"]["trace"]
    assert [item["batch_indices"] for item in omitted_rng] == [
        item["batch_indices"] for item in correct
    ]
    assert [item["python_factor"] for item in omitted_rng] != [
        item["python_factor"] for item in correct
    ]
    assert [item["dropout_mask_sha256"] for item in omitted_rng] != [
        item["dropout_mask_sha256"] for item in correct
    ]

    omitted_data = controls["omit_data_stream_state"]["trace"]
    assert [item["batch_indices"] for item in omitted_data] != [
        item["batch_indices"] for item in correct
    ]
    assert [item["python_factor"] for item in omitted_data] == [
        item["python_factor"] for item in correct
    ]
    assert [item["dropout_mask_sha256"] for item in omitted_data] == [
        item["dropout_mask_sha256"] for item in correct
    ]


def test_assertions_and_scope_do_not_overclaim(report: dict[str, Any]) -> None:
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "real_independent_os_processes_executed": True,
        "phase1_process_exit_and_checkpoint_reopen_executed": True,
        "real_cpu_float16_autocast_executed": True,
        "real_cpu_grad_scaler_executed": True,
        "real_adamw_and_step_lr_executed": True,
        "intentional_nonfinite_optimizer_skip_executed": True,
        "scheduler_skip_and_wrong_advance_control_executed": True,
        "torch_cpu_and_python_rng_restored": True,
        "stateful_shuffle_generator_permutation_cursor_epoch_restored": True,
        "weights_only_checkpoint_load_executed": True,
        "scheduler_scaler_rng_and_data_omission_controls_executed": True,
        "cuda_or_gpu_kernel_executed": False,
        "dataloader_worker_prefetch_or_distributed_state_executed": False,
        "target_model_trainer_tokenizer_or_dataset_executed": False,
        "crash_power_loss_atomicity_or_remote_storage_proved": False,
        "checkpoint_origin_authentication_or_confidentiality_proved": False,
        "convergence_quality_throughput_or_memory_proved": False,
    }
