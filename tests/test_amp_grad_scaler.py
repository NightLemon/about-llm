from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from about_llm.finetuning.amp_scaler import (  # noqa: E402
    AMP_GRAD_SCALER_CONTROL_VERSION,
    run_cpu_amp_grad_scaler_control,
)

pytestmark = pytest.mark.formula

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "single-gpu-finetuning" / "amp_grad_scaler_control.py"


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return run_cpu_amp_grad_scaler_control().to_dict()


def test_finetuning_package_lazy_exports_control() -> None:
    from about_llm import finetuning

    assert finetuning.AMP_GRAD_SCALER_CONTROL_VERSION == AMP_GRAD_SCALER_CONTROL_VERSION
    assert finetuning.run_cpu_amp_grad_scaler_control is run_cpu_amp_grad_scaler_control


def test_correct_unscale_then_clip_matches_full_batch(report: dict[str, object]) -> None:
    clip = report["clip_ordering"]
    assert isinstance(clip, dict)
    reference = clip["full_batch_reference"]
    correct = clip["correct_unscale_then_clip"]
    assert isinstance(reference, dict)
    assert isinstance(correct, dict)
    assert correct["scaled_gradient_before_ordering"] == 24.0
    assert correct["clip_input_gradient"] == 3.0
    assert correct["reported_pre_clip_norm"] == 3.0
    assert correct["optimizer_gradient"] == reference["gradient_after_clip"]
    assert correct["parameter_after_step"] == reference["parameter_after_step"]
    assert correct["autocast_output_dtype"] == "torch.float16"


def test_clip_before_unscale_is_real_negative_control(report: dict[str, object]) -> None:
    clip = report["clip_ordering"]
    assert isinstance(clip, dict)
    correct = clip["correct_unscale_then_clip"]
    wrong = clip["wrong_clip_then_unscale"]
    assert isinstance(correct, dict)
    assert isinstance(wrong, dict)
    assert wrong["reported_pre_clip_norm"] == 24.0
    assert wrong["clip_input_gradient"] == 24.0
    assert wrong["optimizer_gradient"] == pytest.approx(
        float(correct["optimizer_gradient"]) / 8.0,
        abs=1e-7,
    )
    assert wrong["parameter_after_step"] != correct["parameter_after_step"]


def test_nonfinite_accumulation_windows_skip_full_adamw_update(
    report: dict[str, object],
) -> None:
    section = report["overflow_and_resume"]
    assert isinstance(section, dict)
    initial = section["initial_finite_adamw_step"]
    overflows = section["overflow_windows"]
    assert isinstance(initial, dict)
    assert isinstance(overflows, list)
    assert initial["optimizer_step_executed"] is True
    assert initial["optimizer_state_after"]["step"] == 1
    assert [(item["scale_before"], item["scale_after"]) for item in overflows] == [
        (8.0, 4.0),
        (4.0, 2.0),
        (2.0, 1.0),
    ]
    for item in overflows:
        assert item["microbatch_count"] == 2
        assert item["scaled_gradient_is_finite"] is False
        assert item["scaled_gradient"] is None
        assert item["unscaled_gradient"] is None
        assert item["optimizer_step_executed"] is False
        assert item["parameter_before"] == item["parameter_after"]
        assert item["optimizer_state_before"] == item["optimizer_state_after"]


def test_restored_scaler_matches_uninterrupted_and_omission_diverges(
    report: dict[str, object],
) -> None:
    section = report["overflow_and_resume"]
    assertions = report["assertions"]
    assert isinstance(section, dict)
    assert isinstance(assertions, dict)
    checkpoint = section["checkpoint"]
    restored = section["restored_with_scaler_state"]
    omitted = section["restored_without_scaler_state"]
    assert checkpoint["grad_scaler_state"]["scale"] == 1.0
    assert checkpoint["optimizer_state"]["step"] == 1
    assert restored["scale_before"] == 1.0
    assert restored["scaled_gradient"] == 10000.0
    assert restored["optimizer_step_executed"] is True
    assert restored["optimizer_state_after"]["step"] == 2
    assert omitted["scale_before"] == 8.0
    assert omitted["scaled_gradient"] is None
    assert omitted["optimizer_step_executed"] is False
    assert omitted["optimizer_state_after"]["step"] == 1
    assert assertions["restored_state_matches_uninterrupted_exactly"] is True
    assert assertions["omitted_scaler_state_diverges_from_restored_parameter"] is True


def test_scope_does_not_overclaim(report: dict[str, object]) -> None:
    scope = report["scope"]
    assert isinstance(scope, dict)
    assert scope == {
        "real_cpu_float16_autocast_executed": True,
        "real_cpu_grad_scaler_executed": True,
        "two_microbatch_scaled_accumulation_executed": True,
        "unscale_then_global_norm_clip_executed": True,
        "clip_before_unscale_negative_control_executed": True,
        "real_adamw_moments_and_step_executed": True,
        "intentional_nonfinite_accumulation_windows_executed": True,
        "in_memory_model_optimizer_scaler_resume_executed": True,
        "omitted_scaler_state_negative_control_executed": True,
        "cuda_or_gpu_kernel_executed": False,
        "file_checkpoint_or_process_restart_executed": False,
        "scheduler_rng_dataloader_or_distributed_state_executed": False,
        "target_model_trainer_tokenizer_or_dataset_executed": False,
        "convergence_quality_throughput_or_memory_proved": False,
    }

