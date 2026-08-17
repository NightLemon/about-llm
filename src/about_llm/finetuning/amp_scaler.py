"""CPU AMP/GradScaler controls for accumulation, overflow, and resume state.

The control intentionally uses CPU float16 autocast because its finite range
makes a small, deterministic scale-sensitive overflow fixture possible.  It
does not model CUDA kernels, Tensor Cores, target trainers, or model quality.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

AMP_GRAD_SCALER_CONTROL_VERSION = "about-llm.amp-grad-scaler-control.v1"
AMP_GRAD_SCALER_EVIDENCE_BOUNDARY = (
    "real_cpu_float16_autocast_and_cpu_grad_scaler_executed",
    "two_microbatch_scaled_gradient_accumulation_executed",
    "unscale_then_global_norm_clip_matches_full_batch_reference",
    "clip_before_unscale_negative_control_executed",
    "intentional_nonfinite_accumulation_windows_skip_adamw_steps",
    "in_memory_model_optimizer_and_scaler_state_resume_executed",
    "omitted_scaler_state_changes_next_step_decision",
    "cuda_or_gpu_kernel_executed=false",
    "file_checkpoint_or_process_restart_executed=false",
    "target_model_trainer_or_dataset_executed=false",
    "convergence_quality_throughput_or_memory_proved=false",
)


@dataclass(frozen=True)
class ClipPathObservation:
    """One scaled-gradient clipping order and its optimizer-visible result."""

    unscale_before_clip: bool
    scaled_gradient_before_ordering: float
    clip_input_gradient: float
    reported_pre_clip_norm: float
    gradient_after_clip_before_optional_unscale: float
    optimizer_gradient: float
    parameter_after_step: float
    scaler_scale_after: float
    autocast_output_dtype: str

    def to_dict(self) -> dict[str, bool | float | str]:
        return {
            "unscale_before_clip": self.unscale_before_clip,
            "scaled_gradient_before_ordering": self.scaled_gradient_before_ordering,
            "clip_input_gradient": self.clip_input_gradient,
            "reported_pre_clip_norm": self.reported_pre_clip_norm,
            "gradient_after_clip_before_optional_unscale": (
                self.gradient_after_clip_before_optional_unscale
            ),
            "optimizer_gradient": self.optimizer_gradient,
            "parameter_after_step": self.parameter_after_step,
            "scaler_scale_after": self.scaler_scale_after,
            "autocast_output_dtype": self.autocast_output_dtype,
        }


@dataclass(frozen=True)
class FullBatchClipReference:
    """Unscaled full-batch reference for the accumulation/clip fixture."""

    gradient_before_clip: float
    reported_pre_clip_norm: float
    gradient_after_clip: float
    parameter_after_step: float
    autocast_output_dtype: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "gradient_before_clip": self.gradient_before_clip,
            "reported_pre_clip_norm": self.reported_pre_clip_norm,
            "gradient_after_clip": self.gradient_after_clip,
            "parameter_after_step": self.parameter_after_step,
            "autocast_output_dtype": self.autocast_output_dtype,
        }


@dataclass(frozen=True)
class AdamWStateObservation:
    """The scalar fixture's complete per-parameter AdamW state."""

    step: int
    exp_avg: float | None
    exp_avg_sq: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "step": self.step,
            "exp_avg": self.exp_avg,
            "exp_avg_sq": self.exp_avg_sq,
        }


@dataclass(frozen=True)
class ScaledAdamWWindowObservation:
    """One real GradScaler window, including skip/update evidence."""

    label: str
    microbatch_count: int
    scale_before: float
    scale_after: float
    scaled_gradient_is_finite: bool
    scaled_gradient: float | None
    unscaled_gradient_is_finite: bool
    unscaled_gradient: float | None
    parameter_before: float
    parameter_after: float
    optimizer_state_before: AdamWStateObservation
    optimizer_state_after: AdamWStateObservation
    optimizer_step_executed: bool
    autocast_output_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "microbatch_count": self.microbatch_count,
            "scale_before": self.scale_before,
            "scale_after": self.scale_after,
            "scaled_gradient_is_finite": self.scaled_gradient_is_finite,
            "scaled_gradient": self.scaled_gradient,
            "unscaled_gradient_is_finite": self.unscaled_gradient_is_finite,
            "unscaled_gradient": self.unscaled_gradient,
            "parameter_before": self.parameter_before,
            "parameter_after": self.parameter_after,
            "optimizer_state_before": self.optimizer_state_before.to_dict(),
            "optimizer_state_after": self.optimizer_state_after.to_dict(),
            "optimizer_step_executed": self.optimizer_step_executed,
            "autocast_output_dtype": self.autocast_output_dtype,
        }


@dataclass(frozen=True)
class ScalerCheckpointObservation:
    """Explicit GradScaler and optimizer state at the in-memory split point."""

    scale: float
    growth_factor: float
    backoff_factor: float
    growth_interval: int
    growth_tracker: int
    parameter: float
    optimizer_state: AdamWStateObservation

    def to_dict(self) -> dict[str, Any]:
        return {
            "grad_scaler_state": {
                "scale": self.scale,
                "growth_factor": self.growth_factor,
                "backoff_factor": self.backoff_factor,
                "growth_interval": self.growth_interval,
                "growth_tracker": self.growth_tracker,
            },
            "parameter": self.parameter,
            "optimizer_state": self.optimizer_state.to_dict(),
        }


@dataclass(frozen=True)
class AMPGradScalerAnalysis:
    """Complete result of the CPU AMP sequencing and resume control."""

    torch_version: str
    clip_max_norm: float
    initial_scale: float
    full_batch_reference: FullBatchClipReference
    correct_unscale_then_clip: ClipPathObservation
    wrong_clip_then_unscale: ClipPathObservation
    initial_finite_adamw_step: ScaledAdamWWindowObservation
    overflow_windows: tuple[ScaledAdamWWindowObservation, ...]
    checkpoint: ScalerCheckpointObservation
    uninterrupted_after_checkpoint: ScaledAdamWWindowObservation
    restored_with_scaler_state: ScaledAdamWWindowObservation
    restored_without_scaler_state: ScaledAdamWWindowObservation
    assertions: dict[str, bool]
    scope: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": AMP_GRAD_SCALER_CONTROL_VERSION,
            "runtime": {
                "torch_version": self.torch_version,
                "device": "cpu",
                "parameter_dtype": "torch.float32",
                "autocast_dtype": "torch.float16",
                "optimizer_state_dtype": "torch.float32",
            },
            "fixture": {
                "clip_microbatch_inputs": [1.0, 2.0],
                "clip_max_norm": self.clip_max_norm,
                "initial_scale": self.initial_scale,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 1000,
                "borderline_unscaled_gradient": 10000.0,
            },
            "clip_ordering": {
                "full_batch_reference": self.full_batch_reference.to_dict(),
                "correct_unscale_then_clip": self.correct_unscale_then_clip.to_dict(),
                "wrong_clip_then_unscale": self.wrong_clip_then_unscale.to_dict(),
            },
            "overflow_and_resume": {
                "initial_finite_adamw_step": self.initial_finite_adamw_step.to_dict(),
                "overflow_windows": [item.to_dict() for item in self.overflow_windows],
                "checkpoint": self.checkpoint.to_dict(),
                "uninterrupted_after_checkpoint": (
                    self.uninterrupted_after_checkpoint.to_dict()
                ),
                "restored_with_scaler_state": self.restored_with_scaler_state.to_dict(),
                "restored_without_scaler_state": (
                    self.restored_without_scaler_state.to_dict()
                ),
            },
            "assertions": dict(self.assertions),
            "scope": dict(self.scope),
        }


class _ScalarLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.nn.functional.linear(inputs, self.weight)


@dataclass(frozen=True)
class _TrainingSnapshot:
    model: dict[str, Tensor]
    optimizer: dict[str, Any]
    scaler: dict[str, Any]


def _new_scaler() -> torch.amp.GradScaler:
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=8.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1000,
    )
    if not scaler.is_enabled():
        raise RuntimeError("CPU GradScaler is unavailable or disabled in this PyTorch build")
    return scaler


def _new_adamw_training_state(
    snapshot: _TrainingSnapshot | None = None,
    *,
    restore_scaler: bool = True,
) -> tuple[_ScalarLinear, torch.optim.AdamW, torch.amp.GradScaler]:
    model = _ScalarLinear()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    scaler = _new_scaler()
    if snapshot is not None:
        model.load_state_dict(copy.deepcopy(snapshot.model), strict=True)
        optimizer.load_state_dict(copy.deepcopy(snapshot.optimizer))
        if restore_scaler:
            scaler.load_state_dict(copy.deepcopy(snapshot.scaler))
    return model, optimizer, scaler


def _adamw_state(
    model: _ScalarLinear,
    optimizer: torch.optim.AdamW,
) -> AdamWStateObservation:
    state = optimizer.state.get(model.weight)
    if not state:
        return AdamWStateObservation(step=0, exp_avg=None, exp_avg_sq=None)
    fields = set(state)
    if fields != {"step", "exp_avg", "exp_avg_sq"}:
        raise AssertionError(f"unexpected AdamW state fields: {sorted(fields)}")
    step = state["step"]
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    if not isinstance(step, Tensor) or step.numel() != 1:
        raise AssertionError("AdamW step must be a scalar tensor")
    if not isinstance(exp_avg, Tensor) or exp_avg.shape != model.weight.shape:
        raise AssertionError("AdamW exp_avg shape drifted")
    if not isinstance(exp_avg_sq, Tensor) or exp_avg_sq.shape != model.weight.shape:
        raise AssertionError("AdamW exp_avg_sq shape drifted")
    return AdamWStateObservation(
        step=int(step.item()),
        exp_avg=float(exp_avg.item()),
        exp_avg_sq=float(exp_avg_sq.item()),
    )


def _finite_value(value: Tensor) -> tuple[bool, float | None]:
    finite = bool(torch.isfinite(value).all().item())
    return finite, float(value.item()) if finite else None


def _run_adamw_window(
    model: _ScalarLinear,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler,
    *,
    label: str,
    loss_multipliers: tuple[float, ...],
) -> ScaledAdamWWindowObservation:
    if not loss_multipliers:
        raise ValueError("loss_multipliers cannot be empty")
    optimizer.zero_grad(set_to_none=True)
    parameter_before = float(model.weight.item())
    optimizer_before = _adamw_state(model, optimizer)
    scale_before = float(scaler.get_scale())
    output_dtype = ""
    for multiplier in loss_multipliers:
        with torch.amp.autocast(device_type="cpu", dtype=torch.float16):
            output = model(torch.ones((1, 1), dtype=torch.float32))
            loss = output.float().sum() * multiplier
        output_dtype = str(output.dtype)
        scaler.scale(loss).backward()
    if model.weight.grad is None:
        raise AssertionError("scaled backward did not create a gradient")
    scaled_finite, scaled_gradient = _finite_value(model.weight.grad.detach())
    scaler.unscale_(optimizer)
    if model.weight.grad is None:
        raise AssertionError("unscale removed the gradient")
    unscaled_finite, unscaled_gradient = _finite_value(model.weight.grad.detach())
    scaler.step(optimizer)
    scaler.update()
    optimizer_after = _adamw_state(model, optimizer)
    parameter_after = float(model.weight.item())
    return ScaledAdamWWindowObservation(
        label=label,
        microbatch_count=len(loss_multipliers),
        scale_before=scale_before,
        scale_after=float(scaler.get_scale()),
        scaled_gradient_is_finite=scaled_finite,
        scaled_gradient=scaled_gradient,
        unscaled_gradient_is_finite=unscaled_finite,
        unscaled_gradient=unscaled_gradient,
        parameter_before=parameter_before,
        parameter_after=parameter_after,
        optimizer_state_before=optimizer_before,
        optimizer_state_after=optimizer_after,
        optimizer_step_executed=optimizer_after.step == optimizer_before.step + 1,
        autocast_output_dtype=output_dtype,
    )


def _run_clip_path(*, unscale_before_clip: bool) -> ClipPathObservation:
    model = _ScalarLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _new_scaler()
    optimizer.zero_grad(set_to_none=True)
    output_dtype = ""
    for input_value in (1.0, 2.0):
        with torch.amp.autocast(device_type="cpu", dtype=torch.float16):
            output = model(torch.tensor([[input_value]], dtype=torch.float32))
            loss = output.float().sum()
        output_dtype = str(output.dtype)
        scaler.scale(loss).backward()
    if model.weight.grad is None:
        raise AssertionError("scaled accumulation did not create a gradient")
    scaled_gradient = float(model.weight.grad.item())
    if unscale_before_clip:
        scaler.unscale_(optimizer)
        clip_input = float(model.weight.grad.item())
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=0.5, error_if_nonfinite=True
        )
        gradient_after_clip = float(model.weight.grad.item())
        optimizer_gradient = gradient_after_clip
    else:
        clip_input = float(model.weight.grad.item())
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=0.5, error_if_nonfinite=True
        )
        gradient_after_clip = float(model.weight.grad.item())
        scaler.unscale_(optimizer)
        optimizer_gradient = float(model.weight.grad.item())
    scaler.step(optimizer)
    scaler.update()
    return ClipPathObservation(
        unscale_before_clip=unscale_before_clip,
        scaled_gradient_before_ordering=scaled_gradient,
        clip_input_gradient=clip_input,
        reported_pre_clip_norm=float(norm.item()),
        gradient_after_clip_before_optional_unscale=gradient_after_clip,
        optimizer_gradient=optimizer_gradient,
        parameter_after_step=float(model.weight.item()),
        scaler_scale_after=float(scaler.get_scale()),
        autocast_output_dtype=output_dtype,
    )


def _run_full_batch_clip_reference() -> FullBatchClipReference:
    model = _ScalarLinear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cpu", dtype=torch.float16):
        output = model(torch.tensor([[1.0], [2.0]], dtype=torch.float32))
        loss = output.float().sum()
    loss.backward()
    if model.weight.grad is None:
        raise AssertionError("full-batch backward did not create a gradient")
    gradient_before_clip = float(model.weight.grad.item())
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=0.5, error_if_nonfinite=True
    )
    gradient_after_clip = float(model.weight.grad.item())
    optimizer.step()
    return FullBatchClipReference(
        gradient_before_clip=gradient_before_clip,
        reported_pre_clip_norm=float(norm.item()),
        gradient_after_clip=gradient_after_clip,
        parameter_after_step=float(model.weight.item()),
        autocast_output_dtype=str(output.dtype),
    )


def _snapshot(
    model: _ScalarLinear,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler,
) -> _TrainingSnapshot:
    return _TrainingSnapshot(
        model=copy.deepcopy(model.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        scaler=copy.deepcopy(scaler.state_dict()),
    )


def _checkpoint_observation(
    model: _ScalarLinear,
    optimizer: torch.optim.AdamW,
    scaler: torch.amp.GradScaler,
) -> ScalerCheckpointObservation:
    state = scaler.state_dict()
    expected = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if set(state) != expected:
        raise AssertionError(f"unexpected GradScaler state fields: {sorted(state)}")
    return ScalerCheckpointObservation(
        scale=float(state["scale"]),
        growth_factor=float(state["growth_factor"]),
        backoff_factor=float(state["backoff_factor"]),
        growth_interval=int(state["growth_interval"]),
        growth_tracker=int(state["_growth_tracker"]),
        parameter=float(model.weight.item()),
        optimizer_state=_adamw_state(model, optimizer),
    )


def run_cpu_amp_grad_scaler_control() -> AMPGradScalerAnalysis:
    """Execute real CPU AMP sequencing and a scale-sensitive resume control."""
    full_batch = _run_full_batch_clip_reference()
    correct_clip = _run_clip_path(unscale_before_clip=True)
    wrong_clip = _run_clip_path(unscale_before_clip=False)

    model, optimizer, scaler = _new_adamw_training_state()
    initial_finite = _run_adamw_window(
        model,
        optimizer,
        scaler,
        label="finite-state-initialization",
        loss_multipliers=(1.0,),
    )
    state_before_overflows = _adamw_state(model, optimizer)
    parameter_before_overflows = float(model.weight.item())
    overflow_windows = tuple(
        _run_adamw_window(
            model,
            optimizer,
            scaler,
            label=f"intentional-overflow-{index}",
            loss_multipliers=(1.0, float("inf")),
        )
        for index in range(1, 4)
    )
    checkpoint = _checkpoint_observation(model, optimizer, scaler)
    training_snapshot = _snapshot(model, optimizer, scaler)

    uninterrupted = _run_adamw_window(
        model,
        optimizer,
        scaler,
        label="borderline-uninterrupted",
        loss_multipliers=(10000.0,),
    )
    restored_model, restored_optimizer, restored_scaler = _new_adamw_training_state(
        training_snapshot
    )
    restored = _run_adamw_window(
        restored_model,
        restored_optimizer,
        restored_scaler,
        label="borderline-restored-with-scaler",
        loss_multipliers=(10000.0,),
    )
    omitted_model, omitted_optimizer, omitted_scaler = _new_adamw_training_state(
        training_snapshot,
        restore_scaler=False,
    )
    omitted = _run_adamw_window(
        omitted_model,
        omitted_optimizer,
        omitted_scaler,
        label="borderline-restored-without-scaler",
        loss_multipliers=(10000.0,),
    )

    overflow_scales = tuple(
        (item.scale_before, item.scale_after) for item in overflow_windows
    )
    assertions = {
        "correct_accumulation_pre_clip_matches_full_batch": (
            correct_clip.clip_input_gradient == full_batch.gradient_before_clip == 3.0
        ),
        "correct_post_clip_gradient_matches_full_batch": (
            correct_clip.optimizer_gradient == full_batch.gradient_after_clip
        ),
        "correct_parameter_update_matches_full_batch": (
            correct_clip.parameter_after_step == full_batch.parameter_after_step
        ),
        "clip_before_unscale_shrinks_optimizer_gradient_by_scale": math.isclose(
            wrong_clip.optimizer_gradient,
            correct_clip.optimizer_gradient / 8.0,
            rel_tol=0.0,
            abs_tol=1e-7,
        ),
        "initial_finite_window_executes_one_adamw_step": (
            initial_finite.optimizer_step_executed
            and initial_finite.optimizer_state_after.step == 1
        ),
        "overflow_scale_transitions_are_8_4_2_1": overflow_scales
        == ((8.0, 4.0), (4.0, 2.0), (2.0, 1.0)),
        "overflow_windows_skip_entire_adamw_update": all(
            not item.optimizer_step_executed
            and item.parameter_after == parameter_before_overflows
            and item.optimizer_state_after == state_before_overflows
            for item in overflow_windows
        ),
        "checkpoint_contains_nonempty_adamw_and_scale_one": (
            checkpoint.optimizer_state.step == 1 and checkpoint.scale == 1.0
        ),
        "restored_state_matches_uninterrupted_exactly": (
            torch.equal(model.weight.detach(), restored_model.weight.detach())
            and _adamw_state(model, optimizer)
            == _adamw_state(restored_model, restored_optimizer)
            and scaler.state_dict() == restored_scaler.state_dict()
        ),
        "restored_scaler_executes_borderline_step": (
            restored.optimizer_step_executed
            and restored.optimizer_state_after.step == 2
            and restored.scale_before == 1.0
        ),
        "omitted_scaler_state_overflows_and_skips_borderline_step": (
            not omitted.scaled_gradient_is_finite
            and not omitted.optimizer_step_executed
            and omitted.optimizer_state_after.step == 1
            and omitted.scale_before == 8.0
            and omitted.scale_after == 4.0
        ),
        "omitted_scaler_state_diverges_from_restored_parameter": (
            omitted.parameter_after != restored.parameter_after
        ),
        "strict_json_payload_contains_no_nonfinite_numbers": True,
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"CPU AMP/GradScaler control failed: {failed}")

    scope = {
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
    return AMPGradScalerAnalysis(
        torch_version=torch.__version__,
        clip_max_norm=0.5,
        initial_scale=8.0,
        full_batch_reference=full_batch,
        correct_unscale_then_clip=correct_clip,
        wrong_clip_then_unscale=wrong_clip,
        initial_finite_adamw_step=initial_finite,
        overflow_windows=overflow_windows,
        checkpoint=checkpoint,
        uninterrupted_after_checkpoint=uninterrupted,
        restored_with_scaler_state=restored,
        restored_without_scaler_state=omitted,
        assertions=assertions,
        scope=scope,
    )
