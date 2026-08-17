"""Cross-process CPU AMP training-checkpoint resume control.

This module is deliberately small enough to audit but executes the real
PyTorch state machines that are easy to omit from a training checkpoint:
model and AdamW state, StepLR progress, GradScaler scale, Torch/Python RNG,
and a stateful shuffled data cursor.  The project entry point launches the
first and second training segments in different operating-system processes.

The fixture is CPU/float16 specific.  It is not evidence about CUDA kernels,
DataLoader workers, distributed checkpoints, target LLMs, or model quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

TRAINING_RESUME_CONTROL_VERSION = "about-llm.training-resume-process-control.v1"
TRAINING_RESUME_CHECKPOINT_VERSION = "about-llm.training-resume-checkpoint.v1"
TRAINING_RESUME_EVIDENCE_BOUNDARY = (
    "real_cpu_float16_autocast_and_grad_scaler_executed",
    "real_adamw_and_step_lr_executed",
    "scheduler_advanced_only_after_executed_optimizer_steps",
    "torch_python_and_stateful_shuffle_rng_restored",
    "authored_checkpoint_written_and_loaded_with_weights_only",
    "first_segment_process_exited_before_distinct_resume_process",
    "resumed_terminal_state_matches_uninterrupted_bit_exactly",
    "scheduler_scaler_rng_and_data_omission_controls_executed",
    "cuda_dataloader_workers_or_distributed_checkpoint_executed=false",
    "target_model_trainer_or_dataset_executed=false",
    "crash_atomicity_origin_authentication_or_quality_proved=false",
)

WorkerMode = Literal[
    "baseline",
    "phase1",
    "resume",
    "omit-scheduler",
    "omit-scaler",
    "omit-rng",
    "omit-data",
    "wrong-scheduler",
]

_TOTAL_ATTEMPTS = 8
_SPLIT_AFTER_ATTEMPTS = 4
_OVERFLOW_ATTEMPTS = frozenset({1, 2, 3})
_BORDERLINE_ATTEMPT = 4
_MODEL_SEED = 31
_TRAINING_SEED = 37
_PYTHON_SEED = 41
_DATA_SEED = 43
_CHECKPOINT_MAX_BYTES = 16 * 1024 * 1024
_CHECKPOINT_FIELDS = {
    "schema_version",
    "dataset_sha256",
    "progress",
    "model",
    "optimizer",
    "scheduler",
    "grad_scaler",
    "torch_cpu_rng_state",
    "python_rng_state",
    "data_stream",
}
_PROGRESS_FIELDS = {"next_attempt_index", "successful_updates"}
_PYTHON_RNG_FIELDS = {"version", "state", "gauss_next"}
_STREAM_FIELDS = {"permutation", "cursor", "epoch", "generator_state"}


class _TinyStochasticRegressor(torch.nn.Module):
    """Two real linear paths: one overflow anchor and one stochastic branch."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1, bias=False)
        self.data_path = torch.nn.Linear(4, 1, bias=True)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("features must have shape [batch, 4]")
        # An explicit mask makes the consumed global Torch RNG visible in the
        # report.  This has the same inverted-dropout expectation as p=0.5.
        mask = (torch.rand_like(features) >= 0.5).to(features.dtype) * 2.0
        with torch.amp.autocast(device_type="cpu", dtype=torch.float16):
            anchor = self.anchor(
                torch.ones((features.shape[0], 1), dtype=torch.float32)
            )
            output = anchor + self.data_path(features * mask)
        return output, mask


@dataclass
class _StatefulShuffleStream:
    permutation: Tensor
    cursor: int
    epoch: int
    generator_state: Tensor

    @classmethod
    def create(cls, example_count: int, *, seed: int) -> _StatefulShuffleStream:
        if type(example_count) is not int or example_count < 2:
            raise ValueError("example_count must be an integer of at least two")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(example_count, generator=generator)
        return cls(
            permutation=permutation,
            cursor=0,
            epoch=0,
            generator_state=generator.get_state().clone(),
        )

    def next_indices(self, *, example_count: int, batch_size: int) -> Tensor:
        self._validate(example_count=example_count, batch_size=batch_size)
        generator = torch.Generator(device="cpu")
        generator.set_state(self.generator_state)
        if self.cursor + batch_size > example_count:
            self.permutation = torch.randperm(example_count, generator=generator)
            self.cursor = 0
            self.epoch += 1
        indices = self.permutation[self.cursor : self.cursor + batch_size].clone()
        self.cursor += batch_size
        self.generator_state = generator.get_state().clone()
        return indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "permutation": self.permutation.clone(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "generator_state": self.generator_state.clone(),
        }

    def load_state_dict(
        self,
        value: object,
        *,
        example_count: int,
        batch_size: int,
    ) -> None:
        root = _exact_dict(value, _STREAM_FIELDS, "data_stream")
        permutation = _tensor(root["permutation"], "data_stream.permutation")
        generator_state = _tensor(
            root["generator_state"], "data_stream.generator_state"
        )
        cursor = _integer(root["cursor"], "data_stream.cursor", minimum=0)
        epoch = _integer(root["epoch"], "data_stream.epoch", minimum=0)
        self.permutation = permutation.clone()
        self.cursor = cursor
        self.epoch = epoch
        self.generator_state = generator_state.clone()
        self._validate(example_count=example_count, batch_size=batch_size)
        generator = torch.Generator(device="cpu")
        try:
            generator.set_state(self.generator_state)
        except RuntimeError as error:
            raise ValueError("data_stream.generator_state is invalid") from error

    def _validate(self, *, example_count: int, batch_size: int) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= example_count:
            raise ValueError("batch_size must be within the dataset")
        if self.permutation.dtype != torch.int64 or self.permutation.shape != (
            example_count,
        ):
            raise ValueError("data permutation dtype or shape drifted")
        expected = torch.arange(example_count, dtype=torch.int64)
        if not torch.equal(torch.sort(self.permutation).values, expected):
            raise ValueError("data permutation is not a complete permutation")
        if self.cursor < 0 or self.cursor > example_count:
            raise ValueError("data cursor is outside the permutation")
        if self.cursor % batch_size != 0:
            raise ValueError("data cursor is not at a batch boundary")
        if self.epoch < 0:
            raise ValueError("data epoch cannot be negative")
        if self.generator_state.dtype != torch.uint8 or self.generator_state.ndim != 1:
            raise ValueError("data generator state dtype or shape drifted")


@dataclass
class _RuntimeState:
    model: _TinyStochasticRegressor
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler.StepLR
    scaler: torch.amp.GradScaler
    data_stream: _StatefulShuffleStream
    next_attempt_index: int = 0
    successful_updates: int = 0


def _dataset() -> Tensor:
    return torch.tensor(
        [
            [0.125, 0.250, 0.375, 0.500],
            [0.500, 0.375, 0.250, 0.125],
            [0.125, 0.500, 0.250, 0.375],
            [0.375, 0.125, 0.500, 0.250],
            [0.250, 0.375, 0.125, 0.500],
            [0.500, 0.125, 0.375, 0.250],
            [0.250, 0.500, 0.375, 0.125],
            [0.375, 0.250, 0.500, 0.125],
        ],
        dtype=torch.float32,
    )


def _dataset_sha256(dataset: Tensor) -> str:
    return _tensor_sha256(dataset)


def _new_runtime_state() -> _RuntimeState:
    torch.manual_seed(_MODEL_SEED)
    model = _TinyStochasticRegressor()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.02,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5,
    )
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=8.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=1000,
    )
    if not scaler.is_enabled():
        raise RuntimeError("CPU GradScaler is unavailable or disabled")
    stream = _StatefulShuffleStream.create(len(_dataset()), seed=_DATA_SEED)
    torch.manual_seed(_TRAINING_SEED)
    random.seed(_PYTHON_SEED)
    return _RuntimeState(model, optimizer, scheduler, scaler, stream)


def _optimizer_step(state: _RuntimeState) -> int:
    steps: set[int] = set()
    for parameter in state.model.parameters():
        parameter_state = state.optimizer.state.get(parameter)
        if not parameter_state:
            continue
        step = parameter_state.get("step")
        if not isinstance(step, Tensor) or step.numel() != 1:
            raise AssertionError("AdamW step must be a scalar tensor")
        steps.add(int(step.item()))
    if not steps:
        return 0
    if len(steps) != 1:
        raise AssertionError("AdamW parameter steps diverged")
    return next(iter(steps))


def _all_gradients_finite(model: torch.nn.Module) -> bool:
    gradients = [parameter.grad for parameter in model.parameters()]
    if any(gradient is None for gradient in gradients):
        raise AssertionError("every authored model parameter must receive a gradient")
    return all(bool(torch.isfinite(cast(Tensor, gradient)).all()) for gradient in gradients)


def _attempt_kind(index: int) -> str:
    if index in _OVERFLOW_ATTEMPTS:
        return "intentional-nonfinite"
    if index == _BORDERLINE_ATTEMPT:
        return "scale-sensitive-borderline"
    return "finite-stochastic"


def _loss_multiplier(index: int, python_factor: float) -> float:
    if index in _OVERFLOW_ATTEMPTS:
        return float("inf")
    if index == _BORDERLINE_ATTEMPT:
        return 10_000.0
    return python_factor


def _run_attempt(
    state: _RuntimeState,
    dataset: Tensor,
    *,
    advance_scheduler_on_overflow: bool,
) -> dict[str, Any]:
    index = state.next_attempt_index
    if not 0 <= index < _TOTAL_ATTEMPTS:
        raise ValueError("training attempt index is outside the authored fixture")
    indices = state.data_stream.next_indices(example_count=len(dataset), batch_size=2)
    features = dataset.index_select(0, indices)
    python_factor = 0.75 + 0.5 * random.random()
    multiplier = _loss_multiplier(index, python_factor)
    state.optimizer.zero_grad(set_to_none=True)
    parameter_before = _fingerprint(state.model.state_dict())
    optimizer_step_before = _optimizer_step(state)
    scheduler_epoch_before = int(state.scheduler.last_epoch)
    scheduler_step_count_before = int(state.scheduler._step_count)
    learning_rate_before = float(state.optimizer.param_groups[0]["lr"])
    scale_before = float(state.scaler.get_scale())

    output, dropout_mask = state.model(features)
    loss = output.float().sum() * multiplier
    state.scaler.scale(loss).backward()
    state.scaler.unscale_(state.optimizer)
    gradients_finite = _all_gradients_finite(state.model)
    gradient_norm: float | None = None
    if gradients_finite:
        norm = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        gradient_norm = float(norm.item())
    state.scaler.step(state.optimizer)
    state.scaler.update()
    optimizer_step_after = _optimizer_step(state)
    optimizer_step_executed = optimizer_step_after == optimizer_step_before + 1
    if optimizer_step_executed:
        state.successful_updates += 1
        state.scheduler.step()
    elif advance_scheduler_on_overflow:
        state.scheduler.step()
    state.optimizer.zero_grad(set_to_none=True)
    state.next_attempt_index += 1

    loss_value = float(loss.detach())
    return {
        "attempt_index": index,
        "kind": _attempt_kind(index),
        "batch_indices": [int(item) for item in indices.tolist()],
        "python_factor": python_factor,
        "dropout_mask_sha256": _tensor_sha256(dropout_mask),
        "autocast_output_dtype": str(output.dtype),
        "loss_is_finite": math.isfinite(loss_value),
        "loss": loss_value if math.isfinite(loss_value) else None,
        "gradients_finite_after_unscale": gradients_finite,
        "gradient_norm_before_clip": gradient_norm,
        "scale_before": scale_before,
        "scale_after": float(state.scaler.get_scale()),
        "optimizer_step_before": optimizer_step_before,
        "optimizer_step_after": optimizer_step_after,
        "optimizer_step_executed": optimizer_step_executed,
        "scheduler_last_epoch_before": scheduler_epoch_before,
        "scheduler_last_epoch_after": int(state.scheduler.last_epoch),
        "scheduler_step_count_before": scheduler_step_count_before,
        "scheduler_step_count_after": int(state.scheduler._step_count),
        "learning_rate_before": learning_rate_before,
        "learning_rate_after": float(state.optimizer.param_groups[0]["lr"]),
        "model_fingerprint_before": parameter_before,
        "model_fingerprint_after": _fingerprint(state.model.state_dict()),
        "data_epoch_after": state.data_stream.epoch,
        "data_cursor_after": state.data_stream.cursor,
    }


def _run_range(
    state: _RuntimeState,
    dataset: Tensor,
    *,
    stop: int,
    advance_scheduler_on_overflow: bool,
) -> list[dict[str, Any]]:
    if not state.next_attempt_index < stop <= _TOTAL_ATTEMPTS:
        raise ValueError("invalid training attempt range")
    return [
        _run_attempt(
            state,
            dataset,
            advance_scheduler_on_overflow=advance_scheduler_on_overflow,
        )
        for _ in range(state.next_attempt_index, stop)
    ]


def _encode_python_rng_state() -> dict[str, Any]:
    version, values, gauss_next = random.getstate()
    if gauss_next is not None:
        raise AssertionError("authored Python RNG must not have a Gaussian cache")
    return {
        "version": int(version),
        "state": torch.tensor(values, dtype=torch.int64),
        "gauss_next": None,
    }


def _restore_python_rng_state(value: object) -> None:
    root = _exact_dict(value, _PYTHON_RNG_FIELDS, "python_rng_state")
    version = _integer(root["version"], "python_rng_state.version", minimum=1)
    state = _tensor(root["state"], "python_rng_state.state")
    if state.dtype != torch.int64 or state.ndim != 1 or state.numel() != 625:
        raise ValueError("python_rng_state.state dtype or shape drifted")
    if root["gauss_next"] is not None:
        raise ValueError("python_rng_state.gauss_next must be null")
    values = tuple(int(item) for item in state.tolist())
    try:
        random.setstate((version, values, None))
    except (TypeError, ValueError) as error:
        raise ValueError("python_rng_state is invalid") from error


def _checkpoint_payload(state: _RuntimeState, dataset: Tensor) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_RESUME_CHECKPOINT_VERSION,
        "dataset_sha256": _dataset_sha256(dataset),
        "progress": {
            "next_attempt_index": state.next_attempt_index,
            "successful_updates": state.successful_updates,
        },
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "grad_scaler": state.scaler.state_dict(),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "python_rng_state": _encode_python_rng_state(),
        "data_stream": state.data_stream.state_dict(),
    }


def _write_checkpoint(path: Path, state: _RuntimeState, dataset: Tensor) -> str:
    if path.exists():
        raise FileExistsError(f"checkpoint path already exists: {path}")
    if not path.parent.is_dir():
        raise ValueError("checkpoint parent directory must already exist")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(_checkpoint_payload(state, dataset), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size > _CHECKPOINT_MAX_BYTES:
            raise ValueError("authored checkpoint exceeds the byte limit")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _load_checkpoint(
    path: Path,
    state: _RuntimeState,
    dataset: Tensor,
    *,
    restore_scheduler: bool,
    restore_scaler: bool,
    restore_rng: bool,
    restore_data: bool,
) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("checkpoint must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= _CHECKPOINT_MAX_BYTES:
        raise ValueError("checkpoint byte size is invalid or exceeds the limit")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    root = _exact_dict(payload, _CHECKPOINT_FIELDS, "checkpoint")
    if root["schema_version"] != TRAINING_RESUME_CHECKPOINT_VERSION:
        raise ValueError("checkpoint schema version drifted")
    if root["dataset_sha256"] != _dataset_sha256(dataset):
        raise ValueError("checkpoint dataset identity drifted")
    progress = _exact_dict(root["progress"], _PROGRESS_FIELDS, "progress")
    next_attempt = _integer(
        progress["next_attempt_index"],
        "progress.next_attempt_index",
        minimum=1,
    )
    successful_updates = _integer(
        progress["successful_updates"],
        "progress.successful_updates",
        minimum=1,
    )
    if next_attempt > _TOTAL_ATTEMPTS or successful_updates > next_attempt:
        raise ValueError("checkpoint progress is inconsistent")

    model_state = _mapping(root["model"], "model")
    state.model.load_state_dict(model_state, strict=True)
    if any(not bool(torch.isfinite(parameter).all()) for parameter in state.model.parameters()):
        raise ValueError("checkpoint model contains a non-finite parameter")
    state.optimizer.load_state_dict(_mapping(root["optimizer"], "optimizer"))
    if restore_scheduler:
        state.scheduler.load_state_dict(_mapping(root["scheduler"], "scheduler"))
    if restore_scaler:
        state.scaler.load_state_dict(_mapping(root["grad_scaler"], "grad_scaler"))
    if restore_data:
        state.data_stream.load_state_dict(
            root["data_stream"],
            example_count=len(dataset),
            batch_size=2,
        )
    if restore_rng:
        torch_rng = _tensor(root["torch_cpu_rng_state"], "torch_cpu_rng_state")
        if torch_rng.dtype != torch.uint8 or torch_rng.ndim != 1:
            raise ValueError("torch_cpu_rng_state dtype or shape drifted")
        try:
            torch.set_rng_state(torch_rng)
        except RuntimeError as error:
            raise ValueError("torch_cpu_rng_state is invalid") from error
        _restore_python_rng_state(root["python_rng_state"])
    state.next_attempt_index = next_attempt
    state.successful_updates = successful_updates
    if _optimizer_step(state) != successful_updates:
        raise ValueError("checkpoint optimizer step and progress disagree")
    if any(parameter.grad is not None for parameter in state.model.parameters()):
        raise ValueError("checkpoint must restore at a cleared-gradient boundary")
    return _file_sha256(path)


def _terminal_summary(state: _RuntimeState) -> dict[str, Any]:
    components = {
        "model": _fingerprint(state.model.state_dict()),
        "optimizer": _fingerprint(state.optimizer.state_dict()),
        "scheduler": _fingerprint(state.scheduler.state_dict()),
        "grad_scaler": _fingerprint(state.scaler.state_dict()),
        "torch_cpu_rng": _tensor_sha256(torch.get_rng_state()),
        "python_rng": _fingerprint(_encode_python_rng_state()),
        "data_stream": _fingerprint(state.data_stream.state_dict()),
    }
    progress = {
        "next_attempt_index": state.next_attempt_index,
        "successful_updates": state.successful_updates,
        "optimizer_step": _optimizer_step(state),
        "scheduler_last_epoch": int(state.scheduler.last_epoch),
        "scheduler_step_count": int(state.scheduler._step_count),
        "learning_rate": float(state.optimizer.param_groups[0]["lr"]),
        "grad_scaler_scale": float(state.scaler.get_scale()),
        "data_epoch": state.data_stream.epoch,
        "data_cursor": state.data_stream.cursor,
    }
    return {
        "components": components,
        "progress": progress,
        "full_state_fingerprint": _fingerprint(
            {"components": components, "progress": progress}
        ),
    }


def run_training_resume_worker(mode: WorkerMode, checkpoint_path: Path) -> dict[str, Any]:
    """Run one isolated worker phase; the project CLI owns process creation."""

    if mode not in {
        "baseline",
        "phase1",
        "resume",
        "omit-scheduler",
        "omit-scaler",
        "omit-rng",
        "omit-data",
        "wrong-scheduler",
    }:
        raise ValueError("unsupported worker mode")
    torch.set_num_threads(1)
    dataset = _dataset()
    state = _new_runtime_state()
    checkpoint_sha256: str | None = None
    if mode in {
        "resume",
        "omit-scheduler",
        "omit-scaler",
        "omit-rng",
        "omit-data",
    }:
        checkpoint_sha256 = _load_checkpoint(
            checkpoint_path,
            state,
            dataset,
            restore_scheduler=mode != "omit-scheduler",
            restore_scaler=mode != "omit-scaler",
            restore_rng=mode != "omit-rng",
            restore_data=mode != "omit-data",
        )
    stop = _SPLIT_AFTER_ATTEMPTS if mode == "phase1" else _TOTAL_ATTEMPTS
    trace = _run_range(
        state,
        dataset,
        stop=stop,
        advance_scheduler_on_overflow=mode == "wrong-scheduler",
    )
    if mode == "phase1":
        checkpoint_sha256 = _write_checkpoint(checkpoint_path, state, dataset)
    return {
        "implementation": TRAINING_RESUME_CONTROL_VERSION,
        "worker_mode": mode,
        "pid": os.getpid(),
        "runtime": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "parameter_dtype": "torch.float32",
            "autocast_dtype": "torch.float16",
        },
        "trace": trace,
        "terminal": _terminal_summary(state),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _run_worker_process(
    worker_script: Path,
    *,
    mode: WorkerMode,
    checkpoint_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(worker_script),
        "--worker-mode",
        mode,
        "--checkpoint-path",
        str(checkpoint_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=worker_script.parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f"{mode} worker exceeded 120 seconds") from None
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"{mode} worker failed with code {process.returncode}: {detail}")
    if stderr:
        raise RuntimeError(
            f"{mode} worker emitted stderr: {stderr.decode('utf-8', errors='replace')}"
        )
    try:
        decoded = json.loads(
            stdout.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"{mode} worker emitted invalid strict JSON") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{mode} worker report must be a JSON object")
    report = cast(dict[str, Any], decoded)
    if report.get("pid") != process.pid or report.get("worker_mode") != mode:
        raise RuntimeError(f"{mode} worker identity drifted")
    return report


def _trace(report: dict[str, Any]) -> list[dict[str, Any]]:
    value = report.get("trace")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise AssertionError("worker trace schema drifted")
    return cast(list[dict[str, Any]], value)


def _terminal(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("terminal")
    if not isinstance(value, dict):
        raise AssertionError("worker terminal schema drifted")
    return cast(dict[str, Any], value)


def run_training_resume_process_control(worker_script: Path) -> dict[str, Any]:
    """Launch independent workers and compare uninterrupted/resumed state."""

    script = worker_script.resolve(strict=True)
    if not script.is_file() or script.is_symlink():
        raise ValueError("worker_script must be a regular non-symlink file")
    with tempfile.TemporaryDirectory(prefix="about-llm-training-resume-") as directory:
        checkpoint_path = Path(directory) / "training-checkpoint.pt"
        # Baseline does not touch the checkpoint.  Phase 1 must complete before
        # any resume worker is launched; after that, all consumers are read-only.
        with ThreadPoolExecutor(max_workers=2) as executor:
            baseline_future = executor.submit(
                _run_worker_process,
                script,
                mode="baseline",
                checkpoint_path=checkpoint_path,
            )
            phase1_future = executor.submit(
                _run_worker_process,
                script,
                mode="phase1",
                checkpoint_path=checkpoint_path,
            )
            baseline = baseline_future.result()
            phase1 = phase1_future.result()
        post_checkpoint_modes: tuple[WorkerMode, ...] = (
            "resume",
            "omit-scheduler",
            "omit-scaler",
            "omit-rng",
            "omit-data",
            "wrong-scheduler",
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                mode: executor.submit(
                    _run_worker_process,
                    script,
                    mode=mode,
                    checkpoint_path=checkpoint_path,
                )
                for mode in post_checkpoint_modes
            }
            post_checkpoint = {
                mode: future.result() for mode, future in futures.items()
            }
        resumed = post_checkpoint["resume"]
        omit_scheduler = post_checkpoint["omit-scheduler"]
        omit_scaler = post_checkpoint["omit-scaler"]
        omit_rng = post_checkpoint["omit-rng"]
        omit_data = post_checkpoint["omit-data"]
        wrong_scheduler = post_checkpoint["wrong-scheduler"]
        checkpoint_bytes = checkpoint_path.stat().st_size

    baseline_trace = _trace(baseline)
    phase1_trace = _trace(phase1)
    resumed_trace = _trace(resumed)
    omitted_scheduler_trace = _trace(omit_scheduler)
    omitted_scaler_trace = _trace(omit_scaler)
    omitted_rng_trace = _trace(omit_rng)
    omitted_data_trace = _trace(omit_data)
    wrong_scheduler_trace = _trace(wrong_scheduler)
    baseline_terminal = _terminal(baseline)
    resumed_terminal = _terminal(resumed)
    checkpoint_hashes = {
        report.get("checkpoint_sha256")
        for report in (
            phase1,
            resumed,
            omit_scheduler,
            omit_scaler,
            omit_rng,
            omit_data,
        )
    }
    overflow_trace = phase1_trace[1:4]
    assertions = {
        "phase1_process_exited_before_distinct_resume_process": (
            phase1["pid"] != resumed["pid"]
        ),
        "checkpoint_bytes_reopened_identically_by_all_resume_workers": (
            len(checkpoint_hashes) == 1 and None not in checkpoint_hashes
        ),
        "split_prefix_matches_uninterrupted_exactly": (
            phase1_trace == baseline_trace[:_SPLIT_AFTER_ATTEMPTS]
        ),
        "resumed_tail_matches_uninterrupted_exactly": (
            resumed_trace == baseline_trace[_SPLIT_AFTER_ATTEMPTS:]
        ),
        "resumed_terminal_state_matches_uninterrupted_bit_exactly": (
            resumed_terminal == baseline_terminal
        ),
        "overflow_skips_optimizer_and_scheduler_together": all(
            item["optimizer_step_executed"] is False
            and item["scheduler_last_epoch_after"]
            == item["scheduler_last_epoch_before"]
            and item["scheduler_step_count_after"]
            == item["scheduler_step_count_before"]
            for item in overflow_trace
        ),
        "wrong_scheduler_advances_on_overflow_and_diverges": (
            all(
                item["optimizer_step_executed"] is False
                and item["scheduler_last_epoch_after"]
                == item["scheduler_last_epoch_before"] + 1
                for item in wrong_scheduler_trace[1:4]
            )
            and _terminal(wrong_scheduler)["full_state_fingerprint"]
            != baseline_terminal["full_state_fingerprint"]
        ),
        "omitted_scheduler_state_changes_lr_and_terminal_state": (
            omitted_scheduler_trace[0]["learning_rate_after"]
            != resumed_trace[0]["learning_rate_after"]
            and _terminal(omit_scheduler)["full_state_fingerprint"]
            != baseline_terminal["full_state_fingerprint"]
        ),
        "omitted_scaler_state_changes_borderline_step_decision": (
            resumed_trace[0]["scale_before"] == 1.0
            and resumed_trace[0]["optimizer_step_executed"] is True
            and omitted_scaler_trace[0]["scale_before"] == 8.0
            and omitted_scaler_trace[0]["optimizer_step_executed"] is False
        ),
        "omitted_rng_state_preserves_batches_but_changes_stochastic_trace": (
            [item["batch_indices"] for item in omitted_rng_trace]
            == [item["batch_indices"] for item in resumed_trace]
            and [
                (item["python_factor"], item["dropout_mask_sha256"])
                for item in omitted_rng_trace
            ]
            != [
                (item["python_factor"], item["dropout_mask_sha256"])
                for item in resumed_trace
            ]
        ),
        "omitted_data_state_changes_batches_and_terminal_state": (
            [item["batch_indices"] for item in omitted_data_trace]
            != [item["batch_indices"] for item in resumed_trace]
            and _terminal(omit_data)["full_state_fingerprint"]
            != baseline_terminal["full_state_fingerprint"]
        ),
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"training-resume control assertion(s) failed: {failed}")
    return {
        "implementation": TRAINING_RESUME_CONTROL_VERSION,
        "runtime": baseline["runtime"],
        "fixture": {
            "total_attempts": _TOTAL_ATTEMPTS,
            "split_after_attempts": _SPLIT_AFTER_ATTEMPTS,
            "intentional_overflow_attempts": sorted(_OVERFLOW_ATTEMPTS),
            "scale_sensitive_borderline_attempt": _BORDERLINE_ATTEMPT,
            "batch_size": 2,
            "dataset_examples": len(_dataset()),
            "initial_grad_scaler_scale": 8.0,
            "step_lr_step_size": 2,
            "step_lr_gamma": 0.5,
        },
        "processes": {
            "baseline_pid": baseline["pid"],
            "phase1_pid": phase1["pid"],
            "resumed_pid": resumed["pid"],
            "phase1_process_exited_before_resume_launch": True,
            "phase1_and_resume_pids_are_distinct": phase1["pid"] != resumed["pid"],
        },
        "checkpoint": {
            "schema_version": TRAINING_RESUME_CHECKPOINT_VERSION,
            "bytes": checkpoint_bytes,
            "sha256": phase1["checkpoint_sha256"],
            "loaded_with_torch_weights_only": True,
            "atomic_same_directory_temporary_replace_executed": True,
            "components": [
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
            ],
        },
        "uninterrupted": {
            "trace": baseline_trace,
            "terminal": baseline_terminal,
        },
        "split_resume": {
            "phase1_trace": phase1_trace,
            "resumed_trace": resumed_trace,
            "terminal": resumed_terminal,
        },
        "negative_controls": {
            "advance_scheduler_on_overflow": {
                "trace": wrong_scheduler_trace,
                "terminal": _terminal(wrong_scheduler),
            },
            "omit_scheduler_state": {
                "trace": omitted_scheduler_trace,
                "terminal": _terminal(omit_scheduler),
            },
            "omit_grad_scaler_state": {
                "trace": omitted_scaler_trace,
                "terminal": _terminal(omit_scaler),
            },
            "omit_rng_state": {
                "trace": omitted_rng_trace,
                "terminal": _terminal(omit_rng),
            },
            "omit_data_stream_state": {
                "trace": omitted_data_trace,
                "terminal": _terminal(omit_data),
            },
        },
        "assertions": assertions,
        "scope": {
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
        },
    }


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(metadata + b"\0" + raw).hexdigest()


def _projection(value: object) -> object:
    if isinstance(value, Tensor):
        return {
            "tensor": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _tensor_sha256(value),
            }
        }
    if isinstance(value, dict):
        entries: list[dict[str, object]] = []
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError("fingerprinted mapping keys must be strings or integers")
            entries.append(
                {
                    "key_type": type(key).__name__,
                    "key": key,
                    "value": _projection(value[key]),
                }
            )
        return {"mapping": entries}
    if isinstance(value, (list, tuple)):
        return {"sequence": [_projection(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot fingerprint non-finite float state")
        return value
    raise TypeError(f"unsupported fingerprint state type: {type(value).__name__}")


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        _projection(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    result = cast(dict[str, Any], value)
    if set(result) != fields:
        raise ValueError(f"{label} fields drifted")
    return result


def _mapping(value: object, label: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _tensor(value: object, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{label} must be a tensor")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value
