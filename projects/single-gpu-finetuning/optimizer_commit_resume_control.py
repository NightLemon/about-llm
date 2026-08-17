"""Cross-process optimizer-commit resume control with real DataLoader workers.

The fixture crashes after a third microbatch has been delivered and backward
has run, while only the first two microbatches belong to a committed optimizer
step.  The base checkpoint intentionally excludes the in-flight gradients.  A
correct restart can either replay from the optimizer-committed cursor or load a
digest-bound gradient sidecar.  The sidecar protocol additionally requires a
strict manifest published last; incomplete publication snapshots fail closed.
A negative control restarts from the later main-loop-consumed cursor without
the gradients and silently loses one sample.

This is a deterministic CPU/float64 control, not evidence about CUDA,
distributed checkpointing, arbitrary trainers, or crash-durable storage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

CONTROL_VERSION = "about-llm.optimizer-commit-resume-control.v1"
CHECKPOINT_VERSION = "about-llm.optimizer-commit-checkpoint.v1"
INFLIGHT_SIDECAR_VERSION = "about-llm.inflight-gradient-sidecar.v1"
BUNDLE_MANIFEST_VERSION = "about-llm.optimizer-commit-bundle-manifest.v1"
PERMUTATION = (8, 3, 1, 7, 0, 9, 4, 2, 6, 5)
DATASET_SIZE = len(PERMUTATION)
ACCUMULATION_STEPS = 2
CRASH_AFTER_CONSUMED = 3
COMMITTED_CURSOR_AT_CRASH = 2
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
LOADER_GENERATOR_SEED = 20260814
STOCHASTIC_MASK_SEED = 20260815
CHECKPOINT_MAX_BYTES = 4 * 1024 * 1024
BUNDLE_MANIFEST_MAX_BYTES = 16 * 1024
WorkerMode = Literal[
    "baseline",
    "phase1",
    "resume_committed",
    "resume_consumed",
    "resume_inflight",
    "resume_inflight_wrong_rng",
]


def _encode_json(payload: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _dataset_records() -> tuple[tuple[float, float, float], ...]:
    records: list[tuple[float, float, float]] = []
    for sample_id in range(DATASET_SIZE):
        first = (sample_id + 1) / 16.0
        second = ((sample_id * 3 + 1) % 11) / 12.0
        target = 0.7 * first - 1.2 * second + 0.3
        records.append((first, second, target))
    return tuple(records)


def _dataset_identity() -> str:
    payload = {
        "dtype": "torch.float64",
        "records": _dataset_records(),
        "permutation": PERMUTATION,
    }
    encoded = _encode_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _loader_contract() -> dict[str, object]:
    return {
        "batch_size": 1,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "persistent_workers": False,
        "pin_memory": False,
        "multiprocessing_context": "spawn",
        "in_order": True,
        "loader_generator_seed": LOADER_GENERATOR_SEED,
    }


class CommitDataset(Dataset[dict[str, int | Tensor]]):
    """Return deterministic examples and observable worker identities."""

    def __len__(self) -> int:
        return DATASET_SIZE

    def __getitem__(self, sample_id: int) -> dict[str, int | Tensor]:
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise TypeError("sample_id must be an integer")
        if not 0 <= sample_id < DATASET_SIZE:
            raise IndexError("sample_id is outside the authored dataset")
        worker = get_worker_info()
        if worker is None:
            raise RuntimeError("this fixture requires a real DataLoader worker")
        first, second, target = _dataset_records()[sample_id]
        return {
            "sample_id": sample_id,
            "features": torch.tensor([first, second], dtype=torch.float64),
            "target": torch.tensor(target, dtype=torch.float64),
            "worker_id": worker.id,
            "worker_pid": os.getpid(),
        }


class TrackingOffsetSampler(Sampler[int]):
    """Yield a fixed permutation while exposing DataLoader prefetch progress."""

    def __init__(self, start_cursor: int) -> None:
        if isinstance(start_cursor, bool) or not isinstance(start_cursor, int):
            raise TypeError("start_cursor must be an integer")
        if not 0 <= start_cursor <= len(PERMUTATION):
            raise ValueError("start_cursor is outside the authored permutation")
        self.start_cursor = start_cursor
        self.emitted_cursor = start_cursor

    def __iter__(self) -> Iterator[int]:
        while self.emitted_cursor < len(PERMUTATION):
            sample_id = PERMUTATION[self.emitted_cursor]
            self.emitted_cursor += 1
            yield sample_id

    def __len__(self) -> int:
        return len(PERMUTATION) - self.start_cursor


@dataclass
class RuntimeState:
    model: nn.Linear
    optimizer: torch.optim.SGD
    scheduler: torch.optim.lr_scheduler.StepLR
    optimizer_steps: int
    committed_sample_ids: list[int]
    commit_boundary_torch_rng_state: Tensor | None


def _new_state() -> RuntimeState:
    model = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.125, -0.25]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.05], dtype=torch.float64))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5,
    )
    return RuntimeState(model, optimizer, scheduler, 0, [], None)


def _scalar_int(batch: Mapping[str, Any], field: str) -> int:
    value = batch.get(field)
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise AssertionError(f"{field} must collate to one scalar tensor")
    return int(value.item())


def _tensor_batch(batch: Mapping[str, Any], field: str) -> Tensor:
    value = batch.get(field)
    if not isinstance(value, Tensor):
        raise AssertionError(f"{field} must collate to a tensor")
    if value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise AssertionError(f"{field} must be finite torch.float64")
    return value


def _commit_window(
    state: RuntimeState,
    window_sample_ids: list[int],
    *,
    rescale_partial: bool,
) -> None:
    if not window_sample_ids:
        raise AssertionError("cannot commit an empty accumulation window")
    if len(window_sample_ids) > ACCUMULATION_STEPS:
        raise AssertionError("accumulation window exceeds the authored size")
    if rescale_partial and len(window_sample_ids) < ACCUMULATION_STEPS:
        scale = ACCUMULATION_STEPS / len(window_sample_ids)
        for parameter in state.model.parameters():
            if parameter.grad is None:
                raise AssertionError("partial window parameter gradient is missing")
            parameter.grad.mul_(scale)
    state.optimizer.step()
    state.scheduler.step()
    state.optimizer.zero_grad(set_to_none=True)
    state.optimizer_steps += 1
    state.committed_sample_ids.extend(window_sample_ids)
    state.commit_boundary_torch_rng_state = torch.get_rng_state().clone()


def _run_segment(
    state: RuntimeState,
    *,
    start_cursor: int,
    stop_after_records: int | None,
    initial_pending_sample_ids: list[int] | None = None,
) -> dict[str, object]:
    initial_window_ids = list(initial_pending_sample_ids or [])
    if len(initial_window_ids) >= ACCUMULATION_STEPS:
        raise ValueError("initial pending window must be incomplete")
    gradients_present = [
        parameter.grad is not None for parameter in state.model.parameters()
    ]
    if initial_window_ids and not all(gradients_present):
        raise AssertionError("pending window requires every parameter gradient")
    if not initial_window_ids and any(gradients_present):
        raise AssertionError("empty pending window cannot start with gradients")
    sampler = TrackingOffsetSampler(start_cursor)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(LOADER_GENERATOR_SEED)
    loader = DataLoader(
        CommitDataset(),
        batch_size=1,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,
        pin_memory=False,
        timeout=30,
        multiprocessing_context="spawn",
        generator=generator,
        in_order=True,
    )
    iterator = iter(loader)
    consumed_ids: list[int] = []
    worker_ids: list[int] = []
    worker_pids: list[int] = []
    stochastic_mask_sha256: list[str] = []
    new_committed_windows: list[list[int]] = []
    learning_rates_after_commit: list[float] = []
    window_ids = list(initial_window_ids)
    stopped_early = False
    try:
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if not isinstance(batch, Mapping):
                raise AssertionError("default collate must return a mapping")
            sample_id = _scalar_int(batch, "sample_id")
            features = _tensor_batch(batch, "features")
            target = _tensor_batch(batch, "target").reshape(1, 1)
            stochastic_mask = (
                (torch.rand_like(features) >= 0.5).to(features.dtype) * 2.0
            )
            prediction = state.model(features * stochastic_mask)
            loss = F.mse_loss(prediction, target, reduction="mean")
            if not bool(torch.isfinite(loss)):
                raise AssertionError("training loss must be finite")
            # The installed PyTorch stubs do not type Tensor.backward().
            (loss / ACCUMULATION_STEPS).backward()  # type: ignore[no-untyped-call]
            consumed_ids.append(sample_id)
            worker_ids.append(_scalar_int(batch, "worker_id"))
            worker_pids.append(_scalar_int(batch, "worker_pid"))
            stochastic_mask_sha256.append(_tensor_sha256(stochastic_mask))
            window_ids.append(sample_id)
            if len(window_ids) == ACCUMULATION_STEPS:
                committed = list(window_ids)
                _commit_window(state, window_ids, rescale_partial=False)
                new_committed_windows.append(committed)
                learning_rates_after_commit.append(
                    float(state.optimizer.param_groups[0]["lr"])
                )
                window_ids.clear()
            if (
                stop_after_records is not None
                and len(consumed_ids) == stop_after_records
            ):
                stopped_early = True
                break
        emitted_cursor = sampler.emitted_cursor
        if not stopped_early and window_ids:
            committed = list(window_ids)
            _commit_window(state, window_ids, rescale_partial=True)
            new_committed_windows.append(committed)
            learning_rates_after_commit.append(
                float(state.optimizer.param_groups[0]["lr"])
            )
            window_ids.clear()
    finally:
        del iterator
        del loader
        gc.collect()

    in_flight_gradient_tensors = sum(
        parameter.grad is not None for parameter in state.model.parameters()
    )
    return {
        "start_cursor": start_cursor,
        "initial_pending_sample_ids": initial_window_ids,
        "stop_after_records": stop_after_records,
        "stopped_early": stopped_early,
        "consumed_sample_ids": consumed_ids,
        "sampler_emitted_cursor_when_observed": emitted_cursor,
        "new_committed_windows": new_committed_windows,
        "learning_rates_after_commit": learning_rates_after_commit,
        "pending_uncommitted_sample_ids": list(window_ids),
        "in_flight_gradient_tensor_count": in_flight_gradient_tensors,
        "stochastic_mask_sha256": stochastic_mask_sha256,
        "worker_ids_seen": sorted(set(worker_ids)),
        "worker_pids_seen": sorted(set(worker_pids)),
    }


def _update_fingerprint(hasher: Any, value: object) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor:")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_fingerprint(hasher, key)
            _update_fingerprint(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        hasher.update(b"sequence[")
        for item in value:
            _update_fingerprint(hasher, item)
        hasher.update(b"]")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError("state fingerprint forbids non-finite floats")
        hasher.update(value.hex().encode("ascii"))
        return
    if value is None or isinstance(value, (str, int, bool)):
        hasher.update(repr(value).encode("utf-8"))
        return
    raise TypeError(f"unsupported fingerprint value: {type(value)!r}")


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(repr(tuple(value.shape)).encode("ascii"))
    hasher.update(value.numpy().tobytes())
    return "sha256:" + hasher.hexdigest()


def _state_fingerprint(state: RuntimeState) -> str:
    hasher = hashlib.sha256()
    _update_fingerprint(hasher, state.model.state_dict())
    _update_fingerprint(hasher, state.optimizer.state_dict())
    _update_fingerprint(hasher, state.scheduler.state_dict())
    _update_fingerprint(hasher, torch.get_rng_state())
    _update_fingerprint(hasher, state.optimizer_steps)
    _update_fingerprint(hasher, state.committed_sample_ids)
    return "sha256:" + hasher.hexdigest()


def _parameter_snapshot(state: RuntimeState) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, tensor in state.model.state_dict().items():
        values = tensor.detach().cpu().reshape(-1).tolist()
        if not all(math.isfinite(float(value)) for value in values):
            raise AssertionError("model snapshot contains a non-finite value")
        result[name] = [float(value) for value in values]
    return result


def _terminal_state(state: RuntimeState) -> dict[str, object]:
    momentum_buffers: dict[str, list[float]] = {}
    for name, parameter in state.model.named_parameters():
        raw = state.optimizer.state.get(parameter, {}).get("momentum_buffer")
        if not isinstance(raw, Tensor):
            raise AssertionError("SGD momentum buffer is missing")
        momentum_buffers[name] = [
            float(value) for value in raw.detach().cpu().reshape(-1).tolist()
        ]
    return {
        "fingerprint": _state_fingerprint(state),
        "model_parameters": _parameter_snapshot(state),
        "optimizer_momentum_buffers": momentum_buffers,
        "scheduler": {
            "last_epoch": int(state.scheduler.last_epoch),
            "step_count": int(state.scheduler._step_count),
            "learning_rate": float(state.optimizer.param_groups[0]["lr"]),
        },
        "torch_cpu_rng_sha256": _tensor_sha256(torch.get_rng_state()),
        "optimizer_steps": state.optimizer_steps,
        "committed_sample_ids": list(state.committed_sample_ids),
        "all_parameter_gradients_cleared": all(
            parameter.grad is None for parameter in state.model.parameters()
        ),
    }


def _checkpoint_payload(
    state: RuntimeState,
    segment: Mapping[str, object],
) -> dict[str, object]:
    consumed_ids = segment["consumed_sample_ids"]
    emitted_cursor = segment["sampler_emitted_cursor_when_observed"]
    if not isinstance(consumed_ids, list) or not isinstance(emitted_cursor, int):
        raise AssertionError("phase segment progress is malformed")
    if state.commit_boundary_torch_rng_state is None:
        raise AssertionError("phase state is missing the commit-boundary RNG snapshot")
    return {
        "schema_version": CHECKPOINT_VERSION,
        "dataset_identity": _dataset_identity(),
        "permutation": list(PERMUTATION),
        "loader_contract": _loader_contract(),
        "progress": {
            "optimizer_committed_cursor": COMMITTED_CURSOR_AT_CRASH,
            "main_loop_consumed_cursor": len(consumed_ids),
            "sampler_emitted_cursor": emitted_cursor,
            "optimizer_steps": state.optimizer_steps,
            "committed_sample_ids": list(state.committed_sample_ids),
            "consumed_sample_ids": list(consumed_ids),
            "in_flight_gradients_serialized": False,
        },
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "commit_boundary_torch_rng_state": (
            state.commit_boundary_torch_rng_state.clone()
        ),
    }


def _inflight_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("inflight-gradients.pt")


def _inflight_bundle_manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.bundle-manifest.json")


def _inflight_sidecar_payload(
    state: RuntimeState,
    segment: Mapping[str, object],
    *,
    base_checkpoint_sha256: str,
) -> dict[str, object]:
    consumed_ids = segment["consumed_sample_ids"]
    emitted_cursor = segment["sampler_emitted_cursor_when_observed"]
    pending_ids = segment["pending_uncommitted_sample_ids"]
    if (
        not isinstance(consumed_ids, list)
        or not isinstance(emitted_cursor, int)
        or not isinstance(pending_ids, list)
    ):
        raise AssertionError("phase segment progress is malformed")
    gradients: dict[str, Tensor] = {}
    for name, parameter in state.model.named_parameters():
        if parameter.grad is None:
            raise AssertionError("in-flight sidecar requires every parameter gradient")
        gradients[name] = parameter.grad.detach().cpu().clone()
    return {
        "schema_version": INFLIGHT_SIDECAR_VERSION,
        "base_checkpoint_schema_version": CHECKPOINT_VERSION,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "dataset_identity": _dataset_identity(),
        "permutation": list(PERMUTATION),
        "progress": {
            "optimizer_committed_cursor": COMMITTED_CURSOR_AT_CRASH,
            "main_loop_consumed_cursor": len(consumed_ids),
            "sampler_emitted_cursor": emitted_cursor,
            "optimizer_steps": state.optimizer_steps,
            "committed_sample_ids": list(state.committed_sample_ids),
            "consumed_sample_ids": list(consumed_ids),
            "pending_window_sample_ids": list(pending_ids),
            "accumulation_position": len(pending_ids),
            "accumulation_steps": ACCUMULATION_STEPS,
            "loss_divisor": ACCUMULATION_STEPS,
        },
        "crash_observed_torch_rng_state": torch.get_rng_state().clone(),
        "gradients": gradients,
    }


def _inflight_bundle_manifest_payload(
    checkpoint_path: Path,
    *,
    base_size_bytes: int,
    base_sha256: str,
    sidecar_size_bytes: int,
    sidecar_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": BUNDLE_MANIFEST_VERSION,
        "publication_state": "complete",
        "dataset_identity": _dataset_identity(),
        "artifacts": {
            "base_checkpoint": {
                "file_name": checkpoint_path.name,
                "schema_version": CHECKPOINT_VERSION,
                "size_bytes": base_size_bytes,
                "sha256": base_sha256,
            },
            "inflight_gradient_sidecar": {
                "file_name": _inflight_sidecar_path(checkpoint_path).name,
                "schema_version": INFLIGHT_SIDECAR_VERSION,
                "size_bytes": sidecar_size_bytes,
                "sha256": sidecar_sha256,
                "base_checkpoint_sha256": base_sha256,
            },
        },
        "publication_sequence": [
            "base_checkpoint",
            "inflight_gradient_sidecar",
            "bundle_manifest",
        ],
    }


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> tuple[int, str]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        size = temporary_path.stat().st_size
        if size <= 0 or size > CHECKPOINT_MAX_BYTES:
            raise ValueError("checkpoint size is outside the authored cap")
        digest = "sha256:" + hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        os.replace(temporary_path, path)
        return size, digest
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_canonical_json(
    path: Path,
    payload: Mapping[str, object],
) -> tuple[int, str]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    encoded = _encode_json(dict(payload)).encode("utf-8")
    if not encoded or len(encoded) > BUNDLE_MANIFEST_MAX_BYTES:
        raise ValueError("bundle manifest size is outside the authored cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        size = temporary_path.stat().st_size
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        os.replace(temporary_path, path)
        return size, digest
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_exact_keys(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields drifted")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 identity")
    return value


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _read_bounded_artifact(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[bytes, int, str]:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing; publication is incomplete") from error
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{label} size is outside the authored cap")
    with path.open("rb") as handle:
        raw = handle.read(maximum_bytes + 1)
    if len(raw) != size:
        raise ValueError(f"{label} changed while it was being read")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return raw, size, digest


def _read_torch_checkpoint(path: Path) -> tuple[object, int, str]:
    encoded, size, digest = _read_bounded_artifact(
        path,
        maximum_bytes=CHECKPOINT_MAX_BYTES,
        label="checkpoint",
    )
    raw = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    return raw, size, digest


def _load_inflight_bundle_manifest(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], int, str]:
    manifest_path = _inflight_bundle_manifest_path(checkpoint_path)
    encoded, size, digest = _read_bounded_artifact(
        manifest_path,
        maximum_bytes=BUNDLE_MANIFEST_MAX_BYTES,
        label="in-flight bundle manifest",
    )
    try:
        decoded = encoded.decode("utf-8", errors="strict")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("in-flight bundle manifest is not strict JSON") from error
    if _encode_json(raw).encode("utf-8") != encoded:
        raise ValueError("in-flight bundle manifest must use canonical JSON")
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "publication_state",
            "dataset_identity",
            "artifacts",
            "publication_sequence",
        },
        "in-flight bundle manifest",
    )
    if root["schema_version"] != BUNDLE_MANIFEST_VERSION:
        raise ValueError("in-flight bundle manifest schema version drifted")
    if root["publication_state"] != "complete":
        raise ValueError("in-flight bundle publication is not complete")
    if root["dataset_identity"] != _dataset_identity():
        raise ValueError("in-flight bundle dataset identity drifted")
    if root["publication_sequence"] != [
        "base_checkpoint",
        "inflight_gradient_sidecar",
        "bundle_manifest",
    ]:
        raise ValueError("in-flight bundle publication sequence drifted")
    artifacts = _require_exact_keys(
        root["artifacts"],
        {"base_checkpoint", "inflight_gradient_sidecar"},
        "in-flight bundle artifacts",
    )
    base = _require_exact_keys(
        artifacts["base_checkpoint"],
        {"file_name", "schema_version", "size_bytes", "sha256"},
        "in-flight bundle base artifact",
    )
    sidecar = _require_exact_keys(
        artifacts["inflight_gradient_sidecar"],
        {
            "file_name",
            "schema_version",
            "size_bytes",
            "sha256",
            "base_checkpoint_sha256",
        },
        "in-flight bundle sidecar artifact",
    )
    if base["file_name"] != checkpoint_path.name:
        raise ValueError("in-flight bundle base file name drifted")
    sidecar_path = _inflight_sidecar_path(checkpoint_path)
    if sidecar["file_name"] != sidecar_path.name:
        raise ValueError("in-flight bundle sidecar file name drifted")
    if base["schema_version"] != CHECKPOINT_VERSION:
        raise ValueError("in-flight bundle base schema version drifted")
    if sidecar["schema_version"] != INFLIGHT_SIDECAR_VERSION:
        raise ValueError("in-flight bundle sidecar schema version drifted")
    base_size = _require_integer(base["size_bytes"], "bundle base size")
    sidecar_size = _require_integer(sidecar["size_bytes"], "bundle sidecar size")
    base_digest = _require_sha256(base["sha256"], "bundle base digest")
    sidecar_digest = _require_sha256(sidecar["sha256"], "bundle sidecar digest")
    bound_base_digest = _require_sha256(
        sidecar["base_checkpoint_sha256"],
        "bundle sidecar base digest",
    )
    if bound_base_digest != base_digest:
        raise ValueError("in-flight bundle sidecar binding disagrees with base")
    _, actual_base_size, actual_base_digest = _read_bounded_artifact(
        checkpoint_path,
        maximum_bytes=CHECKPOINT_MAX_BYTES,
        label="bundle base checkpoint",
    )
    _, actual_sidecar_size, actual_sidecar_digest = _read_bounded_artifact(
        sidecar_path,
        maximum_bytes=CHECKPOINT_MAX_BYTES,
        label="bundle in-flight gradient sidecar",
    )
    if (actual_base_size, actual_base_digest) != (base_size, base_digest):
        raise ValueError("in-flight bundle base identity drifted")
    if (actual_sidecar_size, actual_sidecar_digest) != (
        sidecar_size,
        sidecar_digest,
    ):
        raise ValueError("in-flight bundle sidecar identity drifted")
    return root, size, digest


def _load_checkpoint(
    path: Path,
    *,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[RuntimeState, dict[str, Any], int, str]:
    raw, size, digest = _read_torch_checkpoint(path)
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise ValueError("checkpoint size disagrees with bundle manifest")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("checkpoint digest disagrees with bundle manifest")
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "dataset_identity",
            "permutation",
            "loader_contract",
            "progress",
            "model",
            "optimizer",
            "scheduler",
            "commit_boundary_torch_rng_state",
        },
        "checkpoint",
    )
    if root["schema_version"] != CHECKPOINT_VERSION:
        raise ValueError("checkpoint schema version drifted")
    if root["dataset_identity"] != _dataset_identity():
        raise ValueError("checkpoint dataset identity drifted")
    if root["permutation"] != list(PERMUTATION):
        raise ValueError("checkpoint permutation drifted")
    loader_contract = _require_exact_keys(
        root["loader_contract"],
        {
            "batch_size",
            "num_workers",
            "prefetch_factor",
            "persistent_workers",
            "pin_memory",
            "multiprocessing_context",
            "in_order",
            "loader_generator_seed",
        },
        "loader_contract",
    )
    if loader_contract != _loader_contract():
        raise ValueError("loader contract values drifted")
    progress = _require_exact_keys(
        root["progress"],
        {
            "optimizer_committed_cursor",
            "main_loop_consumed_cursor",
            "sampler_emitted_cursor",
            "optimizer_steps",
            "committed_sample_ids",
            "consumed_sample_ids",
            "in_flight_gradients_serialized",
        },
        "progress",
    )
    committed = _require_integer(
        progress["optimizer_committed_cursor"],
        "optimizer_committed_cursor",
    )
    consumed = _require_integer(
        progress["main_loop_consumed_cursor"],
        "main_loop_consumed_cursor",
    )
    emitted = _require_integer(
        progress["sampler_emitted_cursor"],
        "sampler_emitted_cursor",
    )
    steps = _require_integer(progress["optimizer_steps"], "optimizer_steps")
    if not 0 <= committed < consumed <= emitted <= len(PERMUTATION):
        raise ValueError("checkpoint cursor ordering drifted")
    if committed != COMMITTED_CURSOR_AT_CRASH:
        raise ValueError("optimizer committed cursor drifted")
    if consumed != CRASH_AFTER_CONSUMED:
        raise ValueError("main-loop consumed cursor drifted")
    if committed % ACCUMULATION_STEPS != 0:
        raise ValueError("committed cursor is not an accumulation boundary")
    if steps != committed // ACCUMULATION_STEPS:
        raise ValueError("optimizer step count drifted")
    if progress["committed_sample_ids"] != list(PERMUTATION[:committed]):
        raise ValueError("committed sample ledger drifted")
    if progress["consumed_sample_ids"] != list(PERMUTATION[:consumed]):
        raise ValueError("consumed sample ledger drifted")
    if progress["in_flight_gradients_serialized"] is not False:
        raise ValueError("control checkpoint must exclude in-flight gradients")

    state = _new_state()
    if not isinstance(root["model"], dict) or not isinstance(
        root["optimizer"], dict
    ):
        raise ValueError("model and optimizer states must be mappings")
    model_payload = root["model"]
    expected_model = state.model.state_dict()
    if set(model_payload) != set(expected_model):
        raise ValueError("model state fields drifted")
    for name, expected_tensor in expected_model.items():
        tensor = model_payload[name]
        if not isinstance(tensor, Tensor):
            raise ValueError(f"model tensor {name} is not a tensor")
        if tensor.dtype != expected_tensor.dtype or tensor.shape != expected_tensor.shape:
            raise ValueError(f"model tensor {name} dtype or shape drifted")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"model tensor {name} must be finite")

    optimizer_payload = root["optimizer"]
    if set(optimizer_payload) != {"state", "param_groups"}:
        raise ValueError("optimizer state fields drifted")
    expected_optimizer = state.optimizer.state_dict()
    if optimizer_payload["param_groups"] != expected_optimizer["param_groups"]:
        raise ValueError("optimizer parameter-group contract drifted")
    raw_optimizer_state = optimizer_payload["state"]
    expected_parameter_ids = expected_optimizer["param_groups"][0]["params"]
    if not isinstance(raw_optimizer_state, Mapping) or set(
        raw_optimizer_state
    ) != set(expected_parameter_ids):
        raise ValueError("optimizer state parameter IDs drifted")
    for parameter_id, parameter in zip(
        expected_parameter_ids,
        state.model.parameters(),
        strict=True,
    ):
        parameter_state = raw_optimizer_state[parameter_id]
        if not isinstance(parameter_state, Mapping) or set(parameter_state) != {
            "momentum_buffer"
        }:
            raise ValueError("optimizer momentum state fields drifted")
        momentum = parameter_state["momentum_buffer"]
        if not isinstance(momentum, Tensor):
            raise ValueError("optimizer momentum buffer is not a tensor")
        if momentum.dtype != parameter.dtype or momentum.shape != parameter.shape:
            raise ValueError("optimizer momentum buffer dtype or shape drifted")
        if not bool(torch.isfinite(momentum).all()):
            raise ValueError("optimizer momentum buffer must be finite")

    state.model.load_state_dict(model_payload, strict=True)
    state.optimizer.load_state_dict(optimizer_payload)
    scheduler_payload = _require_exact_keys(
        root["scheduler"],
        {
            "step_size",
            "gamma",
            "base_lrs",
            "last_epoch",
            "_step_count",
            "_is_initial",
            "_get_lr_called_within_step",
            "_last_lr",
        },
        "scheduler",
    )
    expected_scheduler = state.scheduler.state_dict()
    for field in (
        "step_size",
        "gamma",
        "base_lrs",
        "_is_initial",
        "_get_lr_called_within_step",
    ):
        if scheduler_payload[field] != expected_scheduler[field]:
            raise ValueError(f"scheduler {field} contract drifted")
    scheduler_epoch = _require_integer(
        scheduler_payload["last_epoch"], "scheduler last_epoch"
    )
    scheduler_step_count = _require_integer(
        scheduler_payload["_step_count"], "scheduler step_count"
    )
    current_learning_rate = float(state.optimizer.param_groups[0]["lr"])
    if not math.isfinite(current_learning_rate):
        raise ValueError("optimizer learning rate must be finite")
    if scheduler_epoch != steps or scheduler_step_count != steps + 1:
        raise ValueError("scheduler progress disagrees with optimizer steps")
    if scheduler_payload["_last_lr"] != [current_learning_rate]:
        raise ValueError("scheduler last learning rate drifted")
    state.scheduler.load_state_dict(scheduler_payload)

    commit_rng = root["commit_boundary_torch_rng_state"]
    expected_rng = torch.get_rng_state()
    if not isinstance(commit_rng, Tensor):
        raise ValueError("commit-boundary Torch RNG state is not a tensor")
    if (
        commit_rng.dtype != torch.uint8
        or commit_rng.ndim != 1
        or commit_rng.shape != expected_rng.shape
    ):
        raise ValueError("commit-boundary Torch RNG dtype or shape drifted")
    try:
        torch.set_rng_state(commit_rng)
    except RuntimeError as error:
        raise ValueError("commit-boundary Torch RNG state is invalid") from error
    state.optimizer_steps = steps
    state.committed_sample_ids = list(PERMUTATION[:committed])
    state.commit_boundary_torch_rng_state = commit_rng.clone()
    if any(parameter.grad is not None for parameter in state.model.parameters()):
        raise AssertionError("checkpoint unexpectedly restored gradients")
    return state, progress, size, digest


def _load_inflight_sidecar(
    path: Path,
    *,
    expected_base_sha256: str,
    base_progress: Mapping[str, Any],
    state: RuntimeState,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
    restore_gradients: bool = True,
    restore_rng: bool = True,
) -> tuple[list[int], dict[str, Any], int, str]:
    if any(parameter.grad is not None for parameter in state.model.parameters()):
        raise AssertionError("base checkpoint must load without gradients")
    raw, size, digest = _read_torch_checkpoint(path)
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise ValueError("in-flight sidecar size disagrees with bundle manifest")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("in-flight sidecar digest disagrees with bundle manifest")
    root = _require_exact_keys(
        raw,
        {
            "schema_version",
            "base_checkpoint_schema_version",
            "base_checkpoint_sha256",
            "dataset_identity",
            "permutation",
            "progress",
            "crash_observed_torch_rng_state",
            "gradients",
        },
        "in-flight sidecar",
    )
    if root["schema_version"] != INFLIGHT_SIDECAR_VERSION:
        raise ValueError("in-flight sidecar schema version drifted")
    if root["base_checkpoint_schema_version"] != CHECKPOINT_VERSION:
        raise ValueError("in-flight sidecar base schema version drifted")
    if root["base_checkpoint_sha256"] != expected_base_sha256:
        raise ValueError("in-flight sidecar base checkpoint digest drifted")
    if root["dataset_identity"] != _dataset_identity():
        raise ValueError("in-flight sidecar dataset identity drifted")
    if root["permutation"] != list(PERMUTATION):
        raise ValueError("in-flight sidecar permutation drifted")
    progress = _require_exact_keys(
        root["progress"],
        {
            "optimizer_committed_cursor",
            "main_loop_consumed_cursor",
            "sampler_emitted_cursor",
            "optimizer_steps",
            "committed_sample_ids",
            "consumed_sample_ids",
            "pending_window_sample_ids",
            "accumulation_position",
            "accumulation_steps",
            "loss_divisor",
        },
        "in-flight progress",
    )
    integer_fields = (
        "optimizer_committed_cursor",
        "main_loop_consumed_cursor",
        "sampler_emitted_cursor",
        "optimizer_steps",
        "accumulation_position",
        "accumulation_steps",
        "loss_divisor",
    )
    integers = {
        field: _require_integer(progress[field], field) for field in integer_fields
    }
    for field in (
        "optimizer_committed_cursor",
        "main_loop_consumed_cursor",
        "sampler_emitted_cursor",
        "optimizer_steps",
        "committed_sample_ids",
        "consumed_sample_ids",
    ):
        if progress[field] != base_progress[field]:
            raise ValueError(f"in-flight sidecar {field} disagrees with base checkpoint")
    pending_ids = progress["pending_window_sample_ids"]
    if pending_ids != list(
        PERMUTATION[COMMITTED_CURSOR_AT_CRASH:CRASH_AFTER_CONSUMED]
    ):
        raise ValueError("in-flight pending sample ledger drifted")
    if integers["accumulation_position"] != len(pending_ids):
        raise ValueError("in-flight accumulation position drifted")
    if not 0 < integers["accumulation_position"] < ACCUMULATION_STEPS:
        raise ValueError("in-flight accumulation position is not incomplete")
    if integers["accumulation_steps"] != ACCUMULATION_STEPS:
        raise ValueError("in-flight accumulation steps drifted")
    if integers["loss_divisor"] != ACCUMULATION_STEPS:
        raise ValueError("in-flight loss divisor drifted")

    crash_rng = root["crash_observed_torch_rng_state"]
    expected_rng = torch.get_rng_state()
    if not isinstance(crash_rng, Tensor):
        raise ValueError("crash-observed Torch RNG state is not a tensor")
    if (
        crash_rng.dtype != torch.uint8
        or crash_rng.ndim != 1
        or crash_rng.shape != expected_rng.shape
    ):
        raise ValueError("crash-observed Torch RNG dtype or shape drifted")

    gradients = _require_exact_keys(
        root["gradients"],
        set(dict(state.model.named_parameters())),
        "in-flight gradients",
    )
    for name, parameter in state.model.named_parameters():
        gradient = gradients[name]
        if not isinstance(gradient, Tensor):
            raise ValueError(f"in-flight gradient {name} is not a tensor")
        if gradient.dtype != parameter.dtype or gradient.shape != parameter.shape:
            raise ValueError(f"in-flight gradient {name} dtype or shape drifted")
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"in-flight gradient {name} must be finite")
        if restore_gradients:
            parameter.grad = gradient.detach().cpu().clone()
    if restore_rng:
        try:
            torch.set_rng_state(crash_rng)
        except RuntimeError as error:
            raise ValueError("crash-observed Torch RNG state is invalid") from error
    return list(pending_ids), progress, size, digest


def _publish_inflight_bundle(
    checkpoint_path: Path,
    state: RuntimeState,
    segment: Mapping[str, object],
) -> dict[str, object]:
    size, digest = _write_checkpoint(
        checkpoint_path,
        _checkpoint_payload(state, segment),
    )
    sidecar_path = _inflight_sidecar_path(checkpoint_path)
    sidecar_size, sidecar_digest = _write_checkpoint(
        sidecar_path,
        _inflight_sidecar_payload(
            state,
            segment,
            base_checkpoint_sha256=digest,
        ),
    )
    manifest_path = _inflight_bundle_manifest_path(checkpoint_path)
    manifest_size, manifest_digest = _write_canonical_json(
        manifest_path,
        _inflight_bundle_manifest_payload(
            checkpoint_path,
            base_size_bytes=size,
            base_sha256=digest,
            sidecar_size_bytes=sidecar_size,
            sidecar_sha256=sidecar_digest,
        ),
    )
    return {
        "size_bytes": size,
        "sha256": digest,
        "commit_boundary_torch_rng_sha256": _tensor_sha256(
            state.commit_boundary_torch_rng_state
        ),
        "published_with_temp_file_and_os_replace": True,
        "inflight_sidecar": {
            "schema_version": INFLIGHT_SIDECAR_VERSION,
            "size_bytes": sidecar_size,
            "sha256": sidecar_digest,
            "base_checkpoint_sha256": digest,
            "crash_observed_torch_rng_sha256": _tensor_sha256(
                torch.get_rng_state()
            ),
            "published_after_base_checkpoint": True,
        },
        "bundle_manifest": {
            "schema_version": BUNDLE_MANIFEST_VERSION,
            "size_bytes": manifest_size,
            "sha256": manifest_digest,
            "publication_state": "complete",
            "published_last_after_payload_artifacts": True,
        },
    }


def _copy_fault_snapshot_artifact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def _expect_inflight_bundle_rejection(
    checkpoint_path: Path,
    *,
    expected_message: str,
) -> str:
    try:
        _load_inflight_bundle_manifest(checkpoint_path)
    except ValueError as error:
        message = str(error)
        if expected_message not in message:
            raise AssertionError(
                "bundle fault rejected for the wrong reason: " + message
            ) from error
        return message
    raise AssertionError("incomplete or tampered in-flight bundle was accepted")


def _run_bundle_publication_fault_injection(
    checkpoint_path: Path,
) -> dict[str, object]:
    source_sidecar = _inflight_sidecar_path(checkpoint_path)
    source_manifest = _inflight_bundle_manifest_path(checkpoint_path)
    root = checkpoint_path.parent

    base_only = root / "fault-base-only" / checkpoint_path.name
    _copy_fault_snapshot_artifact(checkpoint_path, base_only)
    base_only_error = _expect_inflight_bundle_rejection(
        base_only,
        expected_message="manifest is missing; publication is incomplete",
    )

    payloads_without_manifest = (
        root / "fault-payloads-without-manifest" / checkpoint_path.name
    )
    _copy_fault_snapshot_artifact(checkpoint_path, payloads_without_manifest)
    _copy_fault_snapshot_artifact(
        source_sidecar,
        _inflight_sidecar_path(payloads_without_manifest),
    )
    payloads_without_manifest_error = _expect_inflight_bundle_rejection(
        payloads_without_manifest,
        expected_message="manifest is missing; publication is incomplete",
    )

    manifest_without_sidecar = (
        root / "fault-manifest-without-sidecar" / checkpoint_path.name
    )
    _copy_fault_snapshot_artifact(checkpoint_path, manifest_without_sidecar)
    _copy_fault_snapshot_artifact(
        source_manifest,
        _inflight_bundle_manifest_path(manifest_without_sidecar),
    )
    manifest_without_sidecar_error = _expect_inflight_bundle_rejection(
        manifest_without_sidecar,
        expected_message="gradient sidecar is missing; publication is incomplete",
    )

    tampered_sidecar = root / "fault-tampered-sidecar" / checkpoint_path.name
    _copy_fault_snapshot_artifact(checkpoint_path, tampered_sidecar)
    tampered_sidecar_path = _inflight_sidecar_path(tampered_sidecar)
    _copy_fault_snapshot_artifact(source_sidecar, tampered_sidecar_path)
    _copy_fault_snapshot_artifact(
        source_manifest,
        _inflight_bundle_manifest_path(tampered_sidecar),
    )
    tampered = bytearray(tampered_sidecar_path.read_bytes())
    tampered[-1] ^= 1
    tampered_sidecar_path.write_bytes(tampered)
    tampered_sidecar_error = _expect_inflight_bundle_rejection(
        tampered_sidecar,
        expected_message="sidecar identity drifted",
    )

    _load_inflight_bundle_manifest(checkpoint_path)
    return {
        "complete_bundle_accepted": True,
        "base_only_rejected": True,
        "base_and_sidecar_without_manifest_rejected": True,
        "manifest_without_sidecar_rejected": True,
        "tampered_sidecar_after_manifest_rejected": True,
        "rejection_reasons": {
            "base_only": base_only_error,
            "base_and_sidecar_without_manifest": payloads_without_manifest_error,
            "manifest_without_sidecar": manifest_without_sidecar_error,
            "tampered_sidecar_after_manifest": tampered_sidecar_error,
        },
    }


def _run_worker(mode: WorkerMode, checkpoint_path: Path) -> dict[str, object]:
    torch.manual_seed(STOCHASTIC_MASK_SEED)
    if mode == "baseline":
        state = _new_state()
        segment = _run_segment(
            state,
            start_cursor=0,
            stop_after_records=None,
        )
        checkpoint: dict[str, object] | None = None
    elif mode == "phase1":
        state = _new_state()
        segment = _run_segment(
            state,
            start_cursor=0,
            stop_after_records=CRASH_AFTER_CONSUMED,
        )
        checkpoint = _publish_inflight_bundle(checkpoint_path, state, segment)
    elif mode == "resume_committed":
        state, progress, size, digest = _load_checkpoint(checkpoint_path)
        start_field = "optimizer_committed_cursor"
        start_cursor = _require_integer(progress[start_field], start_field)
        segment = _run_segment(
            state,
            start_cursor=start_cursor,
            stop_after_records=None,
        )
        checkpoint = {
            "size_bytes": size,
            "sha256": digest,
            "loaded_with_torch_weights_only": True,
            "selected_resume_cursor_field": start_field,
            "gradients_present_immediately_after_load": False,
            "torch_rng_restored_from": "commit_boundary_base_checkpoint",
        }
    else:
        bundle_manifest, manifest_size, manifest_digest = (
            _load_inflight_bundle_manifest(checkpoint_path)
        )
        artifacts = bundle_manifest["artifacts"]
        if not isinstance(artifacts, dict):
            raise AssertionError("validated bundle artifacts must be a mapping")
        base_artifact = artifacts["base_checkpoint"]
        sidecar_artifact = artifacts["inflight_gradient_sidecar"]
        if not isinstance(base_artifact, dict) or not isinstance(
            sidecar_artifact, dict
        ):
            raise AssertionError("validated bundle entries must be mappings")
        state, progress, size, digest = _load_checkpoint(
            checkpoint_path,
            expected_size_bytes=_require_integer(
                base_artifact["size_bytes"], "bundle base size"
            ),
            expected_sha256=_require_sha256(
                base_artifact["sha256"], "bundle base digest"
            ),
        )
        rng_after_base_load = _tensor_sha256(torch.get_rng_state())
        restore_gradients = mode != "resume_consumed"
        restore_rng = mode != "resume_inflight_wrong_rng"
        pending_ids, sidecar_progress, sidecar_size, sidecar_digest = (
            _load_inflight_sidecar(
                _inflight_sidecar_path(checkpoint_path),
                expected_base_sha256=digest,
                base_progress=progress,
                state=state,
                expected_size_bytes=_require_integer(
                    sidecar_artifact["size_bytes"], "bundle sidecar size"
                ),
                expected_sha256=_require_sha256(
                    sidecar_artifact["sha256"], "bundle sidecar digest"
                ),
                restore_gradients=restore_gradients,
                restore_rng=restore_rng,
            )
        )
        start_field = "main_loop_consumed_cursor"
        start_cursor = _require_integer(sidecar_progress[start_field], start_field)
        rng_after_sidecar_policy = _tensor_sha256(torch.get_rng_state())
        segment = _run_segment(
            state,
            start_cursor=start_cursor,
            stop_after_records=None,
            initial_pending_sample_ids=(
                pending_ids if restore_gradients else None
            ),
        )
        checkpoint = {
            "size_bytes": size,
            "sha256": digest,
            "loaded_with_torch_weights_only": True,
            "selected_resume_cursor_field": start_field,
            "gradients_present_immediately_after_base_load": False,
            "inflight_sidecar": {
                "schema_version": INFLIGHT_SIDECAR_VERSION,
                "size_bytes": sidecar_size,
                "sha256": sidecar_digest,
                "base_checkpoint_sha256": digest,
                "loaded_with_torch_weights_only": True,
                "gradients_restored_before_resume": restore_gradients,
                "crash_rng_restored_before_resume": restore_rng,
            },
            "rng_after_base_load_sha256": rng_after_base_load,
            "rng_after_sidecar_policy_sha256": rng_after_sidecar_policy,
            "torch_rng_restored_from": (
                "crash_observed_sidecar"
                if restore_rng
                else "commit_boundary_base_checkpoint_negative_control"
            ),
            "bundle_manifest": {
                "schema_version": BUNDLE_MANIFEST_VERSION,
                "size_bytes": manifest_size,
                "sha256": manifest_digest,
                "publication_state": bundle_manifest["publication_state"],
                "validated_before_payload_deserialization": True,
                "artifact_identities_rechecked_at_payload_load": True,
            },
        }
    return {
        "worker_mode": mode,
        "pid": os.getpid(),
        "segment": segment,
        "checkpoint": checkpoint,
        "terminal_state": _terminal_state(state),
    }


def _run_worker_process(
    script_path: Path,
    mode: WorkerMode,
    checkpoint_path: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--worker-mode",
            mode,
            "--checkpoint",
            str(checkpoint_path),
        ],
        cwd=script_path.parents[2],
        check=False,
        capture_output=True,
        timeout=180,
    )
    stderr = completed.stderr.decode("utf-8", errors="strict")
    if completed.returncode != 0:
        raise RuntimeError(f"{mode} failed ({completed.returncode}): {stderr}")
    if stderr:
        raise AssertionError(f"{mode} wrote stderr: {stderr}")
    report = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    if not isinstance(report, dict):
        raise AssertionError(f"{mode} report must be a JSON object")
    return report


def _max_parameter_difference(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> float:
    differences: list[float] = []
    if set(left) != set(right):
        raise AssertionError("parameter snapshot keys differ")
    for name in sorted(left):
        left_values = left[name]
        right_values = right[name]
        if not isinstance(left_values, list) or not isinstance(right_values, list):
            raise AssertionError("parameter snapshots must contain lists")
        if len(left_values) != len(right_values):
            raise AssertionError("parameter snapshot shapes differ")
        differences.extend(
            abs(float(a) - float(b))
            for a, b in zip(left_values, right_values, strict=True)
        )
    return max(differences, default=0.0)


def run_control(script_path: Path | None = None) -> dict[str, object]:
    """Execute uninterrupted, crash, two correct resumes, and two negatives."""

    entry = Path(__file__).resolve() if script_path is None else script_path.resolve()
    with tempfile.TemporaryDirectory(
        prefix="about-llm-optimizer-commit-resume-"
    ) as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        baseline = _run_worker_process(entry, "baseline", checkpoint_path)
        phase1 = _run_worker_process(entry, "phase1", checkpoint_path)
        publication_fault_injection = _run_bundle_publication_fault_injection(
            checkpoint_path
        )
        resume_committed = _run_worker_process(
            entry, "resume_committed", checkpoint_path
        )
        resume_consumed = _run_worker_process(
            entry, "resume_consumed", checkpoint_path
        )
        resume_inflight = _run_worker_process(
            entry, "resume_inflight", checkpoint_path
        )
        resume_inflight_wrong_rng = _run_worker_process(
            entry, "resume_inflight_wrong_rng", checkpoint_path
        )

    baseline_terminal = baseline["terminal_state"]
    committed_terminal = resume_committed["terminal_state"]
    consumed_terminal = resume_consumed["terminal_state"]
    inflight_terminal = resume_inflight["terminal_state"]
    wrong_rng_terminal = resume_inflight_wrong_rng["terminal_state"]
    phase_segment = phase1["segment"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            baseline_terminal,
            committed_terminal,
            consumed_terminal,
            inflight_terminal,
            wrong_rng_terminal,
            phase_segment,
        )
    ):
        raise AssertionError("worker state reports must be mappings")
    baseline_parameters = baseline_terminal["model_parameters"]
    committed_parameters = committed_terminal["model_parameters"]
    consumed_parameters = consumed_terminal["model_parameters"]
    inflight_parameters = inflight_terminal["model_parameters"]
    wrong_rng_parameters = wrong_rng_terminal["model_parameters"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            baseline_parameters,
            committed_parameters,
            consumed_parameters,
            inflight_parameters,
            wrong_rng_parameters,
        )
    ):
        raise AssertionError("model parameter reports must be mappings")
    committed_difference = _max_parameter_difference(
        baseline_parameters, committed_parameters
    )
    consumed_difference = _max_parameter_difference(
        baseline_parameters, consumed_parameters
    )
    inflight_difference = _max_parameter_difference(
        baseline_parameters, inflight_parameters
    )
    wrong_rng_difference = _max_parameter_difference(
        baseline_parameters, wrong_rng_parameters
    )
    pids = [
        int(baseline["pid"]),
        int(phase1["pid"]),
        int(resume_committed["pid"]),
        int(resume_consumed["pid"]),
        int(resume_inflight["pid"]),
        int(resume_inflight_wrong_rng["pid"]),
    ]
    baseline_masks = baseline["segment"]["stochastic_mask_sha256"]
    phase_masks = phase_segment["stochastic_mask_sha256"]
    committed_masks = resume_committed["segment"]["stochastic_mask_sha256"]
    consumed_masks = resume_consumed["segment"]["stochastic_mask_sha256"]
    inflight_masks = resume_inflight["segment"]["stochastic_mask_sha256"]
    wrong_rng_masks = resume_inflight_wrong_rng["segment"][
        "stochastic_mask_sha256"
    ]
    if not all(
        isinstance(value, list)
        for value in (
            baseline_masks,
            phase_masks,
            committed_masks,
            consumed_masks,
            inflight_masks,
            wrong_rng_masks,
        )
    ):
        raise AssertionError("stochastic mask traces must be lists")
    expected_emitted = CRASH_AFTER_CONSUMED + NUM_WORKERS * PREFETCH_FACTOR
    assertions = {
        "all_top_level_segments_use_distinct_processes": len(set(pids)) == 6,
        "phase_stochastic_prefix_matches_uninterrupted": (
            phase_masks == baseline_masks[:CRASH_AFTER_CONSUMED]
        ),
        "phase_consumed_cursor_leads_optimizer_commit_by_one": (
            phase_segment["consumed_sample_ids"]
            == list(PERMUTATION[:CRASH_AFTER_CONSUMED])
            and phase1["terminal_state"]["committed_sample_ids"]
            == list(PERMUTATION[:COMMITTED_CURSOR_AT_CRASH])
        ),
        "phase_contains_unserialized_in_flight_gradients": (
            phase_segment["pending_uncommitted_sample_ids"]
            == [PERMUTATION[COMMITTED_CURSOR_AT_CRASH]]
            and phase_segment["in_flight_gradient_tensor_count"] == 2
        ),
        "sampler_prefetch_leads_main_loop_consumption": (
            phase_segment["sampler_emitted_cursor_when_observed"]
            == expected_emitted
        ),
        "phase_used_two_real_dataloader_workers": (
            phase_segment["worker_ids_seen"] == [0, 1]
            and len(phase_segment["worker_pids_seen"]) == 2
        ),
        "committed_cursor_resume_replays_full_sample_ledger": (
            committed_terminal["committed_sample_ids"] == list(PERMUTATION)
        ),
        "committed_cursor_resume_matches_uninterrupted_bit_exact": (
            committed_terminal["fingerprint"]
            == baseline_terminal["fingerprint"]
            and committed_difference == 0.0
            and committed_masks
            == baseline_masks[COMMITTED_CURSOR_AT_CRASH:]
        ),
        "consumed_cursor_resume_silently_omits_uncommitted_sample": (
            consumed_terminal["committed_sample_ids"]
            == list(PERMUTATION[:COMMITTED_CURSOR_AT_CRASH])
            + list(PERMUTATION[CRASH_AFTER_CONSUMED:])
        ),
        "negative_control_keeps_optimizer_step_count_but_diverges": (
            consumed_terminal["optimizer_steps"]
            == baseline_terminal["optimizer_steps"]
            and consumed_terminal["fingerprint"]
            != baseline_terminal["fingerprint"]
            and consumed_difference > 0.0
            and consumed_terminal["torch_cpu_rng_sha256"]
            == baseline_terminal["torch_cpu_rng_sha256"]
            and consumed_masks == baseline_masks[CRASH_AFTER_CONSUMED:]
        ),
        "inflight_sidecar_resume_continues_pending_window": (
            resume_inflight["segment"]["start_cursor"]
            == CRASH_AFTER_CONSUMED
            and resume_inflight["segment"]["initial_pending_sample_ids"]
            == [PERMUTATION[COMMITTED_CURSOR_AT_CRASH]]
            and resume_inflight["segment"]["new_committed_windows"][0]
            == list(PERMUTATION[COMMITTED_CURSOR_AT_CRASH:4])
        ),
        "inflight_sidecar_resume_matches_uninterrupted_bit_exact": (
            inflight_terminal["committed_sample_ids"] == list(PERMUTATION)
            and inflight_terminal["fingerprint"]
            == baseline_terminal["fingerprint"]
            and inflight_difference == 0.0
            and inflight_masks == baseline_masks[CRASH_AFTER_CONSUMED:]
        ),
        "wrong_rng_with_complete_gradients_isolated_negative_control": (
            wrong_rng_terminal["committed_sample_ids"] == list(PERMUTATION)
            and wrong_rng_terminal["optimizer_steps"]
            == baseline_terminal["optimizer_steps"]
            and wrong_rng_terminal["fingerprint"]
            != baseline_terminal["fingerprint"]
            and wrong_rng_terminal["torch_cpu_rng_sha256"]
            != baseline_terminal["torch_cpu_rng_sha256"]
            and wrong_rng_difference > 0.0
            and wrong_rng_masks
            == baseline_masks[
                COMMITTED_CURSOR_AT_CRASH : len(PERMUTATION) - 1
            ]
            and wrong_rng_masks != baseline_masks[CRASH_AFTER_CONSUMED:]
        ),
        "resume_checkpoint_digest_is_identical_in_all_loaders": (
            resume_committed["checkpoint"]["sha256"]
            == resume_consumed["checkpoint"]["sha256"]
            == resume_inflight["checkpoint"]["sha256"]
            == resume_inflight_wrong_rng["checkpoint"]["sha256"]
            == phase1["checkpoint"]["sha256"]
        ),
        "inflight_sidecar_digest_and_base_binding_match_phase1": (
            resume_inflight["checkpoint"]["inflight_sidecar"]["sha256"]
            == phase1["checkpoint"]["inflight_sidecar"]["sha256"]
            and resume_inflight["checkpoint"]["inflight_sidecar"][
                "base_checkpoint_sha256"
            ]
            == phase1["checkpoint"]["sha256"]
        ),
        "rng_snapshots_select_commit_or_crash_boundary_as_declared": (
            phase1["checkpoint"]["commit_boundary_torch_rng_sha256"]
            != phase1["checkpoint"]["inflight_sidecar"][
                "crash_observed_torch_rng_sha256"
            ]
            and resume_consumed["checkpoint"][
                "rng_after_sidecar_policy_sha256"
            ]
            == phase1["checkpoint"]["inflight_sidecar"][
                "crash_observed_torch_rng_sha256"
            ]
            and resume_inflight["checkpoint"][
                "rng_after_sidecar_policy_sha256"
            ]
            == phase1["checkpoint"]["inflight_sidecar"][
                "crash_observed_torch_rng_sha256"
            ]
            and resume_inflight_wrong_rng["checkpoint"][
                "rng_after_sidecar_policy_sha256"
            ]
            == phase1["checkpoint"]["commit_boundary_torch_rng_sha256"]
        ),
        "bundle_manifest_matches_phase1_and_precedes_sidecar_load": (
            resume_inflight["checkpoint"]["bundle_manifest"]["sha256"]
            == phase1["checkpoint"]["bundle_manifest"]["sha256"]
            and resume_inflight["checkpoint"]["bundle_manifest"][
                "publication_state"
            ]
            == "complete"
            and resume_inflight["checkpoint"]["bundle_manifest"][
                "validated_before_payload_deserialization"
            ]
            is True
        ),
        "incomplete_and_tampered_bundle_snapshots_fail_closed": all(
            publication_fault_injection[field] is True
            for field in (
                "complete_bundle_accepted",
                "base_only_rejected",
                "base_and_sidecar_without_manifest_rejected",
                "manifest_without_sidecar_rejected",
                "tampered_sidecar_after_manifest_rejected",
            )
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"optimizer commit resume control failed: {failed}")

    return {
        "implementation": CONTROL_VERSION,
        "runtime": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "torch.float64",
            "model": "torch.nn.Linear(2,1)",
            "optimizer": "torch.optim.SGD(momentum=0.9)",
            "scheduler": "torch.optim.lr_scheduler.StepLR(step_size=2,gamma=0.5)",
            "loss": "torch.nn.functional.mse_loss",
            "stochastic_forward": "main-process inverted Bernoulli mask p=0.5",
            "accumulation_steps": ACCUMULATION_STEPS,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "multiprocessing_context": "spawn",
            "in_order": True,
        },
        "fixture": {
            "dataset_identity": _dataset_identity(),
            "permutation": list(PERMUTATION),
            "optimizer_committed_cursor_at_crash": COMMITTED_CURSOR_AT_CRASH,
            "main_loop_consumed_cursor_at_crash": CRASH_AFTER_CONSUMED,
            "sampler_emitted_cursor_at_crash": expected_emitted,
            "uncommitted_sample_id_requiring_replay": PERMUTATION[
                COMMITTED_CURSOR_AT_CRASH
            ],
            "stochastic_mask_seed": STOCHASTIC_MASK_SEED,
        },
        "processes": {
            "baseline_pid": pids[0],
            "phase1_pid": pids[1],
            "resume_committed_pid": pids[2],
            "resume_consumed_negative_control_pid": pids[3],
            "resume_inflight_gradient_pid": pids[4],
            "resume_inflight_wrong_rng_negative_control_pid": pids[5],
            "all_distinct": True,
        },
        "checkpoint": {
            "schema_version": CHECKPOINT_VERSION,
            "size_bytes": phase1["checkpoint"]["size_bytes"],
            "sha256": phase1["checkpoint"]["sha256"],
            "commit_boundary_torch_rng_sha256": phase1["checkpoint"][
                "commit_boundary_torch_rng_sha256"
            ],
            "torch_load_weights_only": True,
            "in_flight_gradients_serialized": False,
            "temporary_file_then_os_replace": True,
            "preload_size_cap_bytes": CHECKPOINT_MAX_BYTES,
            "inflight_sidecar": {
                "schema_version": INFLIGHT_SIDECAR_VERSION,
                "size_bytes": phase1["checkpoint"]["inflight_sidecar"][
                    "size_bytes"
                ],
                "sha256": phase1["checkpoint"]["inflight_sidecar"]["sha256"],
                "base_checkpoint_sha256": phase1["checkpoint"]["sha256"],
                "crash_observed_torch_rng_sha256": phase1["checkpoint"][
                    "inflight_sidecar"
                ]["crash_observed_torch_rng_sha256"],
                "accumulation_position": 1,
                "pending_window_sample_ids": [
                    PERMUTATION[COMMITTED_CURSOR_AT_CRASH]
                ],
                "gradient_tensor_count": len(
                    dict(_new_state().model.named_parameters())
                ),
                "published_after_base_checkpoint": True,
            },
            "bundle_manifest": {
                "schema_version": BUNDLE_MANIFEST_VERSION,
                "size_bytes": phase1["checkpoint"]["bundle_manifest"][
                    "size_bytes"
                ],
                "sha256": phase1["checkpoint"]["bundle_manifest"]["sha256"],
                "publication_state": "complete",
                "publication_sequence": [
                    "base_checkpoint",
                    "inflight_gradient_sidecar",
                    "bundle_manifest",
                ],
                "preload_size_cap_bytes": BUNDLE_MANIFEST_MAX_BYTES,
                "published_last_after_payload_artifacts": True,
            },
        },
        "comparisons": {
            "uninterrupted_fingerprint": baseline_terminal["fingerprint"],
            "committed_resume_fingerprint": committed_terminal["fingerprint"],
            "consumed_resume_fingerprint": consumed_terminal["fingerprint"],
            "inflight_resume_fingerprint": inflight_terminal["fingerprint"],
            "wrong_rng_resume_fingerprint": wrong_rng_terminal["fingerprint"],
            "committed_resume_model_max_abs_difference": committed_difference,
            "consumed_resume_model_max_abs_difference": consumed_difference,
            "inflight_resume_model_max_abs_difference": inflight_difference,
            "wrong_rng_resume_model_max_abs_difference": wrong_rng_difference,
            "uninterrupted_optimizer_steps": baseline_terminal["optimizer_steps"],
            "consumed_resume_optimizer_steps": consumed_terminal[
                "optimizer_steps"
            ],
            "committed_resume_sample_ledger": committed_terminal[
                "committed_sample_ids"
            ],
            "consumed_resume_sample_ledger": consumed_terminal[
                "committed_sample_ids"
            ],
            "inflight_resume_sample_ledger": inflight_terminal[
                "committed_sample_ids"
            ],
            "wrong_rng_resume_sample_ledger": wrong_rng_terminal[
                "committed_sample_ids"
            ],
            "uninterrupted_terminal_torch_rng_sha256": baseline_terminal[
                "torch_cpu_rng_sha256"
            ],
            "consumed_omission_terminal_torch_rng_sha256": consumed_terminal[
                "torch_cpu_rng_sha256"
            ],
            "wrong_rng_terminal_torch_rng_sha256": wrong_rng_terminal[
                "torch_cpu_rng_sha256"
            ],
        },
        "paths": {
            "uninterrupted": baseline,
            "phase1_crash_after_backward": phase1,
            "resume_from_optimizer_committed_cursor": resume_committed,
            "resume_from_main_loop_consumed_cursor_negative_control": (
                resume_consumed
            ),
            "resume_from_consumed_cursor_with_inflight_gradients": (
                resume_inflight
            ),
            "resume_with_inflight_gradients_but_commit_boundary_rng_negative_control": (
                resume_inflight_wrong_rng
            ),
            "bundle_publication_fault_injection": publication_fault_injection,
        },
        "assertions": assertions,
        "scope": {
            "real_two_worker_dataloader_prefetch_executed": True,
            "real_float64_backward_and_sgd_momentum_steps_executed": True,
            "gradient_accumulation_window_executed": True,
            "crash_after_consumption_before_optimizer_commit_executed": True,
            "in_flight_gradients_intentionally_excluded_from_checkpoint": True,
            "committed_cursor_replay_matches_uninterrupted_bit_exact": True,
            "consumed_cursor_skip_negative_control_executed": True,
            "negative_control_equal_optimizer_step_count_executed": True,
            "wrong_rng_with_complete_gradients_negative_control_executed": True,
            "checkpoint_loaded_with_torch_weights_only": True,
            "checkpoint_temp_file_and_os_replace_executed": True,
            "sampler_queue_or_worker_state_serialized": False,
            "in_flight_gradient_checkpoint_resume_executed": True,
            "in_flight_sidecar_bound_to_base_checkpoint_digest": True,
            "manifest_last_bundle_completeness_gate_executed": True,
            "incomplete_and_tampered_bundle_fault_injection_executed": True,
            "manifest_artifact_hashes_rechecked_at_payload_load": True,
            "base_checkpoint_and_gradient_sidecar_atomic_publication_proved": False,
            "concurrent_directory_replacement_or_storage_snapshot_proved": False,
            "checkpoint_and_sample_commit_atomic_transaction_proved": False,
            "power_loss_directory_fsync_or_storage_durability_proved": False,
            "main_process_stochastic_mask_and_torch_cpu_rng_resume_executed": True,
            "real_step_lr_advanced_after_optimizer_commit_executed": True,
            "worker_rng_or_multi_epoch_policy_executed": False,
            "grad_scaler_or_cuda_amp_executed": False,
            "distributed_sampler_ddp_fsdp_zero_or_sharded_state_executed": False,
            "target_llm_trainer_dataset_quality_or_convergence_proved": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-mode",
        choices=(
            "baseline",
            "phase1",
            "resume_committed",
            "resume_consumed",
            "resume_inflight",
            "resume_inflight_wrong_rng",
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker_mode is None:
        if args.checkpoint is not None:
            raise SystemExit("--checkpoint requires --worker-mode")
        report = run_control()
    else:
        if args.checkpoint is None:
            raise SystemExit("--worker-mode requires --checkpoint")
        report = _run_worker(args.worker_mode, args.checkpoint)
    sys.stdout.buffer.write(_encode_json(report, pretty=True).encode("utf-8"))


if __name__ == "__main__":
    main()
