"""Small runtime helpers shared by the executable fine-tuning entry points."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any


def write_strict_json(path: Path, payload: object) -> None:
    """Write a human-readable artifact while rejecting non-finite numbers."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def normalize_trainer_metrics(metrics: Mapping[str, Any]) -> dict[str, object]:
    """Convert scalar-like Trainer metrics into strict-JSON values."""

    normalized: dict[str, object] = {}
    for name, raw_value in metrics.items():
        value = raw_value.item() if hasattr(raw_value, "item") else raw_value
        if value is None or isinstance(value, (bool, int, float, str)):
            normalized[str(name)] = value
            continue
        raise TypeError(
            f"trainer metric {name!r} must be a JSON scalar, got {type(value).__name__}"
        )
    return normalized


def training_runtime_identity(
    torch_module: Any, package_names: tuple[str, ...]
) -> dict[str, object]:
    """Record the runtime versions needed to interpret one local training run."""

    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "packages": {name: version(name) for name in package_names},
    }


def reset_cuda_peak_memory(torch_module: Any, device: Any) -> None:
    """Start a local allocator peak window when Trainer selected CUDA."""

    if getattr(device, "type", None) != "cuda":
        return
    index = _cuda_device_index(torch_module, device)
    torch_module.cuda.reset_peak_memory_stats(index)


def cuda_memory_snapshot(torch_module: Any, device: Any) -> dict[str, object]:
    """Read process-local CUDA allocator counters without importing torch here."""

    if getattr(device, "type", None) != "cuda":
        return {"cuda_executed": False}
    index = _cuda_device_index(torch_module, device)
    properties = torch_module.cuda.get_device_properties(index)
    major, minor = torch_module.cuda.get_device_capability(index)
    return {
        "cuda_executed": True,
        "device_index": index,
        "device_name": str(properties.name),
        "compute_capability": f"{major}.{minor}",
        "total_device_memory_bytes": int(properties.total_memory),
        "allocated_bytes": int(torch_module.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch_module.cuda.memory_reserved(index)),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(index)),
    }


def _cuda_device_index(torch_module: Any, device: Any) -> int:
    index = getattr(device, "index", None)
    return int(torch_module.cuda.current_device() if index is None else index)
