from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from about_llm.finetuning.training_runtime import (
    cuda_memory_snapshot,
    normalize_trainer_metrics,
    reset_cuda_peak_memory,
    training_runtime_identity,
    write_strict_json,
)

pytestmark = pytest.mark.contract


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _FakeCuda:
    def __init__(self) -> None:
        self.reset_index: int | None = None

    def current_device(self) -> int:
        return 0

    def reset_peak_memory_stats(self, index: int) -> None:
        self.reset_index = index

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(name="teaching-gpu", total_memory=8 * 2**30)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (8, 6)

    def memory_allocated(self, index: int) -> int:
        return 100 + index

    def memory_reserved(self, index: int) -> int:
        return 200 + index

    def max_memory_allocated(self, index: int) -> int:
        return 300 + index

    def max_memory_reserved(self, index: int) -> int:
        return 400 + index


def test_normalize_trainer_metrics_accepts_only_json_scalars() -> None:
    assert normalize_trainer_metrics(
        {"loss": _Scalar(0.25), "step": 1, "finished": True}
    ) == {"loss": 0.25, "step": 1, "finished": True}

    with pytest.raises(TypeError, match="must be a JSON scalar"):
        normalize_trainer_metrics({"history": [0.5, 0.25]})


def test_cuda_memory_snapshot_distinguishes_cpu_from_process_local_cuda() -> None:
    torch_module = SimpleNamespace(cuda=_FakeCuda())
    cpu = SimpleNamespace(type="cpu", index=None)
    cuda = SimpleNamespace(type="cuda", index=None)

    assert cuda_memory_snapshot(torch_module, cpu) == {"cuda_executed": False}

    reset_cuda_peak_memory(torch_module, cuda)
    snapshot = cuda_memory_snapshot(torch_module, cuda)

    assert torch_module.cuda.reset_index == 0
    assert snapshot["device_name"] == "teaching-gpu"
    assert snapshot["compute_capability"] == "8.6"
    assert snapshot["total_device_memory_bytes"] == 8 * 2**30
    assert snapshot["peak_allocated_bytes"] == 300
    assert snapshot["peak_reserved_bytes"] == 400


def test_write_strict_json_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_strict_json(output, {"loss": 0.25})

    assert json.loads(output.read_text(encoding="utf-8")) == {"loss": 0.25}
    with pytest.raises(ValueError, match="Out of range float values"):
        write_strict_json(output, {"loss": float("nan")})


def test_training_runtime_identity_names_version_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "about_llm.finetuning.training_runtime.version",
        lambda name: {"transformers": "4.test", "trl": "0.test"}[name],
    )
    torch_module = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    identity = training_runtime_identity(torch_module, ("transformers", "trl"))

    assert identity["torch_version"] == "2.test"
    assert identity["torch_cuda_version"] == "12.test"
    assert identity["cuda_available"] is True
    assert identity["packages"] == {"transformers": "4.test", "trl": "0.test"}
