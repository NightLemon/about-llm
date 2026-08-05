from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("peft")


def test_offline_peft_smoke_trains_only_adapter_and_merges() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "projects" / "single-gpu-finetuning" / "smoke_peft.py"
    )
    spec = importlib.util.spec_from_file_location("peft_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_smoke(steps=8)

    parameters = report["parameter_report"]
    assert parameters["trainable_parameters"] < parameters["total_parameters"]
    assert report["final_loss"] < report["initial_loss"]
    assert report["adapter_tensor_count"] > 0
    assert report["maximum_merge_error"] < 1e-5
