from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("transformers")


def test_offline_transformers_smoke_reduces_loss() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "projects" / "transformers-basics" / "smoke_tiny.py"
    )
    spec = importlib.util.spec_from_file_location("transformers_smoke_tiny", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_smoke(steps=8)

    assert report["final_loss"] < report["initial_loss"]
    assert report["generated_shape"] == [2, 5]
