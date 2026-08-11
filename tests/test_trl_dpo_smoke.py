from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "single-gpu-finetuning" / "smoke_trl_dpo.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_trl_dpo", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DPO smoke script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_offline_trl_dpo_smoke_executes_real_pairwise_training() -> None:
    report = _load_script().run_smoke(steps=20)

    assert report["pair_count"] == 2
    assert report["prompt_masked_token_count"] > 0
    assert report["completion_token_count"] > 0
    assert report["initial_dpo_loss"] == pytest.approx(0.693147, rel=1e-3)
    assert report["final_dpo_loss"] < report["initial_dpo_loss"]
    assert report["reference_parameters_unchanged"] is True
    assert "do not prove target-model preference quality" in report["evidence_boundary"]
