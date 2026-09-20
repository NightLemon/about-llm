from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "single-gpu-finetuning" / "optimizer_commit_resume_control.py"


def _load_control() -> ModuleType:
    spec = importlib.util.spec_from_file_location("optimizer_commit_resume_control", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import optimizer commit resume control")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_checkpoint_payload(control: ModuleType) -> dict[str, object]:
    state = control._new_state()
    features = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
    target = torch.tensor([[0.1]], dtype=torch.float64)
    loss = torch.nn.functional.mse_loss(state.model(features), target)
    (loss / control.ACCUMULATION_STEPS).backward()
    control._commit_window(state, list(control.PERMUTATION[:2]), rescale_partial=False)
    return control._checkpoint_payload(
        state,
        {
            "consumed_sample_ids": list(control.PERMUTATION[:3]),
            "sampler_emitted_cursor_when_observed": 3,
        },
    )


def test_checkpoint_loader_accepts_the_current_steplr_schema(tmp_path: Path) -> None:
    control = _load_control()
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _valid_checkpoint_payload(control)
    control._write_checkpoint(checkpoint, payload)

    restored, progress, _, _ = control._load_checkpoint(checkpoint)

    assert progress["optimizer_steps"] == 1
    assert restored.scheduler.state_dict() == payload["scheduler"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("static", "scheduler gamma contract drifted"),
        ("unknown", "scheduler fields drifted"),
    ],
)
def test_checkpoint_loader_rejects_scheduler_static_or_unknown_fields(
    tmp_path: Path, mutation: str, message: str
) -> None:
    control = _load_control()
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _valid_checkpoint_payload(control)
    scheduler = payload["scheduler"]
    assert isinstance(scheduler, dict)
    if mutation == "static":
        scheduler["gamma"] = 0.25
    else:
        scheduler["unrecognized_scheduler_field"] = False
    control._write_checkpoint(checkpoint, payload)

    with pytest.raises(ValueError, match=message):
        control._load_checkpoint(checkpoint)
