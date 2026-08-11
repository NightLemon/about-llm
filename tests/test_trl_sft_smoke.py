from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("trl")


def test_offline_trl_smoke_preserves_only_assistant_labels_and_overfits() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "projects"
        / "single-gpu-finetuning"
        / "smoke_trl_sft.py"
    )
    spec = importlib.util.spec_from_file_location("trl_sft_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_smoke(steps=8)

    assert report["record_count"] == 2
    assert report["supervised_label_count"] > 0
    assert report["ignored_label_count"] > 0
    assert report["final_loss"] < report["initial_loss"]
    assert str(report["data_manifest_fingerprint"]).startswith("sha256:")
    assert str(report["split_manifest_fingerprint"]).startswith("sha256:")
    assert str(report["binding_fingerprint"]).startswith("sha256:")
    assert str(report["near_duplicate_manifest_fingerprint"]).startswith("sha256:")
    assert str(report["readiness_manifest_fingerprint"]).startswith("sha256:")
    assert str(report["mask_manifest_fingerprint"]).startswith("sha256:")
    assert "do not prove target-model quality" in report["evidence_boundary"]
