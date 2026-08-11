from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("peft")


def _load_smoke():
    path = (
        Path(__file__).resolve().parents[1] / "projects" / "single-gpu-finetuning" / "smoke_peft.py"
    )
    spec = importlib.util.spec_from_file_location("peft_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_peft_smoke_saves_reloads_merges_and_exports() -> None:
    report = _load_smoke().run_smoke(steps=8)

    parameters = report["parameter_report"]
    assert parameters["trainable_parameters"] < parameters["total_parameters"]
    training = report["training"]
    assert training["final_loss"] < training["initial_loss"]
    assert training["base_parameters_unchanged"] is True
    assert training["adapter_tensor_count"] == 4
    assert len(training["adapter_tensor_keys"]) == 4
    round_trip = report["round_trip"]
    assert round_trip["builder_adapter_matches_trained_maximum_logit_error"] == 0.0
    assert round_trip["verified_adapter_reload_maximum_logit_error"] == 0.0
    assert round_trip["merge_maximum_logit_error"] < 1e-5
    assert round_trip["verified_merged_reload_maximum_logit_error"] == 0.0
    assert round_trip["tokenizer_chat_template_token_ids"] == [5, 7, 2, 9, 2]
    assert round_trip["verified_tokenizer_chat_template_token_ids"] == [5, 7, 2, 9, 2]
    assert round_trip["tokenizer_chat_template_exact"] is True
    artifacts = report["artifacts"]
    assert artifacts["persisted"] is False
    assert artifacts["root"] is None
    assert artifacts["pickle_weight_files"] == []
    for item in artifacts["weights"].values():
        assert item["bytes"] > 0
        assert item["sha256"].startswith("sha256:")
    verification = artifacts["strict_verification"]
    assert verification["file_count"] == 13
    assert verification["total_file_bytes"] >= sum(
        item["bytes"] for item in artifacts["weights"].values()
    )
    assert verification["manifest_bytes"] > 0
    assert verification["files"] == sorted(verification["files"])
    scope = report["scope"]
    assert scope["safetensors_payloads_parsed_before_reload"] is True
    assert scope["base_merged_config_payload_and_tensor_signature_match"] is True
    assert scope["lora_target_a_b_tensor_coverage_validated"] is True


def test_offline_peft_smoke_persists_standard_artifacts_without_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "export"
    report = _load_smoke().run_smoke(steps=2, artifact_root=root)
    assert report["artifacts"]["persisted"] is True
    assert report["artifacts"]["root"] == str(root)
    assert (root / "base" / "model.safetensors").exists()
    assert (root / "adapter" / "adapter_model.safetensors").exists()
    assert (root / "adapter" / "adapter_config.json").exists()
    assert (root / "merged" / "model.safetensors").exists()
    assert (root / "tokenizer" / "tokenizer.json").exists()
    assert (root / "tokenizer" / "chat_template.jinja").exists()
    assert (root / "about-llm-export-manifest.json").exists()
    with pytest.raises(FileExistsError):
        _load_smoke().run_smoke(steps=2, artifact_root=root)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_offline_peft_smoke_rejects_invalid_steps(steps: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _load_smoke().run_smoke(steps=steps)
