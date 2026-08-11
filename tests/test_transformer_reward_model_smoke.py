from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "smoke_transformer_reward_model.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_transformer_reward_model", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Transformer reward-model smoke script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_offline_transformer_reward_model_executes_real_text_pair_training(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    readiness_path = tmp_path / "readiness.json"
    train_path.write_bytes(
        (SCRIPT.parent / "preference.train.example.jsonl").read_bytes()
    )
    readiness_path.write_bytes(
        (SCRIPT.parent / "preference-training-readiness.example.json").read_bytes()
    )
    report = _load_script().run_smoke(
        steps=4,
        train_path=train_path,
        readiness_path=readiness_path,
    )

    assert not (tmp_path / "preference.example.jsonl").exists()
    assert report["model_class"] == "GPT2ForSequenceClassification"
    assert report["pair_count"] == 2
    assert report["input_sequence_count"] == 4
    assert report["non_padding_token_count"] > 0
    assert report["initial_metrics"]["mean_loss"] == pytest.approx(math.log(2))
    assert report["initial_metrics"]["tie_count"] == 2
    assert report["final_metrics"]["mean_loss"] < report["initial_metrics"]["mean_loss"]
    assert report["final_metrics"]["strict_pair_accuracy"] == 1
    assert report["authored_counterfactual_metrics"]["strict_pair_accuracy"] == 0
    assert report["reward_head_parameters_changed"] is True
    assert report["transformer_backbone_parameters_changed"] is True
    assert str(report["readiness_manifest_fingerprint"]).startswith("sha256:")
    assert str(report["tokenization_manifest_fingerprint"]).startswith("sha256:")
    assert str(report["combined_manifest_fingerprint_from_readiness"]).startswith(
        "sha256:"
    )
    assert report["scope"] == {
        "device": "cpu",
        "train_only_tokenizer_vocabulary": True,
        "training_process_without_held_out_access": True,
        "actual_text_tokenization_executed": True,
        "transformer_forward_and_optimizer_executed": True,
        "pairwise_bradley_terry_loss_executed": True,
        "full_prompt_and_response_scored": True,
        "authored_preferences_not_human_labels": True,
        "target_reward_model_quality_proved": False,
        "broad_counterfactual_robustness_proved": False,
        "reward_hacking_or_policy_optimization_evaluated": False,
        "cuda_executed": False,
    }


def test_transformer_reward_model_rejects_missing_tampered_and_stale_readiness(
    tmp_path: Path,
) -> None:
    module = _load_script()
    source_train = SCRIPT.parent / "preference.train.example.jsonl"
    source_readiness = SCRIPT.parent / "preference-training-readiness.example.json"
    train_path = tmp_path / "train.jsonl"
    train_path.write_bytes(source_train.read_bytes())

    with pytest.raises(FileNotFoundError):
        module.run_smoke(
            steps=1,
            train_path=train_path,
            readiness_path=tmp_path / "missing.json",
        )

    payload = json.loads(source_readiness.read_text(encoding="utf-8"))
    payload["binary_train_record_count"] = 3
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_fingerprint mismatch"):
        module.run_smoke(
            steps=1,
            train_path=train_path,
            readiness_path=tampered_path,
        )

    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_bytes(source_readiness.read_bytes())
    stale_train = tmp_path / "stale-train.jsonl"
    lines = source_train.read_text(encoding="utf-8").splitlines()
    stale_train.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordered fingerprint differs"):
        module.run_smoke(
            steps=1,
            train_path=stale_train,
            readiness_path=readiness_path,
        )
