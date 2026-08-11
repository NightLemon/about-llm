from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from about_llm.finetuning import load_preference_records

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
SCRIPT = PROJECT / "train_reward_model.py"
SMOKE = PROJECT / "smoke_transformer_reward_model.py"


def _load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_transformer_reward_model", SMOKE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Transformer reward-model smoke script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    tmp_path: Path,
    *,
    model_id: str,
    readiness: Path,
    mode: str,
    max_length: int = 48,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-id",
            model_id,
            "--revision",
            "local-fixture",
            "--train-jsonl",
            str(PROJECT / "preference.train.example.jsonl"),
            "--readiness-json",
            str(readiness),
            "--output-dir",
            str(tmp_path / "run"),
            "--max-length",
            str(max_length),
            mode,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


def test_reward_model_data_preflight_needs_no_model_or_held_out_file(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        model_id="must-not-download",
        readiness=PROJECT / "preference-training-readiness.example.json",
        mode="--data-preflight-only",
    )

    assert result.returncode == 0, result.stderr
    output = tmp_path / "run"
    contract = json.loads(
        (output / "reward-model-data-contract.json").read_text(encoding="utf-8")
    )
    assert contract["binary_train_pair_count"] == 2
    assert contract["scalar_reward_count_per_sequence"] == 1
    assert contract["held_out_dataset_passed_to_trainer"] is False
    assert not (output / "preference-tokenization-audit.json").exists()
    assert not (output / "reward-model-run-contract.json").exists()


def test_reward_model_entry_rejects_tampered_readiness_before_output(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (PROJECT / "preference-training-readiness.example.json").read_text(
            encoding="utf-8"
        )
    )
    payload["binary_train_record_count"] = 99
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(
        tmp_path,
        model_id="must-not-download",
        readiness=tampered,
        mode="--data-preflight-only",
    )

    assert result.returncode != 0
    assert "manifest_fingerprint mismatch" in result.stderr
    assert not (tmp_path / "run").exists()


def test_reward_model_target_tokenizer_preflight_executes_without_model(
    tmp_path: Path,
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    records = load_preference_records(PROJECT / "preference.train.example.jsonl")
    _load_smoke()._tokenizer(records).save_pretrained(tokenizer_dir)

    result = _run(
        tmp_path,
        model_id=str(tokenizer_dir),
        readiness=PROJECT / "preference-training-readiness.example.json",
        mode="--tokenization-preflight-only",
    )

    assert result.returncode == 0, result.stderr
    output = tmp_path / "run"
    tokenization = json.loads(
        (output / "preference-tokenization-audit.json").read_text(encoding="utf-8")
    )
    assert tokenization["record_count"] == 2
    assert tokenization["scope"]["target_tokenizer_executed"] is True
    assert not (output / "reward-model-run-contract.json").exists()


def test_reward_model_target_tokenizer_preflight_rejects_overlength_pairs(
    tmp_path: Path,
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    records = load_preference_records(PROJECT / "preference.train.example.jsonl")
    _load_smoke()._tokenizer(records).save_pretrained(tokenizer_dir)

    result = _run(
        tmp_path,
        model_id=str(tokenizer_dir),
        readiness=PROJECT / "preference-training-readiness.example.json",
        mode="--tokenization-preflight-only",
        max_length=2,
    )

    assert result.returncode != 0
    assert "exceeds max_length" in result.stderr
    assert "trainer truncation or filtering" in result.stderr
    assert not (tmp_path / "run" / "preference-tokenization-audit.json").exists()


@pytest.mark.slow
def test_reward_model_entry_executes_real_trl_lora_optimizer_and_save(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("trl")
    model_dir = tmp_path / "tiny-model"
    records = load_preference_records(PROJECT / "preference.train.example.jsonl")
    tokenizer = _load_smoke()._tokenizer(records)
    model = transformers.GPT2ForSequenceClassification(
        transformers.GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=24,
            n_ctx=24,
            n_embd=8,
            n_layer=1,
            n_head=1,
            num_labels=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )
    )
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    output = tmp_path / "full-run"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-id",
            str(model_dir),
            "--revision",
            "local-fixture",
            "--train-jsonl",
            str(PROJECT / "preference.train.example.jsonl"),
            "--readiness-json",
            str(PROJECT / "preference-training-readiness.example.json"),
            "--output-dir",
            str(output),
            "--target-modules",
            "c_attn",
            "--modules-to-save",
            "score",
            "--rank",
            "2",
            "--alpha",
            "4",
            "--max-length",
            "24",
            "--batch-size",
            "2",
            "--gradient-accumulation",
            "1",
            "--learning-rate",
            "0.01",
            "--max-steps",
            "1",
            "--no-gradient-checkpointing",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    outcome = json.loads(
        (output / "reward-model-train-result.json").read_text(encoding="utf-8")
    )
    assert outcome["global_step"] == 1
    assert outcome["prepared_pair_count"] == 2
    assert 0 < outcome["trainable_parameter_count"] < outcome["total_parameter_count"]
    assert math.isfinite(outcome["metrics"]["train_loss"])
    assert outcome["scope"] == {
        "trl_reward_trainer_executed": True,
        "held_out_dataset_passed_to_trainer": False,
        "target_model_quality_proved": False,
        "cuda_or_qlora_executed": False,
    }
    assert (output / "adapter_config.json").exists()
    assert (output / "adapter_model.safetensors").exists()
    from safetensors.torch import load_file

    adapter = load_file(output / "adapter_model.safetensors")
    assert any(
        "lora_B" in name and torch.count_nonzero(tensor).item() > 0
        for name, tensor in adapter.items()
    )
