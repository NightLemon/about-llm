from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from about_llm.finetuning.adapter_bundle import (
    SFT_ADAPTER_BUNDLE_MANIFEST,
    bind_peft_adapter_identity,
    publish_sft_adapter_bundle,
    verify_sft_adapter_bundle,
)
from about_llm.finetuning.training_runtime import write_strict_json

pytestmark = pytest.mark.contract

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
READINESS = "sha256:" + "1" * 64
MASK = "sha256:" + "2" * 64
LABELS = "sha256:" + "3" * 64
ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ENTRY = ROOT / "projects" / "single-gpu-finetuning" / "sft_adapter_bundle.py"


def _training_output(
    root: Path, *, adapter_revision: str = REVISION, zero_b: bool = False
) -> Path:
    output = root / "training-output"
    adapter = output / "adapter"
    tokenizer = output / "tokenizer"
    adapter.mkdir(parents=True)
    tokenizer.mkdir()
    write_strict_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": MODEL_ID,
            "revision": adapter_revision,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "bias": "none",
            "target_modules": ["q_proj"],
        },
    )
    save_file(
        {
            "base_model.model.layers.0.q_proj.lora_A.weight": np.ones(
                (8, 4), dtype=np.float32
            ),
            "base_model.model.layers.0.q_proj.lora_B.weight": (
                np.zeros((4, 8), dtype=np.float32)
                if zero_b
                else np.ones((4, 8), dtype=np.float32)
            ),
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "README.md").write_text("# Adapter\n", encoding="utf-8")
    write_strict_json(
        tokenizer / "tokenizer_config.json", {"chat_template": "{{ messages }}"}
    )
    write_strict_json(tokenizer / "tokenizer.json", {"version": "1.0"})

    write_strict_json(output / "sft-data-audit.json", {"gate_passed": True})
    write_strict_json(
        output / "sft-training-readiness.json",
        {"gate_passed": True, "manifest_fingerprint": READINESS},
    )
    write_strict_json(
        output / "sft-template-mask-audit.json",
        {"gate_passed": True, "manifest_fingerprint": MASK},
    )
    write_strict_json(
        output / "sft-final-label-audit.json",
        {"gate_passed": True, "labels_fingerprint": LABELS},
    )
    write_strict_json(
        output / "sft-training-run.json",
        {
            "report_version": "about-llm.sft-training-run.v1",
            "status": "completed",
            "model": {"model_id": MODEL_ID, "revision": REVISION},
            "data": {
                "readiness_manifest_fingerprint": READINESS,
                "assistant_mask_manifest_fingerprint": MASK,
                "final_labels_fingerprint": LABELS,
            },
            "training": {
                "target_modules": ["q_proj"],
                "rank": 8,
                "alpha": 16,
            },
            "outcome": {"optimizer_step_count": 1},
        },
    )
    return output


def test_publish_and_verify_binds_adapter_tokenizer_and_training_evidence(
    tmp_path: Path,
) -> None:
    output = _training_output(tmp_path)

    published = publish_sft_adapter_bundle(output)
    verified = verify_sft_adapter_bundle(
        output / "adapter-bundle",
        expected_model_id=MODEL_ID,
        expected_revision=REVISION,
    )

    assert published == verified
    assert verified.identity.model_id == MODEL_ID
    assert verified.identity.revision == REVISION
    assert verified.contract.target_modules == ("q_proj",)
    assert verified.optimizer_step_count == 1
    assert verified.adapter_tensor_count == 2
    assert verified.file_count == 10


def test_verify_rejects_expected_base_revision_drift(tmp_path: Path) -> None:
    output = _training_output(tmp_path)
    publish_sft_adapter_bundle(output)

    with pytest.raises(ValueError, match="expected base revision"):
        verify_sft_adapter_bundle(
            output / "adapter-bundle",
            expected_model_id=MODEL_ID,
            expected_revision="another-revision",
        )


@pytest.mark.smoke
def test_verify_cli_reports_the_bound_base_identity(tmp_path: Path) -> None:
    output = _training_output(tmp_path)
    publish_sft_adapter_bundle(output)

    completed = subprocess.run(
        [
            sys.executable,
            str(BUNDLE_ENTRY),
            "verify",
            "--bundle",
            str(output / "adapter-bundle"),
            "--expected-model-id",
            MODEL_ID,
            "--expected-revision",
            REVISION,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    report = json.loads(completed.stdout)
    assert report["identity"] == {"model_id": MODEL_ID, "revision": REVISION}
    assert report["optimizer_step_count"] == 1


def test_publish_rejects_adapter_config_revision_drift(tmp_path: Path) -> None:
    output = _training_output(tmp_path, adapter_revision="moving-main")

    with pytest.raises(ValueError, match="revision differs"):
        publish_sft_adapter_bundle(output)

    assert not (output / "adapter-bundle").exists()


def test_publish_rejects_an_adapter_with_no_learned_b_update(tmp_path: Path) -> None:
    output = _training_output(tmp_path, zero_b=True)

    with pytest.raises(ValueError, match="LoRA B tensors are all zero"):
        publish_sft_adapter_bundle(output)

    assert not (output / "adapter-bundle").exists()


def test_verify_rejects_file_tampering_and_duplicate_manifest_keys(
    tmp_path: Path,
) -> None:
    output = _training_output(tmp_path)
    publish_sft_adapter_bundle(output)
    bundle = output / "adapter-bundle"
    weights = bundle / "adapter" / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="file set, size, or digest"):
        verify_sft_adapter_bundle(bundle)

    output = _training_output(tmp_path / "duplicate")
    publish_sft_adapter_bundle(output)
    manifest = output / "adapter-bundle" / SFT_ADAPTER_BUNDLE_MANIFEST
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace("{", '{"schema_version":"duplicate",', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_sft_adapter_bundle(output / "adapter-bundle")


def test_verify_rejects_an_unlisted_directory(tmp_path: Path) -> None:
    output = _training_output(tmp_path)
    publish_sft_adapter_bundle(output)
    (output / "adapter-bundle" / "unlisted").mkdir()

    with pytest.raises(ValueError, match="unsupported directory"):
        verify_sft_adapter_bundle(output / "adapter-bundle")


def test_bind_peft_adapter_identity_updates_the_saved_default_config() -> None:
    configuration = SimpleNamespace(
        base_model_name_or_path="moving-main", revision=None
    )
    model = SimpleNamespace(peft_config={"default": configuration})

    bind_peft_adapter_identity(model, model_id=MODEL_ID, revision=REVISION)

    assert configuration.base_model_name_or_path == MODEL_ID
    assert configuration.revision == REVISION


@pytest.mark.integration
@pytest.mark.slow
def test_real_peft_save_produces_a_publishable_bundle(tmp_path: Path) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    output = _training_output(tmp_path)
    shutil.rmtree(output / "adapter")
    base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    local_base = tmp_path / "local-base"
    base.save_pretrained(local_base, safe_serialization=True)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    input_ids = torch.tensor([[1, 3, 5, 7, 2]])
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    local_model_id = str(local_base.resolve())
    local_revision = "fixture-revision-1"
    bind_peft_adapter_identity(
        model, model_id=local_model_id, revision=local_revision
    )
    model.save_pretrained(
        output / "adapter",
        safe_serialization=True,
        save_embedding_layers=False,
    )
    shutil.rmtree(output / "tokenizer")
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
        **{f"tok{index}": index for index in range(6, 32)},
    }
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        chat_template=(
            "{% for message in messages %}"
            "{{ '<|' + message['role'] + '|> ' + message['content'] + ' ' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
        ),
    )
    tokenizer.save_pretrained(output / "tokenizer")

    report_path = output / "sft-training-run.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["model"] = {
        "model_id": local_model_id,
        "revision": local_revision,
    }
    report["training"]["target_modules"] = ["c_attn"]
    write_strict_json(report_path, report)

    verification = publish_sft_adapter_bundle(output)

    assert verification.contract.target_modules == ("c_attn",)
    assert verification.adapter_tensor_count == 2

    reload_report = tmp_path / "fresh-reload.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(BUNDLE_ENTRY),
            "reload",
            "--bundle",
            str(output / "adapter-bundle"),
            "--expected-model-id",
            local_model_id,
            "--expected-revision",
            local_revision,
            "--report",
            str(reload_report),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--local-files-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    reload_result = json.loads(completed.stdout)
    assert reload_result["status"] == "completed"
    assert reload_result["execution"]["adapter_loaded_with_peft"] is True
    assert reload_result["execution"]["maximum_last_logit_delta"] > 0
    assert json.loads(reload_report.read_text(encoding="utf-8")) == reload_result
