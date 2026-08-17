from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from about_llm.integrations.transformers_checkpoint_control import (
    TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY,
    CheckpointControlSpec,
    CheckpointFileEvidence,
    CheckpointMessage,
    execute_loaded_checkpoint_control,
    load_checkpoint_control_spec,
    verify_checkpoint_snapshot,
    verify_recorded_checkpoint_report,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
SCRIPT = ROOT / "projects" / "transformers-basics" / "run_target_checkpoint.py"
RECORDED_REPORT = MANIFEST.with_name(
    "qwen2.5-0.5b-instruct.recorded-report.json"
)


def _tiny_spec(*, expected_model_class: str = "GPT2LMHeadModel") -> CheckpointControlSpec:
    return CheckpointControlSpec(
        checked_at="2026-08-13",
        model_id="fixture/tiny-gpt2",
        revision="a" * 40,
        source_base_url=(
            "https://huggingface.co/fixture/tiny-gpt2/resolve/" + "a" * 40 + "/"
        ),
        expected_model_class=expected_model_class,
        expected_model_type="gpt2",
        dtype="float32",
        device="cpu",
        attention_implementation="eager",
        max_new_tokens=2,
        messages=(
            CheckpointMessage(role="system", content="fixture system"),
            CheckpointMessage(role="user", content="fixture user"),
        ),
        files=(),
        manifest_fingerprint="sha256:" + "a" * 64,
    )


class TinyTokenizer:
    chat_template = "authored fixture template"
    pad_token_id = 0
    eos_token_id = 4

    def __len__(self) -> int:
        return 32

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> Any:
        import torch

        assert messages == [
            {"role": "system", "content": "fixture system"},
            {"role": "user", "content": "fixture user"},
        ]
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(str(token_id) for token_id in token_ids)


def _tiny_model() -> Any:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(17)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=4,
            pad_token_id=0,
        )
    )


def test_reviewed_qwen_manifest_binds_weight_and_runtime_contract() -> None:
    spec = load_checkpoint_control_spec(MANIFEST)

    assert spec.manifest_fingerprint == (
        "sha256:ddf41f2cff963bc2a8fc186c28369abba8a920b850152fc815e2b17c7d037876"
    )
    assert spec.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert spec.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert (spec.dtype, spec.device, spec.attention_implementation) == (
        "float32",
        "cpu",
        "eager",
    )
    files = {item.filename: item for item in spec.files}
    assert files["model.safetensors"] == CheckpointFileEvidence(
        filename="model.safetensors",
        size_bytes=988_097_824,
        sha256=(
            "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
        ),
    )
    assert sum(item.size_bytes for item in spec.files) == 999_586_347


def test_snapshot_verifier_hashes_open_file_and_returns_no_local_path(
    tmp_path: Path,
) -> None:
    payload = b"reviewed checkpoint bytes\n"
    path = tmp_path / "weights.safetensors"
    path.write_bytes(payload)
    spec = _tiny_spec()
    spec = CheckpointControlSpec(
        **{
            **spec.__dict__,
            "files": (
                CheckpointFileEvidence(
                    filename=path.name,
                    size_bytes=len(payload),
                    sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                ),
            ),
        }
    )

    verified = verify_checkpoint_snapshot(spec, tmp_path)

    assert verified.directory == tmp_path.resolve()
    assert verified.files == (
        {
            "filename": "weights.safetensors",
            "size_bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "verified": True,
        },
    )
    assert str(tmp_path) not in json.dumps(verified.files)


@pytest.mark.parametrize("mutation", ["size", "hash"])
def test_snapshot_verifier_rejects_tampered_file(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"actual")
    expected_size = 7 if mutation == "size" else 6
    expected_hash = (
        "sha256:" + hashlib.sha256(b"actual").hexdigest()
        if mutation == "size"
        else "sha256:" + "0" * 64
    )
    spec = _tiny_spec()
    spec = CheckpointControlSpec(
        **{
            **spec.__dict__,
            "files": (
                CheckpointFileEvidence(
                    filename=path.name,
                    size_bytes=expected_size,
                    sha256=expected_hash,
                ),
            ),
        }
    )

    with pytest.raises(ValueError, match=r"size mismatch|SHA-256 mismatch"):
        verify_checkpoint_snapshot(spec, tmp_path)


def test_tiny_real_transformers_control_executes_prefill_cache_and_generate() -> None:
    report = execute_loaded_checkpoint_control(
        _tiny_spec(), model=_tiny_model(), tokenizer=TinyTokenizer()
    )

    assert report["prompt_token_count"] == 3
    assert report["prefill_logits_shape"] == [1, 3, 32]
    assert len(report["generated_token_ids"]) == 2
    assert report["manual_prefill_argmax_matches_generate"] is True
    assert report["manual_cached_argmax_matches_generate"] is True
    assert report["cached_full_argmax_match"] is True
    assert report["cached_full_max_abs_error"] <= report["cached_full_tolerance"]
    assert report["past_key_values_executed"] is True
    assert report["prefill_last_logits_sha256"].startswith("sha256:")


def test_loaded_model_identity_mismatch_fails_before_execution() -> None:
    with pytest.raises(ValueError, match="loaded model class mismatch"):
        execute_loaded_checkpoint_control(
            _tiny_spec(expected_model_class="NotGPT2"),
            model=_tiny_model(),
            tokenizer=TinyTokenizer(),
        )


def test_recorded_qwen_run_is_self_authenticating_and_scope_limited() -> None:
    spec = load_checkpoint_control_spec(MANIFEST)
    report = verify_recorded_checkpoint_report(
        RECORDED_REPORT,
        expected_manifest_fingerprint=spec.manifest_fingerprint,
    )

    assert report["report_fingerprint"] == (
        "sha256:56528a3e02ed6ef9d205dcf83ba456658d639d7681ab6a1ad9eb110211edba62"
    )
    assert report["artifacts"]["selected_total_bytes"] == 999_586_347
    assert report["model"]["parameter_report"]["total_parameters"] == 494_032_768
    assert report["model"]["parameter_report"]["trainable_parameters"] == 0
    assert report["execution"]["parameters_frozen_for_control"] is True
    assert report["execution"]["generated_token_ids"] == [17, 151645]
    assert report["execution"]["cached_full_max_abs_error"] == pytest.approx(
        3.719329833984375e-05
    )
    assert report["scope"]["model_quality_proven"] is False
    assert report["scope"]["verification_to_loader_reopen_toctou_eliminated"] is False


def test_recorded_qwen_run_tampering_is_rejected(tmp_path: Path) -> None:
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    report["execution"]["decoded_continuation"] = "tampered"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="report fingerprint mismatch"):
        verify_recorded_checkpoint_report(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["source"].update({"unreviewed": True}),
            r"report.source: field set mismatch",
        ),
        (
            lambda report: report["runtime"].update({"device": "cuda"}),
            r"not the reviewed cpu/float32/eager runtime",
        ),
        (
            lambda report: report["model"]["parameter_report"].update(
                {"trainable_parameters": 1, "trainable_fraction": 1e-9}
            ),
            r"parameters were not fully frozen",
        ),
        (
            lambda report: report["model"]["parameter_report"].update(
                {"parameter_storage_bytes": 1_976_131_068}
            ),
            r"FP32 parameter storage is inconsistent",
        ),
        (
            lambda report: report["execution"].update(
                {"prefill_logits_shape": [1, 30, 151936]}
            ),
            r"prefill_logits_shape is inconsistent",
        ),
        (
            lambda report: report["execution"].update(
                {"cached_full_max_abs_error": 0.5, "cached_full_tolerance": 1.0}
            ),
            r"tolerance is not the reviewed value",
        ),
        (
            lambda report: report["scope"].pop(
                "verification_to_loader_reopen_toctou_eliminated"
            ),
            r"report.scope: field set mismatch",
        ),
    ],
)
def test_recorded_report_closed_schema_rejects_cooperatively_rehashed_drift(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    mutation(report)
    projection = dict(report)
    del projection["report_fingerprint"]
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(projection)
    path = tmp_path / "cooperatively-rehashed-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_recorded_checkpoint_report(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unexpected": True}), "field set mismatch"),
        (
            lambda value: value.update(
                {"source_base_url": "https://example.com/wrong/"}
            ),
            "does not match model_id/revision",
        ),
        (
            lambda value: value["files"][0].update({"filename": "../config.json"}),
            "simple file name",
        ),
        (lambda value: value.update({"max_new_tokens": 3}), "must be exactly 2"),
        (
            lambda value: value.update({"evidence_boundary": "overclaim"}),
            "evidence_boundary drift",
        ),
    ],
)
def test_manifest_drift_fails_closed(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutation(value)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_checkpoint_control_spec(path)


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"control_version":"a","control_version":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_checkpoint_control_spec(path)


def test_target_checkpoint_cli_help_does_not_download_weights() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert "--local-files-only" in completed.stdout
    assert completed.stderr == ""
    assert TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY.startswith(
        "This control hashes selected files from an immutable-revision repository snapshot"
    )
