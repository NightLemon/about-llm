from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import about_llm.finetuning.target_lora_control as target_module  # noqa: E402
from about_llm.finetuning.target_lora_control import (  # noqa: E402
    TARGET_LORA_ARTIFACT_MANIFEST,
    TARGET_LORA_EVIDENCE_BOUNDARY,
    TargetLoRAControlSpec,
    execute_loaded_target_lora_training,
    load_recorded_target_lora_report,
    load_target_lora_control_spec,
    run_target_lora_control,
    verify_recorded_target_lora_report,
    verify_target_lora_adapter_artifact,
)
from about_llm.integrations.transformers_checkpoint_control import (  # noqa: E402
    VerifiedCheckpointSnapshot,
    load_checkpoint_control_spec,
)
from about_llm.llmops import canonical_json_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
CONTROL = PROJECT / "qwen2.5-0.5b-lora.control.json"
RECORDED_REPORT = PROJECT / "qwen2.5-0.5b-lora.recorded-report.json"
ARTIFACT = PROJECT / "target-adapters" / "qwen2.5-0.5b-instruct-step1"
CHECKPOINT_CONTROL = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
CHECKPOINT_REPORT = CHECKPOINT_CONTROL.with_name(
    "qwen2.5-0.5b-instruct.recorded-report.json"
)
SCRIPT = PROJECT / "run_qwen_target_lora_control.py"


def _reviewed() -> tuple[Any, TargetLoRAControlSpec]:
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_CONTROL)
    spec = load_target_lora_control_spec(CONTROL, checkpoint_spec=checkpoint)
    return checkpoint, spec


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_artifact_manifest(root: Path) -> None:
    path = root / TARGET_LORA_ARTIFACT_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for descriptor in manifest["files"]:
        payload = root / descriptor["path"]
        descriptor["bytes"] = payload.stat().st_size
        descriptor["sha256"] = _sha256_file(payload)
    manifest.pop("manifest_fingerprint")
    manifest["manifest_fingerprint"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    path.write_bytes(canonical_json_bytes(manifest))


def _rehash_report(report: dict[str, Any]) -> None:
    report.pop("report_fingerprint", None)
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()


class _TinyTokenizer:
    chat_template = "authored-target-lora-test-template"

    def __len__(self) -> int:
        return 64

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> Any:
        assert tokenize is True
        assert return_tensors == "pt"
        if add_generation_prompt:
            assert len(messages) == 1
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        assert len(messages) == 2
        return torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)


def _tiny_spec(spec: TargetLoRAControlSpec, model: Any) -> TargetLoRAControlSpec:
    model_contract = dataclasses.replace(
        spec.model_contract,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    return dataclasses.replace(spec, model_contract=model_contract)


def _save_tiny_checkpoint(root: Path) -> tuple[Any, TargetLoRAControlSpec]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    _, reviewed = _reviewed()
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(73)
    model = transformers.Qwen2ForCausalLM(config)
    spec = _tiny_spec(reviewed, model)
    model.save_pretrained(root, safe_serialization=True)
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "<user>": 4,
        "<assistant>": 5,
        "5": 6,
        **{f"tok{index}": index for index in range(7, 64)},
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=64,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<user> {{ message['content'] }} {{ eos_token }} "
        "{% else %}"
        "<assistant> {{ message['content'] }} {{ eos_token }} "
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    tokenizer.save_pretrained(root)
    return model, spec


def test_reviewed_manifest_binds_real_qwen_and_narrow_scope() -> None:
    checkpoint, spec = _reviewed()

    assert spec.manifest_fingerprint == (
        "sha256:801b95fe6f35fdff9b0bef5db47d028e98ad0335e78055d5b33eeb48c3034885"
    )
    assert spec.source_checkpoint_manifest_fingerprint == checkpoint.manifest_fingerprint
    assert spec.source_checkpoint_report_fingerprint == (
        "sha256:56528a3e02ed6ef9d205dcf83ba456658d639d7681ab6a1ad9eb110211edba62"
    )
    assert spec.model_contract.base_parameter_count == 494_032_768
    assert spec.adapter.target_modules == ("q_proj", "v_proj")
    assert spec.optimizer.steps == 1
    assert spec.device == "cpu"
    assert TARGET_LORA_EVIDENCE_BOUNDARY.endswith("prove production safety.")


def test_loaded_tiny_qwen_executes_real_peft_backward_and_verified_export(
    tmp_path: Path,
) -> None:
    _, reviewed = _reviewed()
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    torch.manual_seed(71)
    model = transformers.Qwen2ForCausalLM(config)
    spec = _tiny_spec(reviewed, model)
    artifact = tmp_path / "adapter"

    result = execute_loaded_target_lora_training(
        spec,
        model=model,
        tokenizer=_TinyTokenizer(),
        artifact_directory=artifact,
    )

    assert result.sample["prompt_token_count"] == 4
    assert result.sample["supervised_token_count"] == 2
    assert result.model["adapter_parameter_count"] == 896
    assert result.execution["backward_executed"] is True
    assert result.execution["frozen_base_parameters_unchanged"] is True
    assert result.execution["adapter_nonzero_b_tensor_count_after_step"] == 4
    assert result.execution["post_step_vs_base_max_abs_error"] > 0
    assert result.artifact.tensor_count == 8
    assert result.artifact.total_tensor_numel == 896
    assert verify_target_lora_adapter_artifact(artifact, spec=spec) == result.artifact


def test_full_runner_loads_trains_publishes_reloads_and_self_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _ = _reviewed()
    checkpoint_directory = tmp_path / "checkpoint"
    checkpoint_directory.mkdir()
    model, spec = _save_tiny_checkpoint(checkpoint_directory)
    del model
    verification_calls: list[Path] = []

    def fake_download(*args: Any, **kwargs: Any) -> Path:
        assert args[0] == checkpoint
        assert kwargs == {"local_files_only": True}
        return checkpoint_directory

    def fake_verify(*args: Any, **kwargs: Any) -> VerifiedCheckpointSnapshot:
        assert args == (checkpoint, checkpoint_directory)
        assert kwargs == {}
        verification_calls.append(checkpoint_directory)
        return VerifiedCheckpointSnapshot(
            directory=checkpoint_directory.resolve(), files=tuple()
        )

    monkeypatch.setattr(target_module, "download_checkpoint_snapshot", fake_download)
    monkeypatch.setattr(target_module, "verify_checkpoint_snapshot", fake_verify)
    artifact = tmp_path / "adapter"

    report = run_target_lora_control(
        spec,
        checkpoint_spec=checkpoint,
        checkpoint_report_path=CHECKPOINT_REPORT,
        artifact_directory=artifact,
        local_files_only=True,
    )

    assert len(verification_calls) == 2
    assert report["model"]["base_parameter_count"] == spec.model_contract.base_parameter_count
    assert report["execution"]["optimizer_step_count"] == 1
    assert report["round_trip"]["maximum_logit_error"] == 0.0
    assert report["round_trip"]["reloaded_loss"] == report["execution"][
        "post_step_loss"
    ]
    assert report["scope"]["target_checkpoint_backward_executed"] is True
    assert verify_target_lora_adapter_artifact(artifact, spec=spec).tensor_count == 8


def test_recorded_qwen_report_preserves_success_and_non_quality_result() -> None:
    checkpoint, spec = _reviewed()
    report = load_recorded_target_lora_report(
        RECORDED_REPORT,
        spec=spec,
        checkpoint_spec=checkpoint,
        checkpoint_report_path=CHECKPOINT_REPORT,
        artifact_directory=ARTIFACT,
    )

    assert report["report_fingerprint"] == (
        "sha256:8a3897b10dbc2f55bb5ad3a8851fe659670e6951c19e58ae7fd269f9fb026230"
    )
    model = report["model"]
    execution = report["execution"]
    artifact = report["adapter_artifact"]
    assert model["adapter_parameter_count"] == 270_336
    assert execution["optimizer_step_count"] == 1
    assert execution["adapter_nonzero_b_tensor_count_after_step"] == 48
    assert execution["frozen_base_parameter_fingerprint_before"] == execution[
        "frozen_base_parameter_fingerprint_after"
    ]
    assert execution["post_step_loss"] > execution["initial_loss"]
    assert artifact["total_tensor_numel"] == 270_336
    assert artifact["files"][2]["bytes"] == 1_093_728
    assert report["round_trip"]["maximum_logit_error"] == 0.0
    assert report["scope"]["model_quality_or_convergence_proven"] is False
    assert report["scope"]["qlora_or_quantized_base_executed"] is False


def test_adapter_verifier_rejects_tamper_and_cooperative_config_drift(
    tmp_path: Path,
) -> None:
    _, spec = _reviewed()
    tampered = tmp_path / "tampered"
    shutil.copytree(ARTIFACT, tampered)
    with (tampered / "README.md").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="file set, size, or digest"):
        verify_target_lora_adapter_artifact(tampered, spec=spec)

    drifted = tmp_path / "drifted"
    shutil.copytree(ARTIFACT, drifted)
    config_path = drifted / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_modules"] = ["q_proj"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _rewrite_artifact_manifest(drifted)
    with pytest.raises(ValueError, match="target_modules drift"):
        verify_target_lora_adapter_artifact(drifted, spec=spec)

    rank_drifted = tmp_path / "rank-drifted"
    shutil.copytree(ARTIFACT, rank_drifted)
    rank_config_path = rank_drifted / "adapter_config.json"
    rank_config = json.loads(rank_config_path.read_text(encoding="utf-8"))
    rank_config["rank_pattern"] = {"q_proj": 8}
    rank_config_path.write_text(json.dumps(rank_config), encoding="utf-8")
    _rewrite_artifact_manifest(rank_drifted)
    with pytest.raises(ValueError, match="rank_pattern drift"):
        verify_target_lora_adapter_artifact(rank_drifted, spec=spec)


def test_adapter_verifier_rejects_missing_b_tensor_after_cooperative_rehash(
    tmp_path: Path,
) -> None:
    from safetensors.torch import load_file, save_file

    _, spec = _reviewed()
    drifted = tmp_path / "missing-b"
    shutil.copytree(ARTIFACT, drifted)
    weights = drifted / "adapter_model.safetensors"
    state = load_file(weights, device="cpu")
    key = next(name for name in state if ".lora_B." in name)
    del state[key]
    save_file(state, weights, metadata={"format": "pt"})
    _rewrite_artifact_manifest(drifted)
    with pytest.raises(ValueError, match="cover every reviewed A/B target"):
        verify_target_lora_adapter_artifact(drifted, spec=spec)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["scope"].update({"cuda_executed": True}), "scope drift"),
        (
            lambda report: report["execution"].update(
                {"frozen_base_parameter_fingerprint_after": "sha256:" + "0" * 64}
            ),
            "frozen base fingerprint drift",
        ),
        (
            lambda report: report["sample"].update({"prompt_token_count": 40}),
            "token boundary accounting mismatch",
        ),
        (
            lambda report: report["execution"].update(
                {"adapter_nonzero_b_element_count_after_step": 98_303}
            ),
            "execution/artifact tensor evidence drift",
        ),
    ],
)
def test_recorded_report_rejects_nested_drift_after_cooperative_rehash(
    mutation: Any, message: str
) -> None:
    checkpoint, spec = _reviewed()
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    mutation(report)
    _rehash_report(report)
    with pytest.raises(ValueError, match=message):
        verify_recorded_target_lora_report(
            spec,
            checkpoint_spec=checkpoint,
            checkpoint_report_path=CHECKPOINT_REPORT,
            report=report,
            artifact_directory=ARTIFACT,
        )


def test_recorded_loader_rejects_duplicate_nonfinite_and_invalid_utf8(
    tmp_path: Path,
) -> None:
    checkpoint, spec = _reviewed()
    invalid_payloads = (
        b'{"report_version":"x","report_version":"y"}',
        b'{"value":NaN}',
        b"{\xff}",
    )
    for index, payload in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError):
            load_recorded_target_lora_report(
                path,
                spec=spec,
                checkpoint_spec=checkpoint,
                checkpoint_report_path=CHECKPOINT_REPORT,
                artifact_directory=ARTIFACT,
            )


def test_cli_verifies_recorded_target_lora_without_replaying_training() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--verify",
            str(RECORDED_REPORT),
            "--artifact-directory",
            str(ARTIFACT),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["report_fingerprint"].endswith("026230")
    assert report["scope"]["target_checkpoint_backward_executed"] is True
