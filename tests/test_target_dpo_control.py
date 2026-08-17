from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from about_llm.finetuning import load_preference_records
from about_llm.finetuning.target_dpo_control import (
    TARGET_DPO_EVIDENCE_BOUNDARY,
    TargetDPOAdapterSpec,
    TargetDPOControlSpec,
    TargetDPOModelSpec,
    TargetDPOPairSpec,
    execute_loaded_target_dpo_control,
    load_target_dpo_control_spec,
    verify_recorded_target_dpo_report,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
CONTROL = PROJECT / "qwen2.5-0.5b-dpo.control.json"
TRAINING = PROJECT / "preference.train.example.jsonl"
READINESS = PROJECT / "preference-training-readiness.example.json"
RECORDED_REPORT = PROJECT / "qwen2.5-0.5b-dpo.recorded-report.json"
SCRIPT = PROJECT / "run_qwen_target_dpo_control.py"
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

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|> ' + message['content'] + ' ' + eos_token + ' ' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
)


def _reviewed() -> tuple[Any, TargetDPOControlSpec]:
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_CONTROL)
    spec = load_target_dpo_control_spec(CONTROL, checkpoint_spec=checkpoint)
    return checkpoint, spec


def _rehash_report(report: dict[str, Any]) -> None:
    report.pop("report_fingerprint", None)
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(report)


def _tokenizer(records: tuple[Any, ...]) -> PreTrainedTokenizerFast:
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "</s>": 2}
    tokens = {"<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"}
    for record in records:
        for message in record.prompt:
            tokens.update(message.content.split())
        tokens.update(record.candidate_a.split())
        tokens.update(record.candidate_b.split())
    for token in sorted(tokens):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="</s>",
        eos_token="</s>",
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def _expected_pairs(
    records: tuple[Any, ...], tokenizer: PreTrainedTokenizerFast
) -> tuple[TargetDPOPairSpec, ...]:
    pairs: list[TargetDPOPairSpec] = []
    for record in records:
        row = record.to_dpo_row()
        prompt_ids = tuple(
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True
            )
        )
        chosen_full = tuple(
            tokenizer.apply_chat_template(
                row["prompt"] + row["chosen"], tokenize=True
            )
        )
        rejected_full = tuple(
            tokenizer.apply_chat_template(
                row["prompt"] + row["rejected"], tokenize=True
            )
        )
        assert chosen_full[: len(prompt_ids)] == prompt_ids
        assert rejected_full[: len(prompt_ids)] == prompt_ids
        pairs.append(
            TargetDPOPairSpec(
                record_id=record.record_id,
                prompt_ids=prompt_ids,
                chosen_ids=chosen_full[len(prompt_ids) :],
                rejected_ids=rejected_full[len(prompt_ids) :],
            )
        )
    return tuple(pairs)


def _tiny_control() -> tuple[TargetDPOControlSpec, Any, Any, tuple[Any, ...]]:
    _, reviewed = _reviewed()
    records = load_preference_records(TRAINING)
    tokenizer = _tokenizer(records)
    torch.manual_seed(91)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=32,
            n_ctx=32,
            n_embd=32,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )
    )
    model_contract = TargetDPOModelSpec(
        expected_model_class="GPT2LMHeadModel",
        expected_model_type="gpt2",
        num_hidden_layers=1,
        hidden_size=32,
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        config_bos_token_id=2,
        config_eos_token_id=2,
        config_pad_token_id=1,
        tokenizer_bos_token_id=2,
        tokenizer_eos_token_id=2,
        tokenizer_pad_token_id=1,
        generation_config_bos_token_id=2,
        generation_config_eos_token_ids=(2,),
        generation_config_pad_token_id=1,
    )
    adapter = TargetDPOAdapterSpec(
        task_type="CAUSAL_LM",
        target_modules=("c_attn",),
        rank=4,
        alpha=8,
        dropout=0.0,
        bias="none",
        expected_trainable_parameters=512,
        expected_trainable_tensor_count=2,
        expected_a_tensor_count=1,
        expected_b_tensor_count=1,
        expected_b_element_count=384,
    )
    trainer = dataclasses.replace(reviewed.trainer, learning_rate=1e-2)
    spec = dataclasses.replace(
        reviewed,
        model_contract=model_contract,
        adapter=adapter,
        trainer=trainer,
        expected_pairs=_expected_pairs(records, tokenizer),
    )
    return spec, model, tokenizer, records


def test_reviewed_manifest_binds_qwen_runtime_data_and_narrow_scope() -> None:
    checkpoint, spec = _reviewed()

    assert spec.manifest_fingerprint == (
        "sha256:ebbf365523707c08d8c18c13a26551cf9af7420ce530274e9718bdc4f8d818b3"
    )
    assert spec.source_checkpoint_manifest_fingerprint == checkpoint.manifest_fingerprint
    assert spec.model_contract.expected_model_class == "Qwen2ForCausalLM"
    assert spec.model_contract.base_parameter_count == 494_032_768
    assert spec.model_contract.config_eos_token_id == 151_645
    assert spec.model_contract.config_pad_token_id is None
    assert spec.model_contract.tokenizer_pad_token_id == 151_643
    assert spec.runtime.torch_version == "2.13.0+cpu"
    assert spec.trainer.trl_version == "0.29.1"
    assert spec.training_artifact.record_ids == (
        "pref-train-alpha",
        "pref-train-beta",
    )
    assert spec.adapter.expected_trainable_parameters == 270_336
    assert "human preference validity" in TARGET_DPO_EVIDENCE_BOUNDARY


def test_loaded_tiny_gpt2_executes_real_trl_peft_step_and_gradient_callback() -> None:
    spec, model, tokenizer, records = _tiny_control()

    result = execute_loaded_target_dpo_control(
        spec, model=model, tokenizer=tokenizer, records=records
    )

    tokenization = result["tokenization"]
    execution = result["execution"]
    assert tokenization["collated_input_shape"] == [4, 17]
    assert tokenization["completion_token_counts"] == [4, 4, 4, 4]
    assert math.isclose(
        execution["initial_trainer_loss"], math.log(2), rel_tol=0, abs_tol=1e-6
    )
    assert execution["final_trainer_loss"] < execution["initial_trainer_loss"]
    assert all(value > 0 for value in execution["final_relative_margins"])
    assert execution["gradient_callback_count"] == 1
    assert execution["trainable_gradient_tensor_count"] == 2
    assert execution["finite_gradient_tensor_count"] == 2
    assert execution["frozen_base_gradient_tensor_count"] == 0
    assert execution["frozen_base_parameters_unchanged"] is True
    assert execution["reference_replay_drift_reported_not_equated_to_weight_drift"] is True
    assert execution["reference_replay_max_abs_error_after_step"] > 0
    assert execution["frozen_non_adapter_state_unchanged"] is True
    assert execution["adapter_after"]["nonzero_b_element_count"] == 384


@pytest.mark.parametrize(
    "payload",
    [
        b'{"control_version":"x","control_version":"y"}',
        b'{"value":NaN}',
        b"{\xff}",
    ],
)
def test_manifest_loader_rejects_duplicate_nonfinite_and_invalid_utf8(
    tmp_path: Path, payload: bytes
) -> None:
    checkpoint, _ = _reviewed()
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_target_dpo_control_spec(path, checkpoint_spec=checkpoint)


def test_recorded_qwen_report_verifies_target_step_and_explicit_replay_drift() -> None:
    checkpoint, spec = _reviewed()

    report = verify_recorded_target_dpo_report(
        RECORDED_REPORT,
        spec=spec,
        checkpoint_spec=checkpoint,
        checkpoint_report_path=CHECKPOINT_REPORT,
        training_path=TRAINING,
        readiness_path=READINESS,
    )

    assert report["report_fingerprint"] == (
        "sha256:3cafbade034045df61e09907185d6ae37a71e81075e96586bd9c46a3b549b7bc"
    )
    execution = report["result"]["execution"]
    assert execution["initial_trainer_loss"] == 0.6931471824645996
    assert execution["final_trainer_loss"] == 0.333351731300354
    assert execution["final_relative_margins"] == [
        8.566291809082031,
        10.01645278930664,
    ]
    assert execution["reference_replay_max_abs_error_after_step"] == (
        0.5470771789550781
    )
    assert execution["reference_replay_bitwise_equal"] is False
    assert execution["frozen_non_adapter_state_unchanged"] is True
    assert report["scope"]["alignment_quality_or_generalization_proven"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["runtime"].update({"torch_version": "cooperative-drift"}),
            "runtime drifted",
        ),
        (
            lambda report: report["scope"].update({"safety_alignment_proven": True}),
            "scope drifted",
        ),
        (
            lambda report: report["result"]["execution"].update(
                {"frozen_non_adapter_state_unchanged": False}
            ),
            "frozen identity drifted",
        ),
        (
            lambda report: report["result"]["execution"][
                "final_relative_margins"
            ].__setitem__(0, 99.0),
            "final margins drifted",
        ),
        (
            lambda report: report["result"]["execution"].update(
                {"reference_replay_max_abs_error_after_step": 0.0}
            ),
            "reference replay evidence drifted",
        ),
        (
            lambda report: report["result"]["model"]["special_token_contract"][
                "after_trainer"
            ]["model_config"].update({"pad_token_id": None}),
            "model evidence drifted",
        ),
    ],
)
def test_recorded_report_rejects_nested_drift_after_cooperative_rehash(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    checkpoint, spec = _reviewed()
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    mutation(report)
    _rehash_report(report)
    path = tmp_path / "cooperatively-rehashed.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        verify_recorded_target_dpo_report(
            path,
            spec=spec,
            checkpoint_spec=checkpoint,
            checkpoint_report_path=CHECKPOINT_REPORT,
            training_path=TRAINING,
            readiness_path=READINESS,
        )


def test_recorded_report_rejects_current_training_byte_tamper(tmp_path: Path) -> None:
    checkpoint, spec = _reviewed()
    training = tmp_path / TRAINING.name
    training.write_bytes(TRAINING.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact bytes drifted"):
        verify_recorded_target_dpo_report(
            RECORDED_REPORT,
            spec=spec,
            checkpoint_spec=checkpoint,
            checkpoint_report_path=CHECKPOINT_REPORT,
            training_path=training,
            readiness_path=READINESS,
        )


def test_cli_verifies_recorded_target_dpo_without_replaying_training() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify", str(RECORDED_REPORT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["report_fingerprint"].endswith("b549b7bc")
    assert report["scope"]["real_trl_dpo_trainer_executed"] is True
