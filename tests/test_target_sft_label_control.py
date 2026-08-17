from __future__ import annotations

import dataclasses
import hashlib
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

from about_llm.finetuning import load_sft_records
from about_llm.finetuning.target_sft_label_control import (
    TARGET_SFT_LABEL_CONTROL_VERSION,
    TARGET_SFT_LABEL_EVIDENCE_BOUNDARY,
    TARGET_SFT_LABEL_REPORT_VERSION,
    TargetSFTLabelControlSpec,
    TargetSFTSampleSpec,
    execute_loaded_target_sft_label_control,
    load_target_sft_label_control_spec,
    verify_recorded_target_sft_label_report,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.llmops import artifact_fingerprint
from scripts.smoke_wheel import EXPECTED_TARGET_SFT_LABEL_VERSION_LINES

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
CONTROL = PROJECT / "qwen2.5-0.5b-sft-label.control.json"
TRAINING = PROJECT / "tool-sft.train.jsonl"
READINESS = PROJECT / "tool-sft-training-readiness.json"
TINY_TRAINING = PROJECT / "train.example.jsonl"
TEMPLATE = PROJECT / "qwen2.5-generation-aware-sft.jinja"
RECORDED_REPORT = PROJECT / "qwen2.5-0.5b-sft-label.recorded-report.json"
SCRIPT = PROJECT / "run_qwen_target_sft_label_control.py"
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

NATIVE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|> ' + message['content'] + ' <|im_end|> ' }}"
    "{% endfor %}"
)
REVIEWED_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|assistant|> ' }}"
    "{% generation %}{{ message['content'] + ' <|im_end|> ' }}{% endgeneration %}"
    "{% else %}"
    "{{ '<|' + message['role'] + '|> ' + message['content'] + ' <|im_end|> ' }}"
    "{% endif %}"
    "{% endfor %}"
)


def test_wheel_smoke_versions_track_exported_target_sft_contract() -> None:
    assert EXPECTED_TARGET_SFT_LABEL_VERSION_LINES == (
        TARGET_SFT_LABEL_CONTROL_VERSION,
        TARGET_SFT_LABEL_REPORT_VERSION,
    )


def _reviewed() -> tuple[Any, TargetSFTLabelControlSpec]:
    checkpoint = load_checkpoint_control_spec(CHECKPOINT_CONTROL)
    spec = load_target_sft_label_control_spec(CONTROL, checkpoint_spec=checkpoint)
    return checkpoint, spec


def _verify(path: Path = RECORDED_REPORT, **overrides: Path) -> dict[str, Any]:
    checkpoint, spec = _reviewed()
    arguments = {
        "checkpoint_report_path": CHECKPOINT_REPORT,
        "training_path": TRAINING,
        "readiness_path": READINESS,
        "template_path": TEMPLATE,
        **overrides,
    }
    return verify_recorded_target_sft_label_report(
        path,
        spec=spec,
        checkpoint_spec=checkpoint,
        **arguments,
    )


def _rehash_report(report: dict[str, Any]) -> None:
    report.pop("report_fingerprint", None)
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(report)


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    records = load_sft_records(TINY_TRAINING)
    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_end|>": 2,
        "<|system|>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
    }
    tokens: set[str] = set()
    for record in records:
        for message in record.messages:
            tokens.update(message.content.split())
    for token in sorted(tokens):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
    )
    tokenizer.chat_template = NATIVE_TEMPLATE
    return tokenizer


def _tiny_control() -> tuple[
    TargetSFTLabelControlSpec,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    tuple[Any, ...],
]:
    _, reviewed = _reviewed()
    records = load_sft_records(TINY_TRAINING)
    tokenizer = _tiny_tokenizer()
    samples: list[TargetSFTSampleSpec] = []
    for record in records:
        rendered = tokenizer.apply_chat_template(
            [message.to_dict() for message in record.messages],
            chat_template=REVIEWED_TEMPLATE,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        samples.append(
            TargetSFTSampleSpec(
                record_id=record.record_id,
                input_ids=tuple(rendered["input_ids"]),
                assistant_masks=tuple(rendered["assistant_masks"]),
                assistant_generation_text=(
                    record.messages[-1].content + " <|im_end|> "
                ),
            )
        )
    torch.manual_seed(29)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
            loss_type="ForCausalLM",
        )
    )
    width = max(len(sample.input_ids) for sample in samples)
    native_sha256 = "sha256:" + hashlib.sha256(NATIVE_TEMPLATE.encode("utf-8")).hexdigest()
    spec = dataclasses.replace(
        reviewed,
        expected_model_class="GPT2LMHeadModel",
        expected_model_type="gpt2",
        expected_base_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        native_chat_template_sha256=native_sha256,
        assistant_terminator=" <|im_end|> ",
        max_length=64,
        batch_size=2,
        pad_token_id=int(tokenizer.pad_token_id),
        expected_batch_shape=(2, width),
        runtime=dataclasses.replace(reviewed.runtime, torch_num_threads=1),
        samples=tuple(samples),
    )
    return spec, model, tokenizer, records


def test_reviewed_manifest_binds_target_tokens_labels_runtime_and_narrow_scope() -> None:
    checkpoint, spec = _reviewed()

    assert spec.manifest_fingerprint == (
        "sha256:b1c1a6b36db5a8671d8ccdda0669355e14c2efb21b012040ab1764b3e8c936e6"
    )
    assert spec.source_checkpoint_manifest_fingerprint == checkpoint.manifest_fingerprint
    assert spec.expected_model_class == "Qwen2ForCausalLM"
    assert spec.expected_base_parameter_count == 494_032_768
    assert spec.expected_batch_shape == (3, 301)
    assert [sum(sample.assistant_masks) for sample in spec.samples] == [8, 51, 31]
    assert spec.assistant_terminator == "<|im_end|>\n"
    assert spec.runtime.trl_version == "0.29.1"
    assert "all-zero assistant mask" in TARGET_SFT_LABEL_EVIDENCE_BOUNDARY
    assert "does not authenticate" in TARGET_SFT_LABEL_EVIDENCE_BOUNDARY


def test_loaded_tiny_gpt2_executes_real_trl_target_label_contract() -> None:
    spec, model, tokenizer, records = _tiny_control()

    result = execute_loaded_target_sft_label_control(
        spec,
        model=model,
        tokenizer=tokenizer,
        records=records,
        reviewed_template=REVIEWED_TEMPLATE,
        torch_module=torch,
    )

    collator = result["collator"]
    execution = result["execution"]
    assert collator["batch_shape"][0] == 2
    assert collator["supervised_label_count"] == sum(
        sum(sample.assistant_masks) for sample in spec.samples
    )
    assert collator["non_assistant_and_padding_labels_are_minus_100"] is True
    assert collator["assistant_labels_equal_input_ids"] is True
    assert execution["target_forward_executed"] is True
    assert execution["backward_executed"] is False
    assert math.isfinite(execution["forward_loss"])


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
        load_target_sft_label_control_spec(path, checkpoint_spec=checkpoint)


def test_recorded_qwen_report_verifies_exact_collator_and_forward_scope() -> None:
    report = _verify()

    assert report["report_fingerprint"] == (
        "sha256:8b61fa58ea8278444ce63ba5daa0fab88952c1c51e72bc7197b3a2678810421a"
    )
    assert report["result"]["collator"]["supervised_label_count"] == 90
    assert report["result"]["collator"]["ignored_label_count"] == 813
    assert report["result"]["execution"]["forward_loss"] == 1.251716136932373
    assert report["scope"]["fixed_tool_calls_executed"] is True
    assert report["scope"]["pre_arrow_tokenization_executed"] is True
    assert report["scope"]["real_trl_sft_collator_executed"] is True
    assert report["scope"]["backward_or_optimizer_executed"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda report: report["result"]["collator"].__setitem__(
                "supervised_label_count", 64
            ),
            "collator projection",
        ),
        (
            lambda report: report["result"]["execution"].__setitem__(
                "forward_loss", 4.0
            ),
            "execution projection",
        ),
        (
            lambda report: report["scope"].__setitem__(
                "backward_or_optimizer_executed", True
            ),
            "scope projection",
        ),
        (
            lambda report: report["runtime"].__setitem__("trl_version", "0.30.0"),
            "runtime projection",
        ),
    ],
)
def test_semantic_verifier_rejects_cooperatively_rehashed_drift(
    tmp_path: Path, mutator: Any, message: str
) -> None:
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    mutator(report)
    _rehash_report(report)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _verify(path)


def test_verifier_rejects_template_byte_drift(tmp_path: Path) -> None:
    template = tmp_path / TEMPLATE.name
    template.write_text(TEMPLATE.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte identity"):
        _verify(template_path=template)


def test_cli_verifies_recorded_report() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify", str(RECORDED_REPORT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    report = json.loads(completed.stdout)
    assert report["report_fingerprint"].endswith("10421a")
