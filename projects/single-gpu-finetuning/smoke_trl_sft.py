"""Offline TRL smoke test from strict SFT records through assistant-only labels."""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, disable_progress_bars
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from about_llm.finetuning import (
    NearDuplicateProfile,
    SFTTrainingReadinessReport,
    audit_sft_governance,
    audit_sft_near_duplicates,
    load_sft_governance_policy,
    load_sft_records,
    prepare_assistant_mask_features,
    validate_training_subset,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FIXTURE = ROOT / "projects" / "single-gpu-finetuning" / "train.example.jsonl"
AUDIT_FIXTURE = ROOT / "projects" / "single-gpu-finetuning" / "audit.example.jsonl"
MAX_LENGTH = 64
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|> ' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + ' ' + eos_token }}{% endgeneration %}"
    "{% else %}{{ message['content'] + ' ' + eos_token }}{% endif %}"
    "{% endfor %}"
)


def _tokenizer(records: tuple[Any, ...]) -> PreTrainedTokenizerFast:
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "</s>": 2}
    tokens = {f"<|{message.role.value}|>" for record in records for message in record.messages}
    tokens.update(
        token
        for record in records
        for message in record.messages
        for token in message.content.split()
    )
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


def _assert_collator_labels(
    trainer: SFTTrainer, dataset: Any
) -> tuple[dict[str, torch.Tensor], int, int]:
    features = [dict(dataset[index]) for index in range(len(dataset))]
    batch = trainer.data_collator(features)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    supervised = 0
    ignored = 0
    for row, feature in enumerate(features):
        masks = feature["assistant_masks"]
        for column, mask in enumerate(masks):
            if mask == 1:
                if labels[row, column] != input_ids[row, column]:
                    raise AssertionError("assistant token was not preserved as a label")
                supervised += 1
            else:
                if labels[row, column] != -100:
                    raise AssertionError("non-assistant token was not ignored")
                ignored += 1
    if supervised == 0 or ignored == 0:
        raise AssertionError("smoke fixture must exercise supervised and ignored labels")
    return batch, supervised, ignored


def _loss(model: GPT2LMHeadModel, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        value = float(
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss
        )
    if not math.isfinite(value):
        raise AssertionError("TRL smoke loss must be finite")
    return value


def run_smoke(steps: int = 12) -> dict[str, object]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    torch.manual_seed(19)
    records = load_sft_records(TRAIN_FIXTURE)
    audited_records = load_sft_records(AUDIT_FIXTURE)
    binding = validate_training_subset(records, audited_records)
    data_audit = binding.training_report
    near_audit = audit_sft_near_duplicates(
        audited_records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=0.9,
    )
    governance_audit = audit_sft_governance(
        audited_records,
        policy=load_sft_governance_policy(
            ROOT
            / "projects"
            / "single-gpu-finetuning"
            / "governance-policy.example.json"
        ),
        evaluated_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
    )
    readiness = SFTTrainingReadinessReport.from_reports(
        binding, near_audit, governance_audit
    )
    if not readiness.gate_passed:
        raise AssertionError("offline fixture unexpectedly failed lexical candidate gate")
    tokenizer = _tokenizer(records)
    mask_preparation = prepare_assistant_mask_features(
        records,
        render=lambda messages, tools: tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        ),
        renderer_identity={
            "kind": "offline-wordlevel-fixture",
            "chat_template": tokenizer.chat_template,
            "vocabulary": tokenizer.get_vocab(),
        },
        max_length=MAX_LENGTH,
    )
    mask_audit = mask_preparation.audit_report
    dataset = Dataset.from_list(mask_preparation.to_training_rows())
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=MAX_LENGTH,
            n_ctx=MAX_LENGTH,
            n_embd=32,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
            loss_type="ForCausalLM",
        )
    )
    disable_progress_bars()
    with tempfile.TemporaryDirectory(prefix="about-llm-trl-smoke-") as directory:
        config = SFTConfig(
            output_dir=directory,
            max_length=MAX_LENGTH,
            # The pre-tokenized rows already carry assistant_masks; TRL's
            # collator consumes them independently of this preprocessing flag.
            assistant_only_loss=False,
            completion_only_loss=False,
            per_device_train_batch_size=len(records),
            learning_rate=5e-3,
            max_steps=steps,
            gradient_checkpointing=False,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            use_cpu=True,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            seed=19,
            data_seed=19,
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        prepared = trainer.train_dataset
        if prepared is None:
            raise AssertionError("TRL did not prepare the training dataset")
        batch, supervised, ignored = _assert_collator_labels(trainer, prepared)
        initial_loss = _loss(model, batch)
        trainer.train()
        final_loss = _loss(model, batch)
    if final_loss >= initial_loss:
        raise AssertionError(
            f"tiny SFT failed to overfit: initial={initial_loss}, final={final_loss}"
        )
    return {
        "evidence_boundary": (
            "Random tiny GPT-2, local WordLevel tokenizer, and authored fixture prove only "
            "the offline TRL control path and label contract; they do not prove target-model "
            "quality, CUDA behavior, real-data legality, or production convergence."
        ),
        "data_manifest_fingerprint": data_audit.manifest_fingerprint,
        "split_manifest_fingerprint": binding.split_report.manifest_fingerprint,
        "binding_fingerprint": binding.binding_fingerprint,
        "near_duplicate_manifest_fingerprint": near_audit.manifest_fingerprint,
        "governance_manifest_fingerprint": governance_audit.manifest_fingerprint,
        "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
        "mask_manifest_fingerprint": mask_audit.manifest_fingerprint,
        "record_count": len(records),
        "supervised_label_count": supervised,
        "ignored_label_count": ignored,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": steps,
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))
