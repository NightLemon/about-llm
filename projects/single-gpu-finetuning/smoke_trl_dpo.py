"""离线跑通严格 preference records 到 TRL DPO optimizer step。

实验用随机微型 GPT-2 和本地 tokenizer，先审计 chosen/rejected 数据、治理、近重复与截断，
再冻结 reference model，执行若干 DPO 更新并比较训练前后的 loss。
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, disable_progress_bars
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import DPOConfig, DPOTrainer

from about_llm.finetuning import (
    NearDuplicateProfile,
    PreferenceTrainingReadinessReport,
    audit_preference_governance,
    audit_preference_near_duplicates,
    audit_preference_records,
    audit_preference_tokenization,
    load_preference_records,
    load_sft_governance_policy,
    validate_dpo_training_subset,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "projects" / "single-gpu-finetuning" / "preference.example.jsonl"
TRAIN_FIXTURE = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "preference.train.example.jsonl"
)
MAX_LENGTH = 48
BETA = 0.2
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|> ' + message['content'] + ' ' + eos_token + ' ' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
)


def _tokenizer(records: tuple[Any, ...]) -> PreTrainedTokenizerFast:
    """根据固定 preference 文本构建最小 tokenizer 和共享 chat template。"""

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


def _collated_batch(
    trainer: DPOTrainer, prepared: Any
) -> tuple[dict[str, torch.Tensor], int, int]:
    """从 DPOTrainer 取出一批真实 collated chosen/rejected 张量。"""

    features = [dict(prepared[index]) for index in range(len(prepared))]
    batch = trainer.data_collator(features)
    pair_count = len(features)
    if batch["input_ids"].shape[0] != pair_count * 2:
        raise AssertionError("DPO collator must concatenate chosen then rejected rows")
    completion_tokens = 0
    prompt_tokens = 0
    for index, feature in enumerate(features):
        prompt_ids = feature["prompt_ids"]
        chosen_ids = feature["chosen_ids"]
        rejected_ids = feature["rejected_ids"]
        if not prompt_ids or not chosen_ids or not rejected_ids:
            raise AssertionError("tokenized prompt/chosen/rejected must all be non-empty")
        for row, completion_ids in (
            (index, chosen_ids),
            (index + pair_count, rejected_ids),
        ):
            prompt_length = len(prompt_ids)
            completion_length = len(completion_ids)
            actual_ids = batch["input_ids"][
                row, : prompt_length + completion_length
            ].tolist()
            if actual_ids != prompt_ids + completion_ids:
                raise AssertionError("chosen/rejected ids were reordered by the collator")
            mask = batch["completion_mask"][row]
            if mask[:prompt_length].count_nonzero().item() != 0:
                raise AssertionError("prompt tokens leaked into the DPO completion mask")
            if mask[prompt_length : prompt_length + completion_length].sum().item() != (
                completion_length
            ):
                raise AssertionError("completion tokens were missing from the DPO mask")
            prompt_tokens += prompt_length
            completion_tokens += completion_length
    return batch, prompt_tokens, completion_tokens


def _loss(trainer: DPOTrainer, batch: dict[str, torch.Tensor]) -> float:
    """用 Trainer 自己的 loss 路径计算固定 batch 指标。"""

    trainer.model.eval()
    with torch.no_grad():
        value = float(trainer.compute_loss(trainer.model, batch))
    if not math.isfinite(value):
        raise AssertionError("DPO smoke loss must be finite")
    return value


def run_smoke(steps: int = 20) -> dict[str, object]:
    """完成数据门禁、policy/reference 构造和 DPO 训练闭环。"""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    torch.manual_seed(23)
    # 先确保数据与 split 审计通过，再初始化 policy/reference 模型。
    records = load_preference_records(FIXTURE)
    split_audit = audit_preference_records(records)
    if not split_audit.gate_passed:
        raise AssertionError("preference fixture unexpectedly failed split audit")
    training = load_preference_records(TRAIN_FIXTURE)
    binding = validate_dpo_training_subset(training, records)
    train_audit = binding.training_report
    near_audit = audit_preference_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=0.9,
    )
    governance_audit = audit_preference_governance(
        records,
        policy=load_sft_governance_policy(
            ROOT
            / "projects"
            / "single-gpu-finetuning"
            / "governance-policy.example.json"
        ),
        evaluated_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
    )
    readiness = PreferenceTrainingReadinessReport.from_reports(
        binding, near_audit, governance_audit
    )
    if not readiness.gate_passed:
        raise AssertionError("preference fixture unexpectedly failed readiness")
    tokenizer = _tokenizer(records)
    tokenization_audit = audit_preference_tokenization(
        training,
        render=lambda row: {
            "prompt_ids": tokenizer.apply_chat_template(
                row["prompt"],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            ),
            "prompt_chosen_ids": tokenizer.apply_chat_template(
                row["prompt"] + row["chosen"],
                tokenize=True,
                return_dict=True,
            )["input_ids"],
            "prompt_rejected_ids": tokenizer.apply_chat_template(
                row["prompt"] + row["rejected"],
                tokenize=True,
                return_dict=True,
            )["input_ids"],
        },
        renderer_identity={
            "tokenizer": "local-wordlevel",
            "chat_template": tokenizer.chat_template,
            "trl_version": "0.29",
        },
        max_length=MAX_LENGTH,
    )
    dataset = Dataset.from_list([record.to_dpo_row() for record in training])
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
        )
    )
    # reference 是更新前 policy 的冻结深拷贝，训练期间逐参数检查它没有变化。
    reference = copy.deepcopy(model)
    reference_before = {
        name: parameter.detach().clone() for name, parameter in reference.named_parameters()
    }
    disable_progress_bars()
    with tempfile.TemporaryDirectory(prefix="about-llm-trl-dpo-") as directory:
        config = DPOConfig(
            output_dir=directory,
            max_length=MAX_LENGTH,
            beta=BETA,
            loss_type=["sigmoid"],
            per_device_train_batch_size=len(training),
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
            seed=23,
            data_seed=23,
        )
        # Trainer 计算同一 prompt 下 policy/reference 的 chosen/rejected log probability。
        trainer = DPOTrainer(
            model=model,
            ref_model=reference,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        prepared = trainer.train_dataset
        if prepared is None:
            raise AssertionError("TRL did not prepare the preference dataset")
        batch, prompt_tokens, completion_tokens = _collated_batch(trainer, prepared)
        initial_loss = _loss(trainer, batch)
        with redirect_stdout(sys.stderr):
            train_result = trainer.train()
        final_loss = _loss(trainer, batch)
    if not math.isclose(initial_loss, math.log(2), rel_tol=1e-3, abs_tol=5e-4):
        raise AssertionError(
            f"identical policy/reference should start at log(2), got {initial_loss}"
        )
    if final_loss >= initial_loss:
        raise AssertionError(
            f"tiny DPO failed to improve: initial={initial_loss}, final={final_loss}"
        )
    for name, parameter in reference.named_parameters():
        if not torch.equal(parameter.detach(), reference_before[name]):
            raise AssertionError(f"reference parameter changed during DPO: {name}")
    return {
        "task_contract": {
            "training_unit": "one authored prompt with chosen and rejected completions",
            "supervision": "prompt tokens are masked; completion log probabilities enter DPO",
            "reference_model": "frozen copy of the initial policy",
            "objective": "increase chosen-vs-rejected preference under beta-scaled DPO loss",
        },
        "evidence_boundary": (
            "Random tiny GPT-2, local WordLevel tokenizer, and authored pairs prove only "
            "the offline TRL 0.29 control path, completion masking, frozen reference, and "
            "tiny-pair optimization. They do not prove target-model preference quality, "
            "human label validity, CUDA behavior, safety alignment, or production convergence."
        ),
        "split_manifest_fingerprint": split_audit.manifest_fingerprint,
        "train_manifest_fingerprint": train_audit.manifest_fingerprint,
        "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
        "tokenization_manifest_fingerprint": (
            tokenization_audit.manifest_fingerprint
        ),
        "pair_count": len(training),
        "prompt_masked_token_count": prompt_tokens,
        "completion_token_count": completion_tokens,
        "beta": BETA,
        "initial_dpo_loss": initial_loss,
        "final_dpo_loss": final_loss,
        "steps": steps,
        "trainer_metrics": train_result.metrics,
        "reference_parameters_unchanged": True,
        "outcome": {
            "prompt_masking_contract_passed": True,
            "reference_parameters_unchanged": True,
            "training_loss_decreased": final_loss < initial_loss,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))
