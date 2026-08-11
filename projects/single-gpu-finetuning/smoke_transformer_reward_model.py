"""Offline text-tokenization and Transformer reward-model optimizer control."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2ForSequenceClassification, PreTrainedTokenizerFast

from about_llm.finetuning import (
    audit_preference_tokenization,
    load_preference_records,
    load_preference_training_readiness,
    validate_preference_training_readiness,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FIXTURE = (
    ROOT / "projects" / "single-gpu-finetuning" / "preference.train.example.jsonl"
)
READINESS_FIXTURE = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "preference-training-readiness.example.json"
)
MAX_LENGTH = 48
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|> ' + message['content'] + ' ' + eos_token + ' ' }}"
    "{% endfor %}"
)


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


def _render(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict[str, str]],
) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
    )
    if not isinstance(token_ids, list) or not token_ids:
        raise AssertionError("reward-model input must render to non-empty token ids")
    if len(token_ids) > MAX_LENGTH:
        raise AssertionError("reward-model input must not be silently truncated")
    return [int(token_id) for token_id in token_ids]


def _batch(
    sequences: list[list[int]],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), maximum),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


def _scores(
    model: GPT2ForSequenceClassification,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    if logits.shape != (input_ids.shape[0], 1):
        raise AssertionError("reward model must emit exactly one scalar per sequence")
    return logits[:, 0]


def _metrics(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> dict[str, object]:
    margins = chosen_scores - rejected_scores
    losses = functional.softplus(-margins)
    return {
        "pair_count": int(margins.numel()),
        "mean_loss": float(losses.mean().detach()),
        "strict_pair_accuracy": float((margins > 0).float().mean().detach()),
        "tie_count": int((margins == 0).sum().detach()),
        "mean_margin": float(margins.mean().detach()),
        "minimum_margin": float(margins.min().detach()),
    }


def run_smoke(
    steps: int = 4,
    *,
    train_path: Path = TRAIN_FIXTURE,
    readiness_path: Path = READINESS_FIXTURE,
) -> dict[str, object]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    torch.manual_seed(29)
    training = load_preference_records(train_path)
    readiness = load_preference_training_readiness(readiness_path)
    train_audit = validate_preference_training_readiness(training, readiness)

    tokenizer = _tokenizer(training)
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
                row["prompt"] + row["chosen"], tokenize=True, return_dict=True
            )["input_ids"],
            "prompt_rejected_ids": tokenizer.apply_chat_template(
                row["prompt"] + row["rejected"], tokenize=True, return_dict=True
            )["input_ids"],
        },
        renderer_identity={
            "tokenizer": "train-only-local-wordlevel",
            "chat_template": tokenizer.chat_template,
            "transformers": "GPT2ForSequenceClassification",
        },
        max_length=MAX_LENGTH,
    )

    rows = [record.to_dpo_row() for record in training]
    chosen_ids = [_render(tokenizer, row["prompt"] + row["chosen"]) for row in rows]
    rejected_ids = [
        _render(tokenizer, row["prompt"] + row["rejected"]) for row in rows
    ]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise AssertionError("offline reward-model tokenizer requires a pad token")
    input_ids, attention_mask = _batch(
        chosen_ids + rejected_ids,
        pad_token_id=pad_token_id,
    )

    model = GPT2ForSequenceClassification(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=MAX_LENGTH,
            n_ctx=MAX_LENGTH,
            n_embd=32,
            n_layer=1,
            n_head=2,
            num_labels=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=False,
        )
    )
    torch.nn.init.zeros_(model.score.weight)
    pair_count = len(training)
    reward_head_before = model.score.weight.detach().clone()
    embedding_before = model.transformer.wte.weight.detach().clone()

    model.eval()
    with torch.no_grad():
        initial_scores = _scores(model, input_ids, attention_mask)
        initial_metrics = _metrics(
            initial_scores[:pair_count], initial_scores[pair_count:]
        )
    if not math.isclose(float(initial_metrics["mean_loss"]), math.log(2), abs_tol=1e-7):
        raise AssertionError("zero reward head must start at log(2) pairwise loss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scores = _scores(model, input_ids, attention_mask)
        loss = functional.softplus(
            -(scores[:pair_count] - scores[pair_count:])
        ).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_scores = _scores(model, input_ids, attention_mask)
        final_metrics = _metrics(final_scores[:pair_count], final_scores[pair_count:])
        counterfactual_chosen = _render(
            tokenizer,
            [
                {"role": "user", "content": "Select response alpha."},
                {"role": "assistant", "content": "bad alpha answer"},
            ],
        )
        counterfactual_rejected = _render(
            tokenizer,
            [
                {"role": "user", "content": "Select response alpha."},
                {"role": "assistant", "content": "good alpha answer"},
            ],
        )
        counterfactual_ids, counterfactual_mask = _batch(
            [counterfactual_chosen, counterfactual_rejected],
            pad_token_id=pad_token_id,
        )
        counterfactual_scores = _scores(
            model, counterfactual_ids, counterfactual_mask
        )
        counterfactual_metrics = _metrics(
            counterfactual_scores[:1], counterfactual_scores[1:]
        )

    if float(final_metrics["mean_loss"]) >= float(initial_metrics["mean_loss"]):
        raise AssertionError("tiny Transformer reward-model loss did not decrease")
    if final_metrics["strict_pair_accuracy"] != 1:
        raise AssertionError("tiny Transformer reward model did not fit the train pairs")
    reward_head_changed = not torch.equal(model.score.weight.detach(), reward_head_before)
    embedding_changed = not torch.equal(
        model.transformer.wte.weight.detach(), embedding_before
    )
    if not reward_head_changed or not embedding_changed:
        raise AssertionError("reward head and Transformer backbone must both receive updates")
    return {
        "schema_version": 1,
        "train_manifest_fingerprint": train_audit.manifest_fingerprint,
        "combined_manifest_fingerprint_from_readiness": (
            readiness.combined_manifest_fingerprint
        ),
        "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
        "tokenization_manifest_fingerprint": tokenization_audit.manifest_fingerprint,
        "model_class": type(model).__name__,
        "pair_count": pair_count,
        "steps": steps,
        "input_sequence_count": int(input_ids.shape[0]),
        "non_padding_token_count": int(attention_mask.sum()),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "authored_counterfactual_metrics": counterfactual_metrics,
        "reward_head_parameters_changed": reward_head_changed,
        "transformer_backbone_parameters_changed": embedding_changed,
        "scope": {
            "device": str(input_ids.device),
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
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, allow_nan=False))
