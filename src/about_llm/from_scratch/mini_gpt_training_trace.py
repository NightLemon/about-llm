"""Run one causal-language-model training step on the shared teaching sample."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT, TransformerBlock
from about_llm.from_scratch.language_model_sample import build_language_model_sample

SCHEMA_VERSION = "about-llm.minigpt-training-trace.v1"
DEFAULT_SEED = 37
DEFAULT_LEARNING_RATE = 0.1
IGNORE_INDEX = -100


def run_minigpt_training_trace(
    *,
    seed: int = DEFAULT_SEED,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict[str, Any]:
    """Trace forward, masked NLL, backward, and one SGD update on CPU."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or not 0.0 < float(learning_rate) <= 1.0
    ):
        raise ValueError("learning_rate must be finite and in (0, 1]")

    sample = build_language_model_sample()
    teaching_model = sample["teaching_model"]
    input_ids = list(teaching_model["model_input_ids"])
    original_labels = list(teaching_model["labels"])
    loss_mask = list(teaching_model["loss_mask"])
    supervised_labels = [
        target_id if included else IGNORE_INDEX
        for target_id, included in zip(original_labels, loss_mask, strict=True)
    ]

    torch.manual_seed(seed)
    config = GPTConfig(
        vocab_size=int(teaching_model["embedding_row_count_required"]),
        context_length=len(input_ids),
        model_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2,
        dropout=0.0,
    )
    model = MiniGPT(config).cpu().train()
    first_block = cast(TransformerBlock, model.blocks[0])
    model_mask = first_block.attention.causal_mask[
        0, 0, : len(input_ids), : len(input_ids)
    ].tolist()
    if model_mask != teaching_model["causal_attention_mask"]:
        raise AssertionError("MiniGPT causal mask does not match the teaching trace")

    inputs = torch.tensor([input_ids], dtype=torch.long)
    targets = torch.tensor([supervised_labels], dtype=torch.long)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))

    logits_before, loss_before_tensor = model(inputs, targets)
    if loss_before_tensor is None:
        raise AssertionError("MiniGPT did not return a supervised loss")
    before_rows = _position_probabilities(
        logits_before,
        teaching_model["position_trace"],
        original_labels,
        loss_mask,
    )
    scored_nlls = [
        float(row["negative_log_probability"])
        for row in before_rows
        if row["negative_log_probability"] is not None
    ]
    manual_mean_nll = sum(scored_nlls) / len(scored_nlls)
    loss_before = float(loss_before_tensor.detach())

    optimizer.zero_grad(set_to_none=True)
    loss_before_tensor.backward()
    gradient_global_l2 = math.sqrt(
        sum(
            float(torch.sum(parameter.grad.detach().double().square()))
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer.step()

    with torch.no_grad():
        logits_after, loss_after_tensor = model(inputs, targets)
    if loss_after_tensor is None:
        raise AssertionError("MiniGPT did not return a post-update loss")
    after_rows = _position_probabilities(
        logits_after,
        teaching_model["position_trace"],
        original_labels,
        loss_mask,
    )
    loss_after = float(loss_after_tensor)
    parameter_changes = {
        name: float((parameter.detach() - parameters_before[name]).abs().max())
        for name, parameter in model.named_parameters()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "sample": {
            "text": sample["text"],
            "input_ids": input_ids,
            "original_labels": original_labels,
            "supervised_labels": supervised_labels,
            "effective_target_count": int(teaching_model["effective_target_count"]),
        },
        "model": {
            "implementation": "about_llm.from_scratch.gpt_torch.MiniGPT",
            "device": "cpu",
            "seed": seed,
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "model_dim": config.model_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "token_embedding_and_lm_head_are_tied": (
                model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
            ),
            "causal_mask_matches_input_trace": True,
        },
        "forward_before_update": {
            "logits_shape": list(logits_before.shape),
            "positions": before_rows,
            "mean_nll_from_model": loss_before,
            "mean_nll_recomputed_from_positions": manual_mean_nll,
            "perplexity_on_three_targets": math.exp(loss_before),
        },
        "backward_and_update": {
            "optimizer": "SGD",
            "learning_rate": float(learning_rate),
            "gradient_global_l2": gradient_global_l2,
            "updated_parameter_tensor_count": sum(
                change > 0.0 for change in parameter_changes.values()
            ),
            "parameter_tensor_count": len(parameter_changes),
            "maximum_parameter_change": max(parameter_changes.values()),
            "mean_nll_after_one_step": loss_after,
            "perplexity_after_one_step": math.exp(loss_after),
            "loss_decreased_on_same_sample": loss_after < loss_before,
            "positions_after_update": after_rows,
        },
        "scope": {
            "byte_bpe_and_training_targets_reused": True,
            "embedding_attention_mlp_and_lm_head_executed": True,
            "masked_cross_entropy_recomputed": True,
            "backward_and_optimizer_step_executed": True,
            "randomly_initialized_teaching_model": True,
            "pretrained_checkpoint_loaded": False,
            "language_quality_or_generalization_measured": False,
        },
    }


def _position_probabilities(
    logits: torch.Tensor,
    position_trace: list[dict[str, Any]],
    original_labels: list[int],
    loss_mask: list[bool],
) -> list[dict[str, Any]]:
    log_probabilities = torch.log_softmax(logits.detach().double(), dim=-1)
    rows: list[dict[str, Any]] = []
    for position, (trace, target_id, included) in enumerate(
        zip(position_trace, original_labels, loss_mask, strict=True)
    ):
        log_probability = float(log_probabilities[0, position, target_id])
        rows.append(
            {
                "position": position,
                "input_piece": trace["input_piece"],
                "target_piece": trace["target_piece"],
                "included_in_loss": included,
                "target_probability": math.exp(log_probability),
                "negative_log_probability": -log_probability if included else None,
            }
        )
    return rows
