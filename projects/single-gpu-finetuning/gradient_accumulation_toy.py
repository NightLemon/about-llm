"""Masked token-mean gradient accumulation with an autograd counterexample."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from about_llm.finetuning import (
    CategoricalMicrobatch,
    CategoricalTokenRecord,
    analyze_masked_token_gradient_accumulation,
)

IGNORE_INDEX = -100


def authored_microbatches() -> tuple[CategoricalMicrobatch, ...]:
    return (
        CategoricalMicrobatch(
            "short",
            (
                CategoricalTokenRecord("short.valid", (9, 1), 0),
                CategoricalTokenRecord("short.padding-1", (1, 1), None),
                CategoricalTokenRecord("short.padding-2", (1, 1), None),
            ),
        ),
        CategoricalMicrobatch(
            "long",
            (
                CategoricalTokenRecord("long.valid-1", (4, 1), 1),
                CategoricalTokenRecord("long.valid-2", (4, 1), 1),
                CategoricalTokenRecord("long.valid-3", (4, 1), 1),
                CategoricalTokenRecord("long.padding", (1, 1), None),
            ),
        ),
    )


@dataclass(frozen=True)
class AutogradPath:
    loss: float
    gradient: Tensor


def _torch_fixture(
    microbatches: tuple[CategoricalMicrobatch, ...],
) -> tuple[Tensor, Tensor, tuple[tuple[int, ...], ...]]:
    logits: list[list[float]] = []
    targets: list[int] = []
    index_groups: list[tuple[int, ...]] = []
    next_index = 0
    for microbatch in microbatches:
        indices: list[int] = []
        for token in microbatch.tokens:
            probabilities = [float(value) for value in token.probabilities]
            logits.append([math.log(probability) for probability in probabilities])
            targets.append(
                IGNORE_INDEX if token.target_index is None else token.target_index
            )
            indices.append(next_index)
            next_index += 1
        index_groups.append(tuple(indices))
    return (
        torch.tensor(logits, dtype=torch.float64),
        torch.tensor(targets, dtype=torch.long),
        tuple(index_groups),
    )


def _autograd_path(
    initial_logits: Tensor,
    targets: Tensor,
    index_groups: tuple[tuple[int, ...], ...],
    *,
    reduction: Literal["full", "count_scaled", "equal_microbatch"],
) -> AutogradPath:
    logits = nn.Parameter(initial_logits.clone())
    valid_count = int((targets != IGNORE_INDEX).sum().item())
    if reduction == "full":
        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        # PyTorch's installed type stubs do not type Tensor.backward().
        loss.backward()  # type: ignore[no-untyped-call]
        reported_loss = float(loss.detach().item())
    else:
        recorded_losses: list[Tensor] = []
        for indices in index_groups:
            index_tensor = torch.tensor(indices, dtype=torch.long)
            microbatch_logits = logits[index_tensor]
            microbatch_targets = targets[index_tensor]
            microbatch_loss = F.cross_entropy(
                microbatch_logits,
                microbatch_targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum" if reduction == "count_scaled" else "mean",
            )
            # Keep the real accumulated-backward path; see the stub note above.
            microbatch_loss.backward()  # type: ignore[no-untyped-call]
            recorded_losses.append(microbatch_loss.detach())
        if logits.grad is None:
            raise AssertionError("autograd did not populate logits.grad")
        if reduction == "count_scaled":
            logits.grad.div_(valid_count)
            reported_loss = float(torch.stack(recorded_losses).sum().item() / valid_count)
        else:
            logits.grad.div_(len(index_groups))
            reported_loss = float(torch.stack(recorded_losses).mean().item())
    if logits.grad is None:
        raise AssertionError("autograd did not populate logits.grad")
    return AutogradPath(reported_loss, logits.grad.detach().clone())


def run_toy() -> dict[str, object]:
    microbatches = authored_microbatches()
    exact = analyze_masked_token_gradient_accumulation(microbatches)
    initial_logits, targets, index_groups = _torch_fixture(microbatches)
    full = _autograd_path(
        initial_logits, targets, index_groups, reduction="full"
    )
    count_scaled = _autograd_path(
        initial_logits, targets, index_groups, reduction="count_scaled"
    )
    naive = _autograd_path(
        initial_logits, targets, index_groups, reduction="equal_microbatch"
    )
    full_aggregate = full.gradient.sum(dim=0)
    count_scaled_aggregate = count_scaled.gradient.sum(dim=0)
    naive_aggregate = naive.gradient.sum(dim=0)
    ignored_mask = targets == IGNORE_INDEX
    return {
        "implementation": "about-llm.gradient-accumulation-toy.v1",
        "exact_oracle": exact.to_dict(),
        "pytorch_autograd": {
            "torch_version": torch.__version__,
            "dtype": "torch.float64",
            "ignore_index": IGNORE_INDEX,
            "full_batch_loss": full.loss,
            "count_scaled_accumulated_loss": count_scaled.loss,
            "naive_equal_microbatch_loss": naive.loss,
            "full_class_aggregate_gradient": full_aggregate.tolist(),
            "count_scaled_class_aggregate_gradient": (
                count_scaled_aggregate.tolist()
            ),
            "naive_equal_microbatch_class_aggregate_gradient": (
                naive_aggregate.tolist()
            ),
            "full_vs_count_scaled_max_abs_gradient_error": float(
                (full.gradient - count_scaled.gradient).abs().max().item()
            ),
            "full_vs_naive_max_abs_gradient_difference": float(
                (full.gradient - naive.gradient).abs().max().item()
            ),
            "ignored_row_max_abs_gradient": float(
                full.gradient[ignored_mask].abs().max().item()
            ),
        },
        "observations": {
            "valid_token_counts_are_one_and_three": [
                contribution.valid_token_count
                for contribution in exact.microbatches
            ]
            == [1, 3],
            "count_scaled_matches_full_exactly_in_fraction_oracle": (
                exact.count_scaled_accumulated_class_aggregate_logit_gradient
                == exact.full_batch_class_aggregate_logit_gradient
            ),
            "equal_microbatch_mean_changes_the_objective": (
                exact.naive_equal_microbatch_class_aggregate_logit_gradient
                != exact.full_batch_class_aggregate_logit_gradient
            ),
            "pytorch_count_scaled_gradient_matches_full": torch.equal(
                full.gradient, count_scaled.gradient
            ),
            "pytorch_naive_gradient_differs_from_full": not torch.equal(
                full.gradient, naive.gradient
            ),
            "ignored_positions_have_zero_gradient": bool(
                torch.count_nonzero(full.gradient[ignored_mask]).item() == 0
            ),
        },
        "scope": {
            "authored_probabilities_targets_and_padding_executed": True,
            "exact_fraction_logit_gradient_oracle_executed": True,
            "pytorch_float64_cross_entropy_backward_executed": True,
            "optimizer_step_or_parameter_update_executed": False,
            "dropout_batchnorm_or_stochastic_model_equivalence_proved": False,
            "ddp_fsdp_zero_collective_or_no_sync_executed": False,
            "amp_cuda_gpu_memory_throughput_or_quality_measured": False,
            "target_llm_tokenizer_dataset_or_training_run_executed": False,
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_toy(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
