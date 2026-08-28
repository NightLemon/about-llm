"""用真实 autograd 说明 token-mean gradient accumulation 为什么不能平均 microbatch loss。

两个 microbatch 分别只有 1 和 3 个有效 token。正确目标是四个 token 的总 loss 除以总 token 数；
若先各自求 mean 再平均，只有 1 个 token 的 microbatch 权重就会从 1/4 升到 1/2（放大一倍），
而 3 个 token 的那个从 3/4 降到 1/2。实验与精确分数 oracle 交叉核对。
"""

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
    """构造有效 token 数为 1 与 3 的两个 microbatch，并加入 padding。"""

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
    """保存一种 reduction 路径得到的 loss 与逐 logit 梯度。"""

    loss: float
    gradient: Tensor


def _torch_fixture(
    microbatches: tuple[CategoricalMicrobatch, ...],
) -> tuple[Tensor, Tensor, tuple[tuple[int, ...], ...]]:
    """把手写概率转为 logits/targets，并记录每个 microbatch 的行索引。"""

    logits: list[list[float]] = []
    targets: list[int] = []
    index_groups: list[tuple[int, ...]] = []
    next_index = 0
    for microbatch in microbatches:
        indices: list[int] = []
        for token in microbatch.tokens:
            probabilities = [float(value) for value in token.probabilities]
            logits.append([math.log(probability) for probability in probabilities])
            # padding 用 ignore_index 表示，cross_entropy 应给这些行零梯度。
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
    """分别执行 full、按 token 计数缩放和朴素等 microbatch 权重三条路径。"""

    logits = nn.Parameter(initial_logits.clone())
    valid_count = int((targets != IGNORE_INDEX).sum().item())
    if reduction == "full":
        # 一次性 full-batch mean 是要复现的参考目标。
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
        # accumulation 路径逐 microbatch backward，让梯度真实累加到同一 Parameter。
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
            # 先累加 token loss sum，最后只除一次全局有效 token 数。
            logits.grad.div_(valid_count)
            reported_loss = float(torch.stack(recorded_losses).sum().item() / valid_count)
        else:
            # 错误路径把每个 microbatch mean 等权平均，忽略它们有效 token 数不同。
            logits.grad.div_(len(index_groups))
            reported_loss = float(torch.stack(recorded_losses).mean().item())
    if logits.grad is None:
        raise AssertionError("autograd did not populate logits.grad")
    return AutogradPath(reported_loss, logits.grad.detach().clone())


def run_toy() -> dict[str, object]:
    """并排运行精确 oracle 与 PyTorch autograd 三种 reduction。"""

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
    """输出 loss、聚合梯度、padding 梯度和路径差异。"""

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
