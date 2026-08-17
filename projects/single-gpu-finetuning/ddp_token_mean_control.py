"""Two-process CPU/Gloo control for global masked-token mean gradients."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.multiprocessing.spawn import spawn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from about_llm.finetuning import (
    CategoricalMicrobatch,
    CategoricalTokenRecord,
    analyze_default_ddp_token_mean,
)

WORLD_SIZE = 2
IGNORE_INDEX = -100
GradientPath = Literal[
    "correct_d_over_n",
    "missing_world_size",
    "rank_local_mean",
]
GRADIENT_PATHS: tuple[GradientPath, ...] = (
    "correct_d_over_n",
    "missing_world_size",
    "rank_local_mean",
)


def authored_rank_shards() -> tuple[CategoricalMicrobatch, ...]:
    return (
        CategoricalMicrobatch(
            "rank-0",
            (
                CategoricalTokenRecord("rank-0.valid", (9, 1), 0),
                CategoricalTokenRecord("rank-0.padding-1", (1, 1), None),
                CategoricalTokenRecord("rank-0.padding-2", (1, 1), None),
            ),
        ),
        CategoricalMicrobatch(
            "rank-1",
            (
                CategoricalTokenRecord("rank-1.valid-1", (4, 1), 1),
                CategoricalTokenRecord("rank-1.valid-2", (4, 1), 1),
                CategoricalTokenRecord("rank-1.valid-3", (4, 1), 1),
                CategoricalTokenRecord("rank-1.padding", (1, 1), None),
            ),
        ),
    )


class SharedBiasClassifier(nn.Module):
    """Add one shared two-class parameter to rank-specific fixed logits."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def forward(self, base_logits: Tensor) -> Tensor:
        return base_logits + self.bias


@dataclass(frozen=True, slots=True)
class RankFixture:
    base_logits: Tensor
    targets: Tensor
    local_valid_token_count: int


def _rank_fixture(rank: int) -> RankFixture:
    shard = authored_rank_shards()[rank]
    logits: list[list[float]] = []
    targets: list[int] = []
    for token in shard.tokens:
        logits.append([math.log(float(value)) for value in token.probabilities])
        targets.append(
            IGNORE_INDEX if token.target_index is None else token.target_index
        )
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return RankFixture(
        base_logits=torch.tensor(logits, dtype=torch.float64),
        targets=target_tensor,
        local_valid_token_count=int((target_tensor != IGNORE_INDEX).sum().item()),
    )


def _run_gradient_path(
    ddp_model: DistributedDataParallel,
    model: SharedBiasClassifier,
    fixture: RankFixture,
    *,
    path: GradientPath,
    global_valid_token_count: int,
) -> list[float]:
    ddp_model.zero_grad(set_to_none=True)
    logits = ddp_model(fixture.base_logits)
    if path == "rank_local_mean":
        loss = F.cross_entropy(
            logits,
            fixture.targets,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
    else:
        loss_sum = F.cross_entropy(
            logits,
            fixture.targets,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        numerator = WORLD_SIZE if path == "correct_d_over_n" else 1
        loss = loss_sum * numerator / global_valid_token_count
    # The installed PyTorch stubs do not type Tensor.backward().
    loss.backward()  # type: ignore[no-untyped-call]
    if model.bias.grad is None:
        raise AssertionError("DDP backward did not populate the shared bias gradient")
    gradient = model.bias.grad.detach().clone()
    if not bool(torch.isfinite(gradient).all().item()):
        raise AssertionError("DDP gradient must be finite")
    return [float(value) for value in gradient.tolist()]


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_directory: str,
) -> None:
    if world_size != WORLD_SIZE:
        raise ValueError(f"world_size must be {WORLD_SIZE}")
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        fixture = _rank_fixture(rank)
        count = torch.tensor([fixture.local_valid_token_count], dtype=torch.int64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        global_count = int(count.item())
        model = SharedBiasClassifier()
        ddp_model = DistributedDataParallel(model)
        gradients = {
            path: _run_gradient_path(
                ddp_model,
                model,
                fixture,
                path=path,
                global_valid_token_count=global_count,
            )
            for path in GRADIENT_PATHS
        }
        payload = {
            "rank": rank,
            "world_size": world_size,
            "local_valid_token_count": fixture.local_valid_token_count,
            "global_valid_token_count_after_all_reduce": global_count,
            "gradients": gradients,
        }
        output_path = Path(output_directory) / f"rank-{rank}.json"
        output_path.write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _full_batch_reference() -> list[float]:
    fixtures = tuple(_rank_fixture(rank) for rank in range(WORLD_SIZE))
    model = SharedBiasClassifier()
    logits = model(torch.cat([fixture.base_logits for fixture in fixtures], dim=0))
    targets = torch.cat([fixture.targets for fixture in fixtures], dim=0)
    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    # The installed PyTorch stubs do not type Tensor.backward().
    loss.backward()  # type: ignore[no-untyped-call]
    if model.bias.grad is None:
        raise AssertionError("full-batch backward did not populate the bias gradient")
    return [float(value) for value in model.bias.grad.detach().tolist()]


def _max_abs_difference(left: list[float], right: list[float]) -> float:
    return max(
        abs(left_value - right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def run_control() -> dict[str, object]:
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed with the Gloo backend is required")
    exact = analyze_default_ddp_token_mean(
        authored_rank_shards(),
        data_parallel_world_size=WORLD_SIZE,
    )
    with tempfile.TemporaryDirectory(prefix="about-llm-ddp-token-mean-") as temporary:
        root = Path(temporary)
        rendezvous_path = root / "rendezvous"
        init_method = rendezvous_path.resolve().as_uri()
        output_directory = root / "rank-results"
        output_directory.mkdir()
        spawn(  # type: ignore[no-untyped-call]
            _worker,
            args=(WORLD_SIZE, init_method, str(output_directory)),
            nprocs=WORLD_SIZE,
            join=True,
        )
        rank_reports = [
            json.loads(
                (output_directory / f"rank-{rank}.json").read_text(encoding="utf-8")
            )
            for rank in range(WORLD_SIZE)
        ]

    full_batch = _full_batch_reference()
    first_gradients = rank_reports[0]["gradients"]
    if not isinstance(first_gradients, dict):
        raise AssertionError("rank gradient report must be an object")
    correct = first_gradients["correct_d_over_n"]
    missing_world_size = first_gradients["missing_world_size"]
    rank_local_mean = first_gradients["rank_local_mean"]
    if not all(isinstance(value, list) for value in first_gradients.values()):
        raise AssertionError("rank gradients must be lists")
    if any(report["gradients"] != first_gradients for report in rank_reports[1:]):
        raise AssertionError("DDP must synchronize the observed gradients across ranks")
    if not all(
        report["global_valid_token_count_after_all_reduce"] == 4
        for report in rank_reports
    ):
        raise AssertionError("global valid-token count all-reduce must equal four")

    correct_error = _max_abs_difference(correct, full_batch)
    missing_expected = [value / WORLD_SIZE for value in full_batch]
    missing_error = _max_abs_difference(missing_world_size, missing_expected)
    return {
        "implementation": "about-llm.ddp-token-mean-control.v1",
        "exact_oracle": exact.to_dict(),
        "runtime": {
            "torch_version": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
            "dtype": "torch.float64",
            "world_size": WORLD_SIZE,
            "process_start_method": "spawn",
            "rendezvous": "temporary-file-store",
        },
        "rank_reports": rank_reports,
        "full_batch_shared_bias_gradient": full_batch,
        "observed_default_ddp_gradients": first_gradients,
        "comparison": {
            "correct_d_over_n_vs_full_max_abs_error": correct_error,
            "missing_world_size_vs_full_divided_by_d_max_abs_error": missing_error,
            "rank_local_mean_vs_full_max_abs_difference": (
                _max_abs_difference(rank_local_mean, full_batch)
            ),
        },
        "observations": {
            "correct_d_over_n_matches_full_within_1e_15": correct_error <= 1e-15,
            "missing_world_size_is_full_divided_by_d_within_1e_15": (
                missing_error <= 1e-15
            ),
            "rank_local_mean_changes_the_objective": rank_local_mean != full_batch,
            "all_ranks_observe_identical_synchronized_gradients": all(
                report["gradients"] == first_gradients for report in rank_reports
            ),
            "global_valid_token_count_all_reduce_is_four": all(
                report["global_valid_token_count_after_all_reduce"] == 4
                for report in rank_reports
            ),
        },
        "scope": {
            "real_two_process_same_host_gloo_process_group_executed": True,
            "default_ddp_gradient_averaging_observed": True,
            "global_valid_token_count_all_reduce_executed": True,
            "temporary_file_store_rendezvous_executed": True,
            "optimizer_step_parameter_update_or_gradient_clipping_executed": False,
            "gradient_accumulation_no_sync_amp_or_scaler_executed": False,
            "fsdp_zero_tensor_pipeline_expert_parallel_executed": False,
            "cuda_gpu_multi_node_or_remote_host_executed": False,
            "target_llm_tokenizer_dataset_or_quality_evaluation_executed": False,
            "bitwise_equivalence_across_hardware_or_world_sizes_proved": False,
            "transport_security_packet_capture_or_fault_injection_executed": False,
        },
    }


def main() -> None:
    print(
        json.dumps(
            run_control(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
