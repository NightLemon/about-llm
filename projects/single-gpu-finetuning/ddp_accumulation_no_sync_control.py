"""Two-process CPU/Gloo control for DDP accumulation and ``no_sync``."""

import json
import math
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import GradBucket  # type: ignore[attr-defined]
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.multiprocessing.spawn import spawn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from about_llm.finetuning import (
    CategoricalMicrobatch,
    CategoricalTokenRecord,
    analyze_default_ddp_gradient_accumulation,
)

WORLD_SIZE = 2
ACCUMULATION_STEPS = 2
IGNORE_INDEX = -100
LEARNING_RATE_FRACTION = Fraction(7, 20)
LEARNING_RATE = float(LEARNING_RATE_FRACTION)
MAX_GRAD_NORM = 0.5
CaseName = Literal[
    "builtin_no_sync",
    "counting_hook_no_sync",
    "counting_hook_backward_only",
]
CASE_NAMES: tuple[CaseName, ...] = (
    "builtin_no_sync",
    "counting_hook_no_sync",
    "counting_hook_backward_only",
)


def authored_rank_windows() -> tuple[tuple[CategoricalMicrobatch, ...], ...]:
    """Return two rank windows with effective-token counts ``[[1,2],[3,1]]``."""

    return (
        (
            CategoricalMicrobatch(
                "rank-0.micro-0",
                (
                    CategoricalTokenRecord("r0.m0.valid", (9, 1), 0),
                    CategoricalTokenRecord("r0.m0.ignored", (1, 1), None),
                ),
            ),
            CategoricalMicrobatch(
                "rank-0.micro-1",
                (
                    CategoricalTokenRecord("r0.m1.valid-1", (4, 1), 1),
                    CategoricalTokenRecord("r0.m1.valid-2", (4, 1), 1),
                    CategoricalTokenRecord("r0.m1.ignored", (1, 1), None),
                ),
            ),
        ),
        (
            CategoricalMicrobatch(
                "rank-1.micro-0",
                (
                    CategoricalTokenRecord("r1.m0.valid-1", (4, 1), 1),
                    CategoricalTokenRecord("r1.m0.valid-2", (4, 1), 1),
                    CategoricalTokenRecord("r1.m0.valid-3", (4, 1), 1),
                    CategoricalTokenRecord("r1.m0.ignored", (1, 1), None),
                ),
            ),
            CategoricalMicrobatch(
                "rank-1.micro-1",
                (
                    CategoricalTokenRecord("r1.m1.valid", (9, 1), 0),
                    CategoricalTokenRecord("r1.m1.ignored", (1, 1), None),
                ),
            ),
        ),
    )


class SharedBiasClassifier(nn.Module):
    """Add one shared two-class parameter to authored fixed logits."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def forward(self, base_logits: Tensor) -> Tensor:
        return base_logits + self.bias


@dataclass(frozen=True, slots=True)
class MicrobatchFixture:
    base_logits: Tensor
    targets: Tensor
    valid_token_count: int


@dataclass(slots=True)
class AllReduceHookState:
    process_group: dist.ProcessGroup
    calls: int = 0
    bucket_numel: list[int] = field(default_factory=list)


def _counting_allreduce_hook(
    state: AllReduceHookState,
    bucket: GradBucket,
) -> torch.futures.Future[Tensor]:
    state.calls += 1
    state.bucket_numel.append(bucket.buffer().numel())
    return default_hooks.allreduce_hook(state.process_group, bucket)


def _rank_fixtures(rank: int) -> tuple[MicrobatchFixture, ...]:
    fixtures: list[MicrobatchFixture] = []
    for microbatch in authored_rank_windows()[rank]:
        logits: list[list[float]] = []
        targets: list[int] = []
        for token in microbatch.tokens:
            logits.append([math.log(float(value)) for value in token.probabilities])
            targets.append(
                IGNORE_INDEX if token.target_index is None else token.target_index
            )
        target_tensor = torch.tensor(targets, dtype=torch.long)
        fixtures.append(
            MicrobatchFixture(
                base_logits=torch.tensor(logits, dtype=torch.float64),
                targets=target_tensor,
                valid_token_count=int(
                    (target_tensor != IGNORE_INDEX).sum().item()
                ),
            )
        )
    return tuple(fixtures)


def _backward_loss(
    ddp_model: DistributedDataParallel,
    fixture: MicrobatchFixture,
    *,
    global_valid_token_count: int,
) -> None:
    logits = ddp_model(fixture.base_logits)
    loss_sum = F.cross_entropy(
        logits,
        fixture.targets,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    loss = loss_sum * WORLD_SIZE / global_valid_token_count
    # The installed PyTorch stubs do not type Tensor.backward().
    loss.backward()  # type: ignore[no-untyped-call]


def _run_case(
    fixtures: tuple[MicrobatchFixture, ...],
    *,
    case: CaseName,
    global_valid_token_count: int,
) -> dict[str, object]:
    model = SharedBiasClassifier()
    ddp_model = DistributedDataParallel(model)
    hook_state: AllReduceHookState | None = None
    if case != "builtin_no_sync":
        process_group = dist.group.WORLD
        if process_group is None:
            raise AssertionError("the default process group must be initialized")
        hook_state = AllReduceHookState(process_group)
        ddp_model.register_comm_hook(hook_state, _counting_allreduce_hook)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)

    for index, fixture in enumerate(fixtures):
        is_final = index == len(fixtures) - 1
        if not is_final and case != "counting_hook_backward_only":
            with ddp_model.no_sync():
                _backward_loss(
                    ddp_model,
                    fixture,
                    global_valid_token_count=global_valid_token_count,
                )
        elif not is_final:
            logits = ddp_model(fixture.base_logits)
            loss_sum = F.cross_entropy(
                logits,
                fixture.targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            loss = loss_sum * WORLD_SIZE / global_valid_token_count
            with ddp_model.no_sync():
                # This is deliberately too late: DDP saw sync=True in forward.
                loss.backward()  # type: ignore[no-untyped-call]
        else:
            _backward_loss(
                ddp_model,
                fixture,
                global_valid_token_count=global_valid_token_count,
            )

    if model.bias.grad is None:
        raise AssertionError("DDP accumulation did not populate the bias gradient")
    gradient_before_clip = model.bias.grad.detach().clone()
    returned_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=MAX_GRAD_NORM,
    )
    if model.bias.grad is None:
        raise AssertionError("gradient clipping removed the bias gradient")
    gradient_after_clip = model.bias.grad.detach().clone()
    optimizer.step()
    return {
        "gradient_before_clip": [
            float(value) for value in gradient_before_clip.tolist()
        ],
        "clip_grad_norm_returned_pre_clip_norm": float(returned_norm.item()),
        "gradient_after_clip": [
            float(value) for value in gradient_after_clip.tolist()
        ],
        "bias_after_sgd_step": [float(value) for value in model.bias.tolist()],
        "registered_reference_allreduce_hook": hook_state is not None,
        "reference_allreduce_hook_calls": (
            None if hook_state is None else hook_state.calls
        ),
        "reference_allreduce_hook_bucket_numel": (
            None if hook_state is None else hook_state.bucket_numel
        ),
    }


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
        fixtures = _rank_fixtures(rank)
        local_count = sum(item.valid_token_count for item in fixtures)
        count = torch.tensor([local_count], dtype=torch.int64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        global_count = int(count.item())
        cases: dict[str, object] = {}
        for case in CASE_NAMES:
            cases[case] = _run_case(
                fixtures,
                case=case,
                global_valid_token_count=global_count,
            )
            dist.barrier()
        report = {
            "rank": rank,
            "local_valid_token_counts": [
                item.valid_token_count for item in fixtures
            ],
            "global_valid_token_count_after_all_reduce": global_count,
            "cases": cases,
        }
        output_path = Path(output_directory) / f"rank-{rank}.json"
        output_path.write_text(
            json.dumps(report, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _full_batch_reference() -> dict[str, object]:
    fixtures = tuple(
        fixture
        for rank in range(WORLD_SIZE)
        for fixture in _rank_fixtures(rank)
    )
    model = SharedBiasClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
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
    gradient_before_clip = model.bias.grad.detach().clone()
    returned_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=MAX_GRAD_NORM,
    )
    if model.bias.grad is None:
        raise AssertionError("full-batch clipping removed the bias gradient")
    gradient_after_clip = model.bias.grad.detach().clone()
    optimizer.step()
    return {
        "gradient_before_clip": [
            float(value) for value in gradient_before_clip.tolist()
        ],
        "clip_grad_norm_returned_pre_clip_norm": float(returned_norm.item()),
        "gradient_after_clip": [
            float(value) for value in gradient_after_clip.tolist()
        ],
        "bias_after_sgd_step": [float(value) for value in model.bias.tolist()],
    }


def _max_abs_difference(left: list[float], right: list[float]) -> float:
    return max(
        abs(left_value - right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _require_float_list(payload: dict[str, object], field_name: str) -> list[float]:
    value = payload.get(field_name)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise AssertionError(f"{field_name} must be a finite numeric list")
    result = [float(item) for item in value]
    if not result or not all(math.isfinite(item) for item in result):
        raise AssertionError(f"{field_name} must be a finite numeric list")
    return result


def run_control() -> dict[str, object]:
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("torch.distributed with the Gloo backend is required")
    exact = analyze_default_ddp_gradient_accumulation(
        authored_rank_windows(),
        data_parallel_world_size=WORLD_SIZE,
        unclipped_sgd_learning_rate=LEARNING_RATE_FRACTION,
    )
    with tempfile.TemporaryDirectory(
        prefix="about-llm-ddp-accumulation-no-sync-"
    ) as temporary:
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
    first_cases = rank_reports[0]["cases"]
    if not isinstance(first_cases, dict):
        raise AssertionError("rank cases must be an object")
    if any(report["cases"] != first_cases for report in rank_reports[1:]):
        raise AssertionError("all ranks must observe identical synchronized outcomes")
    if not all(
        report["global_valid_token_count_after_all_reduce"] == 7
        for report in rank_reports
    ):
        raise AssertionError("global valid-token count must equal seven")

    comparisons: dict[str, dict[str, float]] = {}
    full_pre_clip = _require_float_list(full_batch, "gradient_before_clip")
    full_post_clip = _require_float_list(full_batch, "gradient_after_clip")
    full_post_step = _require_float_list(full_batch, "bias_after_sgd_step")
    for case_name, case_payload in first_cases.items():
        if not isinstance(case_name, str) or not isinstance(case_payload, dict):
            raise AssertionError("case payloads must be named objects")
        comparisons[case_name] = {
            "pre_clip_gradient_vs_full_max_abs_error": _max_abs_difference(
                _require_float_list(case_payload, "gradient_before_clip"),
                full_pre_clip,
            ),
            "post_clip_gradient_vs_full_max_abs_error": _max_abs_difference(
                _require_float_list(case_payload, "gradient_after_clip"),
                full_post_clip,
            ),
            "post_step_bias_vs_full_max_abs_error": _max_abs_difference(
                _require_float_list(case_payload, "bias_after_sgd_step"),
                full_post_step,
            ),
        }
    if any(
        value > 1e-15
        for comparison in comparisons.values()
        for value in comparison.values()
    ):
        raise AssertionError("DDP accumulation outcomes must match full batch")
    counting_no_sync = first_cases["counting_hook_no_sync"]
    backward_only = first_cases["counting_hook_backward_only"]
    if not isinstance(counting_no_sync, dict) or not isinstance(backward_only, dict):
        raise AssertionError("counting hook cases must be objects")
    if counting_no_sync["reference_allreduce_hook_calls"] != 1:
        raise AssertionError("correct no_sync scope must invoke one all-reduce hook")
    if backward_only["reference_allreduce_hook_calls"] != 2:
        raise AssertionError("backward-only no_sync must invoke two all-reduce hooks")

    return {
        "implementation": "about-llm.ddp-accumulation-no-sync-control.v1",
        "exact_oracle": exact.to_dict(),
        "runtime": {
            "torch_version": torch.__version__,
            "backend": "gloo",
            "device": "cpu",
            "dtype": "torch.float64",
            "world_size": WORLD_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "process_start_method": "spawn",
            "rendezvous": "temporary-file-store",
            "optimizer": "torch.optim.SGD",
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "rank_reports": rank_reports,
        "full_batch_reference": full_batch,
        "comparisons": comparisons,
        "observations": {
            "all_cases_match_full_pre_clip_gradient_within_1e_15": True,
            "all_cases_match_full_post_clip_gradient_within_1e_15": True,
            "all_cases_match_full_post_step_bias_within_1e_15": True,
            "correct_no_sync_scope_has_one_reference_hook_call": True,
            "backward_only_no_sync_scope_has_two_reference_hook_calls": True,
            "all_ranks_observe_identical_synchronized_outcomes": True,
            "global_valid_token_count_all_reduce_is_seven": True,
        },
        "scope": {
            "real_two_process_same_host_gloo_process_group_executed": True,
            "two_microbatches_per_rank_accumulated": True,
            "builtin_ddp_no_sync_forward_and_backward_scope_executed": True,
            "pytorch_reference_allreduce_hook_counting_control_executed": True,
            "backward_only_no_sync_negative_control_executed": True,
            "global_valid_token_count_all_reduce_executed": True,
            "gradient_clipping_after_synchronized_normalization_executed": True,
            "plain_sgd_optimizer_step_and_parameter_update_executed": True,
            "single_process_full_batch_reference_executed": True,
            "builtin_reducer_collective_count_directly_instrumented": False,
            "multiple_parameters_or_multiple_gradient_buckets_executed": False,
            "dropout_batchnorm_or_stochastic_rng_equivalence_executed": False,
            "amp_scaler_or_overflow_path_executed": False,
            "fsdp_zero_tensor_pipeline_expert_parallel_executed": False,
            "cuda_gpu_multi_node_or_remote_host_executed": False,
            "target_llm_tokenizer_dataset_trainer_or_quality_evaluation_executed": (
                False
            ),
            "optimizer_state_checkpoint_resume_or_failure_recovery_executed": False,
            "throughput_latency_memory_or_communication_bytes_measured": False,
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
