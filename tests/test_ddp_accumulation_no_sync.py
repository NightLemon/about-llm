from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch.distributed as dist

from about_llm.finetuning import (
    CategoricalMicrobatch,
    CategoricalTokenRecord,
    DDPGradientAccumulationAnalysis,
    analyze_default_ddp_gradient_accumulation,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "ddp_accumulation_no_sync_control.py"
)


def rank_windows() -> tuple[tuple[CategoricalMicrobatch, ...], ...]:
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


def test_exact_ddp_accumulation_oracle_matches_full_gradient_and_sgd_delta() -> None:
    analysis = analyze_default_ddp_gradient_accumulation(
        rank_windows(),
        data_parallel_world_size=2,
        unclipped_sgd_learning_rate=Fraction(7, 20),
    )

    assert isinstance(analysis, DDPGradientAccumulationAnalysis)
    assert analysis.data_parallel_world_size == 2
    assert analysis.accumulation_steps == 2
    assert analysis.valid_token_count == 7
    assert analysis.ignored_token_count == 4
    assert analysis.valid_token_counts_by_rank_and_microbatch == ((1, 2), (3, 1))
    assert analysis.valid_token_counts_by_rank == (3, 4)
    assert analysis.correct_local_loss_sum_scale == Fraction(2, 7)
    assert analysis.rank_microbatch_loss_sum_class_aggregate_logit_gradients == (
        (
            (Fraction(-1, 10), Fraction(1, 10)),
            (Fraction(8, 5), Fraction(-8, 5)),
        ),
        (
            (Fraction(12, 5), Fraction(-12, 5)),
            (Fraction(-1, 10), Fraction(1, 10)),
        ),
    )
    assert analysis.rank_accumulated_loss_sum_class_aggregate_logit_gradients == (
        (Fraction(3, 2), Fraction(-3, 2)),
        (Fraction(23, 10), Fraction(-23, 10)),
    )
    assert analysis.full_batch_class_aggregate_logit_gradient == (
        Fraction(19, 35),
        Fraction(-19, 35),
    )
    assert (
        analysis.one_sync_after_accumulation_class_aggregate_logit_gradient
        == analysis.full_batch_class_aggregate_logit_gradient
    )
    assert (
        analysis.sync_every_microbatch_class_aggregate_logit_gradient
        == analysis.full_batch_class_aggregate_logit_gradient
    )
    assert analysis.unclipped_sgd_parameter_delta == (
        Fraction(-19, 100),
        Fraction(19, 100),
    )


def test_exact_ddp_accumulation_report_preserves_nested_fractions() -> None:
    payload = analyze_default_ddp_gradient_accumulation(
        rank_windows(),
        data_parallel_world_size=2,
        unclipped_sgd_learning_rate=Fraction(7, 20),
    ).to_dict()

    assert payload["valid_token_counts_by_rank_and_microbatch"] == [[1, 2], [3, 1]]
    assert payload["correct_local_loss_sum_scale"] == {
        "numerator": 2,
        "denominator": 7,
        "decimal": pytest.approx(2 / 7),
    }
    assert payload["unclipped_sgd_parameter_delta"] == [
        {"numerator": -19, "denominator": 100, "decimal": -0.19},
        {"numerator": 19, "denominator": 100, "decimal": 0.19},
    ]


@pytest.mark.parametrize("world_size", [True, 1, 257])
def test_exact_ddp_accumulation_rejects_invalid_world_size(
    world_size: object,
) -> None:
    with pytest.raises(ValueError, match="data_parallel_world_size"):
        analyze_default_ddp_gradient_accumulation(
            rank_windows(),
            data_parallel_world_size=world_size,  # type: ignore[arg-type]
            unclipped_sgd_learning_rate=Fraction(7, 20),
        )


@pytest.mark.parametrize("learning_rate", [Fraction(0), Fraction(-1), 1])
def test_exact_ddp_accumulation_rejects_invalid_learning_rate(
    learning_rate: object,
) -> None:
    with pytest.raises(ValueError, match="unclipped_sgd_learning_rate"):
        analyze_default_ddp_gradient_accumulation(
            rank_windows(),
            data_parallel_world_size=2,
            unclipped_sgd_learning_rate=learning_rate,  # type: ignore[arg-type]
        )


def test_exact_ddp_accumulation_requires_one_window_per_rank() -> None:
    with pytest.raises(ValueError, match="one accumulation window per rank"):
        analyze_default_ddp_gradient_accumulation(
            rank_windows()[:1],
            data_parallel_world_size=2,
            unclipped_sgd_learning_rate=Fraction(7, 20),
        )


def test_exact_ddp_accumulation_requires_equal_nonempty_step_counts() -> None:
    windows = rank_windows()
    with pytest.raises(ValueError, match="same accumulation step count"):
        analyze_default_ddp_gradient_accumulation(
            (windows[0], windows[1][:1]),
            data_parallel_world_size=2,
            unclipped_sgd_learning_rate=Fraction(7, 20),
        )
    with pytest.raises(ValueError, match="at least one micro-batch"):
        analyze_default_ddp_gradient_accumulation(
            ((), ()),
            data_parallel_world_size=2,
            unclipped_sgd_learning_rate=Fraction(7, 20),
        )


def test_exact_ddp_accumulation_caps_total_rank_microbatches() -> None:
    microbatch = rank_windows()[0][0]
    with pytest.raises(ValueError, match="total rank micro-batch count"):
        analyze_default_ddp_gradient_accumulation(
            ((microbatch,) * 129, (microbatch,) * 129),
            data_parallel_world_size=2,
            unclipped_sgd_learning_rate=Fraction(7, 20),
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_process_no_sync_control_executes_accumulation_clip_and_sgd() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    report = json.loads(completed.stdout)

    assert completed.stderr == ""
    assert report["implementation"] == (
        "about-llm.ddp-accumulation-no-sync-control.v1"
    )
    assert report["runtime"] == {
        "accumulation_steps": 2,
        "backend": "gloo",
        "device": "cpu",
        "dtype": "torch.float64",
        "learning_rate": 0.35,
        "max_grad_norm": 0.5,
        "optimizer": "torch.optim.SGD",
        "process_start_method": "spawn",
        "rendezvous": "temporary-file-store",
        "torch_version": report["runtime"]["torch_version"],
        "world_size": 2,
    }
    assert [
        rank["local_valid_token_counts"] for rank in report["rank_reports"]
    ] == [[1, 2], [3, 1]]
    assert all(
        rank["global_valid_token_count_after_all_reduce"] == 7
        for rank in report["rank_reports"]
    )
    full = report["full_batch_reference"]
    assert full["gradient_before_clip"] == pytest.approx([19 / 35, -19 / 35])
    assert full["clip_grad_norm_returned_pre_clip_norm"] == pytest.approx(
        0.76771593385968
    )
    assert full["gradient_after_clip"] == pytest.approx(
        [0.35355293006755784, -0.3535529300675579]
    )
    assert full["bias_after_sgd_step"] == pytest.approx(
        [-0.12374352552364523, 0.12374352552364526]
    )
    cases = report["rank_reports"][0]["cases"]
    assert cases["builtin_no_sync"]["reference_allreduce_hook_calls"] is None
    assert cases["counting_hook_no_sync"]["reference_allreduce_hook_calls"] == 1
    assert cases["counting_hook_no_sync"][
        "reference_allreduce_hook_bucket_numel"
    ] == [2]
    assert cases["counting_hook_backward_only"][
        "reference_allreduce_hook_calls"
    ] == 2
    assert cases["counting_hook_backward_only"][
        "reference_allreduce_hook_bucket_numel"
    ] == [2, 2]
    assert all(
        value == pytest.approx(0, abs=1e-15)
        for case in report["comparisons"].values()
        for value in case.values()
    )
    assert all(report["observations"].values())
    assert report["scope"] == {
        "amp_scaler_or_overflow_path_executed": False,
        "backward_only_no_sync_negative_control_executed": True,
        "bitwise_equivalence_across_hardware_or_world_sizes_proved": False,
        "builtin_ddp_no_sync_forward_and_backward_scope_executed": True,
        "builtin_reducer_collective_count_directly_instrumented": False,
        "cuda_gpu_multi_node_or_remote_host_executed": False,
        "dropout_batchnorm_or_stochastic_rng_equivalence_executed": False,
        "fsdp_zero_tensor_pipeline_expert_parallel_executed": False,
        "global_valid_token_count_all_reduce_executed": True,
        "gradient_clipping_after_synchronized_normalization_executed": True,
        "multiple_parameters_or_multiple_gradient_buckets_executed": False,
        "optimizer_state_checkpoint_resume_or_failure_recovery_executed": False,
        "plain_sgd_optimizer_step_and_parameter_update_executed": True,
        "pytorch_reference_allreduce_hook_counting_control_executed": True,
        "real_two_process_same_host_gloo_process_group_executed": True,
        "single_process_full_batch_reference_executed": True,
        "target_llm_tokenizer_dataset_trainer_or_quality_evaluation_executed": False,
        "throughput_latency_memory_or_communication_bytes_measured": False,
        "transport_security_packet_capture_or_fault_injection_executed": False,
        "two_microbatches_per_rank_accumulated": True,
    }
