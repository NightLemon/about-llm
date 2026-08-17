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
    analyze_default_ddp_token_mean,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "ddp_token_mean_control.py"
)


def rank_shards() -> tuple[CategoricalMicrobatch, ...]:
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


def test_exact_default_ddp_oracle_separates_three_scaling_paths() -> None:
    analysis = analyze_default_ddp_token_mean(
        rank_shards(),
        data_parallel_world_size=2,
    )

    assert analysis.valid_token_count == 4
    assert analysis.ignored_token_count == 3
    assert analysis.valid_token_counts_by_rank == (1, 3)
    assert analysis.correct_local_loss_sum_scale == Fraction(1, 2)
    assert analysis.missing_world_size_local_loss_sum_scale == Fraction(1, 4)
    assert analysis.rank_local_sum_class_aggregate_logit_gradients == (
        (Fraction(-1, 10), Fraction(1, 10)),
        (Fraction(12, 5), Fraction(-12, 5)),
    )
    assert analysis.full_batch_class_aggregate_logit_gradient == (
        Fraction(23, 40),
        Fraction(-23, 40),
    )
    assert (
        analysis.correctly_scaled_default_ddp_class_aggregate_logit_gradient
        == analysis.full_batch_class_aggregate_logit_gradient
    )
    assert (
        analysis.missing_world_size_default_ddp_class_aggregate_logit_gradient
        == (Fraction(23, 80), Fraction(-23, 80))
    )
    assert analysis.equal_rank_local_mean_class_aggregate_logit_gradient == (
        Fraction(7, 20),
        Fraction(-7, 20),
    )


def test_exact_default_ddp_report_preserves_fraction_payloads() -> None:
    payload = analyze_default_ddp_token_mean(
        rank_shards(),
        data_parallel_world_size=2,
    ).to_dict()

    assert payload["correct_local_loss_sum_scale"] == {
        "numerator": 1,
        "denominator": 2,
        "decimal": 0.5,
    }
    assert payload["missing_world_size_default_ddp_class_aggregate_logit_gradient"] == [
        {"numerator": 23, "denominator": 80, "decimal": 0.2875},
        {"numerator": -23, "denominator": 80, "decimal": -0.2875},
    ]


@pytest.mark.parametrize("world_size", [True, 1, 257])
def test_exact_default_ddp_oracle_rejects_invalid_world_size(
    world_size: object,
) -> None:
    with pytest.raises(ValueError, match="data_parallel_world_size"):
        analyze_default_ddp_token_mean(
            rank_shards(),
            data_parallel_world_size=world_size,  # type: ignore[arg-type]
        )


def test_exact_default_ddp_oracle_requires_one_shard_per_rank() -> None:
    with pytest.raises(ValueError, match="exactly one shard"):
        analyze_default_ddp_token_mean(
            rank_shards()[:1],
            data_parallel_world_size=2,
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_two_process_gloo_control_executes_real_default_ddp_reduction() -> None:
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
    assert report["implementation"] == "about-llm.ddp-token-mean-control.v1"
    assert report["runtime"] == {
        "backend": "gloo",
        "device": "cpu",
        "dtype": "torch.float64",
        "process_start_method": "spawn",
        "rendezvous": "temporary-file-store",
        "torch_version": report["runtime"]["torch_version"],
        "world_size": 2,
    }
    assert [item["local_valid_token_count"] for item in report["rank_reports"]] == [
        1,
        3,
    ]
    assert all(
        item["global_valid_token_count_after_all_reduce"] == 4
        for item in report["rank_reports"]
    )
    observed = report["observed_default_ddp_gradients"]
    assert observed["correct_d_over_n"] == pytest.approx([0.575, -0.575])
    assert observed["missing_world_size"] == pytest.approx([0.2875, -0.2875])
    assert observed["rank_local_mean"] == pytest.approx([0.35, -0.35])
    assert report["full_batch_shared_bias_gradient"] == pytest.approx(
        [0.575, -0.575]
    )
    assert report["comparison"] == {
        "correct_d_over_n_vs_full_max_abs_error": pytest.approx(0, abs=1e-15),
        "missing_world_size_vs_full_divided_by_d_max_abs_error": pytest.approx(
            0, abs=1e-15
        ),
        "rank_local_mean_vs_full_max_abs_difference": pytest.approx(0.225),
    }
    assert all(report["observations"].values())
    assert report["scope"] == {
        "bitwise_equivalence_across_hardware_or_world_sizes_proved": False,
        "cuda_gpu_multi_node_or_remote_host_executed": False,
        "default_ddp_gradient_averaging_observed": True,
        "fsdp_zero_tensor_pipeline_expert_parallel_executed": False,
        "global_valid_token_count_all_reduce_executed": True,
        "gradient_accumulation_no_sync_amp_or_scaler_executed": False,
        "optimizer_step_parameter_update_or_gradient_clipping_executed": False,
        "real_two_process_same_host_gloo_process_group_executed": True,
        "target_llm_tokenizer_dataset_or_quality_evaluation_executed": False,
        "temporary_file_store_rendezvous_executed": True,
        "transport_security_packet_capture_or_fault_injection_executed": False,
    }
