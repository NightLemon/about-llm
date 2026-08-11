from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from about_llm.continual_learning import (
    reservoir_sample_indices,
    summarize_accuracy_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "single-gpu-finetuning" / "continual_replay_toy.py"


def test_accuracy_matrix_metrics_have_explicit_indexing_and_direction() -> None:
    report = summarize_accuracy_matrix(
        (
            (0.80, 0.20, 0.30),
            (0.75, 0.70, 0.40),
            (0.90, 0.60, 0.85),
        ),
        pretraining_baseline=(0.50, 0.30, 0.25),
    )

    assert report.diagonal_accuracy == pytest.approx((0.80, 0.70, 0.85))
    assert report.final_average_accuracy == pytest.approx(2.35 / 3)
    assert report.backward_transfer == pytest.approx(0.0)
    assert report.per_task_forgetting == pytest.approx((0.0, 0.10, 0.0))
    assert report.average_forgetting_old_tasks == pytest.approx(0.05)
    assert report.forward_transfer == pytest.approx(0.025)
    assert report.to_dict()["scope"] == {
        "metric_direction": "higher_accuracy_is_better",
        "future_tasks_evaluated_before_training": True,
        "last_task_forgetting_defined_as_zero": True,
        "confidence_intervals_computed": False,
    }


@pytest.mark.parametrize(
    ("matrix", "baseline"),
    [
        ([(0.5,)], [0.5]),
        ([(0.5, 0.4), (0.3,)], [0.5, 0.5]),
        ([(0.5, 0.4), (0.3, 0.6)], [0.5]),
        ([(float("nan"), 0.4), (0.3, 0.6)], [0.5, 0.5]),
        ([(0.5, 0.4), (0.3, 0.6)], [0.5, float("inf")]),
        ([(0.5, 1.01), (0.3, 0.6)], [0.5, 0.5]),
        ([(False, 0.4), (0.3, 0.6)], [0.5, 0.5]),
    ],
)
def test_accuracy_matrix_metrics_reject_ambiguous_or_invalid_inputs(
    matrix: list[tuple[float, ...]], baseline: list[float]
) -> None:
    with pytest.raises(ValueError):
        summarize_accuracy_matrix(matrix, pretraining_baseline=baseline)


def test_reservoir_sample_is_reproducible_unique_and_bounded() -> None:
    first = reservoir_sample_indices(20, 5, seed=7)
    second = reservoir_sample_indices(20, 5, seed=7)

    assert first == second
    assert first == (3, 4, 14, 16, 17)
    assert first == tuple(sorted(first))
    assert len(first) == len(set(first)) == 5
    assert all(0 <= index < 20 for index in first)
    assert reservoir_sample_indices(20, 0, seed=7) == ()
    assert reservoir_sample_indices(5, 5, seed=7) == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    ("stream_length", "capacity", "seed"),
    [
        (-1, 0, 0),
        (1, -1, 0),
        (1, 2, 0),
        (True, 0, 0),
        (1, False, 0),
        (1, 0, True),
    ],
)
def test_reservoir_sample_rejects_ambiguous_inputs(
    stream_length: int, capacity: int, seed: int
) -> None:
    with pytest.raises(ValueError):
        reservoir_sample_indices(stream_length, capacity, seed=seed)


def test_replay_toy_emits_machine_readable_bounded_evidence() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    no_replay = payload["no_replay"]
    full_replay = payload["full_replay"]

    assert payload["experiment"] == "about-llm.task-incremental-replay-toy.v1"
    assert no_replay["per_task_forgetting"][0] > 0.9
    assert full_replay["per_task_forgetting"][0] == pytest.approx(0.0)
    assert no_replay["accuracy_matrix"][-1][1] > 0.99
    assert full_replay["accuracy_matrix"][-1] == pytest.approx((1.0, 1.0))
    assert no_replay["forward_transfer"] == pytest.approx(
        full_replay["forward_transfer"]
    )
    assert payload["scope"] == {
        "actual_gradient_updates": True,
        "full_batch_deterministic_cpu_fixture": True,
        "replay_uses_all_old_examples": True,
        "multiple_seeds_or_confidence_intervals": False,
        "language_model_or_real_corpus": False,
        "privacy_retention_cost_modeled": False,
        "replay_always_prevents_forgetting": False,
    }


def test_finite_replay_benchmark_exposes_paired_uncertainty_and_cost() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark",
            "--seed-count",
            "4",
            "--bootstrap-samples",
            "500",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)

    assert payload["experiment"] == "about-llm.task-incremental-replay-benchmark.v1"
    assert len(payload["runs"]) == 4
    first = payload["runs"][0]["strategies"]
    assert first["no_replay"]["total_examples_per_task_b_step"] == 256
    assert first["finite_reservoir"]["unique_old_examples"] == 64
    assert first["finite_reservoir"]["old_fraction_per_task_b_step"] == 0.2
    assert len(first["finite_reservoir"]["selected_old_indices"]) == 64
    assert first["full_replay"]["selected_old_indices"] is None
    assert first["full_replay"]["total_examples_per_task_b_step"] == 512
    assert payload["cost_contract"] == {
        "task_b_optimizer_steps_matched": True,
        "task_b_total_example_presentations_matched": False,
        "wall_time_or_energy_measured": False,
    }
    retention = payload["paired_vs_no_replay"]["full_replay"][
        "old_task_final_accuracy_gain"
    ]
    assert retention["mean_difference"] > 0.7
    assert retention["confidence_low"] > 0.5
    assert payload["aggregate"]["full_replay"]["mean_new_task_final_accuracy"] > 0.95
    assert payload["scope"]["task_or_data_distribution_resampled_across_seeds"] is False
    assert payload["scope"]["confidence_interval_covers_task_or_data_uncertainty"] is False
