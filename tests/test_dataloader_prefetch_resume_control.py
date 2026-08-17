from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "dataloader_prefetch_resume_control.py"
)
MODULE = runpy.run_path(str(CONTROL), run_name="dataloader_prefetch_test_module")


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise AssertionError(f"non-standard JSON number: {value}")


def _valid_checkpoint_payload() -> dict[str, object]:
    checkpoint_payload = MODULE["_checkpoint_payload"]
    return checkpoint_payload(
        {
            "pid": 12345,
            "sampler_emitted_cursor_when_observed": 7,
            "records": [
                {"sample_id": 8},
                {"sample_id": 3},
                {"sample_id": 1},
            ],
        }
    )


def test_prefetch_checkpoint_round_trip_is_closed_canonical_json(
    tmp_path: Path,
) -> None:
    write_checkpoint = MODULE["_write_checkpoint"]
    load_checkpoint = MODULE["_load_checkpoint"]
    checkpoint = tmp_path / "prefetch.json"

    size = write_checkpoint(checkpoint, _valid_checkpoint_payload())
    loaded, loaded_size = load_checkpoint(checkpoint)

    assert loaded == _valid_checkpoint_payload()
    assert loaded_size == size == checkpoint.stat().st_size
    assert size < 64 * 1024
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_checkpoint(checkpoint, _valid_checkpoint_payload())


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ('{"field":1,"field":2}', "duplicate JSON key"),
        ('{"field":NaN}', "non-finite JSON number"),
    ],
)
def test_strict_json_parser_rejects_duplicate_and_nonfinite(
    payload: str,
    match: str,
) -> None:
    parse_strict_json = MODULE["_parse_strict_json"]
    with pytest.raises(ValueError, match=match):
        parse_strict_json(payload)


def test_prefetch_checkpoint_rejects_noncanonical_and_cursor_drift(
    tmp_path: Path,
) -> None:
    load_checkpoint = MODULE["_load_checkpoint"]
    encode_canonical = MODULE["_encode_canonical"]
    payload = _valid_checkpoint_payload()

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_checkpoint(noncanonical)

    payload["sampler_emitted_cursor"] = 3
    invalid_cursor = tmp_path / "cursor.json"
    invalid_cursor.write_text(encode_canonical(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prefetch-ahead"):
        load_checkpoint(invalid_cursor)


@pytest.mark.slow
def test_real_dataloader_prefetch_resume_control() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
    )
    report = json.loads(
        completed.stdout,
        parse_constant=_reject_nonfinite_json,
    )

    assert completed.stderr == ""
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
    assert report["implementation"] == (
        "about-llm.dataloader-prefetch-resume-control.v1"
    )
    assert report["runtime"] == {
        "batch_size": 1,
        "dataloader": "torch.utils.data.DataLoader",
        "dataset_kind": "torch.utils.data.Dataset-map-style",
        "device": "cpu",
        "in_order": True,
        "loader_generator_seed": 20260814,
        "multiprocessing_context": "spawn",
        "num_workers": 2,
        "persistent_workers": False,
        "pin_memory": False,
        "prefetch_factor": 2,
        "torch_version": report["runtime"]["torch_version"],
    }
    assert report["fixture"] == {
        "dataset_identity": (
            "sha256:715db0ba19cc381b9b706969ea4fe8d3b5d8115dba00d9448500db4491ceb057"
        ),
        "dataset_size": 10,
        "permutation": [8, 3, 1, 7, 0, 9, 4, 2, 6, 5],
        "prefetched_but_uncommitted_sample_ids": [7, 0, 9, 4],
        "sample_key_namespace": "about-llm.sample-keyed-randomness.v1",
        "sampler_emitted_cursor_at_split": 7,
        "split_committed_cursor": 3,
    }
    assert report["checkpoint"]["implementation"] == (
        "about-llm.dataloader-prefetch-checkpoint.v1"
    )
    assert report["checkpoint"]["canonical_strict_json"] is True
    assert 0 < report["checkpoint"]["size_bytes"] < 64 * 1024
    assert report["checkpoint"]["fields"] == [
        "committed_cursor",
        "consumed_sample_ids",
        "dataset_identity",
        "implementation",
        "loader_contract",
        "permutation",
        "phase_pid",
        "sampler_emitted_cursor",
    ]
    assert report["processes"]["all_distinct"] is True
    assert len(
        {
            report["processes"]["uninterrupted_pid"],
            report["processes"]["phase1_pid"],
            report["processes"]["resume_committed_pid"],
            report["processes"]["resume_emitted_pid"],
        }
    ) == 4
    assert report["comparisons"] == {
        "committed_resume_combined_sample_ids": [
            8,
            3,
            1,
            7,
            0,
            9,
            4,
            2,
            6,
            5,
        ],
        "emitted_resume_combined_sample_ids": [8, 3, 1, 2, 6, 5],
        "sample_keyed_tail_max_abs_difference": 0.0,
        "uninterrupted_sample_ids": [8, 3, 1, 7, 0, 9, 4, 2, 6, 5],
        "worker_rng_tail_max_abs_difference": pytest.approx(
            0.6544313251137845
        ),
    }
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "arbitrary_stochastic_transform_exact_resume_proved": False,
        "checkpoint_crash_power_loss_atomicity_or_authentication_proved": False,
        "committed_sample_cursor_order_resume_executed": True,
        "cross_pid_checkpoint_and_resume_executed": True,
        "cuda_gpu_or_target_trainer_dataset_executed": False,
        "distributed_sampler_ddp_fsdp_zero_or_sharded_state_executed": False,
        "multi_epoch_or_repeated_sample_randomness_policy_executed": False,
        "num_workers_two_and_prefetch_factor_two_executed": True,
        "optimizer_scheduler_scaler_or_model_training_executed": False,
        "persistent_workers_pin_memory_or_iterable_dataset_executed": False,
        "prefetch_depth_as_public_stable_api_contract_proved": False,
        "prefetched_queue_payload_or_worker_process_state_checkpointed": False,
        "private_dataloader_queue_fields_read_or_mutated": False,
        "real_dataloader_worker_processes_executed": True,
        "sample_consumption_and_optimizer_commit_atomicity_proved": False,
        "sample_keyed_stateless_randomness_exact_replay_executed": True,
        "sampler_emitted_cursor_skip_negative_control_executed": True,
        "sampler_prefetch_ahead_of_committed_consumption_observed": True,
        "strict_canonical_json_checkpoint_executed": True,
        "throughput_memory_quality_or_convergence_proved": False,
        "worker_local_rng_state_restored": False,
        "worker_local_torch_rng_nonreplay_observed": True,
    }
