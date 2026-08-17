from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "optimizer_commit_resume_control.py"
)
MODULE = runpy.run_path(str(CONTROL), run_name="optimizer_commit_test_module")


def _reject_nonfinite(value: str) -> NoReturn:
    raise AssertionError(f"non-standard JSON number: {value}")


def _validation_payload() -> dict[str, object]:
    state = MODULE["_new_state"]()
    for parameter in state.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    state.optimizer.step()
    state.scheduler.step()
    state.optimizer.zero_grad(set_to_none=True)
    return {
        "schema_version": MODULE["CHECKPOINT_VERSION"],
        "dataset_identity": MODULE["_dataset_identity"](),
        "permutation": list(MODULE["PERMUTATION"]),
        "loader_contract": MODULE["_loader_contract"](),
        "progress": {
            "optimizer_committed_cursor": 2,
            "main_loop_consumed_cursor": 3,
            "sampler_emitted_cursor": 7,
            "optimizer_steps": 1,
            "committed_sample_ids": [8, 3],
            "consumed_sample_ids": [8, 3, 1],
            "in_flight_gradients_serialized": False,
        },
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "commit_boundary_torch_rng_state": torch.get_rng_state().clone(),
    }


def _write_inflight_bundle_fixture(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint.pt"
    base_size, base_digest = MODULE["_write_checkpoint"](
        checkpoint,
        _validation_payload(),
    )
    phase_state = MODULE["_new_state"]()
    phase_state.optimizer_steps = 1
    phase_state.committed_sample_ids = [8, 3]
    for index, parameter in enumerate(phase_state.model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index) / 8.0)
    segment = {
        "consumed_sample_ids": [8, 3, 1],
        "sampler_emitted_cursor_when_observed": 7,
        "pending_uncommitted_sample_ids": [1],
    }
    sidecar = MODULE["_inflight_sidecar_path"](checkpoint)
    sidecar_size, sidecar_digest = MODULE["_write_checkpoint"](
        sidecar,
        MODULE["_inflight_sidecar_payload"](
            phase_state,
            segment,
            base_checkpoint_sha256=base_digest,
        ),
    )
    manifest = MODULE["_inflight_bundle_manifest_path"](checkpoint)
    MODULE["_write_canonical_json"](
        manifest,
        MODULE["_inflight_bundle_manifest_payload"](
            checkpoint,
            base_size_bytes=base_size,
            base_sha256=base_digest,
            sidecar_size_bytes=sidecar_size,
            sidecar_sha256=sidecar_digest,
        ),
    )
    return checkpoint


def _copy_bundle_payloads(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    source_sidecar = MODULE["_inflight_sidecar_path"](source)
    target_sidecar = MODULE["_inflight_sidecar_path"](target)
    target_sidecar.write_bytes(source_sidecar.read_bytes())


def test_checkpoint_rejects_consumed_cursor_as_committed_boundary(
    tmp_path: Path,
) -> None:
    payload = _validation_payload()
    progress = payload["progress"]
    assert isinstance(progress, dict)
    progress["optimizer_committed_cursor"] = 3
    checkpoint = tmp_path / "drifted.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=r"cursor ordering|committed cursor"):
        MODULE["_load_checkpoint"](checkpoint)


def test_checkpoint_rejects_loader_contract_value_drift(tmp_path: Path) -> None:
    payload = _validation_payload()
    contract = payload["loader_contract"]
    assert isinstance(contract, dict)
    contract["prefetch_factor"] = 99
    checkpoint = tmp_path / "loader-drift.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="loader contract values drifted"):
        MODULE["_load_checkpoint"](checkpoint)


def test_checkpoint_rejects_nonfinite_model_and_missing_momentum(
    tmp_path: Path,
) -> None:
    nonfinite_payload = _validation_payload()
    model = nonfinite_payload["model"]
    assert isinstance(model, dict)
    model["bias"] = torch.tensor([float("nan")], dtype=torch.float64)
    nonfinite = tmp_path / "nonfinite.pt"
    torch.save(nonfinite_payload, nonfinite)
    with pytest.raises(ValueError, match="model tensor bias must be finite"):
        MODULE["_load_checkpoint"](nonfinite)

    missing_momentum_payload = _validation_payload()
    missing_optimizer = missing_momentum_payload["optimizer"]
    assert isinstance(missing_optimizer, dict)
    missing_optimizer["state"] = {}
    missing_momentum = tmp_path / "missing-momentum.pt"
    torch.save(missing_momentum_payload, missing_momentum)
    with pytest.raises(ValueError, match="optimizer state parameter IDs drifted"):
        MODULE["_load_checkpoint"](missing_momentum)

    scheduler_drift_payload = _validation_payload()
    scheduler = scheduler_drift_payload["scheduler"]
    assert isinstance(scheduler, dict)
    scheduler["gamma"] = 0.75
    scheduler_drift = tmp_path / "scheduler-drift.pt"
    torch.save(scheduler_drift_payload, scheduler_drift)
    with pytest.raises(ValueError, match="scheduler gamma contract drifted"):
        MODULE["_load_checkpoint"](scheduler_drift)

    rng_drift_payload = _validation_payload()
    rng_drift_payload["commit_boundary_torch_rng_state"] = torch.zeros(
        4, dtype=torch.float64
    )
    rng_drift = tmp_path / "commit-rng-drift.pt"
    torch.save(rng_drift_payload, rng_drift)
    with pytest.raises(ValueError, match="Torch RNG dtype or shape drifted"):
        MODULE["_load_checkpoint"](rng_drift)


def test_checkpoint_writer_refuses_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    MODULE["_write_checkpoint"](checkpoint, {"field": torch.tensor([1.0])})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE["_write_checkpoint"](
            checkpoint,
            {"field": torch.tensor([2.0])},
        )


def test_inflight_sidecar_round_trip_and_base_digest_binding(
    tmp_path: Path,
) -> None:
    phase_state = MODULE["_new_state"]()
    phase_state.optimizer_steps = 1
    phase_state.committed_sample_ids = [8, 3]
    for index, parameter in enumerate(phase_state.model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index) / 8.0)
    base_digest = "sha256:" + "a" * 64
    torch.manual_seed(811)
    expected_crash_rng = torch.get_rng_state().clone()
    payload = MODULE["_inflight_sidecar_payload"](
        phase_state,
        {
            "consumed_sample_ids": [8, 3, 1],
            "sampler_emitted_cursor_when_observed": 7,
            "pending_uncommitted_sample_ids": [1],
        },
        base_checkpoint_sha256=base_digest,
    )
    sidecar = tmp_path / "inflight.pt"
    MODULE["_write_checkpoint"](sidecar, payload)
    torch.manual_seed(977)
    loaded_state = MODULE["_new_state"]()
    base_progress = _validation_payload()["progress"]
    assert isinstance(base_progress, dict)

    pending_ids, progress, size, digest = MODULE["_load_inflight_sidecar"](
        sidecar,
        expected_base_sha256=base_digest,
        base_progress=base_progress,
        state=loaded_state,
    )

    assert pending_ids == [1]
    assert progress["accumulation_position"] == 1
    assert progress["loss_divisor"] == 2
    assert 0 < size < 4 * 1024 * 1024
    assert digest.startswith("sha256:")
    assert torch.equal(torch.get_rng_state(), expected_crash_rng)
    assert all(
        parameter.grad is not None for parameter in loaded_state.model.parameters()
    )
    with pytest.raises(ValueError, match="base checkpoint digest drifted"):
        MODULE["_load_inflight_sidecar"](
            sidecar,
            expected_base_sha256="sha256:" + "b" * 64,
            base_progress=base_progress,
            state=MODULE["_new_state"](),
        )

    drifted_payload = dict(payload)
    drifted_payload["crash_observed_torch_rng_state"] = torch.zeros(
        4, dtype=torch.float64
    )
    drifted_sidecar = tmp_path / "inflight-rng-drift.pt"
    MODULE["_write_checkpoint"](drifted_sidecar, drifted_payload)
    with pytest.raises(ValueError, match="Torch RNG dtype or shape drifted"):
        MODULE["_load_inflight_sidecar"](
            drifted_sidecar,
            expected_base_sha256=base_digest,
            base_progress=base_progress,
            state=MODULE["_new_state"](),
        )


def test_inflight_bundle_manifest_round_trip_and_fault_injection(
    tmp_path: Path,
) -> None:
    checkpoint = _write_inflight_bundle_fixture(tmp_path)
    manifest, size, digest = MODULE["_load_inflight_bundle_manifest"](
        checkpoint
    )

    assert manifest["schema_version"] == (
        "about-llm.optimizer-commit-bundle-manifest.v1"
    )
    assert manifest["publication_state"] == "complete"
    assert manifest["publication_sequence"] == [
        "base_checkpoint",
        "inflight_gradient_sidecar",
        "bundle_manifest",
    ]
    assert 0 < size < 16 * 1024
    assert digest.startswith("sha256:")
    fault_report = MODULE["_run_bundle_publication_fault_injection"](
        checkpoint
    )
    assert all(
        fault_report[field] is True
        for field in (
            "complete_bundle_accepted",
            "base_only_rejected",
            "base_and_sidecar_without_manifest_rejected",
            "manifest_without_sidecar_rejected",
            "tampered_sidecar_after_manifest_rejected",
        )
    )


def test_inflight_bundle_manifest_rejects_noncanonical_duplicate_and_unknown(
    tmp_path: Path,
) -> None:
    source = _write_inflight_bundle_fixture(tmp_path / "source")
    source_manifest = MODULE["_inflight_bundle_manifest_path"](source)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))

    noncanonical = tmp_path / "noncanonical" / "checkpoint.pt"
    _copy_bundle_payloads(source, noncanonical)
    noncanonical_manifest = MODULE["_inflight_bundle_manifest_path"](
        noncanonical
    )
    noncanonical_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must use canonical JSON"):
        MODULE["_load_inflight_bundle_manifest"](noncanonical)

    duplicate = tmp_path / "duplicate" / "checkpoint.pt"
    _copy_bundle_payloads(source, duplicate)
    duplicate_manifest = MODULE["_inflight_bundle_manifest_path"](duplicate)
    canonical = source_manifest.read_bytes()
    duplicate_manifest.write_bytes(
        canonical.replace(
            b'{"artifacts"',
            b'{"schema_version":"duplicate","artifacts"',
            1,
        )
    )
    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        MODULE["_load_inflight_bundle_manifest"](duplicate)

    unknown = tmp_path / "unknown" / "checkpoint.pt"
    _copy_bundle_payloads(source, unknown)
    unknown_manifest = MODULE["_inflight_bundle_manifest_path"](unknown)
    payload["unexpected"] = True
    MODULE["_write_canonical_json"](unknown_manifest, payload)
    with pytest.raises(ValueError, match="manifest fields drifted"):
        MODULE["_load_inflight_bundle_manifest"](unknown)

    with pytest.raises(FileExistsError, match="refusing to overwrite JSON"):
        MODULE["_write_canonical_json"](source_manifest, payload)


@pytest.mark.slow
def test_real_optimizer_commit_resume_control() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=480,
    )
    report = json.loads(
        completed.stdout,
        parse_constant=_reject_nonfinite,
    )

    assert completed.stderr == ""
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
    assert report["implementation"] == (
        "about-llm.optimizer-commit-resume-control.v1"
    )
    assert report["runtime"] == {
        "accumulation_steps": 2,
        "device": "cpu",
        "dtype": "torch.float64",
        "in_order": True,
        "loss": "torch.nn.functional.mse_loss",
        "model": "torch.nn.Linear(2,1)",
        "multiprocessing_context": "spawn",
        "num_workers": 2,
        "optimizer": "torch.optim.SGD(momentum=0.9)",
        "prefetch_factor": 2,
        "scheduler": "torch.optim.lr_scheduler.StepLR(step_size=2,gamma=0.5)",
        "stochastic_forward": "main-process inverted Bernoulli mask p=0.5",
        "torch_version": report["runtime"]["torch_version"],
    }
    assert report["fixture"] == {
        "dataset_identity": report["fixture"]["dataset_identity"],
        "main_loop_consumed_cursor_at_crash": 3,
        "optimizer_committed_cursor_at_crash": 2,
        "permutation": [8, 3, 1, 7, 0, 9, 4, 2, 6, 5],
        "sampler_emitted_cursor_at_crash": 7,
        "stochastic_mask_seed": 20260815,
        "uncommitted_sample_id_requiring_replay": 1,
    }
    assert report["fixture"]["dataset_identity"].startswith("sha256:")
    assert report["processes"]["all_distinct"] is True
    assert len(
        {
            report["processes"]["baseline_pid"],
            report["processes"]["phase1_pid"],
            report["processes"]["resume_committed_pid"],
            report["processes"][
                "resume_consumed_negative_control_pid"
            ],
            report["processes"]["resume_inflight_gradient_pid"],
            report["processes"][
                "resume_inflight_wrong_rng_negative_control_pid"
            ],
        }
    ) == 6
    assert report["checkpoint"]["schema_version"] == (
        "about-llm.optimizer-commit-checkpoint.v1"
    )
    assert 0 < report["checkpoint"]["size_bytes"] < 4 * 1024 * 1024
    assert report["checkpoint"]["torch_load_weights_only"] is True
    assert report["checkpoint"]["in_flight_gradients_serialized"] is False
    assert report["checkpoint"][
        "commit_boundary_torch_rng_sha256"
    ].startswith("sha256:")
    assert report["checkpoint"]["inflight_sidecar"]["schema_version"] == (
        "about-llm.inflight-gradient-sidecar.v1"
    )
    assert 0 < report["checkpoint"]["inflight_sidecar"]["size_bytes"] < (
        4 * 1024 * 1024
    )
    assert report["checkpoint"]["inflight_sidecar"][
        "base_checkpoint_sha256"
    ] == report["checkpoint"]["sha256"]
    assert report["checkpoint"]["inflight_sidecar"][
        "pending_window_sample_ids"
    ] == [1]
    assert report["checkpoint"]["inflight_sidecar"][
        "gradient_tensor_count"
    ] == 2
    assert report["checkpoint"]["inflight_sidecar"][
        "crash_observed_torch_rng_sha256"
    ].startswith("sha256:")
    assert report["checkpoint"][
        "commit_boundary_torch_rng_sha256"
    ] != report["checkpoint"]["inflight_sidecar"][
        "crash_observed_torch_rng_sha256"
    ]
    assert report["checkpoint"]["bundle_manifest"] == {
        "preload_size_cap_bytes": 16 * 1024,
        "publication_sequence": [
            "base_checkpoint",
            "inflight_gradient_sidecar",
            "bundle_manifest",
        ],
        "publication_state": "complete",
        "published_last_after_payload_artifacts": True,
        "schema_version": "about-llm.optimizer-commit-bundle-manifest.v1",
        "sha256": report["checkpoint"]["bundle_manifest"]["sha256"],
        "size_bytes": report["checkpoint"]["bundle_manifest"]["size_bytes"],
    }
    assert 0 < report["checkpoint"]["bundle_manifest"]["size_bytes"] < (
        16 * 1024
    )
    fault_report = report["paths"]["bundle_publication_fault_injection"]
    assert all(
        fault_report[field] is True
        for field in (
            "complete_bundle_accepted",
            "base_only_rejected",
            "base_and_sidecar_without_manifest_rejected",
            "manifest_without_sidecar_rejected",
            "tampered_sidecar_after_manifest_rejected",
        )
    )
    assert report["comparisons"][
        "committed_resume_model_max_abs_difference"
    ] == 0.0
    assert report["comparisons"][
        "consumed_resume_model_max_abs_difference"
    ] == 0.005767858566116724
    assert report["comparisons"][
        "inflight_resume_model_max_abs_difference"
    ] == 0.0
    assert report["comparisons"][
        "wrong_rng_resume_model_max_abs_difference"
    ] == 0.017878893573032573
    assert report["comparisons"]["uninterrupted_optimizer_steps"] == 5
    assert report["comparisons"]["consumed_resume_optimizer_steps"] == 5
    assert report["comparisons"]["committed_resume_sample_ledger"] == [
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
    ]
    assert report["comparisons"]["consumed_resume_sample_ledger"] == [
        8,
        3,
        7,
        0,
        9,
        4,
        2,
        6,
        5,
    ]
    assert report["comparisons"]["inflight_resume_sample_ledger"] == [
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
    ]
    assert report["comparisons"]["wrong_rng_resume_sample_ledger"] == [
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
    ]
    assert report["comparisons"][
        "consumed_omission_terminal_torch_rng_sha256"
    ] == report["comparisons"]["uninterrupted_terminal_torch_rng_sha256"]
    assert report["comparisons"][
        "wrong_rng_terminal_torch_rng_sha256"
    ] != report["comparisons"]["uninterrupted_terminal_torch_rng_sha256"]
    for path_name in (
        "uninterrupted",
        "resume_from_optimizer_committed_cursor",
        "resume_from_consumed_cursor_with_inflight_gradients",
    ):
        assert report["paths"][path_name]["terminal_state"]["scheduler"] == {
            "last_epoch": 5,
            "learning_rate": 0.0125,
            "step_count": 6,
        }
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "base_checkpoint_and_gradient_sidecar_atomic_publication_proved": False,
        "checkpoint_and_sample_commit_atomic_transaction_proved": False,
        "checkpoint_loaded_with_torch_weights_only": True,
        "checkpoint_temp_file_and_os_replace_executed": True,
        "committed_cursor_replay_matches_uninterrupted_bit_exact": True,
        "concurrent_directory_replacement_or_storage_snapshot_proved": False,
        "consumed_cursor_skip_negative_control_executed": True,
        "crash_after_consumption_before_optimizer_commit_executed": True,
        "distributed_sampler_ddp_fsdp_zero_or_sharded_state_executed": False,
        "grad_scaler_or_cuda_amp_executed": False,
        "gradient_accumulation_window_executed": True,
        "in_flight_gradient_checkpoint_resume_executed": True,
        "in_flight_sidecar_bound_to_base_checkpoint_digest": True,
        "in_flight_gradients_intentionally_excluded_from_checkpoint": True,
        "incomplete_and_tampered_bundle_fault_injection_executed": True,
        "main_process_stochastic_mask_and_torch_cpu_rng_resume_executed": True,
        "manifest_artifact_hashes_rechecked_at_payload_load": True,
        "manifest_last_bundle_completeness_gate_executed": True,
        "negative_control_equal_optimizer_step_count_executed": True,
        "power_loss_directory_fsync_or_storage_durability_proved": False,
        "real_float64_backward_and_sgd_momentum_steps_executed": True,
        "real_step_lr_advanced_after_optimizer_commit_executed": True,
        "real_two_worker_dataloader_prefetch_executed": True,
        "sampler_queue_or_worker_state_serialized": False,
        "target_llm_trainer_dataset_quality_or_convergence_proved": False,
        "worker_rng_or_multi_epoch_policy_executed": False,
        "wrong_rng_with_complete_gradients_negative_control_executed": True,
    }
