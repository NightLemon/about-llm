from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from about_llm.from_scratch.jax_training_resume import (
    JAX_TRAINING_RESUME_VERSION,
    create_jax_training_state,
    load_jax_training_checkpoint,
    parse_jax_training_checkpoint,
    serialize_jax_training_checkpoint,
    train_jax_steps,
    write_jax_training_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "projects" / "jax-minigpt" / "checkpoint_resume_control.py"


def _reject_nonfinite(value: str) -> NoReturn:
    raise AssertionError(f"non-standard JSON number: {value}")


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@pytest.mark.slow
def test_cross_process_jax_checkpoint_resume_is_bit_exact() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    report = json.loads(
        completed.stdout,
        parse_constant=_reject_nonfinite,
    )

    assert completed.stderr == ""
    assert report["schema_version"] == JAX_TRAINING_RESUME_VERSION
    assert report["runtime"]["jax_backend"] == "cpu"
    assert report["runtime"]["process_start_method"] == "spawn"
    assert report["fixture"]["total_steps"] == 6
    assert report["fixture"]["split_step"] == 3
    assert report["fixture"]["dataset_examples"] == 7
    assert report["fixture"]["dataset_fingerprint"] == (
        "sha256:d91b77df27af887b21758e4b2f0cb69004db9da08b99942a4ba085526bf99da3"
    )
    assert report["artifact"] == {
        "artifact_bytes": 13476,
        "artifact_sha256": (
            "sha256:e9252e5dddfa4aa507bfaa864cd205f9ba5a7c0aef7a03b4b98366f770568a35"
        ),
    }
    assert report["process_observation"] == {
        "distinct_phase_worker_count": 2,
        "raw_process_ids_published": False,
    }
    assert report["uninterrupted"] == report["resumed"]
    assert report["uninterrupted"]["final_state_fingerprint"] == (
        "sha256:720817cca4c067cf1e532a5ce73e13d0dd1eba1c7b65c964445f67171d058f33"
    )
    assert report["uninterrupted"]["trace"] == {
        "gradient_norms": [
            1.6905815601348877,
            1.9288957118988037,
            1.3948036432266235,
            1.6974924802780151,
            1.3795976638793945,
            1.1869877576828003,
        ],
        "losses": [
            2.584887981414795,
            2.517184019088745,
            2.497069835662842,
            2.4534246921539307,
            2.3306925296783447,
            2.374443292617798,
        ],
        "sample_ids": [[0, 4], [3, 2], [5, 1], [6, 3], [2, 1], [6, 4]],
    }
    assert report["counterfactuals"]["reset_dropout_prng"][
        "parameter_max_abs_difference"
    ] > 0
    assert report["counterfactuals"]["reset_data_cursor"][
        "parameter_max_abs_difference"
    ] == 0.03700308472616598
    assert report["counterfactuals"]["reset_dropout_prng"][
        "parameter_max_abs_difference"
    ] == 0.037261832505464554
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "artifact_origin_authentication_or_confidentiality_proved": False,
        "bit_exact_full_state_and_trace_compared": True,
        "cross_process_split_resume_executed": True,
        "cuda_tpu_multi_device_or_sharding_executed": False,
        "directory_fsync_or_power_loss_atomicity_proved": False,
        "dropout_prng_and_data_shuffle_state_restored": True,
        "exclusive_create_and_file_fsync_executed": True,
        "orbax_flax_tensorstore_or_distributed_checkpoint_executed": False,
        "parameter_and_optax_state_restored": True,
        "python_numpy_worker_or_accelerator_rng_restored": False,
        "strict_canonical_manifest_and_outer_digest_executed": True,
        "target_model_dataset_convergence_or_performance_proved": False,
        "wrong_prng_and_cursor_counterfactuals_executed": True,
    }
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == (
        "sha256:652c22d525598adfbd473738c6b3ef4cbffaf13c0e2ae06a8de63b1d467e6fee"
    )
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout


def test_checkpoint_round_trip_and_exclusive_create(tmp_path: Path) -> None:
    state, _ = train_jax_steps(create_jax_training_state(), steps=2)
    checkpoint = tmp_path / "state.allmjax"
    metadata = write_jax_training_checkpoint(checkpoint, state)
    restored = load_jax_training_checkpoint(checkpoint)

    assert metadata["artifact_bytes"] == checkpoint.stat().st_size
    assert serialize_jax_training_checkpoint(restored) == checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        write_jax_training_checkpoint(checkpoint, state)


@pytest.mark.parametrize(
    "mutation",
    ["truncate", "payload", "inner_payload", "digest"],
)
def test_checkpoint_rejects_truncation_and_tamper(mutation: str) -> None:
    state = create_jax_training_state()
    artifact = bytearray(serialize_jax_training_checkpoint(state))
    if mutation == "truncate":
        mutated = bytes(artifact[:-1])
    elif mutation == "payload":
        artifact[len(artifact) // 2] ^= 0x01
        mutated = bytes(artifact)
    elif mutation == "inner_payload":
        prefix = artifact[:-32]
        prefix[-1] ^= 0x01
        mutated = bytes(prefix) + hashlib.sha256(prefix).digest()
    else:
        artifact[-1] ^= 0x01
        mutated = bytes(artifact)
    with pytest.raises(ValueError):
        parse_jax_training_checkpoint(mutated)
