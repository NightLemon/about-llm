from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from about_llm.finetuning.minigpt_training_checkpoint import (
    MINIGPT_TRAINING_CHECKPOINT_SCHEMA_VERSION,
    AdamWTrainingConfig,
    LinearLearningRateSchedule,
    MiniGPTTrainingCheckpointIdentity,
    MiniGPTTrainingCheckpointLimits,
    MiniGPTTrainingConfig,
    MiniGPTTrainingState,
    create_minigpt_training_state,
    load_minigpt_training_checkpoint,
    read_minigpt_training_checkpoint,
    run_minigpt_training_updates,
    serialize_minigpt_training_checkpoint,
    write_minigpt_training_checkpoint_new,
)
from about_llm.from_scratch import ByteBPETokenizer, GPTConfig, MiniGPT
from about_llm.llmops import canonical_json_bytes

pytestmark = [pytest.mark.contract, pytest.mark.integration]

_HEADER = struct.Struct("<8sB3xIII")
_DIGEST_BYTES = 32


def _fixture() -> tuple[
    MiniGPTTrainingState,
    torch.Tensor,
    MiniGPTTrainingCheckpointIdentity,
]:
    torch.manual_seed(13)
    tokenizer = ByteBPETokenizer()
    model = MiniGPT(
        GPTConfig(
            vocab_size=256,
            context_length=4,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            mlp_ratio=2,
            dropout=0.2,
            bias=True,
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(99)
    dataset = torch.randint(0, 256, (7, 5), generator=generator)
    training_config = MiniGPTTrainingConfig(
        optimizer=AdamWTrainingConfig(
            learning_rate=0.003,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            weight_decay=0.01,
        ),
        schedule=LinearLearningRateSchedule(
            initial_learning_rate=0.003,
            final_learning_rate=0.001,
            total_updates=6,
        ),
        batch_size=2,
        max_grad_norm=1.0,
    )
    state = create_minigpt_training_state(
        model,
        tokenizer,
        dataset,
        training_config=training_config,
        training_seed=17,
        data_seed=19,
    )
    identity = MiniGPTTrainingCheckpointIdentity(
        run_id="authored-resume-control",
        model_revision="fixture-seed-13",
        tokenizer_revision="byte-v1",
        data_revision="fixture-seed-99",
    )
    return state, dataset, identity


def _assert_states_exact(
    observed: MiniGPTTrainingState, expected: MiniGPTTrainingState
) -> None:
    assert observed.global_step == expected.global_step
    assert observed.training_config == expected.training_config
    assert observed.dataset_binding == expected.dataset_binding
    assert observed.tokenizer.merges == expected.tokenizer.merges
    assert observed.batch_stream.cursor == expected.batch_stream.cursor
    assert observed.batch_stream.epoch == expected.batch_stream.epoch
    assert torch.equal(
        observed.batch_stream.permutation, expected.batch_stream.permutation
    )
    assert torch.equal(
        observed.batch_stream.generator_state, expected.batch_stream.generator_state
    )
    assert torch.equal(observed.torch_cpu_rng_state, expected.torch_cpu_rng_state)
    observed_parameters = dict(observed.model.named_parameters())
    expected_parameters = dict(expected.model.named_parameters())
    assert observed_parameters.keys() == expected_parameters.keys()
    for name, parameter in observed_parameters.items():
        expected_parameter = expected_parameters[name]
        assert torch.equal(parameter, expected_parameter), name
        observed_optimizer = observed.optimizer.state[parameter]
        expected_optimizer = expected.optimizer.state[expected_parameter]
        assert observed_optimizer.keys() == expected_optimizer.keys()
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(observed_optimizer[field], expected_optimizer[field]), (
                name,
                field,
            )
    assert observed.optimizer.param_groups[0]["lr"] == expected.optimizer.param_groups[0][
        "lr"
    ]


def _parts(artifact: bytes) -> tuple[dict[str, Any], bytes]:
    _, _, manifest_length, _, _ = _HEADER.unpack_from(artifact)
    start = _HEADER.size
    stop = start + manifest_length
    return json.loads(artifact[start:stop]), artifact[stop:-_DIGEST_BYTES]


def _rebuild(manifest: dict[str, Any], payload: bytes) -> bytes:
    return _rebuild_raw(canonical_json_bytes(manifest), payload)


def _rebuild_raw(manifest: bytes, payload: bytes) -> bytes:
    decoded = json.loads(manifest)
    header = _HEADER.pack(
        b"ALLMTRN1", 1, len(manifest), len(decoded["tensors"]), len(payload)
    )
    body = header + manifest + payload
    return body + hashlib.sha256(body).digest()


def test_training_checkpoint_split_run_is_bit_exact_and_preserves_external_rng() -> None:
    baseline, baseline_data, _ = _fixture()
    split, split_data, identity = _fixture()
    external_rng_before = torch.get_rng_state().clone()

    baseline_reports = run_minigpt_training_updates(
        baseline, baseline_data, updates=6
    )
    first_reports = run_minigpt_training_updates(split, split_data, updates=3)
    assert torch.equal(torch.get_rng_state(), external_rng_before)
    artifact = serialize_minigpt_training_checkpoint(
        split, split_data, identity=identity
    )
    restored, restored_identity = load_minigpt_training_checkpoint(
        artifact, split_data
    )
    _assert_states_exact(restored, split)
    tail_reports = run_minigpt_training_updates(restored, split_data, updates=3)

    assert restored_identity == identity
    assert MINIGPT_TRAINING_CHECKPOINT_SCHEMA_VERSION == (
        "about-llm.minigpt-training-checkpoint.v1"
    )
    assert first_reports == baseline_reports[:3]
    assert tail_reports == baseline_reports[3:]
    assert [report.batch_indices for report in baseline_reports] == [
        (6, 5),
        (2, 1),
        (4, 0),
        (1, 0),
        (6, 5),
        (3, 4),
    ]
    assert [report.epoch for report in baseline_reports] == [0, 0, 0, 1, 1, 1]
    assert [report.learning_rate for report in baseline_reports] == [
        0.003,
        0.0026,
        0.0022,
        0.0018000000000000002,
        0.0014,
        0.001,
    ]
    assert len(artifact) == 53_917
    _assert_states_exact(restored, baseline)
    assert torch.equal(torch.get_rng_state(), external_rng_before)


def test_training_checkpoint_manifest_and_payload_cover_required_state() -> None:
    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=3)
    artifact = serialize_minigpt_training_checkpoint(state, dataset, identity=identity)
    manifest, _ = _parts(artifact)
    names = [item["name"] for item in manifest["tensors"]]

    assert names == sorted(names)
    assert manifest["progress"] == {
        "global_step": 3,
        "current_learning_rate": 0.0022,
        "epoch": 0,
        "cursor": 6,
        "gradient_accumulation_position": 0,
        "checkpoint_at_optimizer_boundary": True,
    }
    assert manifest["training"]["optimizer"]["type"] == "torch.optim.AdamW"
    assert manifest["training"]["schedule"]["type"] == "linear-per-update-v1"
    assert manifest["training"]["precision"] == "cpu-float32"
    assert manifest["training"]["gradient_accumulation_steps"] == 1
    assert len(names) == 51
    assert sum(name.startswith("model/") for name in names) == 16
    assert sum(name.endswith("/exp_avg") for name in names) == 16
    assert sum(name.endswith("/exp_avg_sq") for name in names) == 16
    assert "rng/torch_cpu" in names
    assert "rng/data_generator" in names
    assert "stream/permutation" in names


def test_training_checkpoint_rejects_zero_step_gradients_and_optimizer_drift() -> None:
    state, dataset, identity = _fixture()
    with pytest.raises(ValueError, match="at least one completed update"):
        serialize_minigpt_training_checkpoint(state, dataset, identity=identity)

    batch = dataset[:2]
    _, loss = state.model(batch[:, :-1], batch[:, 1:])
    assert loss is not None
    loss.backward()
    state.global_step = 1
    with pytest.raises(ValueError, match="gradients cleared"):
        serialize_minigpt_training_checkpoint(state, dataset, identity=identity)

    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=1)
    state.optimizer.param_groups[0]["weight_decay"] = 0.5
    with pytest.raises(ValueError, match="hyperparameters"):
        serialize_minigpt_training_checkpoint(state, dataset, identity=identity)


def test_training_checkpoint_rejects_dataset_drift() -> None:
    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=2)
    artifact = serialize_minigpt_training_checkpoint(state, dataset, identity=identity)
    changed = dataset.clone()
    changed[0, 0] = (changed[0, 0] + 1) % 256
    with pytest.raises(ValueError, match="dataset identity"):
        load_minigpt_training_checkpoint(artifact, changed)


def test_training_checkpoint_rejects_outer_trailing_and_truncation() -> None:
    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=2)
    artifact = serialize_minigpt_training_checkpoint(state, dataset, identity=identity)
    tampered = bytearray(artifact)
    tampered[-33] ^= 1
    with pytest.raises(ValueError, match="SHA-256"):
        load_minigpt_training_checkpoint(bytes(tampered), dataset)
    with pytest.raises(ValueError, match="trailing"):
        load_minigpt_training_checkpoint(artifact + b"x", dataset)
    with pytest.raises(ValueError, match=r"truncated|inconsistent"):
        load_minigpt_training_checkpoint(artifact[:-1], dataset)


def test_training_checkpoint_rejects_duplicate_noncanonical_and_semantic_drift() -> None:
    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=3)
    artifact = serialize_minigpt_training_checkpoint(state, dataset, identity=identity)
    manifest, payload = _parts(artifact)
    canonical = canonical_json_bytes(manifest)
    duplicate = canonical.replace(
        b'"schema_version":',
        b'"schema_version":"about-llm.minigpt-training-checkpoint.v1",'
        b'"schema_version":',
        1,
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_minigpt_training_checkpoint(_rebuild_raw(duplicate, payload), dataset)
    noncanonical = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_minigpt_training_checkpoint(_rebuild_raw(noncanonical, payload), dataset)

    manifest["progress"]["current_learning_rate"] = 0.001
    with pytest.raises(ValueError, match="learning rate drifted"):
        load_minigpt_training_checkpoint(_rebuild(manifest, payload), dataset)


def test_training_checkpoint_rejects_tensor_nan_permutation_and_rng_after_rehash() -> None:
    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=3)
    artifact = serialize_minigpt_training_checkpoint(state, dataset, identity=identity)

    for tensor_name, replacement, message in (
        (
            "optimizer/final_norm.bias/exp_avg",
            np.asarray([np.nan], dtype="<f4").tobytes(),
            "non-finite",
        ),
        (
            "stream/permutation",
            np.asarray([0], dtype="<i8").tobytes(),
            "permutation",
        ),
    ):
        manifest, payload = _parts(artifact)
        descriptor = next(
            item for item in manifest["tensors"] if item["name"] == tensor_name
        )
        tampered = bytearray(payload)
        start = descriptor["offset"]
        tampered[start : start + len(replacement)] = replacement
        tensor_bytes = bytes(tampered[start : start + descriptor["length"]])
        descriptor["sha256"] = "sha256:" + hashlib.sha256(tensor_bytes).hexdigest()
        with pytest.raises(ValueError, match=message):
            load_minigpt_training_checkpoint(
                _rebuild(manifest, bytes(tampered)), dataset
            )

    manifest, payload = _parts(artifact)
    descriptor = next(
        item for item in manifest["tensors"] if item["name"] == "rng/torch_cpu"
    )
    tampered = bytearray(payload)
    start = descriptor["offset"]
    tampered[start : start + descriptor["length"]] = b"\x00" * descriptor["length"]
    tensor_bytes = bytes(tampered[start : start + descriptor["length"]])
    descriptor["sha256"] = "sha256:" + hashlib.sha256(tensor_bytes).hexdigest()
    with pytest.raises(ValueError, match="RNG state is invalid"):
        load_minigpt_training_checkpoint(_rebuild(manifest, bytes(tampered)), dataset)


def test_training_checkpoint_resource_limits() -> None:
    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=2)
    artifact = serialize_minigpt_training_checkpoint(state, dataset, identity=identity)
    with pytest.raises(ValueError, match="artifact exceeds"):
        load_minigpt_training_checkpoint(
            artifact,
            dataset,
            limits=MiniGPTTrainingCheckpointLimits(
                max_artifact_bytes=len(artifact) - 1
            ),
        )
    with pytest.raises(ValueError, match="manifest exceeds"):
        load_minigpt_training_checkpoint(
            artifact,
            dataset,
            limits=MiniGPTTrainingCheckpointLimits(max_manifest_bytes=1),
        )
    with pytest.raises(ValueError, match="tensor count"):
        load_minigpt_training_checkpoint(
            artifact,
            dataset,
            limits=MiniGPTTrainingCheckpointLimits(max_tensors=1),
        )
    with pytest.raises(ValueError, match="tensor exceeds"):
        load_minigpt_training_checkpoint(
            artifact,
            dataset,
            limits=MiniGPTTrainingCheckpointLimits(max_tensor_bytes=1),
        )
    with pytest.raises(ValueError, match="model parameter count"):
        load_minigpt_training_checkpoint(
            artifact,
            dataset,
            limits=MiniGPTTrainingCheckpointLimits(max_model_parameter_count=1),
        )
    with pytest.raises(ValueError, match="example count"):
        load_minigpt_training_checkpoint(
            artifact,
            dataset,
            limits=MiniGPTTrainingCheckpointLimits(max_examples=1),
        )
    with pytest.raises(ValueError, match="RNG tensor"):
        load_minigpt_training_checkpoint(
            artifact,
            dataset,
            limits=MiniGPTTrainingCheckpointLimits(max_rng_state_bytes=1),
        )


def test_training_checkpoint_disk_round_trip_no_overwrite_and_handle_release(
    tmp_path: Path,
) -> None:
    state, dataset, identity = _fixture()
    run_minigpt_training_updates(state, dataset, updates=2)
    artifact = serialize_minigpt_training_checkpoint(state, dataset, identity=identity)
    path = tmp_path / "nested" / "training.allmtrn"
    write_minigpt_training_checkpoint_new(path, artifact, dataset)
    restored, restored_identity = read_minigpt_training_checkpoint(path, dataset)
    assert restored.global_step == 2
    assert restored_identity == identity
    with pytest.raises(FileExistsError):
        write_minigpt_training_checkpoint_new(path, artifact, dataset)
    path.unlink()
    assert not path.exists()


def test_training_schedule_rejects_unrepresentable_adamw_step() -> None:
    with pytest.raises(ValueError, match="float32 AdamW step range"):
        LinearLearningRateSchedule(0.1, 0.01, (1 << 24) + 1)
