from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT
from about_llm.from_scratch.tokenizer import ByteBPETokenizer
from about_llm.inference import (
    MINIGPT_ARCHITECTURE_ID,
    MINIGPT_ARCHITECTURE_REVISION,
    MINIGPT_CHECKPOINT_SCHEMA_VERSION,
    MiniGPTCheckpointIdentity,
    MiniGPTCheckpointLimits,
    load_quantized_minigpt_checkpoint,
    read_quantized_minigpt_checkpoint,
    serialize_quantized_minigpt_checkpoint,
    write_quantized_minigpt_checkpoint_new,
)
from about_llm.llmops import canonical_json_bytes

_HEADER = struct.Struct("<8sB3xIII")
_DIGEST_BYTES = 32


def _fixture() -> tuple[MiniGPT, ByteBPETokenizer, MiniGPTCheckpointIdentity, bytes]:
    torch.manual_seed(7)
    tokenizer = ByteBPETokenizer(((97, 98), (256, 99)))
    model = MiniGPT(
        GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=8,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            mlp_ratio=2,
            dropout=0,
            bias=True,
        )
    ).eval()
    identity = MiniGPTCheckpointIdentity(
        model_id="authored-minigpt",
        model_revision="fixture-seed-7",
        tokenizer_revision="authored-merges-v1",
    )
    artifact = serialize_quantized_minigpt_checkpoint(
        model,
        tokenizer,
        identity=identity,
        bit_width=4,
        group_size=4,
    )
    return model, tokenizer, identity, artifact


def _parts(artifact: bytes) -> tuple[dict[str, Any], bytes]:
    _, _, manifest_length, _, _ = _HEADER.unpack_from(artifact)
    start = _HEADER.size
    stop = start + manifest_length
    return json.loads(artifact[start:stop]), artifact[stop:-_DIGEST_BYTES]


def _rebuild(manifest: dict[str, Any], payload: bytes) -> bytes:
    return _rebuild_raw(canonical_json_bytes(manifest), payload)


def _rebuild_raw(manifest: bytes, payload: bytes) -> bytes:
    decoded = json.loads(manifest)
    parameter_count = len(decoded["parameters"])
    header = _HEADER.pack(b"ALLMGPT1", 1, len(manifest), parameter_count, len(payload))
    body = header + manifest + payload
    return body + hashlib.sha256(body).digest()


def test_checkpoint_restores_tokenizer_all_parameters_tying_and_forward() -> None:
    fp32_model, tokenizer, identity, artifact = _fixture()
    loaded = load_quantized_minigpt_checkpoint(artifact)

    assert loaded.identity == identity
    assert loaded.config == fp32_model.config
    assert loaded.tokenizer.merges == tokenizer.merges
    assert loaded.tokenizer.encode("abc abc") == tokenizer.encode("abc abc")
    assert loaded.tokenizer.decode(tokenizer.encode("abc abc")) == "abc abc"
    assert loaded.model.lm_head.weight is loaded.model.token_embedding.weight
    assert loaded.model.training is False
    assert loaded.bit_width == 4
    assert loaded.group_size == 4
    assert loaded.serialized_artifact_bytes == 8720
    assert loaded.artifact_sha256.startswith("sha256:")
    assert MINIGPT_CHECKPOINT_SCHEMA_VERSION == "about-llm.minigpt-checkpoint.v1"
    assert MINIGPT_ARCHITECTURE_ID == "about-llm.minigpt"
    assert MINIGPT_ARCHITECTURE_REVISION == "about-llm.minigpt.forward.v1"

    input_ids = torch.tensor(
        [tokenizer.encode("abc"), tokenizer.encode("abc")], dtype=torch.long
    )
    with torch.inference_mode():
        fp32_logits, _ = fp32_model(input_ids)
        restored_logits, _ = loaded.model(input_ids)
    assert restored_logits.shape == (2, 1, tokenizer.vocab_size)
    assert torch.isfinite(restored_logits).all()
    assert not torch.equal(fp32_logits, restored_logits)
    assert torch.sqrt(torch.mean((fp32_logits - restored_logits) ** 2)).item() > 0

    # Reloading the same immutable artifact is exact. Re-exporting the
    # dequantized model is intentionally not asserted byte-identical: recomputing
    # absmax scales is a new quantization operation, not an artifact copy.
    second = load_quantized_minigpt_checkpoint(artifact)
    with torch.inference_mode():
        second_logits, _ = second.model(input_ids)
    assert torch.equal(restored_logits, second_logits)


def test_checkpoint_manifest_covers_quantized_vectors_and_derived_state() -> None:
    _, _, _, artifact = _fixture()
    manifest, _ = _parts(artifact)
    parameters = manifest["parameters"]

    assert manifest["architecture"] == {
        "id": MINIGPT_ARCHITECTURE_ID,
        "revision": MINIGPT_ARCHITECTURE_REVISION,
    }
    assert manifest["tokenizer"]["merges"] == [[97, 98], [256, 99]]
    assert manifest["tied_parameters"] == [
        {"alias": "lm_head.weight", "target": "token_embedding.weight"}
    ]
    assert len(parameters) == 16
    assert [item["name"] for item in parameters] == sorted(
        item["name"] for item in parameters
    )
    assert {
        item["kind"] for item in parameters
    } == {
        "groupwise-quantized-matrix-v1",
        "little-endian-float32-vector-v1",
    }
    names = {item["name"] for item in parameters}
    assert "lm_head.weight" not in names
    assert not any(name.endswith("causal_mask") for name in names)
    assert "token_embedding.weight" in names
    assert "final_norm.bias" in names
    assert all(
        item["offset"]
        == sum(previous["length"] for previous in parameters[:index])
        for index, item in enumerate(parameters)
    )


def test_checkpoint_restores_bias_free_multilayer_model() -> None:
    torch.manual_seed(11)
    tokenizer = ByteBPETokenizer()
    model = MiniGPT(
        GPTConfig(
            vocab_size=256,
            context_length=6,
            model_dim=8,
            num_heads=2,
            num_layers=2,
            mlp_ratio=2,
            dropout=0.0,
            bias=False,
        )
    ).eval()
    artifact = serialize_quantized_minigpt_checkpoint(
        model,
        tokenizer,
        identity=MiniGPTCheckpointIdentity("bias-free", "seed-11", "byte-v1"),
        bit_width=5,
        group_size=3,
    )
    loaded = load_quantized_minigpt_checkpoint(artifact)
    input_ids = torch.tensor([tokenizer.encode("hi")], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = loaded.model(input_ids)

    assert loaded.config.bias is False
    assert loaded.config.num_layers == 2
    assert loaded.bit_width == 5
    assert loaded.group_size == 3
    assert len(tuple(loaded.model.named_parameters())) == 20
    assert loaded.model.lm_head.weight is loaded.model.token_embedding.weight
    assert logits.shape == (1, 2, 256)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("suffix", [b"extra", b"\x00"])
def test_checkpoint_rejects_trailing_data(suffix: bytes) -> None:
    _, _, _, artifact = _fixture()
    with pytest.raises(ValueError, match="trailing data"):
        load_quantized_minigpt_checkpoint(artifact + suffix)


def test_checkpoint_rejects_truncation_outer_and_inner_tamper() -> None:
    _, _, _, artifact = _fixture()
    with pytest.raises(ValueError, match=r"truncated|inconsistent"):
        load_quantized_minigpt_checkpoint(artifact[:-1])

    outer_tamper = bytearray(artifact)
    outer_tamper[-33] ^= 1
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_quantized_minigpt_checkpoint(bytes(outer_tamper))

    manifest, payload = _parts(artifact)
    descriptor = next(
        item
        for item in manifest["parameters"]
        if item["kind"] == "groupwise-quantized-matrix-v1"
    )
    tampered_payload = bytearray(payload)
    tampered_payload[descriptor["offset"]] ^= 1
    parameter_bytes = bytes(
        tampered_payload[
            descriptor["offset"] : descriptor["offset"] + descriptor["length"]
        ]
    )
    descriptor["sha256"] = "sha256:" + hashlib.sha256(parameter_bytes).hexdigest()
    with pytest.raises(ValueError, match=r"quantized matrix|magic|SHA-256"):
        load_quantized_minigpt_checkpoint(_rebuild(manifest, bytes(tampered_payload)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update({"unknown": 1}), "manifest fields"),
        (
            lambda manifest: manifest["architecture"].update({"revision": "other"}),
            "architecture identity",
        ),
        (
            lambda manifest: manifest["tied_parameters"].clear(),
            "tied-parameter contract",
        ),
        (
            lambda manifest: manifest["tokenizer"].update({"vocab_size": 257}),
            "tokenizer payload vocabulary",
        ),
        (
            lambda manifest: manifest["parameters"][0].update({"offset": 1}),
            "offsets must be contiguous",
        ),
        (
            lambda manifest: manifest["parameters"][0].update({"name": "wrong"}),
            "shape/name does not match",
        ),
        (
            lambda manifest: manifest["parameters"][0].update({"shape": [1, 1]}),
            "shape/name does not match",
        ),
        (
            lambda manifest: manifest["quantization"].update({"group_size": 3}),
            "quantization config mismatch",
        ),
    ],
)
def test_checkpoint_rejects_manifest_semantic_drift(
    mutate: Any, message: str
) -> None:
    _, _, _, artifact = _fixture()
    manifest, payload = _parts(artifact)
    mutate(manifest)
    with pytest.raises(ValueError, match=message):
        load_quantized_minigpt_checkpoint(_rebuild(manifest, payload))


def test_checkpoint_rejects_duplicate_and_noncanonical_json() -> None:
    _, _, _, artifact = _fixture()
    manifest, payload = _parts(artifact)
    canonical = canonical_json_bytes(manifest)
    duplicate = canonical.replace(
        b'"schema_version":',
        b'"schema_version":"about-llm.minigpt-checkpoint.v1","schema_version":',
        1,
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_quantized_minigpt_checkpoint(_rebuild_raw(duplicate, payload))

    noncanonical = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_quantized_minigpt_checkpoint(_rebuild_raw(noncanonical, payload))


def test_checkpoint_rejects_nonfinite_unquantized_vector_after_rehash() -> None:
    _, _, _, artifact = _fixture()
    manifest, payload = _parts(artifact)
    descriptor = next(
        item
        for item in manifest["parameters"]
        if item["name"] == "final_norm.bias"
    )
    tampered_payload = bytearray(payload)
    start = descriptor["offset"]
    tampered_payload[start : start + 4] = np.asarray([np.nan], dtype="<f4").tobytes()
    parameter_bytes = bytes(tampered_payload[start : start + descriptor["length"]])
    descriptor["sha256"] = "sha256:" + hashlib.sha256(parameter_bytes).hexdigest()
    with pytest.raises(ValueError, match="non-finite"):
        load_quantized_minigpt_checkpoint(_rebuild(manifest, bytes(tampered_payload)))


def test_checkpoint_resource_limits_fail_before_model_construction() -> None:
    model, tokenizer, identity, artifact = _fixture()
    manifest, payload = _parts(artifact)

    with pytest.raises(ValueError, match="artifact exceeds"):
        load_quantized_minigpt_checkpoint(
            artifact,
            limits=MiniGPTCheckpointLimits(max_artifact_bytes=len(artifact) - 1),
        )
    with pytest.raises(ValueError, match="manifest exceeds"):
        load_quantized_minigpt_checkpoint(
            artifact,
            limits=MiniGPTCheckpointLimits(max_manifest_bytes=1),
        )
    with pytest.raises(ValueError, match="parameter count"):
        load_quantized_minigpt_checkpoint(
            artifact,
            limits=MiniGPTCheckpointLimits(max_parameters=1),
        )
    with pytest.raises(ValueError, match="parameter artifact"):
        load_quantized_minigpt_checkpoint(
            artifact,
            limits=MiniGPTCheckpointLimits(max_parameter_bytes=1),
        )
    with pytest.raises(ValueError, match="model parameter count"):
        load_quantized_minigpt_checkpoint(
            artifact,
            limits=MiniGPTCheckpointLimits(max_model_parameter_count=1),
        )
    with pytest.raises(ValueError, match="tokenizer merge count"):
        load_quantized_minigpt_checkpoint(
            artifact,
            limits=MiniGPTCheckpointLimits(max_tokenizer_merges=1),
        )

    with pytest.raises(ValueError, match="tokenizer merge count"):
        serialize_quantized_minigpt_checkpoint(
            model,
            tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
            limits=MiniGPTCheckpointLimits(max_tokenizer_merges=1),
        )

    manifest["config"]["vocab_size"] = 100_000_000
    with pytest.raises(ValueError, match="model parameter count"):
        load_quantized_minigpt_checkpoint(_rebuild(manifest, payload))


def test_checkpoint_disk_round_trip_no_overwrite_and_handle_release(
    tmp_path: Path,
) -> None:
    _, _, _, artifact = _fixture()
    path = tmp_path / "nested" / "minigpt.allmgpt"
    write_quantized_minigpt_checkpoint_new(path, artifact)
    loaded = read_quantized_minigpt_checkpoint(path)

    assert loaded.serialized_artifact_bytes == len(artifact)
    assert loaded.tokenizer.decode(loaded.tokenizer.encode("abc")) == "abc"
    with pytest.raises(FileExistsError):
        write_quantized_minigpt_checkpoint_new(path, artifact)
    path.unlink()
    assert not path.exists()


def test_checkpoint_writer_rejects_invalid_artifact_before_creating_target(
    tmp_path: Path,
) -> None:
    _, _, _, artifact = _fixture()
    target = tmp_path / "invalid.allmgpt"
    with pytest.raises(ValueError, match="SHA-256"):
        write_quantized_minigpt_checkpoint_new(target, artifact[:-1] + b"0")
    assert not target.exists()


def test_checkpoint_serialization_rejects_vocab_dtype_tying_and_nonfinite() -> None:
    model, tokenizer, identity, _ = _fixture()
    wrong_tokenizer = ByteBPETokenizer()
    with pytest.raises(ValueError, match="vocabulary"):
        serialize_quantized_minigpt_checkpoint(
            model,
            wrong_tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
        )

    model.double()
    with pytest.raises(ValueError, match="float32"):
        serialize_quantized_minigpt_checkpoint(
            model,
            tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
        )
    model.float()
    model.lm_head.weight = torch.nn.Parameter(model.token_embedding.weight.detach().clone())
    with pytest.raises(ValueError, match="must remain tied"):
        serialize_quantized_minigpt_checkpoint(
            model,
            tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
        )

    model.lm_head.weight = model.token_embedding.weight
    with torch.no_grad():
        model.final_norm.weight[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        serialize_quantized_minigpt_checkpoint(
            model,
            tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
        )


def test_checkpoint_serialization_normalizes_config_and_rejects_runtime_drift() -> None:
    model, tokenizer, identity, artifact = _fixture()
    model.config = replace(model.config, dropout=0.0)
    assert (
        serialize_quantized_minigpt_checkpoint(
            model,
            tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
        )
        == artifact
    )

    model.config = replace(model.config, num_heads=1)
    with pytest.raises(ValueError, match="attention runtime"):
        serialize_quantized_minigpt_checkpoint(
            model,
            tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
        )

    model.config = replace(model.config, num_heads=2)
    model.blocks[0].attention.causal_mask.zero_()
    with pytest.raises(ValueError, match="attention runtime"):
        serialize_quantized_minigpt_checkpoint(
            model,
            tokenizer,
            identity=identity,
            bit_width=4,
            group_size=4,
        )
