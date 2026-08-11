from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from about_llm.inference import (
    QUANTIZED_BUNDLE_FORMAT_VERSION,
    QUANTIZED_BUNDLE_SCHEMA_VERSION,
    NamedQuantizedMatrix,
    QuantizedBundleIdentity,
    QuantizedBundleLimits,
    QuantizedMatrixBundle,
    quantize_symmetric_groupwise,
    quantized_linear,
)
from about_llm.llmops import canonical_json_bytes

HEADER = struct.Struct("<8sB3xIII")
ROOT = Path(__file__).resolve().parents[1]


def _bundle() -> QuantizedMatrixBundle:
    first = quantize_symmetric_groupwise(
        np.array(
            [
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
                [0.2, 0.3, -0.1, 0.7, -0.5, 0.4],
                [-0.3, 0.8, 0.2, -0.1, 0.6, -0.7],
                [0.5, 0.1, -0.8, 0.2, -0.2, 0.9],
            ],
            dtype=np.float32,
        ),
        bit_width=4,
        group_size=3,
    ).pack()
    second = quantize_symmetric_groupwise(
        np.array(
            [[0.5, -0.4, 0.3, -0.2], [-0.1, 0.2, -0.3, 0.4]],
            dtype=np.float32,
        ),
        bit_width=4,
        group_size=2,
    ).pack()
    return QuantizedMatrixBundle(
        identity=QuantizedBundleIdentity(
            model_family="authored-two-layer-mlp",
            model_revision="fixture-v1",
            tokenizer_id="integer-input-fixture",
            tokenizer_revision="not-a-text-tokenizer-v1",
            architecture_config={
                "input_features": 6,
                "hidden_features": 4,
                "output_features": 2,
                "activation": "tanh",
                "nested": {"values": [1, 2]},
            },
        ),
        tensors=(
            NamedQuantizedMatrix("layer.1.weight", second),
            NamedQuantizedMatrix("layer.0.weight", first),
        ),
    )


def _parts(artifact: bytes) -> tuple[dict, bytes]:
    _, _, manifest_length, _, _ = HEADER.unpack_from(artifact)
    start = HEADER.size
    stop = start + manifest_length
    return json.loads(artifact[start:stop]), artifact[stop:-32]


def _rebuild(manifest: dict | bytes, payload: bytes) -> bytes:
    manifest_bytes = (
        manifest if isinstance(manifest, bytes) else canonical_json_bytes(manifest)
    )
    tensor_count = (
        len(manifest.get("tensors", [])) if isinstance(manifest, dict) else 2
    )
    header = HEADER.pack(
        b"ALLMQB01",
        QUANTIZED_BUNDLE_FORMAT_VERSION,
        len(manifest_bytes),
        tensor_count,
        len(payload),
    )
    body = header + manifest_bytes + payload
    return body + hashlib.sha256(body).digest()


def test_bundle_round_trips_named_matrices_and_canonical_identity() -> None:
    bundle = _bundle()
    artifact = bundle.to_bytes()
    restored = QuantizedMatrixBundle.from_bytes(artifact)

    assert QUANTIZED_BUNDLE_FORMAT_VERSION == 1
    assert QUANTIZED_BUNDLE_SCHEMA_VERSION.endswith(".v1")
    assert artifact[:8] == b"ALLMQB01"
    assert artifact[8] == 1
    assert restored.tensor_names == ("layer.0.weight", "layer.1.weight")
    assert restored.identity.to_dict() == bundle.identity.to_dict()
    assert restored.to_bytes() == artifact
    assert bundle.serialized_artifact_bytes == len(artifact)
    assert bundle.serialized_artifact_sha256.startswith("sha256:")
    np.testing.assert_array_equal(
        restored.get("layer.0.weight").unpack().values,
        bundle.get("layer.0.weight").unpack().values,
    )


def test_reloaded_two_layer_forward_matches_in_memory_quantized_forward() -> None:
    bundle = _bundle()
    restored = QuantizedMatrixBundle.from_bytes(bundle.to_bytes())
    inputs = np.array(
        [[0.5, -0.2, 0.1, 0.7, -0.4, 0.3], [-0.1, 0.9, 0.2, 0.0, 0.4, -0.8]],
        dtype=np.float32,
    )

    expected = quantized_linear(
        np.tanh(quantized_linear(inputs, bundle.get("layer.0.weight"))),
        bundle.get("layer.1.weight"),
    )
    actual = quantized_linear(
        np.tanh(quantized_linear(inputs, restored.get("layer.0.weight"))),
        restored.get("layer.1.weight"),
    )

    np.testing.assert_array_equal(actual, expected)


def test_identity_takes_a_recursive_immutable_json_snapshot() -> None:
    config = {"layers": [4, 2], "nested": {"activation": "tanh"}}
    identity = QuantizedBundleIdentity(
        "family", "revision", "tokenizer", "tokenizer-revision", config
    )
    config["layers"].append(99)
    config["nested"]["activation"] = "changed"

    assert identity.to_dict()["architecture_config"]["layers"] == [4, 2]
    assert identity.to_dict()["architecture_config"]["nested"] == {
        "activation": "tanh"
    }
    with pytest.raises(TypeError):
        identity.architecture_config["new"] = 1  # type: ignore[index]


def test_outer_format_rejects_tamper_version_trailing_and_truncation() -> None:
    artifact = _bundle().to_bytes()

    tampered = bytearray(artifact)
    tampered[-33] ^= 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        QuantizedMatrixBundle.from_bytes(bytes(tampered))
    wrong_version = bytearray(artifact)
    wrong_version[8] = 2
    with pytest.raises(ValueError, match=r"unsupported.*version"):
        QuantizedMatrixBundle.from_bytes(bytes(wrong_version))
    with pytest.raises(ValueError, match="trailing"):
        QuantizedMatrixBundle.from_bytes(artifact + b"x")
    with pytest.raises(ValueError, match=r"truncated|length"):
        QuantizedMatrixBundle.from_bytes(artifact[:-1])


def test_recomputed_outer_digest_does_not_bypass_manifest_structure() -> None:
    artifact = _bundle().to_bytes()
    manifest, payload = _parts(artifact)

    manifest["unexpected"] = True
    with pytest.raises(ValueError, match="manifest fields"):
        QuantizedMatrixBundle.from_bytes(_rebuild(manifest, payload))

    manifest, payload = _parts(artifact)
    manifest["tensors"][1]["name"] = manifest["tensors"][0]["name"]
    with pytest.raises(ValueError, match="name-sorted and unique"):
        QuantizedMatrixBundle.from_bytes(_rebuild(manifest, payload))

    manifest, payload = _parts(artifact)
    manifest["tensors"][1]["offset"] += 1
    with pytest.raises(ValueError, match="offsets must be contiguous"):
        QuantizedMatrixBundle.from_bytes(_rebuild(manifest, payload))


def test_noncanonical_and_duplicate_key_manifests_are_rejected() -> None:
    artifact = _bundle().to_bytes()
    manifest, payload = _parts(artifact)
    noncanonical = json.dumps(manifest, ensure_ascii=False).encode()
    with pytest.raises(ValueError, match="canonical JSON"):
        QuantizedMatrixBundle.from_bytes(_rebuild(noncanonical, payload))

    canonical = canonical_json_bytes(manifest)
    duplicate = canonical[:-1] + b',"schema_version":"duplicate"}'
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        QuantizedMatrixBundle.from_bytes(_rebuild(duplicate, payload))


def test_resource_limits_apply_before_or_during_payload_materialization() -> None:
    bundle = _bundle()
    artifact = bundle.to_bytes()
    manifest, _ = _parts(artifact)
    tensor_length = manifest["tensors"][0]["length"]

    with pytest.raises(ValueError, match="tensor count"):
        bundle.to_bytes(limits=QuantizedBundleLimits(max_tensors=1))
    with pytest.raises(ValueError, match="artifact exceeds"):
        QuantizedMatrixBundle.from_bytes(
            artifact,
            limits=QuantizedBundleLimits(max_artifact_bytes=len(artifact) - 1),
        )
    with pytest.raises(ValueError, match="manifest exceeds"):
        QuantizedMatrixBundle.from_bytes(
            artifact, limits=QuantizedBundleLimits(max_manifest_bytes=8)
        )
    with pytest.raises(ValueError, match="tensor artifact exceeds"):
        QuantizedMatrixBundle.from_bytes(
            artifact,
            limits=QuantizedBundleLimits(max_tensor_bytes=tensor_length - 1),
        )


def test_exclusive_file_round_trip_releases_windows_handle(tmp_path: Path) -> None:
    bundle = _bundle()
    path = tmp_path / "bundle.allmqb"
    bundle.write_new(path)

    restored = QuantizedMatrixBundle.read(path)
    assert restored.to_bytes() == bundle.to_bytes()
    with pytest.raises(FileExistsError):
        bundle.write_new(path)

    moved = tmp_path / "moved.allmqb"
    path.rename(moved)
    assert QuantizedMatrixBundle.read(moved).tensor_names == bundle.tensor_names
    with pytest.raises(ValueError, match="file exceeds"):
        QuantizedMatrixBundle.read(
            moved,
            limits=QuantizedBundleLimits(max_artifact_bytes=moved.stat().st_size - 1),
        )


def test_bundle_rejects_duplicate_names_invalid_identity_and_missing_tensor() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="unique"):
        QuantizedMatrixBundle(
            bundle.identity,
            (
                bundle.tensors[0],
                NamedQuantizedMatrix(bundle.tensors[0].name, bundle.tensors[1].matrix),
            ),
        )
    with pytest.raises(ValueError, match="finite JSON"):
        QuantizedBundleIdentity(
            "family", "revision", "tokenizer", "revision", {"value": float("nan")}
        )
    with pytest.raises(KeyError):
        bundle.get("missing.weight")


def test_quantized_bundle_toy_executes_reloaded_two_layer_forward(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "toy.allmqb"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "projects"
                / "inference-serving"
                / "quantized_bundle_toy.py"
            ),
            "--artifact-path",
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["tensor_names"] == ["layer.0.weight", "layer.1.weight"]
    assert artifact["tensor_count"] == 2
    assert artifact["reference_fp32_weight_bytes"] == 288
    assert artifact["exact_byte_round_trip"] is True
    assert artifact["exact_quantized_forward_round_trip"] is True
    assert artifact["disk_round_trip"] is True
    assert artifact["bundle_container_overhead_bytes"] > 0
    assert artifact["scope"]["tokenizer_payload_embedded"] is False
    assert artifact["scope"]["full_llm_checkpoint"] is False
