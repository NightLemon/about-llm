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
    QUANTIZED_MATRIX_FORMAT_VERSION,
    GroupwiseQuantizedMatrix,
    PackedGroupwiseQuantizedMatrix,
    quantization_error,
    quantize_symmetric_groupwise,
    quantized_linear,
)

ROOT = Path(__file__).resolve().parents[1]


def test_groupwise_symmetric_quantization_has_explicit_codes_and_scales() -> None:
    weights = np.array([[0.0, 1.0, -1.0, 0.6, 0.0]], dtype=np.float32)
    quantized = quantize_symmetric_groupwise(weights, bit_width=4, group_size=2)

    np.testing.assert_array_equal(quantized.values, [[0, 7, -7, 4, 0]])
    np.testing.assert_allclose(quantized.scales, [[1 / 7, 1 / 7, 1]])
    np.testing.assert_allclose(
        quantized.dequantize(), [[0, 1, -1, 4 / 7, 0]], rtol=1e-6
    )
    assert quantized.maximum_code == 7
    assert not quantized.values.flags.writeable
    assert not quantized.scales.flags.writeable


def test_storage_ledger_distinguishes_packed_lower_bound_from_numpy_container() -> None:
    weights = np.arange(15, dtype=np.float32).reshape(3, 5)
    quantized = quantize_symmetric_groupwise(weights, bit_width=4, group_size=4)

    assert quantized.groups_per_row == 2
    assert quantized.reference_fp32_weight_bytes == 60
    assert quantized.ideal_packed_weight_bytes == 8
    assert quantized.scale_metadata_bytes == 24
    assert quantized.ideal_total_bytes == 32
    assert quantized.ideal_compression_ratio == pytest.approx(60 / 32)
    assert quantized.unpacked_reference_bytes == 39


def test_dense_bitstream_has_known_nibbles_and_exact_round_trip() -> None:
    quantized = GroupwiseQuantizedMatrix(
        values=np.array([[-7, 0, 7]], dtype=np.int8),
        scales=np.array([[0.5]], dtype=np.float32),
        original_shape=(1, 3),
        bit_width=4,
        group_size=3,
    )

    packed = quantized.pack()

    # Offset-binary codes are [0, 7, 14]. Low nibble comes first, and the
    # high nibble after the final code is canonical zero padding.
    assert packed.packed_values.hex() == "700e"
    assert packed.padding_bits == 4
    assert packed.packed_weight_bytes == quantized.ideal_packed_weight_bytes == 2
    assert packed.raw_payload_bytes == 6
    assert packed.packed_values_sha256.startswith("sha256:")
    np.testing.assert_array_equal(packed.unpack().values, quantized.values)
    np.testing.assert_array_equal(packed.unpack().scales, quantized.scales)
    np.testing.assert_array_equal(packed.dequantize(), quantized.dequantize())
    assert not packed.scales.flags.writeable


def test_binary_tensor_artifact_is_little_endian_and_exactly_reloadable() -> None:
    quantized = GroupwiseQuantizedMatrix(
        values=np.array([[-7, 0, 7]], dtype=np.int8),
        scales=np.array([[0.5]], dtype=np.float32),
        original_shape=(1, 3),
        bit_width=4,
        group_size=3,
    )
    packed = quantized.pack()

    artifact = packed.to_bytes()
    restored = PackedGroupwiseQuantizedMatrix.from_bytes(artifact)

    assert QUANTIZED_MATRIX_FORMAT_VERSION == 1
    assert artifact[:8] == b"ALLMQTZ1"
    assert artifact[8] == QUANTIZED_MATRIX_FORMAT_VERSION
    assert artifact[32:34].hex() == "700e"
    assert artifact[34:38] == struct.pack("<f", 0.5)
    assert len(artifact) == packed.serialized_artifact_bytes == 70
    assert packed.serialized_artifact_sha256.startswith("sha256:")
    np.testing.assert_array_equal(restored.unpack().values, quantized.values)
    np.testing.assert_array_equal(restored.scales, quantized.scales)


def test_binary_tensor_artifact_rejects_version_tamper_trailing_and_nonfinite_scale() -> None:
    packed = quantize_symmetric_groupwise(
        [[-1.0, 0.0, 1.0]], bit_width=4, group_size=3
    ).pack()
    artifact = packed.to_bytes()

    wrong_version = bytearray(artifact)
    wrong_version[8] = 2
    with pytest.raises(ValueError, match="unsupported format version"):
        PackedGroupwiseQuantizedMatrix.from_bytes(bytes(wrong_version))
    wrong_mapping = bytearray(artifact)
    wrong_mapping[10] = 2
    with pytest.raises(ValueError, match="unsupported code mapping"):
        PackedGroupwiseQuantizedMatrix.from_bytes(bytes(wrong_mapping))
    wrong_dtype = bytearray(artifact)
    wrong_dtype[11] = 2
    with pytest.raises(ValueError, match="unsupported scale dtype"):
        PackedGroupwiseQuantizedMatrix.from_bytes(bytes(wrong_dtype))
    wrong_packed_length = bytearray(artifact)
    struct.pack_into("<I", wrong_packed_length, 24, 999)
    with pytest.raises(ValueError, match="packed length is inconsistent"):
        PackedGroupwiseQuantizedMatrix.from_bytes(bytes(wrong_packed_length))

    payload_tamper = bytearray(artifact)
    payload_tamper[32] ^= 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PackedGroupwiseQuantizedMatrix.from_bytes(bytes(payload_tamper))
    with pytest.raises(ValueError, match="trailing data"):
        PackedGroupwiseQuantizedMatrix.from_bytes(artifact + b"x")
    with pytest.raises(ValueError, match=r"truncated|length"):
        PackedGroupwiseQuantizedMatrix.from_bytes(artifact[:-1])

    # Recompute the unkeyed digest to show that the structural validator, not
    # the checksum alone, rejects a non-finite scale.
    nonfinite = bytearray(artifact[:-32])
    packed_length = packed.packed_weight_bytes
    scale_start = 32 + packed_length
    nonfinite[scale_start : scale_start + 4] = struct.pack("<f", float("nan"))
    forged = bytes(nonfinite) + hashlib.sha256(nonfinite).digest()
    with pytest.raises(ValueError, match="finite and positive"):
        PackedGroupwiseQuantizedMatrix.from_bytes(forged)


@pytest.mark.parametrize("bit_width", range(2, 9))
def test_dense_bitstream_round_trips_cross_byte_codes(bit_width: int) -> None:
    weights = np.array(
        [[-1.0, -0.25, 0.0, 0.2, 0.7, 1.0, -0.8], [0.0] * 7],
        dtype=np.float32,
    )
    quantized = quantize_symmetric_groupwise(
        weights, bit_width=bit_width, group_size=3
    )

    packed = quantized.pack()
    unpacked = packed.unpack()

    assert packed.packed_weight_bytes == quantized.ideal_packed_weight_bytes
    assert packed.padding_bits == (-quantized.values.size * bit_width) % 8
    np.testing.assert_array_equal(unpacked.values, quantized.values)
    np.testing.assert_array_equal(unpacked.scales, quantized.scales)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "length"),
        (b"\x81", "padding"),
        (b"\x03", "unused all-ones"),
    ],
)
def test_packed_constructor_rejects_noncanonical_payloads(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PackedGroupwiseQuantizedMatrix(
            packed_values=payload,
            scales=np.array([[1]], dtype=np.float32),
            original_shape=(1, 1),
            bit_width=2,
            group_size=1,
        )


def test_quantized_linear_matches_explicit_dequantization_and_reports_error() -> None:
    weights = np.array([[0.2, -0.6, 0.8], [0.0, 0.0, 0.0]], dtype=np.float32)
    inputs = np.array([[1.0, 2.0, -1.0]], dtype=np.float32)
    quantized = quantize_symmetric_groupwise(weights, bit_width=4, group_size=2)

    actual = quantized_linear(inputs, quantized, bias=[0.25, -0.5])
    expected = inputs @ quantized.dequantize().T + np.array([0.25, -0.5])
    np.testing.assert_allclose(actual, expected)
    report = quantization_error(weights, quantized.dequantize())
    assert report.mean_absolute_error >= 0
    assert report.root_mean_squared_error >= report.mean_absolute_error
    assert report.maximum_absolute_error >= report.root_mean_squared_error
    assert report.relative_l2_error is not None
    assert quantization_error(np.zeros(2), np.zeros(2)).relative_l2_error is None


@pytest.mark.parametrize(
    ("weights", "bit_width", "group_size"),
    [
        (np.array([]), 4, 2),
        (np.array([1.0, 2.0]), 4, 2),
        (np.array([[np.nan]]), 4, 2),
        (np.ones((1, 2)), 1, 2),
        (np.ones((1, 2)), 9, 2),
        (np.ones((1, 2)), 4, 0),
    ],
)
def test_quantizer_rejects_invalid_contracts(
    weights: np.ndarray, bit_width: int, group_size: int
) -> None:
    with pytest.raises(ValueError):
        quantize_symmetric_groupwise(
            weights, bit_width=bit_width, group_size=group_size
        )


def test_quantized_matrix_constructor_and_linear_input_fail_closed() -> None:
    with pytest.raises(ValueError, match="outside"):
        GroupwiseQuantizedMatrix(
            values=np.array([[8]], dtype=np.int8),
            scales=np.array([[1]], dtype=np.float32),
            original_shape=(1, 1),
            bit_width=4,
            group_size=1,
        )
    quantized = quantize_symmetric_groupwise([[1.0, 2.0]], group_size=2)
    with pytest.raises(ValueError, match="last dimension"):
        quantized_linear([[1.0]], quantized)
    with pytest.raises(ValueError, match="bias"):
        quantized_linear([[1.0, 2.0]], quantized, bias=[1.0, 2.0])


def test_quantization_toy_emits_storage_error_and_scope_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "matrix.allmqtz"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "projects"
                / "inference-serving"
                / "quantization_toy.py"
            ),
            "--seed",
            "7",
            "--output-features",
            "3",
            "--input-features",
            "5",
            "--batch-size",
            "2",
            "--group-size",
            "4",
            "--artifact-path",
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["configuration"]["code_range"] == [-7, 7]
    assert artifact["schema_version"] == 2
    assert artifact["storage"] == {
        "reference_fp32_weight_bytes": 60,
        "ideal_packed_weight_bytes": 8,
        "float32_scale_metadata_bytes": 24,
        "ideal_total_bytes": 32,
        "ideal_compression_ratio": pytest.approx(60 / 32),
        "numpy_unpacked_reference_bytes": 39,
        "actual_dense_packed_weight_bytes": 8,
        "actual_code_plus_fp32_scale_payload_bytes": 32,
        "final_byte_padding_bits": 4,
        "raw_payload_includes_container_header_or_alignment": False,
        "serialized_tensor_artifact_bytes": 96,
    }
    assert len(bytes.fromhex(artifact["packed_codes"]["hex"])) == 8
    assert artifact["packed_codes"]["sha256"].startswith("sha256:")
    assert artifact["packed_codes"]["round_trip_codes_exact"] is True
    assert artifact["packed_codes"]["round_trip_scales_exact"] is True
    assert artifact["tensor_artifact"] == {
        "format_version": 1,
        "byte_order": "little-endian header and float32 scales",
        "integrity": "trailing unkeyed SHA-256 over header and payload",
        "sha256": "sha256:"
        + hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "round_trip_exact": True,
        "written_to_disk": True,
        "path": str(artifact_path),
    }
    written_bytes = artifact_path.read_bytes()
    reloaded = PackedGroupwiseQuantizedMatrix.from_bytes(written_bytes)
    assert reloaded.original_shape == (3, 5)
    second = subprocess.run(
        completed.args,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
    assert artifact_path.read_bytes() == written_bytes
    assert artifact["weight_error"]["root_mean_squared_error"] >= 0
    assert artifact["linear_output_error"]["root_mean_squared_error"] >= 0
    assert artifact["scope"] == {
        "device": "CPU",
        "weights": "seeded synthetic matrix",
        "quantizer": "symmetric absmax per contiguous row group",
        "actual_low_bit_packing_executed": True,
        "self_contained_quantized_tensor_artifact_constructed": True,
        "self_contained_model_artifact_written": False,
        "fused_low_bit_kernel_executed": False,
        "calibration_or_gptq_awq_executed": False,
        "model_quality_or_latency_proved": False,
    }
