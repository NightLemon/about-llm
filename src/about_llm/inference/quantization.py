"""Transparent CPU reference for symmetric group-wise weight quantization."""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

QUANTIZED_MATRIX_FORMAT_VERSION = 1
_QUANTIZED_MATRIX_MAGIC = b"ALLMQTZ1"
_OFFSET_BINARY_LSB_FIRST = 1
_LITTLE_ENDIAN_FLOAT32 = 1
_QUANTIZED_MATRIX_HEADER = struct.Struct("<8sBBBBIIIII")
_SHA256_BYTES = 32
_UINT32_MAX = 2**32 - 1


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class QuantizationError:
    mean_absolute_error: float
    root_mean_squared_error: float
    maximum_absolute_error: float
    relative_l2_error: float | None


@dataclass(frozen=True)
class GroupwiseQuantizedMatrix:
    """Unpacked teaching representation plus an ideal packed-storage ledger.

    ``values`` uses int8 so NumPy can inspect every code. The ideal byte
    properties instead describe a dense bit stream and must not be confused
    with ``values.nbytes`` or a runtime-specific packed tensor layout.
    """

    values: NDArray[np.int8]
    scales: NDArray[np.float32]
    original_shape: tuple[int, int]
    bit_width: int
    group_size: int

    def __post_init__(self) -> None:
        if len(self.original_shape) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.original_shape
        ):
            raise ValueError("original_shape must contain two positive integers")
        _positive_integer(self.group_size, "group_size")
        if (
            isinstance(self.bit_width, bool)
            or not isinstance(self.bit_width, int)
            or not 2 <= self.bit_width <= 8
        ):
            raise ValueError("bit_width must be an integer from 2 through 8")
        rows, columns = self.original_shape
        groups_per_row = math.ceil(columns / self.group_size)
        if self.values.shape != self.original_shape or self.values.dtype != np.int8:
            raise ValueError("values must be an int8 matrix matching original_shape")
        if self.scales.shape != (rows, groups_per_row):
            raise ValueError("scales shape does not match rows and quantization groups")
        if self.scales.dtype != np.float32:
            raise ValueError("scales must use float32")
        if not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("scales must be finite and positive")
        qmax = self.maximum_code
        if np.any(self.values < -qmax) or np.any(self.values > qmax):
            raise ValueError("values contain a code outside the symmetric range")

        values = self.values.copy()
        scales = self.scales.copy()
        values.setflags(write=False)
        scales.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "scales", scales)

    @property
    def maximum_code(self) -> int:
        return int(2 ** (self.bit_width - 1) - 1)

    @property
    def groups_per_row(self) -> int:
        return int(self.scales.shape[1])

    @property
    def reference_fp32_weight_bytes(self) -> int:
        return int(self.values.size * np.dtype(np.float32).itemsize)

    @property
    def ideal_packed_weight_bytes(self) -> int:
        """Dense-bitstream lower bound, excluding alignment and container headers."""

        return math.ceil(self.values.size * self.bit_width / 8)

    @property
    def scale_metadata_bytes(self) -> int:
        return int(self.scales.nbytes)

    @property
    def ideal_total_bytes(self) -> int:
        return self.ideal_packed_weight_bytes + self.scale_metadata_bytes

    @property
    def ideal_compression_ratio(self) -> float:
        return self.reference_fp32_weight_bytes / self.ideal_total_bytes

    @property
    def unpacked_reference_bytes(self) -> int:
        return int(self.values.nbytes + self.scales.nbytes)

    def dequantize(self) -> NDArray[np.float32]:
        reconstructed = np.empty(self.original_shape, dtype=np.float32)
        for group_index in range(self.groups_per_row):
            start = group_index * self.group_size
            stop = min(start + self.group_size, self.original_shape[1])
            reconstructed[:, start:stop] = (
                self.values[:, start:stop] * self.scales[:, group_index, None]
            )
        return reconstructed

    def pack(self) -> PackedGroupwiseQuantizedMatrix:
        """Pack row-major signed codes into the canonical dense teaching stream."""

        return PackedGroupwiseQuantizedMatrix(
            packed_values=_pack_signed_codes(
                self.values,
                bit_width=self.bit_width,
                maximum_code=self.maximum_code,
            ),
            scales=self.scales,
            original_shape=self.original_shape,
            bit_width=self.bit_width,
            group_size=self.group_size,
        )


@dataclass(frozen=True)
class PackedGroupwiseQuantizedMatrix:
    """Canonical dense code stream plus FP32 scales for the CPU reference.

    Signed ``q`` is stored as unsigned ``q + qmax``. Codes are flattened in C
    row-major order and each code contributes ``bit_width`` least-significant
    bits to consecutive low-to-high bit positions in the byte stream. The
    all-ones unsigned code is unused, and high padding bits in the last byte
    must be zero. This is a teaching payload, not a runtime-specific model
    container or a fused low-bit tensor layout.
    """

    packed_values: bytes
    scales: NDArray[np.float32]
    original_shape: tuple[int, int]
    bit_width: int
    group_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.packed_values, bytes):
            raise TypeError("packed_values must be immutable bytes")
        if len(self.original_shape) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.original_shape
        ):
            raise ValueError("original_shape must contain two positive integers")
        _positive_integer(self.group_size, "group_size")
        if (
            isinstance(self.bit_width, bool)
            or not isinstance(self.bit_width, int)
            or not 2 <= self.bit_width <= 8
        ):
            raise ValueError("bit_width must be an integer from 2 through 8")
        rows, columns = self.original_shape
        groups_per_row = math.ceil(columns / self.group_size)
        if self.scales.shape != (rows, groups_per_row):
            raise ValueError("scales shape does not match rows and quantization groups")
        if self.scales.dtype != np.float32:
            raise ValueError("scales must use float32")
        if not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("scales must be finite and positive")
        if len(self.packed_values) != self.packed_weight_bytes:
            raise ValueError("packed_values length does not match shape and bit width")
        _validate_zero_padding(
            self.packed_values,
            value_count=self.value_count,
            bit_width=self.bit_width,
        )
        # Decode once at the trust boundary so the unused all-ones code cannot
        # hide in an otherwise length-correct payload.
        _unpack_signed_codes(
            self.packed_values,
            value_count=self.value_count,
            bit_width=self.bit_width,
            maximum_code=self.maximum_code,
        )
        scales = self.scales.copy()
        scales.setflags(write=False)
        object.__setattr__(self, "scales", scales)

    @property
    def maximum_code(self) -> int:
        return int(2 ** (self.bit_width - 1) - 1)

    @property
    def value_count(self) -> int:
        return self.original_shape[0] * self.original_shape[1]

    @property
    def groups_per_row(self) -> int:
        return int(self.scales.shape[1])

    @property
    def packed_weight_bytes(self) -> int:
        return math.ceil(self.value_count * self.bit_width / 8)

    @property
    def padding_bits(self) -> int:
        return self.packed_weight_bytes * 8 - self.value_count * self.bit_width

    @property
    def scale_metadata_bytes(self) -> int:
        return int(self.scales.nbytes)

    @property
    def raw_payload_bytes(self) -> int:
        """Code bytes plus scale bytes, excluding any container/header/alignment."""

        return self.packed_weight_bytes + self.scale_metadata_bytes

    @property
    def packed_values_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.packed_values).hexdigest()

    @property
    def serialized_artifact_bytes(self) -> int:
        return len(self.to_bytes())

    @property
    def serialized_artifact_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_bytes()).hexdigest()

    def unpack(self) -> GroupwiseQuantizedMatrix:
        values = _unpack_signed_codes(
            self.packed_values,
            value_count=self.value_count,
            bit_width=self.bit_width,
            maximum_code=self.maximum_code,
        ).reshape(self.original_shape)
        return GroupwiseQuantizedMatrix(
            values=values,
            scales=self.scales,
            original_shape=self.original_shape,
            bit_width=self.bit_width,
            group_size=self.group_size,
        )

    def dequantize(self) -> NDArray[np.float32]:
        """Unpack and dequantize; this still does not execute a low-bit kernel."""

        return self.unpack().dequantize()

    def to_bytes(self) -> bytes:
        """Serialize one matrix with a strict little-endian v1 header and digest.

        The trailing SHA-256 detects accidental or relative-to-a-trusted-copy
        modification. It is not a signature or an authenticity guarantee.
        """

        rows, columns = self.original_shape
        scale_count = int(self.scales.size)
        header_integers = (rows, columns, self.group_size, len(self.packed_values), scale_count)
        if any(value > _UINT32_MAX for value in header_integers):
            raise ValueError("quantized matrix fields exceed the v1 uint32 format")
        header = _QUANTIZED_MATRIX_HEADER.pack(
            _QUANTIZED_MATRIX_MAGIC,
            QUANTIZED_MATRIX_FORMAT_VERSION,
            self.bit_width,
            _OFFSET_BINARY_LSB_FIRST,
            _LITTLE_ENDIAN_FLOAT32,
            *header_integers,
        )
        scale_bytes = self.scales.astype("<f4", copy=False).tobytes(order="C")
        body = header + self.packed_values + scale_bytes
        return body + hashlib.sha256(body).digest()

    @classmethod
    def from_bytes(cls, artifact: bytes) -> PackedGroupwiseQuantizedMatrix:
        """Strictly reload one v1 matrix and reject drift, truncation, or trailing data."""

        if not isinstance(artifact, bytes):
            raise TypeError("quantized matrix artifact must be immutable bytes")
        minimum_size = _QUANTIZED_MATRIX_HEADER.size + _SHA256_BYTES
        if len(artifact) < minimum_size:
            raise ValueError("quantized matrix artifact is truncated")
        (
            magic,
            format_version,
            bit_width,
            code_mapping,
            scale_dtype,
            rows,
            columns,
            group_size,
            packed_length,
            scale_count,
        ) = _QUANTIZED_MATRIX_HEADER.unpack_from(artifact)
        if magic != _QUANTIZED_MATRIX_MAGIC:
            raise ValueError("quantized matrix artifact has unknown magic")
        if format_version != QUANTIZED_MATRIX_FORMAT_VERSION:
            raise ValueError("quantized matrix artifact has unsupported format version")
        if code_mapping != _OFFSET_BINARY_LSB_FIRST:
            raise ValueError("quantized matrix artifact has unsupported code mapping")
        if scale_dtype != _LITTLE_ENDIAN_FLOAT32:
            raise ValueError("quantized matrix artifact has unsupported scale dtype")
        if rows == 0 or columns == 0 or group_size == 0:
            raise ValueError("quantized matrix artifact dimensions must be positive")
        if not 2 <= bit_width <= 8:
            raise ValueError("quantized matrix artifact bit width must be in [2, 8]")
        expected_packed_length = math.ceil(rows * columns * bit_width / 8)
        expected_scale_count = rows * math.ceil(columns / group_size)
        if packed_length != expected_packed_length:
            raise ValueError("quantized matrix artifact packed length is inconsistent")
        if scale_count != expected_scale_count:
            raise ValueError("quantized matrix artifact scale count is inconsistent")
        scale_length = scale_count * np.dtype("<f4").itemsize
        expected_length = (
            _QUANTIZED_MATRIX_HEADER.size
            + packed_length
            + scale_length
            + _SHA256_BYTES
        )
        if len(artifact) != expected_length:
            raise ValueError("quantized matrix artifact length or trailing data is invalid")
        body = artifact[:-_SHA256_BYTES]
        if not hmac.compare_digest(hashlib.sha256(body).digest(), artifact[-_SHA256_BYTES:]):
            raise ValueError("quantized matrix artifact SHA-256 mismatch")
        packed_start = _QUANTIZED_MATRIX_HEADER.size
        packed_stop = packed_start + packed_length
        scale_stop = packed_stop + scale_length
        packed_values = artifact[packed_start:packed_stop]
        scales = (
            np.frombuffer(artifact[packed_stop:scale_stop], dtype="<f4")
            .astype(np.float32, copy=True)
            .reshape((rows, expected_scale_count // rows))
        )
        return cls(
            packed_values=packed_values,
            scales=scales,
            original_shape=(rows, columns),
            bit_width=bit_width,
            group_size=group_size,
        )


def quantize_symmetric_groupwise(
    weights: ArrayLike,
    *,
    bit_width: int = 4,
    group_size: int = 128,
) -> GroupwiseQuantizedMatrix:
    """Quantize each contiguous row group with one absmax-derived scale.

    Codes use ``[-qmax, qmax]`` where ``qmax = 2**(bits - 1) - 1``. This is
    a symmetric convention, so the most-negative two's-complement code is
    deliberately unused. An all-zero group receives scale 1 and stays zero.
    """

    group_size = _positive_integer(group_size, "group_size")
    if (
        isinstance(bit_width, bool)
        or not isinstance(bit_width, int)
        or not 2 <= bit_width <= 8
    ):
        raise ValueError("bit_width must be an integer from 2 through 8")
    matrix = np.asarray(weights, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0 or 0 in matrix.shape:
        raise ValueError("weights must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weights must contain only finite values")

    rows, columns = matrix.shape
    groups_per_row = math.ceil(columns / group_size)
    values = np.empty(matrix.shape, dtype=np.int8)
    scales = np.empty((rows, groups_per_row), dtype=np.float32)
    qmax = 2 ** (bit_width - 1) - 1
    for group_index in range(groups_per_row):
        start = group_index * group_size
        stop = min(start + group_size, columns)
        group = matrix[:, start:stop]
        maximum = np.max(np.abs(group), axis=1)
        scale = np.where(maximum == 0, 1.0, maximum / qmax).astype(np.float32)
        scales[:, group_index] = scale
        values[:, start:stop] = np.clip(
            np.rint(group / scale[:, None]), -qmax, qmax
        ).astype(np.int8)

    return GroupwiseQuantizedMatrix(
        values=values,
        scales=scales,
        original_shape=(rows, columns),
        bit_width=bit_width,
        group_size=group_size,
    )


def _pack_signed_codes(
    values: NDArray[np.int8],
    *,
    bit_width: int,
    maximum_code: int,
) -> bytes:
    """Pack offset-binary codes into an LSB-first dense stream."""

    output = bytearray()
    buffer = 0
    buffered_bits = 0
    for signed_value in values.ravel(order="C"):
        unsigned_code = int(signed_value) + maximum_code
        buffer |= unsigned_code << buffered_bits
        buffered_bits += bit_width
        while buffered_bits >= 8:
            output.append(buffer & 0xFF)
            buffer >>= 8
            buffered_bits -= 8
    if buffered_bits:
        output.append(buffer & 0xFF)
    return bytes(output)


def _unpack_signed_codes(
    payload: bytes,
    *,
    value_count: int,
    bit_width: int,
    maximum_code: int,
) -> NDArray[np.int8]:
    """Decode the canonical stream and reject the unused all-ones code."""

    mask = (1 << bit_width) - 1
    values = np.empty(value_count, dtype=np.int8)
    buffer = 0
    buffered_bits = 0
    byte_index = 0
    for value_index in range(value_count):
        while buffered_bits < bit_width:
            buffer |= payload[byte_index] << buffered_bits
            byte_index += 1
            buffered_bits += 8
        unsigned_code = buffer & mask
        buffer >>= bit_width
        buffered_bits -= bit_width
        if unsigned_code > 2 * maximum_code:
            raise ValueError("packed_values contain the unused all-ones code")
        values[value_index] = unsigned_code - maximum_code
    return values


def _validate_zero_padding(
    payload: bytes,
    *,
    value_count: int,
    bit_width: int,
) -> None:
    used_final_bits = value_count * bit_width % 8
    if used_final_bits and payload[-1] >> used_final_bits:
        raise ValueError("packed_values contain non-zero high padding bits")


def quantization_error(
    reference: ArrayLike,
    reconstructed: ArrayLike,
) -> QuantizationError:
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(reconstructed, dtype=np.float64)
    if expected.shape != observed.shape or expected.size == 0:
        raise ValueError("reference and reconstructed must have the same non-empty shape")
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(observed)):
        raise ValueError("error inputs must contain only finite values")
    difference = observed - expected
    absolute = np.abs(difference)
    denominator = float(np.linalg.norm(expected.ravel(), ord=2))
    relative_l2 = (
        None
        if denominator == 0
        else float(np.linalg.norm(difference.ravel(), ord=2) / denominator)
    )
    return QuantizationError(
        mean_absolute_error=float(np.mean(absolute)),
        root_mean_squared_error=float(np.sqrt(np.mean(np.square(difference)))),
        maximum_absolute_error=float(np.max(absolute)),
        relative_l2_error=relative_l2,
    )


def quantized_linear(
    inputs: ArrayLike,
    weights: GroupwiseQuantizedMatrix,
    *,
    bias: ArrayLike | None = None,
) -> NDArray[np.float32]:
    """Compute with dequantized weights; this is not a fused low-bit kernel."""

    activations = np.asarray(inputs, dtype=np.float32)
    if activations.ndim == 0 or activations.shape[-1] != weights.original_shape[1]:
        raise ValueError("inputs last dimension must equal weight input features")
    if not np.all(np.isfinite(activations)):
        raise ValueError("inputs must contain only finite values")
    output = np.matmul(activations, weights.dequantize().T)
    if bias is not None:
        offset = np.asarray(bias, dtype=np.float32)
        if offset.shape != (weights.original_shape[0],):
            raise ValueError("bias must match weight output features")
        if not np.all(np.isfinite(offset)):
            raise ValueError("bias must contain only finite values")
        output = output + offset
    return np.asarray(output, dtype=np.float32)
