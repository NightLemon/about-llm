"""Strict multi-matrix bundle for the transparent CPU quantization reference."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from about_llm.inference.quantization import (
    QUANTIZED_MATRIX_FORMAT_VERSION,
    PackedGroupwiseQuantizedMatrix,
)
from about_llm.llmops import canonical_json_bytes

QUANTIZED_BUNDLE_FORMAT_VERSION = 1
QUANTIZED_BUNDLE_SCHEMA_VERSION = "about-llm.quantized-matrix-bundle.v1"
_BUNDLE_MAGIC = b"ALLMQB01"
_BUNDLE_HEADER = struct.Struct("<8sB3xIII")
_SHA256_BYTES = 32
_UINT32_MAX = (1 << 32) - 1
_IDENTITY_FIELDS = {
    "architecture_config",
    "model_family",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
}
_MANIFEST_FIELDS = {"identity", "schema_version", "tensors"}
_TENSOR_FIELDS = {
    "length",
    "matrix_format_version",
    "name",
    "offset",
    "sha256",
}


@dataclass(frozen=True)
class QuantizedBundleLimits:
    max_artifact_bytes: int = _UINT32_MAX
    max_manifest_bytes: int = 16 * 1024 * 1024
    max_tensors: int = 100_000
    max_tensor_bytes: int = _UINT32_MAX

    def __post_init__(self) -> None:
        for name, value in (
            ("max_artifact_bytes", self.max_artifact_bytes),
            ("max_manifest_bytes", self.max_manifest_bytes),
            ("max_tensors", self.max_tensors),
            ("max_tensor_bytes", self.max_tensor_bytes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= _UINT32_MAX
            ):
                raise ValueError(f"{name} must be an integer in [1, 2^32-1]")


@dataclass(frozen=True)
class QuantizedBundleIdentity:
    model_family: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    architecture_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "model_family",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value.strip() != value
                or "\x00" in value
                or len(value.encode("utf-8")) > 4096
            ):
                raise ValueError(
                    f"{name} must be trimmed, NUL-free, and <=4096 UTF-8 bytes"
                )
        if not isinstance(self.architecture_config, Mapping):
            raise TypeError("architecture_config must be a JSON object mapping")
        try:
            snapshot = json.loads(canonical_json_bytes(self.architecture_config))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("architecture_config must be finite JSON") from error
        if not isinstance(snapshot, dict) or not snapshot:
            raise ValueError("architecture_config must be a non-empty JSON object")
        object.__setattr__(self, "architecture_config", _freeze_object(snapshot))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "architecture_config": _thaw(self.architecture_config),
        }


@dataclass(frozen=True)
class NamedQuantizedMatrix:
    name: str
    matrix: PackedGroupwiseQuantizedMatrix

    def __post_init__(self) -> None:
        _validate_tensor_name(self.name)
        if not isinstance(self.matrix, PackedGroupwiseQuantizedMatrix):
            raise TypeError("matrix must be PackedGroupwiseQuantizedMatrix")


@dataclass(frozen=True)
class QuantizedMatrixBundle:
    identity: QuantizedBundleIdentity
    tensors: tuple[NamedQuantizedMatrix, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, QuantizedBundleIdentity):
            raise TypeError("identity must be QuantizedBundleIdentity")
        tensors = tuple(self.tensors)
        if not tensors or any(not isinstance(item, NamedQuantizedMatrix) for item in tensors):
            raise ValueError("tensors must contain NamedQuantizedMatrix values")
        names = tuple(item.name for item in tensors)
        if len(names) != len(set(names)):
            raise ValueError("tensor names must be unique")
        object.__setattr__(
            self, "tensors", tuple(sorted(tensors, key=lambda item: item.name))
        )

    def get(self, name: str) -> PackedGroupwiseQuantizedMatrix:
        if not isinstance(name, str) or not name:
            raise ValueError("tensor name must be a non-empty string")
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor.matrix
        raise KeyError(name)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.tensors)

    @property
    def serialized_artifact_bytes(self) -> int:
        return len(self.to_bytes())

    @property
    def serialized_artifact_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_bytes()).hexdigest()

    def to_bytes(
        self, *, limits: QuantizedBundleLimits | None = None
    ) -> bytes:
        exact_limits = _limits(limits)
        if len(self.tensors) > exact_limits.max_tensors:
            raise ValueError("tensor count exceeds configured bundle limit")

        payload_parts: list[bytes] = []
        descriptors: list[dict[str, Any]] = []
        offset = 0
        for item in self.tensors:
            tensor_bytes = item.matrix.to_bytes()
            if len(tensor_bytes) > exact_limits.max_tensor_bytes:
                raise ValueError("tensor artifact exceeds configured bundle limit")
            if offset + len(tensor_bytes) > _UINT32_MAX:
                raise ValueError("bundle payload exceeds v1 uint32 range")
            descriptors.append(
                {
                    "name": item.name,
                    "matrix_format_version": QUANTIZED_MATRIX_FORMAT_VERSION,
                    "offset": offset,
                    "length": len(tensor_bytes),
                    "sha256": "sha256:" + hashlib.sha256(tensor_bytes).hexdigest(),
                }
            )
            payload_parts.append(tensor_bytes)
            offset += len(tensor_bytes)

        manifest = canonical_json_bytes(
            {
                "schema_version": QUANTIZED_BUNDLE_SCHEMA_VERSION,
                "identity": self.identity.to_dict(),
                "tensors": descriptors,
            }
        )
        payload = b"".join(payload_parts)
        if len(manifest) > exact_limits.max_manifest_bytes:
            raise ValueError("bundle manifest exceeds configured limit")
        if len(manifest) > _UINT32_MAX or len(payload) > _UINT32_MAX:
            raise ValueError("bundle manifest or payload exceeds v1 uint32 range")
        header = _BUNDLE_HEADER.pack(
            _BUNDLE_MAGIC,
            QUANTIZED_BUNDLE_FORMAT_VERSION,
            len(manifest),
            len(self.tensors),
            len(payload),
        )
        body = header + manifest + payload
        artifact = body + hashlib.sha256(body).digest()
        if len(artifact) > exact_limits.max_artifact_bytes:
            raise ValueError("bundle artifact exceeds configured limit")
        return artifact

    @classmethod
    def from_bytes(
        cls,
        artifact: bytes,
        *,
        limits: QuantizedBundleLimits | None = None,
    ) -> QuantizedMatrixBundle:
        if not isinstance(artifact, bytes):
            raise TypeError("bundle artifact must be immutable bytes")
        exact_limits = _limits(limits)
        if len(artifact) > exact_limits.max_artifact_bytes:
            raise ValueError("bundle artifact exceeds configured limit")
        minimum = _BUNDLE_HEADER.size + _SHA256_BYTES
        if len(artifact) < minimum:
            raise ValueError("bundle artifact is truncated")
        magic, version, manifest_length, tensor_count, payload_length = (
            _BUNDLE_HEADER.unpack_from(artifact)
        )
        if magic != _BUNDLE_MAGIC:
            raise ValueError("invalid quantized bundle magic")
        if version != QUANTIZED_BUNDLE_FORMAT_VERSION:
            raise ValueError("unsupported quantized bundle format version")
        if manifest_length > exact_limits.max_manifest_bytes:
            raise ValueError("bundle manifest exceeds configured limit")
        if tensor_count == 0 or tensor_count > exact_limits.max_tensors:
            raise ValueError("bundle tensor count is invalid or exceeds limit")
        expected_length = (
            _BUNDLE_HEADER.size + manifest_length + payload_length + _SHA256_BYTES
        )
        if len(artifact) != expected_length:
            if len(artifact) > expected_length:
                raise ValueError("bundle artifact contains trailing data")
            raise ValueError("bundle artifact length is inconsistent or truncated")
        body = artifact[:-_SHA256_BYTES]
        if not hmac.compare_digest(hashlib.sha256(body).digest(), artifact[-32:]):
            raise ValueError("bundle SHA-256 mismatch")

        manifest_start = _BUNDLE_HEADER.size
        payload_start = manifest_start + manifest_length
        manifest_bytes = artifact[manifest_start:payload_start]
        payload = artifact[payload_start:-_SHA256_BYTES]
        manifest = _strict_manifest(manifest_bytes)
        descriptors = manifest["tensors"]
        if not isinstance(descriptors, list) or len(descriptors) != tensor_count:
            raise ValueError("bundle tensor descriptor count is inconsistent")

        identity = _parse_identity(manifest["identity"])
        tensors: list[NamedQuantizedMatrix] = []
        expected_offset = 0
        previous_name: str | None = None
        for descriptor in descriptors:
            parsed = _parse_descriptor(descriptor)
            name = cast(str, parsed["name"])
            offset = cast(int, parsed["offset"])
            length = cast(int, parsed["length"])
            if previous_name is not None and name <= previous_name:
                raise ValueError("bundle tensor descriptors must be name-sorted and unique")
            if offset != expected_offset:
                raise ValueError("bundle tensor offsets must be contiguous")
            if length > exact_limits.max_tensor_bytes:
                raise ValueError("tensor artifact exceeds configured bundle limit")
            stop = offset + length
            if stop > len(payload):
                raise ValueError("bundle tensor range exceeds payload")
            tensor_bytes = payload[offset:stop]
            if not hmac.compare_digest(
                "sha256:" + hashlib.sha256(tensor_bytes).hexdigest(),
                cast(str, parsed["sha256"]),
            ):
                raise ValueError("bundle tensor SHA-256 mismatch")
            tensors.append(
                NamedQuantizedMatrix(
                    name,
                    PackedGroupwiseQuantizedMatrix.from_bytes(tensor_bytes),
                )
            )
            expected_offset = stop
            previous_name = name
        if expected_offset != len(payload):
            raise ValueError("bundle payload has unreferenced trailing bytes")
        return cls(identity=identity, tensors=tuple(tensors))

    def write_new(
        self,
        path: Path,
        *,
        limits: QuantizedBundleLimits | None = None,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be pathlib.Path")
        artifact = self.to_bytes(limits=limits)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(artifact)
            stream.flush()
            os.fsync(stream.fileno())

    @classmethod
    def read(
        cls,
        path: Path,
        *,
        limits: QuantizedBundleLimits | None = None,
    ) -> QuantizedMatrixBundle:
        if not isinstance(path, Path):
            raise TypeError("path must be pathlib.Path")
        exact_limits = _limits(limits)
        with path.open("rb") as stream:
            file_size = os.fstat(stream.fileno()).st_size
            if file_size > exact_limits.max_artifact_bytes:
                raise ValueError("bundle file exceeds configured artifact limit")
            artifact = stream.read(exact_limits.max_artifact_bytes + 1)
        if len(artifact) != file_size:
            raise ValueError("bundle file changed or could not be read completely")
        return cls.from_bytes(artifact, limits=exact_limits)


def _limits(value: QuantizedBundleLimits | None) -> QuantizedBundleLimits:
    if value is None:
        return QuantizedBundleLimits()
    if not isinstance(value, QuantizedBundleLimits):
        raise TypeError("limits must be QuantizedBundleLimits")
    return value


def _strict_manifest(value: bytes) -> dict[str, Any]:
    try:
        decoded = value.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: _invalid_constant(constant),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("bundle manifest must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("bundle manifest fields are invalid")
    if payload["schema_version"] != QUANTIZED_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported quantized bundle schema version")
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("bundle manifest contains unsupported JSON") from error
    if canonical != value:
        raise ValueError("bundle manifest must use canonical JSON encoding")
    return payload


def _parse_identity(value: Any) -> QuantizedBundleIdentity:
    if not isinstance(value, dict) or set(value) != _IDENTITY_FIELDS:
        raise ValueError("bundle identity fields are invalid")
    return QuantizedBundleIdentity(
        model_family=_string(value["model_family"], "model_family"),
        model_revision=_string(value["model_revision"], "model_revision"),
        tokenizer_id=_string(value["tokenizer_id"], "tokenizer_id"),
        tokenizer_revision=_string(value["tokenizer_revision"], "tokenizer_revision"),
        architecture_config=value["architecture_config"],
    )


def _parse_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _TENSOR_FIELDS:
        raise ValueError("bundle tensor descriptor fields are invalid")
    if value["matrix_format_version"] != QUANTIZED_MATRIX_FORMAT_VERSION:
        raise ValueError("unsupported bundled matrix format version")
    name = _string(value["name"], "tensor name")
    _validate_tensor_name(name)
    offset = _integer(value["offset"], "tensor offset")
    length = _integer(value["length"], "tensor length")
    if length <= 0:
        raise ValueError("tensor length must be positive")
    digest = _sha256(value["sha256"], "tensor sha256")
    return {"name": name, "offset": offset, "length": length, "sha256": digest}


def _validate_tensor_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
        or len(value.encode("utf-8")) > 1024
    ):
        raise ValueError(
            "tensor name must be trimmed, non-empty, NUL-free, and <=1024 UTF-8 bytes"
        )
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return cast(int, value)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be canonical sha256:<lowercase-hex>")
    return value


def _freeze_object(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _freeze_object(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("identity contains a non-finite float")
    return value
