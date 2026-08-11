"""Strict repo-native checkpoint for the teaching MiniGPT inference model."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT, TransformerBlock
from about_llm.from_scratch.tokenizer import ByteBPETokenizer
from about_llm.inference.quantization import (
    QUANTIZED_MATRIX_FORMAT_VERSION,
    PackedGroupwiseQuantizedMatrix,
    quantize_symmetric_groupwise,
)
from about_llm.llmops import canonical_json_bytes

MINIGPT_CHECKPOINT_FORMAT_VERSION = 1
MINIGPT_CHECKPOINT_SCHEMA_VERSION = "about-llm.minigpt-checkpoint.v1"
MINIGPT_ARCHITECTURE_ID = "about-llm.minigpt"
MINIGPT_ARCHITECTURE_REVISION = "about-llm.minigpt.forward.v1"
MINIGPT_TOKENIZER_KIND = "about-llm.byte-bpe"
MINIGPT_TOKENIZER_FORMAT_VERSION = 1

_MAGIC = b"ALLMGPT1"
_HEADER = struct.Struct("<8sB3xIII")
_SHA256_BYTES = 32
_UINT32_MAX = (1 << 32) - 1
_QUANTIZED_KIND = "groupwise-quantized-matrix-v1"
_FLOAT32_VECTOR_KIND = "little-endian-float32-vector-v1"
_TIED_PARAMETERS = (
    {"alias": "lm_head.weight", "target": "token_embedding.weight"},
)
_MANIFEST_FIELDS = {
    "architecture",
    "config",
    "identity",
    "parameters",
    "quantization",
    "schema_version",
    "tied_parameters",
    "tokenizer",
}
_ARCHITECTURE_FIELDS = {"id", "revision"}
_IDENTITY_FIELDS = {"model_id", "model_revision", "tokenizer_revision"}
_CONFIG_FIELDS = {
    "bias",
    "context_length",
    "dropout",
    "mlp_ratio",
    "model_dim",
    "num_heads",
    "num_layers",
    "vocab_size",
}
_TOKENIZER_FIELDS = {"format_version", "kind", "merges", "vocab_size"}
_QUANTIZATION_FIELDS = {
    "bit_width",
    "group_size",
    "matrix_format_version",
    "method",
}
_PARAMETER_FIELDS = {"kind", "length", "name", "offset", "sha256", "shape"}


@dataclass(frozen=True)
class MiniGPTCheckpointLimits:
    """Allocation and parsing limits applied before model construction."""

    max_artifact_bytes: int = 512 * 1024 * 1024
    max_manifest_bytes: int = 16 * 1024 * 1024
    max_parameters: int = 100_000
    max_parameter_bytes: int = 256 * 1024 * 1024
    max_model_parameter_count: int = 25_000_000
    max_tokenizer_merges: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= _UINT32_MAX
            ):
                raise ValueError(f"{name} must be an integer in [1, 2^32-1]")


@dataclass(frozen=True)
class MiniGPTCheckpointIdentity:
    model_id: str
    model_revision: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "tokenizer_revision"):
            _validate_identity_string(getattr(self, name), name)


@dataclass(frozen=True)
class LoadedMiniGPTCheckpoint:
    """A restored tokenizer and dequantized repo-native MiniGPT inference model."""

    identity: MiniGPTCheckpointIdentity
    config: GPTConfig
    tokenizer: ByteBPETokenizer
    model: MiniGPT
    bit_width: int
    group_size: int
    artifact_sha256: str
    serialized_artifact_bytes: int


def serialize_quantized_minigpt_checkpoint(
    model: MiniGPT,
    tokenizer: ByteBPETokenizer,
    *,
    identity: MiniGPTCheckpointIdentity,
    bit_width: int,
    group_size: int,
    limits: MiniGPTCheckpointLimits | None = None,
) -> bytes:
    """Serialize every unique MiniGPT parameter plus the Byte-BPE payload."""

    exact_limits = _limits(limits)
    if type(model) is not MiniGPT:
        raise TypeError("model must be MiniGPT")
    if type(tokenizer) is not ByteBPETokenizer:
        raise TypeError("tokenizer must be ByteBPETokenizer")
    if not isinstance(identity, MiniGPTCheckpointIdentity):
        raise TypeError("identity must be MiniGPTCheckpointIdentity")
    _validate_quantization(bit_width, group_size)
    config = model.config
    config_payload = _config_payload(config)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary must match MiniGPT config")
    if len(tokenizer.merges) > exact_limits.max_tokenizer_merges:
        raise ValueError("tokenizer merge count exceeds configured limit")

    expected_shapes = _expected_parameter_shapes(config)
    _validate_model_size(expected_shapes, exact_limits)
    _validate_runtime_model(model, expected_shapes)
    named_parameters = dict(model.named_parameters())
    if len(named_parameters) > exact_limits.max_parameters:
        raise ValueError("parameter count exceeds configured checkpoint limit")
    payload_parts: list[bytes] = []
    descriptors: list[dict[str, Any]] = []
    offset = 0
    for name in sorted(named_parameters):
        parameter = named_parameters[name]
        if parameter.dtype != torch.float32:
            raise ValueError(f"parameter {name} must be float32")
        values = parameter.detach().cpu().numpy()
        if tuple(values.shape) != expected_shapes[name]:
            raise ValueError(f"parameter {name} shape does not match config")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"parameter {name} must contain only finite values")
        if values.ndim == 2:
            kind = _QUANTIZED_KIND
            parameter_bytes = quantize_symmetric_groupwise(
                values,
                bit_width=bit_width,
                group_size=group_size,
            ).pack().to_bytes()
        elif values.ndim == 1:
            kind = _FLOAT32_VECTOR_KIND
            parameter_bytes = np.asarray(values, dtype="<f4").tobytes(order="C")
        else:
            raise ValueError(f"parameter {name} rank is unsupported")
        if len(parameter_bytes) > exact_limits.max_parameter_bytes:
            raise ValueError(f"parameter {name} artifact exceeds configured limit")
        if offset + len(parameter_bytes) > _UINT32_MAX:
            raise ValueError("checkpoint payload exceeds v1 uint32 range")
        descriptors.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(values.shape),
                "offset": offset,
                "length": len(parameter_bytes),
                "sha256": "sha256:" + hashlib.sha256(parameter_bytes).hexdigest(),
            }
        )
        payload_parts.append(parameter_bytes)
        offset += len(parameter_bytes)

    manifest = canonical_json_bytes(
        {
            "schema_version": MINIGPT_CHECKPOINT_SCHEMA_VERSION,
            "architecture": {
                "id": MINIGPT_ARCHITECTURE_ID,
                "revision": MINIGPT_ARCHITECTURE_REVISION,
            },
            "identity": asdict(identity),
            "config": config_payload,
            "tokenizer": {
                "kind": MINIGPT_TOKENIZER_KIND,
                "format_version": MINIGPT_TOKENIZER_FORMAT_VERSION,
                "vocab_size": tokenizer.vocab_size,
                "merges": [list(pair) for pair in tokenizer.merges],
            },
            "quantization": {
                "method": "symmetric-absmax-contiguous-row-group",
                "bit_width": bit_width,
                "group_size": group_size,
                "matrix_format_version": QUANTIZED_MATRIX_FORMAT_VERSION,
            },
            "tied_parameters": list(_TIED_PARAMETERS),
            "parameters": descriptors,
        }
    )
    payload = b"".join(payload_parts)
    if len(manifest) > exact_limits.max_manifest_bytes:
        raise ValueError("checkpoint manifest exceeds configured limit")
    if len(manifest) > _UINT32_MAX or len(payload) > _UINT32_MAX:
        raise ValueError("checkpoint manifest or payload exceeds v1 uint32 range")
    header = _HEADER.pack(
        _MAGIC,
        MINIGPT_CHECKPOINT_FORMAT_VERSION,
        len(manifest),
        len(descriptors),
        len(payload),
    )
    body = header + manifest + payload
    artifact = body + hashlib.sha256(body).digest()
    if len(artifact) > exact_limits.max_artifact_bytes:
        raise ValueError("checkpoint artifact exceeds configured limit")
    return artifact


def load_quantized_minigpt_checkpoint(
    artifact: bytes,
    *,
    limits: MiniGPTCheckpointLimits | None = None,
) -> LoadedMiniGPTCheckpoint:
    """Strictly validate and restore a CPU float32 MiniGPT inference model."""

    if not isinstance(artifact, bytes):
        raise TypeError("checkpoint artifact must be immutable bytes")
    exact_limits = _limits(limits)
    if len(artifact) > exact_limits.max_artifact_bytes:
        raise ValueError("checkpoint artifact exceeds configured limit")
    if len(artifact) < _HEADER.size + _SHA256_BYTES:
        raise ValueError("checkpoint artifact is truncated")
    magic, version, manifest_length, parameter_count, payload_length = (
        _HEADER.unpack_from(artifact)
    )
    if magic != _MAGIC:
        raise ValueError("invalid MiniGPT checkpoint magic")
    if version != MINIGPT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported MiniGPT checkpoint format version")
    if manifest_length > exact_limits.max_manifest_bytes:
        raise ValueError("checkpoint manifest exceeds configured limit")
    if parameter_count == 0 or parameter_count > exact_limits.max_parameters:
        raise ValueError("checkpoint parameter count is invalid or exceeds limit")
    expected_length = _HEADER.size + manifest_length + payload_length + _SHA256_BYTES
    if len(artifact) != expected_length:
        if len(artifact) > expected_length:
            raise ValueError("checkpoint artifact contains trailing data")
        raise ValueError("checkpoint artifact length is inconsistent or truncated")
    body = artifact[:-_SHA256_BYTES]
    if not hmac.compare_digest(hashlib.sha256(body).digest(), artifact[-32:]):
        raise ValueError("checkpoint SHA-256 mismatch")

    manifest_start = _HEADER.size
    payload_start = manifest_start + manifest_length
    manifest = _strict_manifest(artifact[manifest_start:payload_start])
    payload = artifact[payload_start:-_SHA256_BYTES]
    identity = _parse_identity(manifest["identity"])
    config = _parse_config(manifest["config"])
    expected_shapes = _expected_parameter_shapes(config)
    _validate_model_size(expected_shapes, exact_limits)
    tokenizer = _parse_tokenizer(
        manifest["tokenizer"], config=config, limits=exact_limits
    )
    bit_width, group_size = _parse_quantization(manifest["quantization"])
    _parse_architecture(manifest["architecture"])
    _parse_tied_parameters(manifest["tied_parameters"])

    descriptor_values = manifest["parameters"]
    if not isinstance(descriptor_values, list) or len(descriptor_values) != parameter_count:
        raise ValueError("checkpoint parameter descriptor count is inconsistent")
    if len(descriptor_values) != len(expected_shapes):
        raise ValueError("checkpoint parameter count does not match MiniGPT config")

    arrays: dict[str, NDArray[np.float32]] = {}
    expected_offset = 0
    previous_name: str | None = None
    for value in descriptor_values:
        descriptor = _parse_parameter_descriptor(value)
        name = cast(str, descriptor["name"])
        kind = cast(str, descriptor["kind"])
        shape = cast(tuple[int, ...], descriptor["shape"])
        offset = cast(int, descriptor["offset"])
        length = cast(int, descriptor["length"])
        if previous_name is not None and name <= previous_name:
            raise ValueError("parameter descriptors must be name-sorted and unique")
        if name not in expected_shapes or shape != expected_shapes[name]:
            raise ValueError(f"parameter {name} shape/name does not match config")
        expected_kind = _QUANTIZED_KIND if len(shape) == 2 else _FLOAT32_VECTOR_KIND
        if kind != expected_kind:
            raise ValueError(f"parameter {name} storage kind does not match rank")
        if offset != expected_offset:
            raise ValueError("checkpoint parameter offsets must be contiguous")
        if length > exact_limits.max_parameter_bytes:
            raise ValueError("parameter artifact exceeds configured limit")
        stop = offset + length
        if stop > len(payload):
            raise ValueError("checkpoint parameter range exceeds payload")
        parameter_bytes = payload[offset:stop]
        digest = "sha256:" + hashlib.sha256(parameter_bytes).hexdigest()
        if not hmac.compare_digest(digest, cast(str, descriptor["sha256"])):
            raise ValueError(f"parameter {name} SHA-256 mismatch")
        arrays[name] = _decode_parameter(
            parameter_bytes,
            name=name,
            kind=kind,
            shape=shape,
            bit_width=bit_width,
            group_size=group_size,
        )
        expected_offset = stop
        previous_name = name
    if expected_offset != len(payload):
        raise ValueError("checkpoint payload has unreferenced trailing bytes")
    if set(arrays) != set(expected_shapes):
        raise ValueError("checkpoint parameter set does not match MiniGPT config")

    model = MiniGPT(config).cpu().float()
    _validate_runtime_model(model, expected_shapes)
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            parameter.copy_(torch.from_numpy(arrays[name]))
    if model.lm_head.weight is not model.token_embedding.weight:
        raise RuntimeError("runtime MiniGPT failed to restore tied LM head")
    model.eval()
    return LoadedMiniGPTCheckpoint(
        identity=identity,
        config=config,
        tokenizer=tokenizer,
        model=model,
        bit_width=bit_width,
        group_size=group_size,
        artifact_sha256="sha256:" + hashlib.sha256(artifact).hexdigest(),
        serialized_artifact_bytes=len(artifact),
    )


def write_quantized_minigpt_checkpoint_new(
    path: Path,
    artifact: bytes,
    *,
    limits: MiniGPTCheckpointLimits | None = None,
) -> None:
    """Create a checkpoint file without overwriting an existing target."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    load_quantized_minigpt_checkpoint(artifact, limits=limits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(artifact)
        stream.flush()
        os.fsync(stream.fileno())


def read_quantized_minigpt_checkpoint(
    path: Path,
    *,
    limits: MiniGPTCheckpointLimits | None = None,
) -> LoadedMiniGPTCheckpoint:
    """Read with a pre-allocation file-size gate and close before model creation."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    exact_limits = _limits(limits)
    with path.open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        if file_size > exact_limits.max_artifact_bytes:
            raise ValueError("checkpoint file exceeds configured artifact limit")
        artifact = stream.read(exact_limits.max_artifact_bytes + 1)
    if len(artifact) != file_size:
        raise ValueError("checkpoint file changed or could not be read completely")
    return load_quantized_minigpt_checkpoint(artifact, limits=exact_limits)


def _limits(value: MiniGPTCheckpointLimits | None) -> MiniGPTCheckpointLimits:
    if value is None:
        return MiniGPTCheckpointLimits()
    if not isinstance(value, MiniGPTCheckpointLimits):
        raise TypeError("limits must be MiniGPTCheckpointLimits")
    return value


def _validate_identity_string(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise ValueError(f"{name} must be trimmed, NUL-free, and <=4096 UTF-8 bytes")
    return value


def _validate_quantization(bit_width: Any, group_size: Any) -> None:
    if (
        isinstance(bit_width, bool)
        or not isinstance(bit_width, int)
        or not 2 <= bit_width <= 8
    ):
        raise ValueError("bit_width must be an integer from 2 through 8")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")


def _expected_parameter_shapes(config: GPTConfig) -> dict[str, tuple[int, ...]]:
    dimension = config.model_dim
    hidden = config.mlp_ratio * dimension
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (config.vocab_size, dimension),
        "position_embedding.weight": (config.context_length, dimension),
        "final_norm.bias": (dimension,),
        "final_norm.weight": (dimension,),
    }
    for layer in range(config.num_layers):
        prefix = f"blocks.{layer}"
        shapes.update(
            {
                f"{prefix}.attention_norm.weight": (dimension,),
                f"{prefix}.attention_norm.bias": (dimension,),
                f"{prefix}.attention.qkv.weight": (3 * dimension, dimension),
                f"{prefix}.attention.output.weight": (dimension, dimension),
                f"{prefix}.mlp_norm.weight": (dimension,),
                f"{prefix}.mlp_norm.bias": (dimension,),
                f"{prefix}.mlp.layers.0.weight": (hidden, dimension),
                f"{prefix}.mlp.layers.2.weight": (dimension, hidden),
            }
        )
        if config.bias:
            shapes.update(
                {
                    f"{prefix}.attention.qkv.bias": (3 * dimension,),
                    f"{prefix}.attention.output.bias": (dimension,),
                    f"{prefix}.mlp.layers.0.bias": (hidden,),
                    f"{prefix}.mlp.layers.2.bias": (dimension,),
                }
            )
    return shapes


def _validate_runtime_model(
    model: MiniGPT, expected_shapes: dict[str, tuple[int, ...]]
) -> None:
    config = model.config
    if model.lm_head.weight is not model.token_embedding.weight:
        raise ValueError("MiniGPT token embedding and LM head must remain tied")
    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(expected_shapes) or any(
        tuple(named_parameters[name].shape) != shape
        for name, shape in expected_shapes.items()
    ):
        raise ValueError("MiniGPT parameter names/shapes do not match architecture revision")
    if len(model.blocks) != config.num_layers:
        raise ValueError("MiniGPT block count does not match config")
    if not math.isclose(float(model.dropout.p), float(config.dropout)):
        raise ValueError("MiniGPT embedding dropout does not match config")
    expected_mask = torch.tril(
        torch.ones(config.context_length, config.context_length, dtype=torch.bool)
    ).view(1, 1, config.context_length, config.context_length)
    for raw_block in model.blocks:
        block = cast(TransformerBlock, raw_block)
        attention = block.attention
        mlp = block.mlp
        if (
            attention.num_heads != config.num_heads
            or attention.head_dim != config.model_dim // config.num_heads
            or not math.isclose(
                float(attention.attention_dropout), float(config.dropout)
            )
            or not math.isclose(
                float(attention.residual_dropout.p), float(config.dropout)
            )
            or not torch.equal(attention.causal_mask.cpu(), expected_mask)
        ):
            raise ValueError("MiniGPT attention runtime does not match config")
        if (
            block.attention_norm.eps != 1e-5
            or block.mlp_norm.eps != 1e-5
            or getattr(mlp.layers[1], "approximate", None) != "tanh"
            or not math.isclose(
                float(cast(torch.nn.Dropout, mlp.layers[3]).p),
                float(config.dropout),
            )
        ):
            raise ValueError("MiniGPT norm/MLP runtime does not match architecture")
    if model.final_norm.eps != 1e-5:
        raise ValueError("MiniGPT final norm does not match architecture")


def _validate_model_size(
    shapes: dict[str, tuple[int, ...]], limits: MiniGPTCheckpointLimits
) -> None:
    if len(shapes) > limits.max_parameters:
        raise ValueError("parameter count exceeds configured checkpoint limit")
    parameter_count = sum(math.prod(shape) for shape in shapes.values())
    if parameter_count > limits.max_model_parameter_count:
        raise ValueError("model parameter count exceeds configured limit")


def _strict_manifest(value: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: _invalid_constant(constant),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("checkpoint manifest must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("checkpoint manifest fields are invalid")
    if payload["schema_version"] != MINIGPT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported MiniGPT checkpoint schema version")
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("checkpoint manifest contains unsupported JSON") from error
    if canonical != value:
        raise ValueError("checkpoint manifest must use canonical JSON encoding")
    return payload


def _parse_architecture(value: Any) -> None:
    record = _record(value, _ARCHITECTURE_FIELDS, "architecture")
    if (
        record["id"] != MINIGPT_ARCHITECTURE_ID
        or record["revision"] != MINIGPT_ARCHITECTURE_REVISION
    ):
        raise ValueError("unsupported MiniGPT architecture identity")


def _parse_identity(value: Any) -> MiniGPTCheckpointIdentity:
    record = _record(value, _IDENTITY_FIELDS, "identity")
    return MiniGPTCheckpointIdentity(
        model_id=_validate_identity_string(record["model_id"], "model_id"),
        model_revision=_validate_identity_string(
            record["model_revision"], "model_revision"
        ),
        tokenizer_revision=_validate_identity_string(
            record["tokenizer_revision"], "tokenizer_revision"
        ),
    )


def _parse_config(value: Any) -> GPTConfig:
    record = _record(value, _CONFIG_FIELDS, "config")
    dropout = record["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("config.dropout must be a finite number")
    dropout_float = float(dropout)
    if not math.isfinite(dropout_float):
        raise ValueError("config.dropout must be finite")
    return GPTConfig(
        vocab_size=_positive_integer(record["vocab_size"], "config.vocab_size"),
        context_length=_positive_integer(
            record["context_length"], "config.context_length"
        ),
        model_dim=_positive_integer(record["model_dim"], "config.model_dim"),
        num_heads=_positive_integer(record["num_heads"], "config.num_heads"),
        num_layers=_positive_integer(record["num_layers"], "config.num_layers"),
        mlp_ratio=_positive_integer(record["mlp_ratio"], "config.mlp_ratio"),
        dropout=dropout_float,
        bias=_boolean(record["bias"], "config.bias"),
    )


def _config_payload(config: GPTConfig) -> dict[str, Any]:
    if isinstance(config.dropout, bool) or not isinstance(config.dropout, (int, float)):
        raise ValueError("config.dropout must be a finite number")
    payload: dict[str, Any] = {
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "model_dim": config.model_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "mlp_ratio": config.mlp_ratio,
        "dropout": float(config.dropout),
        "bias": config.bias,
    }
    _parse_config(payload)
    return payload


def _parse_tokenizer(
    value: Any,
    *,
    config: GPTConfig,
    limits: MiniGPTCheckpointLimits,
) -> ByteBPETokenizer:
    record = _record(value, _TOKENIZER_FIELDS, "tokenizer")
    if (
        record["kind"] != MINIGPT_TOKENIZER_KIND
        or record["format_version"] != MINIGPT_TOKENIZER_FORMAT_VERSION
    ):
        raise ValueError("unsupported checkpoint tokenizer format")
    merges_value = record["merges"]
    if not isinstance(merges_value, list):
        raise ValueError("tokenizer.merges must be an array")
    if len(merges_value) > limits.max_tokenizer_merges:
        raise ValueError("tokenizer merge count exceeds configured limit")
    merges: list[tuple[int, int]] = []
    for index, pair in enumerate(merges_value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"tokenizer.merges[{index}] must contain two ids")
        merges.append(
            (
                _nonnegative_integer(pair[0], f"tokenizer.merges[{index}][0]"),
                _nonnegative_integer(pair[1], f"tokenizer.merges[{index}][1]"),
            )
        )
    tokenizer = ByteBPETokenizer(merges)
    vocab_size = _positive_integer(record["vocab_size"], "tokenizer.vocab_size")
    if tokenizer.vocab_size != vocab_size or vocab_size != config.vocab_size:
        raise ValueError("tokenizer payload vocabulary does not match config")
    return tokenizer


def _parse_quantization(value: Any) -> tuple[int, int]:
    record = _record(value, _QUANTIZATION_FIELDS, "quantization")
    if (
        record["method"] != "symmetric-absmax-contiguous-row-group"
        or record["matrix_format_version"] != QUANTIZED_MATRIX_FORMAT_VERSION
    ):
        raise ValueError("unsupported checkpoint quantization format")
    bit_width = record["bit_width"]
    group_size = record["group_size"]
    _validate_quantization(bit_width, group_size)
    return cast(int, bit_width), cast(int, group_size)


def _parse_tied_parameters(value: Any) -> None:
    if value != list(_TIED_PARAMETERS):
        raise ValueError("checkpoint tied-parameter contract is invalid")


def _parse_parameter_descriptor(value: Any) -> dict[str, Any]:
    record = _record(value, _PARAMETER_FIELDS, "parameter descriptor")
    name = _parameter_name(record["name"])
    kind = record["kind"]
    if kind not in {_QUANTIZED_KIND, _FLOAT32_VECTOR_KIND}:
        raise ValueError("unsupported checkpoint parameter storage kind")
    shape_value = record["shape"]
    if not isinstance(shape_value, list) or not 1 <= len(shape_value) <= 2:
        raise ValueError("parameter shape must contain one or two dimensions")
    shape = tuple(
        _positive_integer(dimension, "parameter shape dimension")
        for dimension in shape_value
    )
    length = _positive_integer(record["length"], "parameter length")
    return {
        "name": name,
        "kind": kind,
        "shape": shape,
        "offset": _nonnegative_integer(record["offset"], "parameter offset"),
        "length": length,
        "sha256": _sha256(record["sha256"], "parameter sha256"),
    }


def _decode_parameter(
    value: bytes,
    *,
    name: str,
    kind: str,
    shape: tuple[int, ...],
    bit_width: int,
    group_size: int,
) -> NDArray[np.float32]:
    if kind == _QUANTIZED_KIND:
        packed = PackedGroupwiseQuantizedMatrix.from_bytes(value)
        if packed.bit_width != bit_width or packed.group_size != group_size:
            raise ValueError(f"parameter {name} quantization config mismatch")
        if packed.original_shape != cast(tuple[int, int], shape):
            raise ValueError(f"parameter {name} packed shape mismatch")
        result = packed.unpack().dequantize()
    else:
        expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
        if len(value) != expected_bytes:
            raise ValueError(f"parameter {name} float32 byte length mismatch")
        result = np.frombuffer(value, dtype="<f4").copy().reshape(shape)
    result = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"parameter {name} contains non-finite values")
    return result


def _record(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
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


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return cast(int, value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _parameter_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
        or len(value.encode("utf-8")) > 1024
    ):
        raise ValueError("parameter name is invalid")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be canonical sha256:<lowercase-hex>")
    return value
