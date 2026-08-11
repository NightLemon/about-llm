"""Strict, architecture-scoped inspection of decoder checkpoint configs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

MODEL_CONFIG_INSPECTION_VERSION = "about-llm.model-config-inspection.v1"
MODEL_CONFIG_EVIDENCE_BOUNDARY = (
    "This inspection binds explicit JSON config fields and derives only an ideal dense "
    "K/V payload layout when standard attention fields are sufficient and no known MLA "
    "marker is present. It does not inspect weights, remote code, tokenizer or chat "
    "template, prove architecture semantics, estimate exact parameter count, model "
    "runtime metadata/workspace or quantization scales, establish effective context "
    "length or quality, authenticate provenance, or determine license compatibility."
)

_CORE_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "tie_word_embeddings",
    "rope_scaling",
)
_MLA_MARKERS = (
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)
_MOE_MARKERS = (
    "n_routed_experts",
    "n_shared_experts",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "num_experts_per_token",
    "moe_intermediate_size",
    "router_aux_loss_coef",
)


@dataclass(frozen=True)
class StandardKVLayout:
    """Fields needed by the ideal standard MHA/GQA/MQA cache formula."""

    applicable: bool
    reason: str
    attention_kind: str | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    query_heads_per_kv_head: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "applicable": self.applicable,
            "reason": self.reason,
            "attention_kind": self.attention_kind,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "query_heads_per_kv_head": self.query_heads_per_kv_head,
        }


@dataclass(frozen=True)
class DecoderConfigInspection:
    """Canonical config identity plus conservative architecture deductions."""

    config_fingerprint: str
    model_type: str | None
    architectures: tuple[str, ...]
    core_fields: Mapping[str, Any]
    mla_marker_fields: Mapping[str, Any]
    moe_marker_fields: Mapping[str, Any]
    standard_kv_layout: StandardKVLayout

    def to_dict(self) -> dict[str, object]:
        return {
            "inspection_version": MODEL_CONFIG_INSPECTION_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "core_fields": _thaw(self.core_fields),
            "mla_marker_fields": _thaw(self.mla_marker_fields),
            "moe_marker_fields": _thaw(self.moe_marker_fields),
            "known_mla_markers_present": bool(self.mla_marker_fields),
            "known_moe_markers_present": bool(self.moe_marker_fields),
            "parameter_count_estimated": False,
            "standard_kv_layout": self.standard_kv_layout.to_dict(),
            "evidence_boundary": MODEL_CONFIG_EVIDENCE_BOUNDARY,
        }


@dataclass(frozen=True)
class StandardKVEstimate:
    """Ideal K/V tensor payload only; excludes every runtime overhead."""

    token_count: int
    batch_size: int
    element_bytes: int
    bytes_per_token_per_layer: int
    total_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "batch_size": self.batch_size,
            "element_bytes": self.element_bytes,
            "bytes_per_token_per_layer": self.bytes_per_token_per_layer,
            "total_bytes": self.total_bytes,
            "ideal_tensor_payload_only": True,
            "includes_allocator_metadata_alignment_workspace_or_scales": False,
        }


def inspect_decoder_config(config: Mapping[str, Any]) -> DecoderConfigInspection:
    """Inspect explicit JSON config fields without guessing model-family semantics."""

    snapshot_value = json.loads(canonical_json_bytes(config))
    if not isinstance(snapshot_value, dict):
        raise ValueError("config must be a JSON object")
    snapshot = cast(dict[str, Any], snapshot_value)
    model_type_value = snapshot.get("model_type")
    if model_type_value is not None and (
        not isinstance(model_type_value, str) or not model_type_value.strip()
    ):
        raise ValueError("model_type must be a non-empty string or null")
    architectures = _architectures(snapshot.get("architectures"))
    _validate_common_fields(snapshot)
    core_fields = _selected_fields(snapshot, _CORE_FIELDS)
    mla_fields = _selected_fields(snapshot, _MLA_MARKERS)
    moe_fields = _selected_fields(snapshot, _MOE_MARKERS)
    layout = _standard_kv_layout(snapshot, mla_fields)
    return DecoderConfigInspection(
        config_fingerprint="sha256:" + artifact_fingerprint(snapshot),
        model_type=model_type_value,
        architectures=architectures,
        core_fields=_freeze(core_fields),
        mla_marker_fields=_freeze(mla_fields),
        moe_marker_fields=_freeze(moe_fields),
        standard_kv_layout=layout,
    )


def estimate_standard_kv_cache(
    inspection: DecoderConfigInspection,
    *,
    token_count: int,
    batch_size: int = 1,
    element_bytes: int = 2,
) -> StandardKVEstimate:
    """Calculate ideal standard dense K/V payload bytes for one request batch."""

    token_count = _positive_integer(token_count, "token_count")
    batch_size = _positive_integer(batch_size, "batch_size")
    element_bytes = _positive_integer(element_bytes, "element_bytes")
    layout = inspection.standard_kv_layout
    if not layout.applicable:
        raise ValueError(f"standard KV estimate is not applicable: {layout.reason}")
    if (
        layout.num_hidden_layers is None
        or layout.num_key_value_heads is None
        or layout.head_dim is None
    ):
        raise RuntimeError("applicable standard KV layout is internally incomplete")
    per_token_per_layer = (
        2 * layout.num_key_value_heads * layout.head_dim * element_bytes
    )
    total = (
        per_token_per_layer
        * layout.num_hidden_layers
        * token_count
        * batch_size
    )
    return StandardKVEstimate(
        token_count=token_count,
        batch_size=batch_size,
        element_bytes=element_bytes,
        bytes_per_token_per_layer=per_token_per_layer,
        total_bytes=total,
    )


def load_model_config_json(path: Path) -> Mapping[str, Any]:
    """Load a config JSON while rejecting duplicate keys and non-finite constants."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise ValueError(f"{path}: config is not valid UTF-8: {error}") from error
    try:
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: config must be a JSON object")
    canonical_json_bytes(value)
    return MappingProxyType(cast(dict[str, Any], value))


def _standard_kv_layout(
    config: Mapping[str, Any], mla_fields: Mapping[str, Any]
) -> StandardKVLayout:
    if mla_fields:
        return StandardKVLayout(
            applicable=False,
            reason=(
                "known MLA marker fields are present; the standard dense K/V formula "
                "must not be applied"
            ),
        )
    layers = _optional_positive_integer(config, "num_hidden_layers")
    query_heads = _optional_positive_integer(config, "num_attention_heads")
    kv_heads = _optional_positive_integer(config, "num_key_value_heads")
    explicit_head_dim = _optional_positive_integer(config, "head_dim")
    hidden_size = _optional_positive_integer(config, "hidden_size")
    missing = [
        name
        for name, value in (
            ("num_hidden_layers", layers),
            ("num_attention_heads", query_heads),
            ("num_key_value_heads", kv_heads),
        )
        if value is None
    ]
    if missing:
        return StandardKVLayout(
            applicable=False,
            reason="missing explicit standard attention fields: " + ", ".join(missing),
        )
    if explicit_head_dim is None:
        if hidden_size is None:
            return StandardKVLayout(
                applicable=False,
                reason="head_dim is absent and hidden_size is unavailable for derivation",
            )
        if hidden_size % cast(int, query_heads) != 0:
            return StandardKVLayout(
                applicable=False,
                reason="hidden_size is not divisible by num_attention_heads",
            )
        head_dim = hidden_size // cast(int, query_heads)
    else:
        head_dim = explicit_head_dim
    query_heads_int = cast(int, query_heads)
    kv_heads_int = cast(int, kv_heads)
    if kv_heads_int > query_heads_int or query_heads_int % kv_heads_int != 0:
        return StandardKVLayout(
            applicable=False,
            reason=(
                "num_attention_heads must be divisible by num_key_value_heads "
                "for this standard MHA/GQA/MQA contract"
            ),
        )
    group_size = query_heads_int // kv_heads_int
    if kv_heads_int == query_heads_int:
        attention_kind = "mha"
    elif kv_heads_int == 1:
        attention_kind = "mqa"
    else:
        attention_kind = "gqa"
    return StandardKVLayout(
        applicable=True,
        reason="explicit fields satisfy the ideal standard MHA/GQA/MQA layout contract",
        attention_kind=attention_kind,
        num_hidden_layers=cast(int, layers),
        num_attention_heads=query_heads_int,
        num_key_value_heads=kv_heads_int,
        head_dim=head_dim,
        query_heads_per_kv_head=group_size,
    )


def _architectures(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("architectures must be an array of non-empty strings or null")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError("architectures must contain non-empty strings")
    return cast(tuple[str, ...], result)


def _selected_fields(
    config: Mapping[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    return {
        name: config[name]
        for name in names
        if name in config and config[name] is not None
    }


def _optional_positive_integer(config: Mapping[str, Any], name: str) -> int | None:
    if name not in config or config[name] is None:
        return None
    return _positive_integer(config[name], name)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _validate_common_fields(config: Mapping[str, Any]) -> None:
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        *_MLA_MARKERS,
    ):
        _optional_positive_integer(config, name)
    tie_word_embeddings = config.get("tie_word_embeddings")
    if tie_word_embeddings is not None and not isinstance(tie_word_embeddings, bool):
        raise ValueError("tie_word_embeddings must be a boolean or null")
    rope_scaling = config.get("rope_scaling")
    if rope_scaling is not None and not isinstance(rope_scaling, Mapping):
        raise ValueError("rope_scaling must be an object or null")


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_value(item) for key, item in sorted(value.items())}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _freeze(cast(dict[str, Any], value))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
