"""Conservative comparison of explicit local generation-protocol snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

GENERATION_PROTOCOL_INSPECTION_VERSION = "about-llm.generation-protocol.v1"
GENERATION_PROTOCOL_EVIDENCE_BOUNDARY = (
    "This inspection compares explicitly supplied normalized tokenizer, model-config, "
    "and generation-config snapshots. It does not resolve generate() kwargs, model "
    "class fallbacks, runtime defaults or precedence, tokenizer file bytes, remote "
    "code, stop-string tokenization, weights, provenance, quality, or behavior in "
    "Transformers, vLLM, or a provider API. A reported mismatch is a review signal, "
    "not proof that either side is wrong; missing or matching fields are not proof of "
    "an effective runtime contract."
)

_SPECIAL_TOKEN_FIELDS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "decoder_start_token_id",
)
_GENERATION_FIELDS = (
    "do_sample",
    "num_beams",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "max_length",
    "max_new_tokens",
    "min_length",
    "min_new_tokens",
    "stop_strings",
    "use_cache",
)
_TOP_LEVEL_FIELDS = {
    "contract_id",
    "tokenizer_size",
    "model_vocab_size",
    "tokenizer",
    "model_config",
    "generation_config",
}


@dataclass(frozen=True)
class SpecialTokenComparison:
    """One special-token field across three explicit sources."""

    field: str
    tokenizer_ids: tuple[int, ...] | None
    model_config_ids: tuple[int, ...] | None
    generation_config_ids: tuple[int, ...] | None
    tokenizer_vs_model: str
    tokenizer_vs_generation: str
    model_vs_generation: str
    ids_outside_tokenizer_size: tuple[int, ...]
    ids_outside_model_vocab: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "tokenizer_ids": _ids_to_json(self.tokenizer_ids),
            "model_config_ids": _ids_to_json(self.model_config_ids),
            "generation_config_ids": _ids_to_json(self.generation_config_ids),
            "tokenizer_vs_model": self.tokenizer_vs_model,
            "tokenizer_vs_generation": self.tokenizer_vs_generation,
            "model_vs_generation": self.model_vs_generation,
            "ids_outside_tokenizer_size": list(self.ids_outside_tokenizer_size),
            "ids_outside_model_vocab": list(self.ids_outside_model_vocab),
        }


@dataclass(frozen=True)
class GenerationProtocolInspection:
    """Identity, three-way token comparison, and non-prescriptive observations."""

    contract_id: str
    contract_fingerprint: str
    tokenizer_size: int
    model_vocab_size: int | None
    generation_config_present: bool
    generation_fields: Mapping[str, Any]
    special_tokens: tuple[SpecialTokenComparison, ...]
    observations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "inspection_version": GENERATION_PROTOCOL_INSPECTION_VERSION,
            "contract_id": self.contract_id,
            "contract_fingerprint": self.contract_fingerprint,
            "tokenizer_size": self.tokenizer_size,
            "model_vocab_size": self.model_vocab_size,
            "generation_config_present": self.generation_config_present,
            "generation_fields": _thaw(self.generation_fields),
            "special_tokens": [item.to_dict() for item in self.special_tokens],
            "observations": list(self.observations),
            "effective_runtime_contract_proved": False,
            "evidence_boundary": GENERATION_PROTOCOL_EVIDENCE_BOUNDARY,
        }


def inspect_generation_protocol(
    *,
    contract_id: str,
    tokenizer_size: int,
    model_vocab_size: int | None,
    tokenizer: Mapping[str, Any],
    model_config: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None,
) -> GenerationProtocolInspection:
    """Compare explicit snapshots without choosing or repairing runtime defaults."""

    if not isinstance(contract_id, str) or not contract_id.strip():
        raise ValueError("contract_id must be a non-empty string")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("tokenizer must be an object")
    if not isinstance(model_config, Mapping):
        raise ValueError("model_config must be an object")
    if generation_config is not None and not isinstance(generation_config, Mapping):
        raise ValueError("generation_config must be an object or null")
    tokenizer_size = _positive_integer(tokenizer_size, "tokenizer_size")
    if model_vocab_size is not None:
        model_vocab_size = _positive_integer(model_vocab_size, "model_vocab_size")
    snapshot_value = json.loads(
        canonical_json_bytes(
            {
                "contract_id": contract_id,
                "tokenizer_size": tokenizer_size,
                "model_vocab_size": model_vocab_size,
                "tokenizer": tokenizer,
                "model_config": model_config,
                "generation_config": generation_config,
            }
        )
    )
    if not isinstance(snapshot_value, dict):
        raise RuntimeError("canonical generation contract snapshot is not an object")
    snapshot = cast(dict[str, Any], snapshot_value)
    tokenizer_snapshot = cast(dict[str, Any], snapshot["tokenizer"])
    model_snapshot = cast(dict[str, Any], snapshot["model_config"])
    generation_snapshot = cast(dict[str, Any] | None, snapshot["generation_config"])

    comparisons = tuple(
        _compare_special_token(
            field,
            tokenizer_snapshot,
            model_snapshot,
            generation_snapshot,
            tokenizer_size=tokenizer_size,
            model_vocab_size=model_vocab_size,
        )
        for field in _SPECIAL_TOKEN_FIELDS
    )
    generation_fields = (
        {}
        if generation_snapshot is None
        else {
            field: generation_snapshot[field]
            for field in _GENERATION_FIELDS
            if field in generation_snapshot and generation_snapshot[field] is not None
        }
    )
    observations = _observations(comparisons, generation_snapshot)
    return GenerationProtocolInspection(
        contract_id=contract_id,
        contract_fingerprint="sha256:" + artifact_fingerprint(snapshot),
        tokenizer_size=tokenizer_size,
        model_vocab_size=model_vocab_size,
        generation_config_present=generation_snapshot is not None,
        generation_fields=_freeze(generation_fields),
        special_tokens=comparisons,
        observations=observations,
    )


def load_generation_protocol_json(path: Path) -> Mapping[str, Any]:
    """Load the strict combined-input schema used by the offline CLI."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise ValueError(f"{path}: protocol is not valid UTF-8: {error}") from error
    try:
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: protocol must be a JSON object")
    unknown = set(value) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(value)
    if unknown or missing:
        raise ValueError(
            f"{path}: protocol fields mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    for field in ("tokenizer", "model_config"):
        if not isinstance(value[field], dict):
            raise ValueError(f"{path}: {field} must be an object")
    if value["generation_config"] is not None and not isinstance(
        value["generation_config"], dict
    ):
        raise ValueError(f"{path}: generation_config must be an object or null")
    canonical_json_bytes(value)
    return MappingProxyType(cast(dict[str, Any], value))


def inspect_generation_protocol_document(
    document: Mapping[str, Any],
) -> GenerationProtocolInspection:
    """Inspect one strict combined document after schema loading."""

    return inspect_generation_protocol(
        contract_id=document["contract_id"],
        tokenizer_size=document["tokenizer_size"],
        model_vocab_size=document["model_vocab_size"],
        tokenizer=document["tokenizer"],
        model_config=document["model_config"],
        generation_config=document["generation_config"],
    )


def _compare_special_token(
    field: str,
    tokenizer: Mapping[str, Any],
    model_config: Mapping[str, Any],
    generation_config: Mapping[str, Any] | None,
    *,
    tokenizer_size: int,
    model_vocab_size: int | None,
) -> SpecialTokenComparison:
    tokenizer_ids = _optional_token_ids(tokenizer.get(field), f"tokenizer.{field}")
    model_ids = _optional_token_ids(
        model_config.get(field), f"model_config.{field}"
    )
    generation_ids = (
        None
        if generation_config is None
        else _optional_token_ids(
            generation_config.get(field), f"generation_config.{field}"
        )
    )
    all_ids = sorted(
        {
            token_id
            for ids in (tokenizer_ids, model_ids, generation_ids)
            if ids is not None
            for token_id in ids
        }
    )
    return SpecialTokenComparison(
        field=field,
        tokenizer_ids=tokenizer_ids,
        model_config_ids=model_ids,
        generation_config_ids=generation_ids,
        tokenizer_vs_model=_set_relation(tokenizer_ids, model_ids),
        tokenizer_vs_generation=_set_relation(tokenizer_ids, generation_ids),
        model_vs_generation=_set_relation(model_ids, generation_ids),
        ids_outside_tokenizer_size=tuple(
            token_id for token_id in all_ids if token_id >= tokenizer_size
        ),
        ids_outside_model_vocab=tuple(
            token_id
            for token_id in all_ids
            if model_vocab_size is not None and token_id >= model_vocab_size
        ),
    )


def _optional_token_ids(value: Any, location: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{location} must be an integer, integer array, or null")
    if isinstance(value, int):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        values = tuple(value)
        if not values:
            raise ValueError(f"{location} integer array cannot be empty")
    else:
        raise ValueError(f"{location} must be an integer, integer array, or null")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
        raise ValueError(f"{location} must contain non-negative integers")
    integer_values = cast(tuple[int, ...], values)
    if len(set(integer_values)) != len(integer_values):
        raise ValueError(f"{location} must not contain duplicate token ids")
    return tuple(sorted(integer_values))


def _set_relation(
    left: tuple[int, ...] | None, right: tuple[int, ...] | None
) -> str:
    if left is None or right is None:
        return "unavailable"
    left_set = set(left)
    right_set = set(right)
    if left_set == right_set:
        return "exact_set_match"
    if left_set < right_set:
        return "left_strict_subset"
    if left_set > right_set:
        return "left_strict_superset"
    if left_set & right_set:
        return "overlap_not_equal"
    return "disjoint"


def _observations(
    comparisons: tuple[SpecialTokenComparison, ...],
    generation_config: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    observations: list[str] = []
    if generation_config is None:
        observations.append("generation_config_snapshot_unavailable")
    elif (
        generation_config.get("max_length") is not None
        and generation_config.get("max_new_tokens") is not None
    ):
        observations.append(
            "generation_config_contains_both_max_length_and_max_new_tokens; "
            "runtime precedence is not inferred"
        )
    for comparison in comparisons:
        for relation_name, relation in (
            ("tokenizer_vs_model", comparison.tokenizer_vs_model),
            ("tokenizer_vs_generation", comparison.tokenizer_vs_generation),
            ("model_vs_generation", comparison.model_vs_generation),
        ):
            if relation == "disjoint":
                observations.append(f"{comparison.field}:{relation_name}:disjoint")
        if comparison.ids_outside_tokenizer_size:
            observations.append(
                f"{comparison.field}:ids_outside_tokenizer_size="
                + ",".join(map(str, comparison.ids_outside_tokenizer_size))
            )
        if comparison.ids_outside_model_vocab:
            observations.append(
                f"{comparison.field}:ids_outside_model_vocab="
                + ",".join(map(str, comparison.ids_outside_model_vocab))
            )
    for source_name, source_ids in (
        ("tokenizer", {item.field: item.tokenizer_ids for item in comparisons}),
        ("model_config", {item.field: item.model_config_ids for item in comparisons}),
        (
            "generation_config",
            {item.field: item.generation_config_ids for item in comparisons},
        ),
    ):
        pad_ids = source_ids["pad_token_id"]
        eos_ids = source_ids["eos_token_id"]
        if pad_ids is not None and eos_ids is not None and set(pad_ids) & set(eos_ids):
            observations.append(
                f"{source_name}:pad_and_eos_sets_overlap; this may be intentional"
            )
    return tuple(observations)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _ids_to_json(value: tuple[int, ...] | None) -> list[int] | None:
    return None if value is None else list(value)


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
