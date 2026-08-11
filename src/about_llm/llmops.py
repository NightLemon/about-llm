"""Deterministic, JSON-only fingerprints for LLM experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias

JsonScalar: TypeAlias = bool | int | float | str | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _json_snapshot(value: object, path: str) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        snapshot: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            snapshot[key] = _json_snapshot(item, f"{path}.{key}")
        return snapshot
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_snapshot(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-JSON value of type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-only value with deterministic key and whitespace rules."""

    snapshot = _json_snapshot(value, "artifact")
    return json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def artifact_fingerprint(components: Mapping[str, object]) -> str:
    """Return a SHA-256 identity for explicitly supplied JSON components.

    The digest proves equality under this canonical serialization only. It does
    not prove semantic equivalence, completeness, safety, provenance, or that a
    remote model/provider can reproduce the same output.
    """

    if not components:
        raise ValueError("components must not be empty")
    return hashlib.sha256(canonical_json_bytes(components)).hexdigest()
