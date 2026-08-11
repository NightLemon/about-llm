from __future__ import annotations

import math

import pytest

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes


def test_fingerprint_is_independent_of_mapping_insertion_order() -> None:
    first = {
        "model": {"revision": "abc", "id": "model-a"},
        "generation": {"temperature": 0.0, "stop": ["END"]},
    }
    second = {
        "generation": {"stop": ["END"], "temperature": 0.0},
        "model": {"id": "model-a", "revision": "abc"},
    }
    assert artifact_fingerprint(first) == artifact_fingerprint(second)


def test_fingerprint_changes_when_an_explicit_component_changes() -> None:
    baseline = {"model": "model-a@abc", "prompt": "prompt-v1", "tools": None}
    candidate = {"model": "model-a@abc", "prompt": "prompt-v2", "tools": None}
    assert artifact_fingerprint(baseline) != artifact_fingerprint(candidate)


def test_sequence_order_remains_semantically_significant_to_identity() -> None:
    assert artifact_fingerprint({"messages": ["system", "user"]}) != artifact_fingerprint(
        {"messages": ["user", "system"]}
    )


def test_unicode_is_encoded_as_utf8_without_ascii_escape_noise() -> None:
    encoded = canonical_json_bytes({"prompt": "中文"})
    assert "中文".encode() in encoded
    assert b"\\u" not in encoded


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        artifact_fingerprint({"temperature": bad})


@pytest.mark.parametrize(
    "bad",
    [
        {"payload": b"bytes"},
        {"payload": {1, 2}},
        {"payload": object()},
        {"payload": {1: "non-string key"}},
    ],
)
def test_non_json_values_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        artifact_fingerprint(bad)


def test_empty_component_map_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        artifact_fingerprint({})
