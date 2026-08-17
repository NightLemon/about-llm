from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.generation_contract import (
    GENERATION_PROTOCOL_EVIDENCE_BOUNDARY,
    GenerationProtocolInspection,
    inspect_generation_protocol,
    inspect_generation_protocol_document,
    load_generation_protocol_json,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "transformers-basics"
PROTOCOLS = PROJECT / "protocols"
SCRIPT = PROJECT / "inspect_generation_protocol.py"


def _inspection(name: str) -> GenerationProtocolInspection:
    return inspect_generation_protocol_document(
        load_generation_protocol_json(PROTOCOLS / name)
    )


def _token(report: dict[str, object], field: str) -> dict[str, object]:
    values = report["special_tokens"]
    assert isinstance(values, list)
    matches = [item for item in values if isinstance(item, dict) and item["field"] == field]
    assert len(matches) == 1
    return matches[0]


def test_aligned_fixture_reports_eos_superset_without_calling_it_an_error() -> None:
    report = _inspection("aligned-superset-eos.example.json").to_dict()
    eos = _token(report, "eos_token_id")

    assert report["contract_id"] == "authored-aligned-superset-eos@v1"
    assert report["contract_fingerprint"] == (
        "sha256:fc3d4f4477d59687fd5b311badb212efb4fb5bf808d2ecc3ae0b28959aa6f807"
    )
    assert report["generation_config_present"] is True
    assert report["effective_runtime_contract_proved"] is False
    assert eos["tokenizer_ids"] == [2]
    assert eos["model_config_ids"] == [2]
    assert eos["generation_config_ids"] == [2, 3]
    assert eos["tokenizer_vs_model"] == "exact_set_match"
    assert eos["tokenizer_vs_generation"] == "left_strict_subset"
    assert eos["model_vs_generation"] == "left_strict_subset"
    assert report["observations"] == [
        "generation_config_contains_both_max_length_and_max_new_tokens; "
        "runtime precedence is not inferred"
    ]
    assert report["evidence_boundary"] == GENERATION_PROTOCOL_EVIDENCE_BOUNDARY


def test_drift_fixture_surfaces_disjoint_and_out_of_range_ids() -> None:
    report = _inspection("drift-out-of-range.example.json").to_dict()
    bos = _token(report, "bos_token_id")
    eos = _token(report, "eos_token_id")
    pad = _token(report, "pad_token_id")

    assert report["contract_fingerprint"] == (
        "sha256:9a33ae14d2035794f17a0d0ead561baab647585b409c5e4f4f4e17e3f5422e52"
    )
    assert bos["tokenizer_vs_generation"] == "disjoint"
    assert eos["model_vs_generation"] == "disjoint"
    assert pad["generation_config_ids"] == [9]
    assert pad["ids_outside_tokenizer_size"] == [9]
    assert pad["ids_outside_model_vocab"] == [9]
    assert report["observations"] == [
        "bos_token_id:tokenizer_vs_generation:disjoint",
        "bos_token_id:model_vs_generation:disjoint",
        "eos_token_id:tokenizer_vs_generation:disjoint",
        "eos_token_id:model_vs_generation:disjoint",
        "pad_token_id:tokenizer_vs_generation:disjoint",
        "pad_token_id:model_vs_generation:disjoint",
        "pad_token_id:ids_outside_tokenizer_size=9",
        "pad_token_id:ids_outside_model_vocab=9",
    ]


def test_missing_generation_and_pad_eos_overlap_are_neutral_observations() -> None:
    report = inspect_generation_protocol(
        contract_id="authored-base-no-generation-config@v1",
        tokenizer_size=8,
        model_vocab_size=8,
        tokenizer={"eos_token_id": 2, "pad_token_id": 2},
        model_config={"eos_token_id": 2, "pad_token_id": 2},
        generation_config=None,
    ).to_dict()

    assert report["generation_config_present"] is False
    assert report["observations"] == [
        "generation_config_snapshot_unavailable",
        "tokenizer:pad_and_eos_sets_overlap; this may be intentional",
        "model_config:pad_and_eos_sets_overlap; this may be intentional",
    ]


@pytest.mark.parametrize(
    ("tokenizer", "message"),
    [
        ({"eos_token_id": True}, "tokenizer.eos_token_id"),
        ({"eos_token_id": []}, "cannot be empty"),
        ({"eos_token_id": [2, 2]}, "must not contain duplicate"),
        ({"eos_token_id": [-1]}, "non-negative integers"),
        ({"eos_token_id": "2"}, "integer, integer array, or null"),
    ],
)
def test_token_id_fields_reject_ambiguous_values(
    tokenizer: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        inspect_generation_protocol(
            contract_id="authored-invalid@v1",
            tokenizer_size=8,
            model_vocab_size=8,
            tokenizer=tokenizer,
            model_config={},
            generation_config=None,
        )


def test_inspection_snapshots_nested_generation_fields() -> None:
    generation = {"stop_strings": ["<END>"], "eos_token_id": [2, 3]}
    inspection = inspect_generation_protocol(
        contract_id="authored-snapshot@v1",
        tokenizer_size=8,
        model_vocab_size=8,
        tokenizer={"eos_token_id": 2},
        model_config={"eos_token_id": 2},
        generation_config=generation,
    )
    generation["stop_strings"][0] = "MUTATED"  # type: ignore[index]
    report = inspection.to_dict()

    assert report["generation_fields"] == {
        "stop_strings": ["<END>"],
    }
    assert _token(report, "eos_token_id")["generation_config_ids"] == [2, 3]


@pytest.mark.parametrize(
    "bad_document",
    [
        '{"contract_id":"x","contract_id":"y"}',
        '{"contract_id":NaN}',
        "[]",
        json.dumps(
            {
                "contract_id": "x",
                "tokenizer_size": 8,
                "model_vocab_size": 8,
                "tokenizer": {},
                "model_config": {},
                "generation_config": None,
                "unknown": True,
            }
        ),
    ],
)
def test_strict_document_loader_rejects_ambiguous_schema(
    tmp_path: Path, bad_document: str
) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(bad_document, encoding="utf-8")

    with pytest.raises(ValueError):
        load_generation_protocol_json(path)


def test_strict_document_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_bytes(b'{"contract_id":"\xff"}')

    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_generation_protocol_json(path)


