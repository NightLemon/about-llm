from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from about_llm.finetuning import (
    NearDuplicateProfile,
    SFTTrainingReadinessReport,
    audit_sft_governance,
    audit_sft_near_duplicates,
    load_sft_governance_policy,
    load_sft_records,
    load_sft_training_readiness,
    validate_sft_training_readiness,
    validate_training_subset,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"


def _report() -> SFTTrainingReadinessReport:
    training = load_sft_records(PROJECT / "train.example.jsonl")
    combined = load_sft_records(PROJECT / "audit.example.jsonl")
    binding = validate_training_subset(training, combined)
    near = audit_sft_near_duplicates(
        combined,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=0.9,
    )
    governance = audit_sft_governance(
        combined,
        policy=load_sft_governance_policy(
            PROJECT / "governance-policy.example.json"
        ),
        evaluated_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
    )
    return SFTTrainingReadinessReport.from_reports(binding, near, governance)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_readiness_strict_round_trip_and_train_binding(tmp_path: Path) -> None:
    expected = _report()
    path = tmp_path / "readiness.json"
    _write(path, expected.to_dict())

    loaded = load_sft_training_readiness(path)
    training = load_sft_records(PROJECT / "train.example.jsonl")
    audit = validate_sft_training_readiness(training, loaded)

    assert loaded == expected
    assert audit.manifest_fingerprint == loaded.training_manifest_fingerprint
    assert loaded.to_dict()["scope"]["held_out_plaintext_embedded"] is False
    assert "provide neither origin authentication" in loaded.to_dict()[
        "evidence_boundary"
    ]


def test_readiness_rejects_tampered_decision_field(tmp_path: Path) -> None:
    payload = _report().to_dict()
    payload["near_duplicate_candidate_count"] = 1
    payload["gate_passed"] = False
    path = tmp_path / "tampered.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="manifest_fingerprint mismatch"):
        load_sft_training_readiness(path)


def test_readiness_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    payload = _report().to_dict()
    payload["surprise"] = True
    unknown = tmp_path / "unknown.json"
    _write(unknown, payload)
    with pytest.raises(ValueError, match=r"unknown=.*surprise"):
        load_sft_training_readiness(unknown)

    rendered = json.dumps(_report().to_dict(), ensure_ascii=False)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        rendered[:-1] + ',"gate_passed":true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_sft_training_readiness(duplicate)


def test_readiness_rejects_unsupported_artifact_version(tmp_path: Path) -> None:
    payload = _report().to_dict()
    payload["artifact_version"] = "about-llm.sft-training-readiness.v999"
    path = tmp_path / "future.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="unsupported readiness artifact_version"):
        load_sft_training_readiness(path)


def test_readiness_rejects_stale_or_reordered_training_data() -> None:
    readiness = _report()
    training = load_sft_records(PROJECT / "train.example.jsonl")

    with pytest.raises(ValueError, match="ordered fingerprint differs"):
        validate_sft_training_readiness(tuple(reversed(training)), readiness)


def test_readiness_hash_is_explicitly_not_an_authentication_boundary() -> None:
    payload = _report().to_dict()

    assert payload["scope"]["cryptographic_origin_authenticated"] is False
    assert "able to replace the artifact can recompute" in payload["evidence_boundary"]
