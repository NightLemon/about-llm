from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from about_llm.finetuning import (
    NearDuplicateProfile,
    PreferenceTrainingReadinessReport,
    audit_preference_governance,
    audit_preference_near_duplicates,
    load_preference_records,
    load_preference_training_readiness,
    load_sft_governance_policy,
    validate_dpo_training_subset,
    validate_preference_training_readiness,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
pytestmark = [pytest.mark.contract, pytest.mark.security]


def _report() -> PreferenceTrainingReadinessReport:
    training = load_preference_records(PROJECT / "preference.train.example.jsonl")
    combined = load_preference_records(PROJECT / "preference.example.jsonl")
    binding = validate_dpo_training_subset(training, combined)
    near = audit_preference_near_duplicates(
        combined,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=0.9,
    )
    governance = audit_preference_governance(
        combined,
        policy=load_sft_governance_policy(
            PROJECT / "governance-policy.example.json"
        ),
        evaluated_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
    )
    return PreferenceTrainingReadinessReport.from_reports(binding, near, governance)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_preference_readiness_strict_round_trip_and_train_binding(
    tmp_path: Path,
) -> None:
    expected = _report()
    path = tmp_path / "readiness.json"
    _write(path, expected.to_dict())

    loaded = load_preference_training_readiness(path)
    training = load_preference_records(PROJECT / "preference.train.example.jsonl")
    audit = validate_preference_training_readiness(training, loaded)

    assert loaded == expected
    assert audit.record_count == 2
    assert loaded.to_dict()["scope"]["held_out_plaintext_embedded"] is False
    assert "do not authenticate" in loaded.to_dict()["evidence_boundary"]


def test_preference_readiness_rejects_tamper_unknown_and_duplicate_fields(
    tmp_path: Path,
) -> None:
    payload = _report().to_dict()
    payload["binary_train_record_count"] = 3
    tampered = tmp_path / "tampered.json"
    _write(tampered, payload)
    with pytest.raises(ValueError, match="manifest_fingerprint mismatch"):
        load_preference_training_readiness(tampered)

    payload = _report().to_dict()
    payload["surprise"] = True
    unknown = tmp_path / "unknown.json"
    _write(unknown, payload)
    with pytest.raises(ValueError, match=r"unknown=.*surprise"):
        load_preference_training_readiness(unknown)

    rendered = json.dumps(_report().to_dict(), ensure_ascii=False)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        rendered[:-1] + ',"gate_passed":true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_preference_training_readiness(duplicate)


def test_preference_readiness_rejects_reordered_training_data() -> None:
    training = load_preference_records(PROJECT / "preference.train.example.jsonl")

    with pytest.raises(ValueError, match="ordered fingerprint differs"):
        validate_preference_training_readiness(tuple(reversed(training)), _report())
