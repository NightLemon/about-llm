from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from about_llm.finetuning import (
    CandidateDisposition,
    CandidateException,
    GovernanceOutcome,
    audit_preference_governance,
    load_preference_records,
    load_sft_governance_policy,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
AS_OF = datetime(2026, 8, 6, 12, tzinfo=timezone.utc)
pytestmark = [pytest.mark.contract, pytest.mark.security]


def _policy():
    return load_sft_governance_policy(PROJECT / "governance-policy.example.json")


def test_preference_fixture_governance_covers_prompt_and_both_candidates() -> None:
    records = load_preference_records(PROJECT / "preference.example.jsonl")

    report = audit_preference_governance(
        records, policy=_policy(), evaluated_at=AS_OF
    )
    payload = report.to_dict()

    assert report.gate_passed
    assert len(report.source_decisions) == 4
    assert all(item.outcome is GovernanceOutcome.ALLOWED for item in report.source_decisions)
    assert report.sensitive_candidates == ()
    assert payload["scope"]["prompt_and_both_candidates_scanned"] is True
    assert payload["scope"]["legal_permission_verified"] is False


def test_preference_governance_fails_closed_and_omits_candidate_plaintext() -> None:
    record = load_preference_records(PROJECT / "preference.example.jsonl")[0]
    sensitive = replace(
        record,
        candidate_a="Contact alice@example.test",
        license_id="unknown-license",
    )

    report = audit_preference_governance(
        (sensitive,), policy=_policy(), evaluated_at=AS_OF
    )
    rendered = json.dumps(report.to_dict(), ensure_ascii=False)

    assert not report.gate_passed
    assert report.source_decisions[0].outcome is GovernanceOutcome.UNKNOWN_SOURCE_LICENSE
    assert report.sensitive_candidates[0].role == "candidate_a"
    assert report.sensitive_candidates[0].disposition is CandidateDisposition.UNREVIEWED
    assert "alice@example.test" not in rendered
    assert '"matched_plaintext": null' in rendered


def test_preference_candidate_exception_is_bound_to_record_surface_and_span() -> None:
    record = replace(
        load_preference_records(PROJECT / "preference.example.jsonl")[0],
        candidate_b="Contact alice@example.test",
    )
    initial = audit_preference_governance(
        (record,), policy=_policy(), evaluated_at=AS_OF
    )
    candidate_id = initial.sensitive_candidates[0].candidate_fingerprint
    exception = CandidateException(
        candidate_id,
        "privacy-reviewer",
        "Reserved example.test fixture",
        "review://preference-1",
    )
    accepted_policy = replace(_policy(), candidate_exceptions=(exception,))

    accepted = audit_preference_governance(
        (record,), policy=accepted_policy, evaluated_at=AS_OF
    )
    stale = audit_preference_governance(
        (replace(record, candidate_b="safe text"),),
        policy=accepted_policy,
        evaluated_at=AS_OF,
    )

    assert accepted.gate_passed
    assert (
        accepted.sensitive_candidates[0].disposition
        is CandidateDisposition.ACCEPTED_EXCEPTION
    )
    assert not stale.gate_passed
    assert stale.unused_exception_fingerprints == (candidate_id,)
