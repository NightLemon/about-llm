from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from about_llm.finetuning.data import (
    ChatMessage,
    DataSplit,
    FunctionToolCall,
    FunctionToolDefinition,
    MessageRole,
    SFTRecord,
    load_sft_records,
)
from about_llm.finetuning.governance import (
    CandidateDisposition,
    CandidateException,
    GovernanceOutcome,
    GovernancePurpose,
    SFTGovernancePolicy,
    SourceDecision,
    SourceRule,
    audit_sft_governance,
    load_sft_governance_policy,
    parse_utc_timestamp,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
AS_OF = datetime(2026, 8, 6, 12, tzinfo=timezone.utc)
pytestmark = [pytest.mark.contract, pytest.mark.security]


def _policy() -> SFTGovernancePolicy:
    return load_sft_governance_policy(PROJECT / "governance-policy.example.json")


def _record(content: str, *, record_id: str = "one") -> SFTRecord:
    return SFTRecord(
        record_id,
        (
            ChatMessage(MessageRole.USER, "Inspect this value"),
            ChatMessage(MessageRole.ASSISTANT, content),
        ),
        "about-llm-authored-fixture",
        "CC-BY-4.0",
        "governance-test",
        "en",
        "normal",
        f"group-{record_id}",
        DataSplit.TRAIN,
    )


def test_fixture_policy_passes_with_explicit_scope_and_identity() -> None:
    records = load_sft_records(PROJECT / "audit.example.jsonl")

    report = audit_sft_governance(records, policy=_policy(), evaluated_at=AS_OF)
    payload = report.to_dict()

    assert report.gate_passed
    assert report.blocking_finding_count == 0
    assert len(report.source_decisions) == 4
    assert all(item.outcome is GovernanceOutcome.ALLOWED for item in report.source_decisions)
    assert report.sensitive_candidates == ()
    assert report.manifest_fingerprint.startswith("sha256:")
    assert payload["scope"]["legal_permission_verified"] is False
    assert payload["scope"]["comprehensive_pii_or_secret_detection"] is False
    assert "not proof" in payload["evidence_boundary"]


def test_registry_fails_closed_for_unknown_license_and_risk() -> None:
    base = _record("safe content")
    wrong_license = replace(base, license_id="unknown-license")
    wrong_risk = replace(base, record_id="two", group_id="group-two", risk="high")

    report = audit_sft_governance(
        (wrong_license, wrong_risk), policy=_policy(), evaluated_at=AS_OF
    )

    assert not report.gate_passed
    assert [item.outcome for item in report.source_decisions] == [
        GovernanceOutcome.UNKNOWN_SOURCE_LICENSE,
        GovernanceOutcome.RISK_NOT_ALLOWED,
    ]


@pytest.mark.parametrize(
    ("decision", "purposes", "expires_at", "split", "outcome"),
    [
        (SourceDecision.DENY, (), None, DataSplit.TRAIN, GovernanceOutcome.DENIED),
        (
            SourceDecision.ALLOW,
            (GovernancePurpose.EVALUATION,),
            None,
            DataSplit.TRAIN,
            GovernanceOutcome.PURPOSE_NOT_ALLOWED,
        ),
        (
            SourceDecision.ALLOW,
            (GovernancePurpose.TRAINING,),
            datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
            DataSplit.TRAIN,
            GovernanceOutcome.EXPIRED,
        ),
    ],
)
def test_registry_decision_time_and_purpose_are_enforced(
    decision: SourceDecision,
    purposes: tuple[GovernancePurpose, ...],
    expires_at: datetime | None,
    split: DataSplit,
    outcome: GovernanceOutcome,
) -> None:
    rule = SourceRule(
        "about-llm-authored-fixture",
        "CC-BY-4.0",
        decision,
        purposes,
        "test-evidence",
        expires_at,
    )
    policy = SFTGovernancePolicy(
        "test-policy",
        "test-owner",
        datetime(2026, 8, 5, tzinfo=timezone.utc),
        (rule,),
        ("normal",),
        (),
    )
    record = replace(_record("safe content"), split=split)

    report = audit_sft_governance((record,), policy=policy, evaluated_at=AS_OF)

    assert report.source_decisions[0].outcome is outcome
    assert not report.gate_passed


def test_sensitive_candidates_omit_plaintext_and_fail_closed() -> None:
    sensitive = (
        "contact alice@example.test; key AKIAABCDEFGHIJKLMNOP; "
        "card 4111 1111 1111 1111"
    )

    report = audit_sft_governance(
        (_record(sensitive),), policy=_policy(), evaluated_at=AS_OF
    )
    rendered = json.dumps(report.to_dict(), ensure_ascii=False)

    assert not report.gate_passed
    assert {item.detector_id for item in report.sensitive_candidates} == {
        "email_address",
        "aws_access_key_id",
        "luhn_card_like_number",
    }
    assert all(
        item.disposition is CandidateDisposition.UNREVIEWED
        for item in report.sensitive_candidates
    )
    assert "alice@example.test" not in rendered
    assert "AKIAABCDEFGHIJKLMNOP" not in rendered
    assert "4111 1111 1111 1111" not in rendered
    assert '"matched_plaintext": null' in rendered


def test_sensitive_scan_covers_tool_calls_and_tool_schemas() -> None:
    record = SFTRecord(
        "tool-sensitive",
        (
            ChatMessage(MessageRole.USER, "send notification"),
            ChatMessage(
                MessageRole.ASSISTANT,
                "",
                tool_calls=(
                    FunctionToolCall(
                        "call-1", "notify", {"email": "alice@example.test"}
                    ),
                ),
            ),
            ChatMessage(
                MessageRole.TOOL,
                "queued",
                tool_call_id="call-1",
                name="notify",
            ),
            ChatMessage(MessageRole.ASSISTANT, "done"),
        ),
        "about-llm-authored-fixture",
        "CC-BY-4.0",
        "governance-test",
        "en",
        "normal",
        "group-tool-sensitive",
        DataSplit.TRAIN,
        tools=(
            FunctionToolDefinition(
                "notify",
                "Fixture key AKIAABCDEFGHIJKLMNOP.",
                {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                },
            ),
        ),
    )

    report = audit_sft_governance((record,), policy=_policy(), evaluated_at=AS_OF)

    assert not report.gate_passed
    assert {(item.role, item.detector_id) for item in report.sensitive_candidates} == {
        ("assistant", "email_address"),
        ("tool_schema", "aws_access_key_id"),
    }
    rendered = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "alice@example.test" not in rendered
    assert "AKIAABCDEFGHIJKLMNOP" not in rendered


def test_exact_candidate_exception_passes_but_stale_exception_fails() -> None:
    record = _record("contact alice@example.test")
    initial = audit_sft_governance((record,), policy=_policy(), evaluated_at=AS_OF)
    candidate_id = initial.sensitive_candidates[0].candidate_fingerprint
    exception = CandidateException(
        candidate_id,
        "privacy-reviewer",
        "Reserved example.test address in authored fixture",
        "review://ticket-1",
    )
    accepted_policy = replace(_policy(), candidate_exceptions=(exception,))

    accepted = audit_sft_governance(
        (record,), policy=accepted_policy, evaluated_at=AS_OF
    )
    stale = audit_sft_governance(
        (_record("no candidate here"),),
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


def test_invalid_luhn_like_number_is_not_a_candidate() -> None:
    report = audit_sft_governance(
        (_record("number 4111 1111 1111 1112"),),
        policy=_policy(),
        evaluated_at=AS_OF,
    )

    assert report.gate_passed
    assert report.sensitive_candidates == ()


def test_policy_loader_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    payload = json.loads(
        (PROJECT / "governance-policy.example.json").read_text(encoding="utf-8")
    )
    payload["surprise"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown=.*surprise"):
        load_sft_governance_policy(unknown)

    rendered = (PROJECT / "governance-policy.example.json").read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        rendered.replace(
            '"policy_id": "about-llm-authored-fixture-v1",',
            '"policy_id": "about-llm-authored-fixture-v1",\n'
            '  "policy_id": "duplicate",',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_sft_governance_policy(duplicate)


def test_timestamp_is_exact_utc_and_future_review_is_rejected() -> None:
    assert parse_utc_timestamp("2026-08-06T12:00:00Z") == AS_OF
    with pytest.raises(ValueError, match="exact UTC form"):
        parse_utc_timestamp("2026-08-06T12:00:00+00:00")
    with pytest.raises(ValueError, match="later than evaluated_at"):
        audit_sft_governance(
            (_record("safe"),),
            policy=replace(
                _policy(),
                reviewed_at=datetime(2026, 8, 7, tzinfo=timezone.utc),
            ),
            evaluated_at=AS_OF,
        )
