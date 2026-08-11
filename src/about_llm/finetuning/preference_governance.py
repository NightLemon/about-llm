"""Preference source policy and limited sensitive-candidate auditing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from about_llm.finetuning.data import DataSplit
from about_llm.finetuning.governance import (
    CandidateDisposition,
    GovernanceOutcome,
    GovernancePurpose,
    SensitiveCandidateFinding,
    SensitiveTextSurface,
    SFTGovernancePolicy,
    SourceDecision,
    SourceDecisionFinding,
    SourceRule,
    scan_sensitive_text_surfaces,
)
from about_llm.finetuning.preference_data import PreferenceRecord
from about_llm.llmops import artifact_fingerprint

PREFERENCE_GOVERNANCE_AUDIT_VERSION = (
    "about-llm.preference-governance-audit.v1"
)
PREFERENCE_SENSITIVE_CANDIDATE_DETECTOR_VERSION = (
    "about-llm.preference-sensitive-regex-checksum.v1"
)


@dataclass(frozen=True)
class PreferenceGovernanceAuditReport:
    evaluated_at: datetime
    policy_fingerprint: str
    ordered_dataset_fingerprint: str
    source_decisions: tuple[SourceDecisionFinding, ...]
    sensitive_candidates: tuple[SensitiveCandidateFinding, ...]
    unused_exception_fingerprints: tuple[str, ...]

    @property
    def blocking_finding_count(self) -> int:
        return (
            sum(not finding.allowed for finding in self.source_decisions)
            + sum(
                finding.disposition is CandidateDisposition.UNREVIEWED
                for finding in self.sensitive_candidates
            )
            + len(self.unused_exception_fingerprints)
        )

    @property
    def gate_passed(self) -> bool:
        return self.blocking_finding_count == 0

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self._identity())

    def _identity(self) -> dict[str, object]:
        return {
            "audit_version": PREFERENCE_GOVERNANCE_AUDIT_VERSION,
            "detector_version": PREFERENCE_SENSITIVE_CANDIDATE_DETECTOR_VERSION,
            "evaluated_at": _format_timestamp(self.evaluated_at),
            "policy_fingerprint": self.policy_fingerprint,
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "source_decisions": [item.to_dict() for item in self.source_decisions],
            "sensitive_candidates": [
                item.to_dict() for item in self.sensitive_candidates
            ],
            "unused_exception_fingerprints": list(
                self.unused_exception_fingerprints
            ),
            "blocking_finding_count": self.blocking_finding_count,
            "gate_passed": self.gate_passed,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "scope": {
                "prompt_and_both_candidates_scanned": True,
                "exact_source_license_match": True,
                "unknown_source_or_license_fails_closed": True,
                "limited_regex_checksum_candidate_scan": True,
                "matched_plaintext_in_report": False,
                "legal_permission_verified": False,
                "consent_verified": False,
                "comprehensive_pii_or_secret_detection": False,
                "human_exception_is_absence_proof": False,
                "detector_precision_or_recall_calibrated": False,
            },
            "evidence_boundary": (
                "The shared source registry records an internal decision under one "
                "declared purpose and time; it is not a legal opinion. The scanner "
                "checks prompt messages and both raw candidates only for a fixed "
                "subset of email, key/token, private-key header, JWT, and Luhn-valid "
                "card-like patterns. No candidate is not proof that personal data or "
                "secrets are absent. Findings omit matched plaintext, but record and "
                "surface-span metadata can still be sensitive."
            ),
        }


def audit_preference_governance(
    records: Iterable[PreferenceRecord],
    *,
    policy: SFTGovernancePolicy,
    evaluated_at: datetime,
) -> PreferenceGovernanceAuditReport:
    snapshot = tuple(records)
    if not snapshot or any(not isinstance(item, PreferenceRecord) for item in snapshot):
        raise ValueError("preference governance audit requires PreferenceRecord values")
    ids = tuple(record.record_id for record in snapshot)
    if len(ids) != len(set(ids)):
        raise ValueError("preference governance audit requires unique record ids")
    if not isinstance(policy, SFTGovernancePolicy):
        raise ValueError("policy must be SFTGovernancePolicy")
    _require_utc_datetime(evaluated_at, "evaluated_at")
    if policy.reviewed_at > evaluated_at:
        raise ValueError("policy reviewed_at cannot be later than evaluated_at")
    rules = {(rule.source, rule.license_id): rule for rule in policy.source_rules}
    decisions = tuple(
        _decide_source(record, rules, policy, evaluated_at) for record in snapshot
    )
    exception_ids = {
        item.candidate_fingerprint for item in policy.candidate_exceptions
    }
    surfaces = tuple(
        surface for record in snapshot for surface in _record_surfaces(record)
    )
    candidates = scan_sensitive_text_surfaces(
        surfaces,
        exception_ids=exception_ids,
        detector_version=PREFERENCE_SENSITIVE_CANDIDATE_DETECTOR_VERSION,
    )
    used_exceptions = {
        item.candidate_fingerprint
        for item in candidates
        if item.disposition is CandidateDisposition.ACCEPTED_EXCEPTION
    }
    ordered_dataset_fingerprint = "sha256:" + artifact_fingerprint(
        {
            "ordered_record_fingerprints": [
                record.record_fingerprint for record in snapshot
            ]
        }
    )
    return PreferenceGovernanceAuditReport(
        evaluated_at=evaluated_at,
        policy_fingerprint=policy.policy_fingerprint,
        ordered_dataset_fingerprint=ordered_dataset_fingerprint,
        source_decisions=decisions,
        sensitive_candidates=candidates,
        unused_exception_fingerprints=tuple(sorted(exception_ids - used_exceptions)),
    )


def _record_surfaces(record: PreferenceRecord) -> tuple[SensitiveTextSurface, ...]:
    prompt = tuple(
        SensitiveTextSurface(
            record.record_id,
            record.record_fingerprint,
            index,
            message.role.value,
            message.content,
        )
        for index, message in enumerate(record.prompt, 1)
    )
    offset = len(prompt)
    return (
        *prompt,
        SensitiveTextSurface(
            record.record_id,
            record.record_fingerprint,
            offset + 1,
            "candidate_a",
            record.candidate_a,
        ),
        SensitiveTextSurface(
            record.record_id,
            record.record_fingerprint,
            offset + 2,
            "candidate_b",
            record.candidate_b,
        ),
    )


def _decide_source(
    record: PreferenceRecord,
    rules: dict[tuple[str, str], SourceRule],
    policy: SFTGovernancePolicy,
    evaluated_at: datetime,
) -> SourceDecisionFinding:
    purpose = (
        GovernancePurpose.TRAINING
        if record.split is DataSplit.TRAIN
        else GovernancePurpose.EVALUATION
    )
    rule = rules.get((record.source, record.license_id))
    evidence_ref: str | None = None
    if rule is None:
        outcome = GovernanceOutcome.UNKNOWN_SOURCE_LICENSE
    else:
        evidence_ref = rule.evidence_ref
        if rule.decision is SourceDecision.DENY:
            outcome = GovernanceOutcome.DENIED
        elif rule.expires_at is not None and evaluated_at >= rule.expires_at:
            outcome = GovernanceOutcome.EXPIRED
        elif purpose not in rule.allowed_purposes:
            outcome = GovernanceOutcome.PURPOSE_NOT_ALLOWED
        elif record.risk not in policy.allowed_risk_labels:
            outcome = GovernanceOutcome.RISK_NOT_ALLOWED
        else:
            outcome = GovernanceOutcome.ALLOWED
    return SourceDecisionFinding(
        record.record_id,
        record.source,
        record.license_id,
        record.split.value,
        purpose,
        record.risk,
        outcome,
        evidence_ref,
    )


def _require_utc_datetime(value: object, prefix: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo != timezone.utc:
        raise ValueError(f"{prefix} must be a timezone-aware UTC datetime")
    if value.microsecond:
        raise ValueError(f"{prefix} must have whole-second precision")


def _format_timestamp(value: datetime) -> str:
    _require_utc_datetime(value, "timestamp")
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")
