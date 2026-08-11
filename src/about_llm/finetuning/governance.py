"""Fail-closed SFT source policy and limited sensitive-content candidates.

This module turns source/license labels into a reproducible policy decision and
provides a deliberately narrow regex/checksum candidate scanner.  Neither is a
legal opinion or proof that personal data and secrets are absent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.finetuning.data import DataSplit, SFTRecord
from about_llm.llmops import artifact_fingerprint

SFT_GOVERNANCE_POLICY_VERSION = "about-llm.sft-governance-policy.v1"
SFT_GOVERNANCE_AUDIT_VERSION = "about-llm.sft-governance-audit.v1"
SFT_SENSITIVE_CANDIDATE_DETECTOR_VERSION = (
    "about-llm.sft-sensitive-regex-checksum.v1"
)
_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class GovernancePurpose(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class SourceDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class GovernanceOutcome(str, Enum):
    ALLOWED = "allowed"
    UNKNOWN_SOURCE_LICENSE = "unknown_source_license"
    DENIED = "denied"
    PURPOSE_NOT_ALLOWED = "purpose_not_allowed"
    EXPIRED = "expired"
    RISK_NOT_ALLOWED = "risk_not_allowed"


class CandidateDisposition(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED_EXCEPTION = "accepted_exception"


@dataclass(frozen=True)
class SourceRule:
    source: str
    license_id: str
    decision: SourceDecision
    allowed_purposes: tuple[GovernancePurpose, ...]
    evidence_ref: str
    expires_at: datetime | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.source, "source rule source")
        _require_nonempty_string(self.license_id, "source rule license")
        _require_nonempty_string(self.evidence_ref, "source rule evidence_ref")
        if not isinstance(self.decision, SourceDecision):
            raise ValueError("source rule decision must be SourceDecision")
        purposes = tuple(self.allowed_purposes)
        if any(not isinstance(item, GovernancePurpose) for item in purposes):
            raise ValueError("source rule purposes must be GovernancePurpose values")
        if len(purposes) != len(set(purposes)):
            raise ValueError("source rule purposes must be unique")
        if self.decision is SourceDecision.ALLOW and not purposes:
            raise ValueError("allowed source rule needs at least one purpose")
        if self.decision is SourceDecision.DENY and purposes:
            raise ValueError("denied source rule cannot declare allowed purposes")
        object.__setattr__(self, "allowed_purposes", purposes)
        if self.expires_at is not None:
            _require_utc_datetime(self.expires_at, "source rule expires_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "license": self.license_id,
            "decision": self.decision.value,
            "allowed_purposes": [item.value for item in self.allowed_purposes],
            "evidence_ref": self.evidence_ref,
            "expires_at": (
                _format_timestamp(self.expires_at)
                if self.expires_at is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CandidateException:
    candidate_fingerprint: str
    reviewer: str
    rationale: str
    evidence_ref: str

    def __post_init__(self) -> None:
        _require_fingerprint(
            self.candidate_fingerprint, "candidate exception fingerprint"
        )
        _require_nonempty_string(self.reviewer, "candidate exception reviewer")
        _require_nonempty_string(self.rationale, "candidate exception rationale")
        _require_nonempty_string(self.evidence_ref, "candidate exception evidence_ref")

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_fingerprint": self.candidate_fingerprint,
            "reviewer": self.reviewer,
            "rationale": self.rationale,
            "evidence_ref": self.evidence_ref,
        }


@dataclass(frozen=True)
class SFTGovernancePolicy:
    policy_id: str
    owner: str
    reviewed_at: datetime
    source_rules: tuple[SourceRule, ...]
    allowed_risk_labels: tuple[str, ...]
    candidate_exceptions: tuple[CandidateException, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.policy_id, "policy_id")
        _require_nonempty_string(self.owner, "policy owner")
        _require_utc_datetime(self.reviewed_at, "policy reviewed_at")
        rules = tuple(self.source_rules)
        if not rules or any(not isinstance(rule, SourceRule) for rule in rules):
            raise ValueError("policy needs SourceRule values")
        identities = tuple((rule.source, rule.license_id) for rule in rules)
        if len(identities) != len(set(identities)):
            raise ValueError("source/license rules must be unique")
        object.__setattr__(self, "source_rules", rules)
        risks = tuple(self.allowed_risk_labels)
        if not risks:
            raise ValueError("allowed_risk_labels must not be empty")
        for risk in risks:
            _require_nonempty_string(risk, "allowed risk label")
        if len(risks) != len(set(risks)):
            raise ValueError("allowed_risk_labels must be unique")
        object.__setattr__(self, "allowed_risk_labels", risks)
        exceptions = tuple(self.candidate_exceptions)
        if any(not isinstance(item, CandidateException) for item in exceptions):
            raise ValueError("candidate_exceptions must contain CandidateException values")
        exception_ids = tuple(item.candidate_fingerprint for item in exceptions)
        if len(exception_ids) != len(set(exception_ids)):
            raise ValueError("candidate exception fingerprints must be unique")
        object.__setattr__(self, "candidate_exceptions", exceptions)

    @property
    def policy_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self._identity())

    def _identity(self) -> dict[str, object]:
        return {
            "policy_version": SFT_GOVERNANCE_POLICY_VERSION,
            "policy_id": self.policy_id,
            "owner": self.owner,
            "reviewed_at": _format_timestamp(self.reviewed_at),
            "source_rules": [rule.to_dict() for rule in self.source_rules],
            "allowed_risk_labels": list(self.allowed_risk_labels),
            "candidate_exceptions": [
                item.to_dict() for item in self.candidate_exceptions
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity(), "policy_fingerprint": self.policy_fingerprint}


@dataclass(frozen=True)
class SourceDecisionFinding:
    record_id: str
    source: str
    license_id: str
    split: str
    purpose: GovernancePurpose
    risk: str
    outcome: GovernanceOutcome
    rule_evidence_ref: str | None

    @property
    def allowed(self) -> bool:
        return self.outcome is GovernanceOutcome.ALLOWED

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "license": self.license_id,
            "split": self.split,
            "purpose": self.purpose.value,
            "risk": self.risk,
            "outcome": self.outcome.value,
            "rule_evidence_ref": self.rule_evidence_ref,
        }


@dataclass(frozen=True)
class SensitiveCandidateFinding:
    record_id: str
    message_index: int
    role: str
    detector_id: str
    start_codepoint: int
    end_codepoint: int
    candidate_fingerprint: str
    disposition: CandidateDisposition

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "message_index": self.message_index,
            "role": self.role,
            "detector_id": self.detector_id,
            "start_codepoint": self.start_codepoint,
            "end_codepoint": self.end_codepoint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "disposition": self.disposition.value,
            "matched_plaintext": None,
        }


@dataclass(frozen=True)
class SensitiveTextSurface:
    """One fingerprint-bound text surface passed to the shared narrow scanner."""

    record_id: str
    record_fingerprint: str
    surface_index: int
    role: str
    content: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.record_id, "surface record_id")
        _require_fingerprint(self.record_fingerprint, "surface record_fingerprint")
        if (
            isinstance(self.surface_index, bool)
            or not isinstance(self.surface_index, int)
            or self.surface_index <= 0
        ):
            raise ValueError("surface_index must be a positive integer")
        _require_nonempty_string(self.role, "surface role")
        _require_nonempty_string(self.content, "surface content")


@dataclass(frozen=True)
class SFTGovernanceAuditReport:
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
            "audit_version": SFT_GOVERNANCE_AUDIT_VERSION,
            "detector_version": SFT_SENSITIVE_CANDIDATE_DETECTOR_VERSION,
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
                "The source registry records an internal allow/deny decision under one "
                "declared purpose and time; it is not a legal opinion. The scanner only "
                "generates candidates for a fixed subset of email, key/token, private-key "
                "header, JWT, and Luhn-valid card-like patterns. No candidate is not proof "
                "that personal data or secrets are absent. Findings omit matched plaintext, "
                "but record identity and span metadata can still be sensitive."
            ),
        }


@dataclass(frozen=True)
class _Detector:
    detector_id: str
    pattern: re.Pattern[str]
    requires_luhn: bool = False


_DETECTORS = (
    _Detector(
        "email_address",
        re.compile(
            r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])"
        ),
    ),
    _Detector(
        "private_key_header",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    _Detector(
        "aws_access_key_id",
        re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
    ),
    _Detector(
        "github_token",
        re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36,255}(?![A-Za-z0-9])"),
    ),
    _Detector(
        "openai_style_secret_key",
        re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    ),
    _Detector(
        "jwt_compact",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
        ),
    ),
    _Detector(
        "luhn_card_like_number",
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
        requires_luhn=True,
    ),
)


def load_sft_governance_policy(path: Path) -> SFTGovernancePolicy:
    """Load a strict policy with duplicate-key and unknown-field rejection."""

    value = _load_strict_json(path)
    policy = _object(value, str(path))
    _expect_fields(
        policy,
        required={
            "policy_version",
            "policy_id",
            "owner",
            "reviewed_at",
            "source_rules",
            "allowed_risk_labels",
            "candidate_exceptions",
        },
        allowed={
            "policy_version",
            "policy_id",
            "owner",
            "reviewed_at",
            "source_rules",
            "allowed_risk_labels",
            "candidate_exceptions",
        },
        prefix=str(path),
    )
    if policy["policy_version"] != SFT_GOVERNANCE_POLICY_VERSION:
        raise ValueError(f"{path}: unsupported policy_version")
    rules_raw = _array(policy["source_rules"], f"{path}.source_rules")
    exceptions_raw = _array(
        policy["candidate_exceptions"], f"{path}.candidate_exceptions"
    )
    risks_raw = _array(policy["allowed_risk_labels"], f"{path}.allowed_risk_labels")
    return SFTGovernancePolicy(
        policy_id=_string(policy["policy_id"], f"{path}.policy_id"),
        owner=_string(policy["owner"], f"{path}.owner"),
        reviewed_at=parse_utc_timestamp(
            _string(policy["reviewed_at"], f"{path}.reviewed_at")
        ),
        source_rules=tuple(
            _parse_source_rule(item, f"{path}.source_rules[{index}]")
            for index, item in enumerate(rules_raw)
        ),
        allowed_risk_labels=tuple(
            _string(item, f"{path}.allowed_risk_labels[{index}]")
            for index, item in enumerate(risks_raw)
        ),
        candidate_exceptions=tuple(
            _parse_exception(item, f"{path}.candidate_exceptions[{index}]")
            for index, item in enumerate(exceptions_raw)
        ),
    )


def audit_sft_governance(
    records: Iterable[SFTRecord],
    *,
    policy: SFTGovernancePolicy,
    evaluated_at: datetime,
) -> SFTGovernanceAuditReport:
    """Apply exact registry decisions and generate limited sensitive candidates."""

    snapshot = tuple(records)
    if not snapshot or any(not isinstance(record, SFTRecord) for record in snapshot):
        raise ValueError("governance audit requires SFTRecord values")
    ids = tuple(record.record_id for record in snapshot)
    if len(ids) != len(set(ids)):
        raise ValueError("governance audit requires unique record ids")
    if not isinstance(policy, SFTGovernancePolicy):
        raise ValueError("policy must be SFTGovernancePolicy")
    _require_utc_datetime(evaluated_at, "evaluated_at")
    if policy.reviewed_at > evaluated_at:
        raise ValueError("policy reviewed_at cannot be later than evaluated_at")
    rules = {(rule.source, rule.license_id): rule for rule in policy.source_rules}
    source_decisions = tuple(
        _decide_source(record, rules, policy, evaluated_at) for record in snapshot
    )
    exception_ids = {
        item.candidate_fingerprint for item in policy.candidate_exceptions
    }
    candidates = tuple(
        _sensitive_candidates(snapshot, exception_ids=exception_ids)
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
    return SFTGovernanceAuditReport(
        evaluated_at=evaluated_at,
        policy_fingerprint=policy.policy_fingerprint,
        ordered_dataset_fingerprint=ordered_dataset_fingerprint,
        source_decisions=source_decisions,
        sensitive_candidates=candidates,
        unused_exception_fingerprints=tuple(sorted(exception_ids - used_exceptions)),
    )


def parse_utc_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError("timestamp must use exact UTC form YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"invalid UTC timestamp {value!r}") from error
    return parsed.replace(tzinfo=timezone.utc)


def _decide_source(
    record: SFTRecord,
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


def _sensitive_candidates(
    records: tuple[SFTRecord, ...], *, exception_ids: set[str]
) -> Iterable[SensitiveCandidateFinding]:
    surfaces = tuple(
        SensitiveTextSurface(
            record_id=record.record_id,
            record_fingerprint=record.record_fingerprint,
            surface_index=message_index,
            role=message.role.value,
            content=message.content,
        )
        for record in records
        for message_index, message in enumerate(record.messages, 1)
    )
    return scan_sensitive_text_surfaces(
        surfaces,
        exception_ids=exception_ids,
        detector_version=SFT_SENSITIVE_CANDIDATE_DETECTOR_VERSION,
    )


def scan_sensitive_text_surfaces(
    surfaces: Iterable[SensitiveTextSurface],
    *,
    exception_ids: set[str],
    detector_version: str,
) -> tuple[SensitiveCandidateFinding, ...]:
    """Run the shared limited scanner without retaining matched plaintext."""

    snapshot = tuple(surfaces)
    if any(not isinstance(item, SensitiveTextSurface) for item in snapshot):
        raise ValueError("sensitive scan requires SensitiveTextSurface values")
    if not isinstance(exception_ids, set) or any(
        not isinstance(item, str) for item in exception_ids
    ):
        raise ValueError("exception_ids must be a set of strings")
    _require_nonempty_string(detector_version, "detector_version")
    findings: list[SensitiveCandidateFinding] = []
    for surface in snapshot:
        for detector in _DETECTORS:
            for match in detector.pattern.finditer(surface.content):
                if detector.requires_luhn and not _passes_luhn(match.group(0)):
                    continue
                candidate_fingerprint = "sha256:" + artifact_fingerprint(
                    {
                        "detector_version": detector_version,
                        "detector_id": detector.detector_id,
                        "record_fingerprint": surface.record_fingerprint,
                        "message_index": surface.surface_index,
                        "start_codepoint": match.start(),
                        "end_codepoint": match.end(),
                    }
                )
                disposition = (
                    CandidateDisposition.ACCEPTED_EXCEPTION
                    if candidate_fingerprint in exception_ids
                    else CandidateDisposition.UNREVIEWED
                )
                findings.append(
                    SensitiveCandidateFinding(
                        surface.record_id,
                        surface.surface_index,
                        surface.role,
                        detector.detector_id,
                        match.start(),
                        match.end(),
                        candidate_fingerprint,
                        disposition,
                    )
                )
    findings.sort(
        key=lambda item: (
            item.record_id,
            item.message_index,
            item.start_codepoint,
            item.detector_id,
        )
    )
    return tuple(findings)


def _passes_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isascii() and character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _parse_source_rule(value: Any, prefix: str) -> SourceRule:
    item = _object(value, prefix)
    fields = {
        "source",
        "license",
        "decision",
        "allowed_purposes",
        "evidence_ref",
        "expires_at",
    }
    _expect_fields(item, required=fields, allowed=fields, prefix=prefix)
    purposes_raw = _array(item["allowed_purposes"], f"{prefix}.allowed_purposes")
    try:
        decision = SourceDecision(item["decision"])
        purposes = tuple(GovernancePurpose(purpose) for purpose in purposes_raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix}: invalid decision or purpose") from error
    expires_raw = item["expires_at"]
    if expires_raw is not None and not isinstance(expires_raw, str):
        raise ValueError(f"{prefix}.expires_at must be null or a UTC timestamp")
    return SourceRule(
        source=_string(item["source"], f"{prefix}.source"),
        license_id=_string(item["license"], f"{prefix}.license"),
        decision=decision,
        allowed_purposes=purposes,
        evidence_ref=_string(item["evidence_ref"], f"{prefix}.evidence_ref"),
        expires_at=parse_utc_timestamp(expires_raw) if expires_raw is not None else None,
    )


def _parse_exception(value: Any, prefix: str) -> CandidateException:
    item = _object(value, prefix)
    fields = {"candidate_fingerprint", "reviewer", "rationale", "evidence_ref"}
    _expect_fields(item, required=fields, allowed=fields, prefix=prefix)
    return CandidateException(
        candidate_fingerprint=_string(
            item["candidate_fingerprint"], f"{prefix}.candidate_fingerprint"
        ),
        reviewer=_string(item["reviewer"], f"{prefix}.reviewer"),
        rationale=_string(item["rationale"], f"{prefix}.rationale"),
        evidence_ref=_string(item["evidence_ref"], f"{prefix}.evidence_ref"),
    )


def _load_strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error


def _expect_fields(
    value: dict[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    prefix: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise ValueError(
            f"{prefix}: field mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _object(value: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: expected an object")
    return cast(dict[str, Any], value)


def _array(value: Any, prefix: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{prefix}: expected an array")
    return value


def _string(value: Any, prefix: str) -> str:
    _require_nonempty_string(value, prefix)
    return cast(str, value)


def _require_nonempty_string(value: Any, prefix: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{prefix} contains an unpaired Unicode surrogate") from error


def _require_fingerprint(value: object, prefix: str) -> None:
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{prefix} must be a lowercase sha256 fingerprint")


def _require_utc_datetime(value: object, prefix: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo != timezone.utc:
        raise ValueError(f"{prefix} must be a timezone-aware UTC datetime")
    if value.microsecond:
        raise ValueError(f"{prefix} must have whole-second precision")


def _format_timestamp(value: datetime) -> str:
    _require_utc_datetime(value, "timestamp")
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
