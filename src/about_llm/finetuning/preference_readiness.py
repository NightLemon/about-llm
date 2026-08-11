"""Strict held-out-free readiness artifacts for pairwise preference training."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.finetuning.near_duplicate import NearDuplicateProfile
from about_llm.finetuning.preference_data import (
    PREFERENCE_DATA_CONTRACT_VERSION,
    PreferenceDataAuditReport,
    PreferenceRecord,
    PreferenceTrainingBindingReport,
    preference_audit_manifest_fingerprint,
    validate_dpo_training_records,
)
from about_llm.finetuning.preference_governance import (
    PREFERENCE_GOVERNANCE_AUDIT_VERSION,
    PreferenceGovernanceAuditReport,
)
from about_llm.finetuning.preference_near_duplicate import (
    PREFERENCE_NEAR_DUPLICATE_AUDIT_VERSION,
    PreferenceNearDuplicateAuditReport,
    PreferenceNearDuplicateView,
)
from about_llm.llmops import artifact_fingerprint

PREFERENCE_TRAINING_READINESS_VERSION = (
    "about-llm.preference-training-readiness.v2"
)
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EVIDENCE_BOUNDARY = (
    "This artifact binds one binary train-only preference identity to one passing "
    "exact/group split audit, one declared lexical candidate policy, and one "
    "source/sensitive-candidate governance audit. It contains no held-out plaintext. "
    "Its unkeyed SHA-256 fingerprints detect accidental drift but do not authenticate "
    "the issuer: a party able to replace the artifact can recompute them. Passing does "
    "not establish human-label validity, annotator agreement, position-bias control, "
    "semantic deduplication, consent, legal permission, comprehensive sensitive-data "
    "safety, target-tokenizer compatibility, alignment quality, or production convergence."
)
_SCOPE: dict[str, bool] = {
    "held_out_plaintext_embedded": False,
    "trainer_needs_held_out_access": False,
    "tie_or_invalid_coerced_to_winner": False,
    "cryptographic_origin_authenticated": False,
    "human_label_validity_verified": False,
    "annotator_agreement_estimated": False,
    "position_bias_estimated": False,
    "lexical_candidates_are_semantic_duplicates": False,
    "legal_permission_or_consent_verified": False,
    "limited_sensitive_candidate_scan_executed": True,
    "comprehensive_pii_or_secret_detection": False,
    "target_tokenizer_or_truncation_verified": False,
}
_FIELDS = {
    "artifact_version",
    "gate_passed",
    "training_ordered_dataset_fingerprint",
    "training_manifest_fingerprint",
    "combined_ordered_dataset_fingerprint",
    "combined_manifest_fingerprint",
    "binding_fingerprint",
    "binary_train_record_count",
    "excluded_nonbinary_train_count",
    "near_duplicate_audit_version",
    "near_duplicate_manifest_fingerprint",
    "near_duplicate_profile",
    "near_duplicate_ngram_size",
    "near_duplicate_threshold",
    "near_duplicate_cross_split_only",
    "near_duplicate_views",
    "near_duplicate_candidate_count",
    "governance_audit_version",
    "governance_policy_fingerprint",
    "governance_manifest_fingerprint",
    "governance_blocking_finding_count",
    "manifest_fingerprint",
    "evidence_boundary",
    "scope",
}


@dataclass(frozen=True)
class PreferenceTrainingReadinessReport:
    training_ordered_dataset_fingerprint: str
    training_manifest_fingerprint: str
    combined_ordered_dataset_fingerprint: str
    combined_manifest_fingerprint: str
    binding_fingerprint: str
    binary_train_record_count: int
    excluded_nonbinary_train_count: int
    near_duplicate_manifest_fingerprint: str
    near_duplicate_profile: NearDuplicateProfile
    near_duplicate_ngram_size: int
    near_duplicate_threshold: float
    near_duplicate_cross_split_only: bool
    near_duplicate_views: tuple[PreferenceNearDuplicateView, ...]
    near_duplicate_candidate_count: int
    governance_policy_fingerprint: str
    governance_manifest_fingerprint: str
    governance_blocking_finding_count: int

    def __post_init__(self) -> None:
        for name in (
            "training_ordered_dataset_fingerprint",
            "training_manifest_fingerprint",
            "combined_ordered_dataset_fingerprint",
            "combined_manifest_fingerprint",
            "binding_fingerprint",
            "near_duplicate_manifest_fingerprint",
            "governance_policy_fingerprint",
            "governance_manifest_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name)
        for name in (
            "binary_train_record_count",
            "excluded_nonbinary_train_count",
            "near_duplicate_candidate_count",
            "governance_blocking_finding_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.binary_train_record_count == 0:
            raise ValueError("binary_train_record_count must be positive")
        if not isinstance(self.near_duplicate_profile, NearDuplicateProfile):
            raise ValueError("near_duplicate_profile must be NearDuplicateProfile")
        if (
            isinstance(self.near_duplicate_ngram_size, bool)
            or not isinstance(self.near_duplicate_ngram_size, int)
            or self.near_duplicate_ngram_size <= 0
        ):
            raise ValueError("near_duplicate_ngram_size must be a positive integer")
        if (
            isinstance(self.near_duplicate_threshold, bool)
            or not isinstance(self.near_duplicate_threshold, (int, float))
            or not math.isfinite(float(self.near_duplicate_threshold))
            or not 0 < float(self.near_duplicate_threshold) <= 1
        ):
            raise ValueError("near_duplicate_threshold must be finite and in (0, 1]")
        if not isinstance(self.near_duplicate_cross_split_only, bool):
            raise ValueError("near_duplicate_cross_split_only must be a boolean")
        views = tuple(self.near_duplicate_views)
        if (
            not views
            or any(not isinstance(view, PreferenceNearDuplicateView) for view in views)
            or len(views) != len(set(views))
        ):
            raise ValueError("near_duplicate_views must contain unique views")
        object.__setattr__(self, "near_duplicate_views", views)
        expected_training = preference_audit_manifest_fingerprint(
            self.training_ordered_dataset_fingerprint, ("train",)
        )
        expected_combined = preference_audit_manifest_fingerprint(
            self.combined_ordered_dataset_fingerprint,
            ("train", "validation", "test"),
        )
        expected_binding = _binding_fingerprint(
            self.training_manifest_fingerprint,
            self.combined_manifest_fingerprint,
            self.excluded_nonbinary_train_count,
        )
        if self.training_manifest_fingerprint != expected_training:
            raise ValueError(
                "training_manifest_fingerprint is inconsistent with the dataset"
            )
        if self.combined_manifest_fingerprint != expected_combined:
            raise ValueError(
                "combined_manifest_fingerprint is inconsistent with the dataset"
            )
        if self.binding_fingerprint != expected_binding:
            raise ValueError("binding_fingerprint is inconsistent with its manifests")

    @classmethod
    def from_reports(
        cls,
        binding: PreferenceTrainingBindingReport,
        near_duplicate_report: PreferenceNearDuplicateAuditReport,
        governance_report: PreferenceGovernanceAuditReport,
    ) -> PreferenceTrainingReadinessReport:
        if not isinstance(binding, PreferenceTrainingBindingReport):
            raise ValueError("binding must be PreferenceTrainingBindingReport")
        if not isinstance(near_duplicate_report, PreferenceNearDuplicateAuditReport):
            raise ValueError(
                "near_duplicate_report must be PreferenceNearDuplicateAuditReport"
            )
        if not isinstance(governance_report, PreferenceGovernanceAuditReport):
            raise ValueError(
                "governance_report must be PreferenceGovernanceAuditReport"
            )
        combined_identity = binding.split_report.ordered_dataset_fingerprint
        if near_duplicate_report.ordered_dataset_fingerprint != combined_identity:
            raise ValueError("near-duplicate report dataset differs from the binding")
        if governance_report.ordered_dataset_fingerprint != combined_identity:
            raise ValueError("governance report dataset differs from the binding")
        return cls(
            training_ordered_dataset_fingerprint=(
                binding.training_report.ordered_dataset_fingerprint
            ),
            training_manifest_fingerprint=(
                binding.training_report.manifest_fingerprint
            ),
            combined_ordered_dataset_fingerprint=combined_identity,
            combined_manifest_fingerprint=binding.split_report.manifest_fingerprint,
            binding_fingerprint=binding.binding_fingerprint,
            binary_train_record_count=binding.training_report.record_count,
            excluded_nonbinary_train_count=len(
                binding.excluded_nonbinary_train_record_ids
            ),
            near_duplicate_manifest_fingerprint=(
                near_duplicate_report.manifest_fingerprint
            ),
            near_duplicate_profile=near_duplicate_report.profile,
            near_duplicate_ngram_size=near_duplicate_report.ngram_size,
            near_duplicate_threshold=near_duplicate_report.threshold,
            near_duplicate_cross_split_only=near_duplicate_report.cross_split_only,
            near_duplicate_views=near_duplicate_report.views,
            near_duplicate_candidate_count=len(near_duplicate_report.findings),
            governance_policy_fingerprint=governance_report.policy_fingerprint,
            governance_manifest_fingerprint=governance_report.manifest_fingerprint,
            governance_blocking_finding_count=(
                governance_report.blocking_finding_count
            ),
        )

    @property
    def gate_passed(self) -> bool:
        return (
            self.near_duplicate_candidate_count == 0
            and self.governance_blocking_finding_count == 0
        )

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self._identity())

    def _identity(self) -> dict[str, object]:
        return {
            "artifact_version": PREFERENCE_TRAINING_READINESS_VERSION,
            "gate_passed": self.gate_passed,
            "training_ordered_dataset_fingerprint": (
                self.training_ordered_dataset_fingerprint
            ),
            "training_manifest_fingerprint": self.training_manifest_fingerprint,
            "combined_ordered_dataset_fingerprint": (
                self.combined_ordered_dataset_fingerprint
            ),
            "combined_manifest_fingerprint": self.combined_manifest_fingerprint,
            "binding_fingerprint": self.binding_fingerprint,
            "binary_train_record_count": self.binary_train_record_count,
            "excluded_nonbinary_train_count": self.excluded_nonbinary_train_count,
            "near_duplicate_audit_version": (
                PREFERENCE_NEAR_DUPLICATE_AUDIT_VERSION
            ),
            "near_duplicate_manifest_fingerprint": (
                self.near_duplicate_manifest_fingerprint
            ),
            "near_duplicate_profile": self.near_duplicate_profile.value,
            "near_duplicate_ngram_size": self.near_duplicate_ngram_size,
            "near_duplicate_threshold": float(self.near_duplicate_threshold),
            "near_duplicate_cross_split_only": self.near_duplicate_cross_split_only,
            "near_duplicate_views": [view.value for view in self.near_duplicate_views],
            "near_duplicate_candidate_count": self.near_duplicate_candidate_count,
            "governance_audit_version": PREFERENCE_GOVERNANCE_AUDIT_VERSION,
            "governance_policy_fingerprint": self.governance_policy_fingerprint,
            "governance_manifest_fingerprint": self.governance_manifest_fingerprint,
            "governance_blocking_finding_count": (
                self.governance_blocking_finding_count
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "evidence_boundary": _EVIDENCE_BOUNDARY,
            "scope": dict(_SCOPE),
        }


def load_preference_training_readiness(
    path: Path,
) -> PreferenceTrainingReadinessReport:
    try:
        value: Any = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: readiness artifact must be a JSON object")
    payload = cast(dict[str, Any], value)
    missing = _FIELDS - set(payload)
    unknown = set(payload) - _FIELDS
    if missing or unknown:
        raise ValueError(
            f"{path}: readiness field mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if payload["artifact_version"] != PREFERENCE_TRAINING_READINESS_VERSION:
        raise ValueError(f"{path}: unsupported readiness artifact_version")
    if (
        payload["near_duplicate_audit_version"]
        != PREFERENCE_NEAR_DUPLICATE_AUDIT_VERSION
    ):
        raise ValueError(f"{path}: unsupported near_duplicate_audit_version")
    if payload["governance_audit_version"] != PREFERENCE_GOVERNANCE_AUDIT_VERSION:
        raise ValueError(f"{path}: unsupported governance_audit_version")
    if payload["evidence_boundary"] != _EVIDENCE_BOUNDARY:
        raise ValueError(f"{path}: evidence_boundary does not match the schema")
    if payload["scope"] != _SCOPE:
        raise ValueError(f"{path}: scope does not match the schema")
    gate_passed = _boolean(payload["gate_passed"], "gate_passed")
    views_raw = payload["near_duplicate_views"]
    if not isinstance(views_raw, list):
        raise ValueError("near_duplicate_views must be an array")
    try:
        profile = NearDuplicateProfile(payload["near_duplicate_profile"])
        views = tuple(PreferenceNearDuplicateView(item) for item in views_raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid near-duplicate policy enum: {error}") from error
    report = PreferenceTrainingReadinessReport(
        training_ordered_dataset_fingerprint=_string(
            payload["training_ordered_dataset_fingerprint"],
            "training_ordered_dataset_fingerprint",
        ),
        training_manifest_fingerprint=_string(
            payload["training_manifest_fingerprint"],
            "training_manifest_fingerprint",
        ),
        combined_ordered_dataset_fingerprint=_string(
            payload["combined_ordered_dataset_fingerprint"],
            "combined_ordered_dataset_fingerprint",
        ),
        combined_manifest_fingerprint=_string(
            payload["combined_manifest_fingerprint"],
            "combined_manifest_fingerprint",
        ),
        binding_fingerprint=_string(
            payload["binding_fingerprint"], "binding_fingerprint"
        ),
        binary_train_record_count=_integer(
            payload["binary_train_record_count"], "binary_train_record_count"
        ),
        excluded_nonbinary_train_count=_integer(
            payload["excluded_nonbinary_train_count"],
            "excluded_nonbinary_train_count",
        ),
        near_duplicate_manifest_fingerprint=_string(
            payload["near_duplicate_manifest_fingerprint"],
            "near_duplicate_manifest_fingerprint",
        ),
        near_duplicate_profile=profile,
        near_duplicate_ngram_size=_integer(
            payload["near_duplicate_ngram_size"], "near_duplicate_ngram_size"
        ),
        near_duplicate_threshold=_number(
            payload["near_duplicate_threshold"], "near_duplicate_threshold"
        ),
        near_duplicate_cross_split_only=_boolean(
            payload["near_duplicate_cross_split_only"],
            "near_duplicate_cross_split_only",
        ),
        near_duplicate_views=views,
        near_duplicate_candidate_count=_integer(
            payload["near_duplicate_candidate_count"],
            "near_duplicate_candidate_count",
        ),
        governance_policy_fingerprint=_string(
            payload["governance_policy_fingerprint"],
            "governance_policy_fingerprint",
        ),
        governance_manifest_fingerprint=_string(
            payload["governance_manifest_fingerprint"],
            "governance_manifest_fingerprint",
        ),
        governance_blocking_finding_count=_integer(
            payload["governance_blocking_finding_count"],
            "governance_blocking_finding_count",
        ),
    )
    if gate_passed != report.gate_passed:
        raise ValueError("gate_passed is inconsistent with blocking counts")
    manifest = _string(payload["manifest_fingerprint"], "manifest_fingerprint")
    if manifest != report.manifest_fingerprint:
        raise ValueError(
            "readiness manifest_fingerprint mismatch; artifact may be stale or tampered"
        )
    return report


def validate_preference_training_readiness(
    records: tuple[PreferenceRecord, ...],
    readiness: PreferenceTrainingReadinessReport,
) -> PreferenceDataAuditReport:
    if not isinstance(readiness, PreferenceTrainingReadinessReport):
        raise ValueError("readiness must be PreferenceTrainingReadinessReport")
    report = validate_dpo_training_records(records)
    if report.ordered_dataset_fingerprint != (
        readiness.training_ordered_dataset_fingerprint
    ):
        raise ValueError("training data ordered fingerprint differs from readiness")
    if report.manifest_fingerprint != readiness.training_manifest_fingerprint:
        raise ValueError("training data manifest differs from readiness")
    if report.record_count != readiness.binary_train_record_count:
        raise ValueError("training record count differs from readiness")
    if not readiness.gate_passed:
        raise ValueError(
            "preference readiness gate failed with "
            f"{readiness.near_duplicate_candidate_count} lexical candidate(s) and "
            f"{readiness.governance_blocking_finding_count} governance finding(s)"
        )
    return report


def _binding_fingerprint(
    training_manifest: str, combined_manifest: str, excluded_count: int
) -> str:
    return "sha256:" + artifact_fingerprint(
        {
            "contract_version": PREFERENCE_DATA_CONTRACT_VERSION,
            "training_manifest_fingerprint": training_manifest,
            "split_manifest_fingerprint": combined_manifest,
            "binding_rule": "ordered_binary_train_subset_exactly_matches",
            "excluded_nonbinary_train_count": excluded_count,
        }
    )


def _require_fingerprint(value: object, name: str) -> None:
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase sha256 fingerprint")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
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
