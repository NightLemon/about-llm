"""Strict raw judgment artifacts and transparent paired-preference statistics."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np

from about_llm.finetuning.data import DataSplit
from about_llm.finetuning.preference_data import (
    PreferenceLabel,
    PreferenceRecord,
    PreferenceStrength,
    audit_preference_records,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

PREFERENCE_JUDGMENT_CONTRACT_VERSION = "about-llm.preference-judgment-jsonl.v1"
PREFERENCE_JUDGMENT_AUDIT_VERSION = "about-llm.preference-judgment-audit.v1"
PREFERENCE_EVALUATION_VERSION = "about-llm.preference-evaluation.v1"


@dataclass(frozen=True)
class PreferenceJudgment:
    judgment_id: str
    pair_id: str
    annotator_id: str
    assignment_batch_id: str
    presentation_order: tuple[str, str]
    label: PreferenceLabel
    preference_strength: PreferenceStrength
    rubric_revision: str
    blind_model_identity: bool
    independent_judgment: bool
    duration_ms: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _record_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("judgment_id", self.judgment_id),
            ("pair_id", self.pair_id),
            ("annotator_id", self.annotator_id),
            ("assignment_batch_id", self.assignment_batch_id),
            ("rubric_revision", self.rubric_revision),
        ):
            _require_nonempty_string(value, name)
        order = tuple(self.presentation_order)
        if len(order) != 2 or set(order) != {"a", "b"}:
            raise ValueError("presentation_order must be ['a','b'] or ['b','a']")
        object.__setattr__(self, "presentation_order", order)
        if not isinstance(self.label, PreferenceLabel):
            raise ValueError("label must be PreferenceLabel")
        if not isinstance(self.preference_strength, PreferenceStrength):
            raise ValueError("preference_strength must be PreferenceStrength")
        binary = self.label in (PreferenceLabel.A, PreferenceLabel.B)
        applicable = self.preference_strength in (
            PreferenceStrength.SLIGHT,
            PreferenceStrength.CLEAR,
        )
        if binary is not applicable:
            raise ValueError(
                "binary labels require slight/clear; tie/invalid requires not_applicable"
            )
        if not isinstance(self.blind_model_identity, bool):
            raise ValueError("blind_model_identity must be a boolean")
        if not isinstance(self.independent_judgment, bool):
            raise ValueError("independent_judgment must be a boolean")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms <= 0
        ):
            raise ValueError("duration_ms must be a positive integer")
        metadata = _strict_json_snapshot(self.metadata, "metadata")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze_object(metadata))
        object.__setattr__(
            self,
            "_record_fingerprint",
            "sha256:" + artifact_fingerprint(self.to_dict()),
        )

    @property
    def record_fingerprint(self) -> str:
        return self._record_fingerprint

    @property
    def order_key(self) -> str:
        return "a_first" if self.presentation_order[0] == "a" else "b_first"

    @property
    def preferred_display_position(self) -> str:
        if self.label is PreferenceLabel.TIE:
            return "tie"
        if self.label is PreferenceLabel.INVALID:
            return "invalid"
        return "first" if self.presentation_order[0] == self.label.value else "second"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.judgment_id,
            "pair_id": self.pair_id,
            "annotator_id": self.annotator_id,
            "assignment_batch_id": self.assignment_batch_id,
            "presentation_order": list(self.presentation_order),
            "label": self.label.value,
            "preference_strength": self.preference_strength.value,
            "rubric_revision": self.rubric_revision,
            "blind_model_identity": self.blind_model_identity,
            "independent_judgment": self.independent_judgment,
            "duration_ms": self.duration_ms,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True)
class PreferenceJudgmentAuditReport:
    case_audit_manifest_fingerprint: str
    selected_case_ids: tuple[str, ...]
    selected_splits: tuple[str, ...]
    judgments_per_pair: int
    minimum_judgments_per_order: int
    judgment_count: int
    annotator_count: int
    duplicate_judgment_ids: tuple[str, ...]
    unknown_pair_judgment_ids: tuple[str, ...]
    out_of_scope_pair_judgment_ids: tuple[str, ...]
    duplicate_annotator_pair_assignments: tuple[str, ...]
    rubric_mismatch_judgment_ids: tuple[str, ...]
    unblinded_judgment_ids: tuple[str, ...]
    nonindependent_judgment_ids: tuple[str, ...]
    count_mismatch_pair_ids: tuple[str, ...]
    order_coverage_mismatch_pair_ids: tuple[str, ...]
    pair_judgment_counts: Mapping[str, int]
    pair_order_counts: Mapping[str, Mapping[str, int]]
    ordered_judgment_fingerprint: str

    @property
    def gate_passed(self) -> bool:
        return not any(
            (
                self.duplicate_judgment_ids,
                self.unknown_pair_judgment_ids,
                self.out_of_scope_pair_judgment_ids,
                self.duplicate_annotator_pair_assignments,
                self.rubric_mismatch_judgment_ids,
                self.unblinded_judgment_ids,
                self.nonindependent_judgment_ids,
                self.count_mismatch_pair_ids,
                self.order_coverage_mismatch_pair_ids,
            )
        )

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "audit_version": PREFERENCE_JUDGMENT_AUDIT_VERSION,
                "case_audit_manifest_fingerprint": (
                    self.case_audit_manifest_fingerprint
                ),
                "selected_case_ids": list(self.selected_case_ids),
                "selected_splits": list(self.selected_splits),
                "judgments_per_pair": self.judgments_per_pair,
                "minimum_judgments_per_order": (
                    self.minimum_judgments_per_order
                ),
                "ordered_judgment_fingerprint": self.ordered_judgment_fingerprint,
                "gate_rules": [
                    "judgment_ids_unique",
                    "pair_ids_known_and_in_selected_splits",
                    "annotator_rates_each_pair_at_most_once",
                    "rubric_matches_pair",
                    "blind_and_independent_declarations_true",
                    "exact_judgment_count_per_pair",
                    "minimum_coverage_for_each_presentation_order",
                ],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_version": PREFERENCE_JUDGMENT_AUDIT_VERSION,
            "judgment_contract_version": PREFERENCE_JUDGMENT_CONTRACT_VERSION,
            "gate_passed": self.gate_passed,
            "case_audit_manifest_fingerprint": self.case_audit_manifest_fingerprint,
            "selected_case_ids": list(self.selected_case_ids),
            "selected_splits": list(self.selected_splits),
            "judgments_per_pair": self.judgments_per_pair,
            "minimum_judgments_per_order": self.minimum_judgments_per_order,
            "judgment_count": self.judgment_count,
            "annotator_count": self.annotator_count,
            "duplicate_judgment_ids": list(self.duplicate_judgment_ids),
            "unknown_pair_judgment_ids": list(self.unknown_pair_judgment_ids),
            "out_of_scope_pair_judgment_ids": list(
                self.out_of_scope_pair_judgment_ids
            ),
            "duplicate_annotator_pair_assignments": list(
                self.duplicate_annotator_pair_assignments
            ),
            "rubric_mismatch_judgment_ids": list(
                self.rubric_mismatch_judgment_ids
            ),
            "unblinded_judgment_ids": list(self.unblinded_judgment_ids),
            "nonindependent_judgment_ids": list(
                self.nonindependent_judgment_ids
            ),
            "count_mismatch_pair_ids": list(self.count_mismatch_pair_ids),
            "order_coverage_mismatch_pair_ids": list(
                self.order_coverage_mismatch_pair_ids
            ),
            "pair_judgment_counts": dict(self.pair_judgment_counts),
            "pair_order_counts": {
                pair_id: dict(counts)
                for pair_id, counts in self.pair_order_counts.items()
            },
            "ordered_judgment_fingerprint": self.ordered_judgment_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "scope": {
                "blindness_field_is_a_declaration_not_observation": True,
                "random_assignment_verified": False,
                "annotator_identity_is_pseudonymous_verified": False,
                "annotator_training_or_expertise_verified": False,
                "raw_judgments_preserved": True,
                "tie_and_invalid_preserved": True,
            },
        }


@dataclass(frozen=True)
class PreferenceEvaluationSummary:
    audit_manifest_fingerprint: str
    label_counts: Mapping[str, int]
    preferred_display_position_counts: Mapping[str, int]
    binary_judgment_count: int
    pairwise_agreement_numerator: int
    pairwise_agreement_denominator: int
    pairwise_agreement: float
    fleiss_kappa: float | None
    fleiss_expected_agreement: float
    a_first_binary_count: int
    a_second_binary_count: int
    a_selected_when_first_count: int
    a_selected_when_second_count: int
    a_selection_rate_when_first: float | None
    a_selection_rate_when_second: float | None
    mean_pair_position_effect: float | None
    position_effect_pair_count: int
    position_effect_confidence: float
    position_effect_confidence_low: float | None
    position_effect_confidence_high: float | None
    bootstrap_samples: int
    bootstrap_seed: int
    pair_summaries: tuple[Mapping[str, object], ...]

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self._identity())

    def _identity(self) -> dict[str, object]:
        return {
            "evaluation_version": PREFERENCE_EVALUATION_VERSION,
            "audit_manifest_fingerprint": self.audit_manifest_fingerprint,
            "label_counts": dict(self.label_counts),
            "preferred_display_position_counts": dict(
                self.preferred_display_position_counts
            ),
            "binary_judgment_count": self.binary_judgment_count,
            "pairwise_agreement_numerator": self.pairwise_agreement_numerator,
            "pairwise_agreement_denominator": self.pairwise_agreement_denominator,
            "pairwise_agreement": self.pairwise_agreement,
            "fleiss_kappa": self.fleiss_kappa,
            "fleiss_expected_agreement": self.fleiss_expected_agreement,
            "a_first_binary_count": self.a_first_binary_count,
            "a_second_binary_count": self.a_second_binary_count,
            "a_selected_when_first_count": self.a_selected_when_first_count,
            "a_selected_when_second_count": self.a_selected_when_second_count,
            "a_selection_rate_when_first": self.a_selection_rate_when_first,
            "a_selection_rate_when_second": self.a_selection_rate_when_second,
            "mean_pair_position_effect": self.mean_pair_position_effect,
            "position_effect_pair_count": self.position_effect_pair_count,
            "position_effect_confidence": self.position_effect_confidence,
            "position_effect_confidence_low": self.position_effect_confidence_low,
            "position_effect_confidence_high": self.position_effect_confidence_high,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "pair_summaries": [dict(item) for item in self.pair_summaries],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "metric_definitions": {
                "pairwise_agreement": (
                    "same-label unordered annotator pairs / all unordered "
                    "annotator pairs within cases; invalid is retained as a label"
                ),
                "fleiss_kappa": (
                    "chance-corrected agreement over a/b/tie/invalid with an exact "
                    "equal rater count per case; null when chance agreement is one"
                ),
                "position_effect": (
                    "within-case P(select A | A displayed first) minus "
                    "P(select A | A displayed second), excluding tie/invalid"
                ),
                "position_effect_interval": (
                    "percentile bootstrap over case-level position effects; cases, "
                    "not judgments, are the resampling unit"
                ),
            },
            "scope": {
                "authored_fixture_is_human_evidence": False,
                "causal_position_bias_identified": False,
                "random_assignment_verified": False,
                "annotator_quality_verified": False,
                "rubric_validity_verified": False,
                "multiple_comparison_control_applied": False,
            },
            "evidence_boundary": (
                "These statistics are descriptive evidence for the supplied raw "
                "judgments. Position order is preserved but random assignment is not "
                "proven, so the position effect is not automatically causal. High "
                "agreement can reflect a shared bad rubric, and the case-cluster "
                "bootstrap does not repair case-selection or annotator-selection bias."
            ),
        }


def load_preference_judgments(path: Path) -> tuple[PreferenceJudgment, ...]:
    records: list[PreferenceJudgment] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: invalid strict JSON: {error}"
            ) from error
        prefix = f"{path}:{line_number}"
        item = _object(value, prefix)
        fields = {
            "id",
            "pair_id",
            "annotator_id",
            "assignment_batch_id",
            "presentation_order",
            "label",
            "preference_strength",
            "rubric_revision",
            "blind_model_identity",
            "independent_judgment",
            "duration_ms",
            "metadata",
        }
        _expect_fields(
            item,
            required=fields - {"metadata"},
            allowed=fields,
            prefix=prefix,
        )
        order = _array(item["presentation_order"], f"{prefix}.presentation_order")
        if len(order) != 2:
            raise ValueError(f"{prefix}.presentation_order must contain two values")
        try:
            label = PreferenceLabel(item["label"])
            strength = PreferenceStrength(item["preference_strength"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}: unsupported label or strength") from error
        try:
            records.append(
                PreferenceJudgment(
                    judgment_id=_string(item["id"], f"{prefix}.id"),
                    pair_id=_string(item["pair_id"], f"{prefix}.pair_id"),
                    annotator_id=_string(
                        item["annotator_id"], f"{prefix}.annotator_id"
                    ),
                    assignment_batch_id=_string(
                        item["assignment_batch_id"],
                        f"{prefix}.assignment_batch_id",
                    ),
                    presentation_order=(
                        _string(order[0], f"{prefix}.presentation_order[0]"),
                        _string(order[1], f"{prefix}.presentation_order[1]"),
                    ),
                    label=label,
                    preference_strength=strength,
                    rubric_revision=_string(
                        item["rubric_revision"], f"{prefix}.rubric_revision"
                    ),
                    blind_model_identity=_boolean(
                        item["blind_model_identity"],
                        f"{prefix}.blind_model_identity",
                    ),
                    independent_judgment=_boolean(
                        item["independent_judgment"],
                        f"{prefix}.independent_judgment",
                    ),
                    duration_ms=_integer(
                        item["duration_ms"], f"{prefix}.duration_ms"
                    ),
                    metadata=item.get("metadata", {}),
                )
            )
        except ValueError as error:
            raise ValueError(f"{prefix}: {error}") from error
    if not records:
        raise ValueError(f"{path}: judgment dataset contains no records")
    return tuple(records)


def audit_preference_judgments(
    cases: Iterable[PreferenceRecord],
    judgments: Iterable[PreferenceJudgment],
    *,
    selected_splits: Iterable[DataSplit] = (DataSplit.VALIDATION, DataSplit.TEST),
    judgments_per_pair: int,
    minimum_judgments_per_order: int,
) -> PreferenceJudgmentAuditReport:
    case_snapshot = tuple(cases)
    judgment_snapshot = tuple(judgments)
    splits = tuple(selected_splits)
    if (
        not splits
        or any(not isinstance(split, DataSplit) for split in splits)
        or len(splits) != len(set(splits))
    ):
        raise ValueError("selected_splits must contain unique DataSplit values")
    if DataSplit.TRAIN in splits:
        raise ValueError("human evaluation selected_splits must not include train")
    if (
        isinstance(judgments_per_pair, bool)
        or not isinstance(judgments_per_pair, int)
        or judgments_per_pair < 2
    ):
        raise ValueError("judgments_per_pair must be an integer >= 2")
    if (
        isinstance(minimum_judgments_per_order, bool)
        or not isinstance(minimum_judgments_per_order, int)
        or minimum_judgments_per_order <= 0
        or 2 * minimum_judgments_per_order > judgments_per_pair
    ):
        raise ValueError(
            "minimum_judgments_per_order must be positive and fit the pair total"
        )
    if not judgment_snapshot or any(
        not isinstance(item, PreferenceJudgment) for item in judgment_snapshot
    ):
        raise ValueError("judgment audit requires PreferenceJudgment values")
    case_audit = audit_preference_records(case_snapshot, required_splits=splits)
    if not case_audit.gate_passed:
        raise ValueError("case preference audit failed")
    case_by_id = {case.record_id: case for case in case_snapshot}
    selected_cases = tuple(case for case in case_snapshot if case.split in splits)
    selected_by_id = {case.record_id: case for case in selected_cases}
    if not selected_cases:
        raise ValueError("selected splits contain no evaluation cases")

    judgment_id_counts = Counter(item.judgment_id for item in judgment_snapshot)
    annotator_pair_counts = Counter(
        (item.annotator_id, item.pair_id) for item in judgment_snapshot
    )
    known = tuple(item for item in judgment_snapshot if item.pair_id in selected_by_id)
    grouped: dict[str, list[PreferenceJudgment]] = defaultdict(list)
    for item in known:
        grouped[item.pair_id].append(item)
    pair_counts = {
        case.record_id: len(grouped[case.record_id]) for case in selected_cases
    }
    order_counts = {
        case.record_id: {
            "a_first": sum(
                item.order_key == "a_first" for item in grouped[case.record_id]
            ),
            "b_first": sum(
                item.order_key == "b_first" for item in grouped[case.record_id]
            ),
        }
        for case in selected_cases
    }
    ordered_identity = {
        "ordered_record_fingerprints": [
            item.record_fingerprint for item in judgment_snapshot
        ]
    }
    return PreferenceJudgmentAuditReport(
        case_audit_manifest_fingerprint=case_audit.manifest_fingerprint,
        selected_case_ids=tuple(case.record_id for case in selected_cases),
        selected_splits=tuple(split.value for split in splits),
        judgments_per_pair=judgments_per_pair,
        minimum_judgments_per_order=minimum_judgments_per_order,
        judgment_count=len(judgment_snapshot),
        annotator_count=len({item.annotator_id for item in judgment_snapshot}),
        duplicate_judgment_ids=tuple(
            key for key, count in sorted(judgment_id_counts.items()) if count > 1
        ),
        unknown_pair_judgment_ids=tuple(
            sorted(
                item.judgment_id
                for item in judgment_snapshot
                if item.pair_id not in case_by_id
            )
        ),
        out_of_scope_pair_judgment_ids=tuple(
            sorted(
                item.judgment_id
                for item in judgment_snapshot
                if item.pair_id in case_by_id and item.pair_id not in selected_by_id
            )
        ),
        duplicate_annotator_pair_assignments=tuple(
            f"{annotator_id}:{pair_id}"
            for (annotator_id, pair_id), count in sorted(
                annotator_pair_counts.items()
            )
            if count > 1
        ),
        rubric_mismatch_judgment_ids=tuple(
            sorted(
                item.judgment_id
                for item in known
                if item.rubric_revision != selected_by_id[item.pair_id].rubric_revision
            )
        ),
        unblinded_judgment_ids=tuple(
            sorted(
                item.judgment_id
                for item in judgment_snapshot
                if not item.blind_model_identity
            )
        ),
        nonindependent_judgment_ids=tuple(
            sorted(
                item.judgment_id
                for item in judgment_snapshot
                if not item.independent_judgment
            )
        ),
        count_mismatch_pair_ids=tuple(
            sorted(
                pair_id
                for pair_id, count in pair_counts.items()
                if count != judgments_per_pair
            )
        ),
        order_coverage_mismatch_pair_ids=tuple(
            sorted(
                pair_id
                for pair_id, counts in order_counts.items()
                if min(counts.values()) < minimum_judgments_per_order
            )
        ),
        pair_judgment_counts=MappingProxyType(dict(sorted(pair_counts.items()))),
        pair_order_counts=MappingProxyType(
            {
                pair_id: MappingProxyType(dict(sorted(counts.items())))
                for pair_id, counts in sorted(order_counts.items())
            }
        ),
        ordered_judgment_fingerprint="sha256:"
        + artifact_fingerprint(ordered_identity),
    )


def summarize_preference_judgments(
    judgments: Iterable[PreferenceJudgment],
    audit: PreferenceJudgmentAuditReport,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> PreferenceEvaluationSummary:
    snapshot = tuple(judgments)
    if not isinstance(audit, PreferenceJudgmentAuditReport) or not audit.gate_passed:
        raise ValueError("a passing PreferenceJudgmentAuditReport is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be positive")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    expected_fingerprint = "sha256:" + artifact_fingerprint(
        {
            "ordered_record_fingerprints": [
                item.record_fingerprint for item in snapshot
            ]
        }
    )
    if expected_fingerprint != audit.ordered_judgment_fingerprint:
        raise ValueError("judgments differ from the audited ordered artifact")
    selected = set(audit.selected_case_ids)
    grouped: dict[str, list[PreferenceJudgment]] = defaultdict(list)
    for item in snapshot:
        if item.pair_id in selected:
            grouped[item.pair_id].append(item)

    agreement_numerator = 0
    agreement_denominator = 0
    category_counts = Counter[str]()
    pair_summaries: list[Mapping[str, object]] = []
    pair_effects: list[float] = []
    for pair_id in audit.selected_case_ids:
        items = grouped[pair_id]
        labels = [item.label.value for item in items]
        counts = Counter(labels)
        category_counts.update(labels)
        for left, right in combinations(labels, 2):
            agreement_denominator += 1
            agreement_numerator += left == right
        a_first = [
            item
            for item in items
            if item.order_key == "a_first" and item.label in _BINARY_LABELS
        ]
        a_second = [
            item
            for item in items
            if item.order_key == "b_first" and item.label in _BINARY_LABELS
        ]
        rate_first = _a_rate(a_first)
        rate_second = _a_rate(a_second)
        effect = (
            rate_first - rate_second
            if rate_first is not None and rate_second is not None
            else None
        )
        if effect is not None:
            pair_effects.append(effect)
        pair_summaries.append(
            MappingProxyType(
                {
                    "pair_id": pair_id,
                    "label_counts": dict(sorted(counts.items())),
                    "a_first_binary_count": len(a_first),
                    "a_second_binary_count": len(a_second),
                    "a_selection_rate_when_first": rate_first,
                    "a_selection_rate_when_second": rate_second,
                    "position_effect": effect,
                }
            )
        )
    if agreement_denominator == 0:
        raise ValueError("agreement denominator is zero")
    kappa, expected_agreement = _fleiss_kappa(
        grouped, audit.selected_case_ids, audit.judgments_per_pair
    )
    a_first_binary = [
        item
        for item in snapshot
        if item.pair_id in selected
        and item.order_key == "a_first"
        and item.label in _BINARY_LABELS
    ]
    a_second_binary = [
        item
        for item in snapshot
        if item.pair_id in selected
        and item.order_key == "b_first"
        and item.label in _BINARY_LABELS
    ]
    mean_effect, low, high = _cluster_bootstrap_effect(
        pair_effects,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    display_counts = Counter(
        item.preferred_display_position
        for item in snapshot
        if item.pair_id in selected
    )
    return PreferenceEvaluationSummary(
        audit_manifest_fingerprint=audit.manifest_fingerprint,
        label_counts=_counts(category_counts),
        preferred_display_position_counts=_counts(display_counts),
        binary_judgment_count=sum(
            item.label in _BINARY_LABELS
            for item in snapshot
            if item.pair_id in selected
        ),
        pairwise_agreement_numerator=agreement_numerator,
        pairwise_agreement_denominator=agreement_denominator,
        pairwise_agreement=agreement_numerator / agreement_denominator,
        fleiss_kappa=kappa,
        fleiss_expected_agreement=expected_agreement,
        a_first_binary_count=len(a_first_binary),
        a_second_binary_count=len(a_second_binary),
        a_selected_when_first_count=sum(
            item.label is PreferenceLabel.A for item in a_first_binary
        ),
        a_selected_when_second_count=sum(
            item.label is PreferenceLabel.A for item in a_second_binary
        ),
        a_selection_rate_when_first=_a_rate(a_first_binary),
        a_selection_rate_when_second=_a_rate(a_second_binary),
        mean_pair_position_effect=mean_effect,
        position_effect_pair_count=len(pair_effects),
        position_effect_confidence=confidence,
        position_effect_confidence_low=low,
        position_effect_confidence_high=high,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        pair_summaries=tuple(pair_summaries),
    )


_BINARY_LABELS = (PreferenceLabel.A, PreferenceLabel.B)


def _a_rate(items: list[PreferenceJudgment]) -> float | None:
    if not items:
        return None
    return sum(item.label is PreferenceLabel.A for item in items) / len(items)


def _fleiss_kappa(
    grouped: Mapping[str, list[PreferenceJudgment]],
    pair_ids: tuple[str, ...],
    ratings_per_pair: int,
) -> tuple[float | None, float]:
    categories = tuple(label.value for label in PreferenceLabel)
    total_counts = Counter[str]()
    observed: list[float] = []
    for pair_id in pair_ids:
        counts = Counter(item.label.value for item in grouped[pair_id])
        if sum(counts.values()) != ratings_per_pair:
            raise ValueError("Fleiss kappa requires the audited equal rater count")
        total_counts.update(counts)
        observed.append(
            (sum(counts[category] ** 2 for category in categories) - ratings_per_pair)
            / (ratings_per_pair * (ratings_per_pair - 1))
        )
    total_ratings = len(pair_ids) * ratings_per_pair
    proportions = [total_counts[category] / total_ratings for category in categories]
    expected = sum(value * value for value in proportions)
    observed_mean = sum(observed) / len(observed)
    if math.isclose(expected, 1.0, rel_tol=0, abs_tol=1e-15):
        return None, expected
    return (observed_mean - expected) / (1 - expected), expected


def _cluster_bootstrap_effect(
    effects: list[float], *, confidence: float, samples: int, seed: int
) -> tuple[float | None, float | None, float | None]:
    if not effects:
        return None, None, None
    array = np.asarray(effects, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(array.mean()), float(low), float(high)


def _counts(value: Counter[str]) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(value.items())))


def _strict_json_snapshot(value: object, prefix: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix} must be strict JSON: {error}") from error


def _freeze_object(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _freeze_object(cast(dict[str, Any], value))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _object(value: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: expected an object")
    return cast(dict[str, Any], value)


def _array(value: Any, prefix: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{prefix}: expected an array")
    return value


def _expect_fields(
    value: Mapping[str, Any],
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


def _integer(value: Any, prefix: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{prefix} must be an integer")
    return cast(int, value)


def _boolean(value: Any, prefix: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{prefix} must be a boolean")
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
