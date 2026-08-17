"""Auditable synthetic-data lineage, verifier gates, and mixture exposure math."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum


class FingerprintProfile(str, Enum):
    """Explicit content identity profiles with different information loss."""

    BYTE_EXACT = "byte_exact"
    NFC_WHITESPACE = "nfc_whitespace"


class SourceKind(str, Enum):
    """Whether a mixture component is a real anchor or model-generated data."""

    REAL = "real"
    SYNTHETIC = "synthetic"


def _nonempty(name: str, value: str) -> None:
    if not value or value.isspace():
        raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class VerificationResult:
    """One versioned verifier decision for one candidate."""

    verifier_id: str
    revision: str
    passed: bool

    def __post_init__(self) -> None:
        _nonempty("verifier_id", self.verifier_id)
        _nonempty("revision", self.revision)
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")


@dataclass(frozen=True)
class SyntheticRecord:
    """A generated candidate with explicit parents and verifier artifacts."""

    record_id: str
    content: str
    parent_ids: tuple[str, ...]
    generator_revision: str
    prompt_revision: str
    generation_round: int
    verifications: tuple[VerificationResult, ...]
    human_reviewed: bool = False

    def __post_init__(self) -> None:
        for name in ("record_id", "content", "generator_revision", "prompt_revision"):
            _nonempty(name, getattr(self, name))
        if isinstance(self.generation_round, bool) or not isinstance(self.generation_round, int):
            raise TypeError("generation_round must be an integer")
        if self.generation_round <= 0:
            raise ValueError("generation_round must be positive")
        if not self.parent_ids:
            raise ValueError("parent_ids must not be empty")
        if any(not parent or parent.isspace() for parent in self.parent_ids):
            raise ValueError("parent_ids must not contain empty values")
        if len(self.parent_ids) != len(set(self.parent_ids)):
            raise ValueError("parent_ids must be unique")
        if self.record_id in self.parent_ids:
            raise ValueError("a record cannot be its own parent")
        verifier_ids = [result.verifier_id for result in self.verifications]
        if len(verifier_ids) != len(set(verifier_ids)):
            raise ValueError("verifier_id must be unique within a record")
        if not isinstance(self.human_reviewed, bool):
            raise TypeError("human_reviewed must be a boolean")


@dataclass(frozen=True)
class GenerationRoundSummary:
    generation_round: int
    candidate_count: int
    eligible_count: int


@dataclass(frozen=True)
class SyntheticAuditReport:
    """Contract diagnostics; eligibility is not a semantic quality guarantee."""

    candidate_count: int
    eligible_count: int
    eligible_unique_content_count: int
    human_reviewed_count: int
    eligible_human_reviewed_count: int
    self_verified_record_ids: tuple[str, ...]
    missing_verifier_record_ids: tuple[str, ...]
    failed_verifier_record_ids: tuple[str, ...]
    unresolved_parent_pairs: tuple[tuple[str, str], ...]
    nonmonotonic_parent_pairs: tuple[tuple[str, str], ...]
    lineage_cycle_record_ids: tuple[str, ...]
    duplicate_content_groups: tuple[tuple[str, ...], ...]
    eligible_record_ids: tuple[str, ...]
    rounds: tuple[GenerationRoundSummary, ...]
    fingerprint_profile: FingerprintProfile

    @property
    def eligibility_rate(self) -> float:
        return self.eligible_count / self.candidate_count


def content_fingerprint(text: str, *, profile: FingerprintProfile) -> str:
    """Return a SHA-256 identity under one explicit normalization profile.

    ``NFC_WHITESPACE`` is suitable only when whitespace is known not to carry
    meaning. It must not be silently applied to code, tables, or formatting tasks.
    """

    _nonempty("text", text)
    if not isinstance(profile, FingerprintProfile):
        raise TypeError("profile must be a FingerprintProfile")
    normalized = text
    if profile is FingerprintProfile.NFC_WHITESPACE:
        normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _lineage_cycle_record_ids(
    records_by_id: dict[str, SyntheticRecord],
) -> tuple[str, ...]:
    """Return only records that participate in a synthetic-parent cycle."""

    parents_by_id = {
        record_id: tuple(
            sorted(
                parent_id
                for parent_id in record.parent_ids
                if parent_id in records_by_id
            )
        )
        for record_id, record in records_by_id.items()
    }
    state: dict[str, int] = {record_id: 0 for record_id in records_by_id}
    stack: list[str] = []
    stack_index: dict[str, int] = {}
    cycle_ids: set[str] = set()

    def visit(record_id: str) -> None:
        state[record_id] = 1
        stack_index[record_id] = len(stack)
        stack.append(record_id)
        for parent_id in parents_by_id[record_id]:
            if state[parent_id] == 0:
                visit(parent_id)
            elif state[parent_id] == 1:
                cycle_ids.update(stack[stack_index[parent_id] :])
        stack.pop()
        del stack_index[record_id]
        state[record_id] = 2

    for record_id in sorted(records_by_id):
        if state[record_id] == 0:
            visit(record_id)
    return tuple(sorted(cycle_ids))


def audit_synthetic_records(
    records: Iterable[SyntheticRecord],
    *,
    required_verifiers: Collection[str],
    known_parent_ids: Collection[str] = (),
    fingerprint_profile: FingerprintProfile = FingerprintProfile.BYTE_EXACT,
) -> SyntheticAuditReport:
    """Audit lineage, gate outcomes, exact identities, and generator overlap.

    A record is *eligible* only when every named required verifier is present and
    passes. Eligibility means the declared gate passed; it does not prove factual
    correctness, diversity, safety, licensing, or absence of model collapse.
    """

    materialized = tuple(records)
    if not materialized:
        raise ValueError("records must not be empty")
    required = tuple(required_verifiers)
    if not required or any(not item or item.isspace() for item in required):
        raise ValueError("required_verifiers must contain non-empty verifier ids")
    if len(required) != len(set(required)):
        raise ValueError("required_verifiers must be unique")
    record_ids = [record.record_id for record in materialized]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record_id must be unique")
    known_values = tuple(known_parent_ids)
    if any(not item or item.isspace() for item in known_values):
        raise ValueError("known_parent_ids must not contain empty values")
    if len(known_values) != len(set(known_values)):
        raise ValueError("known_parent_ids must be unique")
    known = set(known_values)

    record_id_set = set(record_ids)
    records_by_id = {record.record_id: record for record in materialized}
    missing: list[str] = []
    failed: list[str] = []
    eligible: list[SyntheticRecord] = []
    self_verified: list[str] = []
    unresolved: list[tuple[str, str]] = []
    nonmonotonic: list[tuple[str, str]] = []
    digest_groups: dict[str, list[str]] = defaultdict(list)

    for record in materialized:
        by_id = {result.verifier_id: result for result in record.verifications}
        if any(result.revision == record.generator_revision for result in record.verifications):
            self_verified.append(record.record_id)
        absent = [verifier_id for verifier_id in required if verifier_id not in by_id]
        rejected = [
            verifier_id
            for verifier_id in required
            if verifier_id in by_id and not by_id[verifier_id].passed
        ]
        if absent:
            missing.append(record.record_id)
        if rejected:
            failed.append(record.record_id)
        if not absent and not rejected:
            eligible.append(record)
        for parent_id in record.parent_ids:
            if parent_id not in record_id_set and parent_id not in known:
                unresolved.append((record.record_id, parent_id))
            elif parent_id in records_by_id and (
                record.generation_round
                <= records_by_id[parent_id].generation_round
            ):
                nonmonotonic.append((record.record_id, parent_id))
        digest_groups[
            content_fingerprint(record.content, profile=fingerprint_profile)
        ].append(record.record_id)

    eligible_ids = {record.record_id for record in eligible}
    eligible_digests = {
        content_fingerprint(record.content, profile=fingerprint_profile)
        for record in eligible
    }
    round_counts = Counter(record.generation_round for record in materialized)
    eligible_round_counts = Counter(record.generation_round for record in eligible)
    rounds = tuple(
        GenerationRoundSummary(
            generation_round=round_id,
            candidate_count=round_counts[round_id],
            eligible_count=eligible_round_counts[round_id],
        )
        for round_id in sorted(round_counts)
    )
    duplicate_groups = tuple(
        tuple(sorted(ids))
        for ids in sorted(digest_groups.values(), key=lambda group: tuple(sorted(group)))
        if len(ids) > 1
    )
    return SyntheticAuditReport(
        candidate_count=len(materialized),
        eligible_count=len(eligible),
        eligible_unique_content_count=len(eligible_digests),
        human_reviewed_count=sum(record.human_reviewed for record in materialized),
        eligible_human_reviewed_count=sum(
            record.human_reviewed for record in materialized if record.record_id in eligible_ids
        ),
        self_verified_record_ids=tuple(sorted(self_verified)),
        missing_verifier_record_ids=tuple(sorted(missing)),
        failed_verifier_record_ids=tuple(sorted(failed)),
        unresolved_parent_pairs=tuple(sorted(unresolved)),
        nonmonotonic_parent_pairs=tuple(sorted(nonmonotonic)),
        lineage_cycle_record_ids=_lineage_cycle_record_ids(records_by_id),
        duplicate_content_groups=duplicate_groups,
        eligible_record_ids=tuple(sorted(eligible_ids)),
        rounds=rounds,
        fingerprint_profile=fingerprint_profile,
    )


@dataclass(frozen=True)
class MixtureComponent:
    name: str
    source_kind: SourceKind
    unique_tokens: int
    weight: float
    generation_round: int | None = None

    def __post_init__(self) -> None:
        _nonempty("name", self.name)
        if not isinstance(self.source_kind, SourceKind):
            raise TypeError("source_kind must be a SourceKind")
        if isinstance(self.unique_tokens, bool) or not isinstance(self.unique_tokens, int):
            raise TypeError("unique_tokens must be an integer")
        if self.unique_tokens <= 0:
            raise ValueError("unique_tokens must be positive")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("weight must be a real number")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")
        if self.source_kind is SourceKind.SYNTHETIC:
            if (
                isinstance(self.generation_round, bool)
                or not isinstance(self.generation_round, int)
                or self.generation_round <= 0
            ):
                raise ValueError("synthetic component requires a positive generation_round")
        elif self.generation_round is not None:
            raise ValueError("real component must not declare a generation_round")


@dataclass(frozen=True)
class MixtureExposure:
    name: str
    source_kind: SourceKind
    normalized_fraction: float
    expected_consumed_tokens: float
    expected_repetition_factor: float
    generation_round: int | None


@dataclass(frozen=True)
class MixturePlan:
    total_consumed_tokens: int
    synthetic_fraction: float
    exposures: tuple[MixtureExposure, ...]


def plan_mixture(
    components: Iterable[MixtureComponent], *, total_consumed_tokens: int
) -> MixturePlan:
    """Normalize component weights and expose expected token repetition.

    This is an expectation under the target sampler. It does not model packing,
    filtering after sampling, distributed sampler skew, or observed token counts.
    """

    materialized = tuple(components)
    if not materialized:
        raise ValueError("components must not be empty")
    if (
        isinstance(total_consumed_tokens, bool)
        or not isinstance(total_consumed_tokens, int)
        or total_consumed_tokens <= 0
    ):
        raise ValueError("total_consumed_tokens must be a positive integer")
    names = [component.name for component in materialized]
    if len(names) != len(set(names)):
        raise ValueError("component names must be unique")
    total_weight = sum(float(component.weight) for component in materialized)
    exposures: list[MixtureExposure] = []
    for component in materialized:
        fraction = float(component.weight) / total_weight
        expected_tokens = total_consumed_tokens * fraction
        exposures.append(
            MixtureExposure(
                name=component.name,
                source_kind=component.source_kind,
                normalized_fraction=fraction,
                expected_consumed_tokens=expected_tokens,
                expected_repetition_factor=expected_tokens / component.unique_tokens,
                generation_round=component.generation_round,
            )
        )
    synthetic_fraction = sum(
        exposure.normalized_fraction
        for exposure in exposures
        if exposure.source_kind is SourceKind.SYNTHETIC
    )
    return MixturePlan(
        total_consumed_tokens=total_consumed_tokens,
        synthetic_fraction=synthetic_fraction,
        exposures=tuple(exposures),
    )


__all__ = [
    "FingerprintProfile",
    "GenerationRoundSummary",
    "MixtureComponent",
    "MixtureExposure",
    "MixturePlan",
    "SourceKind",
    "SyntheticAuditReport",
    "SyntheticRecord",
    "VerificationResult",
    "audit_synthetic_records",
    "content_fingerprint",
    "plan_mixture",
]
