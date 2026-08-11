"""Strict pairwise-preference data contracts and leakage auditing."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.finetuning.data import ChatMessage, DataSplit, MessageRole
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

PREFERENCE_DATA_CONTRACT_VERSION = "about-llm.preference-jsonl.v1"


class PreferenceLabel(str, Enum):
    A = "a"
    B = "b"
    TIE = "tie"
    INVALID = "invalid"


class PreferenceStrength(str, Enum):
    SLIGHT = "slight"
    CLEAR = "clear"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class PreferenceRecord:
    record_id: str
    prompt: tuple[ChatMessage, ...]
    candidate_a: str
    candidate_b: str
    presentation_order: tuple[str, str]
    label: PreferenceLabel
    preference_strength: PreferenceStrength
    source: str
    license_id: str
    task: str
    language: str
    risk: str
    group_id: str
    split: DataSplit
    rubric_revision: str
    annotator_pool: str
    adjudication: str
    generator_revisions: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _prompt_fingerprint: str = field(init=False, repr=False)
    _pair_content_fingerprint: str = field(init=False, repr=False)
    _record_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("candidate_a", self.candidate_a),
            ("candidate_b", self.candidate_b),
            ("source", self.source),
            ("license_id", self.license_id),
            ("task", self.task),
            ("language", self.language),
            ("risk", self.risk),
            ("group_id", self.group_id),
            ("rubric_revision", self.rubric_revision),
            ("annotator_pool", self.annotator_pool),
            ("adjudication", self.adjudication),
        ):
            _require_nonempty_string(value, f"preference {name}")
        prompt = tuple(self.prompt)
        _validate_prompt(prompt)
        object.__setattr__(self, "prompt", prompt)
        order = tuple(self.presentation_order)
        if len(order) != 2 or set(order) != {"a", "b"}:
            raise ValueError("presentation_order must be exactly ['a','b'] or ['b','a']")
        object.__setattr__(self, "presentation_order", order)
        if not isinstance(self.label, PreferenceLabel):
            raise ValueError("label must be PreferenceLabel")
        if not isinstance(self.preference_strength, PreferenceStrength):
            raise ValueError("preference_strength must be PreferenceStrength")
        binary_label = self.label in (PreferenceLabel.A, PreferenceLabel.B)
        applicable_strength = self.preference_strength in (
            PreferenceStrength.SLIGHT,
            PreferenceStrength.CLEAR,
        )
        if binary_label is not applicable_strength:
            raise ValueError(
                "binary labels require slight/clear strength; tie/invalid requires "
                "not_applicable"
            )
        if not isinstance(self.split, DataSplit):
            raise ValueError("split must be DataSplit")
        generators = _strict_json_snapshot(self.generator_revisions, "generator_revisions")
        if not isinstance(generators, dict) or set(generators) != {"a", "b"}:
            raise ValueError("generator_revisions must contain exactly keys a and b")
        for key in ("a", "b"):
            _require_nonempty_string(generators[key], f"generator_revisions.{key}")
        object.__setattr__(self, "generator_revisions", _freeze_object(generators))
        metadata = _strict_json_snapshot(self.metadata, "metadata")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze_object(metadata))
        prompt_identity = {"prompt": [message.to_dict() for message in prompt]}
        object.__setattr__(
            self,
            "_prompt_fingerprint",
            "sha256:" + artifact_fingerprint(prompt_identity),
        )
        pair_identity = {
            **prompt_identity,
            "sorted_candidates": sorted((self.candidate_a, self.candidate_b)),
        }
        object.__setattr__(
            self,
            "_pair_content_fingerprint",
            "sha256:" + artifact_fingerprint(pair_identity),
        )
        object.__setattr__(
            self,
            "_record_fingerprint",
            "sha256:" + artifact_fingerprint(self.to_dict()),
        )

    @property
    def prompt_fingerprint(self) -> str:
        return self._prompt_fingerprint

    @property
    def pair_content_fingerprint(self) -> str:
        return self._pair_content_fingerprint

    @property
    def record_fingerprint(self) -> str:
        return self._record_fingerprint

    @property
    def is_binary_preference(self) -> bool:
        return self.label in (PreferenceLabel.A, PreferenceLabel.B)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "prompt": [message.to_dict() for message in self.prompt],
            "candidate_a": self.candidate_a,
            "candidate_b": self.candidate_b,
            "presentation_order": list(self.presentation_order),
            "label": self.label.value,
            "preference_strength": self.preference_strength.value,
            "source": self.source,
            "license": self.license_id,
            "task": self.task,
            "language": self.language,
            "risk": self.risk,
            "group_id": self.group_id,
            "split": self.split.value,
            "rubric_revision": self.rubric_revision,
            "annotator_pool": self.annotator_pool,
            "adjudication": self.adjudication,
            "generator_revisions": dict(self.generator_revisions),
            "metadata": _thaw_json(self.metadata),
        }

    def to_dpo_row(self) -> dict[str, Any]:
        """Convert a binary label without discarding the conversational prompt."""

        if not self.is_binary_preference:
            raise ValueError(
                f"preference record {self.record_id!r} has non-binary label "
                f"{self.label.value!r}"
            )
        chosen = self.candidate_a if self.label is PreferenceLabel.A else self.candidate_b
        rejected = self.candidate_b if self.label is PreferenceLabel.A else self.candidate_a
        return {
            "prompt": [message.to_dict() for message in self.prompt],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        }


@dataclass(frozen=True)
class PreferenceDuplicateGroup:
    identity: str
    record_ids: tuple[str, ...]
    splits: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "record_ids": list(self.record_ids),
            "splits": list(self.splits),
        }


@dataclass(frozen=True)
class PreferenceDataAuditReport:
    record_count: int
    required_splits: tuple[str, ...]
    missing_splits: tuple[str, ...]
    split_counts: Mapping[str, int]
    label_counts: Mapping[str, int]
    strength_counts: Mapping[str, int]
    preferred_display_position_counts: Mapping[str, int]
    duplicate_record_ids: tuple[str, ...]
    duplicate_pairs: tuple[PreferenceDuplicateGroup, ...]
    identical_candidate_record_ids: tuple[str, ...]
    cross_split_group_ids: tuple[PreferenceDuplicateGroup, ...]
    cross_split_prompts: tuple[PreferenceDuplicateGroup, ...]
    cross_split_pairs: tuple[PreferenceDuplicateGroup, ...]
    ordered_dataset_fingerprint: str
    unordered_dataset_fingerprint: str

    @property
    def gate_passed(self) -> bool:
        return not any(
            (
                self.missing_splits,
                self.duplicate_record_ids,
                self.duplicate_pairs,
                self.identical_candidate_record_ids,
                self.cross_split_group_ids,
                self.cross_split_prompts,
                self.cross_split_pairs,
            )
        )

    @property
    def manifest_fingerprint(self) -> str:
        return preference_audit_manifest_fingerprint(
            self.ordered_dataset_fingerprint, self.required_splits
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": PREFERENCE_DATA_CONTRACT_VERSION,
            "gate_passed": self.gate_passed,
            "record_count": self.record_count,
            "required_splits": list(self.required_splits),
            "missing_splits": list(self.missing_splits),
            "split_counts": dict(self.split_counts),
            "label_counts": dict(self.label_counts),
            "strength_counts": dict(self.strength_counts),
            "preferred_display_position_counts": dict(
                self.preferred_display_position_counts
            ),
            "duplicate_record_ids": list(self.duplicate_record_ids),
            "duplicate_pairs": [item.to_dict() for item in self.duplicate_pairs],
            "identical_candidate_record_ids": list(
                self.identical_candidate_record_ids
            ),
            "cross_split_group_ids": [
                item.to_dict() for item in self.cross_split_group_ids
            ],
            "cross_split_prompts": [
                item.to_dict() for item in self.cross_split_prompts
            ],
            "cross_split_pairs": [
                item.to_dict() for item in self.cross_split_pairs
            ],
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "unordered_dataset_fingerprint": self.unordered_dataset_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "scope": {
                "assistant_text_only_candidates": True,
                "rankings_beyond_pairwise_supported": False,
                "presentation_order_preserved": True,
                "tie_and_invalid_labels_preserved": True,
                "position_bias_estimated": False,
                "annotator_agreement_estimated": False,
                "rubric_quality_verified": False,
                "semantic_candidate_equivalence_verified": False,
                "license_legality_or_consent_verified": False,
                "tokenization_or_truncation_verified": False,
            },
        }


@dataclass(frozen=True)
class PreferenceTrainingBindingReport:
    """Bind a trainer artifact to binary train rows in a passing full audit."""

    training_report: PreferenceDataAuditReport
    split_report: PreferenceDataAuditReport
    excluded_nonbinary_train_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.training_report, PreferenceDataAuditReport):
            raise ValueError("training_report must be PreferenceDataAuditReport")
        if not isinstance(self.split_report, PreferenceDataAuditReport):
            raise ValueError("split_report must be PreferenceDataAuditReport")
        excluded = tuple(self.excluded_nonbinary_train_record_ids)
        if any(not isinstance(item, str) or not item for item in excluded):
            raise ValueError("excluded record ids must be non-empty strings")
        if len(excluded) != len(set(excluded)):
            raise ValueError("excluded record ids must be unique")
        object.__setattr__(self, "excluded_nonbinary_train_record_ids", excluded)
        if not self.training_report.gate_passed:
            raise ValueError("training_report gate must pass")
        if not self.split_report.gate_passed:
            raise ValueError("split_report gate must pass")
        if self.training_report.required_splits != (DataSplit.TRAIN.value,):
            raise ValueError("training_report must require only the train split")
        expected_splits = tuple(split.value for split in DataSplit)
        if self.split_report.required_splits != expected_splits:
            raise ValueError("split_report must require train, validation, and test")

    @property
    def binding_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "contract_version": PREFERENCE_DATA_CONTRACT_VERSION,
                "training_manifest_fingerprint": (
                    self.training_report.manifest_fingerprint
                ),
                "split_manifest_fingerprint": self.split_report.manifest_fingerprint,
                "binding_rule": "ordered_binary_train_subset_exactly_matches",
                "excluded_nonbinary_train_count": len(
                    self.excluded_nonbinary_train_record_ids
                ),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": PREFERENCE_DATA_CONTRACT_VERSION,
            "gate_passed": True,
            "binding_rule": "ordered_binary_train_subset_exactly_matches",
            "training_manifest_fingerprint": (
                self.training_report.manifest_fingerprint
            ),
            "split_manifest_fingerprint": self.split_report.manifest_fingerprint,
            "training_ordered_dataset_fingerprint": (
                self.training_report.ordered_dataset_fingerprint
            ),
            "combined_ordered_dataset_fingerprint": (
                self.split_report.ordered_dataset_fingerprint
            ),
            "binary_train_record_count": self.training_report.record_count,
            "excluded_nonbinary_train_record_ids": list(
                self.excluded_nonbinary_train_record_ids
            ),
            "binding_fingerprint": self.binding_fingerprint,
        }


def load_preference_records(path: Path) -> tuple[PreferenceRecord, ...]:
    records: list[PreferenceRecord] = []
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
        allowed = {
            "id",
            "prompt",
            "candidate_a",
            "candidate_b",
            "presentation_order",
            "label",
            "preference_strength",
            "source",
            "license",
            "task",
            "language",
            "risk",
            "group_id",
            "split",
            "rubric_revision",
            "annotator_pool",
            "adjudication",
            "generator_revisions",
            "metadata",
        }
        _expect_fields(
            item,
            required=allowed - {"metadata"},
            allowed=allowed,
            prefix=prefix,
        )
        prompt_raw = _array(item["prompt"], f"{prefix}.prompt")
        prompt = tuple(
            _parse_message(message, f"{prefix}.prompt[{index}]")
            for index, message in enumerate(prompt_raw)
        )
        order_raw = _array(
            item["presentation_order"], f"{prefix}.presentation_order"
        )
        if len(order_raw) != 2:
            raise ValueError(f"{prefix}.presentation_order must contain two values")
        try:
            label = PreferenceLabel(item["label"])
            strength = PreferenceStrength(item["preference_strength"])
            split = DataSplit(item["split"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}: unsupported enum value") from error
        try:
            records.append(
                PreferenceRecord(
                    record_id=_string(item["id"], f"{prefix}.id"),
                    prompt=prompt,
                    candidate_a=_string(
                        item["candidate_a"], f"{prefix}.candidate_a"
                    ),
                    candidate_b=_string(
                        item["candidate_b"], f"{prefix}.candidate_b"
                    ),
                    presentation_order=(
                        _string(order_raw[0], f"{prefix}.presentation_order[0]"),
                        _string(order_raw[1], f"{prefix}.presentation_order[1]"),
                    ),
                    label=label,
                    preference_strength=strength,
                    source=_string(item["source"], f"{prefix}.source"),
                    license_id=_string(item["license"], f"{prefix}.license"),
                    task=_string(item["task"], f"{prefix}.task"),
                    language=_string(item["language"], f"{prefix}.language"),
                    risk=_string(item["risk"], f"{prefix}.risk"),
                    group_id=_string(item["group_id"], f"{prefix}.group_id"),
                    split=split,
                    rubric_revision=_string(
                        item["rubric_revision"], f"{prefix}.rubric_revision"
                    ),
                    annotator_pool=_string(
                        item["annotator_pool"], f"{prefix}.annotator_pool"
                    ),
                    adjudication=_string(
                        item["adjudication"], f"{prefix}.adjudication"
                    ),
                    generator_revisions=_object(
                        item["generator_revisions"],
                        f"{prefix}.generator_revisions",
                    ),
                    metadata=item.get("metadata", {}),
                )
            )
        except ValueError as error:
            raise ValueError(f"{prefix}: {error}") from error
    if not records:
        raise ValueError(f"{path}: dataset contains no records")
    return tuple(records)


def audit_preference_records(
    records: Iterable[PreferenceRecord],
    *,
    required_splits: Iterable[DataSplit] = (
        DataSplit.TRAIN,
        DataSplit.VALIDATION,
        DataSplit.TEST,
    ),
) -> PreferenceDataAuditReport:
    snapshot = tuple(records)
    if not snapshot or any(not isinstance(item, PreferenceRecord) for item in snapshot):
        raise ValueError("preference audit requires PreferenceRecord values")
    required = tuple(required_splits)
    if (
        not required
        or any(not isinstance(split, DataSplit) for split in required)
        or len(required) != len(set(required))
    ):
        raise ValueError("required_splits must contain unique DataSplit values")
    split_counts = Counter(record.split.value for record in snapshot)
    id_counts = Counter(record.record_id for record in snapshot)
    pair_groups = _groups(snapshot, lambda record: record.pair_content_fingerprint)
    group_groups = _groups(snapshot, lambda record: record.group_id)
    prompt_groups = _groups(snapshot, lambda record: record.prompt_fingerprint)
    duplicate_pairs = tuple(
        _duplicate(identity, grouped)
        for identity, grouped in sorted(pair_groups.items())
        if len(grouped) > 1
    )
    ordered_identity = {
        "ordered_record_fingerprints": [
            record.record_fingerprint for record in snapshot
        ]
    }
    unordered_identity = {
        "sorted_record_fingerprints": sorted(
            record.record_fingerprint for record in snapshot
        )
    }
    return PreferenceDataAuditReport(
        record_count=len(snapshot),
        required_splits=tuple(split.value for split in required),
        missing_splits=tuple(
            split.value for split in required if split_counts[split.value] == 0
        ),
        split_counts=_counts(split_counts),
        label_counts=_counts(Counter(record.label.value for record in snapshot)),
        strength_counts=_counts(
            Counter(record.preference_strength.value for record in snapshot)
        ),
        preferred_display_position_counts=_counts(
            Counter(_preferred_position(record) for record in snapshot)
        ),
        duplicate_record_ids=tuple(
            record_id for record_id, count in sorted(id_counts.items()) if count > 1
        ),
        duplicate_pairs=duplicate_pairs,
        identical_candidate_record_ids=tuple(
            sorted(
                record.record_id
                for record in snapshot
                if record.candidate_a == record.candidate_b
            )
        ),
        cross_split_group_ids=_cross_split_groups(group_groups),
        cross_split_prompts=_cross_split_groups(prompt_groups),
        cross_split_pairs=tuple(
            group for group in duplicate_pairs if len(group.splits) > 1
        ),
        ordered_dataset_fingerprint="sha256:" + artifact_fingerprint(ordered_identity),
        unordered_dataset_fingerprint="sha256:"
        + artifact_fingerprint(unordered_identity),
    )


def validate_dpo_training_records(
    records: Iterable[PreferenceRecord],
) -> PreferenceDataAuditReport:
    snapshot = tuple(records)
    non_train = [
        record.record_id for record in snapshot if record.split is not DataSplit.TRAIN
    ]
    if non_train:
        raise ValueError(f"DPO training file contains non-train records: {non_train}")
    non_binary = [
        record.record_id for record in snapshot if not record.is_binary_preference
    ]
    if non_binary:
        raise ValueError(f"DPO training records need binary labels: {non_binary}")
    report = audit_preference_records(snapshot, required_splits=(DataSplit.TRAIN,))
    if not report.gate_passed:
        raise ValueError("DPO training data failed duplicate or identity gates")
    return report


def validate_dpo_training_subset(
    training_records: Iterable[PreferenceRecord],
    audited_records: Iterable[PreferenceRecord],
) -> PreferenceTrainingBindingReport:
    """Require the trainer file to equal the ordered binary-train audit subset.

    Tie and invalid annotations remain in the combined artifact for analysis, but
    are intentionally excluded from the trainer artifact instead of being coerced
    into arbitrary winners.
    """

    training = tuple(training_records)
    audited = tuple(audited_records)
    training_report = validate_dpo_training_records(training)
    split_report = audit_preference_records(audited)
    if not split_report.gate_passed:
        raise ValueError("combined preference split audit failed; inspect the report")
    expected_records = tuple(
        record
        for record in audited
        if record.split is DataSplit.TRAIN and record.is_binary_preference
    )
    excluded = tuple(
        record.record_id
        for record in audited
        if record.split is DataSplit.TRAIN and not record.is_binary_preference
    )
    actual = tuple(record.record_fingerprint for record in training)
    expected = tuple(record.record_fingerprint for record in expected_records)
    if actual != expected:
        actual_ids = tuple(record.record_id for record in training)
        expected_ids = tuple(record.record_id for record in expected_records)
        if actual_ids == expected_ids:
            changed = tuple(
                record_id
                for record_id, actual_fingerprint, expected_fingerprint in zip(
                    actual_ids, actual, expected, strict=True
                )
                if actual_fingerprint != expected_fingerprint
            )
            raise ValueError(
                "DPO training records differ from audited binary train records "
                f"for id(s): {changed}"
            )
        if set(actual_ids) == set(expected_ids):
            raise ValueError(
                "DPO training record order differs from audited binary train order"
            )
        missing = tuple(sorted(set(expected_ids) - set(actual_ids)))
        unexpected = tuple(sorted(set(actual_ids) - set(expected_ids)))
        raise ValueError(
            "DPO training membership differs from audited binary train subset: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return PreferenceTrainingBindingReport(training_report, split_report, excluded)


def preference_audit_manifest_fingerprint(
    ordered_dataset_fingerprint: str, required_splits: tuple[str, ...]
) -> str:
    """Rebuild the strict audit-policy identity from its minimal inputs."""

    return "sha256:" + artifact_fingerprint(
        {
            "contract_version": PREFERENCE_DATA_CONTRACT_VERSION,
            "ordered_dataset_fingerprint": ordered_dataset_fingerprint,
            "required_splits": list(required_splits),
            "gate_rules": [
                "required_splits_present",
                "record_ids_unique",
                "unordered_prompt_candidate_pairs_unique",
                "candidate_texts_not_identical",
                "group_id_does_not_cross_splits",
                "exact_prompt_does_not_cross_splits",
                "exact_pair_does_not_cross_splits",
            ],
        }
    )


def _validate_prompt(messages: tuple[ChatMessage, ...]) -> None:
    if not messages or any(not isinstance(item, ChatMessage) for item in messages):
        raise ValueError("prompt must contain ChatMessage values")
    system_positions = [
        index for index, message in enumerate(messages) if message.role is MessageRole.SYSTEM
    ]
    if system_positions not in ([], [0]):
        raise ValueError("system message is optional but may appear only once at index zero")
    start = 1 if system_positions else 0
    if start >= len(messages) or messages[start].role is not MessageRole.USER:
        raise ValueError("first non-system prompt message must be user")
    transitions = {
        MessageRole.SYSTEM: {MessageRole.USER},
        MessageRole.USER: {MessageRole.ASSISTANT},
        MessageRole.ASSISTANT: {MessageRole.USER, MessageRole.TOOL},
        MessageRole.TOOL: {MessageRole.TOOL, MessageRole.ASSISTANT},
    }
    for previous, current in pairwise(messages):
        if current.role not in transitions[previous.role]:
            raise ValueError(
                f"unsupported prompt role transition: "
                f"{previous.role.value}->{current.role.value}"
            )
    if messages[-1].role not in (MessageRole.USER, MessageRole.TOOL):
        raise ValueError("preference prompt must end with user or tool")


def _parse_message(value: Any, prefix: str) -> ChatMessage:
    item = _object(value, prefix)
    _expect_fields(
        item,
        required={"role", "content"},
        allowed={"role", "content"},
        prefix=prefix,
    )
    try:
        role = MessageRole(item["role"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix}: unsupported role") from error
    return ChatMessage(role, _string(item["content"], f"{prefix}.content"))


def _preferred_position(record: PreferenceRecord) -> str:
    if record.label is PreferenceLabel.TIE:
        return "tie"
    if record.label is PreferenceLabel.INVALID:
        return "invalid"
    return "first" if record.presentation_order[0] == record.label.value else "second"


def _groups(
    records: tuple[PreferenceRecord, ...], key: Callable[[PreferenceRecord], str]
) -> dict[str, list[PreferenceRecord]]:
    grouped: dict[str, list[PreferenceRecord]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    return grouped


def _cross_split_groups(
    groups: dict[str, list[PreferenceRecord]],
) -> tuple[PreferenceDuplicateGroup, ...]:
    return tuple(
        _duplicate(identity, grouped)
        for identity, grouped in sorted(groups.items())
        if len({record.split for record in grouped}) > 1
    )


def _duplicate(
    identity: str, records: list[PreferenceRecord]
) -> PreferenceDuplicateGroup:
    return PreferenceDuplicateGroup(
        identity,
        tuple(sorted(record.record_id for record in records)),
        tuple(sorted({record.split.value for record in records})),
    )


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


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
