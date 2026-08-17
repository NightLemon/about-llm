"""Model-independent validation of tokenizer-reported assistant masks."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from about_llm.finetuning.data import SFT_DATA_CONTRACT_VERSION, SFTRecord
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

ASSISTANT_MASK_AUDIT_VERSION = "about-llm.assistant-mask-audit.v2"
ASSISTANT_LABEL_AUDIT_VERSION = "about-llm.assistant-label-audit.v1"

ChatTemplateRenderer = Callable[
    [list[dict[str, Any]], list[dict[str, Any]] | None], Mapping[str, Any]
]


@dataclass(frozen=True)
class AssistantMaskSample:
    record_id: str
    input_token_count: int
    assistant_token_count: int
    token_mask_fingerprint: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "record_id": self.record_id,
            "input_token_count": self.input_token_count,
            "assistant_token_count": self.assistant_token_count,
            "token_mask_fingerprint": self.token_mask_fingerprint,
        }


@dataclass(frozen=True)
class AssistantMaskAuditReport:
    max_length: int
    ordered_dataset_fingerprint: str
    renderer_identity_json: str = field(repr=False)
    renderer_fingerprint: str
    samples: tuple[AssistantMaskSample, ...]

    @property
    def record_count(self) -> int:
        return len(self.samples)

    @property
    def input_token_count(self) -> int:
        return sum(sample.input_token_count for sample in self.samples)

    @property
    def assistant_token_count(self) -> int:
        return sum(sample.assistant_token_count for sample in self.samples)

    @property
    def manifest_fingerprint(self) -> str:
        identity = {
            "audit_version": ASSISTANT_MASK_AUDIT_VERSION,
            "data_contract_version": SFT_DATA_CONTRACT_VERSION,
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "renderer_fingerprint": self.renderer_fingerprint,
            "max_length": self.max_length,
            "ordered_token_mask_fingerprints": [
                sample.token_mask_fingerprint for sample in self.samples
            ],
        }
        return "sha256:" + artifact_fingerprint(identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": ASSISTANT_MASK_AUDIT_VERSION,
            "data_contract_version": SFT_DATA_CONTRACT_VERSION,
            "gate_passed": True,
            "record_count": self.record_count,
            "max_length": self.max_length,
            "input_token_count": self.input_token_count,
            "assistant_token_count": self.assistant_token_count,
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "renderer_identity": json.loads(self.renderer_identity_json),
            "renderer_fingerprint": self.renderer_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "samples": [sample.to_dict() for sample in self.samples],
            "scope": {
                "target_tokenizer_executed": True,
                "record_tools_forwarded_to_chat_template": True,
                "tokenizer_reported_assistant_mask_checked": True,
                "right_truncation_rejected": True,
                "collator_labels_verified": False,
                "mask_semantics_independently_verified": False,
            },
        }


@dataclass(frozen=True)
class AssistantMaskFeature:
    record_id: str
    input_ids: tuple[int, ...]
    assistant_masks: tuple[int, ...]

    def to_training_row(self) -> dict[str, list[int]]:
        return {
            "input_ids": list(self.input_ids),
            "assistant_masks": list(self.assistant_masks),
        }


@dataclass(frozen=True)
class AssistantMaskPreparation:
    audit_report: AssistantMaskAuditReport
    features: tuple[AssistantMaskFeature, ...]

    def to_training_rows(self) -> list[dict[str, list[int]]]:
        return [feature.to_training_row() for feature in self.features]


@dataclass(frozen=True)
class AssistantLabelAuditReport:
    mask_manifest_fingerprint: str
    batch_shape: tuple[int, int]
    attention_token_count: int
    padding_token_count: int
    supervised_label_count: int
    ignored_label_count: int
    input_ids_fingerprint: str
    attention_mask_fingerprint: str
    labels_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": ASSISTANT_LABEL_AUDIT_VERSION,
            "gate_passed": True,
            "mask_manifest_fingerprint": self.mask_manifest_fingerprint,
            "batch_shape": list(self.batch_shape),
            "attention_token_count": self.attention_token_count,
            "padding_token_count": self.padding_token_count,
            "supervised_label_count": self.supervised_label_count,
            "ignored_label_count": self.ignored_label_count,
            "input_ids_fingerprint": self.input_ids_fingerprint,
            "attention_mask_fingerprint": self.attention_mask_fingerprint,
            "labels_fingerprint": self.labels_fingerprint,
            "scope": {
                "prepared_features_exactly_rechecked": True,
                "real_configured_collator_executed": True,
                "assistant_labels_equal_input_ids": True,
                "non_assistant_and_padding_labels_are_minus_100": True,
                "model_forward_or_optimizer_executed": False,
            },
        }


def audit_assistant_masks(
    records: Iterable[SFTRecord],
    *,
    render: ChatTemplateRenderer,
    renderer_identity: Mapping[str, Any],
    max_length: int,
) -> AssistantMaskAuditReport:
    """Fail closed when a target template omits/misaligns masks or would truncate.

    ``render`` must tokenize one conversation and request the tokenizer's
    assistant mask. This checks the returned structure, not whether the
    template author marked the semantically correct spans.
    """

    return prepare_assistant_mask_features(
        records,
        render=render,
        renderer_identity=renderer_identity,
        max_length=max_length,
    ).audit_report


def prepare_assistant_mask_features(
    records: Iterable[SFTRecord],
    *,
    render: ChatTemplateRenderer,
    renderer_identity: Mapping[str, Any],
    max_length: int,
) -> AssistantMaskPreparation:
    """Render in Python before Arrow can normalize nested tool arguments."""

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    snapshot = tuple(records)
    if not snapshot:
        raise ValueError("assistant mask audit requires at least one record")
    if any(not isinstance(record, SFTRecord) for record in snapshot):
        raise ValueError("assistant mask audit accepts only SFTRecord values")
    if not callable(render):
        raise ValueError("render must be callable")
    try:
        renderer_identity_bytes = canonical_json_bytes(renderer_identity)
        renderer_snapshot = json.loads(renderer_identity_bytes)
    except (TypeError, ValueError) as error:
        raise ValueError(f"renderer_identity must be strict JSON: {error}") from error
    if not isinstance(renderer_snapshot, dict) or not renderer_snapshot:
        raise ValueError("renderer_identity must be a non-empty JSON object")
    typed_renderer_snapshot = cast(dict[str, Any], renderer_snapshot)
    renderer_fingerprint = "sha256:" + artifact_fingerprint(typed_renderer_snapshot)

    ordered_dataset_fingerprint = "sha256:" + artifact_fingerprint(
        {
            "ordered_record_fingerprints": [
                record.record_fingerprint for record in snapshot
            ]
        }
    )

    samples: list[AssistantMaskSample] = []
    features: list[AssistantMaskFeature] = []
    for record in snapshot:
        messages = [message.to_dict() for message in record.messages]
        tools = [tool.to_dict() for tool in record.tools] or None
        try:
            rendered = render(messages, tools)
        except Exception as error:
            raise ValueError(
                f"record {record.record_id!r}: chat template execution failed: {error}"
            ) from error
        if not isinstance(rendered, Mapping):
            raise ValueError(
                f"record {record.record_id!r}: chat template must return a mapping"
            )
        input_ids = _integer_sequence(
            rendered.get("input_ids"),
            field="input_ids",
            record_id=record.record_id,
            binary=False,
        )
        assistant_masks = _integer_sequence(
            rendered.get("assistant_masks"),
            field="assistant_masks",
            record_id=record.record_id,
            binary=True,
        )
        if len(input_ids) != len(assistant_masks):
            raise ValueError(
                f"record {record.record_id!r}: input_ids and assistant_masks "
                "must have equal lengths"
            )
        if not any(assistant_masks):
            raise ValueError(
                f"record {record.record_id!r}: tokenizer reported no assistant tokens; "
                "the template may lack generation markers"
            )
        if len(input_ids) > max_length:
            raise ValueError(
                f"record {record.record_id!r}: token length {len(input_ids)} exceeds "
                f"max_length {max_length}; explicit preprocessing is required instead "
                "of silent right truncation"
            )
        samples.append(
            AssistantMaskSample(
                record.record_id,
                input_token_count=len(input_ids),
                assistant_token_count=sum(assistant_masks),
                token_mask_fingerprint="sha256:"
                + artifact_fingerprint(
                    {
                        "input_ids": list(input_ids),
                        "assistant_masks": list(assistant_masks),
                    }
                ),
            )
        )
        features.append(
            AssistantMaskFeature(
                record_id=record.record_id,
                input_ids=input_ids,
                assistant_masks=assistant_masks,
            )
        )
    return AssistantMaskPreparation(
        audit_report=AssistantMaskAuditReport(
            max_length=max_length,
            ordered_dataset_fingerprint=ordered_dataset_fingerprint,
            renderer_identity_json=renderer_identity_bytes.decode("utf-8"),
            renderer_fingerprint=renderer_fingerprint,
            samples=tuple(samples),
        ),
        features=tuple(features),
    )


def audit_assistant_label_projection(
    preparation: AssistantMaskPreparation,
    *,
    prepared_features: Iterable[Mapping[str, Any]],
    collate: Callable[[list[dict[str, Any]]], Mapping[str, Any]],
    pad_token_id: int,
    label_pad_token_id: int = -100,
) -> AssistantLabelAuditReport:
    """Execute the configured collator and verify the final assistant-only labels."""

    if not isinstance(preparation, AssistantMaskPreparation):
        raise ValueError("preparation must be AssistantMaskPreparation")
    if not callable(collate):
        raise ValueError("collate must be callable")
    if (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or pad_token_id < 0
    ):
        raise ValueError("pad_token_id must be a non-negative integer")
    if (
        isinstance(label_pad_token_id, bool)
        or not isinstance(label_pad_token_id, int)
        or label_pad_token_id >= 0
    ):
        raise ValueError("label_pad_token_id must be a negative integer")

    rows = [dict(feature) for feature in prepared_features]
    if len(rows) != len(preparation.features):
        raise ValueError("prepared feature count drifted from mask preparation")
    for index, (row, expected) in enumerate(
        zip(rows, preparation.features, strict=True)
    ):
        input_ids = _integer_sequence(
            row.get("input_ids"),
            field="input_ids",
            record_id=expected.record_id,
            binary=False,
        )
        assistant_masks = _integer_sequence(
            row.get("assistant_masks"),
            field="assistant_masks",
            record_id=expected.record_id,
            binary=True,
        )
        if input_ids != expected.input_ids or assistant_masks != expected.assistant_masks:
            raise ValueError(f"prepared feature {index} drifted from mask preparation")

    try:
        batch = collate(rows)
    except Exception as error:
        raise ValueError(f"configured collator failed: {error}") from error
    if not isinstance(batch, Mapping):
        raise ValueError("configured collator must return a mapping")
    input_rows = _integer_matrix(batch.get("input_ids"), field="input_ids")
    attention_rows = _integer_matrix(
        batch.get("attention_mask"), field="attention_mask"
    )
    label_rows = _integer_matrix(batch.get("labels"), field="labels")
    if not (
        len(input_rows)
        == len(attention_rows)
        == len(label_rows)
        == len(preparation.features)
    ):
        raise ValueError("collated batch row count drifted")
    widths = {
        len(row) for matrix in (input_rows, attention_rows, label_rows) for row in matrix
    }
    if len(widths) != 1:
        raise ValueError("collated batch matrices must have one rectangular shape")
    width = next(iter(widths))
    if width < max(len(feature.input_ids) for feature in preparation.features):
        raise ValueError("collated batch silently truncated a prepared feature")

    supervised = 0
    ignored = 0
    attention = 0
    padding = 0
    for row_index, expected in enumerate(preparation.features):
        for column in range(width):
            if column < len(expected.input_ids):
                expected_input = expected.input_ids[column]
                expected_attention = 1
                expected_label = (
                    expected_input
                    if expected.assistant_masks[column]
                    else label_pad_token_id
                )
            else:
                expected_input = pad_token_id
                expected_attention = 0
                expected_label = label_pad_token_id
            if (
                input_rows[row_index][column] != expected_input
                or attention_rows[row_index][column] != expected_attention
                or label_rows[row_index][column] != expected_label
            ):
                raise ValueError(
                    "final collator assistant-label projection drifted at "
                    f"row={row_index}, column={column}"
                )
            attention += expected_attention
            padding += expected_attention == 0
            supervised += expected_label != label_pad_token_id
            ignored += expected_label == label_pad_token_id

    return AssistantLabelAuditReport(
        mask_manifest_fingerprint=preparation.audit_report.manifest_fingerprint,
        batch_shape=(len(input_rows), width),
        attention_token_count=attention,
        padding_token_count=padding,
        supervised_label_count=supervised,
        ignored_label_count=ignored,
        input_ids_fingerprint="sha256:"
        + artifact_fingerprint({"input_ids": input_rows}),
        attention_mask_fingerprint="sha256:"
        + artifact_fingerprint({"attention_mask": attention_rows}),
        labels_fingerprint="sha256:" + artifact_fingerprint({"labels": label_rows}),
    )


def _integer_sequence(
    value: Any, *, field: str, record_id: str, binary: bool
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"record {record_id!r}: {field} must be an integer sequence")
    items = tuple(value)
    if not items:
        raise ValueError(f"record {record_id!r}: {field} must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
        raise ValueError(f"record {record_id!r}: {field} must contain only integers")
    integers = cast(tuple[int, ...], items)
    if binary and any(item not in (0, 1) for item in integers):
        raise ValueError(f"record {record_id!r}: {field} must contain only 0 or 1")
    if not binary and any(item < 0 for item in integers):
        raise ValueError(f"record {record_id!r}: {field} must contain non-negative ids")
    return integers


def _integer_matrix(value: Any, *, field: str) -> list[list[int]]:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError(f"collated {field} must be a non-empty integer matrix")
    rows: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"collated {field}[{row_index}] must be a non-empty array")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in row):
            raise ValueError(
                f"collated {field}[{row_index}] must contain only integers"
            )
        rows.append(cast(list[int], row))
    return rows
