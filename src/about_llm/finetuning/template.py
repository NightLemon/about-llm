"""Model-independent validation of tokenizer-reported assistant masks."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from about_llm.finetuning.data import SFT_DATA_CONTRACT_VERSION, SFTRecord
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

ASSISTANT_MASK_AUDIT_VERSION = "about-llm.assistant-mask-audit.v1"

ChatTemplateRenderer = Callable[[list[dict[str, str]]], Mapping[str, Any]]


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
                "tokenizer_reported_assistant_mask_checked": True,
                "right_truncation_rejected": True,
                "collator_labels_verified": False,
                "mask_semantics_independently_verified": False,
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
    for record in snapshot:
        messages = [message.to_dict() for message in record.messages]
        try:
            rendered = render(messages)
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
    return AssistantMaskAuditReport(
        max_length=max_length,
        ordered_dataset_fingerprint=ordered_dataset_fingerprint,
        renderer_identity_json=renderer_identity_bytes.decode("utf-8"),
        renderer_fingerprint=renderer_fingerprint,
        samples=tuple(samples),
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
