"""Target-tokenizer preflight for conversational DPO rows."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from about_llm.finetuning.preference_data import (
    PREFERENCE_DATA_CONTRACT_VERSION,
    PreferenceRecord,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

PREFERENCE_TOKENIZATION_AUDIT_VERSION = (
    "about-llm.preference-tokenization-audit.v1"
)
PreferenceTemplateRenderer = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class PreferenceTokenizationSample:
    record_id: str
    prompt_token_count: int
    chosen_completion_token_count: int
    rejected_completion_token_count: int
    tokenization_fingerprint: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "record_id": self.record_id,
            "prompt_token_count": self.prompt_token_count,
            "chosen_completion_token_count": self.chosen_completion_token_count,
            "rejected_completion_token_count": self.rejected_completion_token_count,
            "tokenization_fingerprint": self.tokenization_fingerprint,
        }


@dataclass(frozen=True)
class PreferenceTokenizationAuditReport:
    max_length: int
    ordered_dataset_fingerprint: str
    renderer_identity_json: str = field(repr=False)
    renderer_fingerprint: str
    samples: tuple[PreferenceTokenizationSample, ...]

    @property
    def record_count(self) -> int:
        return len(self.samples)

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "audit_version": PREFERENCE_TOKENIZATION_AUDIT_VERSION,
                "data_contract_version": PREFERENCE_DATA_CONTRACT_VERSION,
                "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
                "renderer_fingerprint": self.renderer_fingerprint,
                "max_length": self.max_length,
                "ordered_tokenization_fingerprints": [
                    sample.tokenization_fingerprint for sample in self.samples
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": PREFERENCE_TOKENIZATION_AUDIT_VERSION,
            "data_contract_version": PREFERENCE_DATA_CONTRACT_VERSION,
            "gate_passed": True,
            "record_count": self.record_count,
            "max_length": self.max_length,
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "renderer_identity": json.loads(self.renderer_identity_json),
            "renderer_fingerprint": self.renderer_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "samples": [sample.to_dict() for sample in self.samples],
            "scope": {
                "target_tokenizer_executed": True,
                "trl_0_29_conversational_tokenization_shape_reproduced": True,
                "prompt_prefix_match_required": True,
                "empty_or_token_identical_completions_rejected": True,
                "max_length_truncation_rejected": True,
                "trainer_collator_or_model_forward_executed": False,
                "template_semantics_independently_verified": False,
            },
        }


def audit_preference_tokenization(
    records: Iterable[PreferenceRecord],
    *,
    render: PreferenceTemplateRenderer,
    renderer_identity: Mapping[str, Any],
    max_length: int,
) -> PreferenceTokenizationAuditReport:
    """Fail closed on prefix mismatch or truncation before model weights load.

    The renderer must reproduce TRL 0.29's conversational path: tokenize the
    prompt with ``add_generation_prompt=True`` and tokenize prompt+chosen and
    prompt+rejected as complete conversations. TRL warns on prefix mismatch and
    slices anyway; this preflight intentionally promotes that condition to an
    error because it can change which tokens receive preference loss.
    """

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    snapshot = tuple(records)
    if not snapshot or any(not isinstance(item, PreferenceRecord) for item in snapshot):
        raise ValueError("preference tokenization audit requires PreferenceRecord values")
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

    samples: list[PreferenceTokenizationSample] = []
    for record in snapshot:
        try:
            rendered = render(record.to_dpo_row())
        except Exception as error:
            raise ValueError(
                f"record {record.record_id!r}: chat template execution failed: {error}"
            ) from error
        if not isinstance(rendered, Mapping):
            raise ValueError(
                f"record {record.record_id!r}: renderer must return a mapping"
            )
        prompt_ids = _integer_sequence(
            rendered.get("prompt_ids"), "prompt_ids", record.record_id
        )
        prompt_chosen_ids = _integer_sequence(
            rendered.get("prompt_chosen_ids"),
            "prompt_chosen_ids",
            record.record_id,
        )
        prompt_rejected_ids = _integer_sequence(
            rendered.get("prompt_rejected_ids"),
            "prompt_rejected_ids",
            record.record_id,
        )
        for name, full_ids in (
            ("chosen", prompt_chosen_ids),
            ("rejected", prompt_rejected_ids),
        ):
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    f"record {record.record_id!r}: tokenized prompt is not an exact "
                    f"prefix of prompt+{name}; TRL slicing would be ambiguous"
                )
            if len(full_ids) > max_length:
                raise ValueError(
                    f"record {record.record_id!r}: prompt+{name} token length "
                    f"{len(full_ids)} exceeds max_length {max_length}; preprocess "
                    "explicitly instead of relying on trainer truncation or filtering"
                )
        chosen_ids = prompt_chosen_ids[len(prompt_ids) :]
        rejected_ids = prompt_rejected_ids[len(prompt_ids) :]
        if not chosen_ids or not rejected_ids:
            raise ValueError(
                f"record {record.record_id!r}: chosen and rejected completions must "
                "both contain tokens after prompt slicing"
            )
        if chosen_ids == rejected_ids:
            raise ValueError(
                f"record {record.record_id!r}: chosen and rejected tokenize to the "
                "same completion ids"
            )
        samples.append(
            PreferenceTokenizationSample(
                record_id=record.record_id,
                prompt_token_count=len(prompt_ids),
                chosen_completion_token_count=len(chosen_ids),
                rejected_completion_token_count=len(rejected_ids),
                tokenization_fingerprint="sha256:"
                + artifact_fingerprint(
                    {
                        "prompt_ids": list(prompt_ids),
                        "chosen_ids": list(chosen_ids),
                        "rejected_ids": list(rejected_ids),
                    }
                ),
            )
        )
    return PreferenceTokenizationAuditReport(
        max_length=max_length,
        ordered_dataset_fingerprint=ordered_dataset_fingerprint,
        renderer_identity_json=renderer_identity_bytes.decode("utf-8"),
        renderer_fingerprint=renderer_fingerprint,
        samples=tuple(samples),
    )


def _integer_sequence(value: Any, field: str, record_id: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"record {record_id!r}: {field} must be an integer sequence")
    items = tuple(value)
    if not items:
        raise ValueError(f"record {record_id!r}: {field} must not be empty")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in items
    ):
        raise ValueError(
            f"record {record_id!r}: {field} must contain non-negative integers"
        )
    return cast(tuple[int, ...], items)
