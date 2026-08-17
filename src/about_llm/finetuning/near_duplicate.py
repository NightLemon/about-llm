"""Transparent lexical near-duplicate candidates for SFT split auditing."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

from about_llm.finetuning.data import (
    MessageRole,
    SFTRecord,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

SFT_NEAR_DUPLICATE_AUDIT_VERSION = "about-llm.char-ngram-near-duplicate.v2"


class NearDuplicateProfile(str, Enum):
    """Explicit lossy normalization choices; neither preserves every task."""

    NFC_WHITESPACE = "nfc_whitespace"
    NFKC_CASEFOLD_WHITESPACE = "nfkc_casefold_whitespace"


class NearDuplicateView(str, Enum):
    """Which supervised conversation surface supplies character shingles."""

    FULL_CONVERSATION = "full_conversation"
    USER_CONTENT = "user_content"
    ASSISTANT_CONTENT = "assistant_content"


@dataclass(frozen=True)
class NearDuplicateFinding:
    left_record_id: str
    right_record_id: str
    left_split: str
    right_split: str
    view: NearDuplicateView
    similarity: float
    intersection_size: int
    union_size: int
    left_shingle_count: int
    right_shingle_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_record_id": self.left_record_id,
            "right_record_id": self.right_record_id,
            "left_split": self.left_split,
            "right_split": self.right_split,
            "view": self.view.value,
            "similarity": self.similarity,
            "intersection_size": self.intersection_size,
            "union_size": self.union_size,
            "left_shingle_count": self.left_shingle_count,
            "right_shingle_count": self.right_shingle_count,
        }


@dataclass(frozen=True)
class SFTNearDuplicateAuditReport:
    record_count: int
    record_pair_count: int
    comparison_count: int
    profile: NearDuplicateProfile
    ngram_size: int
    threshold: float
    cross_split_only: bool
    views: tuple[NearDuplicateView, ...]
    findings: tuple[NearDuplicateFinding, ...]
    ordered_dataset_fingerprint: str

    @property
    def gate_passed(self) -> bool:
        return not self.findings

    @property
    def manifest_fingerprint(self) -> str:
        identity = {
            "audit_version": SFT_NEAR_DUPLICATE_AUDIT_VERSION,
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "profile": self.profile.value,
            "ngram_size": self.ngram_size,
            "threshold": self.threshold,
            "cross_split_only": self.cross_split_only,
            "views": [view.value for view in self.views],
            "findings": [finding.to_dict() for finding in self.findings],
        }
        return "sha256:" + artifact_fingerprint(identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": SFT_NEAR_DUPLICATE_AUDIT_VERSION,
            "gate_passed": self.gate_passed,
            "record_count": self.record_count,
            "record_pair_count": self.record_pair_count,
            "comparison_count": self.comparison_count,
            "profile": self.profile.value,
            "ngram_size": self.ngram_size,
            "threshold": self.threshold,
            "cross_split_only": self.cross_split_only,
            "views": [view.value for view in self.views],
            "findings": [finding.to_dict() for finding in self.findings],
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "scope": {
                "exact_jaccard_over_character_ngram_sets": True,
                "approximate_minhash_or_lsh": False,
                "semantic_equivalence_verified": False,
                "normalization_preserves_task_meaning": False,
                "threshold_calibrated_for_caller_domain": False,
                "scalable_all_pairs_implementation": False,
            },
        }


def normalize_near_duplicate_text(text: str, *, profile: NearDuplicateProfile) -> str:
    """Normalize for candidate generation, never as a training-text rewrite."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("near-duplicate text must be a non-empty string")
    if not isinstance(profile, NearDuplicateProfile):
        raise ValueError("profile must be NearDuplicateProfile")
    if profile is NearDuplicateProfile.NFC_WHITESPACE:
        normalized = unicodedata.normalize("NFC", text)
    else:
        normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def character_ngrams(text: str, *, size: int) -> frozenset[str]:
    """Return a Unicode-codepoint shingle set with an explicit short-text rule."""

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("character n-gram size must be a positive integer")
    if not isinstance(text, str) or not text:
        raise ValueError("character n-gram text must be non-empty")
    if len(text) <= size:
        return frozenset((text,))
    return frozenset(text[index : index + size] for index in range(len(text) - size + 1))


def shingle_jaccard(left: frozenset[str], right: frozenset[str]) -> tuple[float, int, int]:
    """Return set Jaccard plus its auditable numerator and denominator."""

    if not left or not right:
        raise ValueError("shingle sets must be non-empty")
    intersection_size = len(left & right)
    union_size = len(left | right)
    return intersection_size / union_size, intersection_size, union_size


def audit_sft_near_duplicates(
    records: Iterable[SFTRecord],
    *,
    profile: NearDuplicateProfile,
    ngram_size: int,
    threshold: float,
    views: Iterable[NearDuplicateView] = (
        NearDuplicateView.FULL_CONVERSATION,
        NearDuplicateView.USER_CONTENT,
        NearDuplicateView.ASSISTANT_CONTENT,
    ),
    cross_split_only: bool = True,
) -> SFTNearDuplicateAuditReport:
    """Enumerate lexical candidates exactly; do not claim semantic duplicates."""

    snapshot = tuple(records)
    if not snapshot or any(not isinstance(record, SFTRecord) for record in snapshot):
        raise ValueError("near-duplicate audit requires SFTRecord values")
    ids = tuple(record.record_id for record in snapshot)
    if len(ids) != len(set(ids)):
        raise ValueError("near-duplicate audit requires unique record ids")
    if not isinstance(profile, NearDuplicateProfile):
        raise ValueError("profile must be NearDuplicateProfile")
    if isinstance(ngram_size, bool) or not isinstance(ngram_size, int) or ngram_size <= 0:
        raise ValueError("ngram_size must be a positive integer")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 < float(threshold) <= 1
    ):
        raise ValueError("threshold must be finite and in (0, 1]")
    if not isinstance(cross_split_only, bool):
        raise ValueError("cross_split_only must be a boolean")
    selected_views = tuple(views)
    if (
        not selected_views
        or any(not isinstance(view, NearDuplicateView) for view in selected_views)
        or len(selected_views) != len(set(selected_views))
    ):
        raise ValueError("views must contain unique NearDuplicateView values")

    normalized_shingles = {
        (record.record_id, view): character_ngrams(
            normalize_near_duplicate_text(
                _record_view_text(record, view), profile=profile
            ),
            size=ngram_size,
        )
        for record in snapshot
        for view in selected_views
    }
    findings: list[NearDuplicateFinding] = []
    pair_count = 0
    comparison_count = 0
    for left, right in combinations(snapshot, 2):
        if cross_split_only and left.split is right.split:
            continue
        pair_count += 1
        for view in selected_views:
            comparison_count += 1
            left_shingles = normalized_shingles[(left.record_id, view)]
            right_shingles = normalized_shingles[(right.record_id, view)]
            similarity, intersection_size, union_size = shingle_jaccard(
                left_shingles, right_shingles
            )
            if similarity >= float(threshold):
                findings.append(
                    NearDuplicateFinding(
                        left.record_id,
                        right.record_id,
                        left.split.value,
                        right.split.value,
                        view,
                        similarity,
                        intersection_size,
                        union_size,
                        len(left_shingles),
                        len(right_shingles),
                    )
                )
    findings.sort(key=lambda item: (item.left_record_id, item.right_record_id, item.view.value))
    ordered_dataset_fingerprint = "sha256:" + artifact_fingerprint(
        {
            "ordered_record_fingerprints": [
                record.record_fingerprint for record in snapshot
            ]
        }
    )
    return SFTNearDuplicateAuditReport(
        record_count=len(snapshot),
        record_pair_count=pair_count,
        comparison_count=comparison_count,
        profile=profile,
        ngram_size=ngram_size,
        threshold=float(threshold),
        cross_split_only=cross_split_only,
        views=selected_views,
        findings=tuple(findings),
        ordered_dataset_fingerprint=ordered_dataset_fingerprint,
    )


def _record_view_text(record: SFTRecord, view: NearDuplicateView) -> str:
    if view is NearDuplicateView.FULL_CONVERSATION:
        messages = record.messages
    elif view is NearDuplicateView.USER_CONTENT:
        messages = tuple(
            message for message in record.messages if message.role is MessageRole.USER
        )
    else:
        messages = tuple(
            message for message in record.messages if message.role is MessageRole.ASSISTANT
        )
    fragments: list[str] = []
    if view is NearDuplicateView.FULL_CONVERSATION and record.tools:
        fragments.append(
            "<tools>\n"
            + canonical_json_bytes([tool.to_dict() for tool in record.tools]).decode(
                "utf-8"
            )
        )
    for message in messages:
        fragment = f"<{message.role.value}>\n{message.content}"
        if message.tool_calls:
            fragment += "\n<tool_calls>\n" + canonical_json_bytes(
                [call.to_dict() for call in message.tool_calls]
            ).decode("utf-8")
        fragments.append(fragment)
    return "\n".join(fragments)
