"""Transparent lexical near-duplicate candidates for preference split auditing."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
from typing import Any

from about_llm.finetuning.near_duplicate import (
    NearDuplicateProfile,
    character_ngrams,
    normalize_near_duplicate_text,
    shingle_jaccard,
)
from about_llm.finetuning.preference_data import PreferenceRecord
from about_llm.llmops import artifact_fingerprint

PREFERENCE_NEAR_DUPLICATE_AUDIT_VERSION = (
    "about-llm.preference-char-ngram-near-duplicate.v1"
)


class PreferenceNearDuplicateView(str, Enum):
    PROMPT = "prompt"
    CANDIDATE_CROSS_SURFACE = "candidate_cross_surface"


@dataclass(frozen=True)
class PreferenceNearDuplicateFinding:
    left_record_id: str
    right_record_id: str
    left_split: str
    right_split: str
    view: PreferenceNearDuplicateView
    left_surface: str
    right_surface: str
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
            "left_surface": self.left_surface,
            "right_surface": self.right_surface,
            "similarity": self.similarity,
            "intersection_size": self.intersection_size,
            "union_size": self.union_size,
            "left_shingle_count": self.left_shingle_count,
            "right_shingle_count": self.right_shingle_count,
        }


@dataclass(frozen=True)
class PreferenceNearDuplicateAuditReport:
    record_count: int
    record_pair_count: int
    comparison_count: int
    profile: NearDuplicateProfile
    ngram_size: int
    threshold: float
    cross_split_only: bool
    views: tuple[PreferenceNearDuplicateView, ...]
    findings: tuple[PreferenceNearDuplicateFinding, ...]
    ordered_dataset_fingerprint: str

    @property
    def gate_passed(self) -> bool:
        return not self.findings

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "audit_version": PREFERENCE_NEAR_DUPLICATE_AUDIT_VERSION,
                "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
                "profile": self.profile.value,
                "ngram_size": self.ngram_size,
                "threshold": self.threshold,
                "cross_split_only": self.cross_split_only,
                "views": [view.value for view in self.views],
                "findings": [finding.to_dict() for finding in self.findings],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": PREFERENCE_NEAR_DUPLICATE_AUDIT_VERSION,
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
                "prompt_to_prompt_compared": (
                    PreferenceNearDuplicateView.PROMPT in self.views
                ),
                "all_four_cross_record_candidate_surfaces_compared": (
                    PreferenceNearDuplicateView.CANDIDATE_CROSS_SURFACE in self.views
                ),
                "prompt_to_candidate_compared": False,
                "approximate_minhash_or_lsh": False,
                "semantic_equivalence_verified": False,
                "normalization_preserves_task_meaning": False,
                "threshold_calibrated_for_caller_domain": False,
                "scalable_all_pairs_implementation": False,
            },
        }


def audit_preference_near_duplicates(
    records: Iterable[PreferenceRecord],
    *,
    profile: NearDuplicateProfile,
    ngram_size: int,
    threshold: float,
    views: Iterable[PreferenceNearDuplicateView] = (
        PreferenceNearDuplicateView.PROMPT,
        PreferenceNearDuplicateView.CANDIDATE_CROSS_SURFACE,
    ),
    cross_split_only: bool = True,
) -> PreferenceNearDuplicateAuditReport:
    snapshot = tuple(records)
    if not snapshot or any(not isinstance(item, PreferenceRecord) for item in snapshot):
        raise ValueError("preference near audit requires PreferenceRecord values")
    ids = tuple(record.record_id for record in snapshot)
    if len(ids) != len(set(ids)):
        raise ValueError("preference near audit requires unique record ids")
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
        or any(not isinstance(view, PreferenceNearDuplicateView) for view in selected_views)
        or len(selected_views) != len(set(selected_views))
    ):
        raise ValueError("views must contain unique PreferenceNearDuplicateView values")

    shingles = {
        (record.record_id, surface): character_ngrams(
            normalize_near_duplicate_text(text, profile=profile), size=ngram_size
        )
        for record in snapshot
        for surface, text in _record_surfaces(record).items()
    }
    findings: list[PreferenceNearDuplicateFinding] = []
    pair_count = 0
    comparison_count = 0
    for left, right in combinations(snapshot, 2):
        if cross_split_only and left.split is right.split:
            continue
        pair_count += 1
        comparisons: list[tuple[PreferenceNearDuplicateView, str, str]] = []
        if PreferenceNearDuplicateView.PROMPT in selected_views:
            comparisons.append((PreferenceNearDuplicateView.PROMPT, "prompt", "prompt"))
        if PreferenceNearDuplicateView.CANDIDATE_CROSS_SURFACE in selected_views:
            comparisons.extend(
                (
                    PreferenceNearDuplicateView.CANDIDATE_CROSS_SURFACE,
                    left_surface,
                    right_surface,
                )
                for left_surface, right_surface in product(
                    ("candidate_a", "candidate_b"), repeat=2
                )
            )
        for view, left_surface, right_surface in comparisons:
            comparison_count += 1
            left_shingles = shingles[(left.record_id, left_surface)]
            right_shingles = shingles[(right.record_id, right_surface)]
            similarity, intersection, union = shingle_jaccard(
                left_shingles, right_shingles
            )
            if similarity >= float(threshold):
                findings.append(
                    PreferenceNearDuplicateFinding(
                        left.record_id,
                        right.record_id,
                        left.split.value,
                        right.split.value,
                        view,
                        left_surface,
                        right_surface,
                        similarity,
                        intersection,
                        union,
                        len(left_shingles),
                        len(right_shingles),
                    )
                )
    findings.sort(
        key=lambda item: (
            item.left_record_id,
            item.right_record_id,
            item.view.value,
            item.left_surface,
            item.right_surface,
        )
    )
    ordered_dataset_fingerprint = "sha256:" + artifact_fingerprint(
        {
            "ordered_record_fingerprints": [
                record.record_fingerprint for record in snapshot
            ]
        }
    )
    return PreferenceNearDuplicateAuditReport(
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


def _record_surfaces(record: PreferenceRecord) -> dict[str, str]:
    return {
        "prompt": "\n".join(
            f"<{message.role.value}>\n{message.content}" for message in record.prompt
        ),
        "candidate_a": record.candidate_a,
        "candidate_b": record.candidate_b,
    }
