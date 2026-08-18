from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.finetuning import (
    DataSplit,
    NearDuplicateProfile,
    PreferenceNearDuplicateView,
    audit_preference_near_duplicates,
    load_preference_records,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "projects" / "single-gpu-finetuning" / "preference.example.jsonl"
pytestmark = [pytest.mark.formula, pytest.mark.contract]


def test_preference_fixture_has_explicit_cross_surface_comparison_denominator() -> None:
    records = load_preference_records(FIXTURE)

    report = audit_preference_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=0.9,
    )

    assert report.gate_passed
    assert report.record_pair_count == 5
    assert report.comparison_count == 25
    assert report.to_dict()["scope"][
        "all_four_cross_record_candidate_surfaces_compared"
    ] is True
    assert report.to_dict()["scope"]["prompt_to_candidate_compared"] is False


def test_preference_near_audit_detects_prompt_and_swapped_candidate_surface() -> None:
    records = load_preference_records(FIXTURE)
    left = records[0]
    right = replace(
        records[3],
        prompt=left.prompt,
        candidate_b=left.candidate_a,
        candidate_a="unrelated response",
    )

    report = audit_preference_near_duplicates(
        (left, right),
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=1,
    )

    assert not report.gate_passed
    assert {
        (item.view, item.left_surface, item.right_surface)
        for item in report.findings
    } == {
        (PreferenceNearDuplicateView.PROMPT, "prompt", "prompt"),
        (
            PreferenceNearDuplicateView.CANDIDATE_CROSS_SURFACE,
            "candidate_a",
            "candidate_b",
        ),
    }


def test_preference_near_audit_same_split_policy_and_invalid_threshold() -> None:
    records = load_preference_records(FIXTURE)
    same_split = replace(records[3], split=DataSplit.TRAIN)
    cross_only = audit_preference_near_duplicates(
        (records[0], same_split),
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=1,
    )
    all_pairs = audit_preference_near_duplicates(
        (records[0], same_split),
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=1,
        cross_split_only=False,
    )

    assert cross_only.record_pair_count == 0
    assert cross_only.comparison_count == 0
    assert all_pairs.record_pair_count == 1
    assert all_pairs.comparison_count == 5
    with pytest.raises(ValueError, match="threshold"):
        audit_preference_near_duplicates(
            (records[0],),
            profile=NearDuplicateProfile.NFC_WHITESPACE,
            ngram_size=3,
            threshold=math.nan,
        )
