from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from about_llm.finetuning.data import (
    ChatMessage,
    DataSplit,
    MessageRole,
    SFTRecord,
    validate_training_subset,
)
from about_llm.finetuning.governance import (
    GovernancePurpose,
    SFTGovernancePolicy,
    SourceDecision,
    SourceRule,
    audit_sft_governance,
)
from about_llm.finetuning.near_duplicate import (
    NearDuplicateProfile,
    NearDuplicateView,
    audit_sft_near_duplicates,
    character_ngrams,
    normalize_near_duplicate_text,
    shingle_jaccard,
)
from about_llm.finetuning.readiness import SFTTrainingReadinessReport

FULLWIDTH_ABC = "\uff21\uff22\uff23"


def _record(
    record_id: str,
    split: DataSplit,
    *,
    user: str,
    assistant: str = "distinct target",
) -> SFTRecord:
    return SFTRecord(
        record_id,
        (
            ChatMessage(MessageRole.USER, user),
            ChatMessage(MessageRole.ASSISTANT, assistant),
        ),
        "unit-test",
        "test-only",
        "near-duplicate",
        "en",
        "normal",
        f"group-{record_id}",
        split,
    )


def _governance(records: tuple[SFTRecord, ...]):
    policy = SFTGovernancePolicy(
        "unit-test-policy",
        "unit-test-owner",
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        (
            SourceRule(
                "unit-test",
                "test-only",
                SourceDecision.ALLOW,
                (GovernancePurpose.TRAINING, GovernancePurpose.EVALUATION),
                "unit-test-evidence",
                None,
            ),
        ),
        ("normal",),
        (),
    )
    return audit_sft_governance(
        records,
        policy=policy,
        evaluated_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
    )


def test_character_ngram_jaccard_exposes_numerator_and_denominator() -> None:
    left = character_ngrams("abcd", size=2)
    right = character_ngrams("abce", size=2)

    similarity, intersection, union = shingle_jaccard(left, right)

    assert left == frozenset({"ab", "bc", "cd"})
    assert right == frozenset({"ab", "bc", "ce"})
    assert intersection == 2
    assert union == 4
    assert similarity == 0.5


def test_short_text_rule_returns_one_whole_text_shingle() -> None:
    assert character_ngrams("中文", size=5) == frozenset({"中文"})
    assert character_ngrams("中文", size=2) == frozenset({"中文"})


def test_normalization_profiles_have_explicitly_different_information_loss() -> None:
    assert normalize_near_duplicate_text(
        "e\u0301\n  value", profile=NearDuplicateProfile.NFC_WHITESPACE
    ) == "é value"
    assert normalize_near_duplicate_text(
        f"{FULLWIDTH_ABC}  Value",
        profile=NearDuplicateProfile.NFKC_CASEFOLD_WHITESPACE,
    ) == "abc value"
    assert normalize_near_duplicate_text(
        f"{FULLWIDTH_ABC}  Value", profile=NearDuplicateProfile.NFC_WHITESPACE
    ) == f"{FULLWIDTH_ABC} Value"


def test_cross_split_user_candidate_has_exact_jaccard_evidence() -> None:
    train = _record("train", DataSplit.TRAIN, user="abcdefghij")
    test = _record("test", DataSplit.TEST, user="abcdefghiX")

    report = audit_sft_near_duplicates(
        (train, test),
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=0.75,
        views=(NearDuplicateView.USER_CONTENT,),
    )

    assert not report.gate_passed
    assert report.record_pair_count == 1
    assert report.comparison_count == 1
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.left_record_id == "train"
    assert finding.right_record_id == "test"
    assert finding.view is NearDuplicateView.USER_CONTENT
    assert finding.similarity == finding.intersection_size / finding.union_size
    assert finding.similarity >= 0.75
    assert report.manifest_fingerprint.startswith("sha256:")
    assert report.to_dict()["scope"]["semantic_equivalence_verified"] is False


def test_cross_split_only_excludes_same_split_pairs_and_reports_denominators() -> None:
    records = (
        _record("train-a", DataSplit.TRAIN, user="abcdefghij"),
        _record("train-b", DataSplit.TRAIN, user="abcdefghiX"),
        _record("test", DataSplit.TEST, user="completely different"),
    )
    cross_only = audit_sft_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=0.75,
        views=(NearDuplicateView.USER_CONTENT,),
    )
    all_pairs = audit_sft_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=0.75,
        views=(NearDuplicateView.USER_CONTENT,),
        cross_split_only=False,
    )

    assert cross_only.record_pair_count == 2
    assert cross_only.comparison_count == 2
    assert cross_only.findings == ()
    assert all_pairs.record_pair_count == 3
    assert all_pairs.comparison_count == 3
    assert len(all_pairs.findings) == 1


def test_multiple_views_are_independent_comparisons() -> None:
    records = (
        _record("train", DataSplit.TRAIN, user="same user", assistant="first target"),
        _record("test", DataSplit.TEST, user="same user", assistant="second target"),
    )
    report = audit_sft_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=1,
        views=(NearDuplicateView.USER_CONTENT, NearDuplicateView.ASSISTANT_CONTENT),
    )

    assert report.record_pair_count == 1
    assert report.comparison_count == 2
    assert [finding.view for finding in report.findings] == [
        NearDuplicateView.USER_CONTENT
    ]


def test_aggressive_profile_can_merge_width_and_case_variants() -> None:
    records = (
        _record("train", DataSplit.TRAIN, user=f"{FULLWIDTH_ABC} Value"),
        _record("test", DataSplit.TEST, user="abc value"),
    )
    conservative = audit_sft_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=1,
        views=(NearDuplicateView.USER_CONTENT,),
    )
    aggressive = audit_sft_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFKC_CASEFOLD_WHITESPACE,
        ngram_size=3,
        threshold=1,
        views=(NearDuplicateView.USER_CONTENT,),
    )

    assert conservative.gate_passed
    assert not aggressive.gate_passed
    assert conservative.manifest_fingerprint != aggressive.manifest_fingerprint


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, math.nan, math.inf, True])
def test_near_duplicate_threshold_must_be_finite_probability(threshold: float) -> None:
    records = (_record("train", DataSplit.TRAIN, user="content"),)
    with pytest.raises(ValueError, match="threshold"):
        audit_sft_near_duplicates(
            records,
            profile=NearDuplicateProfile.NFC_WHITESPACE,
            ngram_size=3,
            threshold=threshold,
        )


def test_near_duplicate_policy_rejects_ambiguous_configuration() -> None:
    record = _record("train", DataSplit.TRAIN, user="content")
    with pytest.raises(ValueError, match="ngram_size"):
        audit_sft_near_duplicates(
            (record,),
            profile=NearDuplicateProfile.NFC_WHITESPACE,
            ngram_size=0,
            threshold=0.8,
        )
    with pytest.raises(ValueError, match="unique NearDuplicateView"):
        audit_sft_near_duplicates(
            (record,),
            profile=NearDuplicateProfile.NFC_WHITESPACE,
            ngram_size=3,
            threshold=0.8,
            views=(NearDuplicateView.USER_CONTENT, NearDuplicateView.USER_CONTENT),
        )
    duplicate_id = _record("train", DataSplit.TEST, user="other")
    with pytest.raises(ValueError, match="unique record ids"):
        audit_sft_near_duplicates(
            (record, duplicate_id),
            profile=NearDuplicateProfile.NFC_WHITESPACE,
            ngram_size=3,
            threshold=0.8,
        )


def test_training_readiness_binds_exact_and_near_audit_artifacts() -> None:
    training = (
        _record(
            "train", DataSplit.TRAIN, user="training prompt", assistant="train target"
        ),
    )
    combined = (
        *training,
        _record(
            "validation",
            DataSplit.VALIDATION,
            user="validation prompt",
            assistant="validation target",
        ),
        _record(
            "test", DataSplit.TEST, user="test prompt", assistant="test target"
        ),
    )
    binding = validate_training_subset(training, combined)
    near = audit_sft_near_duplicates(
        combined,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=1,
    )

    readiness = SFTTrainingReadinessReport.from_reports(
        binding, near, _governance(combined)
    )

    assert readiness.gate_passed
    assert readiness.manifest_fingerprint.startswith("sha256:")
    assert readiness.to_dict()["binding_fingerprint"] == binding.binding_fingerprint
    assert "does not prove semantic" in readiness.to_dict()["evidence_boundary"]
    assert readiness.to_dict()["scope"]["trainer_needs_held_out_access"] is False


def test_training_readiness_rejects_near_report_for_different_dataset() -> None:
    training = (
        _record(
            "train", DataSplit.TRAIN, user="training prompt", assistant="train target"
        ),
    )
    combined = (
        *training,
        _record(
            "validation",
            DataSplit.VALIDATION,
            user="validation prompt",
            assistant="validation target",
        ),
        _record(
            "test", DataSplit.TEST, user="test prompt", assistant="test target"
        ),
    )
    binding = validate_training_subset(training, combined)
    wrong_near = audit_sft_near_duplicates(
        reversed(combined),
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=1,
    )

    with pytest.raises(ValueError, match="dataset differs"):
        SFTTrainingReadinessReport.from_reports(
            binding, wrong_near, _governance(combined)
        )
