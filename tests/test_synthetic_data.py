from __future__ import annotations

import math

import pytest

from about_llm.synthetic_data import (
    FingerprintProfile,
    MixtureComponent,
    SourceKind,
    SyntheticRecord,
    VerificationResult,
    audit_synthetic_records,
    content_fingerprint,
    plan_mixture,
)


def verification(
    verifier_id: str, *, revision: str = "verifier@v1", passed: bool = True
) -> VerificationResult:
    return VerificationResult(verifier_id, revision, passed)


def record(
    record_id: str,
    *,
    content: str = "解释检索增强生成。",
    parent_ids: tuple[str, ...] = ("real-anchor-1",),
    generator_revision: str = "teacher@v1",
    generation_round: int = 1,
    verifications: tuple[VerificationResult, ...] = (
        VerificationResult("schema", "schema@v1", True),
        VerificationResult("grounding", "grounding@v2", True),
    ),
    human_reviewed: bool = False,
) -> SyntheticRecord:
    return SyntheticRecord(
        record_id=record_id,
        content=content,
        parent_ids=parent_ids,
        generator_revision=generator_revision,
        prompt_revision="prompt@v3",
        generation_round=generation_round,
        verifications=verifications,
        human_reviewed=human_reviewed,
    )


def test_audit_distinguishes_eligible_missing_and_failed_records() -> None:
    records = [
        record("eligible", human_reviewed=True),
        record("missing", verifications=(verification("schema"),)),
        record(
            "failed",
            verifications=(verification("schema"), verification("grounding", passed=False)),
        ),
    ]

    report = audit_synthetic_records(
        records,
        required_verifiers=("schema", "grounding"),
        known_parent_ids=("real-anchor-1",),
    )

    assert report.candidate_count == 3
    assert report.eligible_record_ids == ("eligible",)
    assert report.missing_verifier_record_ids == ("missing",)
    assert report.failed_verifier_record_ids == ("failed",)
    assert report.eligibility_rate == pytest.approx(1 / 3)
    assert report.human_reviewed_count == 1
    assert report.eligible_human_reviewed_count == 1


def test_audit_reports_generator_verifier_revision_overlap() -> None:
    item = record(
        "self-checked",
        generator_revision="same-model@v1",
        verifications=(verification("judge", revision="same-model@v1"),),
    )

    report = audit_synthetic_records(
        [item], required_verifiers=("judge",), known_parent_ids=("real-anchor-1",)
    )

    assert report.self_verified_record_ids == ("self-checked",)
    assert report.eligible_record_ids == ("self-checked",)


def test_audit_reports_unresolved_lineage_without_rejecting_quality_gate() -> None:
    item = record("candidate", parent_ids=("unknown-parent",))

    report = audit_synthetic_records(
        [item], required_verifiers=("schema", "grounding")
    )

    assert report.unresolved_parent_pairs == (("candidate", "unknown-parent"),)
    assert report.eligible_record_ids == ("candidate",)


def test_audit_resolves_synthetic_and_known_external_parents() -> None:
    first = record("round-1", parent_ids=("real-anchor-1",))
    second = record(
        "round-2", parent_ids=(first.record_id,), generation_round=2, content="第二代样本"
    )

    report = audit_synthetic_records(
        [first, second],
        required_verifiers=("schema", "grounding"),
        known_parent_ids=("real-anchor-1",),
    )

    assert report.unresolved_parent_pairs == ()
    assert [(item.generation_round, item.candidate_count) for item in report.rounds] == [
        (1, 1),
        (2, 1),
    ]


def test_audit_reports_cycles_and_nonmonotonic_parent_rounds() -> None:
    first = record(
        "round-1",
        parent_ids=("round-2",),
        generation_round=1,
    )
    second = record(
        "round-2",
        parent_ids=("round-1",),
        generation_round=2,
        content="第二代样本",
    )
    descendant = record(
        "round-3",
        parent_ids=("round-2",),
        generation_round=3,
        content="第三代样本",
    )

    report = audit_synthetic_records(
        [descendant, second, first],
        required_verifiers=("schema", "grounding"),
    )

    assert report.unresolved_parent_pairs == ()
    assert report.nonmonotonic_parent_pairs == (("round-1", "round-2"),)
    assert report.lineage_cycle_record_ids == ("round-1", "round-2")
    assert report.eligible_record_ids == ("round-1", "round-2", "round-3")


def test_byte_exact_and_nfc_whitespace_profiles_have_explicit_semantics() -> None:
    composed = "é  value\nnext"
    decomposed = "e\u0301 value next"

    assert content_fingerprint(
        composed, profile=FingerprintProfile.BYTE_EXACT
    ) != content_fingerprint(decomposed, profile=FingerprintProfile.BYTE_EXACT)
    assert content_fingerprint(
        composed, profile=FingerprintProfile.NFC_WHITESPACE
    ) == content_fingerprint(decomposed, profile=FingerprintProfile.NFC_WHITESPACE)


def test_audit_reports_duplicates_but_does_not_hide_them_from_eligibility() -> None:
    first = record("a")
    second = record("b", content="解释检索增强生成。")

    report = audit_synthetic_records(
        [second, first],
        required_verifiers=("schema", "grounding"),
        known_parent_ids=("real-anchor-1",),
    )

    assert report.duplicate_content_groups == (("a", "b"),)
    assert report.eligible_count == 2
    assert report.eligible_unique_content_count == 1


def test_mixture_plan_exposes_synthetic_fraction_and_repetition() -> None:
    plan = plan_mixture(
        [
            MixtureComponent("real", SourceKind.REAL, unique_tokens=800, weight=3),
            MixtureComponent(
                "synthetic-r1",
                SourceKind.SYNTHETIC,
                unique_tokens=100,
                weight=1,
                generation_round=1,
            ),
        ],
        total_consumed_tokens=2_000,
    )

    assert plan.synthetic_fraction == pytest.approx(0.25)
    real, synthetic = plan.exposures
    assert real.expected_consumed_tokens == pytest.approx(1_500)
    assert real.expected_repetition_factor == pytest.approx(1.875)
    assert synthetic.expected_consumed_tokens == pytest.approx(500)
    assert synthetic.expected_repetition_factor == pytest.approx(5)
    assert sum(item.normalized_fraction for item in plan.exposures) == pytest.approx(1)


def test_mixture_plan_uses_normalized_not_pre_normalized_weights() -> None:
    plan = plan_mixture(
        [
            MixtureComponent("real", SourceKind.REAL, unique_tokens=100, weight=2),
            MixtureComponent(
                "synthetic", SourceKind.SYNTHETIC, unique_tokens=100, weight=2, generation_round=1
            ),
        ],
        total_consumed_tokens=100,
    )

    assert [item.normalized_fraction for item in plan.exposures] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize(
    "records",
    [
        [],
        [record("duplicate"), record("duplicate", content="other")],
    ],
)
def test_audit_rejects_empty_or_duplicate_record_ids(records: list[SyntheticRecord]) -> None:
    with pytest.raises(ValueError):
        audit_synthetic_records(
            records,
            required_verifiers=("schema",),
            known_parent_ids=("real-anchor-1",),
        )


def test_audit_requires_a_nonempty_unique_verifier_contract() -> None:
    item = record("item")
    for required in ((), ("schema", "schema"), ("",)):
        with pytest.raises(ValueError, match="required_verifiers"):
            audit_synthetic_records([item], required_verifiers=required)


def test_audit_requires_unique_known_parent_ids() -> None:
    with pytest.raises(ValueError, match="known_parent_ids must be unique"):
        audit_synthetic_records(
            [record("item")],
            required_verifiers=("schema", "grounding"),
            known_parent_ids=("real-anchor-1", "real-anchor-1"),
        )


def test_record_rejects_duplicate_verifier_ids() -> None:
    with pytest.raises(ValueError, match="verifier_id"):
        record(
            "item",
            verifications=(verification("judge"), verification("judge", revision="judge@v2")),
        )


def test_record_requires_lineage_and_positive_generation_round() -> None:
    with pytest.raises(ValueError, match="parent_ids"):
        record("item", parent_ids=())
    with pytest.raises(ValueError, match="generation_round"):
        record("item", generation_round=0)


def test_synthetic_component_requires_generation_round() -> None:
    with pytest.raises(ValueError, match="generation_round"):
        MixtureComponent("synthetic", SourceKind.SYNTHETIC, unique_tokens=100, weight=1)


def test_real_component_forbids_generation_round() -> None:
    with pytest.raises(ValueError, match="must not"):
        MixtureComponent(
            "real", SourceKind.REAL, unique_tokens=100, weight=1, generation_round=1
        )


def test_component_requires_typed_source_kind() -> None:
    with pytest.raises(TypeError, match="SourceKind"):
        MixtureComponent(
            "real",
            "real",  # type: ignore[arg-type]
            unique_tokens=100,
            weight=1,
        )


@pytest.mark.parametrize("total", [0, -1, True])
def test_mixture_plan_requires_positive_integer_budget(total: object) -> None:
    component = MixtureComponent("real", SourceKind.REAL, unique_tokens=100, weight=1)
    with pytest.raises(ValueError, match="positive integer"):
        plan_mixture([component], total_consumed_tokens=total)  # type: ignore[arg-type]


def test_mixture_plan_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="names"):
        plan_mixture(
            [
                MixtureComponent("same", SourceKind.REAL, unique_tokens=100, weight=1),
                MixtureComponent("same", SourceKind.REAL, unique_tokens=200, weight=1),
            ],
            total_consumed_tokens=100,
        )


@pytest.mark.parametrize("weight", [0, -1, math.inf, math.nan])
def test_component_rejects_nonpositive_or_nonfinite_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        MixtureComponent("real", SourceKind.REAL, unique_tokens=100, weight=weight)
