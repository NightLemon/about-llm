from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.finetuning.preference_data import load_preference_records
from about_llm.finetuning.preference_evaluation import (
    audit_preference_judgments,
    load_preference_judgments,
    summarize_preference_judgments,
)

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
CASES = PROJECT / "preference.example.jsonl"
JUDGMENTS = PROJECT / "preference-judgments.example.jsonl"


def _audit():
    return audit_preference_judgments(
        load_preference_records(CASES),
        load_preference_judgments(JUDGMENTS),
        judgments_per_pair=4,
        minimum_judgments_per_order=2,
    )


def test_authored_judgment_fixture_passes_binding_and_coverage_gate() -> None:
    report = _audit()

    assert report.gate_passed
    assert report.selected_case_ids == ("pref-validation-tie", "pref-test-gamma")
    assert report.judgment_count == 8
    assert report.annotator_count == 4
    assert report.pair_judgment_counts == {
        "pref-test-gamma": 4,
        "pref-validation-tie": 4,
    }
    assert report.pair_order_counts["pref-test-gamma"] == {
        "a_first": 2,
        "b_first": 2,
    }
    assert report.to_dict()["scope"]["random_assignment_verified"] is False


def test_judgment_summary_has_explicit_agreement_and_position_denominators() -> None:
    judgments = load_preference_judgments(JUDGMENTS)

    report = summarize_preference_judgments(
        judgments,
        _audit(),
        confidence=0.95,
        bootstrap_samples=2_000,
        bootstrap_seed=17,
    )

    assert report.label_counts == {"a": 1, "b": 5, "tie": 2}
    assert report.preferred_display_position_counts == {
        "first": 4,
        "second": 2,
        "tie": 2,
    }
    assert report.binary_judgment_count == 6
    assert report.pairwise_agreement_numerator == 7
    assert report.pairwise_agreement_denominator == 12
    assert report.pairwise_agreement == pytest.approx(7 / 12)
    assert report.fleiss_expected_agreement == pytest.approx(30 / 64)
    assert report.fleiss_kappa == pytest.approx(11 / 51)
    assert report.a_first_binary_count == 3
    assert report.a_second_binary_count == 3
    assert report.a_selection_rate_when_first == pytest.approx(1 / 3)
    assert report.a_selection_rate_when_second == 0
    assert report.mean_pair_position_effect == 0.5
    assert report.position_effect_pair_count == 2
    assert report.position_effect_confidence_low == 0
    assert report.position_effect_confidence_high == 1
    assert report.to_dict()["scope"]["authored_fixture_is_human_evidence"] is False


def test_audit_reports_relational_failures_instead_of_silently_dropping_rows() -> None:
    cases = load_preference_records(CASES)
    judgments = list(load_preference_judgments(JUDGMENTS))
    judgments[0] = replace(judgments[0], pair_id="missing-pair")
    judgments[1] = replace(
        judgments[1],
        annotator_id=judgments[2].annotator_id,
        blind_model_identity=False,
        rubric_revision="wrong-rubric",
    )

    report = audit_preference_judgments(
        cases,
        judgments,
        judgments_per_pair=4,
        minimum_judgments_per_order=2,
    )

    assert not report.gate_passed
    assert report.unknown_pair_judgment_ids == ("judgment-val-01",)
    assert report.unblinded_judgment_ids == ("judgment-val-02",)
    assert report.rubric_mismatch_judgment_ids == ("judgment-val-02",)
    assert "fixture-annotator-03:pref-validation-tie" in (
        report.duplicate_annotator_pair_assignments
    )
    assert "pref-validation-tie" in report.count_mismatch_pair_ids
    assert "pref-validation-tie" in report.order_coverage_mismatch_pair_ids
    with pytest.raises(ValueError, match="passing"):
        summarize_preference_judgments(judgments, report)


def test_loader_rejects_duplicate_unknown_and_ambiguous_strength(tmp_path: Path) -> None:
    line = JUDGMENTS.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(line.replace('{"id":', '{"id":"x","id":'), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_preference_judgments(duplicate)

    payload = json.loads(line)
    payload["surprise"] = True
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown=.*surprise"):
        load_preference_judgments(unknown)

    payload.pop("surprise")
    payload["label"] = "a"
    payload["preference_strength"] = "not_applicable"
    ambiguous = tmp_path / "ambiguous.jsonl"
    ambiguous.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binary labels require"):
        load_preference_judgments(ambiguous)


def test_summary_is_bound_to_audited_judgment_order() -> None:
    judgments = load_preference_judgments(JUDGMENTS)

    with pytest.raises(ValueError, match="differ from the audited"):
        summarize_preference_judgments(tuple(reversed(judgments)), _audit())


def test_judgment_metadata_is_a_recursive_snapshot() -> None:
    judgment = load_preference_judgments(JUDGMENTS)[0]

    with pytest.raises(TypeError):
        judgment.metadata["fixture"] = False  # type: ignore[index]
