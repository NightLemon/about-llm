from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.finetuning.data import ChatMessage, DataSplit, MessageRole
from about_llm.finetuning.preference_data import (
    PreferenceLabel,
    PreferenceStrength,
    audit_preference_records,
    load_preference_records,
    validate_dpo_training_records,
    validate_dpo_training_subset,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "projects" / "single-gpu-finetuning" / "preference.example.jsonl"


def test_fixture_preserves_annotation_context_and_passes_split_gate() -> None:
    records = load_preference_records(FIXTURE)

    report = audit_preference_records(records)

    assert report.gate_passed
    assert report.record_count == 4
    assert report.split_counts == {"test": 1, "train": 2, "validation": 1}
    assert report.label_counts == {"a": 1, "b": 2, "tie": 1}
    assert report.preferred_display_position_counts == {
        "first": 2,
        "second": 1,
        "tie": 1,
    }
    assert report.manifest_fingerprint.startswith("sha256:")
    assert report.to_dict()["scope"]["position_bias_estimated"] is False


def test_dpo_conversion_maps_label_without_losing_conversational_prompt() -> None:
    records = load_preference_records(FIXTURE)

    row_a = records[0].to_dpo_row()
    row_b = records[1].to_dpo_row()

    assert row_a["prompt"][0]["role"] == "system"
    assert row_a["chosen"][0]["content"] == "good alpha answer"
    assert row_a["rejected"][0]["content"] == "bad alpha answer"
    assert row_b["chosen"][0]["content"] == "good beta answer"
    assert row_b["rejected"][0]["content"] == "bad beta answer"
    with pytest.raises(ValueError, match="non-binary label"):
        records[2].to_dpo_row()


def test_dpo_training_gate_accepts_only_binary_train_records() -> None:
    records = load_preference_records(FIXTURE)

    report = validate_dpo_training_records(records[:2])

    assert report.gate_passed
    with pytest.raises(ValueError, match="non-train"):
        validate_dpo_training_records(records)
    tie_train = replace(records[2], split=DataSplit.TRAIN)
    with pytest.raises(ValueError, match="binary labels"):
        validate_dpo_training_records((tie_train,))


def test_dpo_training_subset_binds_exact_ordered_binary_train_rows() -> None:
    combined = load_preference_records(FIXTURE)

    binding = validate_dpo_training_subset(combined[:2], combined)

    assert binding.training_report.record_count == 2
    assert binding.excluded_nonbinary_train_record_ids == ()
    assert binding.binding_fingerprint.startswith("sha256:")
    with pytest.raises(ValueError, match="order differs"):
        validate_dpo_training_subset(tuple(reversed(combined[:2])), combined)
    with pytest.raises(ValueError, match="membership differs"):
        validate_dpo_training_subset(combined[:1], combined)


def test_dpo_training_subset_rejects_changed_row_with_same_id() -> None:
    combined = load_preference_records(FIXTURE)
    changed = (replace(combined[0], source="changed-source"), combined[1])

    with pytest.raises(ValueError, match="differ from audited binary train records"):
        validate_dpo_training_subset(changed, combined)


def test_dpo_training_subset_preserves_but_excludes_nonbinary_train_rows() -> None:
    records = load_preference_records(FIXTURE)
    train_tie = replace(
        records[2],
        record_id="pref-train-tie",
        prompt=(ChatMessage(MessageRole.USER, "Unique train tie prompt."),),
        group_id="preference-train-tie",
        split=DataSplit.TRAIN,
    )
    combined = (records[0], records[1], train_tie, records[2], records[3])

    binding = validate_dpo_training_subset(records[:2], combined)

    assert binding.excluded_nonbinary_train_record_ids == ("pref-train-tie",)
    assert binding.to_dict()["excluded_nonbinary_train_record_ids"] == [
        "pref-train-tie"
    ]


def test_swapped_candidate_pair_is_still_a_duplicate() -> None:
    first = load_preference_records(FIXTURE)[0]
    swapped = replace(
        first,
        record_id="swapped",
        candidate_a=first.candidate_b,
        candidate_b=first.candidate_a,
        label=PreferenceLabel.B,
    )

    report = audit_preference_records((first, swapped), required_splits=(DataSplit.TRAIN,))

    assert not report.gate_passed
    assert len(report.duplicate_pairs) == 1


def test_identical_candidates_fail_audit_without_destroying_invalid_record() -> None:
    first = load_preference_records(FIXTURE)[0]
    invalid = replace(
        first,
        record_id="identical",
        candidate_b=first.candidate_a,
        label=PreferenceLabel.INVALID,
        preference_strength=PreferenceStrength.NOT_APPLICABLE,
    )

    report = audit_preference_records((invalid,), required_splits=(DataSplit.TRAIN,))

    assert not report.gate_passed
    assert report.identical_candidate_record_ids == ("identical",)


def test_cross_split_prompt_and_group_leakage_fail_gate() -> None:
    first = load_preference_records(FIXTURE)[0]
    leaked = replace(
        first,
        record_id="test-copy",
        candidate_a="different a",
        candidate_b="different b",
        split=DataSplit.TEST,
    )

    report = audit_preference_records(
        (first, leaked), required_splits=(DataSplit.TRAIN, DataSplit.TEST)
    )

    assert not report.gate_passed
    assert len(report.cross_split_group_ids) == 1
    assert len(report.cross_split_prompts) == 1


def test_loader_rejects_duplicate_and_unknown_json_fields(tmp_path: Path) -> None:
    line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(line.replace('{"id":', '{"id":"x","id":'), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_preference_records(duplicate)

    payload = json.loads(line)
    payload["surprise"] = True
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown=.*surprise"):
        load_preference_records(unknown)


def test_loader_rejects_position_strength_and_generator_ambiguity(
    tmp_path: Path,
) -> None:
    base = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    cases = (
        ({**base, "presentation_order": ["a", "a"]}, "presentation_order"),
        ({**base, "preference_strength": "not_applicable"}, "binary labels"),
        ({**base, "generator_revisions": {"a": "only-one"}}, "exactly keys"),
    )
    for index, (payload, message) in enumerate(cases):
        path = tmp_path / f"bad-{index}.jsonl"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_preference_records(path)


def test_prompt_must_end_at_a_model_response_boundary(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    payload["prompt"].append({"role": "assistant", "content": "already answered"})
    path = tmp_path / "bad-prompt.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must end with user or tool"):
        load_preference_records(path)


def test_metadata_and_generator_revisions_are_recursive_snapshots() -> None:
    record = load_preference_records(FIXTURE)[0]

    with pytest.raises(TypeError):
        record.metadata["fixture"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        record.generator_revisions["a"] = "changed"  # type: ignore[index]
