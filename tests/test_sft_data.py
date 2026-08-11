from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from about_llm.finetuning.data import (
    SFT_DATA_CONTRACT_VERSION,
    ChatMessage,
    DataSplit,
    MessageRole,
    SFTRecord,
    audit_sft_records,
    load_sft_records,
    validate_training_records,
    validate_training_subset,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"


def _record(
    record_id: str,
    split: DataSplit,
    *,
    group_id: str | None = None,
    prompt: str | None = None,
    answer: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SFTRecord:
    return SFTRecord(
        record_id=record_id,
        messages=(
            ChatMessage(MessageRole.USER, prompt or f"question-{record_id}"),
            ChatMessage(MessageRole.ASSISTANT, answer or f"answer-{record_id}"),
        ),
        source="unit-test",
        license_id="test-only",
        task="definition",
        language="en",
        risk="normal",
        group_id=group_id or f"group-{record_id}",
        split=split,
        metadata={} if metadata is None else metadata,
    )


def test_example_artifacts_pass_their_declared_gates() -> None:
    audit_records = load_sft_records(PROJECT / "audit.example.jsonl")
    audit = audit_sft_records(audit_records)
    train_records = load_sft_records(PROJECT / "train.example.jsonl")
    train_audit = validate_training_records(train_records)

    assert audit.gate_passed
    assert audit.record_count == 4
    assert audit.split_counts == {"test": 1, "train": 2, "validation": 1}
    assert train_audit.gate_passed
    assert train_audit.split_counts == {"train": 2}
    assert audit.to_dict()["contract_version"] == SFT_DATA_CONTRACT_VERSION
    assert audit.to_dict()["assistant_count_unit"] == "unicode_codepoints_not_tokens"
    assert audit.manifest_fingerprint.startswith("sha256:")


def test_ordered_and_order_independent_fingerprints_have_distinct_semantics() -> None:
    records = (
        _record("train", DataSplit.TRAIN),
        _record("validation", DataSplit.VALIDATION),
        _record("test", DataSplit.TEST),
    )
    forward = audit_sft_records(records)
    reverse = audit_sft_records(reversed(records))

    assert forward.ordered_dataset_fingerprint != reverse.ordered_dataset_fingerprint
    assert (
        forward.unordered_dataset_fingerprint
        == reverse.unordered_dataset_fingerprint
    )
    assert forward.manifest_fingerprint != reverse.manifest_fingerprint


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('{"id":"one","id":"two"}', "duplicate JSON object key"),
        ('{"value":NaN}', "non-standard JSON constant"),
    ],
)
def test_loader_rejects_non_strict_json(
    tmp_path: Path, line: str, expected: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        load_sft_records(path)


def test_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        '{"id":"x","messages":[{"role":"user","content":"q"},'
        '{"role":"assistant","content":"a"}],"source":"s","license":"l",'
        '"task":"t","language":"en","risk":"normal","group_id":"g",'
        '"split":"train","unexpected":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unknown=\['unexpected'\]"):
        load_sft_records(path)


@pytest.mark.parametrize(
    "messages",
    [
        (),
        (ChatMessage(MessageRole.ASSISTANT, "answer"),),
        (
            ChatMessage(MessageRole.USER, "question"),
            ChatMessage(MessageRole.USER, "second question"),
            ChatMessage(MessageRole.ASSISTANT, "answer"),
        ),
        (
            ChatMessage(MessageRole.USER, "question"),
            ChatMessage(MessageRole.ASSISTANT, "answer"),
            ChatMessage(MessageRole.SYSTEM, "late system"),
            ChatMessage(MessageRole.ASSISTANT, "answer"),
        ),
    ],
)
def test_conversation_contract_rejects_ambiguous_training_targets(
    messages: tuple[ChatMessage, ...],
) -> None:
    with pytest.raises(ValueError):
        SFTRecord(
            "bad",
            messages,
            "source",
            "license",
            "task",
            "en",
            "normal",
            "group",
            DataSplit.TRAIN,
        )


def test_audit_reports_exact_and_group_cross_split_leakage_separately() -> None:
    train = _record(
        "train", DataSplit.TRAIN, group_id="shared-group", prompt="same", answer="same"
    )
    validation = _record(
        "validation", DataSplit.VALIDATION, group_id="shared-group"
    )
    test = _record(
        "test", DataSplit.TEST, group_id="different-group", prompt="same", answer="same"
    )
    report = audit_sft_records((train, validation, test))

    assert not report.gate_passed
    assert report.cross_split_group_ids[0].identity == "shared-group"
    assert report.cross_split_group_ids[0].record_ids == ("train", "validation")
    assert report.cross_split_content[0].record_ids == ("test", "train")
    assert report.duplicate_content == report.cross_split_content
    assert report.to_dict()["scope"]["near_duplicate_detection"] is False


def test_duplicate_ids_fail_even_when_content_differs() -> None:
    report = audit_sft_records(
        (
            _record("same-id", DataSplit.TRAIN, prompt="first"),
            _record("same-id", DataSplit.TRAIN, group_id="second", prompt="second"),
        ),
        required_splits=(DataSplit.TRAIN,),
    )

    assert not report.gate_passed
    assert report.duplicate_record_ids == ("same-id",)


def test_training_gate_rejects_non_train_records_before_trainer_input() -> None:
    with pytest.raises(ValueError, match="non-train"):
        validate_training_records((_record("validation", DataSplit.VALIDATION),))


def test_training_row_excludes_governance_metadata() -> None:
    record = _record("train", DataSplit.TRAIN, metadata={"private_note": "audit only"})

    assert set(record.to_training_row()) == {"messages"}
    assert record.to_dict()["metadata"] == {"private_note": "audit only"}


def test_metadata_is_a_deep_snapshot_and_fingerprint_cannot_drift() -> None:
    metadata = {"nested": {"values": [1, 2]}}
    record = _record("train", DataSplit.TRAIN, metadata=metadata)
    fingerprint = record.record_fingerprint
    metadata["nested"]["values"].append(3)

    assert record.to_dict()["metadata"] == {"nested": {"values": [1, 2]}}
    assert record.record_fingerprint == fingerprint
    nested = cast(dict[str, Any], record.metadata["nested"])
    with pytest.raises(TypeError):
        nested["changed"] = True


def test_programmatic_records_reject_unpaired_unicode_surrogates() -> None:
    with pytest.raises(ValueError, match="unpaired Unicode surrogate"):
        ChatMessage(MessageRole.USER, "\ud800")


def test_audit_rejects_empty_or_duplicate_required_split_policy() -> None:
    record = _record("train", DataSplit.TRAIN)
    with pytest.raises(ValueError, match="must contain DataSplit"):
        audit_sft_records((record,), required_splits=())
    with pytest.raises(ValueError, match="must not contain duplicates"):
        audit_sft_records(
            (record,), required_splits=(DataSplit.TRAIN, DataSplit.TRAIN)
        )


def test_training_subset_is_bound_to_passing_combined_artifact() -> None:
    training = load_sft_records(PROJECT / "train.example.jsonl")
    combined = load_sft_records(PROJECT / "audit.example.jsonl")

    binding = validate_training_subset(training, combined)

    assert binding.training_report.record_count == 2
    assert binding.split_report.record_count == 4
    assert binding.training_report.ordered_dataset_fingerprint == audit_sft_records(
        tuple(record for record in combined if record.split is DataSplit.TRAIN),
        required_splits=(DataSplit.TRAIN,),
    ).ordered_dataset_fingerprint
    assert binding.binding_fingerprint.startswith("sha256:")
    assert binding.to_dict()["binding_rule"] == "ordered_train_subset_exactly_matches"


def test_training_subset_rejects_changed_content_order_and_membership() -> None:
    training = load_sft_records(PROJECT / "train.example.jsonl")
    combined = load_sft_records(PROJECT / "audit.example.jsonl")
    changed = (replace(training[0], source="changed-after-audit"), training[1])
    with pytest.raises(ValueError, match="differ from audited"):
        validate_training_subset(changed, combined)
    with pytest.raises(ValueError, match="order differs"):
        validate_training_subset(reversed(training), combined)
    with pytest.raises(ValueError, match="membership differs"):
        validate_training_subset(training[:1], combined)


def test_training_subset_rejects_combined_leakage_before_membership_check() -> None:
    training = load_sft_records(PROJECT / "train.example.jsonl")
    combined = load_sft_records(PROJECT / "audit.example.jsonl")
    leaked = (
        *combined[:2],
        replace(combined[2], group_id=combined[0].group_id),
        combined[3],
    )

    with pytest.raises(ValueError, match="combined split audit failed"):
        validate_training_subset(training, leaked)
