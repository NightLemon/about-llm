from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from about_llm.finetuning.data import (
    SFT_DATA_CONTRACT_VERSION,
    ChatMessage,
    DataSplit,
    FunctionToolCall,
    FunctionToolDefinition,
    MessageRole,
    SFTRecord,
    audit_sft_records,
    load_sft_records,
    validate_training_records,
    validate_training_subset,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
pytestmark = pytest.mark.contract


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


def _tool_definition(name: str) -> FunctionToolDefinition:
    property_name = "city" if name == "weather" else "expression"
    return FunctionToolDefinition(
        name=name,
        description=f"Call {name}.",
        parameters={
            "type": "object",
            "properties": {property_name: {"type": "string"}},
            "required": [property_name],
            "additionalProperties": False,
        },
    )


def _tool_call(call_id: str, name: str, **arguments: Any) -> FunctionToolCall:
    return FunctionToolCall(call_id=call_id, name=name, arguments=arguments)


def _tool_record(*, metadata: dict[str, Any] | None = None) -> SFTRecord:
    return SFTRecord(
        record_id="tools-train",
        messages=(
            ChatMessage(MessageRole.USER, "杭州天气和 2+2 分别是什么?"),
            ChatMessage(
                MessageRole.ASSISTANT,
                "",
                tool_calls=(
                    _tool_call("call-weather", "weather", city="Hangzhou"),
                    _tool_call("call-calculator", "calculator", expression="2+2"),
                ),
            ),
            ChatMessage(
                MessageRole.TOOL,
                '{"temperature_c":30}',
                tool_call_id="call-weather",
                name="weather",
            ),
            ChatMessage(
                MessageRole.TOOL,
                '{"value":4}',
                tool_call_id="call-calculator",
                name="calculator",
            ),
            ChatMessage(MessageRole.ASSISTANT, "杭州 30°C, 2+2=4。"),
        ),
        source="unit-test",
        license_id="test-only",
        task="tool-use",
        language="zh",
        risk="normal",
        group_id="tools-group",
        split=DataSplit.TRAIN,
        metadata={} if metadata is None else metadata,
        tools=(_tool_definition("weather"), _tool_definition("calculator")),
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


def test_tool_aware_loader_accepts_parallel_calls_and_empty_assistant_content(
    tmp_path: Path,
) -> None:
    expected = _tool_record(metadata={"private_note": "audit only"})
    path = tmp_path / "tools.jsonl"
    path.write_text(
        json.dumps(expected.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (loaded,) = load_sft_records(path)
    report = audit_sft_records((loaded,), required_splits=(DataSplit.TRAIN,))

    assert loaded.to_dict() == expected.to_dict()
    assert loaded.messages[1].content == ""
    assert report.gate_passed
    assert report.tool_definition_count == 2
    assert report.tool_call_count == 2
    assert report.tool_response_count == 2
    assert set(loaded.to_training_row()) == {"messages", "tools"}
    assert "metadata" not in loaded.to_training_row()
    assert loaded.to_training_row()["tools"] == expected.to_dict()["tools"]


@pytest.mark.parametrize("surface", ["tool_definition", "tool_call"])
def test_tool_loader_rejects_unknown_nested_fields(
    tmp_path: Path, surface: str
) -> None:
    payload = _tool_record().to_dict()
    if surface == "tool_definition":
        payload["tools"][0]["function"]["unexpected"] = True
    else:
        payload["messages"][1]["tool_calls"][0]["function"]["unexpected"] = True
    path = tmp_path / f"unknown-{surface}.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown=\['unexpected'\]"):
        load_sft_records(path)


@pytest.mark.parametrize(
    ("arguments_json", "expected"),
    [
        ('{"city":"Hangzhou","city":"Shanghai"}', "duplicate JSON object key"),
        ('{"temperature":NaN}', "non-standard JSON constant"),
    ],
)
def test_tool_loader_rejects_non_strict_nested_arguments(
    tmp_path: Path, arguments_json: str, expected: str
) -> None:
    line = json.dumps(
        _tool_record().to_dict(), ensure_ascii=False, separators=(",", ":")
    )
    original = '{"city":"Hangzhou"}'
    assert original in line
    path = tmp_path / "bad-tool-arguments.jsonl"
    path.write_text(line.replace(original, arguments_json, 1) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
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


def test_tool_values_are_deep_snapshots_and_fingerprints_cannot_drift() -> None:
    arguments = {"location": {"city": "Hangzhou", "coordinates": [30.3, 120.2]}}
    parameters = {
        "type": "object",
        "properties": {"location": {"type": "object"}},
    }
    call = FunctionToolCall("call-1", "weather", arguments)
    definition = FunctionToolDefinition("weather", "Get weather.", parameters)
    record = SFTRecord(
        "tool-snapshot",
        (
            ChatMessage(MessageRole.USER, "weather"),
            ChatMessage(MessageRole.ASSISTANT, "", tool_calls=(call,)),
            ChatMessage(
                MessageRole.TOOL,
                "sunny",
                tool_call_id="call-1",
                name="weather",
            ),
            ChatMessage(MessageRole.ASSISTANT, "Sunny."),
        ),
        "unit-test",
        "test-only",
        "tool-use",
        "en",
        "normal",
        "tool-snapshot-group",
        DataSplit.TRAIN,
        tools=(definition,),
    )
    fingerprint = record.record_fingerprint

    arguments["location"]["city"] = "Shanghai"
    arguments["location"]["coordinates"].append(0.0)
    parameters["properties"]["location"]["type"] = "string"

    assert call.to_dict()["function"]["arguments"] == {
        "location": {"city": "Hangzhou", "coordinates": [30.3, 120.2]}
    }
    assert definition.to_dict()["function"]["parameters"] == {
        "type": "object",
        "properties": {"location": {"type": "object"}},
    }
    assert record.record_fingerprint == fingerprint
    frozen_location = cast(dict[str, Any], call.arguments["location"])
    with pytest.raises(TypeError):
        frozen_location["city"] = "Nanjing"


def test_programmatic_tool_values_require_strict_json_objects() -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        FunctionToolCall("call-1", "weather", {"temperature": float("nan")})
    with pytest.raises(ValueError, match="strict JSON"):
        FunctionToolDefinition(
            "weather", "Get weather.", {"type": "object", "limit": float("inf")}
        )
    with pytest.raises(ValueError, match=r"parameters\.type must be 'object'"):
        FunctionToolDefinition("weather", "Get weather.", {"type": "array"})


def test_message_rejects_duplicate_parallel_call_ids() -> None:
    call = _tool_call("same", "weather", city="Hangzhou")
    with pytest.raises(ValueError, match="message tool call ids must be unique"):
        ChatMessage(MessageRole.ASSISTANT, "", tool_calls=(call, call))


@pytest.mark.parametrize(
    ("messages", "tools", "expected"),
    [
        (
            (
                ChatMessage(MessageRole.USER, "first"),
                ChatMessage(
                    MessageRole.ASSISTANT,
                    "",
                    tool_calls=(_tool_call("same", "weather", city="Hangzhou"),),
                ),
                ChatMessage(
                    MessageRole.TOOL,
                    "sunny",
                    tool_call_id="same",
                    name="weather",
                ),
                ChatMessage(MessageRole.ASSISTANT, "first done"),
                ChatMessage(MessageRole.USER, "again"),
                ChatMessage(
                    MessageRole.ASSISTANT,
                    "",
                    tool_calls=(_tool_call("same", "weather", city="Shanghai"),),
                ),
                ChatMessage(
                    MessageRole.TOOL,
                    "rainy",
                    tool_call_id="same",
                    name="weather",
                ),
                ChatMessage(MessageRole.ASSISTANT, "second done"),
            ),
            (_tool_definition("weather"),),
            "unique across the conversation",
        ),
        (
            (
                ChatMessage(MessageRole.USER, "weather"),
                ChatMessage(
                    MessageRole.ASSISTANT,
                    "",
                    tool_calls=(_tool_call("call-1", "weather", city="Hangzhou"),),
                ),
                ChatMessage(
                    MessageRole.TOOL,
                    "sunny",
                    tool_call_id="unknown",
                    name="weather",
                ),
                ChatMessage(MessageRole.ASSISTANT, "done"),
            ),
            (_tool_definition("weather"),),
            "unknown or already-resolved",
        ),
        (
            (
                ChatMessage(MessageRole.USER, "weather"),
                ChatMessage(
                    MessageRole.ASSISTANT,
                    "",
                    tool_calls=(_tool_call("call-1", "weather", city="Hangzhou"),),
                ),
                ChatMessage(
                    MessageRole.TOOL,
                    "sunny",
                    tool_call_id="call-1",
                    name="weather",
                ),
                ChatMessage(MessageRole.ASSISTANT, "done"),
            ),
            (),
            "no matching tool definition",
        ),
        (
            (
                ChatMessage(MessageRole.USER, "weather"),
                ChatMessage(
                    MessageRole.ASSISTANT,
                    "",
                    tool_calls=(_tool_call("call-1", "weather", city="Hangzhou"),),
                ),
                ChatMessage(
                    MessageRole.TOOL,
                    "sunny",
                    tool_call_id="call-1",
                    name="calculator",
                ),
                ChatMessage(MessageRole.ASSISTANT, "done"),
            ),
            (_tool_definition("weather"),),
            "does not match call",
        ),
        (
            (
                ChatMessage(MessageRole.USER, "weather"),
                ChatMessage(
                    MessageRole.ASSISTANT,
                    "",
                    tool_calls=(_tool_call("call-1", "weather", city="Hangzhou"),),
                ),
            ),
            (_tool_definition("weather"),),
            "must be followed by tool responses",
        ),
        (
            (
                ChatMessage(MessageRole.USER, "both"),
                ChatMessage(
                    MessageRole.ASSISTANT,
                    "",
                    tool_calls=(
                        _tool_call("call-1", "weather", city="Hangzhou"),
                        _tool_call("call-2", "calculator", expression="2+2"),
                    ),
                ),
                ChatMessage(
                    MessageRole.TOOL,
                    "sunny",
                    tool_call_id="call-1",
                    name="weather",
                ),
                ChatMessage(MessageRole.ASSISTANT, "done"),
            ),
            (_tool_definition("weather"), _tool_definition("calculator")),
            "one response each",
        ),
    ],
)
def test_conversation_rejects_invalid_tool_call_lifecycle(
    messages: tuple[ChatMessage, ...],
    tools: tuple[FunctionToolDefinition, ...],
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        SFTRecord(
            "bad-tools",
            messages,
            "unit-test",
            "test-only",
            "tool-use",
            "en",
            "normal",
            "bad-tools-group",
            DataSplit.TRAIN,
            tools=tools,
        )


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
