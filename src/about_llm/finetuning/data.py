"""Strict, model-independent SFT data contracts and leakage auditing."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

SFT_DATA_CONTRACT_VERSION = "about-llm.sft-jsonl.v2"


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class FunctionToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.call_id, "tool call id")
        _require_nonempty_string(self.name, "tool call name")
        object.__setattr__(
            self,
            "arguments",
            _strict_json_object_snapshot(self.arguments, "tool call arguments"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": _thaw_json(self.arguments),
            },
        }


@dataclass(frozen=True)
class FunctionToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "tool definition name")
        _require_nonempty_string(self.description, "tool definition description")
        parameters = _strict_json_object_snapshot(
            self.parameters, "tool definition parameters"
        )
        if parameters.get("type") != "object":
            raise ValueError("tool definition parameters.type must be 'object'")
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _thaw_json(self.parameters),
            },
        }


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: str
    tool_calls: tuple[FunctionToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise ValueError("message role must be MessageRole")
        _text(self.content, "message content")
        tool_calls = tuple(self.tool_calls)
        if any(not isinstance(call, FunctionToolCall) for call in tool_calls):
            raise ValueError("message tool_calls must contain FunctionToolCall values")
        if len({call.call_id for call in tool_calls}) != len(tool_calls):
            raise ValueError("message tool call ids must be unique")
        object.__setattr__(self, "tool_calls", tool_calls)

        if self.role in (MessageRole.SYSTEM, MessageRole.USER):
            _require_nonempty_string(self.content, "message content")
            if tool_calls or self.tool_call_id is not None or self.name is not None:
                raise ValueError("system/user messages cannot carry tool-call fields")
        elif self.role is MessageRole.ASSISTANT:
            if not self.content.strip() and not tool_calls:
                raise ValueError(
                    "assistant message requires content or at least one tool call"
                )
            if self.tool_call_id is not None or self.name is not None:
                raise ValueError("assistant messages cannot be tool responses")
        else:
            _require_nonempty_string(self.content, "tool response content")
            if tool_calls:
                raise ValueError("tool response cannot contain tool calls")
            _require_nonempty_string(self.tool_call_id, "tool response tool_call_id")
            _require_nonempty_string(self.name, "tool response name")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class SFTRecord:
    record_id: str
    messages: tuple[ChatMessage, ...]
    source: str
    license_id: str
    task: str
    language: str
    risk: str
    group_id: str
    split: DataSplit
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[FunctionToolDefinition, ...] = ()
    _content_fingerprint: str = field(init=False, repr=False)
    _record_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("record_id", self.record_id),
            ("source", self.source),
            ("license_id", self.license_id),
            ("task", self.task),
            ("language", self.language),
            ("risk", self.risk),
            ("group_id", self.group_id),
        ):
            _require_nonempty_string(value, f"SFT {field_name}")
        messages = tuple(self.messages)
        tools = tuple(self.tools)
        if any(not isinstance(tool, FunctionToolDefinition) for tool in tools):
            raise ValueError(
                "SFT tools must contain FunctionToolDefinition values"
            )
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("SFT tool definition names must be unique")
        _validate_conversation(messages, tools)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)
        if not isinstance(self.split, DataSplit):
            raise ValueError("SFT split must be DataSplit")
        try:
            metadata_snapshot = cast(
                dict[str, Any], json.loads(canonical_json_bytes(self.metadata))
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"SFT metadata must be strict JSON: {error}") from error
        if not isinstance(metadata_snapshot, dict):
            raise ValueError("SFT metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze_json_object(metadata_snapshot))
        content: dict[str, Any] = {
            "messages": [message.to_dict() for message in messages]
        }
        if tools:
            content["tools"] = [tool.to_dict() for tool in tools]
        object.__setattr__(
            self,
            "_content_fingerprint",
            "sha256:" + artifact_fingerprint(content),
        )
        object.__setattr__(
            self,
            "_record_fingerprint",
            "sha256:" + artifact_fingerprint(self.to_dict()),
        )

    @property
    def content_fingerprint(self) -> str:
        return self._content_fingerprint

    @property
    def record_fingerprint(self) -> str:
        return self._record_fingerprint

    @property
    def assistant_character_count(self) -> int:
        return sum(
            len(message.content)
            for message in self.messages
            if message.role is MessageRole.ASSISTANT
        )

    @property
    def tool_call_count(self) -> int:
        return sum(len(message.tool_calls) for message in self.messages)

    @property
    def tool_response_count(self) -> int:
        return sum(message.role is MessageRole.TOOL for message in self.messages)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.record_id,
            "messages": [message.to_dict() for message in self.messages],
            "source": self.source,
            "license": self.license_id,
            "task": self.task,
            "language": self.language,
            "risk": self.risk,
            "group_id": self.group_id,
            "split": self.split.value,
            "metadata": _thaw_json(self.metadata),
        }
        if self.tools:
            payload["tools"] = [tool.to_dict() for tool in self.tools]
        return payload

    def to_training_row(self) -> dict[str, Any]:
        """Return only model inputs; governance fields remain in the audit manifest."""

        payload: dict[str, Any] = {
            "messages": [message.to_dict() for message in self.messages]
        }
        if self.tools:
            payload["tools"] = [tool.to_dict() for tool in self.tools]
        return payload


@dataclass(frozen=True)
class DuplicateGroup:
    identity: str
    record_ids: tuple[str, ...]
    splits: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "record_ids": list(self.record_ids),
            "splits": list(self.splits),
        }


@dataclass(frozen=True)
class SFTDataAuditReport:
    record_count: int
    required_splits: tuple[str, ...]
    missing_splits: tuple[str, ...]
    split_counts: Mapping[str, int]
    source_counts: Mapping[str, int]
    task_counts: Mapping[str, int]
    language_counts: Mapping[str, int]
    risk_counts: Mapping[str, int]
    assistant_character_count: int
    tool_definition_count: int
    tool_call_count: int
    tool_response_count: int
    duplicate_record_ids: tuple[str, ...]
    duplicate_content: tuple[DuplicateGroup, ...]
    cross_split_group_ids: tuple[DuplicateGroup, ...]
    cross_split_content: tuple[DuplicateGroup, ...]
    ordered_dataset_fingerprint: str
    unordered_dataset_fingerprint: str

    @property
    def gate_passed(self) -> bool:
        return not any(
            (
                self.missing_splits,
                self.duplicate_record_ids,
                self.duplicate_content,
                self.cross_split_group_ids,
                self.cross_split_content,
            )
        )

    @property
    def manifest_fingerprint(self) -> str:
        """Identify parsed record order plus the exact audit contract and policy."""

        identity = {
            "contract_version": SFT_DATA_CONTRACT_VERSION,
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "required_splits": list(self.required_splits),
            "gate_rules": [
                "required_splits_present",
                "record_ids_unique",
                "exact_message_content_unique",
                "group_id_does_not_cross_splits",
                "exact_message_content_does_not_cross_splits",
            ],
        }
        return "sha256:" + artifact_fingerprint(identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": SFT_DATA_CONTRACT_VERSION,
            "gate_passed": self.gate_passed,
            "record_count": self.record_count,
            "required_splits": list(self.required_splits),
            "missing_splits": list(self.missing_splits),
            "split_counts": dict(self.split_counts),
            "source_counts": dict(self.source_counts),
            "task_counts": dict(self.task_counts),
            "language_counts": dict(self.language_counts),
            "risk_counts": dict(self.risk_counts),
            "assistant_character_count": self.assistant_character_count,
            "assistant_count_unit": "unicode_codepoints_not_tokens",
            "tool_definition_count": self.tool_definition_count,
            "tool_call_count": self.tool_call_count,
            "tool_response_count": self.tool_response_count,
            "duplicate_record_ids": list(self.duplicate_record_ids),
            "duplicate_content": [item.to_dict() for item in self.duplicate_content],
            "cross_split_group_ids": [
                item.to_dict() for item in self.cross_split_group_ids
            ],
            "cross_split_content": [
                item.to_dict() for item in self.cross_split_content
            ],
            "ordered_dataset_fingerprint": self.ordered_dataset_fingerprint,
            "unordered_dataset_fingerprint": self.unordered_dataset_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "scope": {
                "exact_content_only": True,
                "near_duplicate_detection": False,
                "license_legality_verified": False,
                "pii_or_secret_detection": False,
                "tokenizer_or_assistant_mask_verified": False,
            },
        }


@dataclass(frozen=True)
class SFTTrainingBindingReport:
    training_report: SFTDataAuditReport
    split_report: SFTDataAuditReport

    @property
    def binding_fingerprint(self) -> str:
        identity = {
            "contract_version": SFT_DATA_CONTRACT_VERSION,
            "training_manifest_fingerprint": self.training_report.manifest_fingerprint,
            "split_manifest_fingerprint": self.split_report.manifest_fingerprint,
            "binding_rule": "ordered_train_subset_exactly_matches",
        }
        return "sha256:" + artifact_fingerprint(identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": SFT_DATA_CONTRACT_VERSION,
            "gate_passed": True,
            "binding_rule": "ordered_train_subset_exactly_matches",
            "training_manifest_fingerprint": self.training_report.manifest_fingerprint,
            "split_manifest_fingerprint": self.split_report.manifest_fingerprint,
            "training_ordered_dataset_fingerprint": (
                self.training_report.ordered_dataset_fingerprint
            ),
            "combined_ordered_dataset_fingerprint": (
                self.split_report.ordered_dataset_fingerprint
            ),
            "binding_fingerprint": self.binding_fingerprint,
        }


def load_sft_records(path: Path) -> tuple[SFTRecord, ...]:
    """Load a strict JSONL dataset with duplicate-key and unknown-field rejection."""

    records: list[SFTRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: invalid strict JSON: {error}"
            ) from error
        prefix = f"{path}:{line_number}"
        record = _record(value, prefix)
        allowed = {
            "id",
            "messages",
            "source",
            "license",
            "task",
            "language",
            "risk",
            "group_id",
            "split",
            "metadata",
            "tools",
        }
        required = allowed - {"metadata", "tools"}
        _expect_fields(record, required=required, allowed=allowed, prefix=prefix)
        messages_raw = record["messages"]
        if not isinstance(messages_raw, list):
            raise ValueError(f"{prefix}: messages must be an array")
        messages = tuple(
            _parse_message(message, prefix, index)
            for index, message in enumerate(messages_raw, 1)
        )
        tools_raw = record.get("tools", [])
        if not isinstance(tools_raw, list):
            raise ValueError(f"{prefix}: tools must be an array")
        tools = tuple(
            _parse_tool_definition(tool, prefix, index)
            for index, tool in enumerate(tools_raw, 1)
        )
        try:
            split = DataSplit(record["split"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}: unsupported split {record['split']!r}") from error
        try:
            records.append(
                SFTRecord(
                    record_id=_string(record["id"], f"{prefix}.id"),
                    messages=messages,
                    source=_string(record["source"], f"{prefix}.source"),
                    license_id=_string(record["license"], f"{prefix}.license"),
                    task=_string(record["task"], f"{prefix}.task"),
                    language=_string(record["language"], f"{prefix}.language"),
                    risk=_string(record["risk"], f"{prefix}.risk"),
                    group_id=_string(record["group_id"], f"{prefix}.group_id"),
                    split=split,
                    metadata=record.get("metadata", {}),
                    tools=tools,
                )
            )
        except ValueError as error:
            raise ValueError(f"{prefix}: {error}") from error
    if not records:
        raise ValueError(f"{path}: dataset contains no records")
    return tuple(records)


def audit_sft_records(
    records: Iterable[SFTRecord],
    *,
    required_splits: Iterable[DataSplit] = (
        DataSplit.TRAIN,
        DataSplit.VALIDATION,
        DataSplit.TEST,
    ),
) -> SFTDataAuditReport:
    """Audit exact identities and group isolation without claiming semantic dedup."""

    snapshot = tuple(records)
    if not snapshot:
        raise ValueError("SFT audit requires at least one record")
    if any(not isinstance(record, SFTRecord) for record in snapshot):
        raise ValueError("SFT audit accepts only SFTRecord values")
    required = tuple(required_splits)
    if not required or any(not isinstance(split, DataSplit) for split in required):
        raise ValueError("required_splits must contain DataSplit values")
    if len(required) != len(set(required)):
        raise ValueError("required_splits must not contain duplicates")

    id_counts = Counter(record.record_id for record in snapshot)
    split_counts = Counter(record.split.value for record in snapshot)
    content_groups = _group_records(snapshot, key=lambda record: record.content_fingerprint)
    group_groups = _group_records(snapshot, key=lambda record: record.group_id)
    duplicate_content = tuple(
        _duplicate_group(identity, grouped)
        for identity, grouped in sorted(content_groups.items())
        if len(grouped) > 1
    )
    cross_split_group_ids = tuple(
        _duplicate_group(identity, grouped)
        for identity, grouped in sorted(group_groups.items())
        if len({record.split for record in grouped}) > 1
    )
    cross_split_content = tuple(
        group for group in duplicate_content if len(group.splits) > 1
    )
    ordered_identity = {
        "ordered_record_fingerprints": [
            record.record_fingerprint for record in snapshot
        ]
    }
    unordered_identity = {
        "sorted_record_fingerprints": sorted(
            record.record_fingerprint for record in snapshot
        )
    }
    return SFTDataAuditReport(
        record_count=len(snapshot),
        required_splits=tuple(split.value for split in required),
        missing_splits=tuple(
            split.value for split in required if split_counts[split.value] == 0
        ),
        split_counts=_frozen_counts(split_counts),
        source_counts=_frozen_counts(Counter(record.source for record in snapshot)),
        task_counts=_frozen_counts(Counter(record.task for record in snapshot)),
        language_counts=_frozen_counts(Counter(record.language for record in snapshot)),
        risk_counts=_frozen_counts(Counter(record.risk for record in snapshot)),
        assistant_character_count=sum(
            record.assistant_character_count for record in snapshot
        ),
        tool_definition_count=sum(len(record.tools) for record in snapshot),
        tool_call_count=sum(record.tool_call_count for record in snapshot),
        tool_response_count=sum(record.tool_response_count for record in snapshot),
        duplicate_record_ids=tuple(
            record_id for record_id, count in sorted(id_counts.items()) if count > 1
        ),
        duplicate_content=duplicate_content,
        cross_split_group_ids=cross_split_group_ids,
        cross_split_content=cross_split_content,
        ordered_dataset_fingerprint="sha256:" + artifact_fingerprint(ordered_identity),
        unordered_dataset_fingerprint="sha256:"
        + artifact_fingerprint(unordered_identity),
    )


def validate_training_records(records: Iterable[SFTRecord]) -> SFTDataAuditReport:
    """Require a duplicate-free train-only file before a trainer sees it."""

    snapshot = tuple(records)
    non_train = [
        record.record_id for record in snapshot if record.split is not DataSplit.TRAIN
    ]
    if non_train:
        raise ValueError(f"training file contains non-train record(s): {non_train}")
    report = audit_sft_records(snapshot, required_splits=(DataSplit.TRAIN,))
    if not report.gate_passed:
        raise ValueError(
            "training data gate failed: duplicate ids/content or missing train split"
        )
    return report


def validate_training_subset(
    training_records: Iterable[SFTRecord],
    audited_records: Iterable[SFTRecord],
) -> SFTTrainingBindingReport:
    """Bind trainer input to the ordered train subset of a passing split audit."""

    training = tuple(training_records)
    audited = tuple(audited_records)
    training_report = validate_training_records(training)
    split_report = audit_sft_records(audited)
    if not split_report.gate_passed:
        raise ValueError("combined split audit failed; inspect the audit report")
    audited_training = tuple(
        record for record in audited if record.split is DataSplit.TRAIN
    )
    actual = tuple(record.record_fingerprint for record in training)
    expected = tuple(record.record_fingerprint for record in audited_training)
    if actual != expected:
        actual_ids = tuple(record.record_id for record in training)
        expected_ids = tuple(record.record_id for record in audited_training)
        if actual_ids == expected_ids:
            changed = tuple(
                record_id
                for record_id, actual_fingerprint, expected_fingerprint in zip(
                    actual_ids, actual, expected, strict=True
                )
                if actual_fingerprint != expected_fingerprint
            )
            raise ValueError(
                "training records differ from audited train records for id(s): "
                f"{changed}"
            )
        if set(actual_ids) == set(expected_ids):
            raise ValueError("training record order differs from audited train order")
        missing = tuple(sorted(set(expected_ids) - set(actual_ids)))
        unexpected = tuple(sorted(set(actual_ids) - set(expected_ids)))
        raise ValueError(
            "training record membership differs from audited train subset: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return SFTTrainingBindingReport(training_report, split_report)


def _validate_conversation(
    messages: tuple[ChatMessage, ...],
    tools: tuple[FunctionToolDefinition, ...],
) -> None:
    if not messages:
        raise ValueError("messages must not be empty")
    if any(not isinstance(message, ChatMessage) for message in messages):
        raise ValueError("messages must contain ChatMessage values")
    system_positions = [
        index for index, message in enumerate(messages) if message.role is MessageRole.SYSTEM
    ]
    if system_positions not in ([], [0]):
        raise ValueError("system message is optional but may appear only once at index zero")
    start = 1 if system_positions else 0
    if start >= len(messages) or messages[start].role is not MessageRole.USER:
        raise ValueError("first non-system message must be user")
    if messages[-1].role is not MessageRole.ASSISTANT:
        raise ValueError("final message must be assistant to provide an SFT target")
    transitions = {
        MessageRole.SYSTEM: {MessageRole.USER},
        MessageRole.USER: {MessageRole.ASSISTANT},
        MessageRole.ASSISTANT: {MessageRole.USER, MessageRole.TOOL},
        MessageRole.TOOL: {MessageRole.TOOL, MessageRole.ASSISTANT},
    }
    for previous, current in pairwise(messages):
        if current.role not in transitions[previous.role]:
            raise ValueError(
                f"unsupported role transition: {previous.role.value}->{current.role.value}"
            )

    declared_tools = {tool.name for tool in tools}
    seen_call_ids: set[str] = set()
    outstanding: dict[str, str] = {}
    for index, message in enumerate(messages):
        if message.role is MessageRole.ASSISTANT:
            if message.tool_calls:
                if outstanding:
                    raise ValueError("assistant issued tool calls before prior calls resolved")
                if index + 1 >= len(messages) or messages[index + 1].role is not MessageRole.TOOL:
                    raise ValueError("assistant tool calls must be followed by tool responses")
                for call in message.tool_calls:
                    if call.call_id in seen_call_ids:
                        raise ValueError("tool call ids must be unique across the conversation")
                    if call.name not in declared_tools:
                        raise ValueError(
                            f"tool call {call.name!r} has no matching tool definition"
                        )
                    seen_call_ids.add(call.call_id)
                    outstanding[call.call_id] = call.name
            elif outstanding:
                raise ValueError("assistant responded before all tool calls were resolved")
        elif message.role is MessageRole.TOOL:
            call_id = cast(str, message.tool_call_id)
            name = cast(str, message.name)
            expected_name = outstanding.get(call_id)
            if expected_name is None:
                raise ValueError(
                    f"tool response references unknown or already-resolved call {call_id!r}"
                )
            if name != expected_name:
                raise ValueError(
                    f"tool response name {name!r} does not match call {expected_name!r}"
                )
            del outstanding[call_id]
            next_role = messages[index + 1].role if index + 1 < len(messages) else None
            if outstanding and next_role is not MessageRole.TOOL:
                raise ValueError("all outstanding tool calls require one response each")
        elif outstanding:
            raise ValueError("tool calls must be resolved before the conversation continues")
    if outstanding:
        raise ValueError("conversation ends with unresolved tool calls")


def _parse_message(value: Any, prefix: str, index: int) -> ChatMessage:
    message_prefix = f"{prefix}.messages[{index}]"
    record = _record(value, message_prefix)
    _expect_fields(
        record,
        required={"role", "content"},
        allowed={"role", "content", "tool_calls", "tool_call_id", "name"},
        prefix=message_prefix,
    )
    try:
        role = MessageRole(record["role"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{message_prefix}: unsupported role {record['role']!r}") from error
    tool_calls_raw = record.get("tool_calls", [])
    if not isinstance(tool_calls_raw, list):
        raise ValueError(f"{message_prefix}.tool_calls must be an array")
    tool_calls = tuple(
        _parse_tool_call(tool_call, message_prefix, tool_index)
        for tool_index, tool_call in enumerate(tool_calls_raw, 1)
    )
    tool_call_id = record.get("tool_call_id")
    name = record.get("name")
    return ChatMessage(
        role,
        _text(record["content"], f"{message_prefix}.content"),
        tool_calls=tool_calls,
        tool_call_id=(
            None
            if tool_call_id is None
            else _string(tool_call_id, f"{message_prefix}.tool_call_id")
        ),
        name=None if name is None else _string(name, f"{message_prefix}.name"),
    )


def _parse_tool_call(value: Any, prefix: str, index: int) -> FunctionToolCall:
    call_prefix = f"{prefix}.tool_calls[{index}]"
    record = _record(value, call_prefix)
    _expect_fields(
        record,
        required={"id", "type", "function"},
        allowed={"id", "type", "function"},
        prefix=call_prefix,
    )
    if record["type"] != "function":
        raise ValueError(f"{call_prefix}.type must be 'function'")
    function = _record(record["function"], f"{call_prefix}.function")
    _expect_fields(
        function,
        required={"name", "arguments"},
        allowed={"name", "arguments"},
        prefix=f"{call_prefix}.function",
    )
    arguments = _record(function["arguments"], f"{call_prefix}.function.arguments")
    return FunctionToolCall(
        call_id=_string(record["id"], f"{call_prefix}.id"),
        name=_string(function["name"], f"{call_prefix}.function.name"),
        arguments=arguments,
    )


def _parse_tool_definition(
    value: Any, prefix: str, index: int
) -> FunctionToolDefinition:
    tool_prefix = f"{prefix}.tools[{index}]"
    record = _record(value, tool_prefix)
    _expect_fields(
        record,
        required={"type", "function"},
        allowed={"type", "function"},
        prefix=tool_prefix,
    )
    if record["type"] != "function":
        raise ValueError(f"{tool_prefix}.type must be 'function'")
    function = _record(record["function"], f"{tool_prefix}.function")
    _expect_fields(
        function,
        required={"name", "description", "parameters"},
        allowed={"name", "description", "parameters"},
        prefix=f"{tool_prefix}.function",
    )
    parameters = _record(function["parameters"], f"{tool_prefix}.function.parameters")
    return FunctionToolDefinition(
        name=_string(function["name"], f"{tool_prefix}.function.name"),
        description=_string(
            function["description"], f"{tool_prefix}.function.description"
        ),
        parameters=parameters,
    )


def _group_records(
    records: tuple[SFTRecord, ...], *, key: Callable[[SFTRecord], str]
) -> dict[str, list[SFTRecord]]:
    grouped: dict[str, list[SFTRecord]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    return grouped


def _duplicate_group(identity: str, records: list[SFTRecord]) -> DuplicateGroup:
    return DuplicateGroup(
        identity=identity,
        record_ids=tuple(sorted(record.record_id for record in records)),
        splits=tuple(sorted({record.split.value for record in records})),
    )


def _frozen_counts(counts: Counter[str]) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(counts.items())))


def _freeze_json_object(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _freeze_json_object(cast(dict[str, Any], value))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _strict_json_object_snapshot(
    value: Mapping[str, Any], field_name: str
) -> Mapping[str, Any]:
    try:
        snapshot = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be strict JSON: {error}") from error
    if not isinstance(snapshot, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return _freeze_json_object(cast(dict[str, Any], snapshot))


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _record(value: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: expected a JSON object")
    return cast(dict[str, Any], value)


def _expect_fields(
    record: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    prefix: str,
) -> None:
    missing = required - set(record)
    unknown = set(record) - allowed
    if missing or unknown:
        raise ValueError(
            f"{prefix}: field mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _string(value: Any, prefix: str) -> str:
    _require_nonempty_string(value, prefix)
    return cast(str, value)


def _text(value: Any, prefix: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{prefix} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{prefix} contains an unpaired Unicode surrogate") from error
    return value


def _require_nonempty_string(value: Any, prefix: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{prefix} contains an unpaired Unicode surrogate") from error


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
