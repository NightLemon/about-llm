"""Strict offline replay for a reviewed subset of OpenAI Responses events.

The reference consumes SDK-shaped event objects.  It does not send an API
request or claim complete coverage of the evolving Responses API surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

MAX_EVENT_FILE_BYTES = 4 * 1024 * 1024
MAX_EVENT_LINE_BYTES = 1024 * 1024
MAX_EVENTS = 10_000
SCHEMA_VERSION = "about-llm.openai-responses-event-replay.v1"

TerminalStatus = Literal["completed", "incomplete", "failed"]
UpdateKind = Literal[
    "text_delta",
    "refusal_delta",
    "function_arguments_delta",
    "terminal",
]


@dataclass(frozen=True)
class ResponsesReplayUpdate:
    kind: UpdateKind
    item_id: str | None = None
    delta: str | None = None
    terminal_status: TerminalStatus | None = None


@dataclass(frozen=True)
class FunctionCallReceipt:
    item_id: str
    call_id: str
    name: str
    arguments: str
    arguments_is_strict_object: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "arguments_is_strict_object": self.arguments_is_strict_object,
        }


@dataclass(frozen=True)
class ResponsesReplayReceipt:
    response_id: str
    model: str
    terminal_status: TerminalStatus
    terminal_reason: str | None
    output_text: str
    refusals: tuple[str, ...]
    function_calls: tuple[FunctionCallReceipt, ...]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    event_count: int
    output_item_count: int
    event_types: tuple[str, ...]
    event_projection_fingerprint: str
    input_size_bytes: int | None = None
    input_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "response_id": self.response_id,
            "model": self.model,
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
            "output_text": self.output_text,
            "refusals": list(self.refusals),
            "function_calls": [call.to_dict() for call in self.function_calls],
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
            "event_count": self.event_count,
            "output_item_count": self.output_item_count,
            "event_types": list(self.event_types),
            "event_projection_fingerprint": self.event_projection_fingerprint,
            "input": {
                "size_bytes": self.input_size_bytes,
                "sha256": self.input_sha256,
            },
            "scope": {
                "sdk_shaped_event_replay_executed": True,
                "strict_json_duplicate_nonfinite_unknown_event_field_rejection": True,
                "sequence_and_item_lifecycle_checked": True,
                "terminal_output_and_usage_reconciled": True,
                "http_sse_or_websocket_transport_executed": False,
                "openai_sdk_or_remote_api_executed": False,
                "model_output_quality_or_safety_proved": False,
                "provider_identity_usage_or_billing_authenticated": False,
                "complete_responses_api_surface_supported": False,
            },
            "evidence_boundary": (
                "This is an authored offline replay of a reviewed Responses event subset. "
                "It proves local parsing and lifecycle invariants for the supplied bytes, "
                "not OpenAI service execution, complete API compatibility, provider identity, "
                "usage or billing, model quality, safety, or production reliability."
            ),
        }
        payload["receipt_fingerprint"] = "sha256:" + artifact_fingerprint(payload)
        return payload


@dataclass
class _ContentState:
    content_index: int
    kind: Literal["output_text", "refusal"]
    parts: list[str]
    semantic_done: bool = False
    part_done: bool = False

    @property
    def value(self) -> str:
        return "".join(self.parts)


@dataclass
class _ItemState:
    item_id: str
    output_index: int
    kind: str
    call_id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)
    arguments_done: bool = False
    contents: dict[int, _ContentState] = field(default_factory=dict)
    done_item: dict[str, Any] | None = None


_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "response.created": frozenset({"type", "sequence_number", "response"}),
    "response.in_progress": frozenset({"type", "sequence_number", "response"}),
    "response.completed": frozenset({"type", "sequence_number", "response"}),
    "response.incomplete": frozenset({"type", "sequence_number", "response"}),
    "response.failed": frozenset({"type", "sequence_number", "response"}),
    "response.output_item.added": frozenset({"type", "sequence_number", "output_index", "item"}),
    "response.output_item.done": frozenset({"type", "sequence_number", "output_index", "item"}),
    "response.content_part.added": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "content_index",
            "part",
        }
    ),
    "response.content_part.done": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "content_index",
            "part",
        }
    ),
    "response.output_text.delta": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "content_index",
            "delta",
            "logprobs",
        }
    ),
    "response.output_text.done": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "content_index",
            "text",
            "logprobs",
        }
    ),
    "response.refusal.delta": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "content_index",
            "delta",
        }
    ),
    "response.refusal.done": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "content_index",
            "refusal",
        }
    ),
    "response.function_call_arguments.delta": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "delta",
        }
    ),
    "response.function_call_arguments.done": frozenset(
        {
            "type",
            "sequence_number",
            "output_index",
            "item_id",
            "name",
            "arguments",
        }
    ),
}


class OpenAIResponsesEventReplay:
    """Validate one ordered, SDK-shaped Responses event sequence.

    Requiring sequence numbers to start at zero and be contiguous is a stricter
    local evidence rule for these replay artifacts.  It is not presented as a
    universal transport recovery guarantee.
    """

    def __init__(self) -> None:
        self._last_sequence = -1
        self._response_id: str | None = None
        self._model: str | None = None
        self._terminal_status: TerminalStatus | None = None
        self._terminal_reason: str | None = None
        self._usage: tuple[int | None, int | None, int | None] = (
            None,
            None,
            None,
        )
        self._items: dict[str, _ItemState] = {}
        self._output_ids: dict[int, str] = {}
        self._events: list[dict[str, Any]] = []

    def consume(self, event: Mapping[str, Any]) -> tuple[ResponsesReplayUpdate, ...]:
        if self._terminal_status is not None:
            raise ValueError("Responses event received after terminal event")
        snapshot = _snapshot_mapping(event, "Responses event")
        event_type = _nonempty_string(snapshot.get("type"), "event type")
        expected_fields = _EVENT_FIELDS.get(event_type)
        if expected_fields is None:
            if event_type == "error":
                raise ValueError("Responses stream emitted an error event")
            raise ValueError(f"unsupported Responses event type: {event_type}")
        if set(snapshot) != expected_fields:
            raise ValueError(
                f"{event_type} fields differ: expected {sorted(expected_fields)}, "
                f"got {sorted(snapshot)}"
            )
        sequence = _index(snapshot["sequence_number"], "sequence_number")
        if sequence != self._last_sequence + 1:
            raise ValueError("Responses sequence_number must start at zero and be contiguous")

        updates = self._dispatch(event_type, snapshot)
        self._last_sequence = sequence
        self._events.append(snapshot)
        return updates

    def finish(self) -> ResponsesReplayReceipt:
        if self._terminal_status is None:
            raise ValueError("Responses event replay ended without a terminal event")
        if self._response_id is None or self._model is None:
            raise AssertionError("terminal replay lost response identity")
        ordered = [self._items[self._output_ids[index]] for index in sorted(self._output_ids)]
        if any(item.done_item is None for item in ordered):
            raise ValueError("Responses replay ended with unfinished output items")
        output_text = "".join(
            content.value
            for item in ordered
            if item.kind == "message"
            for content in (item.contents[index] for index in sorted(item.contents))
            if content.kind == "output_text"
        )
        refusals = tuple(
            content.value
            for item in ordered
            if item.kind == "message"
            for content in (item.contents[index] for index in sorted(item.contents))
            if content.kind == "refusal"
        )
        calls = tuple(
            FunctionCallReceipt(
                item_id=item.item_id,
                call_id=cast(str, item.call_id),
                name=cast(str, item.name),
                arguments="".join(item.arguments),
                arguments_is_strict_object=_is_strict_json_object("".join(item.arguments)),
            )
            for item in ordered
            if item.kind == "function_call"
        )
        input_tokens, output_tokens, total_tokens = self._usage
        return ResponsesReplayReceipt(
            response_id=self._response_id,
            model=self._model,
            terminal_status=self._terminal_status,
            terminal_reason=self._terminal_reason,
            output_text=output_text,
            refusals=refusals,
            function_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            event_count=len(self._events),
            output_item_count=len(ordered),
            event_types=tuple(event["type"] for event in self._events),
            event_projection_fingerprint="sha256:" + artifact_fingerprint({"events": self._events}),
        )

    def _dispatch(
        self, event_type: str, event: dict[str, Any]
    ) -> tuple[ResponsesReplayUpdate, ...]:
        if event_type == "response.created":
            self._consume_created(event)
            return ()
        if self._response_id is None:
            raise ValueError("Responses event arrived before response.created")
        if event_type == "response.in_progress":
            self._consume_in_progress(event)
            return ()
        if event_type == "response.output_item.added":
            self._consume_item_added(event)
            return ()
        if event_type == "response.content_part.added":
            self._consume_content_added(event)
            return ()
        if event_type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            return self._consume_content_delta(event_type, event)
        if event_type in {
            "response.output_text.done",
            "response.refusal.done",
        }:
            self._consume_content_semantic_done(event_type, event)
            return ()
        if event_type == "response.content_part.done":
            self._consume_content_part_done(event)
            return ()
        if event_type == "response.function_call_arguments.delta":
            return self._consume_arguments_delta(event)
        if event_type == "response.function_call_arguments.done":
            self._consume_arguments_done(event)
            return ()
        if event_type == "response.output_item.done":
            self._consume_item_done(event)
            return ()
        if event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            status = cast(TerminalStatus, event_type.removeprefix("response."))
            self._consume_terminal(event, status=status)
            return (ResponsesReplayUpdate("terminal", terminal_status=status),)
        raise AssertionError(f"unhandled reviewed event type: {event_type}")

    def _consume_created(self, event: Mapping[str, Any]) -> None:
        if self._response_id is not None:
            raise ValueError("duplicate response.created event")
        response = _mapping(event["response"], "created response")
        response_id, model, status, output = _response_identity(response)
        if status not in {"queued", "in_progress"}:
            raise ValueError("created response must be queued or in_progress")
        if output:
            raise ValueError("created response output must be empty in this replay contract")
        self._response_id = response_id
        self._model = model

    def _consume_in_progress(self, event: Mapping[str, Any]) -> None:
        response = _mapping(event["response"], "in-progress response")
        response_id, model, status, _ = _response_identity(response)
        self._check_identity(response_id, model)
        if status != "in_progress":
            raise ValueError("response.in_progress must carry in_progress status")

    def _consume_item_added(self, event: Mapping[str, Any]) -> None:
        output_index = _index(event["output_index"], "output_index")
        if output_index != len(self._output_ids):
            raise ValueError("output_index must be contiguous in this replay contract")
        item = _mapping(event["item"], "added output item")
        item_id = _nonempty_string(item.get("id"), "output item id")
        if item_id in self._items:
            raise ValueError("duplicate output item id")
        kind = _nonempty_string(item.get("type"), "output item type")
        state = _ItemState(item_id=item_id, output_index=output_index, kind=kind)
        if kind == "message":
            if item.get("role") != "assistant":
                raise ValueError("output message role must be assistant")
            if item.get("content") != []:
                raise ValueError("added output message content must start empty")
        elif kind == "function_call":
            state.call_id = _nonempty_string(item.get("call_id"), "function call id")
            state.name = _nonempty_string(item.get("name"), "function name")
            initial_arguments = item.get("arguments", "")
            if not isinstance(initial_arguments, str):
                raise ValueError("function arguments must be a string")
            if initial_arguments:
                state.arguments.append(initial_arguments)
        self._items[item_id] = state
        self._output_ids[output_index] = item_id

    def _consume_content_added(self, event: Mapping[str, Any]) -> None:
        item = self._referenced_item(event, expected_kind="message")
        content_index = _index(event["content_index"], "content_index")
        if content_index != len(item.contents):
            raise ValueError("content_index must be contiguous in this replay contract")
        part = _mapping(event["part"], "added content part")
        kind, value = _content_part_value(part)
        item.contents[content_index] = _ContentState(content_index, kind, [value])

    def _consume_content_delta(
        self, event_type: str, event: Mapping[str, Any]
    ) -> tuple[ResponsesReplayUpdate, ...]:
        expected: Literal["output_text", "refusal"] = (
            "output_text" if event_type == "response.output_text.delta" else "refusal"
        )
        content = self._referenced_content(event, expected_kind=expected)
        if content.semantic_done:
            raise ValueError("content delta arrived after semantic done event")
        delta = event.get("delta")
        if not isinstance(delta, str):
            raise ValueError("Responses content delta must be a string")
        if event_type == "response.output_text.delta":
            logprobs = event.get("logprobs")
            if not isinstance(logprobs, list):
                raise ValueError("output_text.delta logprobs must be an array")
        content.parts.append(delta)
        if not delta:
            return ()
        kind: UpdateKind = "text_delta" if expected == "output_text" else "refusal_delta"
        return (ResponsesReplayUpdate(kind, item_id=event["item_id"], delta=delta),)

    def _consume_content_semantic_done(self, event_type: str, event: Mapping[str, Any]) -> None:
        expected: Literal["output_text", "refusal"] = (
            "output_text" if event_type == "response.output_text.done" else "refusal"
        )
        content = self._referenced_content(event, expected_kind=expected)
        if content.semantic_done:
            raise ValueError("duplicate content semantic done event")
        key = "text" if expected == "output_text" else "refusal"
        value = event.get(key)
        if not isinstance(value, str) or value != content.value:
            raise ValueError(f"{event_type} value does not match accumulated deltas")
        if expected == "output_text" and not isinstance(event.get("logprobs"), list):
            raise ValueError("output_text.done logprobs must be an array")
        content.semantic_done = True

    def _consume_content_part_done(self, event: Mapping[str, Any]) -> None:
        content = self._referenced_content(event, expected_kind=None)
        if not content.semantic_done:
            raise ValueError("content_part.done arrived before semantic done")
        if content.part_done:
            raise ValueError("duplicate content_part.done event")
        kind, value = _content_part_value(_mapping(event["part"], "done content part"))
        if kind != content.kind or value != content.value:
            raise ValueError("content_part.done does not match accumulated content")
        content.part_done = True

    def _consume_arguments_delta(
        self, event: Mapping[str, Any]
    ) -> tuple[ResponsesReplayUpdate, ...]:
        item = self._referenced_item(event, expected_kind="function_call")
        if item.arguments_done:
            raise ValueError("function arguments delta arrived after done")
        delta = event.get("delta")
        if not isinstance(delta, str):
            raise ValueError("function arguments delta must be a string")
        item.arguments.append(delta)
        if not delta:
            return ()
        return (
            ResponsesReplayUpdate("function_arguments_delta", item_id=item.item_id, delta=delta),
        )

    def _consume_arguments_done(self, event: Mapping[str, Any]) -> None:
        item = self._referenced_item(event, expected_kind="function_call")
        if item.arguments_done:
            raise ValueError("duplicate function arguments done event")
        if event.get("name") != item.name:
            raise ValueError("function arguments done name differs from added item")
        arguments = event.get("arguments")
        if not isinstance(arguments, str) or arguments != "".join(item.arguments):
            raise ValueError("function arguments done value differs from accumulated deltas")
        item.arguments_done = True

    def _consume_item_done(self, event: Mapping[str, Any]) -> None:
        output_index = _index(event["output_index"], "output_index")
        item_value = _mapping(event["item"], "done output item")
        item_id = _nonempty_string(item_value.get("id"), "done output item id")
        item = self._items.get(item_id)
        if item is None or item.output_index != output_index:
            raise ValueError("output_item.done references an unknown output item")
        if item.done_item is not None:
            raise ValueError("duplicate output_item.done event")
        if item_value.get("type") != item.kind:
            raise ValueError("output item type changed before done")
        if item.kind == "message":
            self._validate_done_message(item, item_value)
        elif item.kind == "function_call":
            self._validate_done_function(item, item_value)
        item.done_item = _snapshot_mapping(item_value, "done output item")

    def _validate_done_message(self, item: _ItemState, value: Mapping[str, Any]) -> None:
        if value.get("role") != "assistant":
            raise ValueError("done output message role must be assistant")
        if any(not content.part_done for content in item.contents.values()):
            raise ValueError("output message finished with active content parts")
        raw_content = value.get("content")
        if not isinstance(raw_content, list) or len(raw_content) != len(item.contents):
            raise ValueError("done output message content count differs from stream")
        for index, raw_part in enumerate(raw_content):
            kind, text = _content_part_value(_mapping(raw_part, "done message part"))
            streamed = item.contents[index]
            if kind != streamed.kind or text != streamed.value:
                raise ValueError("done output message content differs from stream")

    def _validate_done_function(self, item: _ItemState, value: Mapping[str, Any]) -> None:
        if not item.arguments_done:
            raise ValueError("function output item finished before arguments done")
        if value.get("call_id") != item.call_id or value.get("name") != item.name:
            raise ValueError("done function identity differs from added item")
        if value.get("arguments") != "".join(item.arguments):
            raise ValueError("done function arguments differ from streamed arguments")

    def _consume_terminal(self, event: Mapping[str, Any], *, status: TerminalStatus) -> None:
        response = _mapping(event["response"], "terminal response")
        response_id, model, response_status, output = _response_identity(response)
        self._check_identity(response_id, model)
        if response_status != status:
            raise ValueError("terminal event type and response status differ")
        ordered_items = [self._items[self._output_ids[index]] for index in sorted(self._output_ids)]
        if any(item.done_item is None for item in ordered_items):
            raise ValueError("terminal event arrived before all output items were done")
        expected_output = [cast(dict[str, Any], item.done_item) for item in ordered_items]
        if canonical_json_bytes(output) != canonical_json_bytes(expected_output):
            raise ValueError("terminal response output differs from output_item.done events")

        error = response.get("error")
        incomplete = response.get("incomplete_details")
        if status == "completed":
            if error is not None or incomplete is not None:
                raise ValueError("completed response cannot carry error/incomplete details")
            reason = None
        elif status == "incomplete":
            details = _mapping(incomplete, "incomplete_details")
            reason = _nonempty_string(details.get("reason"), "incomplete reason")
        else:
            failure = _mapping(error, "response error")
            reason = _nonempty_string(failure.get("code"), "response error code")
            _nonempty_string(failure.get("message"), "response error message")
        self._usage = _response_usage(response.get("usage"))
        self._terminal_status = status
        self._terminal_reason = reason

    def _referenced_item(self, event: Mapping[str, Any], *, expected_kind: str) -> _ItemState:
        output_index = _index(event["output_index"], "output_index")
        item_id = _nonempty_string(event.get("item_id"), "item_id")
        item = self._items.get(item_id)
        if item is None or item.output_index != output_index:
            raise ValueError("event references an unknown output item")
        if item.kind != expected_kind:
            raise ValueError(f"event requires a {expected_kind} output item")
        if item.done_item is not None:
            raise ValueError("event references an output item that is already done")
        return item

    def _referenced_content(
        self,
        event: Mapping[str, Any],
        *,
        expected_kind: Literal["output_text", "refusal"] | None,
    ) -> _ContentState:
        item = self._referenced_item(event, expected_kind="message")
        index = _index(event["content_index"], "content_index")
        content = item.contents.get(index)
        if content is None:
            raise ValueError("event references an unknown content part")
        if expected_kind is not None and content.kind != expected_kind:
            raise ValueError(f"event requires a {expected_kind} content part")
        return content

    def _check_identity(self, response_id: str, model: str) -> None:
        if response_id != self._response_id or model != self._model:
            raise ValueError("response id or model drifted within event stream")


def replay_response_events(
    events: Iterable[Mapping[str, Any]],
) -> ResponsesReplayReceipt:
    state = OpenAIResponsesEventReplay()
    count = 0
    for count, event in enumerate(events, start=1):
        if count > MAX_EVENTS:
            raise ValueError("Responses event count exceeds resource limit")
        state.consume(event)
    if count == 0:
        raise ValueError("Responses event sequence must not be empty")
    return state.finish()


def load_response_event_jsonl(path: Path) -> tuple[tuple[dict[str, Any], ...], bytes]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_EVENT_FILE_BYTES:
        raise ValueError("Responses event file is empty or exceeds byte limit")
    if not raw.endswith(b"\n"):
        raise ValueError("Responses event JSONL must end with a newline")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Responses event file contains invalid UTF-8") from error
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.isspace():
            raise ValueError(f"Responses event line {line_number} is blank")
        if len(line.encode("utf-8")) > MAX_EVENT_LINE_BYTES:
            raise ValueError("Responses event line exceeds byte limit")
        value = _strict_json(line, context=f"line {line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"Responses event line {line_number} must be an object")
        events.append(_snapshot_mapping(value, f"line {line_number}"))
        if len(events) > MAX_EVENTS:
            raise ValueError("Responses event count exceeds resource limit")
    return tuple(events), raw


def replay_response_event_file(path: Path) -> ResponsesReplayReceipt:
    events, raw = load_response_event_jsonl(path)
    receipt = replay_response_events(events)
    return replace(
        receipt,
        input_size_bytes=len(raw),
        input_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _strict_json(text: str, *, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeEncodeError) as error:
        raise ValueError(f"invalid strict JSON in {context}") from error


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _snapshot_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        snapshot = json.loads(canonical_json_bytes(value))
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise ValueError(f"{label} must contain strict finite JSON values") from error
    if not isinstance(snapshot, dict):
        raise AssertionError("canonical mapping snapshot changed JSON type")
    return cast(dict[str, Any], snapshot)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.isspace():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _index(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _response_identity(
    response: Mapping[str, Any],
) -> tuple[str, str, str, list[Any]]:
    response_id = _nonempty_string(response.get("id"), "response id")
    model = _nonempty_string(response.get("model"), "response model")
    status = _nonempty_string(response.get("status"), "response status")
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("response output must be an array")
    return response_id, model, status, output


def _content_part_value(
    part: Mapping[str, Any],
) -> tuple[Literal["output_text", "refusal"], str]:
    kind = part.get("type")
    if kind == "output_text":
        value = part.get("text")
    elif kind == "refusal":
        value = part.get("refusal")
    else:
        raise ValueError("unsupported Responses message content part type")
    if not isinstance(value, str):
        raise ValueError("Responses content part value must be a string")
    return kind, value


def _token_count(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer or null")
    return value


def _response_usage(value: Any) -> tuple[int | None, int | None, int | None]:
    if value is None:
        return None, None, None
    usage = _mapping(value, "response usage")
    input_tokens = _token_count(usage.get("input_tokens"), "input_tokens")
    output_tokens = _token_count(usage.get("output_tokens"), "output_tokens")
    total_tokens = _token_count(usage.get("total_tokens"), "total_tokens")
    if (
        input_tokens is not None
        and output_tokens is not None
        and total_tokens != input_tokens + output_tokens
    ):
        raise ValueError("total_tokens differs from input_tokens + output_tokens")
    return input_tokens, output_tokens, total_tokens


def _is_strict_json_object(text: str) -> bool:
    try:
        value = _strict_json(text, context="function arguments")
    except ValueError:
        return False
    return isinstance(value, dict)
