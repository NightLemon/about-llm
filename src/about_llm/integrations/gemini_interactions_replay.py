"""Offline replay for a reviewed subset of Gemini Interactions SSE events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from about_llm.inference.sse import SSEDecoder, SSEEvent, parse_sse_json_object

SCHEMA_VERSION = "about-llm.gemini-interactions-replay.v1"
MAX_EVENT_FILE_BYTES = 4 * 1024 * 1024
MAX_FUNCTION_ARGUMENT_BYTES = 64 * 1024
USAGE_TOKEN_FIELDS = frozenset(
    {
        "total_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "total_cached_tokens",
        "total_tool_use_tokens",
        "total_thought_tokens",
    }
)


class GeminiInteractionsProtocolError(ValueError):
    """A sanitized lifecycle or projection error without raw provider data."""


@dataclass(frozen=True)
class GeminiInteractionStep:
    index: int
    step_type: str
    text: str | None
    function_call_id: str | None
    function_name: str | None
    function_arguments: Mapping[str, Any] | None
    projection_complete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "type": self.step_type,
            "text": self.text,
            "function_call": (
                {
                    "id": self.function_call_id,
                    "name": self.function_name,
                    "arguments": (
                        dict(self.function_arguments)
                        if self.function_arguments is not None
                        else None
                    ),
                }
                if self.function_call_id is not None
                else None
            ),
            "projection_complete": self.projection_complete,
        }


@dataclass(frozen=True)
class GeminiInteractionsReceipt:
    interaction_id: str
    model: str
    resource_status: str
    steps: tuple[GeminiInteractionStep, ...]
    usage: Mapping[str, int]
    event_types: tuple[str, ...]
    unknown_event_types: tuple[str, ...]
    unsupported_step_types: tuple[str, ...]
    unsupported_delta_types: tuple[str, ...]
    unprojected_usage_fields: tuple[str, ...]
    sse_byte_framing_checked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "interaction_id": self.interaction_id,
            "model": self.model,
            "stream_terminal_event": "interaction.completed",
            "resource_status": self.resource_status,
            "provider_result_available": self.resource_status == "completed",
            "business_result_verified": False,
            "steps": [step.to_dict() for step in self.steps],
            "usage": dict(self.usage),
            "transport": {
                "done_marker_seen": True,
                "eof_seen": True,
            },
            "compatibility": {
                "unknown_event_types": list(self.unknown_event_types),
                "unsupported_step_types": list(self.unsupported_step_types),
                "unsupported_delta_types": list(self.unsupported_delta_types),
                "unprojected_usage_fields": list(self.unprojected_usage_fields),
            },
            "event_types": list(self.event_types),
            "scope": {
                "offline_sse_events_replayed": True,
                "sse_byte_framing_checked": self.sse_byte_framing_checked,
                "reviewed_lifecycle_checked": True,
                "function_call_count": sum(
                    step.step_type == "function_call" for step in self.steps
                ),
                "all_function_arguments_parsed_as_strict_json_objects": (
                    any(step.step_type == "function_call" for step in self.steps)
                    and all(
                        step.function_arguments is not None
                        for step in self.steps
                        if step.step_type == "function_call"
                    )
                ),
                "remote_gemini_api_called": False,
                "tool_executed": False,
                "provider_usage_or_billing_authenticated": False,
                "complete_interactions_api_supported": False,
            },
        }


@dataclass
class _StepState:
    index: int
    step_type: str
    function_call_id: str | None = None
    function_name: str | None = None
    text_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)
    projection_complete: bool = True


class GeminiInteractionsReplay:
    """Validate one ordered Interactions stream and build a typed projection.

    The replay implements the text ``model_output`` and client-side
    ``function_call`` shapes used by this repository's fixed example. New event,
    step, and delta types are retained in compatibility lists instead of being
    silently treated as supported.
    """

    def __init__(self) -> None:
        self._interaction_id: str | None = None
        self._model: str | None = None
        self._active_step: _StepState | None = None
        self._steps: list[GeminiInteractionStep] = []
        self._resource_status: str | None = None
        self._usage: dict[str, int] = {}
        self._done_seen = False
        self._event_types: list[str] = []
        self._unknown_event_types: list[str] = []
        self._unsupported_step_types: list[str] = []
        self._unsupported_delta_types: list[str] = []
        self._unprojected_usage_fields: list[str] = []

    def consume(self, event: SSEEvent) -> None:
        if self._done_seen:
            raise GeminiInteractionsProtocolError("event received after done marker")
        if event.event == "done":
            if event.data != "[DONE]":
                raise GeminiInteractionsProtocolError("Gemini done event has invalid data")
            if self._resource_status is None:
                raise GeminiInteractionsProtocolError(
                    "done marker arrived before interaction.completed"
                )
            self._done_seen = True
            self._event_types.append("done")
            return
        if event.data == "[DONE]":
            raise GeminiInteractionsProtocolError(
                "Gemini [DONE] marker must use the done event type"
            )

        try:
            payload = parse_sse_json_object(event.data)
        except ValueError as error:
            raise GeminiInteractionsProtocolError("invalid Gemini SSE JSON payload") from error
        event_type = _nonempty_string(payload.get("event_type"), "event_type")
        if event_type != event.event:
            raise GeminiInteractionsProtocolError(
                "Gemini SSE event name and payload event_type differ"
            )
        if self._resource_status is not None:
            raise GeminiInteractionsProtocolError(
                "provider event received after interaction.completed"
            )
        self._event_types.append(event_type)

        if event_type == "interaction.created":
            self._consume_created(payload)
        elif event_type == "interaction.status_update":
            self._consume_status_update(payload)
        elif event_type == "step.start":
            self._consume_step_start(payload)
        elif event_type == "step.delta":
            self._consume_step_delta(payload)
        elif event_type == "step.stop":
            self._consume_step_stop(payload)
        elif event_type == "interaction.completed":
            self._consume_completed(payload)
        elif event_type == "error":
            raise GeminiInteractionsProtocolError(
                "Gemini Interactions stream reported an error"
            )
        else:
            self._unknown_event_types.append(event_type)

    def finish(
        self, *, sse_byte_framing_checked: bool = False
    ) -> GeminiInteractionsReceipt:
        if self._interaction_id is None or self._model is None:
            raise GeminiInteractionsProtocolError(
                "Interactions stream closed before interaction.created"
            )
        if self._active_step is not None:
            raise GeminiInteractionsProtocolError(
                "Interactions stream closed with an active step"
            )
        if self._resource_status is None:
            raise GeminiInteractionsProtocolError(
                "Interactions stream closed before interaction.completed"
            )
        if not self._done_seen:
            raise GeminiInteractionsProtocolError(
                "Interactions stream closed before the done marker"
            )
        return GeminiInteractionsReceipt(
            interaction_id=self._interaction_id,
            model=self._model,
            resource_status=self._resource_status,
            steps=tuple(self._steps),
            usage=dict(self._usage),
            event_types=tuple(self._event_types),
            unknown_event_types=tuple(self._unknown_event_types),
            unsupported_step_types=tuple(self._unsupported_step_types),
            unsupported_delta_types=tuple(self._unsupported_delta_types),
            unprojected_usage_fields=tuple(self._unprojected_usage_fields),
            sse_byte_framing_checked=sse_byte_framing_checked,
        )

    def _consume_created(self, payload: Mapping[str, Any]) -> None:
        if self._interaction_id is not None:
            raise GeminiInteractionsProtocolError("duplicate interaction.created event")
        interaction = _mapping(payload.get("interaction"), "created interaction")
        if interaction.get("object") != "interaction":
            raise GeminiInteractionsProtocolError(
                "created interaction object must be interaction"
            )
        if interaction.get("status") != "in_progress":
            raise GeminiInteractionsProtocolError(
                "created interaction must start in_progress"
            )
        self._interaction_id = _nonempty_string(
            interaction.get("id"), "interaction id"
        )
        self._model = _nonempty_string(interaction.get("model"), "interaction model")

    def _consume_status_update(self, payload: Mapping[str, Any]) -> None:
        self._require_created()
        self._check_interaction_id(payload.get("interaction_id"))
        _nonempty_string(payload.get("status"), "interaction status")

    def _consume_step_start(self, payload: Mapping[str, Any]) -> None:
        self._require_created()
        if self._active_step is not None:
            raise GeminiInteractionsProtocolError(
                "step.start arrived while another step is active"
            )
        index = _index(payload.get("index"), "step index")
        if index != len(self._steps):
            raise GeminiInteractionsProtocolError(
                "step indexes must start at zero and be contiguous in this replay"
            )
        step = _mapping(payload.get("step"), "step")
        step_type = _nonempty_string(step.get("type"), "step type")
        state = _StepState(index=index, step_type=step_type)
        if step_type == "function_call":
            state.function_call_id = _nonempty_string(
                step.get("id"), "function call id"
            )
            state.function_name = _nonempty_string(
                step.get("name"), "function name"
            )
            if step.get("arguments") != {}:
                raise GeminiInteractionsProtocolError(
                    "function_call must start with empty arguments"
                )
        elif step_type != "model_output":
            state.projection_complete = False
            self._unsupported_step_types.append(step_type)
        self._active_step = state

    def _consume_step_delta(self, payload: Mapping[str, Any]) -> None:
        state = self._require_active_step(payload.get("index"))
        delta = _mapping(payload.get("delta"), "step delta")
        delta_type = _nonempty_string(delta.get("type"), "delta type")
        if state.step_type == "model_output" and delta_type == "text":
            state.text_parts.append(_string(delta.get("text"), "text delta"))
            return
        if state.step_type == "function_call" and delta_type == "arguments_delta":
            arguments = _string(delta.get("arguments"), "function arguments delta")
            state.argument_parts.append(arguments)
            if len("".join(state.argument_parts).encode("utf-8")) > MAX_FUNCTION_ARGUMENT_BYTES:
                raise GeminiInteractionsProtocolError(
                    "function arguments exceed the replay byte limit"
                )
            return
        state.projection_complete = False
        self._unsupported_delta_types.append(f"{state.step_type}:{delta_type}")

    def _consume_step_stop(self, payload: Mapping[str, Any]) -> None:
        state = self._require_active_step(payload.get("index"))
        arguments: Mapping[str, Any] | None = None
        if state.step_type == "function_call" and state.projection_complete:
            raw_arguments = "".join(state.argument_parts) or "{}"
            try:
                arguments = parse_sse_json_object(raw_arguments)
            except ValueError as error:
                raise GeminiInteractionsProtocolError(
                    "function arguments are not a strict JSON object"
                ) from error
        self._steps.append(
            GeminiInteractionStep(
                index=state.index,
                step_type=state.step_type,
                text=(
                    "".join(state.text_parts)
                    if state.step_type == "model_output"
                    else None
                ),
                function_call_id=state.function_call_id,
                function_name=state.function_name,
                function_arguments=arguments,
                projection_complete=state.projection_complete,
            )
        )
        self._active_step = None

    def _consume_completed(self, payload: Mapping[str, Any]) -> None:
        self._require_created()
        if self._active_step is not None:
            raise GeminiInteractionsProtocolError(
                "interaction.completed arrived with an active step"
            )
        interaction = _mapping(payload.get("interaction"), "completed interaction")
        self._check_interaction_id(interaction.get("id"))
        raw_model = interaction.get("model")
        if raw_model is not None:
            model = _nonempty_string(raw_model, "interaction model")
            if model != self._model:
                raise GeminiInteractionsProtocolError(
                    "interaction model changed during the stream"
                )
        status = _nonempty_string(interaction.get("status"), "resource status")
        if status == "in_progress":
            raise GeminiInteractionsProtocolError(
                "interaction.completed cannot carry in_progress status"
            )
        usage = _mapping(interaction.get("usage"), "interaction usage")
        self._usage = {
            name: _token_count(value, name)
            for name, value in usage.items()
            if name in USAGE_TOKEN_FIELDS
        }
        self._unprojected_usage_fields = sorted(set(usage) - USAGE_TOKEN_FIELDS)
        self._resource_status = status

    def _require_created(self) -> None:
        if self._interaction_id is None:
            raise GeminiInteractionsProtocolError(
                "Interactions event arrived before interaction.created"
            )

    def _check_interaction_id(self, value: Any) -> None:
        interaction_id = _nonempty_string(value, "interaction id")
        if interaction_id != self._interaction_id:
            raise GeminiInteractionsProtocolError(
                "interaction id changed during the stream"
            )

    def _require_active_step(self, raw_index: Any) -> _StepState:
        if self._active_step is None:
            raise GeminiInteractionsProtocolError(
                "step event references no active step"
            )
        index = _index(raw_index, "step index")
        if index != self._active_step.index:
            raise GeminiInteractionsProtocolError(
                "step event index does not match the active step"
            )
        return self._active_step


def replay_gemini_interactions_sse(
    raw: bytes, *, chunk_size: int = 17
) -> GeminiInteractionsReceipt:
    """Replay fixed SSE bytes through framing and Interactions lifecycle state."""
    if not isinstance(raw, bytes):
        raise TypeError("raw Interactions SSE must be bytes")
    if len(raw) > MAX_EVENT_FILE_BYTES:
        raise GeminiInteractionsProtocolError("Interactions SSE file is too large")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    decoder = SSEDecoder(max_total_bytes=MAX_EVENT_FILE_BYTES)
    replay = GeminiInteractionsReplay()
    try:
        for offset in range(0, len(raw), chunk_size):
            for event in decoder.feed(raw[offset : offset + chunk_size]):
                replay.consume(event)
        for event in decoder.finish():
            replay.consume(event)
    except GeminiInteractionsProtocolError:
        raise
    except (TypeError, ValueError) as error:
        raise GeminiInteractionsProtocolError(
            "invalid Gemini Interactions SSE framing"
        ) from error
    return replay.finish(sse_byte_framing_checked=True)


def load_gemini_interactions_sse(path: Path) -> GeminiInteractionsReceipt:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise GeminiInteractionsProtocolError(
            "cannot inspect Gemini Interactions SSE file"
        ) from error
    if size > MAX_EVENT_FILE_BYTES:
        raise GeminiInteractionsProtocolError("Interactions SSE file is too large")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise GeminiInteractionsProtocolError(
            "cannot read Gemini Interactions SSE file"
        ) from error
    return replay_gemini_interactions_sse(raw)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeminiInteractionsProtocolError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GeminiInteractionsProtocolError(f"{label} must be a non-empty string")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise GeminiInteractionsProtocolError(f"{label} must be a string")
    return value


def _index(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GeminiInteractionsProtocolError(
            f"{label} must be a non-negative integer"
        )
    return value


def _token_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise GeminiInteractionsProtocolError(
            f"usage {label} must be a non-negative integer"
        )
    return value
