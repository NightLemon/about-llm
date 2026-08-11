"""Strict text-only stream state machines for three cloud API shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from about_llm.inference.sse import SSEEvent, parse_sse_json_object

UpdateKind = Literal["text", "usage", "finish", "transport_end"]


@dataclass(frozen=True)
class StreamUpdate:
    kind: UpdateKind
    text: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class StreamProtocolError(ValueError):
    """A sanitized framing/payload/state error without raw provider data."""


class TextStreamState(Protocol):
    def consume(self, event: SSEEvent) -> tuple[StreamUpdate, ...]: ...

    def finish(self) -> tuple[StreamUpdate, ...]: ...


class OpenAICompatibleTextStream:
    """Text-only, single-choice Chat Completions SSE state."""

    def __init__(self) -> None:
        self._model_finished = False
        self._transport_ended = False

    def consume(self, event: SSEEvent) -> tuple[StreamUpdate, ...]:
        if self._transport_ended:
            raise StreamProtocolError("event received after OpenAI transport end")
        if event.event != "message":
            raise StreamProtocolError("unsupported OpenAI-compatible SSE event type")
        if event.data == "[DONE]":
            if not self._model_finished:
                raise StreamProtocolError("OpenAI transport ended before finish_reason")
            self._transport_ended = True
            return (StreamUpdate("transport_end"),)
        value = parse_sse_json_object(event.data)
        updates: list[StreamUpdate] = []
        usage = _optional_mapping(value.get("usage"), "OpenAI usage")
        if usage is not None:
            updates.extend(
                _usage_update(
                    usage,
                    input_key="prompt_tokens",
                    output_key="completion_tokens",
                )
            )
        choices = value.get("choices")
        if not isinstance(choices, list):
            raise StreamProtocolError("OpenAI choices must be an array")
        if len(choices) > 1:
            raise StreamProtocolError("text-only stream supports one OpenAI choice")
        if choices:
            if self._model_finished:
                raise StreamProtocolError("OpenAI choice received after finish_reason")
            choice = _mapping(choices[0], "OpenAI choice")
            index = choice.get("index", 0)
            if index != 0 or isinstance(index, bool):
                raise StreamProtocolError("OpenAI choice index must be zero")
            delta = _mapping(choice.get("delta"), "OpenAI delta")
            if any(name in delta for name in ("tool_calls", "function_call", "refusal")):
                raise StreamProtocolError("non-text OpenAI delta is outside this adapter")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise StreamProtocolError("OpenAI content delta must be a string")
                if content:
                    updates.append(StreamUpdate("text", text=content))
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if not isinstance(finish_reason, str) or not finish_reason:
                    raise StreamProtocolError("OpenAI finish_reason must be a string")
                if self._model_finished:
                    raise StreamProtocolError("duplicate OpenAI finish_reason")
                self._model_finished = True
                updates.append(StreamUpdate("finish", finish_reason=finish_reason))
        return tuple(updates)

    def finish(self) -> tuple[StreamUpdate, ...]:
        if not self._transport_ended:
            raise StreamProtocolError("OpenAI stream closed without [DONE]")
        return ()


class AnthropicTextStream:
    """Text-block subset of the Anthropic Messages streaming state machine."""

    def __init__(self) -> None:
        self._started = False
        self._model_finished = False
        self._stopped = False
        self._active_blocks: set[int] = set()

    def consume(self, event: SSEEvent) -> tuple[StreamUpdate, ...]:
        if self._stopped:
            raise StreamProtocolError("event received after Anthropic message_stop")
        value = parse_sse_json_object(event.data)
        payload_type = value.get("type")
        if payload_type != event.event:
            raise StreamProtocolError("Anthropic SSE event and payload type differ")
        if payload_type == "ping":
            return ()
        if payload_type == "error":
            raise StreamProtocolError("Anthropic stream reported an error")
        if payload_type == "message_start":
            if self._started:
                raise StreamProtocolError("duplicate Anthropic message_start")
            self._started = True
            message = _mapping(value.get("message"), "Anthropic message")
            usage = _optional_mapping(message.get("usage"), "Anthropic usage")
            return _usage_update(usage, input_key="input_tokens", output_key="output_tokens")
        if not self._started:
            raise StreamProtocolError("Anthropic event arrived before message_start")
        if payload_type == "content_block_start":
            index = _index(value.get("index"), "Anthropic block index")
            if index in self._active_blocks:
                raise StreamProtocolError("duplicate Anthropic content block index")
            block = _mapping(value.get("content_block"), "Anthropic content block")
            if block.get("type") != "text":
                raise StreamProtocolError("non-text Anthropic block is outside this adapter")
            text = block.get("text", "")
            if not isinstance(text, str):
                raise StreamProtocolError("Anthropic block text must be a string")
            self._active_blocks.add(index)
            return (StreamUpdate("text", text=text),) if text else ()
        if payload_type == "content_block_delta":
            index = _index(value.get("index"), "Anthropic block index")
            if index not in self._active_blocks:
                raise StreamProtocolError("Anthropic delta references an inactive block")
            delta = _mapping(value.get("delta"), "Anthropic delta")
            if delta.get("type") != "text_delta":
                raise StreamProtocolError("non-text Anthropic delta is outside this adapter")
            text = delta.get("text")
            if not isinstance(text, str):
                raise StreamProtocolError("Anthropic text delta must be a string")
            return (StreamUpdate("text", text=text),) if text else ()
        if payload_type == "content_block_stop":
            index = _index(value.get("index"), "Anthropic block index")
            if index not in self._active_blocks:
                raise StreamProtocolError("Anthropic stop references an inactive block")
            self._active_blocks.remove(index)
            return ()
        if payload_type == "message_delta":
            delta = _mapping(value.get("delta"), "Anthropic message delta")
            reason = delta.get("stop_reason")
            updates = list(
                _usage_update(
                    _optional_mapping(value.get("usage"), "Anthropic usage"),
                    input_key="input_tokens",
                    output_key="output_tokens",
                )
            )
            if reason is not None:
                if not isinstance(reason, str) or not reason:
                    raise StreamProtocolError("Anthropic stop_reason must be a string")
                if self._model_finished:
                    raise StreamProtocolError("duplicate Anthropic stop_reason")
                self._model_finished = True
                updates.append(StreamUpdate("finish", finish_reason=reason))
            return tuple(updates)
        if payload_type == "message_stop":
            if self._active_blocks:
                raise StreamProtocolError("Anthropic message stopped with active blocks")
            if not self._model_finished:
                raise StreamProtocolError("Anthropic message stopped before stop_reason")
            self._stopped = True
            return (StreamUpdate("transport_end"),)
        raise StreamProtocolError("unknown Anthropic stream event type")

    def finish(self) -> tuple[StreamUpdate, ...]:
        if not self._stopped:
            raise StreamProtocolError("Anthropic stream closed before message_stop")
        return ()


class GeminiGenerateContentTextStream:
    """Text-only, single-candidate ``streamGenerateContent`` SSE subset."""

    def __init__(self) -> None:
        self._observed = False
        self._finished = False

    def consume(self, event: SSEEvent) -> tuple[StreamUpdate, ...]:
        if event.event != "message":
            raise StreamProtocolError("unsupported Gemini SSE event type")
        value = parse_sse_json_object(event.data)
        self._observed = True
        updates: list[StreamUpdate] = []
        candidates = value.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) > 1:
            raise StreamProtocolError("Gemini text stream supports at most one candidate")
        if candidates:
            if self._finished:
                raise StreamProtocolError("Gemini candidate received after finishReason")
            candidate = _mapping(candidates[0], "Gemini candidate")
            index = candidate.get("index", 0)
            if index != 0 or isinstance(index, bool):
                raise StreamProtocolError("Gemini candidate index must be zero")
            content = _optional_mapping(candidate.get("content"), "Gemini content")
            if content is not None:
                parts = content.get("parts", [])
                if not isinstance(parts, list):
                    raise StreamProtocolError("Gemini parts must be an array")
                for raw_part in parts:
                    part = _mapping(raw_part, "Gemini part")
                    if set(part) != {"text"} or not isinstance(part["text"], str):
                        raise StreamProtocolError("non-text Gemini part is outside this adapter")
                    if part["text"]:
                        updates.append(StreamUpdate("text", text=part["text"]))
            reason = candidate.get("finishReason")
            if reason is not None:
                if not isinstance(reason, str) or not reason:
                    raise StreamProtocolError("Gemini finishReason must be a string")
                if self._finished:
                    raise StreamProtocolError("duplicate Gemini finishReason")
                self._finished = True
                updates.append(StreamUpdate("finish", finish_reason=reason))
        usage = _optional_mapping(value.get("usageMetadata"), "Gemini usageMetadata")
        updates.extend(
            _usage_update(
                usage,
                input_key="promptTokenCount",
                output_key="candidatesTokenCount",
            )
        )
        return tuple(updates)

    def finish(self) -> tuple[StreamUpdate, ...]:
        if not self._observed or not self._finished:
            raise StreamProtocolError("Gemini stream closed before a finishReason")
        return (StreamUpdate("transport_end"),)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamProtocolError(f"{label} must be an object")
    return value


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, label)


def _optional_token(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StreamProtocolError("stream usage must contain non-negative integers")
    return value


def _usage_update(
    usage: Mapping[str, Any] | None, *, input_key: str, output_key: str
) -> tuple[StreamUpdate, ...]:
    if usage is None:
        return ()
    input_tokens = _optional_token(usage.get(input_key))
    output_tokens = _optional_token(usage.get(output_key))
    if input_tokens is None and output_tokens is None:
        return ()
    return (
        StreamUpdate(
            "usage", input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


def _index(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StreamProtocolError(f"{label} must be a non-negative integer")
    return value
