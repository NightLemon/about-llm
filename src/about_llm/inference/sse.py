"""Bounded Server-Sent Events framing plus a legacy single-line JSON helper."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast


class StreamFinished:
    """Sentinel returned for an OpenAI-compatible ``[DONE]`` data marker."""


STREAM_FINISHED = StreamFinished()


class SSEProtocolError(ValueError):
    """Malformed UTF-8/framing or a stream that ended mid-event."""


class SSELimitError(ValueError):
    """A configured line, event, or total-byte limit was exceeded."""


@dataclass(frozen=True)
class SSEEvent:
    """One dispatched SSE event after byte framing and field folding."""

    data: str
    event: str
    last_event_id: str
    retry_ms: int | None


class SSEDecoder:
    """Incrementally frame UTF-8 SSE bytes with explicit resource bounds.

    Events are emitted only after a blank line. ``finish`` rejects a truncated
    final line or event rather than treating an interrupted response as complete.
    The decoder does not reconnect; reconnecting a model generation could replay
    work or billing and must be a provider-specific decision.
    """

    def __init__(
        self,
        *,
        max_line_bytes: int = 64 * 1024,
        max_event_bytes: int = 1024 * 1024,
        max_total_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        for name, value in (
            ("max_line_bytes", max_line_bytes),
            ("max_event_bytes", max_event_bytes),
            ("max_total_bytes", max_total_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_event_bytes < max_line_bytes:
            raise ValueError("max_event_bytes must be greater than or equal to max_line_bytes")
        if max_total_bytes < max_event_bytes:
            raise ValueError("max_total_bytes must be greater than or equal to max_event_bytes")
        self._max_line_bytes = max_line_bytes
        self._max_event_bytes = max_event_bytes
        self._max_total_bytes = max_total_bytes
        self._buffer = bytearray()
        self._total_bytes = 0
        self._event_bytes = 0
        self._first_line = True
        self._finished = False
        self._data_lines: list[str] = []
        self._event_name = ""
        self._last_event_id = ""
        self._retry_ms: int | None = None

    def feed(self, chunk: bytes) -> tuple[SSEEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished SSE decoder")
        if not isinstance(chunk, bytes):
            raise TypeError("SSE chunks must be bytes")
        self._total_bytes += len(chunk)
        if self._total_bytes > self._max_total_bytes:
            raise SSELimitError("SSE stream exceeds max_total_bytes")
        self._buffer.extend(chunk)
        events: list[SSEEvent] = []
        while True:
            boundary = _find_line_boundary(self._buffer)
            if boundary is None:
                if len(self._buffer) > self._max_line_bytes:
                    raise SSELimitError("SSE line exceeds max_line_bytes")
                break
            line_end, terminator_bytes = boundary
            line = bytes(self._buffer[:line_end])
            del self._buffer[: line_end + terminator_bytes]
            if len(line) > self._max_line_bytes:
                raise SSELimitError("SSE line exceeds max_line_bytes")
            self._event_bytes += len(line) + terminator_bytes
            if self._event_bytes > self._max_event_bytes:
                raise SSELimitError("SSE event exceeds max_event_bytes")
            event = self._process_line(line)
            if event is not None:
                events.append(event)
        return tuple(events)

    def finish(self) -> tuple[SSEEvent, ...]:
        """Mark EOF; reject any line/event not terminated by an SSE blank line."""
        if self._finished:
            raise RuntimeError("SSE decoder is already finished")
        self._finished = True
        events: list[SSEEvent] = []
        if self._buffer.endswith(b"\r"):
            raw_line = bytes(self._buffer[:-1])
            self._buffer.clear()
            if len(raw_line) > self._max_line_bytes:
                raise SSELimitError("SSE line exceeds max_line_bytes")
            self._event_bytes += len(raw_line) + 1
            if self._event_bytes > self._max_event_bytes:
                raise SSELimitError("SSE event exceeds max_event_bytes")
            event = self._process_line(raw_line)
            if event is not None:
                events.append(event)
        if self._buffer:
            raise SSEProtocolError("SSE stream ended with an unterminated line")
        if self._data_lines or self._event_name:
            raise SSEProtocolError("SSE stream ended with an unterminated event")
        return tuple(events)

    def _process_line(self, raw_line: bytes) -> SSEEvent | None:
        encoding = "utf-8-sig" if self._first_line else "utf-8"
        self._first_line = False
        try:
            line = raw_line.decode(encoding, errors="strict")
        except UnicodeDecodeError as error:
            raise SSEProtocolError("SSE stream contains invalid UTF-8") from error
        if line == "":
            event: SSEEvent | None = None
            if self._data_lines:
                event = SSEEvent(
                    data="\n".join(self._data_lines),
                    event=self._event_name or "message",
                    last_event_id=self._last_event_id,
                    retry_ms=self._retry_ms,
                )
            self._data_lines = []
            self._event_name = ""
            self._event_bytes = 0
            return event
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event_name = value
        elif field == "id" and "\x00" not in value:
            self._last_event_id = value
        elif field == "retry" and value.isascii() and value.isdigit():
            self._retry_ms = int(value)
        return None


def parse_sse_data_line(line: str) -> dict[str, Any] | StreamFinished | None:
    """Parse one OpenAI-compatible SSE data line.

    This compatibility helper does not frame events or join multiple ``data``
    fields. New byte-stream consumers should use :class:`SSEDecoder` first.
    """
    if not isinstance(line, str):
        raise TypeError("SSE line must be a string")
    line = line.removesuffix("\n").removesuffix("\r")
    if not line or line.startswith(":"):
        return None
    field, separator, payload = line.partition(":")
    if field != "data" or not separator:
        return None
    if payload.startswith(" "):
        payload = payload[1:]
    if payload == "[DONE]":
        return STREAM_FINISHED
    return parse_sse_json_object(payload)


def parse_sse_json_object(data: str) -> dict[str, Any]:
    """Parse strict finite object JSON from a fully framed SSE event."""

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON constant")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON float")
        return parsed

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON object key")
            result[name] = value
        return result

    try:
        value = json.loads(
            data,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid SSE JSON payload") from error
    if not isinstance(value, dict):
        raise ValueError("SSE data payload must be a JSON object")
    return cast(dict[str, Any], value)


def _find_line_boundary(buffer: bytearray) -> tuple[int, int] | None:
    for index, value in enumerate(buffer):
        if value == 0x0A:
            return index, 1
        if value == 0x0D:
            if index + 1 == len(buffer):
                return None
            return index, 2 if buffer[index + 1] == 0x0A else 1
    return None
