"""Minimal parsing for OpenAI-compatible Server-Sent Event data lines."""

from __future__ import annotations

import json
from typing import Any


class StreamFinished:
    """Sentinel returned for the protocol's [DONE] marker."""


STREAM_FINISHED = StreamFinished()


def parse_sse_data_line(line: str) -> dict[str, Any] | StreamFinished | None:
    """Parse one SSE line, ignoring comments, event fields, and blank lines."""
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or not stripped.startswith("data:"):
        return None
    payload = stripped.removeprefix("data:").strip()
    if payload == "[DONE]":
        return STREAM_FINISHED
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid SSE JSON payload: {payload[:80]!r}") from error
    if not isinstance(value, dict):
        raise ValueError("SSE data payload must be a JSON object")
    return value
