from __future__ import annotations

import pytest

from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line


def test_parse_sse_json_done_and_ignored_lines() -> None:
    assert parse_sse_data_line("") is None
    assert parse_sse_data_line(": keepalive") is None
    assert parse_sse_data_line("event: message") is None
    assert parse_sse_data_line("data: [DONE]") is STREAM_FINISHED
    assert parse_sse_data_line('data: {"choices": []}') == {"choices": []}


def test_parse_sse_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid SSE JSON"):
        parse_sse_data_line("data: {broken}")
