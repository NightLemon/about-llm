from __future__ import annotations

import pytest

from about_llm.inference.sse import (
    STREAM_FINISHED,
    SSEDecoder,
    parse_sse_data_line,
    parse_sse_json_object,
)

pytestmark = pytest.mark.contract


def test_parse_sse_json_done_and_ignored_lines() -> None:
    assert parse_sse_data_line("") is None
    assert parse_sse_data_line(": keepalive") is None
    assert parse_sse_data_line("event: message") is None
    assert parse_sse_data_line("data: [DONE]") is STREAM_FINISHED
    assert parse_sse_data_line('data: {"choices": []}') == {"choices": []}


def test_parse_sse_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid SSE JSON"):
        parse_sse_data_line("data: {broken}")


def test_decoder_handles_arbitrary_utf8_chunks_crlf_and_multiline_data() -> None:
    decoder = SSEDecoder()
    wire = (
        "\ufeff: keepalive\r\n"
        "id: event-1\r\n"
        "retry: 1500\r\n"
        "event: custom\r\n"
        'data: {"text":"你\r\n'
        'data: 好"}\r\n\r\n'
    ).encode("utf-8")
    events = []
    for byte in wire:
        events.extend(decoder.feed(bytes([byte])))
    decoder.finish()

    assert len(events) == 1
    assert events[0].event == "custom"
    assert events[0].last_event_id == "event-1"
    assert events[0].retry_ms == 1500
    assert events[0].data == '{"text":"你\n好"}'


def test_decoder_accepts_cr_only_blank_line_at_eof() -> None:
    decoder = SSEDecoder()
    assert decoder.feed(b"data:x\r\r") == ()
    events = decoder.finish()
    assert len(events) == 1 and events[0].data == "x"


def test_decoder_preserves_one_leading_data_space_and_last_event_id() -> None:
    decoder = SSEDecoder()
    events = decoder.feed(
        b"id: first\ndata:  two spaces\n\nid:\ndata:x\n\n"
    )
    decoder.finish()
    assert [event.data for event in events] == [" two spaces", "x"]
    assert [event.last_event_id for event in events] == ["first", ""]


@pytest.mark.parametrize(
    "payload",
    [
        b"data: {\"x\":1}",
        b"data: {\"x\":1}\n",
        b"data: \xff\n\n",
    ],
)
def test_decoder_rejects_truncated_event_or_invalid_utf8(payload: bytes) -> None:
    decoder = SSEDecoder()
    if b"\xff" in payload:
        with pytest.raises(ValueError, match="invalid UTF-8"):
            decoder.feed(payload)
    else:
        decoder.feed(payload)
        with pytest.raises(ValueError, match="unterminated"):
            decoder.finish()


def test_decoder_enforces_line_event_and_total_byte_limits() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        SSEDecoder(max_line_bytes=10, max_event_bytes=9, max_total_bytes=20)

    line = SSEDecoder(max_line_bytes=4, max_event_bytes=8, max_total_bytes=8)
    with pytest.raises(ValueError, match="line"):
        line.feed(b"12345")

    event = SSEDecoder(max_line_bytes=8, max_event_bytes=8, max_total_bytes=20)
    with pytest.raises(ValueError, match="event"):
        event.feed(b"data:a\ndata:b\n")

    total = SSEDecoder(max_line_bytes=4, max_event_bytes=4, max_total_bytes=4)
    with pytest.raises(ValueError, match="total"):
        total.feed(b"12345")


@pytest.mark.parametrize(
    "payload",
    [
        '{"x":NaN}',
        '{"x":1e9999}',
        '{"x":1,"x":2}',
        "[]",
    ],
)
def test_sse_json_rejects_nonfinite_duplicate_and_nonobject(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_sse_json_object(payload)
