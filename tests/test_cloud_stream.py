from __future__ import annotations

import json

import pytest

from about_llm.inference.sse import SSEDecoder, SSEEvent
from about_llm.integrations.cloud_stream import (
    AnthropicTextStream,
    GeminiGenerateContentTextStream,
    OpenAICompatibleTextStream,
    StreamProtocolError,
)


def _event(data: object, *, event: str = "message") -> SSEEvent:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return SSEEvent(payload, event, "", None)


def test_openai_text_usage_finish_and_done_are_distinct() -> None:
    state = OpenAICompatibleTextStream()
    text = state.consume(
        _event({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    )
    final = state.consume(
        _event(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )
    )
    done = state.consume(_event("[DONE]"))
    state.finish()
    assert text[0].text == "hi"
    assert [item.kind for item in final] == ["usage", "finish"]
    assert done[0].kind == "transport_end"


def test_openai_rejects_tool_delta_and_truncated_done() -> None:
    state = OpenAICompatibleTextStream()
    with pytest.raises(StreamProtocolError, match="non-text"):
        state.consume(
            _event({"choices": [{"delta": {"tool_calls": []}, "index": 0}]})
        )
    with pytest.raises(StreamProtocolError, match=r"without \[DONE\]"):
        state.finish()


def test_anthropic_text_block_state_machine() -> None:
    state = AnthropicTextStream()
    start = state.consume(
        _event(
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
            event="message_start",
        )
    )
    state.consume(
        _event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            event="content_block_start",
        )
    )
    delta = state.consume(
        _event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            event="content_block_delta",
        )
    )
    state.consume(_event({"type": "content_block_stop", "index": 0}, event="content_block_stop"))
    final = state.consume(
        _event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            event="message_delta",
        )
    )
    stopped = state.consume(_event({"type": "message_stop"}, event="message_stop"))
    state.finish()
    assert start[0].input_tokens == 3
    assert delta[0].text == "hello"
    assert [update.kind for update in final] == ["usage", "finish"]
    assert stopped[0].kind == "transport_end"


def test_anthropic_rejects_event_type_mismatch_and_inactive_delta() -> None:
    state = AnthropicTextStream()
    with pytest.raises(StreamProtocolError, match="differ"):
        state.consume(_event({"type": "ping"}, event="message_start"))
    state.consume(
        _event({"type": "message_start", "message": {}}, event="message_start")
    )
    with pytest.raises(StreamProtocolError, match="inactive"):
        state.consume(
            _event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "x"},
                },
                event="content_block_delta",
            )
        )


def test_anthropic_message_stop_requires_finish_reason() -> None:
    state = AnthropicTextStream()
    state.consume(
        _event({"type": "message_start", "message": {}}, event="message_start")
    )
    with pytest.raises(StreamProtocolError, match="before stop_reason"):
        state.consume(_event({"type": "message_stop"}, event="message_stop"))


def test_gemini_text_usage_and_eof_finish() -> None:
    state = GeminiGenerateContentTextStream()
    updates = state.consume(
        _event(
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "hello"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
            }
        )
    )
    end = state.finish()
    assert [update.kind for update in updates] == ["text", "finish", "usage"]
    assert end[0].kind == "transport_end"


def test_gemini_rejects_nontext_part_and_missing_finish() -> None:
    state = GeminiGenerateContentTextStream()
    with pytest.raises(StreamProtocolError, match="non-text"):
        state.consume(
            _event({"candidates": [{"content": {"parts": [{"functionCall": {}}]}}]})
        )
    with pytest.raises(StreamProtocolError, match="finishReason"):
        GeminiGenerateContentTextStream().finish()


def test_byte_framing_composes_with_provider_state() -> None:
    wire = (
        'data: {"choices":[{"index":0,"delta":{"content":"你"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    decoder = SSEDecoder()
    state = OpenAICompatibleTextStream()
    updates = []
    for offset in range(0, len(wire), 2):
        for event in decoder.feed(wire[offset : offset + 2]):
            updates.extend(state.consume(event))
    decoder.finish()
    state.finish()
    assert [update.kind for update in updates] == ["text", "finish", "transport_end"]
    assert updates[0].text == "你"
