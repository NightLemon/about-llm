from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.inference.sse import SSEEvent
from about_llm.integrations.gemini_interactions_replay import (
    GeminiInteractionsProtocolError,
    GeminiInteractionsReplay,
    load_gemini_interactions_sse,
)

pytestmark = pytest.mark.contract

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "cloud-api-contracts"
    / "gemini-interactions-function-call.example.sse"
)


def _event(event_type: str, **payload: object) -> SSEEvent:
    value = {**payload, "event_type": event_type}
    return SSEEvent(json.dumps(value, ensure_ascii=False), event_type, "", None)


def _created() -> SSEEvent:
    return _event(
        "interaction.created",
        interaction={
            "id": "v1_test",
            "status": "in_progress",
            "object": "interaction",
            "model": "gemini-test",
        },
    )


def _completed(status: str = "completed") -> SSEEvent:
    return _event(
        "interaction.completed",
        interaction={
            "id": "v1_test",
            "status": status,
            "object": "interaction",
            "model": "gemini-test",
            "usage": {"total_input_tokens": 2, "total_output_tokens": 1},
        },
    )


def test_fixed_function_call_ends_the_stream_but_still_requires_action() -> None:
    receipt = load_gemini_interactions_sse(FIXTURE)
    payload = receipt.to_dict()

    assert payload["stream_terminal_event"] == "interaction.completed"
    assert payload["resource_status"] == "requires_action"
    assert payload["provider_result_available"] is False
    assert payload["business_result_verified"] is False
    assert payload["scope"]["sse_byte_framing_checked"] is True
    assert payload["scope"]["function_call_count"] == 1
    assert payload["transport"] == {
        "done_marker_seen": True,
        "eof_seen": True,
    }
    assert payload["compatibility"]["unprojected_usage_fields"] == [
        "input_tokens_by_modality"
    ]
    assert payload["steps"] == [
        {
            "index": 0,
            "type": "function_call",
            "text": None,
            "function_call": {
                "id": "call_weather_001",
                "name": "lookup_weather",
                "arguments": {"city": "上海"},
            },
            "projection_complete": True,
        }
    ]


def test_named_event_must_match_payload_event_type() -> None:
    replay = GeminiInteractionsReplay()
    mismatched = SSEEvent(
        '{"event_type":"interaction.status_update"}',
        "interaction.created",
        "",
        None,
    )
    with pytest.raises(GeminiInteractionsProtocolError, match="differ"):
        replay.consume(mismatched)


def test_done_requires_typed_terminal_and_no_event_may_follow_it() -> None:
    replay = GeminiInteractionsReplay()
    replay.consume(_created())
    with pytest.raises(
        GeminiInteractionsProtocolError, match=r"before interaction\.completed"
    ):
        replay.consume(SSEEvent("[DONE]", "done", "", None))

    replay.consume(_completed())
    replay.consume(SSEEvent("[DONE]", "done", "", None))
    with pytest.raises(GeminiInteractionsProtocolError, match="after done marker"):
        replay.consume(
            _event(
                "interaction.status_update",
                interaction_id="v1_test",
                status="completed",
            )
        )


def test_unknown_event_is_reported_without_breaking_a_valid_stream() -> None:
    replay = GeminiInteractionsReplay()
    replay.consume(_created())
    replay.consume(_event("interaction.progress_hint", interaction_id="v1_test"))
    replay.consume(_completed())
    replay.consume(SSEEvent("[DONE]", "done", "", None))

    receipt = replay.finish()
    assert receipt.unknown_event_types == ("interaction.progress_hint",)
    assert receipt.resource_status == "completed"
    payload = receipt.to_dict()
    assert payload["provider_result_available"] is True
    assert payload["business_result_verified"] is False
    assert payload["scope"]["sse_byte_framing_checked"] is False


def test_terminal_snapshot_may_omit_model_already_fixed_by_created_event() -> None:
    replay = GeminiInteractionsReplay()
    replay.consume(_created())
    replay.consume(
        _event(
            "interaction.completed",
            interaction={
                "id": "v1_test",
                "status": "completed",
                "usage": {"total_input_tokens": 2, "total_output_tokens": 1},
            },
        )
    )
    replay.consume(SSEEvent("[DONE]", "done", "", None))

    receipt = replay.finish()
    assert receipt.model == "gemini-test"


def test_function_arguments_require_strict_object_json() -> None:
    replay = GeminiInteractionsReplay()
    replay.consume(_created())
    replay.consume(
        _event(
            "step.start",
            index=0,
            step={
                "type": "function_call",
                "id": "call_1",
                "name": "lookup",
                "arguments": {},
            },
        )
    )
    replay.consume(
        _event(
            "step.delta",
            index=0,
            delta={
                "type": "arguments_delta",
                "arguments": '{"city":"上海","city":"北京"}',
            },
        )
    )
    with pytest.raises(GeminiInteractionsProtocolError, match="strict JSON object"):
        replay.consume(_event("step.stop", index=0))


def test_terminal_rejects_an_active_step_and_invalid_usage() -> None:
    replay = GeminiInteractionsReplay()
    replay.consume(_created())
    replay.consume(
        _event("step.start", index=0, step={"type": "model_output"})
    )
    with pytest.raises(GeminiInteractionsProtocolError, match="active step"):
        replay.consume(_completed())

    second = GeminiInteractionsReplay()
    second.consume(_created())
    invalid = _event(
        "interaction.completed",
        interaction={
            "id": "v1_test",
            "status": "completed",
            "model": "gemini-test",
            "usage": {"total_tokens": True},
        },
    )
    with pytest.raises(GeminiInteractionsProtocolError, match="non-negative integer"):
        second.consume(invalid)
