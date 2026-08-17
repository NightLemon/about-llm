from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from about_llm.integrations import openai_responses_replay as replay_module
from about_llm.integrations.openai_responses_replay import (
    OpenAIResponsesEventReplay,
    load_response_event_jsonl,
    replay_response_event_file,
    replay_response_events,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "cloud-api-contracts"
FIXTURE = PROJECT / "openai-responses-events.example.jsonl"


def _fixture_events() -> list[dict[str, object]]:
    events, _ = load_response_event_jsonl(FIXTURE)
    return copy.deepcopy(list(events))


def _response(
    *,
    status: str,
    output: list[dict[str, object]],
    error: object = None,
    incomplete_details: object = None,
    usage: object = None,
) -> dict[str, object]:
    return {
        "id": "resp_test",
        "model": "gpt-reviewed-snapshot",
        "status": status,
        "output": output,
        "error": error,
        "incomplete_details": incomplete_details,
        "usage": usage,
    }


def _created() -> dict[str, object]:
    return {
        "type": "response.created",
        "sequence_number": 0,
        "response": _response(status="in_progress", output=[]),
    }


def test_fixed_fixture_replays_text_tool_usage_and_exact_input_identity() -> None:
    receipt = replay_response_event_file(FIXTURE)
    payload = receipt.to_dict()

    assert receipt.response_id == "resp_authored_001"
    assert receipt.model == "gpt-reviewed-snapshot"
    assert receipt.terminal_status == "completed"
    assert receipt.output_text == "天气\N{FULLWIDTH COLON}晴。"
    assert receipt.refusals == ()
    assert receipt.event_count == 15
    assert receipt.output_item_count == 2
    assert (receipt.input_tokens, receipt.output_tokens, receipt.total_tokens) == (
        12,
        9,
        21,
    )
    assert len(receipt.function_calls) == 1
    assert receipt.function_calls[0].arguments == '{"city":"上海"}'
    assert receipt.function_calls[0].arguments_is_strict_object is True
    assert receipt.input_size_bytes == len(FIXTURE.read_bytes())
    assert receipt.input_sha256 is not None
    assert payload["schema_version"] == "about-llm.openai-responses-event-replay.v1"
    assert payload["scope"]["http_sse_or_websocket_transport_executed"] is False


def test_sequence_numbers_are_a_contiguous_local_replay_contract() -> None:
    events = _fixture_events()
    events[5]["sequence_number"] = 99

    with pytest.raises(ValueError, match="start at zero and be contiguous"):
        replay_response_events(events)


@pytest.mark.parametrize("change", ["missing", "extra", "unknown_type"])
def test_event_schema_and_unknown_types_fail_closed(change: str) -> None:
    events = _fixture_events()
    if change == "missing":
        del events[4]["logprobs"]
    elif change == "extra":
        events[4]["unexpected"] = True
    else:
        events[4]["type"] = "response.new_future_event"

    with pytest.raises(ValueError, match=r"fields differ|unsupported Responses event"):
        replay_response_events(events)


def test_accumulated_text_and_terminal_output_are_reconciled() -> None:
    events = _fixture_events()
    events[6]["text"] = "different"
    with pytest.raises(ValueError, match="does not match accumulated deltas"):
        replay_response_events(events)

    events = _fixture_events()
    terminal_response = events[-1]["response"]
    assert isinstance(terminal_response, dict)
    output = terminal_response["output"]
    assert isinstance(output, list)
    assert isinstance(output[0], dict)
    output[0]["status"] = "incomplete"
    with pytest.raises(ValueError, match=r"differs from output_item\.done"):
        replay_response_events(events)


def test_invalid_function_argument_json_is_evidence_not_parser_success() -> None:
    events = _fixture_events()
    events[10]["delta"] = "not "
    events[11]["delta"] = "json"
    events[12]["arguments"] = "not json"
    done_item = events[13]["item"]
    assert isinstance(done_item, dict)
    done_item["arguments"] = "not json"
    terminal = events[14]["response"]
    assert isinstance(terminal, dict)
    output = terminal["output"]
    assert isinstance(output, list) and isinstance(output[1], dict)
    output[1]["arguments"] = "not json"

    receipt = replay_response_events(events)

    assert receipt.function_calls[0].arguments == "not json"
    assert receipt.function_calls[0].arguments_is_strict_object is False


def test_refusal_is_kept_separate_from_output_text() -> None:
    added_message = {
        "id": "msg_refusal",
        "type": "message",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }
    refusal_part = {"type": "refusal", "refusal": ""}
    done_part = {"type": "refusal", "refusal": "cannot comply"}
    done_message = {
        **added_message,
        "status": "completed",
        "content": [done_part],
    }
    events = [
        _created(),
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": added_message,
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": "msg_refusal",
            "content_index": 0,
            "part": refusal_part,
        },
        {
            "type": "response.refusal.delta",
            "sequence_number": 3,
            "output_index": 0,
            "item_id": "msg_refusal",
            "content_index": 0,
            "delta": "cannot comply",
        },
        {
            "type": "response.refusal.done",
            "sequence_number": 4,
            "output_index": 0,
            "item_id": "msg_refusal",
            "content_index": 0,
            "refusal": "cannot comply",
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 5,
            "output_index": 0,
            "item_id": "msg_refusal",
            "content_index": 0,
            "part": done_part,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 6,
            "output_index": 0,
            "item": done_message,
        },
        {
            "type": "response.completed",
            "sequence_number": 7,
            "response": _response(
                status="completed",
                output=[done_message],
                usage={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            ),
        },
    ]

    receipt = replay_response_events(events)

    assert receipt.output_text == ""
    assert receipt.refusals == ("cannot comply",)


@pytest.mark.parametrize(
    ("status", "error", "incomplete", "reason"),
    [
        ("incomplete", None, {"reason": "max_output_tokens"}, "max_output_tokens"),
        ("failed", {"code": "server_error", "message": "failed"}, None, "server_error"),
    ],
)
def test_incomplete_and_failed_are_terminal_but_not_completed(
    status: str, error: object, incomplete: object, reason: str
) -> None:
    events = [
        _created(),
        {
            "type": f"response.{status}",
            "sequence_number": 1,
            "response": _response(
                status=status,
                output=[],
                error=error,
                incomplete_details=incomplete,
            ),
        },
    ]

    receipt = replay_response_events(events)

    assert receipt.terminal_status == status
    assert receipt.terminal_reason == reason


def test_opaque_output_item_is_preserved_without_semantic_projection() -> None:
    added = {"id": "rs_1", "type": "reasoning", "status": "in_progress"}
    done = {"id": "rs_1", "type": "reasoning", "status": "completed"}
    events = [
        _created(),
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": added,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 0,
            "item": done,
        },
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": _response(status="completed", output=[done]),
        },
    ]

    receipt = replay_response_events(events)

    assert receipt.output_item_count == 1
    assert receipt.output_text == ""
    assert receipt.function_calls == ()


def test_lifecycle_rejects_precreated_postterminal_and_unfinished_content() -> None:
    state = OpenAIResponsesEventReplay()
    with pytest.raises(ValueError, match=r"before response\.created"):
        state.consume(
            {
                "type": "response.in_progress",
                "sequence_number": 0,
                "response": _response(status="in_progress", output=[]),
            }
        )

    events = _fixture_events()
    replay = OpenAIResponsesEventReplay()
    for event in events:
        replay.consume(event)
    with pytest.raises(ValueError, match="after terminal"):
        replay.consume(events[-1])

    events = _fixture_events()
    del events[6:8]
    for sequence, event in enumerate(events):
        event["sequence_number"] = sequence
    with pytest.raises(ValueError, match=r"active content|semantic done"):
        replay_response_events(events)


def test_usage_total_must_reconcile() -> None:
    events = _fixture_events()
    terminal = events[-1]["response"]
    assert isinstance(terminal, dict) and isinstance(terminal["usage"], dict)
    terminal["usage"]["total_tokens"] = 999

    with pytest.raises(ValueError, match="total_tokens differs"):
        replay_response_events(events)


def test_jsonl_loader_rejects_ambiguous_invalid_and_truncated_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"type":"a","type":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_response_event_jsonl(path)

    path.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_response_event_jsonl(path)

    path.write_bytes(b'{"value":"\xff"}\n')
    with pytest.raises(ValueError, match="invalid UTF-8"):
        load_response_event_jsonl(path)

    path.write_text('{"value":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="end with a newline"):
        load_response_event_jsonl(path)

    path.write_text('{"value":1}\n\n', encoding="utf-8")
    with pytest.raises(ValueError, match="blank"):
        load_response_event_jsonl(path)

    monkeypatch.setattr(replay_module, "MAX_EVENT_FILE_BYTES", 5)
    with pytest.raises(ValueError, match="exceeds byte limit"):
        load_response_event_jsonl(FIXTURE)


def test_state_snapshots_caller_events_before_later_mutation() -> None:
    events = _fixture_events()
    state = OpenAIResponsesEventReplay()
    state.consume(events[0])
    original = copy.deepcopy(events[0])
    response = events[0]["response"]
    assert isinstance(response, dict)
    response["model"] = "mutated-after-consume"
    for event in events[1:]:
        state.consume(event)
    receipt = state.finish()

    assert receipt.model == original["response"]["model"]


def test_project_cli_emits_machine_readable_scope_receipt() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "openai_responses_replay.py"),
            "--events",
            str(FIXTURE),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["terminal_status"] == "completed"
    assert payload["function_calls"][0]["arguments_is_strict_object"] is True
    assert payload["scope"]["openai_sdk_or_remote_api_executed"] is False
