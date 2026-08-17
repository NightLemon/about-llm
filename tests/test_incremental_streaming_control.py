from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from about_llm.inference.incremental_streaming_control import (
    INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY,
    SCRIPTED_BACKEND_FINGERPRINT,
    load_and_verify_incremental_streaming_report,
    run_incremental_streaming_control,
    verify_incremental_streaming_report,
)
from about_llm.inference.openai_reference import (
    CHAT_COMPLETIONS_PATH,
    ChatCompletionRequest,
    ChatMessage,
    IncrementalOpenAIReferenceService,
    IncrementalTokenDelta,
    build_incremental_reference_app,
)
from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "inference-serving"
RECORDED_REPORT = PROJECT / "incremental-streaming.recorded-report.json"


class FixtureIncrementalBackend:
    model_id = "fixture/incremental"
    backend_fingerprint = "sha256:" + "b" * 64

    def __init__(self, events: tuple[IncrementalTokenDelta, ...]) -> None:
        self.events = events
        self.calls = 0

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[IncrementalTokenDelta]:
        self.calls += 1
        assert request.model == self.model_id
        for event in self.events:
            yield event
            await asyncio.sleep(0)

    def audit_projection(self) -> Mapping[str, Any]:
        return {"implementation": "fixture-async-iterator", "calls": self.calls}


def _body(*, stream: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "fixture/incremental",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="fixture/incremental",
        messages=(ChatMessage(role="user", content="hello"),),
        max_tokens=2,
        stream=True,
        include_usage=True,
    )


def _rehash(report: dict[str, Any]) -> None:
    report.pop("report_fingerprint", None)
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(report)


def test_incremental_token_delta_validates_closed_value_contract() -> None:
    valid = IncrementalTokenDelta("字", 0, 1, "length")
    assert valid.token_id == 0
    invalid = (
        ("", 1, 1, None),
        ("x", True, 1, None),
        ("x", -1, 1, None),
        ("x", 1, True, None),
        ("x", 1, 0, None),
        ("x", 1, 1, "other"),
    )
    for arguments in invalid:
        with pytest.raises(ValueError):
            IncrementalTokenDelta(*arguments)


def test_incremental_app_rejects_nonstream_without_calling_backend() -> None:
    async def exercise() -> None:
        backend = FixtureIncrementalBackend(
            (IncrementalTokenDelta("x", 1, 3, "stop"),)
        )
        app = build_incremental_reference_app(
            backend,
            bearer_token="t" * 32,
            maximum_new_tokens=2,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers={"Authorization": "Bearer " + "t" * 32},
                json=_body(stream=False),
            )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "stream_required"
        assert backend.calls == 0
        assert app.state.reference_service.audit_projection()["accepted_requests"] == 0

    asyncio.run(exercise())


def test_incremental_app_emits_content_finish_usage_and_done() -> None:
    async def exercise() -> None:
        backend = FixtureIncrementalBackend(
            (
                IncrementalTokenDelta("甲", 10, 3),
                IncrementalTokenDelta("🙂", 11, 3, "stop"),
            )
        )
        app = build_incremental_reference_app(
            backend,
            bearer_token="t" * 32,
            maximum_new_tokens=2,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                CHAT_COMPLETIONS_PATH,
                headers={"Authorization": "Bearer " + "t" * 32},
                json=_body(),
            )
        assert response.status_code == 200
        events = [
            parse_sse_data_line(line)
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert events[-1] is STREAM_FINISHED
        payloads = [event for event in events[:-1] if isinstance(event, dict)]
        content = [
            event["choices"][0]["delta"]["content"]
            for event in payloads
            if event["choices"] and "content" in event["choices"][0]["delta"]
        ]
        assert content == ["甲", "🙂"]
        assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
        assert payloads[-1]["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        audit = app.state.reference_service.audit_projection()
        assert audit["completed_incremental_streams"] == 1
        assert audit["cancelled_incremental_streams"] == 0
        assert audit["active_incremental_streams"] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "events",
    [
        (
            IncrementalTokenDelta("a", 1, 3, "stop"),
            IncrementalTokenDelta("b", 2, 3, "stop"),
        ),
        (
            IncrementalTokenDelta("a", 1, 3),
            IncrementalTokenDelta("b", 2, 4, "stop"),
        ),
        (IncrementalTokenDelta("a", 1, 3),),
    ],
)
def test_incremental_service_rejects_backend_protocol_drift(
    events: tuple[IncrementalTokenDelta, ...],
) -> None:
    async def exercise() -> None:
        service = IncrementalOpenAIReferenceService(
            FixtureIncrementalBackend(events),
            bearer_token="t" * 32,
            maximum_new_tokens=2,
        )
        await service._admission.acquire()
        service.active_incremental_streams = 1
        with pytest.raises(RuntimeError):
            async for _ in service._incremental_response(_request()):
                pass
        assert service.failed_backend_requests == 1
        assert service.active_incremental_streams == 0
        assert not service._admission.locked()

    asyncio.run(exercise())


def test_real_loopback_disconnect_control_and_recorded_report_verify() -> None:
    live = run_incremental_streaming_control()
    assert live["disconnect_stream"][
        "postclose_backend_asyncio_cancelled_error_observed"
    ] is True
    assert live["disconnect_stream"]["postclose_backend_emitted_token_ids"] == [201]
    assert live["complete_stream"]["backend_completion_token_ids"] == [101, 102, 103]
    assert live["scope"]["kv_or_gpu_resource_release_proven"] is False
    assert verify_incremental_streaming_report(live) == live

    recorded = load_and_verify_incremental_streaming_report(RECORDED_REPORT)
    assert recorded["report_fingerprint"] == (
        "sha256:258468229bb14af198f7a39a68999fb41375a9256ee7aa4b2c2c0e80f42b5d00"
    )
    assert recorded["backend_fingerprint"] == SCRIPTED_BACKEND_FINGERPRINT
    assert recorded["evidence_boundary"] == INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["scope"].update({"vllm_or_cuda_executed": True}),
        lambda report: report["audit"].update({"cancelled_incremental_streams": 0}),
        lambda report: report["disconnect_stream"].update(
            {"postclose_backend_emitted_token_ids": [201, 202]}
        ),
        lambda report: report.update({"unknown": True}),
    ],
)
def test_report_verifier_rejects_cooperatively_rehashed_drift(mutate: Any) -> None:
    report = copy.deepcopy(json.loads(RECORDED_REPORT.read_text(encoding="utf-8")))
    mutate(report)
    _rehash(report)
    with pytest.raises(ValueError):
        verify_incremental_streaming_report(report)


def test_recorded_loader_rejects_duplicate_nonfinite_and_invalid_utf8(
    tmp_path: Path,
) -> None:
    invalid_payloads = (
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b"{\xff}",
    )
    for index, payload in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="strict JSON"):
            load_and_verify_incremental_streaming_report(path)


def test_project_cli_verifies_recorded_report_without_starting_server() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "incremental_streaming_control.py"),
            "--verify",
            str(RECORDED_REPORT),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["report_fingerprint"].endswith("0e80f42b5d00")
    assert report["scope"]["transformers_generation_thread_cancellation_proven"] is False
