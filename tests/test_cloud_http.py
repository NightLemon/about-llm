from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from about_llm.integrations.cloud_api import RequestSpec
from about_llm.integrations.cloud_http import (
    CloudCallError,
    CloudStreamResult,
    HttpExecutorConfig,
    execute_json_request,
    execute_sse_request,
)
from about_llm.integrations.cloud_stream import OpenAICompatibleTextStream, StreamUpdate
from about_llm.integrations.retry import RetryPolicy

URL = "https://api.example/v1/chat/completions"
NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.value += delay


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, delay: float = 0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _request() -> RequestSpec:
    return RequestSpec(
        URL,
        {"model": "model-a", "messages": [{"role": "user", "content": "hello"}]},
        {"Authorization": "Bearer secret", "Content-Type": "application/json"},
    )


def _config(**overrides: Any) -> HttpExecutorConfig:
    values: dict[str, Any] = {
        "allowed_origins": frozenset({"https://api.example"}),
        "deadline_seconds": 10,
        "request_timeout_seconds": 5,
        "max_response_bytes": 1024,
    }
    values.update(overrides)
    return HttpExecutorConfig(**values)


def _execute(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: FakeClock | None = None,
    request: RequestSpec | None = None,
    config: HttpExecutorConfig | None = None,
    policy: RetryPolicy | None = None,
    replay_safe: bool = True,
) -> Any:
    clock = clock or FakeClock()

    async def run() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute_json_request(
                client=client,
                request=request or _request(),
                policy=policy or RetryPolicy(),
                config=config or _config(),
                replay_safe=replay_safe,
                sleep=clock.sleep,
                monotonic=clock,
                wall_clock=lambda: NOW,
                jitter=lambda: 1.0,
            )

    return asyncio.run(run())


def test_executor_returns_strict_json_and_sanitized_trace() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={"answer": "ok"},
            headers={"request-id": "req-123"},
        )

    result = _execute(handler)

    assert result.payload == {"answer": "ok"}
    assert result.status_code == 200
    assert result.attempts[0].request_id == "req-123"
    assert result.attempts[0].retry_decision is None
    assert observed[0].method == "POST"
    assert observed[0].headers["authorization"] == "Bearer secret"
    assert "secret" not in repr(result)


def test_retry_after_is_slept_before_success() -> None:
    clock = FakeClock()
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"answer": "ok"})

    result = _execute(handler, clock=clock)

    assert calls == 2
    assert clock.sleeps == [2]
    assert [trace.status_code for trace in result.attempts] == [503, 200]
    decision = result.attempts[0].retry_decision
    assert decision is not None and decision.retry_after_state == "valid"


@pytest.mark.parametrize("status", [400, 501, 307])
def test_non_retryable_status_and_redirect_are_not_followed(status: int) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, headers={"Location": "https://evil.example/steal"})

    with pytest.raises(CloudCallError) as captured:
        _execute(handler)

    assert captured.value.reason == "not_retryable"
    assert calls == 1
    assert captured.value.attempts[0].status_code == status


def test_connect_failure_can_retry_but_read_timeout_is_uncertain() -> None:
    clock = FakeClock()
    calls = 0

    def connect_then_success(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connect failed", request=request)
        return httpx.Response(200, json={"answer": "ok"})

    result = _execute(connect_then_success, clock=clock)
    assert len(result.attempts) == 2
    assert result.attempts[0].outcome_uncertain is False
    assert result.attempts[0].retry_decision is not None
    assert result.attempts[0].retry_decision.retry is True

    def read_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read failed", request=request)

    with pytest.raises(CloudCallError) as captured:
        _execute(read_timeout)
    assert captured.value.reason == "outcome_uncertain"
    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].outcome_uncertain is True


def test_replay_unsafe_request_stops_on_retryable_status() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    with pytest.raises(CloudCallError) as captured:
        _execute(handler, replay_safe=False)
    assert captured.value.reason == "replay_unsafe"
    assert len(captured.value.attempts) == 1


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b'{"a":1,"a":2}', "application/json"),
        (b'{"value":NaN}', "application/json"),
        (b'{"value":1e9999}', "application/json"),
        (b"[]", "application/json"),
        (b'{"answer":"ok"}', "text/plain"),
    ],
)
def test_success_response_rejects_ambiguous_or_non_object_json(
    body: bytes, content_type: str
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"Content-Type": content_type})

    with pytest.raises(CloudCallError) as captured:
        _execute(handler)
    assert captured.value.reason == "invalid_response"
    assert captured.value.attempts[0].failure_kind == "invalid_response"
    assert captured.value.attempts[0].retry_decision is None


def test_response_size_limit_is_terminal_without_replay() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"answer":"too large"}',
            headers={"Content-Type": "application/json"},
        )

    with pytest.raises(CloudCallError) as captured:
        _execute(handler, config=_config(max_response_bytes=8))
    assert captured.value.reason == "response_too_large"
    assert len(captured.value.attempts) == 1


def test_origin_allowlist_rejects_before_transport_and_normalizes_default_port() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"answer": "ok"})

    result = _execute(
        handler,
        config=_config(allowed_origins=frozenset({"HTTPS://API.EXAMPLE:443/"})),
    )
    assert result.status_code == 200

    disallowed = RequestSpec(
        "https://other.example/v1/messages", {}, {"Authorization": "secret"}
    )
    with pytest.raises(ValueError, match="not in allowed_origins"):
        _execute(handler, request=disallowed)
    assert calls == 1


def test_https_and_query_are_fail_closed_by_default() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        HttpExecutorConfig(allowed_origins=frozenset({"http://api.example"}))

    query_request = RequestSpec(
        URL + "?api_key=secret", {}, {"Content-Type": "application/json"}
    )

    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not run")

    with pytest.raises(ValueError, match="query is disabled"):
        _execute(handler, request=query_request)


@pytest.mark.parametrize(
    "origin",
    [
        "https://api.example/v1",
        "https://user:password@api.example",
        "https://api.example?key=secret",
    ],
)
def test_allowed_origin_must_be_an_origin_not_a_url(origin: str) -> None:
    with pytest.raises(ValueError):
        HttpExecutorConfig(allowed_origins=frozenset({origin}))


def test_deadline_can_expire_before_first_attempt() -> None:
    values = iter((0.0, 2.0))
    calls = 0

    def monotonic() -> float:
        return next(values)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"answer": "ok"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(CloudCallError) as captured:
                await execute_json_request(
                    client=client,
                    request=_request(),
                    policy=RetryPolicy(),
                    config=_config(deadline_seconds=1),
                    replay_safe=True,
                    monotonic=monotonic,
                )
            assert captured.value.reason == "deadline_exhausted"

    asyncio.run(run())
    assert calls == 0


def test_executor_attempt_timeout_is_outcome_uncertain() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json={"answer": "late"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(CloudCallError) as captured:
                await execute_json_request(
                    client=client,
                    request=_request(),
                    policy=RetryPolicy(),
                    config=_config(request_timeout_seconds=0.001),
                    replay_safe=True,
                )
            assert captured.value.reason == "outcome_uncertain"
            assert captured.value.attempts[0].failure_kind == "timeout"
            assert captured.value.attempts[0].outcome_uncertain is True

    asyncio.run(run())


def test_cancellation_is_never_converted_to_retry() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(asyncio.CancelledError):
                await execute_json_request(
                    client=client,
                    request=_request(),
                    policy=RetryPolicy(),
                    config=_config(),
                    replay_safe=True,
                )

    asyncio.run(run())


def _openai_stream_wire() -> bytes:
    return (
        b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
        b"data: [DONE]\n\n"
    )


def test_stream_executor_delivers_updates_and_closes_response() -> None:
    wire = _openai_stream_wire()
    stream = ChunkStream([wire[:7], wire[7:31], wire[31:]])
    observed: list[StreamUpdate] = []

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream", "request-id": "stream-1"},
            stream=stream,
        )

    async def run() -> CloudStreamResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute_sse_request(
                client=client,
                request=_request(),
                state=OpenAICompatibleTextStream(),
                policy=RetryPolicy(),
                config=_config(),
                replay_safe=True,
                on_update=lambda update: _append_update(observed, update),
            )

    result = asyncio.run(run())
    assert result.text == "hello"
    assert result.input_tokens == 2 and result.output_tokens == 1
    assert result.finish_reason == "stop"
    assert result.bytes_received == len(wire)
    assert [update.kind for update in observed] == [
        "text",
        "usage",
        "finish",
        "transport_end",
    ]
    assert stream.closed is True
    assert result.attempts[0].request_id == "stream-1"


async def _append_update(target: list[StreamUpdate], update: StreamUpdate) -> None:
    target.append(update)


def test_stream_executor_retries_status_before_body_only() -> None:
    clock = FakeClock()
    first = ChunkStream([])
    second = ChunkStream([_openai_stream_wire()])
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "1"}, stream=first)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=second
        )

    async def run() -> CloudStreamResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute_sse_request(
                client=client,
                request=_request(),
                state=OpenAICompatibleTextStream(),
                policy=RetryPolicy(),
                config=_config(),
                replay_safe=True,
                sleep=clock.sleep,
                monotonic=clock,
                wall_clock=lambda: NOW,
                jitter=lambda: 1,
            )

    result = asyncio.run(run())
    assert len(result.attempts) == 2
    assert clock.sleeps == [1]
    assert first.closed and second.closed


def test_stream_truncation_and_size_limit_are_terminal_and_close() -> None:
    streams = [
        ChunkStream([b'data: {"choices":[]}\n']),
        ChunkStream([b"data:" + b"x" * 80 + b"\n\n"]),
    ]

    def execute(stream: ChunkStream, *, limited: bool) -> CloudCallError:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=stream
            )

        async def run() -> None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with pytest.raises(CloudCallError) as captured:
                    await execute_sse_request(
                        client=client,
                        request=_request(),
                        state=OpenAICompatibleTextStream(),
                        policy=RetryPolicy(max_attempts=3),
                        config=_config(),
                        replay_safe=True,
                        max_line_bytes=32 if limited else 1024,
                        max_event_bytes=64 if limited else 2048,
                        max_total_bytes=128 if limited else 4096,
                    )
                return captured.value

        return asyncio.run(run())

    truncated = execute(streams[0], limited=False)
    oversized = execute(streams[1], limited=True)
    assert truncated.reason == "invalid_response"
    assert oversized.reason == "response_too_large"
    assert len(truncated.attempts) == len(oversized.attempts) == 1
    assert all(stream.closed for stream in streams)


def test_stream_idle_timeout_and_callback_cancellation_close_response() -> None:
    slow = ChunkStream([_openai_stream_wire()], delay=1)

    def slow_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=slow
        )

    async def timeout_run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow_handler)) as client:
            with pytest.raises(CloudCallError) as captured:
                await execute_sse_request(
                    client=client,
                    request=_request(),
                    state=OpenAICompatibleTextStream(),
                    policy=RetryPolicy(),
                    config=_config(request_timeout_seconds=0.001),
                    replay_safe=True,
                )
            assert captured.value.reason == "outcome_uncertain"

    asyncio.run(timeout_run())
    assert slow.closed

    cancel_stream = ChunkStream([_openai_stream_wire()])

    def cancel_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=cancel_stream
        )

    async def cancel(_: StreamUpdate) -> None:
        raise asyncio.CancelledError

    async def cancel_run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(cancel_handler)) as client:
            with pytest.raises(asyncio.CancelledError):
                await execute_sse_request(
                    client=client,
                    request=_request(),
                    state=OpenAICompatibleTextStream(),
                    policy=RetryPolicy(),
                    config=_config(),
                    replay_safe=True,
                    on_update=cancel,
                )

    asyncio.run(cancel_run())
    assert cancel_stream.closed
