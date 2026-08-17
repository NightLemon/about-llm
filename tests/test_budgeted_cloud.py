from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from about_llm.integrations.budgeted_cloud import (
    BudgetedCloudCallError,
    BudgetedCloudRetryError,
    execute_budgeted_json_request,
    execute_budgeted_json_request_with_retry,
)
from about_llm.integrations.cloud_api import (
    ChatMessage,
    build_openai_compatible_request,
    parse_openai_compatible_response,
)
from about_llm.integrations.cloud_http import HttpExecutorConfig
from about_llm.integrations.retry import RetryPolicy
from about_llm.integrations.sqlite_usage_budget import SQLiteUsageBudgetLedger
from about_llm.integrations.usage_budget import (
    TokenPricingSnapshot,
    UsageBudgetLedger,
    UsageBudgetLimits,
)

ROOT = Path(__file__).resolve().parents[1]


def _pricing() -> TokenPricingSnapshot:
    return TokenPricingSnapshot(
        pricing_id="provider/model@price-v1",
        provider="provider",
        model="model",
        revision="price-v1",
        checked_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        input_microusd_per_million=1_000_000,
        output_microusd_per_million=2_000_000,
    )


def _limits() -> UsageBudgetLimits:
    return UsageBudgetLimits(100, 20, 140)


def _memory_ledger() -> UsageBudgetLedger:
    return UsageBudgetLedger(limits=_limits(), pricing=_pricing())


def _retry_ledger() -> UsageBudgetLedger:
    return UsageBudgetLedger(
        limits=UsageBudgetLimits(200, 40, 200), pricing=_pricing()
    )


def _request():
    return build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="example-secret-not-real",
        model="model",
        messages=[ChatMessage("user", "authored request")],
        max_tokens=10,
    )


def _config() -> HttpExecutorConfig:
    return HttpExecutorConfig(
        allowed_origins=frozenset({"https://provider.invalid"}),
        deadline_seconds=5,
        request_timeout_seconds=2,
    )


async def _execute(
    ledger,
    handler,
    *,
    parse_response=parse_openai_compatible_response,
    policy: RetryPolicy | None = None,
    request=None,
):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await execute_budgeted_json_request(
            ledger=ledger,
            reservation_id="call-1",
            billing_scope="account/project",
            estimated_input_tokens=60,
            client=client,
            request=request or _request(),
            parse_response=parse_response,
            policy=policy or RetryPolicy(max_attempts=1),
            config=_config(),
            replay_safe=True,
            jitter=lambda: 0,
        )


async def _execute_retry(
    ledger,
    handler,
    *,
    parse_response=parse_openai_compatible_response,
    policy: RetryPolicy | None = None,
    replay_safe: bool = True,
    sleep=None,
):
    async def no_sleep(_delay: float) -> None:
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await execute_budgeted_json_request_with_retry(
            ledger=ledger,
            logical_call_id="logical-call",
            billing_scope="account/project",
            estimated_input_tokens=60,
            client=client,
            request=_request(),
            parse_response=parse_response,
            policy=policy
            or RetryPolicy(max_attempts=2, base_delay_seconds=0),
            config=_config(),
            replay_safe=replay_safe,
            sleep=sleep or no_sleep,
            wall_clock=lambda: datetime(2026, 8, 1, tzinfo=timezone.utc),
            jitter=lambda: 0,
        )


def _success(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        headers={"content-type": "application/json", "request-id": "req-1"},
        json={
            "id": "response-1",
            "model": "model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 58, "completion_tokens": 4},
        },
    )


def test_success_settles_reported_usage_and_preserves_attempt_evidence() -> None:
    ledger = _memory_ledger()

    result = asyncio.run(_execute(ledger, _success))

    assert result.response.text == "answer"
    assert result.status_code == 200
    assert result.attempts[0].request_id == "req-1"
    assert result.budget_snapshot.committed_input_tokens == 58
    assert result.budget_snapshot.committed_output_tokens == 4
    assert result.budget_snapshot.committed_estimated_microusd == 66
    assert result.budget_snapshot.active_reservations == 0


def test_connect_failure_is_the_only_transport_path_that_cancels() -> None:
    ledger = _memory_ledger()

    def connect_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("authored connect failure", request=request)

    with pytest.raises(BudgetedCloudCallError) as captured:
        asyncio.run(_execute(ledger, connect_failure))

    assert captured.value.reconciliation_state == "cancelled"
    assert captured.value.reason == "attempts_exhausted"
    assert ledger.snapshot().committed_estimated_microusd == 0
    assert ledger.snapshot().active_reservations == 0


def test_http_error_is_sent_and_therefore_conservatively_uncertain() -> None:
    ledger = _memory_ledger()

    def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, json={"error": "authored"})

    with pytest.raises(BudgetedCloudCallError) as captured:
        asyncio.run(_execute(ledger, server_error))

    assert captured.value.reconciliation_state == "uncertain"
    assert captured.value.attempts[0].status_code == 500
    assert captured.value.attempts[0].outcome_uncertain is False
    assert ledger.snapshot().committed_estimated_microusd == 80
    assert ledger.snapshot().uncertain_settlements == 1


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={
                "choices": [
                    {"message": {"content": "answer"}, "finish_reason": "stop"}
                ]
            },
        ),
        lambda request: httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            content=b"not-json",
        ),
    ],
)
def test_sent_success_without_trustworthy_usage_is_uncertain(handler) -> None:
    ledger = _memory_ledger()

    with pytest.raises(BudgetedCloudCallError) as captured:
        asyncio.run(_execute(ledger, handler))

    assert captured.value.reconciliation_state == "uncertain"
    assert ledger.snapshot().committed_estimated_microusd == 80


def test_cancellation_after_reservation_never_fabricates_zero_usage() -> None:
    ledger = _memory_ledger()

    async def cancelled(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_execute(ledger, cancelled))

    assert ledger.snapshot().committed_estimated_microusd == 80
    assert ledger.snapshot().uncertain_settlements == 1


def test_retry_policy_is_rejected_before_reservation_or_network() -> None:
    ledger = _memory_ledger()
    network_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        return _success(request)

    with pytest.raises(ValueError, match="reserve every replay"):
        asyncio.run(_execute(ledger, handler, policy=RetryPolicy(max_attempts=2)))

    assert network_calls == 0
    assert ledger.snapshot().active_reservations == 0


def test_target_policy_is_validated_before_reservation() -> None:
    ledger = _memory_ledger()
    request = build_openai_compatible_request(
        base_url="https://other.invalid",
        api_key="example-secret-not-real",
        model="model",
        messages=[ChatMessage("user", "authored request")],
        max_tokens=10,
    )

    with pytest.raises(ValueError, match="origin"):
        asyncio.run(_execute(ledger, _success, request=request))

    assert ledger.snapshot().active_reservations == 0


def test_retry_http_500_then_success_reconciles_each_attempt() -> None:
    ledger = _retry_ledger()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, request=request, json={"error": "authored"})
        return _success(request)

    result = asyncio.run(_execute_retry(ledger, handler))

    assert calls == 2
    assert [attempt.reservation.reservation_id for attempt in result.attempts] == [
        "logical-call:attempt:1",
        "logical-call:attempt:2",
    ]
    assert [attempt.reconciliation_state for attempt in result.attempts] == [
        "uncertain",
        "settled",
    ]
    assert [
        attempt.budget_snapshot.committed_estimated_microusd
        for attempt in result.attempts
    ] == [80, 146]
    assert result.budget_snapshot.committed_estimated_microusd == 146


def test_retry_connect_failure_then_success_cancels_first_attempt() -> None:
    ledger = _retry_ledger()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("authored connect failure", request=request)
        return _success(request)

    result = asyncio.run(_execute_retry(ledger, handler))

    assert calls == 2
    assert [attempt.reconciliation_state for attempt in result.attempts] == [
        "cancelled",
        "settled",
    ]
    assert [
        attempt.budget_snapshot.committed_estimated_microusd
        for attempt in result.attempts
    ] == [0, 66]


def test_retry_budget_gate_blocks_second_network_attempt() -> None:
    ledger = UsageBudgetLedger(
        limits=UsageBudgetLimits(200, 40, 140), pricing=_pricing()
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request, json={"error": "authored"})

    with pytest.raises(BudgetedCloudRetryError) as captured:
        asyncio.run(_execute_retry(ledger, handler))

    assert captured.value.reason == "budget_reservation_rejected"
    assert calls == 1
    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].reconciliation_state == "uncertain"
    assert captured.value.budget_snapshot.committed_estimated_microusd == 80
    assert captured.value.budget_snapshot.active_reservations == 0


def test_retry_after_delay_is_preserved_by_budget_orchestrator() -> None:
    ledger = _retry_ledger()
    calls = 0
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                request=request,
                headers={"Retry-After": "2"},
                json={"error": "authored"},
            )
        return _success(request)

    result = asyncio.run(_execute_retry(ledger, handler, sleep=record_sleep))

    assert calls == 2
    assert sleeps == [2]
    trace = result.attempts[0].trace
    assert trace is not None
    decision = trace.retry_decision
    assert decision is not None and decision.retry_after_state == "valid"
    assert result.budget_snapshot.committed_estimated_microusd == 146


def test_outcome_uncertain_transport_is_not_retried() -> None:
    ledger = _retry_ledger()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("authored read timeout", request=request)

    with pytest.raises(BudgetedCloudRetryError) as captured:
        asyncio.run(
            _execute_retry(
                ledger,
                handler,
                policy=RetryPolicy(max_attempts=3, base_delay_seconds=0),
            )
        )

    assert captured.value.reason == "outcome_uncertain"
    assert calls == 1
    assert captured.value.attempts[0].reconciliation_state == "uncertain"
    assert captured.value.budget_snapshot.committed_estimated_microusd == 80


def test_replay_unsafe_request_is_not_retried() -> None:
    ledger = _retry_ledger()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request, json={"error": "authored"})

    with pytest.raises(BudgetedCloudRetryError) as captured:
        asyncio.run(_execute_retry(ledger, handler, replay_safe=False))

    assert captured.value.reason == "replay_unsafe"
    assert calls == 1
    assert captured.value.attempts[0].reconciliation_state == "uncertain"


def test_retry_cancellation_after_reserve_is_uncertain() -> None:
    ledger = _retry_ledger()

    async def cancelled(_request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_execute_retry(ledger, cancelled))

    assert ledger.snapshot().committed_estimated_microusd == 80
    assert ledger.snapshot().uncertain_settlements == 1
    assert ledger.snapshot().active_reservations == 0


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            content=b"not-json",
        ),
        lambda request: httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={
                "choices": [
                    {"message": {"content": "answer"}, "finish_reason": "stop"}
                ]
            },
        ),
    ],
)
def test_retry_final_malformed_or_missing_usage_is_uncertain(handler) -> None:
    ledger = _retry_ledger()

    with pytest.raises(BudgetedCloudRetryError) as captured:
        asyncio.run(_execute_retry(ledger, handler))

    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].reconciliation_state == "uncertain"
    assert captured.value.budget_snapshot.committed_estimated_microusd == 80


def test_retry_sqlite_attempt_tombstones_survive_reopen(tmp_path: Path) -> None:
    database = tmp_path / "retry-budget.db"
    limits = UsageBudgetLimits(200, 40, 200)
    ledger = SQLiteUsageBudgetLedger(database, limits=limits, pricing=_pricing())
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, request=request, json={"error": "authored"})
        return _success(request)

    result = asyncio.run(_execute_retry(ledger, handler))
    reopened = SQLiteUsageBudgetLedger(database, limits=limits, pricing=_pricing())

    first = reopened.get("logical-call:attempt:1")
    second = reopened.get("logical-call:attempt:2")
    assert first is not None and first.state == "uncertain"
    assert second is not None and second.state == "settled"
    events = sorted(
        (
            *reopened.events("logical-call:attempt:1"),
            *reopened.events("logical-call:attempt:2"),
        ),
        key=lambda event: event.event_id,
    )
    assert [event.event_type for event in events] == [
        "reserved",
        "uncertain",
        "reserved",
        "settled",
    ]
    assert result.budget_snapshot.committed_estimated_microusd == 146
    assert reopened.snapshot() == result.budget_snapshot


def test_sqlite_uncertain_transition_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "budget.db"
    ledger = SQLiteUsageBudgetLedger(database, limits=_limits(), pricing=_pricing())

    def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, json={"error": "authored"})

    with pytest.raises(BudgetedCloudCallError):
        asyncio.run(_execute(ledger, server_error))

    reopened = SQLiteUsageBudgetLedger(database, limits=_limits(), pricing=_pricing())
    record = reopened.get("call-1")
    assert record is not None and record.state == "uncertain"
    assert [event.event_type for event in reopened.events("call-1")] == [
        "reserved",
        "uncertain",
    ]
    assert reopened.snapshot().committed_estimated_microusd == 80


