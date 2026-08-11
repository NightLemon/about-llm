from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from about_llm.integrations.budgeted_cloud import (
    BudgetedCloudCallError,
    execute_budgeted_json_request,
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


def test_budgeted_http_demo_has_success_and_sent_error_ledgers(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "projects"
                / "cloud-api-contracts"
                / "budgeted_http_demo.py"
            ),
            "--database",
            str(database),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["network_used"] is False
    assert artifact["http_calls"] == 2
    assert artifact["records"]["call-success"]["state"] == "settled"
    assert artifact["records"]["call-http-500"]["state"] == "uncertain"
    assert artifact["failure"]["reconciliation_state"] == "uncertain"
    assert artifact["final_snapshot"]["committed_estimated_microusd"] == 146
    assert artifact["scope"]["each_replay_requires_new_reservation"] is True
