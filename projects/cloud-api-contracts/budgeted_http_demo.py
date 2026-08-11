"""Offline SQLite budget + MockTransport reconciliation demo."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

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
    UsageBudgetLimits,
)


def _ledger(database: Path) -> SQLiteUsageBudgetLedger:
    return SQLiteUsageBudgetLedger(
        database,
        limits=UsageBudgetLimits(200, 40, 280),
        pricing=TokenPricingSnapshot(
            pricing_id="authored-provider/model@price-v1",
            provider="authored-provider",
            model="model",
            revision="price-v1",
            checked_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            input_microusd_per_million=1_000_000,
            output_microusd_per_million=2_000_000,
        ),
        clock=lambda: datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
    )


async def run_demo(database: Path) -> dict[str, Any]:
    """Settle one reported usage and reconcile one sent HTTP error."""

    if database.exists():
        raise ValueError(f"refusing to reuse existing database: {database}")

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                request=request,
                headers={
                    "content-type": "application/json",
                    "request-id": "fixture-request-1",
                },
                json={
                    "model": "model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "fixture answer",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 58, "completion_tokens": 4},
                },
            )
        return httpx.Response(
            500,
            request=request,
            headers={"request-id": "fixture-request-2"},
            json={"error": "authored server error"},
        )

    request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="example-secret-not-real",
        model="model",
        messages=[ChatMessage("user", "authored request")],
        max_tokens=10,
    )
    ledger = _ledger(database)
    config = HttpExecutorConfig(
        allowed_origins=frozenset({"https://provider.invalid"}),
        deadline_seconds=5,
        request_timeout_seconds=2,
    )
    policy = RetryPolicy(max_attempts=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        success = await execute_budgeted_json_request(
            ledger=ledger,
            reservation_id="call-success",
            billing_scope="authored-account/project",
            estimated_input_tokens=60,
            client=client,
            request=request,
            parse_response=parse_openai_compatible_response,
            policy=policy,
            config=config,
            replay_safe=True,
            jitter=lambda: 0,
        )
        try:
            await execute_budgeted_json_request(
                ledger=ledger,
                reservation_id="call-http-500",
                billing_scope="authored-account/project",
                estimated_input_tokens=60,
                client=client,
                request=request,
                parse_response=parse_openai_compatible_response,
                policy=policy,
                config=config,
                replay_safe=True,
                jitter=lambda: 0,
            )
        except BudgetedCloudCallError as error:
            failure = {
                "reason": error.reason,
                "reconciliation_state": error.reconciliation_state,
                "attempts": [asdict(attempt) for attempt in error.attempts],
                "budget_snapshot": asdict(error.budget_snapshot),
            }
        else:
            raise RuntimeError("fixture HTTP 500 unexpectedly succeeded")

    return {
        "schema_version": 1,
        "simulated_offline": True,
        "network_used": False,
        "database": str(database),
        "http_calls": calls,
        "success": {
            "response": asdict(success.response),
            "attempts": [asdict(attempt) for attempt in success.attempts],
            "budget_snapshot": asdict(success.budget_snapshot),
        },
        "failure": failure,
        "final_snapshot": asdict(ledger.snapshot()),
        "records": {
            reservation_id: asdict(record)
            for reservation_id in ("call-success", "call-http-500")
            if (record := ledger.get(reservation_id)) is not None
        },
        "events": {
            reservation_id: [
                asdict(event) for event in ledger.events(reservation_id)
            ]
            for reservation_id in ("call-success", "call-http-500")
        },
        "scope": {
            "transport": "httpx.MockTransport",
            "automatic_retry_attempts": 0,
            "each_replay_requires_new_reservation": True,
            "remote_call_atomic_with_sqlite": False,
            "proves_provider_usage_or_invoice": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="new SQLite path; an existing file is rejected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(asyncio.run(run_demo(args.database)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
