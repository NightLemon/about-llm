"""离线演示带 SQLite 预算预留的逐次 HTTP 重试。

第一次请求模拟返回可重试的 500，第二次才成功。两个 attempt 各有独立 reservation，
因此读者能看到“逻辑调用重试一次”在预算账本中为何必须留下两条远端尝试记录。
"""

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
    UsageBudgetLimits,
)


def _ledger(database: Path) -> SQLiteUsageBudgetLedger:
    """创建固定预算、价格和时钟的持久化账本。"""

    return SQLiteUsageBudgetLedger(
        database,
        limits=UsageBudgetLimits(200, 40, 200),
        pricing=TokenPricingSnapshot(
            pricing_id="authored-provider/model@price-v1",
            provider="authored-provider",
            model="model",
            revision="price-v1",
            checked_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            input_microusd_per_million=1_000_000,
            output_microusd_per_million=2_000_000,
        ),
        clock=lambda: datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc),
    )


async def run_demo(database: Path) -> dict[str, Any]:
    """先记录一次 500 的不确定用量，再结算成功的重放请求。"""

    if database.exists():
        raise ValueError(f"refusing to reuse existing database: {database}")

    # 同一个 handler 按调用次数返回 500 → 200，完全离线复现重试顺序。
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                500,
                request=request,
                headers={"request-id": "fixture-attempt-1"},
                json={"error": "authored retryable server error"},
            )
        return httpx.Response(
            200,
            request=request,
            headers={
                "content-type": "application/json",
                "request-id": "fixture-attempt-2",
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

    async def no_sleep(_delay: float) -> None:
        """跳过真实退避等待，让教学实验快速完成。"""

        return None

    request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="example-secret-not-real",
        model="model",
        messages=[ChatMessage("user", "authored request")],
        max_tokens=10,
    )
    ledger = _ledger(database)
    # 重试器会为每个远端 attempt 创建新 reservation，而不是复用第一次的账本记录。
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await execute_budgeted_json_request_with_retry(
            ledger=ledger,
            logical_call_id="logical-call",
            billing_scope="authored-account/project",
            estimated_input_tokens=60,
            client=client,
            request=request,
            parse_response=parse_openai_compatible_response,
            policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
            config=HttpExecutorConfig(
                allowed_origins=frozenset({"https://provider.invalid"}),
                deadline_seconds=5,
                request_timeout_seconds=2,
            ),
            replay_safe=True,
            sleep=no_sleep,
            jitter=lambda: 0,
        )

    reservation_ids = tuple(
        attempt.reservation.reservation_id for attempt in result.attempts
    )
    return {
        "schema_version": 1,
        "simulated_offline": True,
        "network_used": False,
        "database": str(database),
        "http_calls": calls,
        "result": {
            "response": asdict(result.response),
            "attempts": [asdict(attempt) for attempt in result.attempts],
            "budget_snapshot": asdict(result.budget_snapshot),
        },
        "records": {
            reservation_id: asdict(record)
            for reservation_id in reservation_ids
            if (record := ledger.get(reservation_id)) is not None
        },
        "events": [
            asdict(event)
            for reservation_id in reservation_ids
            for event in ledger.events(reservation_id)
        ],
        "scope": {
            "transport": "httpx.MockTransport",
            "json_only": True,
            "streaming_retry_supported": False,
            "each_attempt_has_independent_reservation": True,
            "remote_call_atomic_with_sqlite": False,
            "proves_provider_usage_or_invoice": False,
            "proves_exactly_once_billing": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """读取一个全新的 SQLite 数据库路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="new SQLite path; an existing file is rejected",
    )
    return parser.parse_args()


def main() -> None:
    """执行异步重试实验并输出 reservation 与事件时间线。"""

    args = parse_args()
    print(json.dumps(asyncio.run(run_demo(args.database)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
