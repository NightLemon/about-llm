"""离线演示进程退出后仍存在的 SQLite API 用量预留。

实验先预留一次请求，然后模拟在收到响应前进程退出。重新打开数据库后，reservation 仍在，
并被保守标记为 usage uncertain，避免因本地不知道远端是否计费而错误释放预算。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from about_llm.integrations.cloud_api import (
    ChatMessage,
    build_openai_compatible_request,
)
from about_llm.integrations.sqlite_usage_budget import SQLiteUsageBudgetLedger
from about_llm.integrations.usage_budget import (
    TokenPricingSnapshot,
    UsageBudgetLimits,
)


def _open_ledger(database: Path) -> SQLiteUsageBudgetLedger:
    """用相同配置打开账本，模拟两个先后启动的进程。"""

    return SQLiteUsageBudgetLedger(
        database,
        limits=UsageBudgetLimits(
            max_input_tokens=100,
            max_output_tokens=20,
            max_estimated_microusd=140,
        ),
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


def run_demo(database: Path) -> dict[str, Any]:
    """预留请求、重开数据库，并保守处理状态未知的远端调用。"""

    if database.exists():
        raise ValueError(f"refusing to reuse existing database: {database}")

    # 请求指纹不包含可轮换的密钥明文，但绑定会影响计费的请求字段。
    request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="example-secret-not-real",
        model="model",
        messages=[ChatMessage("user", "authored request")],
        max_tokens=10,
    )
    first_process = _open_ledger(database)
    # 在真正发送网络请求前先占用最大预算，防止并发调用共同突破限额。
    reservation = first_process.reserve_request(
        "call-1",
        request=request,
        billing_scope="authored-account/project",
        estimated_input_tokens=60,
    )
    reserved_snapshot = first_process.snapshot()

    # 模拟进程退出：在重新打开数据库之前没有 success/refund 等终态转换。
    del first_process
    restarted_process = _open_ledger(database)
    active_after_reopen = restarted_process.list_active()
    uncertain_snapshot = restarted_process.mark_usage_uncertain(
        reservation.reservation_id,
        request_fingerprint=reservation.request_fingerprint,
    )
    record = restarted_process.get(reservation.reservation_id)
    if record is None:
        raise RuntimeError("durable reservation disappeared")

    return {
        "schema_version": 1,
        "simulated_offline": True,
        "database": str(database),
        "config_fingerprint": restarted_process.config_fingerprint,
        "reservation": asdict(reservation),
        "reserved_snapshot": asdict(reserved_snapshot),
        "active_after_reopen": [asdict(item) for item in active_after_reopen],
        "reconciled_record": asdict(record),
        "uncertain_snapshot": asdict(uncertain_snapshot),
        "events": [
            asdict(event)
            for event in restarted_process.events(reservation.reservation_id)
        ],
        "scope": {
            "storage": "local SQLite",
            "network_used": False,
            "remote_call_atomic_with_sqlite": False,
            "authenticates_usage_or_pricing": False,
            "proves_provider_invoice": False,
            "proves_exactly_once_billing": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """读取一个尚不存在的数据库路径，避免覆盖旧实验。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="new SQLite path; an existing file is rejected",
    )
    return parser.parse_args()


def main() -> None:
    """执行崩溃恢复时间线并输出完整事件记录。"""

    args = parse_args()
    print(json.dumps(run_demo(args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
