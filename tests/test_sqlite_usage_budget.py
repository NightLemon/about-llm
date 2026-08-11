from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier

import pytest

from about_llm.integrations.cloud_api import (
    ChatMessage,
    RequestSpec,
    build_openai_compatible_request,
)
from about_llm.integrations.sqlite_usage_budget import SQLiteUsageBudgetLedger
from about_llm.integrations.usage_budget import (
    PostCallBudgetExceededError,
    TokenPricingSnapshot,
    UsageBudgetExceededError,
    UsageBudgetLimits,
    UsageBudgetStateError,
)

REQUEST_FINGERPRINT = "sha256:" + "a" * 64
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def _pricing(**overrides: object) -> TokenPricingSnapshot:
    values = {
        "pricing_id": "provider/model@price-2026-08-01",
        "provider": "provider",
        "model": "model",
        "revision": "price-2026-08-01",
        "checked_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "input_microusd_per_million": 1_000_000,
        "output_microusd_per_million": 2_000_000,
    }
    values.update(overrides)
    return TokenPricingSnapshot(**values)  # type: ignore[arg-type]


def _limits(**overrides: int | None) -> UsageBudgetLimits:
    values: dict[str, int | None] = {
        "max_input_tokens": 100,
        "max_output_tokens": 20,
        "max_estimated_microusd": 140,
    }
    values.update(overrides)
    return UsageBudgetLimits(**values)


def _ledger(
    path: Path,
    *,
    limits: UsageBudgetLimits | None = None,
    pricing: TokenPricingSnapshot | None = None,
) -> SQLiteUsageBudgetLedger:
    return SQLiteUsageBudgetLedger(
        path,
        limits=limits or _limits(),
        pricing=pricing or _pricing(),
        clock=lambda: NOW,
    )


def _request(max_tokens: int = 10, *, model: str = "model") -> RequestSpec:
    return build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="example-secret-not-real",
        model=model,
        messages=[ChatMessage("user", "authored request")],
        max_tokens=max_tokens,
    )


def test_reservation_survives_reopen_and_uncertain_reconciliation(tmp_path: Path) -> None:
    database = tmp_path / "budget.db"
    first = _ledger(database)
    receipt = first.reserve_request(
        "call-1",
        request=_request(),
        billing_scope="account/project",
        estimated_input_tokens=60,
    )

    reopened = _ledger(database)
    active = reopened.list_active()

    assert reopened.snapshot().reserved_input_tokens == 60
    assert len(active) == 1
    assert active[0].request_fingerprint == receipt.request_fingerprint
    assert active[0].budget_config_fingerprint == reopened.config_fingerprint
    assert [event.event_type for event in reopened.events("call-1")] == ["reserved"]

    reconciled = reopened.mark_usage_uncertain(
        "call-1", request_fingerprint=receipt.request_fingerprint
    )
    assert reconciled.committed_input_tokens == 60
    assert reconciled.committed_output_tokens == 10
    assert reconciled.committed_estimated_microusd == 80
    assert reopened.get("call-1").state == "uncertain"  # type: ignore[union-attr]
    assert [event.event_type for event in reopened.events("call-1")] == [
        "reserved",
        "uncertain",
    ]


def test_successful_settlement_refunds_reserved_capacity_and_is_auditable(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "budget.db")
    receipt = ledger.reserve_request(
        "call-1",
        request=_request(),
        billing_scope="account/project",
        estimated_input_tokens=60,
    )

    snapshot = ledger.settle(
        "call-1",
        request_fingerprint=receipt.request_fingerprint,
        actual_input_tokens=58,
        actual_output_tokens=4,
    )
    record = ledger.get("call-1")

    assert snapshot.committed_estimated_microusd == 66
    assert snapshot.reserved_estimated_microusd == 0
    assert record is not None and record.state == "settled"
    assert record.actual_input_tokens == 58
    assert record.actual_output_tokens == 4
    assert record.actual_estimated_microusd == 66
    assert record.created_at_utc == "2026-08-07T12:00:00Z"
    assert record.terminal_at_utc == "2026-08-07T12:00:00Z"
    assert [event.event_type for event in ledger.events("call-1")] == [
        "reserved",
        "settled",
    ]


def test_two_connections_cannot_reserve_the_same_remaining_capacity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "budget.db"
    limits = UsageBudgetLimits(max_input_tokens=10)
    first = _ledger(database, limits=limits)
    second = _ledger(database, limits=limits)
    barrier = Barrier(2)

    def attempt(item: tuple[str, SQLiteUsageBudgetLedger]) -> str:
        reservation_id, ledger = item
        barrier.wait()
        try:
            ledger.reserve(
                reservation_id,
                request_fingerprint=REQUEST_FINGERPRINT,
                estimated_input_tokens=10,
                maximum_output_tokens=0,
            )
        except UsageBudgetExceededError:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            executor.map(attempt, (("call-1", first), ("call-2", second)))
        )

    assert sorted(outcomes) == ["rejected", "reserved"]
    assert _ledger(database, limits=limits).snapshot().reserved_input_tokens == 10


def test_wrong_terminal_fingerprint_does_not_release_durable_capacity(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "budget.db")
    ledger.reserve(
        "call-1",
        request_fingerprint=REQUEST_FINGERPRINT,
        estimated_input_tokens=60,
        maximum_output_tokens=10,
    )

    with pytest.raises(UsageBudgetStateError, match="does not match"):
        ledger.cancel_before_send(
            "call-1", request_fingerprint="sha256:" + "b" * 64
        )

    assert ledger.snapshot().active_reservations == 1
    assert ledger.get("call-1").state == "active"  # type: ignore[union-attr]
    assert [event.event_type for event in ledger.events("call-1")] == ["reserved"]


def test_post_call_overrun_is_committed_before_error_and_blocks_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "budget.db"
    limits = UsageBudgetLimits(max_input_tokens=60)
    ledger = _ledger(database, limits=limits)
    ledger.reserve(
        "call-1",
        request_fingerprint=REQUEST_FINGERPRINT,
        estimated_input_tokens=60,
        maximum_output_tokens=0,
    )

    with pytest.raises(PostCallBudgetExceededError) as captured:
        ledger.settle(
            "call-1",
            request_fingerprint=REQUEST_FINGERPRINT,
            actual_input_tokens=61,
            actual_output_tokens=0,
        )

    assert captured.value.snapshot.over_limit is True
    reopened = _ledger(database, limits=limits)
    assert reopened.snapshot().committed_input_tokens == 61
    assert reopened.get("call-1").state == "settled"  # type: ignore[union-attr]
    with pytest.raises(UsageBudgetExceededError):
        reopened.reserve(
            "call-2",
            request_fingerprint=REQUEST_FINGERPRINT,
            estimated_input_tokens=0,
            maximum_output_tokens=0,
        )


def test_configuration_drift_and_physical_tamper_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "budget.db"
    original = _ledger(database)

    with pytest.raises(UsageBudgetStateError, match="does not match"):
        _ledger(database, limits=_limits(max_input_tokens=101))
    with pytest.raises(UsageBudgetStateError, match="does not match"):
        _ledger(database, pricing=_pricing(revision="changed"))

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE cloud_usage_budget_config SET max_input_tokens = 99"
        )

    with pytest.raises(UsageBudgetStateError, match="fingerprint is inconsistent"):
        original.snapshot()


def test_event_insert_failure_rolls_back_reservation_row(tmp_path: Path) -> None:
    database = tmp_path / "budget.db"
    ledger = _ledger(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_budget_event_insert
            BEFORE INSERT ON cloud_usage_budget_events
            BEGIN
                SELECT RAISE(ABORT, 'injected budget event failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected budget event failure"):
        ledger.reserve(
            "call-1",
            request_fingerprint=REQUEST_FINGERPRINT,
            estimated_input_tokens=10,
            maximum_output_tokens=1,
        )

    assert ledger.get("call-1") is None
    assert ledger.snapshot().active_reservations == 0


def test_cancelled_id_is_a_tombstone_and_file_handles_are_released(
    tmp_path: Path,
) -> None:
    database = tmp_path / "budget.db"
    ledger = _ledger(database)
    ledger.reserve(
        "call-1",
        request_fingerprint=REQUEST_FINGERPRINT,
        estimated_input_tokens=10,
        maximum_output_tokens=1,
    )
    ledger.cancel_before_send(
        "call-1", request_fingerprint=REQUEST_FINGERPRINT
    )

    with pytest.raises(UsageBudgetStateError, match="already been used"):
        ledger.reserve(
            "call-1",
            request_fingerprint=REQUEST_FINGERPRINT,
            estimated_input_tokens=1,
            maximum_output_tokens=1,
        )

    moved = tmp_path / "moved.db"
    database.rename(moved)
    assert _ledger(moved).get("call-1").state == "cancelled"  # type: ignore[union-attr]


def test_subprocess_exit_leaves_active_reservation_for_reconciliation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "subprocess.db"
    code = f"""
from datetime import datetime, timezone
from pathlib import Path
from about_llm.integrations.sqlite_usage_budget import SQLiteUsageBudgetLedger
from about_llm.integrations.usage_budget import TokenPricingSnapshot, UsageBudgetLimits
pricing = TokenPricingSnapshot(
    'provider/model@price-2026-08-01', 'provider', 'model', 'price-2026-08-01',
    datetime(2026, 8, 1, tzinfo=timezone.utc), 1000000, 2000000
)
ledger = SQLiteUsageBudgetLedger(
    Path({str(database)!r}),
    limits=UsageBudgetLimits(100, 20, 140),
    pricing=pricing,
)
ledger.reserve(
    'call-1', request_fingerprint={'sha256:' + 'a' * 64!r},
    estimated_input_tokens=60, maximum_output_tokens=10
)
"""
    subprocess.run([sys.executable, "-c", code], check=True)

    reopened = _ledger(database)
    assert reopened.snapshot().active_reservations == 1
    assert reopened.list_active()[0].reservation_id == "call-1"


def test_sqlite_integer_range_and_clock_contract_fail_before_mutation(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="SQLite signed-integer range"):
        _ledger(
            tmp_path / "too-large.db",
            limits=UsageBudgetLimits(max_input_tokens=1 << 63),
        )

    ledger = SQLiteUsageBudgetLedger(
        tmp_path / "bad-clock.db",
        limits=_limits(),
        pricing=_pricing(),
        clock=lambda: datetime(2026, 8, 7),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.reserve(
            "call-1",
            request_fingerprint=REQUEST_FINGERPRINT,
            estimated_input_tokens=1,
            maximum_output_tokens=1,
        )
    assert ledger.snapshot().active_reservations == 0


def test_durable_reserve_request_rejects_model_mismatch(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "budget.db")

    with pytest.raises(ValueError, match="pricing snapshot model"):
        ledger.reserve_request(
            "call-1",
            request=_request(model="different"),
            billing_scope="account/project",
            estimated_input_tokens=1,
        )

    assert ledger.snapshot().active_reservations == 0


def test_sqlite_usage_budget_demo_reopens_and_reconciles(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "projects"
                / "cloud-api-contracts"
                / "sqlite_usage_budget_demo.py"
            ),
            "--database",
            str(database),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["simulated_offline"] is True
    assert len(artifact["active_after_reopen"]) == 1
    assert artifact["active_after_reopen"][0]["state"] == "active"
    assert artifact["reconciled_record"]["state"] == "uncertain"
    assert artifact["uncertain_snapshot"]["committed_estimated_microusd"] == 80
    assert [event["event_type"] for event in artifact["events"]] == [
        "reserved",
        "uncertain",
    ]
    assert artifact["scope"] == {
        "storage": "local SQLite",
        "network_used": False,
        "remote_call_atomic_with_sqlite": False,
        "authenticates_usage_or_pricing": False,
        "proves_provider_invoice": False,
        "proves_exactly_once_billing": False,
    }
