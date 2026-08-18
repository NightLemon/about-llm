from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from about_llm.agents import (
    EffectRequest,
    IdempotencyConflict,
    OutboxState,
    SQLiteTransactionalOutbox,
)

pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]

EXECUTION_FINGERPRINT = "sha256:" + "a" * 64
ROOT = Path(__file__).resolve().parents[1]


def _effect(
    effect_id: str = "effect-1",
    *,
    payload: dict[str, Any] | None = None,
) -> EffectRequest:
    return EffectRequest(
        effect_id=effect_id,
        execution_fingerprint=EXECUTION_FINGERPRINT,
        destination="simulated-mail",
        payload=payload or {"to": "test@example.invalid", "body": "hello"},
    )


def test_task_state_and_outbox_effect_commit_atomically_and_idempotently(
    tmp_path: Path,
) -> None:
    outbox = SQLiteTransactionalOutbox(tmp_path / "outbox.db")
    effect = _effect()

    assert outbox.commit_task_effect(
        "task-1", {"status": "approved"}, effect, now=10
    )
    assert not outbox.commit_task_effect(
        "task-1", {"status": "approved"}, effect, now=11
    )
    assert dict(outbox.task_state("task-1")) == {"status": "approved"}
    record = outbox.get("effect-1")
    assert record is not None
    assert record.state is OutboxState.PENDING
    assert record.attempt_count == 0
    assert record.effect_fingerprint == effect.effect_fingerprint
    assert [event.event_type for event in outbox.events("effect-1")] == ["enqueued"]

    with pytest.raises(IdempotencyConflict):
        outbox.commit_task_effect(
            "task-1", {"status": "changed"}, effect, now=12
        )
    with pytest.raises(IdempotencyConflict):
        outbox.commit_task_effect(
            "task-1", {"status": "approved"}, _effect("effect-2"), now=12
        )


def test_outbox_insert_failure_rolls_back_local_task_state(tmp_path: Path) -> None:
    database = tmp_path / "outbox.db"
    outbox = SQLiteTransactionalOutbox(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_outbox_insert
            BEFORE INSERT ON agent_outbox
            BEGIN
                SELECT RAISE(ABORT, 'injected outbox failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected outbox failure"):
        outbox.commit_task_effect(
            "task-rollback", {"status": "approved"}, _effect(), now=10
        )

    assert outbox.task_state("task-rollback") is None
    assert outbox.get("effect-1") is None


def test_concurrent_workers_claim_one_effect_once_per_lease(tmp_path: Path) -> None:
    database = tmp_path / "outbox.db"
    SQLiteTransactionalOutbox(database).commit_task_effect(
        "task-1", {"status": "approved"}, _effect(), now=10
    )

    def claim(worker_id: str) -> object:
        return SQLiteTransactionalOutbox(database).claim_due(
            worker_id, now=10, lease_seconds=5
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        deliveries = tuple(executor.map(claim, ("worker-a", "worker-b")))

    assert sum(delivery is not None for delivery in deliveries) == 1
    claimed = next(delivery for delivery in deliveries if delivery is not None)
    assert claimed.attempt == 1  # type: ignore[union-attr]
    assert claimed.provider_idempotency_key == "effect-1"  # type: ignore[union-attr]


def test_retry_lease_renewal_delivery_and_stale_worker_rejection(tmp_path: Path) -> None:
    outbox = SQLiteTransactionalOutbox(tmp_path / "outbox.db")
    outbox.commit_task_effect("task-1", {"status": "approved"}, _effect(), now=10)
    first = outbox.claim_due("worker-a", now=10, lease_seconds=5)
    assert first is not None and first.attempt == 1
    outbox.schedule_retry(
        "effect-1",
        "worker-a",
        error_code="provider.rate_limited",
        retry_at=20,
        now=11,
    )
    assert outbox.claim_due("worker-b", now=19, lease_seconds=5) is None
    second = outbox.claim_due("worker-b", now=20, lease_seconds=5)
    assert second is not None and second.attempt == 2
    with pytest.raises(IdempotencyConflict):
        outbox.mark_delivered(
            "effect-1", "worker-a", {"provider_id": "stale"}, now=21
        )
    assert outbox.renew_lease(
        "effect-1", "worker-b", now=21, lease_seconds=10
    ) == pytest.approx(31)
    outbox.mark_delivered(
        "effect-1", "worker-b", {"provider_id": "receipt-1"}, now=30
    )

    record = outbox.get("effect-1")
    assert record is not None
    assert record.state is OutboxState.DELIVERED
    assert record.attempt_count == 2
    assert dict(record.receipt) == {"provider_id": "receipt-1"}
    assert outbox.claim_due("worker-c", now=40, lease_seconds=5) is None


def test_crash_after_provider_success_causes_redelivery_not_exactly_once(
    tmp_path: Path,
) -> None:
    class IdempotentProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.effects: dict[str, dict[str, str]] = {}

        def send(self, idempotency_key: str) -> dict[str, str]:
            self.calls += 1
            return self.effects.setdefault(
                idempotency_key, {"provider_id": "provider-effect-1"}
            )

    outbox = SQLiteTransactionalOutbox(tmp_path / "outbox.db")
    outbox.commit_task_effect("task-1", {"status": "approved"}, _effect(), now=10)
    provider = IdempotentProvider()
    first = outbox.claim_due("worker-a", now=10, lease_seconds=5)
    assert first is not None
    first_receipt = provider.send(first.provider_idempotency_key)
    # Simulated crash: the remote effect succeeded, but worker-a never acknowledged it.

    second = outbox.claim_due("worker-b", now=16, lease_seconds=5)
    assert second is not None and second.attempt == 2
    second_receipt = provider.send(second.provider_idempotency_key)
    assert second_receipt == first_receipt
    outbox.mark_delivered("effect-1", "worker-b", second_receipt, now=17)

    assert provider.calls == 2
    assert len(provider.effects) == 1
    assert [event.event_type for event in outbox.events("effect-1")] == [
        "enqueued",
        "claimed",
        "lease_expired",
        "claimed",
        "delivered",
    ]


def test_terminal_failure_moves_effect_to_dead_letter(tmp_path: Path) -> None:
    outbox = SQLiteTransactionalOutbox(tmp_path / "outbox.db")
    outbox.commit_task_effect("task-1", {"status": "approved"}, _effect(), now=10)
    assert outbox.claim_due("worker-a", now=10, lease_seconds=5) is not None
    outbox.schedule_retry(
        "effect-1",
        "worker-a",
        error_code="provider.invalid_recipient",
        retry_at=None,
        terminal=True,
        now=11,
    )

    record = outbox.get("effect-1")
    assert record is not None
    assert record.state is OutboxState.DEAD_LETTER
    assert record.last_error_code == "provider.invalid_recipient"
    assert outbox.claim_due("worker-b", now=20, lease_seconds=5) is None


def test_outbox_rejects_non_json_and_expired_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strict JSON object"):
        _effect(payload={"bad": float("nan")})
    with pytest.raises(ValueError, match="lowercase sha256"):
        EffectRequest("effect", "not-a-hash", "destination", {})

    outbox = SQLiteTransactionalOutbox(tmp_path / "outbox.db")
    outbox.commit_task_effect("task-1", {"status": "approved"}, _effect(), now=10)
    assert outbox.claim_due("worker-a", now=10, lease_seconds=5) is not None
    with pytest.raises(IdempotencyConflict, match="unexpired lease"):
        outbox.mark_delivered(
            "effect-1", "worker-a", {"provider_id": "late"}, now=15
        )
    with pytest.raises(ValueError, match="machine token"):
        outbox.schedule_retry(
            "effect-1",
            "worker-a",
            error_code="raw provider error with spaces",
            retry_at=20,
            now=11,
        )


def test_outbox_demo_records_crash_redelivery_and_provider_deduplication(
    tmp_path: Path,
) -> None:
    database = tmp_path / "outbox-demo.db"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "projects" / "safe-agent" / "outbox_demo.py"),
            "--database",
            str(database),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["atomic_commit_created"] is True
    assert artifact["provider_idempotency_keys"] == ["effect-1", "effect-1"]
    assert artifact["attempts"] == 2
    assert artifact["provider_calls"] == 2
    assert artifact["provider_effect_count"] == 1
    assert artifact["receipts_equal"] is True
    assert artifact["final_state"] == "delivered"
    assert [event["event_type"] for event in artifact["events"]] == [
        "enqueued",
        "claimed",
        "lease_expired",
        "claimed",
        "delivered",
    ]
    assert artifact["scope"] == {
        "storage": "local SQLite",
        "provider": "in-memory simulated idempotent provider",
        "network_used": False,
        "delivery_semantics": "at-least-once",
        "proves_exactly_once_external_effect": False,
    }
