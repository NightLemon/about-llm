"""Deterministic offline demo of transactional-outbox crash recovery."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from about_llm.agents import EffectRequest, SQLiteTransactionalOutbox

EXECUTION_FINGERPRINT = "sha256:" + "a" * 64


class SimulatedIdempotentProvider:
    """In-memory provider that deduplicates requests by idempotency key."""

    def __init__(self) -> None:
        self.calls = 0
        self.effects: dict[str, dict[str, str]] = {}

    def send(
        self,
        *,
        idempotency_key: str,
        destination: str,
        payload: Any,
    ) -> dict[str, str]:
        self.calls += 1
        del destination, payload
        return self.effects.setdefault(
            idempotency_key,
            {"provider_effect_id": "simulated-provider-effect-1"},
        )


def run_demo(database: Path) -> dict[str, Any]:
    """Run one success-before-ack crash and return its auditable artifact."""

    if database.exists():
        raise ValueError(f"refusing to reuse existing database: {database}")

    provider = SimulatedIdempotentProvider()
    worker_a = SQLiteTransactionalOutbox(database)
    effect = EffectRequest(
        effect_id="effect-1",
        execution_fingerprint=EXECUTION_FINGERPRINT,
        destination="simulated-mail",
        payload={"to": "learner@example.invalid", "body": "hello"},
    )
    atomic_commit_created = worker_a.commit_task_effect(
        "task-1",
        {"status": "approved"},
        effect,
        now=10,
    )
    first = worker_a.claim_due("worker-a", now=10, lease_seconds=5)
    if first is None:
        raise RuntimeError("worker-a did not claim the fixture effect")
    first_receipt = provider.send(
        idempotency_key=first.provider_idempotency_key,
        destination=first.destination,
        payload=first.payload,
    )

    # Simulated crash: provider success is durable, but the local acknowledgement is absent.
    del worker_a
    worker_b = SQLiteTransactionalOutbox(database)
    second = worker_b.claim_due("worker-b", now=16, lease_seconds=5)
    if second is None:
        raise RuntimeError("worker-b did not reclaim the expired fixture effect")
    second_receipt = provider.send(
        idempotency_key=second.provider_idempotency_key,
        destination=second.destination,
        payload=second.payload,
    )
    worker_b.mark_delivered(second.effect_id, "worker-b", second_receipt, now=17)

    record = worker_b.get(second.effect_id)
    if record is None:
        raise RuntimeError("delivered fixture effect disappeared")
    task_state = worker_b.task_state("task-1")
    if not isinstance(task_state, Mapping):
        raise RuntimeError("fixture task state is missing or malformed")
    return {
        "schema_version": 1,
        "simulated_offline": True,
        "database": str(database),
        "atomic_commit_created": atomic_commit_created,
        "task_state": dict(task_state),
        "effect_id": record.effect_id,
        "provider_idempotency_keys": [
            first.provider_idempotency_key,
            second.provider_idempotency_key,
        ],
        "attempts": record.attempt_count,
        "provider_calls": provider.calls,
        "provider_effect_count": len(provider.effects),
        "receipts_equal": first_receipt == second_receipt,
        "final_state": record.state.value,
        "events": [
            {
                "event_type": event.event_type,
                "worker_id": event.worker_id,
                "occurred_at": event.occurred_at,
                "detail": dict(event.detail),
            }
            for event in worker_b.events(record.effect_id)
        ],
        "scope": {
            "storage": "local SQLite",
            "provider": "in-memory simulated idempotent provider",
            "network_used": False,
            "delivery_semantics": "at-least-once",
            "proves_exactly_once_external_effect": False,
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
    print(json.dumps(run_demo(args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
