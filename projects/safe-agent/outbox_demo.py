"""离线演示 transactional outbox 在“外部成功、本地未确认”崩溃点的恢复。

worker-a 把任务状态与待发送 effect 原子写入 SQLite，调用 provider 成功后立刻模拟崩溃。
租约过期后 worker-b 重试同一 idempotency key；provider 去重，因此两次调用只产生一个副作用。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from about_llm.agents import EffectRequest, SQLiteTransactionalOutbox

EXECUTION_FINGERPRINT = "sha256:" + "a" * 64


class SimulatedIdempotentProvider:
    """按 idempotency key 去重的内存 provider。"""

    def __init__(self) -> None:
        """初始化调用计数与已产生副作用的映射。"""

        self.calls = 0
        self.effects: dict[str, dict[str, str]] = {}

    def send(
        self,
        *,
        idempotency_key: str,
        destination: str,
        payload: Any,
    ) -> dict[str, str]:
        """模拟外部发送；重复 key 返回同一 receipt，不再产生新 effect。"""

        self.calls += 1
        del destination, payload
        return self.effects.setdefault(
            idempotency_key,
            {"provider_effect_id": "simulated-provider-effect-1"},
        )


def run_demo(database: Path) -> dict[str, Any]:
    """执行一次 success-before-ack 崩溃，并返回可审计事件时间线。"""

    if database.exists():
        raise ValueError(f"refusing to reuse existing database: {database}")

    provider = SimulatedIdempotentProvider()
    worker_a = SQLiteTransactionalOutbox(database)
    # effect 与任务状态在同一 SQLite 事务提交，避免只更新任务却漏发消息。
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
    # claim 给 effect 加短租约，其他 worker 在租约有效期内不能同时发送。
    first = worker_a.claim_due("worker-a", now=10, lease_seconds=5)
    if first is None:
        raise RuntimeError("worker-a did not claim the fixture effect")
    first_receipt = provider.send(
        idempotency_key=first.provider_idempotency_key,
        destination=first.destination,
        payload=first.payload,
    )

    # 模拟崩溃：provider 已成功，但本地还没来得及写 delivered acknowledgement。
    del worker_a
    worker_b = SQLiteTransactionalOutbox(database)
    # now=16 时原租约已过期，worker-b 会重新 claim 并进行 at-least-once 重试。
    second = worker_b.claim_due("worker-b", now=16, lease_seconds=5)
    if second is None:
        raise RuntimeError("worker-b did not reclaim the expired fixture effect")
    second_receipt = provider.send(
        idempotency_key=second.provider_idempotency_key,
        destination=second.destination,
        payload=second.payload,
    )
    # provider 返回相同 receipt 后，worker-b 才把 outbox 记录推进到 delivered。
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
    """要求一个新 SQLite 路径，避免覆盖已有演示账本。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="new SQLite path; an existing file is rejected",
    )
    return parser.parse_args()


def main() -> None:
    """执行崩溃恢复实验并输出 effect 与 event 记录。"""

    args = parse_args()
    print(json.dumps(run_demo(args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
