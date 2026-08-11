"""SQLite transactional-outbox reference with explicit at-least-once semantics."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from about_llm.agents.runtime import IdempotencyConflict, freeze_json_value
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

_ERROR_CODE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


class OutboxState(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DELIVERED = "delivered"
    DEAD_LETTER = "dead_letter"


@dataclass(frozen=True)
class EffectRequest:
    effect_id: str
    execution_fingerprint: str
    destination: str
    payload: Mapping[str, Any]
    _payload_json: str = field(init=False, repr=False)
    _effect_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for value, label in (
            (self.effect_id, "effect_id"),
            (self.destination, "destination"),
        ):
            if not value.strip():
                raise ValueError(f"{label} cannot be empty")
        _validate_sha256(self.execution_fingerprint, "execution_fingerprint")
        payload_json, snapshot = _canonical_object(self.payload, "effect payload")
        object.__setattr__(self, "payload", freeze_json_value(snapshot))
        object.__setattr__(self, "_payload_json", payload_json)
        object.__setattr__(
            self,
            "_effect_fingerprint",
            "sha256:"
            + artifact_fingerprint(
                {
                    "effect_id": self.effect_id,
                    "execution_fingerprint": self.execution_fingerprint,
                    "destination": self.destination,
                    "payload": snapshot,
                }
            ),
        )

    @property
    def effect_fingerprint(self) -> str:
        return self._effect_fingerprint


@dataclass(frozen=True)
class OutboxDelivery:
    effect_id: str
    task_id: str
    execution_fingerprint: str
    effect_fingerprint: str
    destination: str
    payload: Any
    attempt: int
    lease_owner: str
    lease_until: float
    provider_idempotency_key: str


@dataclass(frozen=True)
class OutboxRecord:
    effect_id: str
    task_id: str
    execution_fingerprint: str
    effect_fingerprint: str
    destination: str
    payload: Any
    state: OutboxState
    attempt_count: int
    next_attempt_at: float
    lease_owner: str | None
    lease_until: float | None
    receipt: Any | None
    last_error_code: str | None
    created_at: float
    updated_at: float
    delivered_at: float | None


@dataclass(frozen=True)
class OutboxEvent:
    event_id: int
    effect_id: str
    event_type: str
    worker_id: str | None
    detail: Any
    occurred_at: float


class SQLiteTransactionalOutbox:
    """Atomically commit one local task state and one externally delivered effect.

    Lease expiry deliberately permits redelivery. Exactly-once external effects
    require the provider to honor ``effect_id`` as an idempotency key.
    """

    def __init__(self, path: Path, *, timeout_seconds: float = 5.0) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_local_tasks (
                    task_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    state_fingerprint TEXT NOT NULL,
                    committed_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_outbox (
                    effect_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE
                        REFERENCES agent_local_tasks(task_id),
                    execution_fingerprint TEXT NOT NULL,
                    effect_fingerprint TEXT NOT NULL UNIQUE,
                    destination TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('pending', 'claimed', 'delivered', 'dead_letter')
                    ),
                    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                    next_attempt_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_until REAL,
                    receipt_json TEXT,
                    last_error_code TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    delivered_at REAL,
                    CHECK (
                        (state = 'claimed' AND lease_owner IS NOT NULL AND lease_until IS NOT NULL)
                        OR
                        (state != 'claimed' AND lease_owner IS NULL AND lease_until IS NULL)
                    ),
                    CHECK (
                        (
                            state = 'delivered'
                            AND receipt_json IS NOT NULL
                            AND delivered_at IS NOT NULL
                        )
                        OR
                        (state != 'delivered' AND receipt_json IS NULL AND delivered_at IS NULL)
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_agent_outbox_due
                    ON agent_outbox(state, next_attempt_at, created_at, effect_id);

                CREATE TABLE IF NOT EXISTS agent_outbox_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    effect_id TEXT NOT NULL REFERENCES agent_outbox(effect_id),
                    event_type TEXT NOT NULL,
                    worker_id TEXT,
                    detail_json TEXT NOT NULL,
                    occurred_at REAL NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def commit_task_effect(
        self,
        task_id: str,
        local_state: Mapping[str, Any],
        effect: EffectRequest,
        *,
        now: float | None = None,
    ) -> bool:
        """Commit local task state and its single outbox effect in one transaction."""

        if not task_id.strip():
            raise ValueError("task_id cannot be empty")
        committed_at = _timestamp(now)
        state_json, state_snapshot = _canonical_object(local_state, "local task state")
        state_fingerprint = "sha256:" + artifact_fingerprint(state_snapshot)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            task_row = connection.execute(
                "SELECT state_fingerprint FROM agent_local_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            effect_row = connection.execute(
                """
                SELECT task_id, effect_fingerprint
                FROM agent_outbox
                WHERE effect_id = ? OR task_id = ?
                """,
                (effect.effect_id, task_id),
            ).fetchone()
            if task_row is not None or effect_row is not None:
                if (
                    task_row is not None
                    and effect_row is not None
                    and task_row["state_fingerprint"] == state_fingerprint
                    and effect_row["task_id"] == task_id
                    and effect_row["effect_fingerprint"] == effect.effect_fingerprint
                ):
                    connection.rollback()
                    return False
                connection.rollback()
                raise IdempotencyConflict(
                    "task_id/effect_id already exists with different or partial identity"
                )

            connection.execute(
                """
                INSERT INTO agent_local_tasks
                    (task_id, state_json, state_fingerprint, committed_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, state_json, state_fingerprint, committed_at),
            )
            connection.execute(
                """
                INSERT INTO agent_outbox (
                    effect_id, task_id, execution_fingerprint, effect_fingerprint,
                    destination, payload_json, state, next_attempt_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    effect.effect_id,
                    task_id,
                    effect.execution_fingerprint,
                    effect.effect_fingerprint,
                    effect.destination,
                    effect._payload_json,
                    committed_at,
                    committed_at,
                    committed_at,
                ),
            )
            self._append_event(
                connection,
                effect.effect_id,
                "enqueued",
                worker_id=None,
                detail={
                    "task_id": task_id,
                    "effect_fingerprint": effect.effect_fingerprint,
                },
                occurred_at=committed_at,
            )
            connection.commit()
        return True

    def claim_due(
        self,
        worker_id: str,
        *,
        now: float | None = None,
        lease_seconds: float,
    ) -> OutboxDelivery | None:
        if not worker_id.strip():
            raise ValueError("worker_id cannot be empty")
        current_time = _timestamp(now)
        lease_duration = _positive_duration(lease_seconds, "lease_seconds")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expired_rows = connection.execute(
                """
                SELECT effect_id, lease_owner, attempt_count
                FROM agent_outbox
                WHERE state = 'claimed' AND lease_until <= ?
                ORDER BY effect_id
                """,
                (current_time,),
            ).fetchall()
            for expired in expired_rows:
                connection.execute(
                    """
                    UPDATE agent_outbox
                    SET state = 'pending', lease_owner = NULL, lease_until = NULL,
                        next_attempt_at = ?, updated_at = ?
                    WHERE effect_id = ? AND state = 'claimed' AND lease_until <= ?
                    """,
                    (
                        current_time,
                        current_time,
                        expired["effect_id"],
                        current_time,
                    ),
                )
                self._append_event(
                    connection,
                    expired["effect_id"],
                    "lease_expired",
                    worker_id=expired["lease_owner"],
                    detail={"attempt": expired["attempt_count"]},
                    occurred_at=current_time,
                )

            row = connection.execute(
                """
                SELECT * FROM agent_outbox
                WHERE state = 'pending' AND next_attempt_at <= ?
                ORDER BY next_attempt_at, created_at, effect_id
                LIMIT 1
                """,
                (current_time,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            lease_until = current_time + lease_duration
            attempt = int(row["attempt_count"]) + 1
            cursor = connection.execute(
                """
                UPDATE agent_outbox
                SET state = 'claimed', attempt_count = ?, lease_owner = ?,
                    lease_until = ?, updated_at = ?
                WHERE effect_id = ? AND state = 'pending'
                """,
                (attempt, worker_id, lease_until, current_time, row["effect_id"]),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("outbox claim lost despite BEGIN IMMEDIATE")
            self._append_event(
                connection,
                row["effect_id"],
                "claimed",
                worker_id=worker_id,
                detail={"attempt": attempt, "lease_until": lease_until},
                occurred_at=current_time,
            )
            connection.commit()
        return OutboxDelivery(
            effect_id=row["effect_id"],
            task_id=row["task_id"],
            execution_fingerprint=row["execution_fingerprint"],
            effect_fingerprint=row["effect_fingerprint"],
            destination=row["destination"],
            payload=_load_json(row["payload_json"]),
            attempt=attempt,
            lease_owner=worker_id,
            lease_until=lease_until,
            provider_idempotency_key=row["effect_id"],
        )

    def renew_lease(
        self,
        effect_id: str,
        worker_id: str,
        *,
        now: float | None = None,
        lease_seconds: float,
    ) -> float:
        current_time = _timestamp(now)
        lease_until = current_time + _positive_duration(lease_seconds, "lease_seconds")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_active_claim(connection, effect_id, worker_id, current_time)
            connection.execute(
                """
                UPDATE agent_outbox SET lease_until = ?, updated_at = ?
                WHERE effect_id = ?
                """,
                (lease_until, current_time, effect_id),
            )
            self._append_event(
                connection,
                effect_id,
                "lease_renewed",
                worker_id=worker_id,
                detail={"lease_until": lease_until},
                occurred_at=current_time,
            )
            connection.commit()
        return lease_until

    def mark_delivered(
        self,
        effect_id: str,
        worker_id: str,
        receipt: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> None:
        delivered_at = _timestamp(now)
        receipt_json, receipt_snapshot = _canonical_object(receipt, "provider receipt")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_active_claim(connection, effect_id, worker_id, delivered_at)
            connection.execute(
                """
                UPDATE agent_outbox
                SET state = 'delivered', receipt_json = ?, delivered_at = ?,
                    lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE effect_id = ?
                """,
                (receipt_json, delivered_at, delivered_at, effect_id),
            )
            self._append_event(
                connection,
                effect_id,
                "delivered",
                worker_id=worker_id,
                detail={
                    "receipt_fingerprint": "sha256:"
                    + artifact_fingerprint(receipt_snapshot)
                },
                occurred_at=delivered_at,
            )
            connection.commit()

    def schedule_retry(
        self,
        effect_id: str,
        worker_id: str,
        *,
        error_code: str,
        retry_at: float | None,
        terminal: bool = False,
        now: float | None = None,
    ) -> None:
        if not _ERROR_CODE.fullmatch(error_code):
            raise ValueError("error_code must be a bounded lowercase machine token")
        if not isinstance(terminal, bool):
            raise ValueError("terminal must be a boolean")
        current_time = _timestamp(now)
        if terminal:
            if retry_at is not None:
                raise ValueError("terminal failure cannot have retry_at")
            next_attempt_at = current_time
            new_state = OutboxState.DEAD_LETTER
            event_type = "dead_lettered"
        else:
            if retry_at is None:
                raise ValueError("retryable failure requires retry_at")
            next_attempt_at = _timestamp(retry_at)
            if next_attempt_at < current_time:
                raise ValueError("retry_at cannot be earlier than now")
            new_state = OutboxState.PENDING
            event_type = "retry_scheduled"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_active_claim(connection, effect_id, worker_id, current_time)
            connection.execute(
                """
                UPDATE agent_outbox
                SET state = ?, next_attempt_at = ?, lease_owner = NULL,
                    lease_until = NULL, last_error_code = ?, updated_at = ?
                WHERE effect_id = ?
                """,
                (
                    new_state.value,
                    next_attempt_at,
                    error_code,
                    current_time,
                    effect_id,
                ),
            )
            self._append_event(
                connection,
                effect_id,
                event_type,
                worker_id=worker_id,
                detail={"error_code": error_code, "retry_at": retry_at},
                occurred_at=current_time,
            )
            connection.commit()

    def get(self, effect_id: str) -> OutboxRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_outbox WHERE effect_id = ?", (effect_id,)
            ).fetchone()
        return self._to_record(row) if row is not None else None

    def task_state(self, task_id: str) -> Any | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM agent_local_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return _load_json(row["state_json"]) if row is not None else None

    def events(self, effect_id: str) -> tuple[OutboxEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, effect_id, event_type, worker_id,
                       detail_json, occurred_at
                FROM agent_outbox_events
                WHERE effect_id = ?
                ORDER BY event_id
                """,
                (effect_id,),
            ).fetchall()
        return tuple(
            OutboxEvent(
                event_id=row["event_id"],
                effect_id=row["effect_id"],
                event_type=row["event_type"],
                worker_id=row["worker_id"],
                detail=_load_json(row["detail_json"]),
                occurred_at=row["occurred_at"],
            )
            for row in rows
        )

    @staticmethod
    def _require_active_claim(
        connection: sqlite3.Connection,
        effect_id: str,
        worker_id: str,
        now: float,
    ) -> sqlite3.Row:
        if not effect_id.strip() or not worker_id.strip():
            raise ValueError("effect_id and worker_id cannot be empty")
        row = connection.execute(
            """
            SELECT * FROM agent_outbox
            WHERE effect_id = ? AND state = 'claimed' AND lease_owner = ?
            """,
            (effect_id, worker_id),
        ).fetchone()
        if row is None or row["lease_until"] <= now:
            raise IdempotencyConflict(
                "effect is not held by this worker with an unexpired lease"
            )
        return cast(sqlite3.Row, row)

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        effect_id: str,
        event_type: str,
        *,
        worker_id: str | None,
        detail: Mapping[str, Any],
        occurred_at: float,
    ) -> None:
        detail_json, _ = _canonical_object(detail, "outbox event detail")
        connection.execute(
            """
            INSERT INTO agent_outbox_events
                (effect_id, event_type, worker_id, detail_json, occurred_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (effect_id, event_type, worker_id, detail_json, occurred_at),
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> OutboxRecord:
        return OutboxRecord(
            effect_id=row["effect_id"],
            task_id=row["task_id"],
            execution_fingerprint=row["execution_fingerprint"],
            effect_fingerprint=row["effect_fingerprint"],
            destination=row["destination"],
            payload=_load_json(row["payload_json"]),
            state=OutboxState(row["state"]),
            attempt_count=row["attempt_count"],
            next_attempt_at=row["next_attempt_at"],
            lease_owner=row["lease_owner"],
            lease_until=row["lease_until"],
            receipt=(
                _load_json(row["receipt_json"])
                if row["receipt_json"] is not None
                else None
            ),
            last_error_code=row["last_error_code"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            delivered_at=row["delivered_at"],
        )


def _canonical_object(
    value: Mapping[str, Any], label: str
) -> tuple[str, dict[str, Any]]:
    try:
        encoded = canonical_json_bytes(value)
        snapshot = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a strict JSON object: {error}") from error
    if not isinstance(snapshot, dict):
        raise ValueError(f"{label} must be a strict JSON object")
    return encoded.decode("utf-8"), snapshot


def _load_json(value: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    return freeze_json_value(
        json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    )


def _timestamp(value: float | None) -> float:
    timestamp = time.time() if value is None else value
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
    ):
        raise ValueError("timestamp must be finite")
    return float(timestamp)


def _positive_duration(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _validate_sha256(value: str, label: str) -> None:
    prefix = "sha256:"
    digest = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} must be a lowercase sha256: digest")
