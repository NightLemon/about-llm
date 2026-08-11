"""A durable SQLite execution ledger with atomic call-id claims."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from about_llm.agents.runtime import (
    IdempotencyConflict,
    LedgerEntry,
    LedgerState,
    freeze_json_value,
)
from about_llm.llmops import canonical_json_bytes


@dataclass(frozen=True)
class PendingCall:
    call_id: str
    fingerprint: str
    created_at: float
    age_seconds: float


@dataclass(frozen=True)
class ReconciliationEvent:
    call_id: str
    resolution: str
    note: str
    resolved_at: float


class SQLiteLedger:
    """Persist pending/completed calls across processes and restarts.

    SQLite protects the claim, not the external side effect itself. A crash
    after the remote action but before complete leaves PENDING, intentionally
    requiring reconciliation instead of risking a duplicate action.
    """

    def __init__(self, path: Path, *, timeout_seconds: float = 5.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.path = path
        self.timeout_seconds = timeout_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
                    value_json TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_call_reconciliations (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL UNIQUE REFERENCES tool_calls(call_id),
                    resolution TEXT NOT NULL CHECK (
                        resolution IN ('externally_confirmed', 'abandoned', 'compensated')
                    ),
                    note TEXT NOT NULL,
                    resolved_at REAL NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def lookup(self, call_id: str) -> LedgerEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.fingerprint,
                       CASE
                           WHEN r.resolution = 'abandoned' THEN 'abandoned'
                           WHEN r.resolution = 'compensated' THEN 'compensated'
                           ELSE c.state
                       END AS state,
                       c.value_json
                FROM tool_calls AS c
                LEFT JOIN tool_call_reconciliations AS r USING (call_id)
                WHERE c.call_id = ?
                """,
                (call_id,),
            ).fetchone()
        return self._to_entry(row) if row is not None else None

    def claim(self, call_id: str, fingerprint: str) -> tuple[LedgerEntry, bool]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO tool_calls
                    (call_id, fingerprint, state, created_at)
                VALUES (?, ?, 'pending', ?)
                """,
                (call_id, fingerprint, time.time()),
            )
            row = connection.execute(
                "SELECT fingerprint, state, value_json FROM tool_calls WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("claimed call was not readable")
        return self._to_entry(row), cursor.rowcount == 1

    def list_stale_pending(
        self, *, older_than_seconds: float, now: float | None = None
    ) -> tuple[PendingCall, ...]:
        """List unresolved calls old enough for an operator to investigate."""
        if older_than_seconds < 0:
            raise ValueError("older_than_seconds must be non-negative")
        current_time = time.time() if now is None else now
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.call_id, c.fingerprint, c.created_at
                FROM tool_calls AS c
                LEFT JOIN tool_call_reconciliations AS r USING (call_id)
                WHERE c.state = 'pending'
                  AND r.call_id IS NULL
                  AND c.created_at <= ?
                ORDER BY c.created_at, c.call_id
                """,
                (current_time - older_than_seconds,),
            ).fetchall()
        return tuple(
            PendingCall(
                call_id=row["call_id"],
                fingerprint=row["fingerprint"],
                created_at=row["created_at"],
                age_seconds=max(0.0, current_time - row["created_at"]),
            )
            for row in rows
        )

    def resolve_external_completion(self, call_id: str, value: Any, *, note: str) -> None:
        """Record that an operator verified the external side effect succeeded."""
        if not note.strip():
            raise ValueError("a reconciliation note is required")
        try:
            value_json = canonical_json_bytes(value).decode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(f"reconciled result must be strict JSON: {error}") from error
        resolved_at = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE tool_calls
                SET state = 'completed', value_json = ?, completed_at = ?
                WHERE call_id = ? AND state = 'pending'
                  AND NOT EXISTS (
                    SELECT 1 FROM tool_call_reconciliations WHERE call_id = ?
                  )
                """,
                (value_json, resolved_at, call_id, call_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise IdempotencyConflict(f"call_id {call_id!r} is not unresolved pending")
            connection.execute(
                """
                INSERT INTO tool_call_reconciliations
                    (call_id, resolution, note, resolved_at)
                VALUES (?, 'externally_confirmed', ?, ?)
                """,
                (call_id, note, resolved_at),
            )
            connection.commit()

    def resolve_without_completion(
        self, call_id: str, *, note: str, compensated: bool = False
    ) -> None:
        """Close an uncertain call while retaining the immutable original claim."""
        if not note.strip():
            raise ValueError("a reconciliation note is required")
        resolution = "compensated" if compensated else "abandoned"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM tool_calls WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            existing_resolution = connection.execute(
                "SELECT 1 FROM tool_call_reconciliations WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            if row is None or row["state"] != "pending" or existing_resolution is not None:
                connection.rollback()
                raise IdempotencyConflict(f"call_id {call_id!r} is not unresolved pending")
            connection.execute(
                """
                INSERT INTO tool_call_reconciliations
                    (call_id, resolution, note, resolved_at)
                VALUES (?, ?, ?, ?)
                """,
                (call_id, resolution, note, time.time()),
            )
            connection.commit()

    def reconciliation_history(self, call_id: str) -> tuple[ReconciliationEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT call_id, resolution, note, resolved_at
                FROM tool_call_reconciliations
                WHERE call_id = ?
                ORDER BY event_id
                """,
                (call_id,),
            ).fetchall()
        return tuple(
            ReconciliationEvent(
                call_id=row["call_id"],
                resolution=row["resolution"],
                note=row["note"],
                resolved_at=row["resolved_at"],
            )
            for row in rows
        )

    def complete(self, call_id: str, fingerprint: str, value: Any) -> None:
        try:
            value_json = canonical_json_bytes(value).decode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"tool result must be strict JSON for SQLite ledger: {error}"
            ) from error
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE tool_calls
                SET state = 'completed', value_json = ?, completed_at = ?
                WHERE call_id = ? AND fingerprint = ? AND state = 'pending'
                  AND NOT EXISTS (
                    SELECT 1 FROM tool_call_reconciliations WHERE call_id = ?
                  )
                """,
                (value_json, time.time(), call_id, fingerprint, call_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise IdempotencyConflict(
                    f"call_id {call_id!r} is missing, conflicting, or already completed"
                )
            connection.commit()

    @staticmethod
    def _to_entry(row: sqlite3.Row) -> LedgerEntry:
        state = LedgerState(row["state"])
        value = (
            freeze_json_value(json.loads(row["value_json"]))
            if row["value_json"] is not None
            else None
        )
        return LedgerEntry(fingerprint=row["fingerprint"], state=state, value=value)
