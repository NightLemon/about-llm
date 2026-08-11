"""Durable SQLite token/cost reservations for concurrent cloud workers."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from about_llm.integrations.cloud_api import RequestSpec
from about_llm.integrations.usage_budget import (
    BudgetReservation,
    PostCallBudgetExceededError,
    TokenPricingSnapshot,
    UsageBudgetExceededError,
    UsageBudgetLimits,
    UsageBudgetSnapshot,
    UsageBudgetStateError,
    _non_empty,
    _non_negative_integer,
    _sha256_fingerprint,
    cloud_request_budget_fingerprint,
    request_maximum_output_tokens,
    request_model_identifier,
)
from about_llm.llmops import artifact_fingerprint

SQLITE_USAGE_BUDGET_SCHEMA_VERSION = "about-llm.cloud-usage-budget.sqlite.v1"
_MAX_SQLITE_INTEGER = (1 << 63) - 1
ReservationState = Literal["active", "settled", "uncertain", "cancelled"]
BudgetEventType = Literal["reserved", "settled", "uncertain", "cancelled"]


@dataclass(frozen=True)
class DurableBudgetReservationRecord:
    reservation_id: str
    request_fingerprint: str
    budget_config_fingerprint: str
    input_tokens: int
    maximum_output_tokens: int
    maximum_estimated_microusd: int
    state: ReservationState
    actual_input_tokens: int | None
    actual_output_tokens: int | None
    actual_estimated_microusd: int | None
    created_at_utc: str
    terminal_at_utc: str | None


@dataclass(frozen=True)
class DurableBudgetEvent:
    event_id: int
    reservation_id: str
    event_type: BudgetEventType
    input_tokens: int
    output_tokens: int
    estimated_microusd: int
    occurred_at_utc: str


class SQLiteUsageBudgetLedger:
    """Persist atomic reservations and reconciliation state across processes.

    The database protects local capacity accounting only. It cannot make a
    remote model call atomic with SQLite, authenticate billing data, or prove
    that a provider honored cancellation or reported complete usage.
    """

    def __init__(
        self,
        path: Path,
        *,
        limits: UsageBudgetLimits,
        pricing: TokenPricingSnapshot,
        timeout_seconds: float = 5.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be pathlib.Path")
        if not isinstance(limits, UsageBudgetLimits):
            raise TypeError("limits must be UsageBudgetLimits")
        if not isinstance(pricing, TokenPricingSnapshot):
            raise TypeError("pricing must be TokenPricingSnapshot")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self._validate_sqlite_config(limits, pricing)
        self.path = path
        self.limits = limits
        self.pricing = pricing
        self.timeout_seconds = float(timeout_seconds)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._config_identity = self._build_config_identity(limits, pricing)
        self.config_fingerprint = "sha256:" + artifact_fingerprint(
            self._config_identity
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._create_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cloud_usage_budget_config WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self._insert_config(connection)
            else:
                self._verify_config_row(row)
            connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cloud_usage_budget_config (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version TEXT NOT NULL,
                pricing_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                pricing_revision TEXT NOT NULL,
                pricing_checked_at_utc TEXT NOT NULL,
                input_microusd_per_million INTEGER NOT NULL CHECK (
                    input_microusd_per_million >= 0
                ),
                output_microusd_per_million INTEGER NOT NULL CHECK (
                    output_microusd_per_million >= 0
                ),
                max_input_tokens INTEGER CHECK (max_input_tokens >= 0),
                max_output_tokens INTEGER CHECK (max_output_tokens >= 0),
                max_estimated_microusd INTEGER CHECK (
                    max_estimated_microusd >= 0
                ),
                config_fingerprint TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cloud_usage_budget_reservations (
                reservation_id TEXT PRIMARY KEY,
                request_fingerprint TEXT NOT NULL,
                budget_config_fingerprint TEXT NOT NULL,
                reserved_input_tokens INTEGER NOT NULL CHECK (
                    reserved_input_tokens >= 0
                ),
                reserved_output_tokens INTEGER NOT NULL CHECK (
                    reserved_output_tokens >= 0
                ),
                reserved_estimated_microusd INTEGER NOT NULL CHECK (
                    reserved_estimated_microusd >= 0
                ),
                state TEXT NOT NULL CHECK (
                    state IN ('active', 'settled', 'uncertain', 'cancelled')
                ),
                actual_input_tokens INTEGER CHECK (actual_input_tokens >= 0),
                actual_output_tokens INTEGER CHECK (actual_output_tokens >= 0),
                actual_estimated_microusd INTEGER CHECK (
                    actual_estimated_microusd >= 0
                ),
                created_at_utc TEXT NOT NULL,
                terminal_at_utc TEXT,
                CHECK (
                    (
                        state = 'active'
                        AND actual_input_tokens IS NULL
                        AND actual_output_tokens IS NULL
                        AND actual_estimated_microusd IS NULL
                        AND terminal_at_utc IS NULL
                    ) OR (
                        state = 'cancelled'
                        AND actual_input_tokens IS NULL
                        AND actual_output_tokens IS NULL
                        AND actual_estimated_microusd IS NULL
                        AND terminal_at_utc IS NOT NULL
                    ) OR (
                        state IN ('settled', 'uncertain')
                        AND actual_input_tokens IS NOT NULL
                        AND actual_output_tokens IS NOT NULL
                        AND actual_estimated_microusd IS NOT NULL
                        AND terminal_at_utc IS NOT NULL
                    )
                )
            );

            CREATE INDEX IF NOT EXISTS idx_cloud_usage_budget_state
                ON cloud_usage_budget_reservations(state, created_at_utc, reservation_id);

            CREATE TABLE IF NOT EXISTS cloud_usage_budget_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                reservation_id TEXT NOT NULL REFERENCES
                    cloud_usage_budget_reservations(reservation_id),
                event_type TEXT NOT NULL CHECK (
                    event_type IN ('reserved', 'settled', 'uncertain', 'cancelled')
                ),
                input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
                output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
                estimated_microusd INTEGER NOT NULL CHECK (estimated_microusd >= 0),
                occurred_at_utc TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def _validate_sqlite_config(
        limits: UsageBudgetLimits, pricing: TokenPricingSnapshot
    ) -> None:
        for name, value in (
            ("input_microusd_per_million", pricing.input_microusd_per_million),
            ("output_microusd_per_million", pricing.output_microusd_per_million),
            ("max_input_tokens", limits.max_input_tokens),
            ("max_output_tokens", limits.max_output_tokens),
            ("max_estimated_microusd", limits.max_estimated_microusd),
        ):
            if value is not None and value > _MAX_SQLITE_INTEGER:
                raise ValueError(f"{name} exceeds SQLite signed-integer range")

    @staticmethod
    def _build_config_identity(
        limits: UsageBudgetLimits, pricing: TokenPricingSnapshot
    ) -> dict[str, object]:
        return {
            "schema_version": SQLITE_USAGE_BUDGET_SCHEMA_VERSION,
            "pricing": {
                "pricing_id": pricing.pricing_id,
                "provider": pricing.provider,
                "model": pricing.model,
                "revision": pricing.revision,
                "checked_at_utc": _utc_datetime(pricing.checked_at),
                "input_microusd_per_million": pricing.input_microusd_per_million,
                "output_microusd_per_million": pricing.output_microusd_per_million,
            },
            "limits": {
                "max_input_tokens": limits.max_input_tokens,
                "max_output_tokens": limits.max_output_tokens,
                "max_estimated_microusd": limits.max_estimated_microusd,
            },
        }

    def _insert_config(self, connection: sqlite3.Connection) -> None:
        pricing = cast(dict[str, object], self._config_identity["pricing"])
        limits = cast(dict[str, object], self._config_identity["limits"])
        connection.execute(
            """
            INSERT INTO cloud_usage_budget_config (
                singleton, schema_version, pricing_id, provider, model,
                pricing_revision, pricing_checked_at_utc,
                input_microusd_per_million, output_microusd_per_million,
                max_input_tokens, max_output_tokens, max_estimated_microusd,
                config_fingerprint
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._config_identity["schema_version"],
                pricing["pricing_id"],
                pricing["provider"],
                pricing["model"],
                pricing["revision"],
                pricing["checked_at_utc"],
                pricing["input_microusd_per_million"],
                pricing["output_microusd_per_million"],
                limits["max_input_tokens"],
                limits["max_output_tokens"],
                limits["max_estimated_microusd"],
                self.config_fingerprint,
            ),
        )

    def _verify_config(self, connection: sqlite3.Connection) -> None:
        row = connection.execute(
            "SELECT * FROM cloud_usage_budget_config WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise UsageBudgetStateError("durable budget configuration is missing")
        self._verify_config_row(row)

    def _verify_config_row(self, row: sqlite3.Row) -> None:
        stored_identity = {
            "schema_version": row["schema_version"],
            "pricing": {
                "pricing_id": row["pricing_id"],
                "provider": row["provider"],
                "model": row["model"],
                "revision": row["pricing_revision"],
                "checked_at_utc": row["pricing_checked_at_utc"],
                "input_microusd_per_million": row[
                    "input_microusd_per_million"
                ],
                "output_microusd_per_million": row[
                    "output_microusd_per_million"
                ],
            },
            "limits": {
                "max_input_tokens": row["max_input_tokens"],
                "max_output_tokens": row["max_output_tokens"],
                "max_estimated_microusd": row["max_estimated_microusd"],
            },
        }
        stored_fingerprint = "sha256:" + artifact_fingerprint(stored_identity)
        if row["config_fingerprint"] != stored_fingerprint:
            raise UsageBudgetStateError(
                "durable budget configuration fingerprint is inconsistent"
            )
        if stored_identity != self._config_identity:
            raise UsageBudgetStateError(
                "durable budget configuration does not match requested pricing/limits"
            )

    def reserve(
        self,
        reservation_id: str,
        *,
        request_fingerprint: str,
        estimated_input_tokens: int,
        maximum_output_tokens: int,
    ) -> BudgetReservation:
        _non_empty(reservation_id, "reservation_id")
        _sha256_fingerprint(request_fingerprint, "request_fingerprint")
        _sqlite_non_negative(estimated_input_tokens, "estimated_input_tokens")
        _sqlite_non_negative(maximum_output_tokens, "maximum_output_tokens")
        estimated_cost = self.pricing.quote_microusd(
            estimated_input_tokens, maximum_output_tokens
        )
        _sqlite_non_negative(estimated_cost, "maximum_estimated_microusd")
        reservation = BudgetReservation(
            reservation_id=reservation_id,
            request_fingerprint=request_fingerprint,
            input_tokens=estimated_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
            maximum_estimated_microusd=estimated_cost,
            pricing_id=self.pricing.pricing_id,
        )
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_config(connection)
            if connection.execute(
                "SELECT 1 FROM cloud_usage_budget_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone() is not None:
                connection.rollback()
                raise UsageBudgetStateError("reservation_id has already been used")
            snapshot = self._snapshot(connection)
            projected = (
                snapshot.committed_input_tokens
                + snapshot.reserved_input_tokens
                + estimated_input_tokens,
                snapshot.committed_output_tokens
                + snapshot.reserved_output_tokens
                + maximum_output_tokens,
                snapshot.committed_estimated_microusd
                + snapshot.reserved_estimated_microusd
                + estimated_cost,
            )
            _sqlite_totals(projected)
            if self._exceeds_limits(*projected):
                connection.rollback()
                raise UsageBudgetExceededError(
                    "reservation would exceed a durable token or estimated-cost limit"
                )
            connection.execute(
                """
                INSERT INTO cloud_usage_budget_reservations (
                    reservation_id, request_fingerprint, budget_config_fingerprint,
                    reserved_input_tokens, reserved_output_tokens,
                    reserved_estimated_microusd, state, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    reservation_id,
                    request_fingerprint,
                    self.config_fingerprint,
                    estimated_input_tokens,
                    maximum_output_tokens,
                    estimated_cost,
                    now,
                ),
            )
            self._insert_event(
                connection,
                reservation_id=reservation_id,
                event_type="reserved",
                input_tokens=estimated_input_tokens,
                output_tokens=maximum_output_tokens,
                estimated_microusd=estimated_cost,
                occurred_at_utc=now,
            )
            connection.commit()
        return reservation

    def reserve_request(
        self,
        reservation_id: str,
        *,
        request: RequestSpec,
        billing_scope: str,
        estimated_input_tokens: int,
    ) -> BudgetReservation:
        request_model = request_model_identifier(request)
        if request_model != self.pricing.model:
            raise ValueError(
                "request model must exactly match the pricing snapshot model"
            )
        return self.reserve(
            reservation_id,
            request_fingerprint=cloud_request_budget_fingerprint(
                request, billing_scope=billing_scope
            ),
            estimated_input_tokens=estimated_input_tokens,
            maximum_output_tokens=request_maximum_output_tokens(request),
        )

    def settle(
        self,
        reservation_id: str,
        *,
        request_fingerprint: str,
        actual_input_tokens: int,
        actual_output_tokens: int,
    ) -> UsageBudgetSnapshot:
        _non_empty(reservation_id, "reservation_id")
        _sha256_fingerprint(request_fingerprint, "request_fingerprint")
        _sqlite_non_negative(actual_input_tokens, "actual_input_tokens")
        _sqlite_non_negative(actual_output_tokens, "actual_output_tokens")
        actual_cost = self.pricing.quote_microusd(
            actual_input_tokens, actual_output_tokens
        )
        _sqlite_non_negative(actual_cost, "actual_estimated_microusd")
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_config(connection)
            self._require_active(connection, reservation_id, request_fingerprint)
            current = self._snapshot(connection)
            projected_committed = (
                current.committed_input_tokens + actual_input_tokens,
                current.committed_output_tokens + actual_output_tokens,
                current.committed_estimated_microusd + actual_cost,
            )
            _sqlite_totals(projected_committed)
            connection.execute(
                """
                UPDATE cloud_usage_budget_reservations
                SET state = 'settled', actual_input_tokens = ?,
                    actual_output_tokens = ?, actual_estimated_microusd = ?,
                    terminal_at_utc = ?
                WHERE reservation_id = ?
                """,
                (
                    actual_input_tokens,
                    actual_output_tokens,
                    actual_cost,
                    now,
                    reservation_id,
                ),
            )
            self._insert_event(
                connection,
                reservation_id=reservation_id,
                event_type="settled",
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                estimated_microusd=actual_cost,
                occurred_at_utc=now,
            )
            snapshot = self._snapshot(connection)
            connection.commit()
        if snapshot.over_limit:
            raise PostCallBudgetExceededError(snapshot)
        return snapshot

    def mark_usage_uncertain(
        self, reservation_id: str, *, request_fingerprint: str
    ) -> UsageBudgetSnapshot:
        return self._terminal_from_reservation(
            reservation_id,
            request_fingerprint=request_fingerprint,
            state="uncertain",
        )

    def cancel_before_send(
        self, reservation_id: str, *, request_fingerprint: str
    ) -> UsageBudgetSnapshot:
        return self._terminal_from_reservation(
            reservation_id,
            request_fingerprint=request_fingerprint,
            state="cancelled",
        )

    def _terminal_from_reservation(
        self,
        reservation_id: str,
        *,
        request_fingerprint: str,
        state: Literal["uncertain", "cancelled"],
    ) -> UsageBudgetSnapshot:
        _non_empty(reservation_id, "reservation_id")
        _sha256_fingerprint(request_fingerprint, "request_fingerprint")
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._verify_config(connection)
            row = self._require_active(
                connection, reservation_id, request_fingerprint
            )
            if state == "uncertain":
                actual_input = cast(int, row["reserved_input_tokens"])
                actual_output = cast(int, row["reserved_output_tokens"])
                actual_cost = cast(int, row["reserved_estimated_microusd"])
                connection.execute(
                    """
                    UPDATE cloud_usage_budget_reservations
                    SET state = 'uncertain', actual_input_tokens = ?,
                        actual_output_tokens = ?, actual_estimated_microusd = ?,
                        terminal_at_utc = ?
                    WHERE reservation_id = ?
                    """,
                    (
                        actual_input,
                        actual_output,
                        actual_cost,
                        now,
                        reservation_id,
                    ),
                )
                event_type: BudgetEventType = "uncertain"
            else:
                actual_input = cast(int, row["reserved_input_tokens"])
                actual_output = cast(int, row["reserved_output_tokens"])
                actual_cost = cast(int, row["reserved_estimated_microusd"])
                connection.execute(
                    """
                    UPDATE cloud_usage_budget_reservations
                    SET state = 'cancelled', terminal_at_utc = ?
                    WHERE reservation_id = ?
                    """,
                    (now, reservation_id),
                )
                event_type = "cancelled"
            self._insert_event(
                connection,
                reservation_id=reservation_id,
                event_type=event_type,
                input_tokens=actual_input,
                output_tokens=actual_output,
                estimated_microusd=actual_cost,
                occurred_at_utc=now,
            )
            snapshot = self._snapshot(connection)
            connection.commit()
            return snapshot

    def snapshot(self) -> UsageBudgetSnapshot:
        with self._connect() as connection:
            self._verify_config(connection)
            return self._snapshot(connection)

    def get(self, reservation_id: str) -> DurableBudgetReservationRecord | None:
        _non_empty(reservation_id, "reservation_id")
        with self._connect() as connection:
            self._verify_config(connection)
            row = connection.execute(
                """
                SELECT * FROM cloud_usage_budget_reservations
                WHERE reservation_id = ?
                """,
                (reservation_id,),
            ).fetchone()
        return self._record(row) if row is not None else None

    def list_active(self) -> tuple[DurableBudgetReservationRecord, ...]:
        with self._connect() as connection:
            self._verify_config(connection)
            rows = connection.execute(
                """
                SELECT * FROM cloud_usage_budget_reservations
                WHERE state = 'active'
                ORDER BY created_at_utc, reservation_id
                """
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def events(self, reservation_id: str) -> tuple[DurableBudgetEvent, ...]:
        _non_empty(reservation_id, "reservation_id")
        with self._connect() as connection:
            self._verify_config(connection)
            rows = connection.execute(
                """
                SELECT * FROM cloud_usage_budget_events
                WHERE reservation_id = ?
                ORDER BY event_id
                """,
                (reservation_id,),
            ).fetchall()
        return tuple(
            DurableBudgetEvent(
                event_id=row["event_id"],
                reservation_id=row["reservation_id"],
                event_type=cast(BudgetEventType, row["event_type"]),
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                estimated_microusd=row["estimated_microusd"],
                occurred_at_utc=row["occurred_at_utc"],
            )
            for row in rows
        )

    def _require_active(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
        request_fingerprint: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT * FROM cloud_usage_budget_reservations
            WHERE reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None or row["state"] != "active":
            connection.rollback()
            raise UsageBudgetStateError(
                "reservation is unknown or already terminal"
            )
        if row["request_fingerprint"] != request_fingerprint:
            connection.rollback()
            raise UsageBudgetStateError(
                "request_fingerprint does not match the active reservation"
            )
        if row["budget_config_fingerprint"] != self.config_fingerprint:
            connection.rollback()
            raise UsageBudgetStateError(
                "reservation budget configuration fingerprint does not match"
            )
        return cast(sqlite3.Row, row)

    def _snapshot(self, connection: sqlite3.Connection) -> UsageBudgetSnapshot:
        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN state IN ('settled', 'uncertain')
                    THEN actual_input_tokens ELSE 0 END), 0) AS committed_input,
                COALESCE(SUM(CASE WHEN state IN ('settled', 'uncertain')
                    THEN actual_output_tokens ELSE 0 END), 0) AS committed_output,
                COALESCE(SUM(CASE WHEN state IN ('settled', 'uncertain')
                    THEN actual_estimated_microusd ELSE 0 END), 0) AS committed_cost,
                COALESCE(SUM(CASE WHEN state = 'active'
                    THEN reserved_input_tokens ELSE 0 END), 0) AS reserved_input,
                COALESCE(SUM(CASE WHEN state = 'active'
                    THEN reserved_output_tokens ELSE 0 END), 0) AS reserved_output,
                COALESCE(SUM(CASE WHEN state = 'active'
                    THEN reserved_estimated_microusd ELSE 0 END), 0) AS reserved_cost,
                COALESCE(SUM(CASE WHEN state = 'active' THEN 1 ELSE 0 END), 0)
                    AS active_count,
                COALESCE(SUM(CASE WHEN state = 'uncertain' THEN 1 ELSE 0 END), 0)
                    AS uncertain_count
            FROM cloud_usage_budget_reservations
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("durable budget aggregate query returned no row")
        committed = (
            int(row["committed_input"]),
            int(row["committed_output"]),
            int(row["committed_cost"]),
        )
        reserved = (
            int(row["reserved_input"]),
            int(row["reserved_output"]),
            int(row["reserved_cost"]),
        )
        projected = tuple(
            committed[index] + reserved[index] for index in range(3)
        )
        _sqlite_totals(cast(tuple[int, int, int], projected))
        return UsageBudgetSnapshot(
            pricing_id=self.pricing.pricing_id,
            max_input_tokens=self.limits.max_input_tokens,
            max_output_tokens=self.limits.max_output_tokens,
            max_estimated_microusd=self.limits.max_estimated_microusd,
            committed_input_tokens=committed[0],
            committed_output_tokens=committed[1],
            committed_estimated_microusd=committed[2],
            reserved_input_tokens=reserved[0],
            reserved_output_tokens=reserved[1],
            reserved_estimated_microusd=reserved[2],
            active_reservations=int(row["active_count"]),
            uncertain_settlements=int(row["uncertain_count"]),
            remaining_input_tokens=_remaining(
                self.limits.max_input_tokens, projected[0]
            ),
            remaining_output_tokens=_remaining(
                self.limits.max_output_tokens, projected[1]
            ),
            remaining_estimated_microusd=_remaining(
                self.limits.max_estimated_microusd, projected[2]
            ),
            over_limit=self._exceeds_limits(*projected),
        )

    def _exceeds_limits(self, input_tokens: int, output_tokens: int, cost: int) -> bool:
        return any(
            limit is not None and value > limit
            for value, limit in (
                (input_tokens, self.limits.max_input_tokens),
                (output_tokens, self.limits.max_output_tokens),
                (cost, self.limits.max_estimated_microusd),
            )
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        reservation_id: str,
        event_type: BudgetEventType,
        input_tokens: int,
        output_tokens: int,
        estimated_microusd: int,
        occurred_at_utc: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO cloud_usage_budget_events (
                reservation_id, event_type, input_tokens, output_tokens,
                estimated_microusd, occurred_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                reservation_id,
                event_type,
                input_tokens,
                output_tokens,
                estimated_microusd,
                occurred_at_utc,
            ),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> DurableBudgetReservationRecord:
        return DurableBudgetReservationRecord(
            reservation_id=row["reservation_id"],
            request_fingerprint=row["request_fingerprint"],
            budget_config_fingerprint=row["budget_config_fingerprint"],
            input_tokens=row["reserved_input_tokens"],
            maximum_output_tokens=row["reserved_output_tokens"],
            maximum_estimated_microusd=row["reserved_estimated_microusd"],
            state=cast(ReservationState, row["state"]),
            actual_input_tokens=row["actual_input_tokens"],
            actual_output_tokens=row["actual_output_tokens"],
            actual_estimated_microusd=row["actual_estimated_microusd"],
            created_at_utc=row["created_at_utc"],
            terminal_at_utc=row["terminal_at_utc"],
        )

    def _timestamp(self) -> str:
        return _utc_datetime(self._clock())


def _utc_datetime(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    if value.utcoffset() is None:
        raise ValueError("clock datetime must have a valid UTC offset")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sqlite_non_negative(value: int, name: str) -> None:
    _non_negative_integer(value, name)
    if value > _MAX_SQLITE_INTEGER:
        raise ValueError(f"{name} exceeds SQLite signed-integer range")


def _sqlite_totals(values: tuple[int, int, int]) -> None:
    for name, value in zip(
        ("input token total", "output token total", "estimated cost total"),
        values,
        strict=True,
    ):
        if value > _MAX_SQLITE_INTEGER:
            raise UsageBudgetExceededError(
                f"{name} exceeds SQLite signed-integer range"
            )


def _remaining(limit: int | None, used: int) -> int | None:
    return None if limit is None else max(0, limit - used)
