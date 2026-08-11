"""Single-attempt cloud HTTP execution with request-bound budget reconciliation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

import httpx

from about_llm.integrations.cloud_api import ChatResponse, RequestSpec
from about_llm.integrations.cloud_http import (
    AttemptTrace,
    CloudCallError,
    CloudHttpResult,
    HttpExecutorConfig,
    execute_json_request,
    validate_request_target,
)
from about_llm.integrations.retry import RetryPolicy
from about_llm.integrations.usage_budget import (
    BudgetReservation,
    UsageBudgetSnapshot,
)

ReconciliationState = Literal["cancelled", "uncertain"]


class UsageBudgetBackend(Protocol):
    """Structural contract shared by the in-memory and SQLite ledgers."""

    def reserve_request(
        self,
        reservation_id: str,
        *,
        request: RequestSpec,
        billing_scope: str,
        estimated_input_tokens: int,
    ) -> BudgetReservation: ...

    def settle(
        self,
        reservation_id: str,
        *,
        request_fingerprint: str,
        actual_input_tokens: int,
        actual_output_tokens: int,
    ) -> UsageBudgetSnapshot: ...

    def mark_usage_uncertain(
        self, reservation_id: str, *, request_fingerprint: str
    ) -> UsageBudgetSnapshot: ...

    def cancel_before_send(
        self, reservation_id: str, *, request_fingerprint: str
    ) -> UsageBudgetSnapshot: ...


@dataclass(frozen=True)
class BudgetedCloudResult:
    reservation: BudgetReservation
    response: ChatResponse
    status_code: int
    attempts: tuple[AttemptTrace, ...]
    budget_snapshot: UsageBudgetSnapshot


class BudgetedCloudCallError(RuntimeError):
    """Sanitized call failure after its durable budget transition completed."""

    def __init__(
        self,
        *,
        reason: str,
        reconciliation_state: ReconciliationState,
        reservation: BudgetReservation,
        budget_snapshot: UsageBudgetSnapshot,
        attempts: tuple[AttemptTrace, ...],
    ) -> None:
        self.reason = reason
        self.reconciliation_state = reconciliation_state
        self.reservation = reservation
        self.budget_snapshot = budget_snapshot
        self.attempts = attempts
        super().__init__(
            "budgeted cloud call failed: "
            f"{reason}; reconciliation={reconciliation_state}; "
            f"attempts={len(attempts)}"
        )


async def execute_budgeted_json_request(
    *,
    ledger: UsageBudgetBackend,
    reservation_id: str,
    billing_scope: str,
    estimated_input_tokens: int,
    client: httpx.AsyncClient,
    request: RequestSpec,
    parse_response: Callable[[Mapping[str, Any]], ChatResponse],
    policy: RetryPolicy,
    config: HttpExecutorConfig,
    replay_safe: bool,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] | None = None,
    wall_clock: Callable[[], datetime] | None = None,
    jitter: Callable[[], float] | None = None,
) -> BudgetedCloudResult:
    """Reserve one attempt, execute it, and commit a terminal budget state.

    ``max_attempts`` must be one. A replay can be another billable model call,
    and the underlying executor does not expose an attempt-level reservation
    hook. Callers that retry must create a new reservation for every attempt.
    """

    if not isinstance(policy, RetryPolicy):
        raise TypeError("policy must be RetryPolicy")
    if policy.max_attempts != 1:
        raise ValueError(
            "budgeted execution requires max_attempts=1; reserve every replay "
            "as a separate billable attempt"
        )
    if not isinstance(replay_safe, bool):
        raise ValueError("replay_safe must be a boolean")
    if not callable(parse_response):
        raise TypeError("parse_response must be callable")
    validate_request_target(request, config)

    reservation = ledger.reserve_request(
        reservation_id,
        request=request,
        billing_scope=billing_scope,
        estimated_input_tokens=estimated_input_tokens,
    )
    executor_kwargs: dict[str, Any] = {
        "client": client,
        "request": request,
        "policy": policy,
        "config": config,
        "replay_safe": replay_safe,
        "sleep": sleep,
    }
    if monotonic is not None:
        executor_kwargs["monotonic"] = monotonic
    if wall_clock is not None:
        executor_kwargs["wall_clock"] = wall_clock
    if jitter is not None:
        executor_kwargs["jitter"] = jitter

    try:
        http_result = await execute_json_request(**executor_kwargs)
    except CloudCallError as error:
        state: ReconciliationState
        if _proves_request_was_never_sent(error):
            state = "cancelled"
            snapshot = ledger.cancel_before_send(
                reservation.reservation_id,
                request_fingerprint=reservation.request_fingerprint,
            )
        else:
            state = "uncertain"
            snapshot = ledger.mark_usage_uncertain(
                reservation.reservation_id,
                request_fingerprint=reservation.request_fingerprint,
            )
        raise BudgetedCloudCallError(
            reason=error.reason,
            reconciliation_state=state,
            reservation=reservation,
            budget_snapshot=snapshot,
            attempts=error.attempts,
        ) from error
    except asyncio.CancelledError:
        # Cancellation can arrive while bytes are being written or the server
        # is computing. Without an attempt receipt, zero usage is not proven.
        ledger.mark_usage_uncertain(
            reservation.reservation_id,
            request_fingerprint=reservation.request_fingerprint,
        )
        raise
    except Exception as error:
        # Preflight ran before reservation. Any later unexpected failure is
        # conservatively reconciled because the transport boundary may have
        # been crossed. Do not include the raw exception in the public message.
        snapshot = ledger.mark_usage_uncertain(
            reservation.reservation_id,
            request_fingerprint=reservation.request_fingerprint,
        )
        raise BudgetedCloudCallError(
            reason="local_failure_after_reservation",
            reconciliation_state="uncertain",
            reservation=reservation,
            budget_snapshot=snapshot,
            attempts=(),
        ) from error

    return _settle_http_result(
        ledger=ledger,
        reservation=reservation,
        http_result=http_result,
        parse_response=parse_response,
    )


def _settle_http_result(
    *,
    ledger: UsageBudgetBackend,
    reservation: BudgetReservation,
    http_result: CloudHttpResult,
    parse_response: Callable[[Mapping[str, Any]], ChatResponse],
) -> BudgetedCloudResult:
    try:
        response = parse_response(http_result.payload)
        if not isinstance(response, ChatResponse):
            raise TypeError("parse_response must return ChatResponse")
        if response.input_tokens is None or response.output_tokens is None:
            raise ValueError("successful response does not contain complete usage")
    except Exception as error:
        snapshot = ledger.mark_usage_uncertain(
            reservation.reservation_id,
            request_fingerprint=reservation.request_fingerprint,
        )
        raise BudgetedCloudCallError(
            reason="response_usage_unavailable",
            reconciliation_state="uncertain",
            reservation=reservation,
            budget_snapshot=snapshot,
            attempts=http_result.attempts,
        ) from error

    snapshot = ledger.settle(
        reservation.reservation_id,
        request_fingerprint=reservation.request_fingerprint,
        actual_input_tokens=response.input_tokens,
        actual_output_tokens=response.output_tokens,
    )
    return BudgetedCloudResult(
        reservation=reservation,
        response=response,
        status_code=http_result.status_code,
        attempts=http_result.attempts,
        budget_snapshot=snapshot,
    )


def _proves_request_was_never_sent(error: CloudCallError) -> bool:
    if not error.attempts:
        return error.reason == "deadline_exhausted"
    return all(
        attempt.status_code is None
        and attempt.failure_kind in {"timeout", "transport"}
        and not attempt.outcome_uncertain
        for attempt in error.attempts
    )
