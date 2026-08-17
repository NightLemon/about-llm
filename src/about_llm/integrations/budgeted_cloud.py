"""Cloud HTTP execution with request-bound per-attempt budget reconciliation."""

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
    PostCallBudgetExceededError,
    UsageBudgetExceededError,
    UsageBudgetSnapshot,
)

ReconciliationState = Literal["cancelled", "uncertain"]
AttemptReconciliationState = Literal["cancelled", "uncertain", "settled"]


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

    def snapshot(self) -> UsageBudgetSnapshot: ...


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


@dataclass(frozen=True)
class BudgetedRetryAttempt:
    """One potentially billable HTTP attempt and its terminal reservation."""

    reservation: BudgetReservation
    trace: AttemptTrace | None
    reconciliation_state: AttemptReconciliationState
    budget_snapshot: UsageBudgetSnapshot


@dataclass(frozen=True)
class BudgetedCloudRetryResult:
    response: ChatResponse
    status_code: int
    attempts: tuple[BudgetedRetryAttempt, ...]
    budget_snapshot: UsageBudgetSnapshot


class BudgetedCloudRetryError(RuntimeError):
    """Sanitized logical-call failure with reconciled attempt evidence."""

    def __init__(
        self,
        *,
        reason: str,
        attempts: tuple[BudgetedRetryAttempt, ...],
        budget_snapshot: UsageBudgetSnapshot,
    ) -> None:
        self.reason = reason
        self.attempts = attempts
        self.budget_snapshot = budget_snapshot
        super().__init__(
            "budgeted cloud retry failed: "
            f"{reason}; reconciled_attempts={len(attempts)}"
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
    so this compatibility helper intentionally does not use the executor's
    lifecycle hooks. Use the retry-aware wrapper for per-attempt reservations.
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


async def execute_budgeted_json_request_with_retry(
    *,
    ledger: UsageBudgetBackend,
    logical_call_id: str,
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
) -> BudgetedCloudRetryResult:
    """Retry JSON calls with a distinct durable reservation per HTTP attempt.

    The underlying HTTP executor remains the sole owner of retry decisions,
    ``Retry-After`` sleeps, and the logical-call deadline. This wrapper reserves
    immediately before every send and terminalizes each failed attempt before
    the executor can sleep or start another one. A successful final attempt is
    settled only after strict response parsing exposes complete usage.

    Remote provider execution and the local ledger transition cannot be one
    atomic transaction. Streaming retries and provider-specific error-usage
    parsing are intentionally outside this JSON-only reference contract.
    """

    if not isinstance(policy, RetryPolicy):
        raise TypeError("policy must be RetryPolicy")
    if not isinstance(replay_safe, bool):
        raise ValueError("replay_safe must be a boolean")
    if not isinstance(logical_call_id, str) or not logical_call_id.strip():
        raise ValueError("logical_call_id must be a non-empty string")
    if not callable(parse_response):
        raise TypeError("parse_response must be callable")
    validate_request_target(request, config)

    completed: list[BudgetedRetryAttempt] = []
    active_reservation: BudgetReservation | None = None
    active_attempt_number: int | None = None
    active_trace: AttemptTrace | None = None

    def before_attempt(attempt_number: int) -> None:
        nonlocal active_attempt_number, active_reservation, active_trace
        if active_reservation is not None:
            raise RuntimeError("previous budget reservation is still active")
        reservation = ledger.reserve_request(
            f"{logical_call_id}:attempt:{attempt_number}",
            request=request,
            billing_scope=billing_scope,
            estimated_input_tokens=estimated_input_tokens,
        )
        active_reservation = reservation
        active_attempt_number = attempt_number
        active_trace = None

    def after_attempt(trace: AttemptTrace) -> None:
        nonlocal active_attempt_number, active_reservation, active_trace
        reservation = active_reservation
        if reservation is None or active_attempt_number != trace.attempt:
            raise RuntimeError("HTTP attempt has no matching active reservation")
        active_trace = trace
        if trace.failure_kind is None and trace.status_code is not None:
            return

        if _trace_proves_request_was_never_sent(trace):
            state: AttemptReconciliationState = "cancelled"
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
        completed.append(
            BudgetedRetryAttempt(
                reservation=reservation,
                trace=trace,
                reconciliation_state=state,
                budget_snapshot=snapshot,
            )
        )
        active_reservation = None
        active_attempt_number = None
        active_trace = None

    executor_kwargs: dict[str, Any] = {
        "client": client,
        "request": request,
        "policy": policy,
        "config": config,
        "replay_safe": replay_safe,
        "sleep": sleep,
        "before_attempt": before_attempt,
        "after_attempt": after_attempt,
    }
    if monotonic is not None:
        executor_kwargs["monotonic"] = monotonic
    if wall_clock is not None:
        executor_kwargs["wall_clock"] = wall_clock
    if jitter is not None:
        executor_kwargs["jitter"] = jitter

    try:
        http_result = await execute_json_request(**executor_kwargs)
    except asyncio.CancelledError:
        if active_reservation is not None:
            reservation = active_reservation
            snapshot = ledger.mark_usage_uncertain(
                reservation.reservation_id,
                request_fingerprint=reservation.request_fingerprint,
            )
            completed.append(
                BudgetedRetryAttempt(
                    reservation=reservation,
                    trace=active_trace,
                    reconciliation_state="uncertain",
                    budget_snapshot=snapshot,
                )
            )
        raise
    except UsageBudgetExceededError as error:
        raise BudgetedCloudRetryError(
            reason="budget_reservation_rejected",
            attempts=tuple(completed),
            budget_snapshot=ledger.snapshot(),
        ) from error
    except CloudCallError as error:
        if active_reservation is not None:
            reservation = active_reservation
            snapshot = ledger.mark_usage_uncertain(
                reservation.reservation_id,
                request_fingerprint=reservation.request_fingerprint,
            )
            completed.append(
                BudgetedRetryAttempt(
                    reservation=reservation,
                    trace=active_trace,
                    reconciliation_state="uncertain",
                    budget_snapshot=snapshot,
                )
            )
        raise BudgetedCloudRetryError(
            reason=error.reason,
            attempts=tuple(completed),
            budget_snapshot=ledger.snapshot(),
        ) from error
    except Exception as error:
        if active_reservation is None:
            raise
        reservation = active_reservation
        snapshot = ledger.mark_usage_uncertain(
            reservation.reservation_id,
            request_fingerprint=reservation.request_fingerprint,
        )
        completed.append(
            BudgetedRetryAttempt(
                reservation=reservation,
                trace=active_trace,
                reconciliation_state="uncertain",
                budget_snapshot=snapshot,
            )
        )
        raise BudgetedCloudRetryError(
            reason="local_failure_after_reservation",
            attempts=tuple(completed),
            budget_snapshot=snapshot,
        ) from error

    successful_reservation = active_reservation
    if successful_reservation is None or active_trace is None:
        raise RuntimeError("successful HTTP attempt has no active budget evidence")
    try:
        response = parse_response(http_result.payload)
        if not isinstance(response, ChatResponse):
            raise TypeError("parse_response must return ChatResponse")
        if response.input_tokens is None or response.output_tokens is None:
            raise ValueError("successful response does not contain complete usage")
    except Exception as error:
        snapshot = ledger.mark_usage_uncertain(
            successful_reservation.reservation_id,
            request_fingerprint=successful_reservation.request_fingerprint,
        )
        completed.append(
            BudgetedRetryAttempt(
                reservation=successful_reservation,
                trace=active_trace,
                reconciliation_state="uncertain",
                budget_snapshot=snapshot,
            )
        )
        raise BudgetedCloudRetryError(
            reason="response_usage_unavailable",
            attempts=tuple(completed),
            budget_snapshot=snapshot,
        ) from error

    try:
        snapshot = ledger.settle(
            successful_reservation.reservation_id,
            request_fingerprint=successful_reservation.request_fingerprint,
            actual_input_tokens=response.input_tokens,
            actual_output_tokens=response.output_tokens,
        )
    except PostCallBudgetExceededError as error:
        completed.append(
            BudgetedRetryAttempt(
                reservation=successful_reservation,
                trace=active_trace,
                reconciliation_state="settled",
                budget_snapshot=error.snapshot,
            )
        )
        raise BudgetedCloudRetryError(
            reason="post_call_budget_exceeded",
            attempts=tuple(completed),
            budget_snapshot=error.snapshot,
        ) from error

    completed.append(
        BudgetedRetryAttempt(
            reservation=successful_reservation,
            trace=active_trace,
            reconciliation_state="settled",
            budget_snapshot=snapshot,
        )
    )
    return BudgetedCloudRetryResult(
        response=response,
        status_code=http_result.status_code,
        attempts=tuple(completed),
        budget_snapshot=snapshot,
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
    return all(_trace_proves_request_was_never_sent(attempt) for attempt in error.attempts)


def _trace_proves_request_was_never_sent(trace: AttemptTrace) -> bool:
    return (
        trace.status_code is None
        and trace.failure_kind in {"timeout", "transport"}
        and not trace.outcome_uncertain
    )
