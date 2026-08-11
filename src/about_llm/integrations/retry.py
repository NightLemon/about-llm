"""Provider-neutral, fail-closed retry decisions for cloud model calls."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import MappingProxyType
from typing import Literal

ErrorCategory = Literal["timeout", "transport", "other"]
RetryReason = Literal[
    "retryable_status",
    "retryable_error",
    "not_retryable",
    "replay_unsafe",
    "outcome_uncertain",
    "attempts_exhausted",
    "retry_after_too_long",
    "deadline_exhausted",
]

DEFAULT_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_RETRYABLE_ERRORS: frozenset[ErrorCategory] = frozenset(
    {"timeout", "transport"}
)
_DELTA_SECONDS = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded local policy; ``max_attempts`` includes the initial call."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    multiplier: float = 2.0
    max_backoff_seconds: float = 8.0
    max_retry_after_seconds: float = 30.0
    retryable_statuses: frozenset[int] = field(
        default_factory=lambda: DEFAULT_RETRYABLE_STATUSES
    )
    retryable_errors: frozenset[ErrorCategory] = field(
        default_factory=lambda: DEFAULT_RETRYABLE_ERRORS
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name, value in (
            ("base_delay_seconds", self.base_delay_seconds),
            ("max_backoff_seconds", self.max_backoff_seconds),
            ("max_retry_after_seconds", self.max_retry_after_seconds),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if (
            not isinstance(self.multiplier, (int, float))
            or isinstance(self.multiplier, bool)
            or not math.isfinite(self.multiplier)
            or self.multiplier < 1
        ):
            raise ValueError("multiplier must be a finite number greater than or equal to 1")
        if not all(
            isinstance(status, int)
            and not isinstance(status, bool)
            and 100 <= status <= 599
            for status in self.retryable_statuses
        ):
            raise ValueError("retryable_statuses must contain valid integer HTTP statuses")
        if not self.retryable_errors <= {"timeout", "transport", "other"}:
            raise ValueError("retryable_errors contains an unsupported error category")
        object.__setattr__(self, "retryable_statuses", frozenset(self.retryable_statuses))
        object.__setattr__(self, "retryable_errors", frozenset(self.retryable_errors))


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: float | None
    reason: RetryReason
    attempt: int
    status_code: int | None
    error_category: ErrorCategory | None
    retry_after_state: Literal["absent", "valid", "malformed"] = "absent"
    retry_after_source: Literal["delta-seconds", "http-date"] | None = None


@dataclass(frozen=True)
class ParsedRetryAfter:
    delay_seconds: float
    source: Literal["delta-seconds", "http-date"]


def parse_retry_after(value: str, *, now: datetime) -> ParsedRetryAfter | None:
    """Parse RFC-style delta-seconds or HTTP-date; malformed values return ``None``."""
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    if _DELTA_SECONDS.fullmatch(value):
        try:
            delay = float(int(value))
        except OverflowError:
            delay = math.inf
        return ParsedRetryAfter(delay, "delta-seconds")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delay = max(0.0, (parsed - now).total_seconds())
    return ParsedRetryAfter(delay, "http-date")


def decide_retry(
    *,
    policy: RetryPolicy,
    attempt: int,
    replay_safe: bool,
    outcome_uncertain: bool,
    status_code: int | None = None,
    error_category: ErrorCategory | None = None,
    response_headers: Mapping[str, str] | None = None,
    now: datetime | None = None,
    remaining_seconds: float | None = None,
    jitter_fraction: float = 1.0,
) -> RetryDecision:
    """Return a bounded retry decision without sleeping or sending a request.

    ``attempt`` is the just-completed 1-based attempt. ``jitter_fraction`` is an
    injected value in [0, 1], so callers can use a seeded RNG while tests remain
    deterministic. A valid Retry-After overrides local backoff and is not jittered.
    """
    _validate_decision_inputs(
        attempt=attempt,
        replay_safe=replay_safe,
        outcome_uncertain=outcome_uncertain,
        status_code=status_code,
        error_category=error_category,
        remaining_seconds=remaining_seconds,
        jitter_fraction=jitter_fraction,
    )
    if response_headers and status_code is None:
        raise ValueError("response_headers require an HTTP status_code")
    retry_after, retry_after_state = _retry_after_from_headers(
        response_headers, now=now
    )
    retry_after_source = retry_after.source if retry_after is not None else None

    def result(
        retry: bool,
        delay_seconds: float | None,
        reason: RetryReason,
    ) -> RetryDecision:
        return RetryDecision(
            retry=retry,
            delay_seconds=delay_seconds,
            reason=reason,
            attempt=attempt,
            status_code=status_code,
            error_category=error_category,
            retry_after_state=retry_after_state,
            retry_after_source=retry_after_source,
        )

    if not replay_safe:
        return result(False, None, "replay_unsafe")
    if outcome_uncertain:
        return result(False, None, "outcome_uncertain")
    retryable = status_code in policy.retryable_statuses or (
        error_category in policy.retryable_errors if error_category is not None else False
    )
    if not retryable:
        return result(False, None, "not_retryable")
    if attempt >= policy.max_attempts:
        return result(False, None, "attempts_exhausted")
    if remaining_seconds is not None and remaining_seconds <= 0:
        return result(False, None, "deadline_exhausted")

    if retry_after is not None:
        if retry_after.delay_seconds > policy.max_retry_after_seconds:
            return result(False, None, "retry_after_too_long")
        delay = retry_after.delay_seconds
    else:
        try:
            uncapped = policy.base_delay_seconds * policy.multiplier ** (attempt - 1)
        except OverflowError:
            uncapped = math.inf
        cap = min(policy.max_backoff_seconds, uncapped)
        delay = cap * jitter_fraction
    if remaining_seconds is not None and delay >= remaining_seconds:
        return result(False, None, "deadline_exhausted")
    reason: RetryReason = (
        "retryable_status" if status_code is not None else "retryable_error"
    )
    return result(True, delay, reason)


def _retry_after_from_headers(
    headers: Mapping[str, str] | None, *, now: datetime | None
) -> tuple[ParsedRetryAfter | None, Literal["absent", "valid", "malformed"]]:
    if headers is None:
        return None, "absent"
    normalized_values: dict[str, str] = {}
    for name, header_value in headers.items():
        if not isinstance(name, str) or not isinstance(header_value, str):
            raise ValueError("response headers must have string names and values")
        normalized_name = name.lower()
        if normalized_name in normalized_values:
            raise ValueError("response headers contain duplicate case-insensitive names")
        normalized_values[normalized_name] = header_value
    normalized = MappingProxyType(normalized_values)
    retry_after_value = normalized.get("retry-after")
    if retry_after_value is None:
        return None, "absent"
    if now is None:
        raise ValueError("now is required when Retry-After is present")
    parsed = parse_retry_after(retry_after_value, now=now)
    return (parsed, "valid") if parsed is not None else (None, "malformed")


def _validate_decision_inputs(
    *,
    attempt: int,
    replay_safe: bool,
    outcome_uncertain: bool,
    status_code: int | None,
    error_category: ErrorCategory | None,
    remaining_seconds: float | None,
    jitter_fraction: float,
) -> None:
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    if not isinstance(replay_safe, bool) or not isinstance(outcome_uncertain, bool):
        raise ValueError("replay_safe and outcome_uncertain must be booleans")
    if status_code is not None and (
        not isinstance(status_code, int)
        or isinstance(status_code, bool)
        or not 100 <= status_code <= 599
    ):
        raise ValueError("status_code must be a valid integer HTTP status")
    if error_category is not None and error_category not in {"timeout", "transport", "other"}:
        raise ValueError("unsupported error_category")
    if status_code is not None and error_category is not None:
        raise ValueError("provide status_code or error_category, not both")
    for name, value in (
        ("remaining_seconds", remaining_seconds),
        ("jitter_fraction", jitter_fraction),
    ):
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite")
    if remaining_seconds is not None and remaining_seconds < 0:
        raise ValueError("remaining_seconds must be non-negative")
    if not 0 <= jitter_fraction <= 1:
        raise ValueError("jitter_fraction must be between 0 and 1")
