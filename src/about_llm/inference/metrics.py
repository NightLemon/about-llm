"""Definitions and aggregation for streaming LLM latency measurements."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class InferenceMeasurement:
    """One completed request measured with a monotonic clock."""

    prompt_tokens: int
    output_tokens: int
    started_at: float
    first_token_at: float
    completed_at: float

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.output_tokens <= 0:
            raise ValueError("prompt_tokens must be non-negative and output_tokens positive")
        if not self.started_at <= self.first_token_at <= self.completed_at:
            raise ValueError("timestamps must satisfy start <= first token <= completion")

    @property
    def ttft_seconds(self) -> float:
        return self.first_token_at - self.started_at

    @property
    def end_to_end_seconds(self) -> float:
        return self.completed_at - self.started_at

    @property
    def tpot_seconds(self) -> float | None:
        """Average interval after the first token.

        N output tokens contain N-1 post-first-token intervals. TPOT is
        undefined for a one-token output rather than incorrectly reported as 0.
        """
        if self.output_tokens == 1:
            return None
        return (self.completed_at - self.first_token_at) / (self.output_tokens - 1)


@dataclass(frozen=True)
class InferenceSummary:
    requests: int
    prompt_tokens: int
    output_tokens: int
    wall_seconds: float
    requests_per_second: float
    output_tokens_per_second: float
    ttft_p50_seconds: float
    ttft_p95_seconds: float
    e2e_p50_seconds: float
    e2e_p95_seconds: float
    tpot_p50_seconds: float | None
    tpot_p95_seconds: float | None


class RequestOutcome(str, Enum):
    """Mutually exclusive terminal outcome for one offered request."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    HTTP_ERROR = "http_error"
    PROTOCOL_ERROR = "protocol_error"
    CLIENT_ERROR = "client_error"
    CANCELLED = "cancelled"


def classify_http_failure(status_code: int) -> RequestOutcome:
    """Map an HTTP error response to a stable workload outcome category."""

    if isinstance(status_code, bool) or not isinstance(status_code, int):
        raise TypeError("status_code must be an integer")
    if not 400 <= status_code <= 599:
        raise ValueError("status_code must be an HTTP error in [400, 599]")
    if status_code == 429:
        return RequestOutcome.RATE_LIMITED
    if status_code >= 500:
        return RequestOutcome.SERVER_ERROR
    return RequestOutcome.HTTP_ERROR


@dataclass(frozen=True)
class InferenceAttempt:
    """One request attempt, including failures and partial streams.

    ``offered_at`` is when the workload generator made the request eligible
    for dispatch. ``started_at`` is when the HTTP attempt actually began.
    Keeping both prevents a client concurrency semaphore from hiding queueing.
    """

    request_id: str
    outcome: RequestOutcome
    started_at: float
    completed_at: float
    first_token_at: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    offered_at: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id or self.request_id.isspace():
            raise ValueError("request_id must not be empty")
        if not all(math.isfinite(value) for value in (self.started_at, self.completed_at)):
            raise ValueError("attempt timestamps must be finite")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not predate started_at")
        if self.offered_at is not None:
            if not math.isfinite(self.offered_at):
                raise ValueError("offered_at must be finite")
            if self.offered_at > self.started_at:
                raise ValueError("offered_at must not postdate started_at")
        if self.first_token_at is not None:
            if not math.isfinite(self.first_token_at):
                raise ValueError("first_token_at must be finite")
            if not self.started_at <= self.first_token_at <= self.completed_at:
                raise ValueError("first_token_at must be within the attempt interval")
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if self.outcome is RequestOutcome.SUCCESS:
            if self.first_token_at is None:
                raise ValueError("successful attempt requires first_token_at")
            if self.prompt_tokens is None:
                raise ValueError("successful attempt requires prompt_tokens")
            if self.output_tokens is None or self.output_tokens <= 0:
                raise ValueError("successful attempt requires positive output_tokens")

    @classmethod
    def from_measurement(
        cls,
        request_id: str,
        value: InferenceMeasurement,
        *,
        offered_at: float | None = None,
    ) -> InferenceAttempt:
        return cls(
            request_id=request_id,
            outcome=RequestOutcome.SUCCESS,
            started_at=value.started_at,
            first_token_at=value.first_token_at,
            completed_at=value.completed_at,
            prompt_tokens=value.prompt_tokens,
            output_tokens=value.output_tokens,
            offered_at=offered_at,
        )

    @property
    def end_to_end_seconds(self) -> float:
        """Dispatch-to-terminal latency."""

        return self.completed_at - self.started_at

    @property
    def client_queue_seconds(self) -> float | None:
        if self.offered_at is None:
            return None
        return self.started_at - self.offered_at

    @property
    def offered_to_terminal_seconds(self) -> float | None:
        if self.offered_at is None:
            return None
        return self.completed_at - self.offered_at

    def as_measurement(self) -> InferenceMeasurement:
        if self.outcome is not RequestOutcome.SUCCESS:
            raise ValueError("only a successful attempt can become an InferenceMeasurement")
        assert self.first_token_at is not None
        assert self.prompt_tokens is not None
        assert self.output_tokens is not None
        return InferenceMeasurement(
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
            started_at=self.started_at,
            first_token_at=self.first_token_at,
            completed_at=self.completed_at,
        )


@dataclass(frozen=True)
class WorkloadSummary:
    """Attempt reliability plus successful-request latency and throughput."""

    attempted_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    wall_seconds: float
    offered_timing_attempt_count: int
    client_queue_p50_seconds: float | None
    client_queue_p95_seconds: float | None
    successful_offered_ttft_p50_seconds: float | None
    successful_offered_ttft_p95_seconds: float | None
    offered_to_terminal_p50_seconds: float | None
    offered_to_terminal_p95_seconds: float | None
    attempted_requests_per_second: float
    successful_requests_per_second: float
    successful_prompt_tokens: int
    successful_output_tokens: int
    successful_output_tokens_per_second: float
    failure_counts: dict[str, int]
    ttft_p50_seconds: float | None
    ttft_p95_seconds: float | None
    e2e_p50_seconds: float | None
    e2e_p95_seconds: float | None
    tpot_p50_seconds: float | None
    tpot_p95_seconds: float | None


@dataclass(frozen=True)
class WorkloadSLO:
    """Transparent offline gate over reliability and successful latency."""

    minimum_success_rate: float
    maximum_ttft_p95_seconds: float | None = None
    maximum_e2e_p95_seconds: float | None = None
    maximum_tpot_p95_seconds: float | None = None
    maximum_client_queue_p95_seconds: float | None = None
    maximum_successful_offered_ttft_p95_seconds: float | None = None
    maximum_offered_to_terminal_p95_seconds: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_success_rate) or not 0 <= self.minimum_success_rate <= 1:
            raise ValueError("minimum_success_rate must be finite and in [0, 1]")
        for name, value in (
            ("maximum_ttft_p95_seconds", self.maximum_ttft_p95_seconds),
            ("maximum_e2e_p95_seconds", self.maximum_e2e_p95_seconds),
            ("maximum_tpot_p95_seconds", self.maximum_tpot_p95_seconds),
            (
                "maximum_client_queue_p95_seconds",
                self.maximum_client_queue_p95_seconds,
            ),
            (
                "maximum_successful_offered_ttft_p95_seconds",
                self.maximum_successful_offered_ttft_p95_seconds,
            ),
            (
                "maximum_offered_to_terminal_p95_seconds",
                self.maximum_offered_to_terminal_p95_seconds,
            ),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when provided")

    def evaluate(self, summary: WorkloadSummary) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if summary.success_rate < self.minimum_success_rate:
            reasons.append(
                f"success rate {summary.success_rate:.4f} < {self.minimum_success_rate:.4f}"
            )
        for label, actual, maximum in (
            ("TTFT p95", summary.ttft_p95_seconds, self.maximum_ttft_p95_seconds),
            ("E2E p95", summary.e2e_p95_seconds, self.maximum_e2e_p95_seconds),
            ("TPOT p95", summary.tpot_p95_seconds, self.maximum_tpot_p95_seconds),
            (
                "client queue p95",
                summary.client_queue_p95_seconds,
                self.maximum_client_queue_p95_seconds,
            ),
            (
                "successful offered TTFT p95",
                summary.successful_offered_ttft_p95_seconds,
                self.maximum_successful_offered_ttft_p95_seconds,
            ),
            (
                "offered-to-terminal p95",
                summary.offered_to_terminal_p95_seconds,
                self.maximum_offered_to_terminal_p95_seconds,
            ),
        ):
            if maximum is None:
                continue
            if actual is None:
                reasons.append(f"{label} is unavailable")
            elif actual > maximum:
                reasons.append(f"{label} {actual:.4f}s > {maximum:.4f}s")
        return not reasons, tuple(reasons)


def summarize_measurements(
    measurements: list[InferenceMeasurement],
    *,
    benchmark_started_at: float,
    benchmark_completed_at: float,
) -> InferenceSummary:
    """Aggregate request latency and system throughput without mixing units."""
    if not measurements:
        raise ValueError("at least one measurement is required")
    wall_seconds = benchmark_completed_at - benchmark_started_at
    if wall_seconds <= 0:
        raise ValueError("benchmark wall time must be positive")

    ttft = np.asarray([item.ttft_seconds for item in measurements])
    e2e = np.asarray([item.end_to_end_seconds for item in measurements])
    tpot = np.asarray([item.tpot_seconds for item in measurements if item.tpot_seconds is not None])
    prompt_tokens = sum(item.prompt_tokens for item in measurements)
    output_tokens = sum(item.output_tokens for item in measurements)

    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q))

    return InferenceSummary(
        requests=len(measurements),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        wall_seconds=wall_seconds,
        requests_per_second=len(measurements) / wall_seconds,
        output_tokens_per_second=output_tokens / wall_seconds,
        ttft_p50_seconds=percentile(ttft, 50),
        ttft_p95_seconds=percentile(ttft, 95),
        e2e_p50_seconds=percentile(e2e, 50),
        e2e_p95_seconds=percentile(e2e, 95),
        tpot_p50_seconds=percentile(tpot, 50) if tpot.size else None,
        tpot_p95_seconds=percentile(tpot, 95) if tpot.size else None,
    )


def summarize_attempts(
    attempts: list[InferenceAttempt],
    *,
    benchmark_started_at: float,
    benchmark_completed_at: float,
) -> WorkloadSummary:
    """Aggregate all attempts without hiding failed requests.

    Latency percentiles and token throughput are computed only from successful
    requests and are named accordingly. Reliability is reported over every
    attempt. Benchmark wall time is the throughput denominator.
    """

    if not attempts:
        raise ValueError("at least one attempt is required")
    if not all(math.isfinite(value) for value in (benchmark_started_at, benchmark_completed_at)):
        raise ValueError("benchmark timestamps must be finite")
    wall_seconds = benchmark_completed_at - benchmark_started_at
    if wall_seconds <= 0:
        raise ValueError("benchmark wall time must be positive")
    request_ids = [attempt.request_id for attempt in attempts]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("attempt request_id values must be unique")
    if any(
        attempt.started_at < benchmark_started_at
        or attempt.completed_at > benchmark_completed_at
        or (
            attempt.offered_at is not None
            and attempt.offered_at < benchmark_started_at
        )
        for attempt in attempts
    ):
        raise ValueError("attempt timestamps must lie within the benchmark interval")
    offered_timing_count = sum(attempt.offered_at is not None for attempt in attempts)
    if offered_timing_count not in (0, len(attempts)):
        raise ValueError(
            "offered_at must be present for every attempt or absent for every attempt"
        )

    client_queue_p50: float | None = None
    client_queue_p95: float | None = None
    offered_terminal_p50: float | None = None
    offered_terminal_p95: float | None = None
    if offered_timing_count:
        queue_values = np.asarray(
            [attempt.client_queue_seconds for attempt in attempts], dtype=np.float64
        )
        terminal_values = np.asarray(
            [attempt.offered_to_terminal_seconds for attempt in attempts],
            dtype=np.float64,
        )
        client_queue_p50 = float(np.percentile(queue_values, 50))
        client_queue_p95 = float(np.percentile(queue_values, 95))
        offered_terminal_p50 = float(np.percentile(terminal_values, 50))
        offered_terminal_p95 = float(np.percentile(terminal_values, 95))

    successful_attempts = [
        attempt for attempt in attempts if attempt.outcome is RequestOutcome.SUCCESS
    ]
    successful = [attempt.as_measurement() for attempt in successful_attempts]
    failures = [attempt for attempt in attempts if attempt.outcome is not RequestOutcome.SUCCESS]
    failure_counts = Counter(attempt.outcome.value for attempt in failures)
    ttft_p50: float | None
    ttft_p95: float | None
    e2e_p50: float | None
    e2e_p95: float | None
    tpot_p50: float | None
    tpot_p95: float | None
    offered_ttft_p50: float | None = None
    offered_ttft_p95: float | None = None
    if successful:
        success_summary = summarize_measurements(
            successful,
            benchmark_started_at=benchmark_started_at,
            benchmark_completed_at=benchmark_completed_at,
        )
        prompt_tokens = success_summary.prompt_tokens
        output_tokens = success_summary.output_tokens
        ttft_p50 = success_summary.ttft_p50_seconds
        ttft_p95 = success_summary.ttft_p95_seconds
        e2e_p50 = success_summary.e2e_p50_seconds
        e2e_p95 = success_summary.e2e_p95_seconds
        tpot_p50 = success_summary.tpot_p50_seconds
        tpot_p95 = success_summary.tpot_p95_seconds
        if offered_timing_count:
            offered_ttft = np.asarray(
                [
                    attempt.first_token_at - attempt.offered_at
                    for attempt in successful_attempts
                    if attempt.first_token_at is not None
                    and attempt.offered_at is not None
                ],
                dtype=np.float64,
            )
            offered_ttft_p50 = float(np.percentile(offered_ttft, 50))
            offered_ttft_p95 = float(np.percentile(offered_ttft, 95))
    else:
        prompt_tokens = 0
        output_tokens = 0
        ttft_p50 = ttft_p95 = e2e_p50 = e2e_p95 = None
        tpot_p50 = tpot_p95 = None

    successful_count = len(successful)
    attempted_count = len(attempts)
    return WorkloadSummary(
        attempted_requests=attempted_count,
        successful_requests=successful_count,
        failed_requests=len(failures),
        success_rate=successful_count / attempted_count,
        wall_seconds=wall_seconds,
        offered_timing_attempt_count=offered_timing_count,
        client_queue_p50_seconds=client_queue_p50,
        client_queue_p95_seconds=client_queue_p95,
        successful_offered_ttft_p50_seconds=offered_ttft_p50,
        successful_offered_ttft_p95_seconds=offered_ttft_p95,
        offered_to_terminal_p50_seconds=offered_terminal_p50,
        offered_to_terminal_p95_seconds=offered_terminal_p95,
        attempted_requests_per_second=attempted_count / wall_seconds,
        successful_requests_per_second=successful_count / wall_seconds,
        successful_prompt_tokens=prompt_tokens,
        successful_output_tokens=output_tokens,
        successful_output_tokens_per_second=output_tokens / wall_seconds,
        failure_counts=dict(sorted(failure_counts.items())),
        ttft_p50_seconds=ttft_p50,
        ttft_p95_seconds=ttft_p95,
        e2e_p50_seconds=e2e_p50,
        e2e_p95_seconds=e2e_p95,
        tpot_p50_seconds=tpot_p50,
        tpot_p95_seconds=tpot_p95,
    )
