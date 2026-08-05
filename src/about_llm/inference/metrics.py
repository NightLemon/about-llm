"""Definitions and aggregation for streaming LLM latency measurements."""

from __future__ import annotations

from dataclasses import dataclass

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
