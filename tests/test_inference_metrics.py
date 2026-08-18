from __future__ import annotations

import pytest

from about_llm.inference import (
    InferenceAttempt,
    InferenceMeasurement,
    RequestOutcome,
    WorkloadSLO,
    classify_http_failure,
    summarize_attempts,
    summarize_measurements,
)

pytestmark = pytest.mark.formula


def test_measurement_defines_ttft_tpot_and_e2e() -> None:
    measurement = InferenceMeasurement(
        prompt_tokens=100,
        output_tokens=5,
        started_at=10.0,
        first_token_at=10.5,
        completed_at=12.5,
    )
    assert measurement.ttft_seconds == pytest.approx(0.5)
    assert measurement.tpot_seconds == pytest.approx(0.5)
    assert measurement.end_to_end_seconds == pytest.approx(2.5)


def test_one_token_output_has_no_tpot() -> None:
    measurement = InferenceMeasurement(1, 1, 0.0, 0.1, 0.1)
    assert measurement.tpot_seconds is None


def test_summary_uses_benchmark_wall_time_for_throughput() -> None:
    measurements = [
        InferenceMeasurement(10, 3, 1.0, 1.2, 1.6),
        InferenceMeasurement(20, 5, 1.1, 1.3, 2.1),
    ]
    summary = summarize_measurements(
        measurements,
        benchmark_started_at=1.0,
        benchmark_completed_at=3.0,
    )
    assert summary.requests_per_second == pytest.approx(1.0)
    assert summary.output_tokens_per_second == pytest.approx(4.0)
    assert summary.prompt_tokens == 30
    assert summary.output_tokens == 8
    assert summary.ttft_p50_seconds == pytest.approx(0.2)


def success_attempt(
    request_id: str,
    *,
    started_at: float,
    first_token_at: float,
    completed_at: float,
    output_tokens: int = 3,
    offered_at: float | None = None,
) -> InferenceAttempt:
    return InferenceAttempt(
        request_id=request_id,
        outcome=RequestOutcome.SUCCESS,
        prompt_tokens=10,
        output_tokens=output_tokens,
        started_at=started_at,
        first_token_at=first_token_at,
        completed_at=completed_at,
        offered_at=offered_at,
    )


def test_workload_summary_keeps_failures_as_first_class_attempts() -> None:
    attempts = [
        success_attempt("ok-1", started_at=0.0, first_token_at=0.2, completed_at=0.6),
        success_attempt("ok-2", started_at=0.1, first_token_at=0.4, completed_at=1.0),
        InferenceAttempt("limited", RequestOutcome.RATE_LIMITED, 0.2, 0.3),
        InferenceAttempt("timeout", RequestOutcome.TIMEOUT, 0.3, 1.5),
    ]
    summary = summarize_attempts(
        attempts, benchmark_started_at=0.0, benchmark_completed_at=2.0
    )

    assert summary.attempted_requests == 4
    assert summary.successful_requests == 2
    assert summary.failed_requests == 2
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.attempted_requests_per_second == pytest.approx(2)
    assert summary.successful_requests_per_second == pytest.approx(1)
    assert summary.successful_output_tokens_per_second == pytest.approx(3)
    assert summary.failure_counts == {"rate_limited": 1, "timeout": 1}
    assert summary.ttft_p50_seconds == pytest.approx(0.25)
    assert summary.offered_timing_attempt_count == 0
    assert summary.client_queue_p95_seconds is None


def test_offered_timing_exposes_client_queue_and_terminal_latency() -> None:
    attempts = [
        success_attempt(
            "queued",
            offered_at=0.0,
            started_at=0.2,
            first_token_at=0.5,
            completed_at=1.0,
        ),
        InferenceAttempt(
            "limited",
            RequestOutcome.RATE_LIMITED,
            started_at=0.3,
            completed_at=0.4,
            offered_at=0.1,
        ),
    ]

    summary = summarize_attempts(
        attempts, benchmark_started_at=0.0, benchmark_completed_at=1.0
    )

    assert summary.offered_timing_attempt_count == 2
    assert summary.client_queue_p50_seconds == pytest.approx(0.2)
    assert summary.client_queue_p95_seconds == pytest.approx(0.2)
    assert summary.successful_offered_ttft_p50_seconds == pytest.approx(0.5)
    assert summary.successful_offered_ttft_p95_seconds == pytest.approx(0.5)
    assert summary.offered_to_terminal_p50_seconds == pytest.approx(0.65)
    assert summary.offered_to_terminal_p95_seconds == pytest.approx(0.965)
    assert summary.e2e_p95_seconds == pytest.approx(0.8)


def test_workload_rejects_partial_offered_timing() -> None:
    with pytest.raises(ValueError, match="every attempt or absent"):
        summarize_attempts(
            [
                InferenceAttempt(
                    "timed", RequestOutcome.TIMEOUT, 0.1, 0.5, offered_at=0.0
                ),
                InferenceAttempt("legacy", RequestOutcome.TIMEOUT, 0.2, 0.6),
            ],
            benchmark_started_at=0.0,
            benchmark_completed_at=1.0,
        )


def test_workload_with_no_success_has_unavailable_latency_not_zero() -> None:
    summary = summarize_attempts(
        [InferenceAttempt("timeout", RequestOutcome.TIMEOUT, 0.0, 1.0)],
        benchmark_started_at=0.0,
        benchmark_completed_at=1.0,
    )

    assert summary.success_rate == 0
    assert summary.successful_output_tokens_per_second == 0
    assert summary.ttft_p95_seconds is None
    assert summary.e2e_p95_seconds is None
    assert summary.tpot_p95_seconds is None


def test_one_token_success_does_not_create_false_zero_tpot() -> None:
    summary = summarize_attempts(
        [
            success_attempt(
                "one-token",
                started_at=0,
                first_token_at=0.2,
                completed_at=0.2,
                output_tokens=1,
            )
        ],
        benchmark_started_at=0,
        benchmark_completed_at=1,
    )

    assert summary.tpot_p50_seconds is None
    assert summary.tpot_p95_seconds is None


def test_slo_reports_reliability_latency_and_unavailable_metric_failures() -> None:
    summary = summarize_attempts(
        [
            success_attempt(
                "slow", started_at=0, first_token_at=0.6, completed_at=1.2, output_tokens=1
            ),
            InferenceAttempt("timeout", RequestOutcome.TIMEOUT, 0.1, 1.5),
        ],
        benchmark_started_at=0,
        benchmark_completed_at=2,
    )
    slo = WorkloadSLO(
        minimum_success_rate=0.9,
        maximum_ttft_p95_seconds=0.5,
        maximum_e2e_p95_seconds=1.0,
        maximum_tpot_p95_seconds=0.1,
    )

    passed, reasons = slo.evaluate(summary)

    assert not passed
    assert len(reasons) == 4
    assert any("success rate" in reason for reason in reasons)
    assert any("TTFT p95" in reason for reason in reasons)
    assert any("E2E p95" in reason for reason in reasons)
    assert any("TPOT p95 is unavailable" in reason for reason in reasons)


def test_slo_passes_when_every_constraint_is_met() -> None:
    summary = summarize_attempts(
        [success_attempt("ok", started_at=0, first_token_at=0.1, completed_at=0.5)],
        benchmark_started_at=0,
        benchmark_completed_at=1,
    )

    assert WorkloadSLO(
        minimum_success_rate=1,
        maximum_ttft_p95_seconds=0.2,
        maximum_e2e_p95_seconds=0.6,
        maximum_tpot_p95_seconds=0.3,
    ).evaluate(summary) == (True, ())


def test_slo_can_gate_client_queue_and_offered_terminal_time() -> None:
    summary = summarize_attempts(
        [InferenceAttempt("slow", RequestOutcome.TIMEOUT, 0.4, 1.0, offered_at=0.0)],
        benchmark_started_at=0,
        benchmark_completed_at=1,
    )

    passed, reasons = WorkloadSLO(
        minimum_success_rate=0,
        maximum_client_queue_p95_seconds=0.3,
        maximum_successful_offered_ttft_p95_seconds=0.1,
        maximum_offered_to_terminal_p95_seconds=0.9,
    ).evaluate(summary)

    assert not passed
    assert len(reasons) == 3
    assert any("client queue p95" in reason for reason in reasons)
    assert any("successful offered TTFT p95 is unavailable" in reason for reason in reasons)
    assert any("offered-to-terminal p95" in reason for reason in reasons)


def test_attempt_from_measurement_round_trips() -> None:
    measurement = InferenceMeasurement(10, 3, 1.0, 1.2, 1.6)
    attempt = InferenceAttempt.from_measurement("request-1", measurement)

    assert attempt.as_measurement() == measurement
    assert attempt.end_to_end_seconds == pytest.approx(0.6)


def test_attempt_validates_offered_time_order() -> None:
    with pytest.raises(ValueError, match="offered_at must not postdate"):
        InferenceAttempt(
            "bad-order", RequestOutcome.TIMEOUT, 0.1, 1.0, offered_at=0.2
        )


@pytest.mark.parametrize(
    "attempt",
    [
        InferenceAttempt("failed", RequestOutcome.TIMEOUT, 0, 1),
        InferenceAttempt("partial", RequestOutcome.SERVER_ERROR, 0, 1, 0.5, 10, 1),
    ],
)
def test_failed_attempt_cannot_be_converted_to_success_measurement(
    attempt: InferenceAttempt,
) -> None:
    with pytest.raises(ValueError, match="successful"):
        attempt.as_measurement()


def test_success_attempt_requires_tokens_and_first_token_timestamp() -> None:
    with pytest.raises(ValueError, match="first_token_at"):
        InferenceAttempt("bad", RequestOutcome.SUCCESS, 0, 1, prompt_tokens=1, output_tokens=1)
    with pytest.raises(ValueError, match="prompt_tokens"):
        InferenceAttempt(
            "bad", RequestOutcome.SUCCESS, 0, 1, first_token_at=0.1, output_tokens=1
        )
    with pytest.raises(ValueError, match="positive output_tokens"):
        InferenceAttempt(
            "bad",
            RequestOutcome.SUCCESS,
            0,
            1,
            first_token_at=0.1,
            prompt_tokens=1,
            output_tokens=0,
        )


def test_workload_rejects_duplicate_ids_and_out_of_window_attempts() -> None:
    item = InferenceAttempt("failed", RequestOutcome.TIMEOUT, 0, 1)
    with pytest.raises(ValueError, match="unique"):
        summarize_attempts([item, item], benchmark_started_at=0, benchmark_completed_at=2)
    with pytest.raises(ValueError, match="benchmark interval"):
        summarize_attempts([item], benchmark_started_at=0.5, benchmark_completed_at=2)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, RequestOutcome.HTTP_ERROR),
        (404, RequestOutcome.HTTP_ERROR),
        (429, RequestOutcome.RATE_LIMITED),
        (500, RequestOutcome.SERVER_ERROR),
        (599, RequestOutcome.SERVER_ERROR),
    ],
)
def test_http_failure_classification(
    status_code: int, expected: RequestOutcome
) -> None:
    assert classify_http_failure(status_code) is expected


@pytest.mark.parametrize("status_code", [200, 399, 600])
def test_http_failure_classification_rejects_non_error_status(status_code: int) -> None:
    with pytest.raises(ValueError, match="HTTP error"):
        classify_http_failure(status_code)


@pytest.mark.parametrize(
    ("minimum_success_rate", "maximum_ttft"),
    [(-0.1, None), (1.1, None), (0.9, 0), (0.9, float("nan"))],
)
def test_slo_rejects_invalid_thresholds(
    minimum_success_rate: float, maximum_ttft: float | None
) -> None:
    with pytest.raises(ValueError):
        WorkloadSLO(
            minimum_success_rate=minimum_success_rate,
            maximum_ttft_p95_seconds=maximum_ttft,
        )
