from __future__ import annotations

import pytest

from about_llm.inference import InferenceMeasurement, summarize_measurements


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
