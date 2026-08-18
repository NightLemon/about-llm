from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from about_llm.inference import (
    BatchingRequest,
    ContinuousBatchingReport,
    simulate_continuous_batching,
)

pytestmark = pytest.mark.formula

ROOT = Path(__file__).resolve().parents[1]


def _fixture_report() -> ContinuousBatchingReport:
    return simulate_continuous_batching(
        [
            BatchingRequest("a", 0, 4, 3),
            BatchingRequest("b", 1, 2, 2),
            BatchingRequest("c", 1, 1, 1),
        ],
        max_batch_tokens=4,
        max_running_sequences=2,
        max_prefill_tokens_per_request=3,
    )


def test_exact_chunked_prefill_decode_and_admission_trace() -> None:
    report = _fixture_report()

    assert [step.used_token_slots for step in report.steps] == [3, 3, 2, 2]
    assert [
        [(item.request_id, item.tokens) for item in step.prefill_slices]
        for step in report.steps
    ] == [[("a", 3)], [("a", 1), ("b", 2)], [], [("c", 1)]]
    assert [step.decoded_request_ids for step in report.steps] == [
        (),
        (),
        ("a", "b"),
        ("a",),
    ]
    assert [step.admitted_request_ids for step in report.steps] == [
        ("a",),
        ("b",),
        (),
        ("c",),
    ]
    assert [step.completed_request_ids for step in report.steps] == [
        (),
        (),
        ("b",),
        ("a", "c"),
    ]


def test_first_token_comes_from_final_prefill_position_and_work_is_p_plus_o_minus_one() -> None:
    report = _fixture_report()
    schedules = {request.request_id: request for request in report.requests}

    assert schedules["a"].output_emitted_at_steps == (2, 3, 4)
    assert schedules["b"].output_emitted_at_steps == (2, 3)
    assert schedules["c"].output_emitted_at_steps == (4,)
    assert report.prompt_tokens == 7
    assert report.output_tokens == 6
    assert report.modeled_forward_tokens == 10
    assert sum(step.used_token_slots for step in report.steps) == 10
    assert report.modeled_forward_tokens == 7 + 6 - 3


def test_queue_ttft_tpot_and_capacity_ledgers_are_distinct() -> None:
    report = _fixture_report()
    schedules = {request.request_id: request for request in report.requests}

    assert schedules["a"].queue_steps == 0
    assert schedules["a"].ttft_steps == 2
    assert schedules["a"].tpot_steps == 1
    assert schedules["b"].queue_steps == 0
    assert schedules["b"].ttft_steps == 1
    assert schedules["c"].queue_steps == 2
    assert schedules["c"].ttft_steps == 3
    assert schedules["c"].tpot_steps is None
    assert report.elapsed_steps == 4
    assert report.active_steps == 4
    assert report.elapsed_token_capacity == 16
    assert report.active_token_capacity == 16
    assert report.elapsed_token_utilization == pytest.approx(10 / 16)


def test_idle_arrival_gap_counts_in_elapsed_but_not_active_capacity() -> None:
    report = simulate_continuous_batching(
        [BatchingRequest("a", 0, 1, 1), BatchingRequest("b", 5, 1, 1)],
        max_batch_tokens=2,
        max_running_sequences=1,
        max_prefill_tokens_per_request=2,
    )

    assert [step.iteration for step in report.steps] == [0, 5]
    assert report.completed_at_step == 6
    assert report.elapsed_steps == 6
    assert report.active_steps == 2
    assert report.elapsed_token_capacity == 12
    assert report.active_token_capacity == 4
    assert report.elapsed_token_utilization == pytest.approx(1 / 6)
    assert report.active_token_utilization == pytest.approx(1 / 2)


def test_same_arrival_is_stable_fcfs_and_sequence_limit_creates_queueing() -> None:
    report = simulate_continuous_batching(
        [BatchingRequest("b", 0, 1, 1), BatchingRequest("a", 0, 1, 1)],
        max_batch_tokens=1,
        max_running_sequences=1,
        max_prefill_tokens_per_request=1,
    )

    assert [step.admitted_request_ids for step in report.steps] == [("b",), ("a",)]
    assert report.requests[0].request_id == "b"
    assert report.requests[1].request_id == "a"
    assert report.requests[1].queue_steps == 1


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: BatchingRequest("", 0, 1, 1), "request_id"),
        (lambda: BatchingRequest("a", -1, 1, 1), "arrival_step"),
        (lambda: BatchingRequest("a", 0, 0, 1), "prompt_tokens"),
        (lambda: BatchingRequest("a", 0, 1, 0), "output_tokens"),
        (
            lambda: simulate_continuous_batching(
                [],
                max_batch_tokens=1,
                max_running_sequences=1,
                max_prefill_tokens_per_request=1,
            ),
            "non-empty",
        ),
        (
            lambda: simulate_continuous_batching(
                [BatchingRequest("a", 0, 1, 1), BatchingRequest("a", 1, 1, 1)],
                max_batch_tokens=1,
                max_running_sequences=1,
                max_prefill_tokens_per_request=1,
            ),
            "unique",
        ),
        (
            lambda: simulate_continuous_batching(
                [BatchingRequest("a", 0, 1, 1)],
                max_batch_tokens=1,
                max_running_sequences=2,
                max_prefill_tokens_per_request=1,
            ),
            "at least max_running_sequences",
        ),
    ],
)
def test_invalid_scheduler_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        operation()


def test_simulation_is_deterministic() -> None:
    assert _fixture_report() == _fixture_report()

