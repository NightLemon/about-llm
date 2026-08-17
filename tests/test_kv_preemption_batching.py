from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import pytest

from about_llm.inference import (
    BatchingRequest,
    KVPreemptionBatchingReport,
    simulate_kv_preemption_batching,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixture_report() -> KVPreemptionBatchingReport:
    return simulate_kv_preemption_batching(
        [
            BatchingRequest("a", 0, 4, 3),
            BatchingRequest("b", 1, 2, 2),
        ],
        total_blocks=3,
        block_size_tokens=2,
        max_batch_tokens=4,
        max_running_sequences=2,
        max_prefill_tokens_per_request=3,
    )


def test_exact_kv_pressure_preemption_rebuild_and_release_trace() -> None:
    report = _fixture_report()

    assert [step.used_token_slots for step in report.steps] == [3, 3, 1, 1, 2, 1]
    assert [step.allocated_blocks_at_end for step in report.steps] == [2, 3, 3, 0, 1, 0]
    assert [step.admitted_request_ids for step in report.steps] == [
        ("a",),
        ("b",),
        (),
        ("b",),
        (),
        (),
    ]
    event = report.steps[2].preemptions[0]
    assert event.request_id == "b"
    assert event.triggered_by_request_id == "a"
    assert event.freed_blocks == 1
    assert event.dropped_cached_positions == 2
    assert report.steps[4].recomputed_slices[0].request_id == "b"
    assert report.steps[4].recomputed_slices[0].tokens == 2
    assert report.steps[3].completed_request_ids == ("a",)
    assert report.steps[5].completed_request_ids == ("b",)


def test_logical_work_recompute_overhead_and_output_emissions_are_distinct() -> None:
    report = _fixture_report()
    schedules = {request.request_id: request for request in report.requests}

    assert report.logical_forward_positions == 9
    assert report.recomputed_positions == 2
    assert report.executed_forward_positions == 11
    assert report.recompute_overhead_fraction == pytest.approx(2 / 9)
    assert report.preemption_count == 1
    assert report.peak_allocated_blocks == 3
    assert report.final_free_blocks == 3
    assert schedules["a"].output_emitted_at_steps == (2, 3, 4)
    assert schedules["b"].output_emitted_at_steps == (2, 6)
    assert schedules["b"].admission_steps == (1, 3)
    assert schedules["b"].recomputed_positions == 2
    assert schedules["b"].executed_forward_positions == 5
    assert sum(step.used_token_slots for step in report.steps) == 11


def test_sufficient_kv_capacity_removes_preemption_and_recompute() -> None:
    report = simulate_kv_preemption_batching(
        [
            BatchingRequest("a", 0, 4, 3),
            BatchingRequest("b", 1, 2, 2),
        ],
        total_blocks=6,
        block_size_tokens=2,
        max_batch_tokens=4,
        max_running_sequences=2,
        max_prefill_tokens_per_request=3,
    )

    assert report.preemption_count == 0
    assert report.recomputed_positions == 0
    assert report.executed_forward_positions == report.logical_forward_positions == 9
    assert all(not step.preemptions for step in report.steps)
    assert report.final_free_blocks == 6


def test_stable_fcfs_preemption_direction_prevents_rebuild_ping_pong() -> None:
    requests = [
        BatchingRequest("r0", 4, 3, 2),
        BatchingRequest("r1", 0, 6, 1),
        BatchingRequest("r2", 2, 5, 2),
        BatchingRequest("r3", 1, 1, 3),
        BatchingRequest("r4", 4, 1, 1),
    ]
    report = simulate_kv_preemption_batching(
        requests,
        total_blocks=6,
        block_size_tokens=1,
        max_batch_tokens=8,
        max_running_sequences=3,
        max_prefill_tokens_per_request=2,
    )
    priorities = {
        request.request_id: (request.arrival_step, ordinal)
        for ordinal, request in enumerate(requests)
    }

    assert len(report.steps) < 100
    assert report.final_free_blocks == 6
    for step in report.steps:
        for event in step.preemptions:
            assert (
                priorities[event.triggered_by_request_id]
                < priorities[event.request_id]
            )


def test_seeded_small_state_space_preserves_accounting_and_terminates() -> None:
    rng = random.Random(20260807)
    for case in range(100):
        total_blocks = rng.randint(1, 6)
        block_size = rng.randint(1, 4)
        capacity = total_blocks * block_size
        request_count = rng.randint(1, 5)
        requests = []
        for index in range(request_count):
            prompt = rng.randint(1, capacity)
            output = rng.randint(1, capacity - prompt + 1)
            requests.append(
                BatchingRequest(
                    f"case-{case}-request-{index}",
                    rng.randint(0, 4),
                    prompt,
                    output,
                )
            )
        report = simulate_kv_preemption_batching(
            requests,
            total_blocks=total_blocks,
            block_size_tokens=block_size,
            max_batch_tokens=rng.randint(1, 8),
            max_running_sequences=rng.randint(1, request_count),
            max_prefill_tokens_per_request=rng.randint(1, 6),
        )

        assert report.executed_forward_positions == (
            report.logical_forward_positions + report.recomputed_positions
        )
        assert report.final_free_blocks == total_blocks
        for request, schedule in zip(requests, report.requests, strict=True):
            assert len(schedule.output_emitted_at_steps) == request.output_tokens
            assert schedule.output_emitted_at_steps == tuple(
                sorted(schedule.output_emitted_at_steps)
            )


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: simulate_kv_preemption_batching(
                [],
                total_blocks=1,
                block_size_tokens=1,
                max_batch_tokens=1,
                max_running_sequences=1,
                max_prefill_tokens_per_request=1,
            ),
            "non-empty",
        ),
        (
            lambda: simulate_kv_preemption_batching(
                [BatchingRequest("a", 0, 4, 4)],
                total_blocks=3,
                block_size_tokens=2,
                max_batch_tokens=1,
                max_running_sequences=1,
                max_prefill_tokens_per_request=1,
            ),
            "fit KV capacity",
        ),
        (
            lambda: simulate_kv_preemption_batching(
                [BatchingRequest("a", 0, 1, 1), BatchingRequest("a", 1, 1, 1)],
                total_blocks=1,
                block_size_tokens=1,
                max_batch_tokens=1,
                max_running_sequences=1,
                max_prefill_tokens_per_request=1,
            ),
            "unique",
        ),
        (
            lambda: simulate_kv_preemption_batching(
                [BatchingRequest("a", 0, 1, 1)],
                total_blocks=True,
                block_size_tokens=1,
                max_batch_tokens=1,
                max_running_sequences=1,
                max_prefill_tokens_per_request=1,
            ),
            "positive integer",
        ),
    ],
)
def test_invalid_kv_scheduler_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        operation()


def test_kv_preemption_simulation_is_deterministic() -> None:
    assert _fixture_report() == _fixture_report()


