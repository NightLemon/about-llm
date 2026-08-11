"""Deterministic KV-aware batching oracle with recompute preemption."""

from __future__ import annotations

from dataclasses import dataclass, field

from about_llm.inference.continuous_batching import BatchingRequest
from about_llm.inference.kv_allocator import KVCapacityError, PagedKVAllocator


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class KVWorkSlice:
    request_id: str
    tokens: int


@dataclass(frozen=True)
class KVPreemptionEvent:
    request_id: str
    triggered_by_request_id: str
    freed_blocks: int
    dropped_cached_positions: int


@dataclass(frozen=True)
class KVPreemptionBatchStep:
    iteration: int
    admitted_request_ids: tuple[str, ...]
    recomputed_slices: tuple[KVWorkSlice, ...]
    prefill_slices: tuple[KVWorkSlice, ...]
    decoded_request_ids: tuple[str, ...]
    preemptions: tuple[KVPreemptionEvent, ...]
    first_token_request_ids: tuple[str, ...]
    completed_request_ids: tuple[str, ...]
    used_token_slots: int
    allocated_blocks_at_end: int
    free_blocks_at_end: int


@dataclass(frozen=True)
class KVPreemptionRequestSchedule:
    request_id: str
    arrival_step: int
    admission_steps: tuple[int, ...]
    first_token_at_step: int
    completed_at_step: int
    prompt_tokens: int
    output_tokens: int
    output_emitted_at_steps: tuple[int, ...]
    preemption_count: int
    recomputed_positions: int
    logical_forward_positions: int
    executed_forward_positions: int

    @property
    def queue_steps(self) -> int:
        return self.admission_steps[0] - self.arrival_step

    @property
    def ttft_steps(self) -> int:
        return self.first_token_at_step - self.arrival_step

    @property
    def end_to_end_steps(self) -> int:
        return self.completed_at_step - self.arrival_step

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "arrival_step": self.arrival_step,
            "admission_steps": self.admission_steps,
            "first_token_at_step": self.first_token_at_step,
            "completed_at_step": self.completed_at_step,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "output_emitted_at_steps": self.output_emitted_at_steps,
            "preemption_count": self.preemption_count,
            "recomputed_positions": self.recomputed_positions,
            "logical_forward_positions": self.logical_forward_positions,
            "executed_forward_positions": self.executed_forward_positions,
            "queue_steps": self.queue_steps,
            "ttft_steps": self.ttft_steps,
            "end_to_end_steps": self.end_to_end_steps,
        }


@dataclass(frozen=True)
class KVPreemptionBatchingReport:
    total_blocks: int
    block_size_tokens: int
    max_batch_tokens: int
    max_running_sequences: int
    max_prefill_tokens_per_request: int
    completed_at_step: int
    logical_forward_positions: int
    recomputed_positions: int
    executed_forward_positions: int
    preemption_count: int
    peak_allocated_blocks: int
    final_free_blocks: int
    requests: tuple[KVPreemptionRequestSchedule, ...]
    steps: tuple[KVPreemptionBatchStep, ...]

    @property
    def recompute_overhead_fraction(self) -> float:
        return self.recomputed_positions / self.logical_forward_positions

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration": {
                "total_blocks": self.total_blocks,
                "block_size_tokens": self.block_size_tokens,
                "max_batch_tokens": self.max_batch_tokens,
                "max_running_sequences": self.max_running_sequences,
                "max_prefill_tokens_per_request": (
                    self.max_prefill_tokens_per_request
                ),
            },
            "summary": {
                "completed_at_step": self.completed_at_step,
                "logical_forward_positions": self.logical_forward_positions,
                "recomputed_positions": self.recomputed_positions,
                "executed_forward_positions": self.executed_forward_positions,
                "recompute_overhead_fraction": self.recompute_overhead_fraction,
                "preemption_count": self.preemption_count,
                "peak_allocated_blocks": self.peak_allocated_blocks,
                "final_free_blocks": self.final_free_blocks,
            },
            "requests": [request.to_dict() for request in self.requests],
            "steps": [
                {
                    "iteration": step.iteration,
                    "admitted_request_ids": step.admitted_request_ids,
                    "recomputed_slices": [
                        {"request_id": item.request_id, "tokens": item.tokens}
                        for item in step.recomputed_slices
                    ],
                    "prefill_slices": [
                        {"request_id": item.request_id, "tokens": item.tokens}
                        for item in step.prefill_slices
                    ],
                    "decoded_request_ids": step.decoded_request_ids,
                    "preemptions": [
                        {
                            "request_id": event.request_id,
                            "triggered_by_request_id": (
                                event.triggered_by_request_id
                            ),
                            "freed_blocks": event.freed_blocks,
                            "dropped_cached_positions": (
                                event.dropped_cached_positions
                            ),
                        }
                        for event in step.preemptions
                    ],
                    "first_token_request_ids": step.first_token_request_ids,
                    "completed_request_ids": step.completed_request_ids,
                    "used_token_slots": step.used_token_slots,
                    "allocated_blocks_at_end": step.allocated_blocks_at_end,
                    "free_blocks_at_end": step.free_blocks_at_end,
                }
                for step in self.steps
            ],
        }


@dataclass
class _State:
    request: BatchingRequest
    ordinal: int
    resident: bool = False
    resident_positions: int = 0
    logical_positions: int = 0
    emitted_tokens: int = 0
    admission_steps: list[int] = field(default_factory=list)
    output_emitted_at_steps: list[int] = field(default_factory=list)
    first_token_at_step: int | None = None
    completed_at_step: int | None = None
    preemption_count: int = 0
    recomputed_positions: int = 0


def simulate_kv_preemption_batching(
    requests: list[BatchingRequest] | tuple[BatchingRequest, ...],
    *,
    total_blocks: int,
    block_size_tokens: int,
    max_batch_tokens: int,
    max_running_sequences: int,
    max_prefill_tokens_per_request: int,
) -> KVPreemptionBatchingReport:
    """Run a decode-first scheduler with metadata-only recompute preemption.

    KV is appended for every modeled causal-LM position. If an append needs a
    new block and capacity is exhausted, the scheduler preempts the youngest
    not-yet-worked resident with lower stable FCFS priority than the requester.
    Its KV metadata is released; after FCFS re-admission, all previously
    evaluated positions are recomputed before new logical progress. Work
    already selected in the same interval is protected from preemption, and
    completed sequences release KV only at the interval boundary. The strict
    priority direction prevents two requests from evicting each other forever.

    This is a deterministic CPU policy contract. It stores no K/V values and
    does not model swap, prefix reuse, CUDA kernels, wall time, or any specific
    vLLM release.
    """

    total_blocks = _positive_integer(total_blocks, "total_blocks")
    block_size_tokens = _positive_integer(block_size_tokens, "block_size_tokens")
    max_batch_tokens = _positive_integer(max_batch_tokens, "max_batch_tokens")
    max_running_sequences = _positive_integer(
        max_running_sequences, "max_running_sequences"
    )
    max_prefill_tokens_per_request = _positive_integer(
        max_prefill_tokens_per_request, "max_prefill_tokens_per_request"
    )
    if not isinstance(requests, (list, tuple)) or not requests:
        raise ValueError("requests must be a non-empty list or tuple")
    if any(not isinstance(request, BatchingRequest) for request in requests):
        raise TypeError("every request must be a BatchingRequest")
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id values must be unique")
    capacity_positions = total_blocks * block_size_tokens
    oversized = [
        request.request_id
        for request in requests
        if request.modeled_forward_tokens > capacity_positions
    ]
    if oversized:
        raise ValueError(
            "each request must fit KV capacity by itself; oversized request(s): "
            f"{oversized}"
        )

    states = [_State(request=request, ordinal=index) for index, request in enumerate(requests)]
    allocator = PagedKVAllocator(
        total_blocks=total_blocks, block_size_tokens=block_size_tokens
    )
    current_step = min(request.arrival_step for request in requests)
    steps: list[KVPreemptionBatchStep] = []
    peak_allocated_blocks = 0
    seen_scheduler_states: set[tuple[tuple[object, ...], ...]] = set()

    while any(state.completed_at_step is None for state in states):
        scheduler_state = tuple(
            (
                state.resident,
                state.resident_positions,
                state.logical_positions,
                state.emitted_tokens,
                state.completed_at_step is not None,
                state.request.arrival_step <= current_step,
            )
            for state in states
        )
        if scheduler_state in seen_scheduler_states:
            raise RuntimeError(
                "KV-aware scheduler entered a repeated state without completion: "
                f"step={current_step}, state={scheduler_state!r}"
            )
        seen_scheduler_states.add(scheduler_state)
        unfinished = [state for state in states if state.completed_at_step is None]
        resident = [state for state in unfinished if state.resident]
        eligible = [
            state
            for state in unfinished
            if not state.resident and state.request.arrival_step <= current_step
        ]
        if not resident and not eligible:
            current_step = min(
                state.request.arrival_step
                for state in unfinished
                if state.request.arrival_step > current_step
            )
            eligible = [
                state
                for state in unfinished
                if not state.resident and state.request.arrival_step <= current_step
            ]

        admitted: list[str] = []
        eligible.sort(key=lambda state: (state.request.arrival_step, state.ordinal))
        resident_count = sum(state.resident for state in unfinished)
        for state in eligible:
            if resident_count >= max_running_sequences:
                break
            allocator.create_sequence(state.request.request_id)
            state.resident = True
            state.resident_positions = 0
            state.admission_steps.append(current_step)
            admitted.append(state.request.request_id)
            resident_count += 1

        boundary = current_step + 1
        recomputed: list[KVWorkSlice] = []
        prefills: list[KVWorkSlice] = []
        decoded: list[str] = []
        preemptions: list[KVPreemptionEvent] = []
        first_tokens: list[str] = []
        completed: list[_State] = []
        protected: set[str] = set()
        remaining_budget = max_batch_tokens

        resident = sorted(
            (state for state in states if state.resident),
            key=lambda state: (state.request.arrival_step, state.ordinal),
        )
        decode_states = [
            state
            for state in resident
            if state.resident_positions == state.logical_positions
            and state.logical_positions >= state.request.prompt_tokens
            and state.emitted_tokens < state.request.output_tokens
        ]
        recompute_states = [
            state
            for state in resident
            if state.resident_positions < state.logical_positions
        ]
        prefill_states = [
            state
            for state in resident
            if state.resident_positions == state.logical_positions
            and state.logical_positions < state.request.prompt_tokens
        ]

        def append_one(
            state: _State,
            protected_request_ids: set[str] = protected,
            step_preemptions: list[KVPreemptionEvent] = preemptions,
        ) -> bool:
            nonlocal peak_allocated_blocks
            while True:
                try:
                    allocator.append(state.request.request_id, 1)
                    state.resident_positions += 1
                    peak_allocated_blocks = max(
                        peak_allocated_blocks, allocator.report().allocated_blocks
                    )
                    return True
                except KVCapacityError:
                    victims = [
                        candidate
                        for candidate in states
                        if candidate.resident
                        and candidate is not state
                        and candidate.resident_positions > 0
                        and (
                            candidate.request.arrival_step,
                            candidate.ordinal,
                        )
                        > (state.request.arrival_step, state.ordinal)
                        and candidate.request.request_id
                        not in protected_request_ids
                    ]
                    if not victims:
                        return False
                    victim = max(
                        victims,
                        key=lambda candidate: (
                            candidate.admission_steps[-1],
                            candidate.ordinal,
                        ),
                    )
                    victim_state = allocator.sequence_state(victim.request.request_id)
                    step_preemptions.append(
                        KVPreemptionEvent(
                            request_id=victim.request.request_id,
                            triggered_by_request_id=state.request.request_id,
                            freed_blocks=len(victim_state.physical_block_ids),
                            dropped_cached_positions=victim.resident_positions,
                        )
                    )
                    allocator.release_sequence(victim.request.request_id)
                    victim.resident = False
                    victim.resident_positions = 0
                    victim.preemption_count += 1

        for state in decode_states:
            if remaining_budget == 0:
                break
            if not state.resident or not append_one(state):
                continue
            protected.add(state.request.request_id)
            state.logical_positions += 1
            state.emitted_tokens += 1
            state.output_emitted_at_steps.append(boundary)
            decoded.append(state.request.request_id)
            remaining_budget -= 1
            if state.emitted_tokens == state.request.output_tokens:
                state.completed_at_step = boundary
                completed.append(state)

        for state in recompute_states:
            if remaining_budget == 0:
                break
            if not state.resident:
                continue
            limit = min(
                state.logical_positions - state.resident_positions,
                max_prefill_tokens_per_request,
                remaining_budget,
            )
            count = 0
            for _ in range(limit):
                if not append_one(state):
                    break
                protected.add(state.request.request_id)
                state.recomputed_positions += 1
                remaining_budget -= 1
                count += 1
            if count:
                recomputed.append(KVWorkSlice(state.request.request_id, count))

        for state in prefill_states:
            if remaining_budget == 0:
                break
            if not state.resident:
                continue
            limit = min(
                state.request.prompt_tokens - state.logical_positions,
                max_prefill_tokens_per_request,
                remaining_budget,
            )
            count = 0
            for _ in range(limit):
                if not append_one(state):
                    break
                protected.add(state.request.request_id)
                state.logical_positions += 1
                remaining_budget -= 1
                count += 1
            if count:
                prefills.append(KVWorkSlice(state.request.request_id, count))
            if (
                state.logical_positions == state.request.prompt_tokens
                and state.emitted_tokens == 0
            ):
                state.emitted_tokens = 1
                state.first_token_at_step = boundary
                state.output_emitted_at_steps.append(boundary)
                first_tokens.append(state.request.request_id)
                if state.request.output_tokens == 1:
                    state.completed_at_step = boundary
                    completed.append(state)

        completed.sort(key=lambda state: state.ordinal)
        for state in completed:
            allocator.release_sequence(state.request.request_id)
            state.resident = False
            state.resident_positions = 0

        used_slots = max_batch_tokens - remaining_budget
        if used_slots == 0 and not preemptions and not completed:
            raise RuntimeError("KV-aware scheduler made no progress")
        allocator_report = allocator.report()
        steps.append(
            KVPreemptionBatchStep(
                iteration=current_step,
                admitted_request_ids=tuple(admitted),
                recomputed_slices=tuple(recomputed),
                prefill_slices=tuple(prefills),
                decoded_request_ids=tuple(decoded),
                preemptions=tuple(preemptions),
                first_token_request_ids=tuple(first_tokens),
                completed_request_ids=tuple(
                    state.request.request_id for state in completed
                ),
                used_token_slots=used_slots,
                allocated_blocks_at_end=allocator_report.allocated_blocks,
                free_blocks_at_end=allocator_report.free_blocks,
            )
        )
        current_step = boundary

    schedules: list[KVPreemptionRequestSchedule] = []
    for state in states:
        if state.first_token_at_step is None or state.completed_at_step is None:
            raise RuntimeError("request timing accounting is incomplete")
        logical = state.request.modeled_forward_tokens
        if state.logical_positions != logical:
            raise RuntimeError("logical forward-position accounting mismatch")
        if len(state.output_emitted_at_steps) != state.request.output_tokens:
            raise RuntimeError("output emission accounting mismatch")
        schedules.append(
            KVPreemptionRequestSchedule(
                request_id=state.request.request_id,
                arrival_step=state.request.arrival_step,
                admission_steps=tuple(state.admission_steps),
                first_token_at_step=state.first_token_at_step,
                completed_at_step=state.completed_at_step,
                prompt_tokens=state.request.prompt_tokens,
                output_tokens=state.request.output_tokens,
                output_emitted_at_steps=tuple(state.output_emitted_at_steps),
                preemption_count=state.preemption_count,
                recomputed_positions=state.recomputed_positions,
                logical_forward_positions=logical,
                executed_forward_positions=logical + state.recomputed_positions,
            )
        )

    logical_total = sum(item.logical_forward_positions for item in schedules)
    recomputed_total = sum(item.recomputed_positions for item in schedules)
    executed_total = sum(step.used_token_slots for step in steps)
    if executed_total != logical_total + recomputed_total:
        raise RuntimeError("executed forward-position accounting mismatch")
    final_report = allocator.report()
    if final_report.allocated_blocks != 0 or final_report.sequence_count != 0:
        raise RuntimeError("completed scheduler retained KV state")
    return KVPreemptionBatchingReport(
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
        max_batch_tokens=max_batch_tokens,
        max_running_sequences=max_running_sequences,
        max_prefill_tokens_per_request=max_prefill_tokens_per_request,
        completed_at_step=max(item.completed_at_step for item in schedules),
        logical_forward_positions=logical_total,
        recomputed_positions=recomputed_total,
        executed_forward_positions=executed_total,
        preemption_count=sum(item.preemption_count for item in schedules),
        peak_allocated_blocks=peak_allocated_blocks,
        final_free_blocks=final_report.free_blocks,
        requests=tuple(schedules),
        steps=tuple(steps),
    )
