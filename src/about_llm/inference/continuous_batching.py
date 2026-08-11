"""Deterministic discrete-step oracle for one continuous-batching policy."""

from __future__ import annotations

from dataclasses import dataclass, field


def _non_negative_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class BatchingRequest:
    """One finite request offered at an integer scheduler boundary."""

    request_id: str
    arrival_step: int
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id cannot be empty")
        _non_negative_integer(self.arrival_step, "arrival_step")
        _positive_integer(self.prompt_tokens, "prompt_tokens")
        _positive_integer(self.output_tokens, "output_tokens")

    @property
    def modeled_forward_tokens(self) -> int:
        """Causal-LM positions evaluated under this oracle's convention.

        The final prompt position produces the distribution for the first output
        token. Each later output token needs one decode position, hence P + O - 1.
        """

        return self.prompt_tokens + self.output_tokens - 1


@dataclass(frozen=True)
class PrefillSlice:
    request_id: str
    tokens: int


@dataclass(frozen=True)
class ContinuousBatchStep:
    """Work selected for one half-open scheduler interval ``[i, i + 1)``."""

    iteration: int
    admitted_request_ids: tuple[str, ...]
    prefill_slices: tuple[PrefillSlice, ...]
    decoded_request_ids: tuple[str, ...]
    first_token_request_ids: tuple[str, ...]
    completed_request_ids: tuple[str, ...]
    used_token_slots: int


@dataclass(frozen=True)
class RequestSchedule:
    """Boundary timestamps and emitted-token trace for one request."""

    request_id: str
    arrival_step: int
    admitted_at_step: int
    prefill_completed_at_step: int
    first_token_at_step: int
    completed_at_step: int
    prompt_tokens: int
    output_tokens: int
    output_emitted_at_steps: tuple[int, ...]

    @property
    def queue_steps(self) -> int:
        return self.admitted_at_step - self.arrival_step

    @property
    def ttft_steps(self) -> int:
        return self.first_token_at_step - self.arrival_step

    @property
    def service_steps(self) -> int:
        return self.completed_at_step - self.admitted_at_step

    @property
    def end_to_end_steps(self) -> int:
        return self.completed_at_step - self.arrival_step

    @property
    def tpot_steps(self) -> float | None:
        if self.output_tokens == 1:
            return None
        return (self.completed_at_step - self.first_token_at_step) / (
            self.output_tokens - 1
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "arrival_step": self.arrival_step,
            "admitted_at_step": self.admitted_at_step,
            "prefill_completed_at_step": self.prefill_completed_at_step,
            "first_token_at_step": self.first_token_at_step,
            "completed_at_step": self.completed_at_step,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "output_emitted_at_steps": self.output_emitted_at_steps,
            "queue_steps": self.queue_steps,
            "ttft_steps": self.ttft_steps,
            "service_steps": self.service_steps,
            "end_to_end_steps": self.end_to_end_steps,
            "tpot_steps": self.tpot_steps,
        }


@dataclass(frozen=True)
class ContinuousBatchingReport:
    """Exact result of the documented CPU scheduling policy."""

    max_batch_tokens: int
    max_running_sequences: int
    max_prefill_tokens_per_request: int
    first_arrival_step: int
    completed_at_step: int
    elapsed_steps: int
    active_steps: int
    prompt_tokens: int
    output_tokens: int
    modeled_forward_tokens: int
    elapsed_token_capacity: int
    active_token_capacity: int
    elapsed_token_utilization: float
    active_token_utilization: float
    requests: tuple[RequestSchedule, ...]
    steps: tuple[ContinuousBatchStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration": {
                "max_batch_tokens": self.max_batch_tokens,
                "max_running_sequences": self.max_running_sequences,
                "max_prefill_tokens_per_request": (
                    self.max_prefill_tokens_per_request
                ),
            },
            "summary": {
                "first_arrival_step": self.first_arrival_step,
                "completed_at_step": self.completed_at_step,
                "elapsed_steps": self.elapsed_steps,
                "active_steps": self.active_steps,
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
                "modeled_forward_tokens": self.modeled_forward_tokens,
                "elapsed_token_capacity": self.elapsed_token_capacity,
                "active_token_capacity": self.active_token_capacity,
                "elapsed_token_utilization": self.elapsed_token_utilization,
                "active_token_utilization": self.active_token_utilization,
            },
            "requests": [request.to_dict() for request in self.requests],
            "steps": [
                {
                    "iteration": step.iteration,
                    "admitted_request_ids": step.admitted_request_ids,
                    "prefill_slices": [
                        {"request_id": item.request_id, "tokens": item.tokens}
                        for item in step.prefill_slices
                    ],
                    "decoded_request_ids": step.decoded_request_ids,
                    "first_token_request_ids": step.first_token_request_ids,
                    "completed_request_ids": step.completed_request_ids,
                    "used_token_slots": step.used_token_slots,
                }
                for step in self.steps
            ],
        }


@dataclass
class _RequestState:
    request: BatchingRequest
    ordinal: int
    admitted_at_step: int
    remaining_prefill_tokens: int
    emitted_tokens: int = 0
    prefill_completed_at_step: int | None = None
    first_token_at_step: int | None = None
    completed_at_step: int | None = None
    output_emitted_at_steps: list[int] = field(default_factory=list)


def simulate_continuous_batching(
    requests: list[BatchingRequest] | tuple[BatchingRequest, ...],
    *,
    max_batch_tokens: int,
    max_running_sequences: int,
    max_prefill_tokens_per_request: int,
) -> ContinuousBatchingReport:
    """Simulate a documented decode-first, chunked-prefill policy.

    Arrivals, scheduling, and outputs occur at integer boundaries. At each
    iteration the oracle admits FCFS requests into free sequence slots, gives
    every decode-ready request one token position, then gives every active
    prefill at least one position before distributing remaining prefill budget
    in FCFS order. ``max_batch_tokens >= max_running_sequences`` guarantees
    progress for every resident sequence. A prefill that finishes in interval
    ``[i, i + 1)`` emits its first token at boundary ``i + 1``; later output
    tokens consume decode positions.

    This is a CPU state-machine reference, not a model of a particular vLLM
    version, wall-clock duration, KV capacity, preemption, or GPU execution.
    """

    max_batch_tokens = _positive_integer(max_batch_tokens, "max_batch_tokens")
    max_running_sequences = _positive_integer(
        max_running_sequences, "max_running_sequences"
    )
    max_prefill_tokens_per_request = _positive_integer(
        max_prefill_tokens_per_request, "max_prefill_tokens_per_request"
    )
    if max_batch_tokens < max_running_sequences:
        raise ValueError(
            "max_batch_tokens must be at least max_running_sequences so every "
            "resident request can make progress"
        )
    if not isinstance(requests, (list, tuple)) or not requests:
        raise ValueError("requests must be a non-empty list or tuple")
    if any(not isinstance(request, BatchingRequest) for request in requests):
        raise TypeError("every request must be a BatchingRequest")
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_id values must be unique")

    ordered = sorted(
        enumerate(requests), key=lambda item: (item[1].arrival_step, item[0])
    )
    waiting_index = 0
    running: list[_RequestState] = []
    finished: dict[str, _RequestState] = {}
    steps: list[ContinuousBatchStep] = []
    current_step = ordered[0][1].arrival_step

    while waiting_index < len(ordered) or running:
        if not running and ordered[waiting_index][1].arrival_step > current_step:
            current_step = ordered[waiting_index][1].arrival_step

        admitted: list[str] = []
        while (
            waiting_index < len(ordered)
            and len(running) < max_running_sequences
            and ordered[waiting_index][1].arrival_step <= current_step
        ):
            ordinal, request = ordered[waiting_index]
            waiting_index += 1
            running.append(
                _RequestState(
                    request=request,
                    ordinal=ordinal,
                    admitted_at_step=current_step,
                    remaining_prefill_tokens=request.prompt_tokens,
                )
            )
            admitted.append(request.request_id)

        decode_states = [
            state
            for state in running
            if state.remaining_prefill_tokens == 0
            and state.emitted_tokens < state.request.output_tokens
        ]
        prefill_states = [
            state for state in running if state.remaining_prefill_tokens > 0
        ]
        remaining_budget = max_batch_tokens - len(decode_states)
        allocations = {state.request.request_id: 0 for state in prefill_states}

        # The configuration constraint makes this first progress pass possible.
        for state in prefill_states:
            allocations[state.request.request_id] = 1
            remaining_budget -= 1

        for state in prefill_states:
            already_allocated = allocations[state.request.request_id]
            slice_limit = min(
                max_prefill_tokens_per_request,
                state.remaining_prefill_tokens,
            )
            extra = min(slice_limit - already_allocated, remaining_budget)
            allocations[state.request.request_id] += extra
            remaining_budget -= extra
            if remaining_budget == 0:
                break

        boundary = current_step + 1
        first_tokens: list[str] = []
        completed_states: list[_RequestState] = []
        prefill_slices: list[PrefillSlice] = []
        for state in prefill_states:
            allocated = allocations[state.request.request_id]
            state.remaining_prefill_tokens -= allocated
            prefill_slices.append(PrefillSlice(state.request.request_id, allocated))
            if state.remaining_prefill_tokens == 0:
                state.prefill_completed_at_step = boundary
                state.first_token_at_step = boundary
                state.emitted_tokens = 1
                state.output_emitted_at_steps.append(boundary)
                first_tokens.append(state.request.request_id)
                if state.request.output_tokens == 1:
                    state.completed_at_step = boundary
                    completed_states.append(state)

        for state in decode_states:
            state.emitted_tokens += 1
            state.output_emitted_at_steps.append(boundary)
            if state.emitted_tokens == state.request.output_tokens:
                state.completed_at_step = boundary
                completed_states.append(state)

        completed_states.sort(key=lambda state: state.ordinal)
        used_tokens = sum(item.tokens for item in prefill_slices) + len(
            decode_states
        )
        steps.append(
            ContinuousBatchStep(
                iteration=current_step,
                admitted_request_ids=tuple(admitted),
                prefill_slices=tuple(prefill_slices),
                decoded_request_ids=tuple(
                    state.request.request_id for state in decode_states
                ),
                first_token_request_ids=tuple(first_tokens),
                completed_request_ids=tuple(
                    state.request.request_id for state in completed_states
                ),
                used_token_slots=used_tokens,
            )
        )
        for state in completed_states:
            finished[state.request.request_id] = state
        completed_ids = {
            state.request.request_id for state in completed_states
        }
        running = [
            state for state in running if state.request.request_id not in completed_ids
        ]
        current_step = boundary

    schedules: list[RequestSchedule] = []
    for request in requests:
        state = finished[request.request_id]
        assert state.prefill_completed_at_step is not None
        assert state.first_token_at_step is not None
        assert state.completed_at_step is not None
        if len(state.output_emitted_at_steps) != request.output_tokens:
            raise RuntimeError("output emission accounting mismatch")
        schedules.append(
            RequestSchedule(
                request_id=request.request_id,
                arrival_step=request.arrival_step,
                admitted_at_step=state.admitted_at_step,
                prefill_completed_at_step=state.prefill_completed_at_step,
                first_token_at_step=state.first_token_at_step,
                completed_at_step=state.completed_at_step,
                prompt_tokens=request.prompt_tokens,
                output_tokens=request.output_tokens,
                output_emitted_at_steps=tuple(state.output_emitted_at_steps),
            )
        )

    first_arrival = min(request.arrival_step for request in requests)
    completed_at = max(schedule.completed_at_step for schedule in schedules)
    elapsed_steps = completed_at - first_arrival
    modeled_tokens = sum(request.modeled_forward_tokens for request in requests)
    observed_tokens = sum(step.used_token_slots for step in steps)
    if modeled_tokens != observed_tokens:
        raise RuntimeError("scheduled token work does not match request accounting")
    elapsed_capacity = elapsed_steps * max_batch_tokens
    active_capacity = len(steps) * max_batch_tokens
    return ContinuousBatchingReport(
        max_batch_tokens=max_batch_tokens,
        max_running_sequences=max_running_sequences,
        max_prefill_tokens_per_request=max_prefill_tokens_per_request,
        first_arrival_step=first_arrival,
        completed_at_step=completed_at,
        elapsed_steps=elapsed_steps,
        active_steps=len(steps),
        prompt_tokens=sum(request.prompt_tokens for request in requests),
        output_tokens=sum(request.output_tokens for request in requests),
        modeled_forward_tokens=modeled_tokens,
        elapsed_token_capacity=elapsed_capacity,
        active_token_capacity=active_capacity,
        elapsed_token_utilization=modeled_tokens / elapsed_capacity,
        active_token_utilization=modeled_tokens / active_capacity,
        requests=tuple(schedules),
        steps=tuple(steps),
    )
