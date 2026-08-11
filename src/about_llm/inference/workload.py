"""Deterministic finite arrival schedules for inference load experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ArrivalProcess(str, Enum):
    """Supported client-side request arrival processes."""

    BURST = "burst"
    CONSTANT = "constant"
    POISSON = "poisson"


@dataclass(frozen=True)
class ArrivalSchedule:
    """A finite schedule of monotonic offsets from benchmark start."""

    process: ArrivalProcess
    offsets_seconds: tuple[float, ...]
    nominal_requests_per_second: float | None
    seed: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.process, ArrivalProcess):
            raise ValueError("process must be an ArrivalProcess")
        if not self.offsets_seconds:
            raise ValueError("arrival schedule must contain at least one request")
        if self.offsets_seconds[0] != 0:
            raise ValueError("the first arrival offset must equal zero")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in self.offsets_seconds
        ):
            raise ValueError("arrival offsets must be finite and non-negative")
        if any(
            current < previous
            for previous, current in zip(
                self.offsets_seconds,
                self.offsets_seconds[1:],
                strict=False,
            )
        ):
            raise ValueError("arrival offsets must be non-decreasing")
        if self.process is ArrivalProcess.BURST:
            if any(self.offsets_seconds):
                raise ValueError("burst arrival offsets must all equal zero")
            if self.nominal_requests_per_second is not None or self.seed is not None:
                raise ValueError("burst schedules must not claim a rate or random seed")
        else:
            rate = self.nominal_requests_per_second
            if (
                rate is None
                or isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(rate)
                or rate <= 0
            ):
                raise ValueError("open-loop schedules require a finite positive rate")
            if self.request_count > 1 and any(
                current <= previous
                for previous, current in zip(
                    self.offsets_seconds,
                    self.offsets_seconds[1:],
                    strict=False,
                )
            ):
                raise ValueError("open-loop arrival offsets must be strictly increasing")
            if self.process is ArrivalProcess.CONSTANT and self.seed is not None:
                raise ValueError("constant schedules must not claim a random seed")
            if self.process is ArrivalProcess.POISSON and (
                isinstance(self.seed, bool) or not isinstance(self.seed, int)
            ):
                raise ValueError("Poisson schedules require an integer seed")

    @property
    def request_count(self) -> int:
        return len(self.offsets_seconds)

    @property
    def scheduled_duration_seconds(self) -> float:
        """Time from the first scheduled arrival to the last one."""

        return self.offsets_seconds[-1]

    @property
    def mean_interarrival_seconds(self) -> float | None:
        if self.request_count < 2:
            return None
        return self.scheduled_duration_seconds / (self.request_count - 1)

    @property
    def realized_requests_per_second(self) -> float | None:
        """Finite-sample inter-arrival rate, excluding the first event."""

        mean_interarrival = self.mean_interarrival_seconds
        if mean_interarrival is None or mean_interarrival == 0:
            return None
        return 1 / mean_interarrival

    def to_dict(self) -> dict[str, object]:
        return {
            "process": self.process.value,
            "request_count": self.request_count,
            "offsets_seconds": self.offsets_seconds,
            "nominal_requests_per_second": self.nominal_requests_per_second,
            "seed": self.seed,
            "scheduled_duration_seconds": self.scheduled_duration_seconds,
            "mean_interarrival_seconds": self.mean_interarrival_seconds,
            "realized_requests_per_second": self.realized_requests_per_second,
        }


def build_arrival_schedule(
    requests: int,
    *,
    process: ArrivalProcess | str = ArrivalProcess.BURST,
    requests_per_second: float | None = None,
    seed: int = 0,
) -> ArrivalSchedule:
    """Build a finite burst, constant-rate, or seeded Poisson schedule.

    The first request is scheduled at offset zero. A Poisson schedule samples
    exponential inter-arrival times; its finite realized rate generally differs
    from the nominal rate even though repeated runs with the same seed match.
    """

    if isinstance(requests, bool) or not isinstance(requests, int) or requests <= 0:
        raise ValueError("requests must be a positive integer")
    try:
        selected = ArrivalProcess(process)
    except ValueError as error:
        raise ValueError(f"unknown arrival process {process!r}") from error
    if selected is ArrivalProcess.BURST:
        if requests_per_second is not None:
            raise ValueError("requests_per_second is not used for burst arrivals")
        return ArrivalSchedule(selected, (0.0,) * requests, None, None)
    if (
        requests_per_second is None
        or isinstance(requests_per_second, bool)
        or not isinstance(requests_per_second, (int, float))
        or not math.isfinite(requests_per_second)
        or requests_per_second <= 0
    ):
        raise ValueError(
            "requests_per_second must be finite and positive for open-loop arrivals"
        )
    rate = float(requests_per_second)
    if selected is ArrivalProcess.CONSTANT:
        offsets = tuple(index / rate for index in range(requests))
        if not all(math.isfinite(value) for value in offsets):
            raise ValueError("constant arrival schedule exceeds finite clock range")
        return ArrivalSchedule(selected, offsets, rate, None)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if requests == 1:
        offsets = (0.0,)
    else:
        rng = np.random.default_rng(seed)
        intervals = rng.exponential(scale=1 / rate, size=requests - 1)
        cumulative = np.cumsum(intervals, dtype=np.float64)
        if not np.all(np.isfinite(cumulative)):
            raise ValueError("Poisson arrival schedule exceeds finite clock range")
        offsets = (0.0, *(float(value) for value in cumulative))
    return ArrivalSchedule(selected, offsets, rate, seed)
