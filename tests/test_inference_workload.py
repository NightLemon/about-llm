from __future__ import annotations

import math

import pytest

from about_llm.inference import (
    ArrivalProcess,
    ArrivalSchedule,
    build_arrival_schedule,
)


def test_burst_schedule_offers_every_request_at_start() -> None:
    schedule = build_arrival_schedule(4)

    assert schedule.process is ArrivalProcess.BURST
    assert schedule.offsets_seconds == (0.0, 0.0, 0.0, 0.0)
    assert schedule.nominal_requests_per_second is None
    assert schedule.realized_requests_per_second is None
    assert schedule.to_dict()["request_count"] == 4


def test_constant_open_loop_schedule_has_exact_interarrival_rate() -> None:
    schedule = build_arrival_schedule(
        4,
        process=ArrivalProcess.CONSTANT,
        requests_per_second=4,
    )

    assert schedule.offsets_seconds == (0.0, 0.25, 0.5, 0.75)
    assert schedule.scheduled_duration_seconds == pytest.approx(0.75)
    assert schedule.mean_interarrival_seconds == pytest.approx(0.25)
    assert schedule.realized_requests_per_second == pytest.approx(4.0)
    assert schedule.seed is None


def test_poisson_schedule_is_seeded_and_monotonic() -> None:
    first = build_arrival_schedule(
        100,
        process="poisson",
        requests_per_second=5,
        seed=17,
    )
    repeated = build_arrival_schedule(
        100,
        process="poisson",
        requests_per_second=5,
        seed=17,
    )
    changed = build_arrival_schedule(
        100,
        process="poisson",
        requests_per_second=5,
        seed=18,
    )

    assert first == repeated
    assert first.offsets_seconds != changed.offsets_seconds
    assert first.offsets_seconds[0] == 0
    assert all(
        current >= previous
        for previous, current in zip(
            first.offsets_seconds,
            first.offsets_seconds[1:],
            strict=False,
        )
    )
    assert first.nominal_requests_per_second == 5
    assert first.realized_requests_per_second is not None
    assert math.isfinite(first.realized_requests_per_second)


def test_single_request_open_loop_rate_is_not_estimable_from_interarrivals() -> None:
    schedule = build_arrival_schedule(
        1,
        process="poisson",
        requests_per_second=2,
        seed=3,
    )

    assert schedule.offsets_seconds == (0.0,)
    assert schedule.mean_interarrival_seconds is None
    assert schedule.realized_requests_per_second is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"requests": 0}, "positive integer"),
        ({"requests": True}, "positive integer"),
        ({"requests": 1, "process": "unknown"}, "unknown arrival process"),
        ({"requests": 1, "process": "constant"}, "requests_per_second"),
        (
            {
                "requests": 1,
                "process": "poisson",
                "requests_per_second": float("nan"),
            },
            "finite and positive",
        ),
        (
            {
                "requests": 1,
                "process": "poisson",
                "requests_per_second": 1,
                "seed": True,
            },
            "seed must be an integer",
        ),
        ({"requests": 1, "requests_per_second": 1}, "not used for burst"),
    ],
)
def test_arrival_schedule_builder_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_arrival_schedule(**kwargs)  # type: ignore[arg-type]


def test_arrival_schedule_dataclass_rejects_process_metadata_drift() -> None:
    with pytest.raises(ValueError, match="must not claim"):
        ArrivalSchedule(ArrivalProcess.BURST, (0.0,), 1.0, None)

    with pytest.raises(ValueError, match="non-decreasing"):
        ArrivalSchedule(ArrivalProcess.CONSTANT, (0.0, 1.0, 0.5), 1.0, None)

    with pytest.raises(ValueError, match="strictly increasing"):
        ArrivalSchedule(ArrivalProcess.CONSTANT, (0.0, 0.0), 1.0, None)

    with pytest.raises(ValueError, match="ArrivalProcess"):
        ArrivalSchedule("constant", (0.0,), 1.0, None)  # type: ignore[arg-type]
