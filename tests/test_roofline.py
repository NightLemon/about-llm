from __future__ import annotations

import math

import pytest

from about_llm.inference import roofline_lower_bound

pytestmark = pytest.mark.formula


def test_roofline_identifies_memory_bound_workload() -> None:
    bound = roofline_lower_bound(
        flop_count=100,
        bytes_moved=100,
        effective_flops_per_second=100,
        effective_bytes_per_second=10,
    )
    assert bound.compute_seconds == pytest.approx(1)
    assert bound.memory_seconds == pytest.approx(10)
    assert bound.lower_bound_seconds == pytest.approx(10)
    assert bound.arithmetic_intensity == pytest.approx(1)
    assert bound.ridge_point == pytest.approx(10)
    assert bound.bottleneck == "memory"


def test_roofline_identifies_compute_and_balanced_workloads() -> None:
    compute = roofline_lower_bound(
        flop_count=1_000,
        bytes_moved=10,
        effective_flops_per_second=100,
        effective_bytes_per_second=100,
    )
    assert compute.bottleneck == "compute"

    balanced = roofline_lower_bound(
        flop_count=100,
        bytes_moved=10,
        effective_flops_per_second=100,
        effective_bytes_per_second=10,
    )
    assert balanced.bottleneck == "balanced"
    assert balanced.lower_bound_seconds == pytest.approx(1)


def test_zero_traffic_has_infinite_intensity_but_valid_bound() -> None:
    bound = roofline_lower_bound(
        flop_count=100,
        bytes_moved=0,
        effective_flops_per_second=100,
        effective_bytes_per_second=10,
    )
    assert math.isinf(bound.arithmetic_intensity)
    assert bound.bottleneck == "compute"


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan, True])
def test_roofline_rejects_invalid_work_amounts(bad: float) -> None:
    with pytest.raises(ValueError, match="non-negative finite"):
        roofline_lower_bound(
            flop_count=bad,
            bytes_moved=1,
            effective_flops_per_second=1,
            effective_bytes_per_second=1,
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan, True])
def test_roofline_rejects_invalid_ceilings(bad: float) -> None:
    with pytest.raises(ValueError):
        roofline_lower_bound(
            flop_count=1,
            bytes_moved=1,
            effective_flops_per_second=bad,
            effective_bytes_per_second=1,
        )
