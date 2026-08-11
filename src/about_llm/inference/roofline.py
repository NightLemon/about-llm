"""Assumption-explicit roofline calculations for performance exercises."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Bottleneck = Literal["compute", "memory", "balanced"]


def _non_negative_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _positive_finite(name: str, value: float) -> float:
    result = _non_negative_finite(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class RooflineBound:
    """Ideal lower-bound components under supplied effective ceilings."""

    compute_seconds: float
    memory_seconds: float
    lower_bound_seconds: float
    arithmetic_intensity: float
    ridge_point: float
    bottleneck: Bottleneck


def roofline_lower_bound(
    *,
    flop_count: float,
    bytes_moved: float,
    effective_flops_per_second: float,
    effective_bytes_per_second: float,
) -> RooflineBound:
    """Return ``max(FLOPs/throughput, bytes/bandwidth)`` and its bottleneck.

    This is an idealized lower bound. It excludes launch latency, dependencies,
    synchronization, communication, scheduling, queueing, thermal throttling,
    and any mismatch between the supplied effective ceilings and the workload.
    """

    flops = _non_negative_finite("flop_count", flop_count)
    traffic = _non_negative_finite("bytes_moved", bytes_moved)
    throughput = _positive_finite(
        "effective_flops_per_second", effective_flops_per_second
    )
    bandwidth = _positive_finite(
        "effective_bytes_per_second", effective_bytes_per_second
    )

    compute_seconds = flops / throughput
    memory_seconds = traffic / bandwidth
    lower_bound = max(compute_seconds, memory_seconds)
    arithmetic_intensity = math.inf if traffic == 0 else flops / traffic
    ridge_point = throughput / bandwidth
    if math.isclose(compute_seconds, memory_seconds, rel_tol=1e-12, abs_tol=0.0):
        bottleneck: Bottleneck = "balanced"
    elif compute_seconds > memory_seconds:
        bottleneck = "compute"
    else:
        bottleneck = "memory"
    return RooflineBound(
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        lower_bound_seconds=lower_bound,
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
        bottleneck=bottleneck,
    )
