"""Small multimodal metrics with explicit coordinate and text conventions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _levenshtein_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(hypothesis, start=1):
            substitution_cost = int(reference_character != hypothesis_character)
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return Unicode-code-point Levenshtein distance divided by reference length.

    The caller owns Unicode normalization, whitespace, casing, and grapheme policy.
    CER can exceed 1 when insertions outnumber reference characters. An empty
    reference is rejected because its denominator is undefined.
    """

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if not reference:
        raise ValueError("reference must not be empty")
    return _levenshtein_distance(reference, hypothesis) / len(reference)


def _finite_coordinates(name: str, values: Sequence[float], length: int) -> list[float]:
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} coordinates")
    result = list(values)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in result):
        raise ValueError(f"{name} coordinates must be finite numbers")
    return result


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return continuous-coordinate IoU for ``(x_min, y_min, x_max, y_max)`` boxes."""

    ax1, ay1, ax2, ay2 = _finite_coordinates("box_a", box_a, 4)
    bx1, by1, bx2, by2 = _finite_coordinates("box_b", box_b, 4)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        raise ValueError("boxes must have positive width and height")

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def temporal_iou(interval_a: Sequence[float], interval_b: Sequence[float]) -> float:
    """Return IoU for continuous time intervals ``(start, end)``.

    For continuous durations, open/closed endpoints have zero measure; callers
    using discrete inclusive frame indices need a different ``+1`` convention.
    """

    start_a, end_a = _finite_coordinates("interval_a", interval_a, 2)
    start_b, end_b = _finite_coordinates("interval_b", interval_b, 2)
    if end_a <= start_a or end_b <= start_b:
        raise ValueError("intervals must have positive duration")

    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - intersection
    return intersection / union
