from __future__ import annotations

import math

import pytest

from about_llm.evaluation import box_iou, character_error_rate, temporal_iou


def test_character_error_rate_counts_unicode_code_point_edits() -> None:
    assert character_error_rate("语言模型", "语言大模") == pytest.approx(0.5)
    assert character_error_rate("abc", "abc") == 0
    assert character_error_rate("a", "abcdef") == pytest.approx(5.0)


def test_character_error_rate_rejects_undefined_empty_reference() -> None:
    with pytest.raises(ValueError, match="reference must not be empty"):
        character_error_rate("", "")
    with pytest.raises(TypeError, match="must be strings"):
        character_error_rate("text", 1)  # type: ignore[arg-type]


def test_box_iou_uses_continuous_coordinate_areas() -> None:
    assert box_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1


def test_temporal_iou_uses_continuous_duration() -> None:
    assert temporal_iou((0, 10), (5, 15)) == pytest.approx(1 / 3)
    assert temporal_iou((0, 1), (1, 2)) == 0


@pytest.mark.parametrize(
    "box",
    [(0, 0, 0, 1), (0, 1, 1, 0), (0, 0, math.inf, 1), (0, 0, 1)],
)
def test_box_iou_rejects_invalid_boxes(box: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        box_iou(box, (0, 0, 1, 1))


@pytest.mark.parametrize("interval", [(0, 0), (2, 1), (0, math.nan), (0,)])
def test_temporal_iou_rejects_invalid_intervals(interval: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        temporal_iou(interval, (0, 1))
