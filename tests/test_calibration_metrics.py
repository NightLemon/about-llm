from __future__ import annotations

import numpy as np
import pytest

from about_llm.evaluation import binary_calibration, risk_coverage_curve


def test_binary_calibration_known_example() -> None:
    result = binary_calibration([0, 1, 1, 0], [0.1, 0.8, 0.6, 0.4], bins=2)

    assert result.count == 4
    assert result.brier_score == pytest.approx(0.0925)
    assert result.expected_calibration_error == pytest.approx(0.275)
    assert [item.count for item in result.bins] == [2, 2]
    assert result.bins[0].mean_probability == pytest.approx(0.25)
    assert result.bins[0].positive_rate == pytest.approx(0.0)
    assert result.bins[1].mean_probability == pytest.approx(0.7)
    assert result.bins[1].positive_rate == pytest.approx(1.0)


def test_perfect_binary_probabilities_have_zero_brier_and_ece() -> None:
    result = binary_calibration([0, 1, 0, 1], [0, 1, 0, 1], bins=10)

    assert result.brier_score == 0
    assert result.expected_calibration_error == 0


def test_probability_one_belongs_to_final_bin() -> None:
    result = binary_calibration([1], [1], bins=5)

    assert len(result.bins) == 1
    assert result.bins[0].lower == pytest.approx(0.8)
    assert result.bins[0].upper == 1


def test_empty_calibration_bins_are_omitted() -> None:
    result = binary_calibration([0, 1], [0.01, 0.99], bins=10)

    assert len(result.bins) == 2
    assert sum(item.count for item in result.bins) == result.count


def test_risk_coverage_curve_groups_confidence_ties() -> None:
    points = risk_coverage_curve([1, 0, 1, 0], [0.9, 0.8, 0.8, 0.1])

    assert [point.threshold for point in points] == pytest.approx([0.9, 0.8, 0.1])
    assert [point.accepted_count for point in points] == [1, 3, 4]
    assert [point.coverage for point in points] == pytest.approx([0.25, 0.75, 1.0])
    assert [point.risk for point in points] == pytest.approx([0, 1 / 3, 0.5])


def test_risk_coverage_final_point_is_full_dataset_error_rate() -> None:
    points = risk_coverage_curve([1, 1, 0], [0.2, 0.7, 0.4])

    assert points[-1].coverage == 1
    assert points[-1].risk == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ([], [], "must not be empty"),
        ([0, 1], [0.5], "same length"),
        ([0, 2], [0.1, 0.9], "only 0 and 1"),
        ([0, 1], [-0.1, 0.9], r"in \[0, 1\]"),
        ([0, 1], [0.1, float("nan")], "finite"),
        ([[0, 1]], [[0.1, 0.9]], "one-dimensional"),
    ],
)
def test_binary_calibration_rejects_invalid_inputs(
    labels: object, probabilities: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        binary_calibration(labels, probabilities)


@pytest.mark.parametrize("bins", [0, -1])
def test_binary_calibration_requires_positive_bins(bins: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        binary_calibration([0], [0.1], bins=bins)


def test_binary_calibration_rejects_boolean_bin_count() -> None:
    with pytest.raises(TypeError, match="integer"):
        binary_calibration([0], [0.1], bins=True)


def test_risk_coverage_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        risk_coverage_curve([1], [1.1])


def test_metrics_accept_numpy_arrays() -> None:
    labels = np.array([0, 1], dtype=np.int64)
    probabilities = np.array([0.2, 0.8], dtype=np.float32)

    assert binary_calibration(labels, probabilities, bins=2).count == 2
    assert risk_coverage_curve(labels, probabilities)[-1].coverage == 1
