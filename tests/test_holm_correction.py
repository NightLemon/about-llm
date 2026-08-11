from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import pytest

from about_llm.evaluation import holm_bonferroni_correction

ROOT = Path(__file__).resolve().parents[1]


def test_holm_fixture_uses_running_maximum_and_maps_back_to_input_order() -> None:
    result = holm_bonferroni_correction([0.04, 0.01, 0.03, 0.20])

    assert result.family_size == 4
    assert [item.original_index for item in result.ordered_hypotheses] == [1, 2, 0, 3]
    assert [item.multiplier for item in result.ordered_hypotheses] == [4, 3, 2, 1]
    assert [item.scaled_p_value for item in result.ordered_hypotheses] == pytest.approx(
        [0.04, 0.09, 0.08, 0.20]
    )
    assert [item.adjusted_p_value for item in result.ordered_hypotheses] == pytest.approx(
        [0.04, 0.09, 0.09, 0.20]
    )
    assert result.adjusted_p_values == pytest.approx((0.09, 0.04, 0.09, 0.20))
    assert result.rejected == (False, True, False, False)


def test_equal_p_values_share_adjustment_and_use_stable_input_tie_order() -> None:
    result = holm_bonferroni_correction([0.01, 0.01, 0.20])

    assert [item.original_index for item in result.ordered_hypotheses] == [0, 1, 2]
    assert result.adjusted_p_values == pytest.approx((0.03, 0.03, 0.20))
    assert result.rejected == (True, True, False)


def test_adjusted_values_are_monotone_in_rank_capped_and_not_below_raw_p() -> None:
    result = holm_bonferroni_correction([0.8, 0.001, 0.4, 1.0, 0.2])
    ordered = result.ordered_hypotheses

    assert all(
        left.adjusted_p_value <= right.adjusted_p_value
        for left, right in pairwise(ordered)
    )
    assert all(item.p_value <= item.adjusted_p_value <= 1 for item in ordered)
    assert ordered[-1].adjusted_p_value == 1


def test_single_hypothesis_is_unchanged() -> None:
    result = holm_bonferroni_correction([0.049], alpha=0.05)

    assert result.adjusted_p_values == pytest.approx((0.049,))
    assert result.rejected == (True,)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: holm_bonferroni_correction([]), "non-empty one-dimensional"),
        (lambda: holm_bonferroni_correction([[0.1]]), "one-dimensional"),
        (lambda: holm_bonferroni_correction([True]), "real numbers"),
        (lambda: holm_bonferroni_correction(["0.1"]), "real numbers"),
        (lambda: holm_bonferroni_correction([float("nan")]), "finite"),
        (lambda: holm_bonferroni_correction([-0.01]), r"\[0, 1\]"),
        (lambda: holm_bonferroni_correction([1.01]), r"\[0, 1\]"),
        (lambda: holm_bonferroni_correction([0.01], alpha=0), "alpha"),
        (lambda: holm_bonferroni_correction([0.01], alpha=True), "alpha"),
    ],
)
def test_invalid_holm_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_holm_toy_records_rank_ledger_and_scope() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "projects" / "evaluation-gate" / "holm_correction_toy.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["input_order"]["p_values"] == [0.04, 0.01, 0.03, 0.2]
    assert artifact["holm"]["adjusted_p_values"] == pytest.approx(
        [0.09, 0.04, 0.09, 0.2]
    )
    assert artifact["holm"]["rejected"] == [False, True, False, False]
    assert artifact["scope"] == {
        "arbitrary_dependence_fwer_control_requires_valid_input_p_values": True,
        "effect_size_or_practical_importance_estimated": False,
        "family_prespecified_or_selection_bias_repaired": False,
        "holm_rank_and_running_maximum_executed": True,
        "repeated_peeking_or_optional_stopping_repaired": False,
    }
