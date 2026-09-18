from __future__ import annotations

from fractions import Fraction

import pytest

pytestmark = pytest.mark.formula


def _qualified_cost(total_cost: int, completed: int) -> Fraction:
    if completed <= 0:
        raise ValueError("completed must be positive")
    return Fraction(total_cost, completed)


def test_independent_tools_shorten_critical_path_without_removing_work() -> None:
    model_before = 2
    tool_a = 6
    tool_b = 10
    model_after = 3

    assert model_before + tool_a + tool_b + model_after == 21
    assert model_before + max(tool_a, tool_b) + model_after == 15
    assert tool_a + tool_b == 16
    assert 2 * 15 == 30


def test_on_time_qualified_cost_uses_the_smaller_denominator() -> None:
    total_cost = 1

    assert _qualified_cost(total_cost, 5) == Fraction(1, 5)
    assert _qualified_cost(total_cost, 4) == Fraction(1, 4)


def test_zero_qualified_tasks_leave_unit_cost_undefined() -> None:
    with pytest.raises(ValueError, match="positive"):
        _qualified_cost(total_cost=7, completed=0)
