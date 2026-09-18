from __future__ import annotations

from itertools import pairwise

import pytest

pytestmark = pytest.mark.formula


def _maximum_interval(output_times_ms: list[int]) -> int:
    return max(later - earlier for earlier, later in pairwise(output_times_ms))


def test_nonpreemptive_long_prefill_can_increase_incumbent_itl() -> None:
    fixed_output_times = [10, 20, 30, 40]
    continuous_output_times = [10, 60, 70, 80]

    assert _maximum_interval(fixed_output_times) == 10
    assert _maximum_interval(continuous_output_times) == 50


def test_two_prefill_chunks_reduce_the_toy_maximum_itl_without_changing_work() -> None:
    chunked_output_times = [10, 40, 70, 80]
    decode_work_ms = 4 * 10 + 1 * 10
    prefill_work_ms = 2 * 20

    assert _maximum_interval(chunked_output_times) == 30
    assert decode_work_ms + prefill_work_ms == 90


def test_maximum_interval_requires_at_least_two_outputs() -> None:
    with pytest.raises(ValueError, match="max"):
        _maximum_interval([10])
