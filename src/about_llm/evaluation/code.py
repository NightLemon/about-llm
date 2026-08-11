"""Code-generation metrics with explicit sample-count semantics."""

from __future__ import annotations

import math


def pass_at_k(*, num_samples: int, num_correct: int, k: int) -> float:
    """Estimate the probability that at least one of ``k`` samples is correct.

    This is ``1 - C(n-c, k) / C(n, k)`` for ``n`` evaluated generations with
    ``c`` correct. Under the usual i.i.d. generation assumption it is the common
    unbiased pass@k estimator. It is not a single-attempt production success rate.
    """

    values = {"num_samples": num_samples, "num_correct": num_correct, "k": k}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be in [0, num_samples]")
    if not 1 <= k <= num_samples:
        raise ValueError("k must be in [1, num_samples]")
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)
