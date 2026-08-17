"""Exact optional-stopping oracle for repeated two-sided sign tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import comb

MAX_SEQUENTIAL_LOOKS = 64
MAX_SEQUENTIAL_SAMPLE_COUNT = 512


def _fraction_payload(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def two_sided_sign_test_p_value(
    *,
    positive_count: int,
    sample_count: int,
) -> Fraction:
    """Return the exact doubled-equal-tail p-value under fair independent signs.

    ``sample_count`` counts informative non-tied pairs and ``positive_count`` is
    the number with a positive candidate-minus-baseline difference. Under the
    null used by this teaching oracle, those signs are i.i.d. Bernoulli(1/2).
    The two-sided convention is twice the smaller inclusive binomial tail,
    capped at one. Other definitions of a two-sided exact binomial p-value can
    differ, so the convention is part of the public contract.
    """

    if (
        type(sample_count) is not int
        or sample_count <= 0
        or sample_count > MAX_SEQUENTIAL_SAMPLE_COUNT
    ):
        raise ValueError(
            "sample_count must be an integer in "
            f"[1, {MAX_SEQUENTIAL_SAMPLE_COUNT}]"
        )
    if (
        type(positive_count) is not int
        or positive_count < 0
        or positive_count > sample_count
    ):
        raise ValueError("positive_count must be an integer in [0, sample_count]")

    equally_extreme_count = min(positive_count, sample_count - positive_count)
    smaller_tail_sequences = sum(
        comb(sample_count, count) for count in range(equally_extreme_count + 1)
    )
    return min(
        Fraction(1, 1),
        Fraction(2 * smaller_tail_sequences, 2**sample_count),
    )


@dataclass(frozen=True, slots=True)
class SequentialSignTestLook:
    """Exact null probabilities at one planned interim look."""

    sample_count: int
    per_look_alpha: Fraction
    lower_rejection_max_positive_count: int | None
    upper_rejection_min_positive_count: int | None
    fixed_look_null_rejection_probability: Fraction
    first_rejection_probability: Fraction
    cumulative_rejection_probability: Fraction
    alive_probability_after_look: Fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "per_look_alpha": _fraction_payload(self.per_look_alpha),
            "rejection_region": {
                "lower_positive_count_at_most": (
                    self.lower_rejection_max_positive_count
                ),
                "upper_positive_count_at_least": (
                    self.upper_rejection_min_positive_count
                ),
            },
            "fixed_look_null_rejection_probability": _fraction_payload(
                self.fixed_look_null_rejection_probability
            ),
            "first_rejection_probability": _fraction_payload(
                self.first_rejection_probability
            ),
            "cumulative_rejection_probability": _fraction_payload(
                self.cumulative_rejection_probability
            ),
            "alive_probability_after_look": _fraction_payload(
                self.alive_probability_after_look
            ),
        }


@dataclass(frozen=True, slots=True)
class SequentialSignTestAnalysis:
    """Exact repeated-testing error under a prespecified look schedule."""

    look_sample_counts: tuple[int, ...]
    per_look_alpha: Fraction
    union_bound: Fraction
    maximum_sample_count: int
    logical_binary_sign_sequences: int
    dynamic_programming_state_cells_evaluated: int
    looks: tuple[SequentialSignTestLook, ...]
    familywise_null_rejection_probability: Fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "look_sample_counts": list(self.look_sample_counts),
            "per_look_alpha": _fraction_payload(self.per_look_alpha),
            "union_bound": _fraction_payload(self.union_bound),
            "maximum_sample_count": self.maximum_sample_count,
            "logical_binary_sign_sequences": self.logical_binary_sign_sequences,
            "dynamic_programming_state_cells_evaluated": (
                self.dynamic_programming_state_cells_evaluated
            ),
            "familywise_null_rejection_probability": _fraction_payload(
                self.familywise_null_rejection_probability
            ),
            "looks": [look.to_dict() for look in self.looks],
        }


def analyze_repeated_two_sided_sign_tests(
    look_sample_counts: Iterable[int],
    *,
    per_look_alpha: Fraction,
) -> SequentialSignTestAnalysis:
    """Compute exact type-I error when testing and stopping at planned looks.

    The dynamic program propagates probability mass over ``(n, positive_count)``
    states and removes paths at their first rejecting look. It therefore does
    not enumerate the ``2**max_n`` logical sign sequences.

    The result is conditional on a very specific null: informative pair signs
    are independent fair coin flips, the look schedule and p-value convention
    are fixed before outcomes are seen, and the rule stops at the first
    ``p <= per_look_alpha``. It is not a confidence sequence, an effect-size
    estimate, a power calculation, or evidence that real cases are independent,
    representative, correctly labeled, or free of cluster dependence.
    """

    look_tuple = tuple(look_sample_counts)
    if not look_tuple:
        raise ValueError("at least one look sample count is required")
    if len(look_tuple) > MAX_SEQUENTIAL_LOOKS:
        raise ValueError(f"look count cannot exceed {MAX_SEQUENTIAL_LOOKS}")
    for sample_count in look_tuple:
        if (
            type(sample_count) is not int
            or sample_count <= 0
            or sample_count > MAX_SEQUENTIAL_SAMPLE_COUNT
        ):
            raise ValueError(
                "look sample counts must be integers in "
                f"[1, {MAX_SEQUENTIAL_SAMPLE_COUNT}]"
            )
    if any(left >= right for left, right in pairwise(look_tuple)):
        raise ValueError("look sample counts must be strictly increasing")
    if type(per_look_alpha) is not Fraction or not 0 < per_look_alpha < 1:
        raise ValueError("per_look_alpha must be a Fraction strictly between 0 and 1")

    maximum_sample_count = look_tuple[-1]
    look_set = frozenset(look_tuple)
    alive_by_positive_count: dict[int, Fraction] = {0: Fraction(1, 1)}
    cumulative_rejection = Fraction(0, 1)
    state_cells_evaluated = 0
    look_results: list[SequentialSignTestLook] = []

    for sample_count in range(1, maximum_sample_count + 1):
        next_alive: dict[int, Fraction] = {}
        for positive_count, probability in alive_by_positive_count.items():
            half_probability = probability / 2
            next_alive[positive_count] = (
                next_alive.get(positive_count, Fraction(0, 1)) + half_probability
            )
            next_alive[positive_count + 1] = (
                next_alive.get(positive_count + 1, Fraction(0, 1))
                + half_probability
            )
        alive_by_positive_count = next_alive
        state_cells_evaluated += len(alive_by_positive_count)
        if sample_count not in look_set:
            continue

        rejecting_counts = tuple(
            positive_count
            for positive_count in range(sample_count + 1)
            if two_sided_sign_test_p_value(
                positive_count=positive_count,
                sample_count=sample_count,
            )
            <= per_look_alpha
        )
        rejecting_set = frozenset(rejecting_counts)
        first_rejection = sum(
            (
                probability
                for positive_count, probability in alive_by_positive_count.items()
                if positive_count in rejecting_set
            ),
            start=Fraction(0, 1),
        )
        alive_by_positive_count = {
            positive_count: probability
            for positive_count, probability in alive_by_positive_count.items()
            if positive_count not in rejecting_set
        }
        cumulative_rejection += first_rejection
        alive_probability = sum(
            alive_by_positive_count.values(), start=Fraction(0, 1)
        )
        if cumulative_rejection + alive_probability != 1:
            raise AssertionError("sequential probability mass must sum to one")

        fixed_look_rejection = sum(
            (
                Fraction(comb(sample_count, positive_count), 2**sample_count)
                for positive_count in rejecting_counts
            ),
            start=Fraction(0, 1),
        )
        lower_counts = tuple(
            count for count in rejecting_counts if count <= sample_count // 2
        )
        upper_counts = tuple(
            count for count in rejecting_counts if count > sample_count // 2
        )
        look_results.append(
            SequentialSignTestLook(
                sample_count=sample_count,
                per_look_alpha=per_look_alpha,
                lower_rejection_max_positive_count=(
                    max(lower_counts) if lower_counts else None
                ),
                upper_rejection_min_positive_count=(
                    min(upper_counts) if upper_counts else None
                ),
                fixed_look_null_rejection_probability=fixed_look_rejection,
                first_rejection_probability=first_rejection,
                cumulative_rejection_probability=cumulative_rejection,
                alive_probability_after_look=alive_probability,
            )
        )

    union_bound = min(Fraction(1, 1), len(look_tuple) * per_look_alpha)
    if cumulative_rejection > union_bound:
        raise AssertionError("exact familywise error cannot exceed the union bound")
    return SequentialSignTestAnalysis(
        look_sample_counts=look_tuple,
        per_look_alpha=per_look_alpha,
        union_bound=union_bound,
        maximum_sample_count=maximum_sample_count,
        logical_binary_sign_sequences=2**maximum_sample_count,
        dynamic_programming_state_cells_evaluated=state_cells_evaluated,
        looks=tuple(look_results),
        familywise_null_rejection_probability=cumulative_rejection,
    )
