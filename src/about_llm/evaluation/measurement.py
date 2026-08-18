"""Exact teaching controls for annotation measurement and sign-test planning.

The functions in this module deliberately separate three questions:

* inter-rater reliability: do two raters apply labels consistently?
* criterion validity: do those labels agree with an external criterion?
* statistical power: if an effect exists under a declared model, can the
  planned sign test detect it?

None of these calculations establishes that a construct was operationalized
well, that an external criterion is itself correct, or that sampled cases
represent a target population.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import comb

MAX_MEASUREMENT_ITEMS = 100_000
MAX_SIGN_TEST_SAMPLE_COUNT = 512
MAX_PROBABILITY_GRID_DENOMINATOR = 100_000


def _fraction_payload(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _validated_label_pairs(
    first: Iterable[str],
    second: Iterable[str],
    *,
    first_name: str,
    second_name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    first_labels = tuple(first)
    second_labels = tuple(second)
    if not first_labels:
        raise ValueError("at least one labeled item is required")
    if len(first_labels) != len(second_labels):
        raise ValueError(f"{first_name} and {second_name} must have equal length")
    if len(first_labels) > MAX_MEASUREMENT_ITEMS:
        raise ValueError(f"labeled item count cannot exceed {MAX_MEASUREMENT_ITEMS}")
    for name, labels in ((first_name, first_labels), (second_name, second_labels)):
        if any(type(label) is not str or not label for label in labels):
            raise ValueError(f"{name} labels must be non-empty strings")
    return first_labels, second_labels


@dataclass(frozen=True, slots=True)
class CohenKappaResult:
    """Observed agreement and chance-corrected Cohen's kappa for two raters."""

    item_count: int
    categories: tuple[str, ...]
    observed_agreement: Fraction
    chance_agreement: Fraction
    kappa: Fraction | None

    def to_dict(self) -> dict[str, object]:
        return {
            "item_count": self.item_count,
            "categories": list(self.categories),
            "observed_agreement": _fraction_payload(self.observed_agreement),
            "chance_agreement": _fraction_payload(self.chance_agreement),
            "kappa": None if self.kappa is None else _fraction_payload(self.kappa),
            "kappa_defined": self.kappa is not None,
        }


def cohen_kappa(
    rater_a: Iterable[str],
    rater_b: Iterable[str],
) -> CohenKappaResult:
    """Compute exact Cohen's kappa from paired categorical labels.

    Expected agreement is computed from the two observed marginal label
    distributions. If expected agreement is one, the usual kappa denominator
    is zero and ``kappa`` is returned as ``None`` instead of inventing a value.

    Kappa is a reliability statistic, not an accuracy or validity statistic.
    It is also sensitive to prevalence and rater marginal distributions. The
    item pairing, independent-item assumptions, rubric, sampling process and
    category semantics must be justified outside this function.
    """

    labels_a, labels_b = _validated_label_pairs(
        rater_a,
        rater_b,
        first_name="rater_a",
        second_name="rater_b",
    )
    item_count = len(labels_a)
    categories = tuple(sorted(set(labels_a) | set(labels_b)))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    observed = Fraction(
        sum(left == right for left, right in zip(labels_a, labels_b, strict=True)),
        item_count,
    )
    chance = sum(
        (
            Fraction(counts_a[category], item_count)
            * Fraction(counts_b[category], item_count)
            for category in categories
        ),
        start=Fraction(0, 1),
    )
    kappa = None if chance == 1 else (observed - chance) / (1 - chance)
    return CohenKappaResult(
        item_count=item_count,
        categories=categories,
        observed_agreement=observed,
        chance_agreement=chance,
        kappa=kappa,
    )


@dataclass(frozen=True, slots=True)
class CriterionValidityResult:
    """Exact multiclass agreement with an explicitly supplied criterion."""

    item_count: int
    categories: tuple[str, ...]
    accuracy: Fraction
    confusion_matrix: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "item_count": self.item_count,
            "accuracy": _fraction_payload(self.accuracy),
            "confusion": {
                "row_semantics": "criterion_label",
                "column_semantics": "observed_label",
                "categories_in_row_and_column_order": list(self.categories),
                "matrix": [list(row) for row in self.confusion_matrix],
            },
        }


def criterion_validity(
    observed_labels: Iterable[str],
    criterion_labels: Iterable[str],
) -> CriterionValidityResult:
    """Compare labels with an external criterion using accuracy and confusion.

    Rows in the returned confusion matrix are criterion labels; columns are
    observed labels. A high score only shows agreement with the supplied
    criterion on these items. It does not prove that the criterion is correct,
    independent, representative, current, or a faithful operationalization of
    the intended construct.
    """

    observed, criterion = _validated_label_pairs(
        observed_labels,
        criterion_labels,
        first_name="observed_labels",
        second_name="criterion_labels",
    )
    categories = tuple(sorted(set(observed) | set(criterion)))
    category_index = {category: index for index, category in enumerate(categories)}
    mutable_matrix = [[0 for _ in categories] for _ in categories]
    correct = 0
    for observed_label, criterion_label in zip(observed, criterion, strict=True):
        mutable_matrix[category_index[criterion_label]][category_index[observed_label]] += 1
        correct += observed_label == criterion_label
    return CriterionValidityResult(
        item_count=len(observed),
        categories=categories,
        accuracy=Fraction(correct, len(observed)),
        confusion_matrix=tuple(tuple(row) for row in mutable_matrix),
    )


def _validate_fraction_probability(
    value: Fraction,
    *,
    name: str,
    lower_inclusive: bool = False,
    upper_inclusive: bool = True,
) -> None:
    if type(value) is not Fraction:
        raise ValueError(f"{name} must be a Fraction")
    lower_valid = value >= 0 if lower_inclusive else value > 0
    upper_valid = value <= 1 if upper_inclusive else value < 1
    if not lower_valid or not upper_valid:
        lower_symbol = "[" if lower_inclusive else "("
        upper_symbol = "]" if upper_inclusive else ")"
        raise ValueError(
            f"{name} must be a Fraction in {lower_symbol}0, 1{upper_symbol}"
        )


def _validate_sign_sample_count(sample_count: int) -> None:
    if (
        type(sample_count) is not int
        or sample_count <= 0
        or sample_count > MAX_SIGN_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "sample_count must be an integer in "
            f"[1, {MAX_SIGN_TEST_SAMPLE_COUNT}]"
        )


def _upper_binomial_tail(
    *,
    sample_count: int,
    minimum_positive_count: int,
    positive_probability: Fraction,
) -> Fraction:
    negative_probability = 1 - positive_probability
    return sum(
        (
            comb(sample_count, positive_count)
            * positive_probability**positive_count
            * negative_probability ** (sample_count - positive_count)
            for positive_count in range(minimum_positive_count, sample_count + 1)
        ),
        start=Fraction(0, 1),
    )


@dataclass(frozen=True, slots=True)
class SignTestPowerResult:
    """Exact conditional power for a one-sided paired sign test."""

    informative_pair_count: int
    alpha: Fraction
    alternative_positive_probability: Fraction
    rejection_min_positive_count: int | None
    realized_null_rejection_probability: Fraction
    power: Fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "informative_pair_count": self.informative_pair_count,
            "alpha": _fraction_payload(self.alpha),
            "alternative_positive_probability": _fraction_payload(
                self.alternative_positive_probability
            ),
            "rejection_min_positive_count": self.rejection_min_positive_count,
            "realized_null_rejection_probability": _fraction_payload(
                self.realized_null_rejection_probability
            ),
            "power": _fraction_payload(self.power),
        }


def one_sided_sign_test_power(
    *,
    informative_pair_count: int,
    alternative_positive_probability: Fraction,
    alpha: Fraction = Fraction(1, 20),
) -> SignTestPowerResult:
    """Return exact rejection threshold and power under a binomial sign model.

    Tied pairs are excluded: ``informative_pair_count`` is the number of
    non-tied candidate/baseline pairs. Under the null, positive and negative
    differences are modeled as independent Bernoulli(1/2) signs. The rejection
    threshold is the smallest positive count whose inclusive upper null tail is
    at most ``alpha``. Power is that same tail under the declared alternative.

    This is conditional power for one fixed-horizon, one-sided test. It does not
    infer the discordance rate needed to convert informative pairs into total
    cases, support repeated peeking, or validate sampling, labels, independence,
    effect size, metric validity or practical importance.
    """

    _validate_sign_sample_count(informative_pair_count)
    _validate_fraction_probability(alpha, name="alpha", upper_inclusive=False)
    _validate_fraction_probability(
        alternative_positive_probability,
        name="alternative_positive_probability",
        lower_inclusive=True,
    )

    threshold: int | None = None
    null_probability = Fraction(0, 1)
    null_denominator = 2**informative_pair_count
    for positive_count in range(informative_pair_count, -1, -1):
        candidate_null_probability = null_probability + Fraction(
            comb(informative_pair_count, positive_count), null_denominator
        )
        if candidate_null_probability > alpha:
            break
        threshold = positive_count
        null_probability = candidate_null_probability

    power = (
        Fraction(0, 1)
        if threshold is None
        else _upper_binomial_tail(
            sample_count=informative_pair_count,
            minimum_positive_count=threshold,
            positive_probability=alternative_positive_probability,
        )
    )
    return SignTestPowerResult(
        informative_pair_count=informative_pair_count,
        alpha=alpha,
        alternative_positive_probability=alternative_positive_probability,
        rejection_min_positive_count=threshold,
        realized_null_rejection_probability=null_probability,
        power=power,
    )


@dataclass(frozen=True, slots=True)
class MinimumSignTestSampleSizeResult:
    """Smallest informative-pair count reaching target conditional power."""

    target_power: Fraction
    maximum_pair_count_searched: int
    analysis: SignTestPowerResult

    def to_dict(self) -> dict[str, object]:
        return {
            "target_power": _fraction_payload(self.target_power),
            "maximum_pair_count_searched": self.maximum_pair_count_searched,
            "analysis": self.analysis.to_dict(),
        }


def minimum_sign_test_sample_size(
    *,
    alternative_positive_probability: Fraction,
    target_power: Fraction,
    alpha: Fraction = Fraction(1, 20),
    maximum_pair_count: int = MAX_SIGN_TEST_SAMPLE_COUNT,
) -> MinimumSignTestSampleSizeResult:
    """Find the minimum informative-pair count for target conditional power."""

    _validate_fraction_probability(target_power, name="target_power")
    _validate_fraction_probability(alpha, name="alpha", upper_inclusive=False)
    _validate_fraction_probability(
        alternative_positive_probability,
        name="alternative_positive_probability",
        lower_inclusive=True,
    )
    if alternative_positive_probability <= Fraction(1, 2):
        raise ValueError("alternative_positive_probability must be greater than 1/2")
    _validate_sign_sample_count(maximum_pair_count)

    for pair_count in range(1, maximum_pair_count + 1):
        analysis = one_sided_sign_test_power(
            informative_pair_count=pair_count,
            alternative_positive_probability=alternative_positive_probability,
            alpha=alpha,
        )
        if analysis.power >= target_power:
            return MinimumSignTestSampleSizeResult(
                target_power=target_power,
                maximum_pair_count_searched=maximum_pair_count,
                analysis=analysis,
            )
    raise ValueError("target power is not reached within maximum_pair_count")


@dataclass(frozen=True, slots=True)
class MinimumDetectableSignEffectResult:
    """Smallest alternative on a declared rational grid reaching target power."""

    target_power: Fraction
    probability_grid_denominator: int
    minimum_positive_probability: Fraction
    minimum_margin_over_chance: Fraction
    analysis: SignTestPowerResult

    def to_dict(self) -> dict[str, object]:
        return {
            "target_power": _fraction_payload(self.target_power),
            "probability_grid_denominator": self.probability_grid_denominator,
            "minimum_positive_probability_on_grid": _fraction_payload(
                self.minimum_positive_probability
            ),
            "minimum_margin_over_chance_on_grid": _fraction_payload(
                self.minimum_margin_over_chance
            ),
            "analysis": self.analysis.to_dict(),
        }


def minimum_detectable_sign_effect(
    *,
    informative_pair_count: int,
    target_power: Fraction,
    alpha: Fraction = Fraction(1, 20),
    probability_grid_denominator: int = 1_000,
) -> MinimumDetectableSignEffectResult:
    """Invert exact power on a declared rational grid to obtain a sign MDE.

    The result is exact on the grid ``k / probability_grid_denominator``; it is
    not presented as the continuous mathematical root. The effect is the
    positive-sign probability margin over 1/2, conditional on non-tied pairs.
    """

    _validate_sign_sample_count(informative_pair_count)
    _validate_fraction_probability(target_power, name="target_power")
    _validate_fraction_probability(alpha, name="alpha", upper_inclusive=False)
    if (
        type(probability_grid_denominator) is not int
        or probability_grid_denominator < 2
        or probability_grid_denominator > MAX_PROBABILITY_GRID_DENOMINATOR
    ):
        raise ValueError(
            "probability_grid_denominator must be an integer in "
            f"[2, {MAX_PROBABILITY_GRID_DENOMINATOR}]"
        )

    lower_numerator = probability_grid_denominator // 2 + 1
    upper_numerator = probability_grid_denominator
    upper_analysis = one_sided_sign_test_power(
        informative_pair_count=informative_pair_count,
        alternative_positive_probability=Fraction(1, 1),
        alpha=alpha,
    )
    if upper_analysis.power < target_power:
        raise ValueError("target power is unreachable because this sample has no rejection region")

    while lower_numerator < upper_numerator:
        midpoint = (lower_numerator + upper_numerator) // 2
        midpoint_analysis = one_sided_sign_test_power(
            informative_pair_count=informative_pair_count,
            alternative_positive_probability=Fraction(
                midpoint, probability_grid_denominator
            ),
            alpha=alpha,
        )
        if midpoint_analysis.power >= target_power:
            upper_numerator = midpoint
        else:
            lower_numerator = midpoint + 1

    minimum_probability = Fraction(lower_numerator, probability_grid_denominator)
    analysis = one_sided_sign_test_power(
        informative_pair_count=informative_pair_count,
        alternative_positive_probability=minimum_probability,
        alpha=alpha,
    )
    return MinimumDetectableSignEffectResult(
        target_power=target_power,
        probability_grid_denominator=probability_grid_denominator,
        minimum_positive_probability=minimum_probability,
        minimum_margin_over_chance=minimum_probability - Fraction(1, 2),
        analysis=analysis,
    )
