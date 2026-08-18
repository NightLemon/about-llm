from __future__ import annotations

from fractions import Fraction

import pytest

from about_llm.evaluation import (
    cohen_kappa,
    criterion_validity,
    minimum_detectable_sign_effect,
    minimum_sign_test_sample_size,
    one_sided_sign_test_power,
)

pytestmark = pytest.mark.formula


def test_cohen_kappa_reports_observed_chance_and_corrected_agreement() -> None:
    result = cohen_kappa(
        ["cat", "cat", "dog", "dog"],
        ["cat", "dog", "dog", "dog"],
    )

    assert result.item_count == 4
    assert result.categories == ("cat", "dog")
    assert result.observed_agreement == Fraction(3, 4)
    assert result.chance_agreement == Fraction(1, 2)
    assert result.kappa == Fraction(1, 2)


def test_degenerate_single_category_kappa_is_explicitly_undefined() -> None:
    result = cohen_kappa(["pass", "pass"], ["pass", "pass"])

    assert result.observed_agreement == 1
    assert result.chance_agreement == 1
    assert result.kappa is None
    assert result.to_dict()["kappa_defined"] is False


def test_perfect_reliability_can_coexist_with_zero_criterion_validity() -> None:
    criterion = ["correct", "correct", "incorrect", "incorrect"]
    rater_a = ["incorrect", "incorrect", "correct", "correct"]
    rater_b = list(rater_a)

    reliability = cohen_kappa(rater_a, rater_b)
    validity_a = criterion_validity(rater_a, criterion)
    validity_b = criterion_validity(rater_b, criterion)

    assert reliability.observed_agreement == 1
    assert reliability.kappa == 1
    assert validity_a.accuracy == 0
    assert validity_b.accuracy == 0


def test_criterion_confusion_uses_criterion_rows_and_observed_columns() -> None:
    result = criterion_validity(
        ["pass", "fail", "pass", "pass"],
        ["pass", "pass", "fail", "pass"],
    )

    assert result.categories == ("fail", "pass")
    assert result.accuracy == Fraction(1, 2)
    assert result.confusion_matrix == ((0, 1), (1, 2))
    assert result.to_dict()["confusion"] == {
        "row_semantics": "criterion_label",
        "column_semantics": "observed_label",
        "categories_in_row_and_column_order": ["fail", "pass"],
        "matrix": [[0, 1], [1, 2]],
    }


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([], []),
        (["pass"], []),
        ([""], ["pass"]),
        (["pass"], [1]),
    ],
)
def test_measurement_labels_fail_closed(
    first: list[object], second: list[object]
) -> None:
    with pytest.raises(ValueError):
        cohen_kappa(first, second)  # type: ignore[arg-type]


def test_exact_sign_power_exposes_discrete_rejection_threshold() -> None:
    result = one_sided_sign_test_power(
        informative_pair_count=5,
        alternative_positive_probability=Fraction(4, 5),
        alpha=Fraction(1, 20),
    )

    assert result.rejection_min_positive_count == 5
    assert result.realized_null_rejection_probability == Fraction(1, 32)
    assert result.power == Fraction(1024, 3125)


def test_small_sign_test_can_have_no_rejection_region() -> None:
    result = one_sided_sign_test_power(
        informative_pair_count=4,
        alternative_positive_probability=Fraction(9, 10),
        alpha=Fraction(1, 20),
    )

    assert result.rejection_min_positive_count is None
    assert result.realized_null_rejection_probability == 0
    assert result.power == 0


def test_minimum_sample_size_is_for_informative_pairs() -> None:
    result = minimum_sign_test_sample_size(
        alternative_positive_probability=Fraction(3, 4),
        target_power=Fraction(4, 5),
    )

    previous = one_sided_sign_test_power(
        informative_pair_count=result.analysis.informative_pair_count - 1,
        alternative_positive_probability=Fraction(3, 4),
    )
    assert result.analysis.power >= Fraction(4, 5)
    assert previous.power < Fraction(4, 5)


def test_minimum_detectable_effect_is_exact_on_declared_grid() -> None:
    result = minimum_detectable_sign_effect(
        informative_pair_count=5,
        target_power=Fraction(3, 10),
        probability_grid_denominator=10,
    )

    assert result.minimum_positive_probability == Fraction(4, 5)
    assert result.minimum_margin_over_chance == Fraction(3, 10)
    assert result.analysis.power == Fraction(1024, 3125)
    below = one_sided_sign_test_power(
        informative_pair_count=5,
        alternative_positive_probability=Fraction(7, 10),
    )
    assert below.power < Fraction(3, 10)


@pytest.mark.parametrize(
    "call",
    [
        lambda: one_sided_sign_test_power(
            informative_pair_count=0,
            alternative_positive_probability=Fraction(3, 4),
        ),
        lambda: one_sided_sign_test_power(
            informative_pair_count=20,
            alternative_positive_probability=Fraction(3, 4),
            alpha=Fraction(1, 1),
        ),
        lambda: minimum_sign_test_sample_size(
            alternative_positive_probability=Fraction(1, 2),
            target_power=Fraction(4, 5),
        ),
        lambda: minimum_detectable_sign_effect(
            informative_pair_count=4,
            target_power=Fraction(4, 5),
        ),
        lambda: minimum_detectable_sign_effect(
            informative_pair_count=20,
            target_power=Fraction(4, 5),
            probability_grid_denominator=1,
        ),
    ],
)
def test_sign_power_planning_rejects_invalid_or_unreachable_contracts(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]
