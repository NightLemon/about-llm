from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from about_llm.inference import (
    SamplingConfig,
    greedy_next_token,
    sample_next_token,
)

ROOT = Path(__file__).resolve().parents[1]


def _exact_fixture() -> object:
    return sample_next_token(
        np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)),
        config=SamplingConfig(temperature=1, top_k=3, top_p=0.7),
        uniform=0.6,
    )


def test_top_k_then_top_p_uses_post_top_k_normalization_and_crossing_token() -> None:
    step = _exact_fixture()

    assert step.ranked_token_ids == (0, 1, 2, 3)
    assert step.top_k_token_ids == (0, 1, 2)
    assert step.top_p_token_ids == (0, 1)
    assert step.support_token_ids == (0, 1)
    np.testing.assert_allclose(step.probabilities, [4 / 7, 3 / 7, 0, 0])
    assert step.sampled_token_id == 1
    assert step.sampled_probability == pytest.approx(3 / 7)


def test_top_p_keeps_first_token_that_reaches_or_crosses_threshold() -> None:
    exact = sample_next_token(
        np.log([0.6, 0.3, 0.1]),
        config=SamplingConfig(top_p=0.6),
        uniform=0,
    )
    crossed = sample_next_token(
        np.log([0.6, 0.3, 0.1]),
        config=SamplingConfig(top_p=0.61),
        uniform=0,
    )

    assert exact.top_p_token_ids == (0,)
    assert crossed.top_p_token_ids == (0, 1)


def test_exact_top_k_breaks_equal_score_ties_by_lowest_token_id() -> None:
    step = sample_next_token(
        [1, 1, 1, 0],
        config=SamplingConfig(top_k=2),
        uniform=0,
    )

    assert step.ranked_token_ids == (0, 1, 2, 3)
    assert step.top_k_token_ids == (0, 1)
    assert step.support_token_ids == (0, 1)
    np.testing.assert_allclose(step.probabilities, [0.5, 0.5, 0, 0])


def test_top_p_tie_at_cutoff_uses_the_same_token_id_tie_break() -> None:
    step = sample_next_token(
        [0, 0, 0, 0],
        config=SamplingConfig(top_p=0.5),
        uniform=0,
    )

    assert step.top_p_token_ids == (0, 1)
    np.testing.assert_allclose(step.probabilities, [0.5, 0.5, 0, 0])


def test_sign_aware_repetition_penalty_applies_once_per_unique_prior_token() -> None:
    step = sample_next_token(
        [2.0, -2.0, 0.5],
        config=SamplingConfig(repetition_penalty=2),
        prior_token_ids=(0, 1, 1),
        uniform=0,
    )

    np.testing.assert_array_equal(step.repetition_adjusted_logits, [1, -4, 0.5])
    assert step.prior_token_ids == (0, 1, 1)


def test_temperature_precedes_top_p_and_can_change_its_support() -> None:
    logits = np.log([0.6, 0.25, 0.15])
    cold = sample_next_token(
        logits,
        config=SamplingConfig(temperature=0.5, top_p=0.8),
        uniform=0,
    )
    hot = sample_next_token(
        logits,
        config=SamplingConfig(temperature=2, top_p=0.8),
        uniform=0,
    )

    assert cold.top_p_token_ids == (0,)
    assert hot.top_p_token_ids == (0, 1, 2)


def test_inverse_cdf_is_explicitly_traversed_in_token_id_order() -> None:
    low = sample_next_token(
        [0, 0], config=SamplingConfig(), uniform=0.499999
    )
    boundary = sample_next_token(
        [0, 0], config=SamplingConfig(), uniform=0.5
    )

    assert low.sampled_token_id == 0
    assert boundary.sampled_token_id == 1


def test_greedy_tie_breaks_by_lowest_token_id_without_temperature_division() -> None:
    assert greedy_next_token([-3, 7, 7, 2]) == 1


def test_large_logits_and_additive_translation_are_numerically_stable() -> None:
    config = SamplingConfig(top_p=0.9)
    baseline = sample_next_token([10000, 9999, 9998], config=config, uniform=0.7)
    translated = sample_next_token([3, 2, 1], config=config, uniform=0.7)

    np.testing.assert_allclose(baseline.probabilities, translated.probabilities)
    assert baseline.support_token_ids == translated.support_token_ids
    assert baseline.sampled_token_id == translated.sampled_token_id


def test_top_p_one_and_no_filters_preserve_full_softmax_support() -> None:
    no_filters = sample_next_token(
        [2, 1, 0], config=SamplingConfig(), uniform=0
    )
    top_p_one = sample_next_token(
        [2, 1, 0], config=SamplingConfig(top_p=1), uniform=0
    )

    np.testing.assert_allclose(no_filters.probabilities, top_p_one.probabilities)
    assert top_p_one.support_token_ids == (0, 1, 2)
    assert sum(top_p_one.probabilities) == pytest.approx(1)


def test_result_arrays_are_immutable_copies() -> None:
    logits = np.array([1.0, 0.0])
    step = sample_next_token(logits, config=SamplingConfig(), uniform=0)
    logits[0] = -10

    assert step.input_logits[0] == 1
    with pytest.raises(ValueError):
        step.probabilities[0] = 0


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (lambda: SamplingConfig(temperature=0), ValueError, "temperature"),
        (lambda: SamplingConfig(temperature=float("nan")), ValueError, "temperature"),
        (lambda: SamplingConfig(top_k=0), ValueError, "top_k"),
        (lambda: SamplingConfig(top_k=True), ValueError, "top_k"),
        (lambda: SamplingConfig(top_p=0), ValueError, "top_p"),
        (lambda: SamplingConfig(top_p=float("inf")), ValueError, "top_p"),
        (
            lambda: SamplingConfig(repetition_penalty=0),
            ValueError,
            "repetition_penalty",
        ),
        (
            lambda: sample_next_token(
                [1, 0], config=SamplingConfig(top_k=3), uniform=0
            ),
            ValueError,
            "vocabulary size",
        ),
        (
            lambda: sample_next_token(
                [[1, 0]], config=SamplingConfig(), uniform=0
            ),
            ValueError,
            "rank-1",
        ),
        (
            lambda: sample_next_token(
                [1, float("nan")], config=SamplingConfig(), uniform=0
            ),
            ValueError,
            "finite",
        ),
        (
            lambda: sample_next_token(
                [1, 0],
                config=SamplingConfig(),
                prior_token_ids=(2,),
                uniform=0,
            ),
            ValueError,
            "vocabulary",
        ),
        (
            lambda: sample_next_token(
                [1, 0], config=SamplingConfig(), uniform=1
            ),
            ValueError,
            "uniform",
        ),
    ],
)
def test_invalid_sampling_contracts_fail_closed(
    operation: Callable[[], object], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        operation()


def test_sampling_is_deterministic_for_fixed_logits_config_history_and_uniform() -> None:
    assert _exact_fixture().to_dict() == _exact_fixture().to_dict()


