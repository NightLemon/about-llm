from __future__ import annotations

from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

from about_llm.inference import (
    BinaryVoteRegime,
    analyze_latent_regime_binary_majority,
)

ROOT = Path(__file__).resolve().parents[1]
TOY = (
    ROOT
    / "projects"
    / "inference-serving"
    / "self_consistency_correlation_toy.py"
)

INDEPENDENT = (BinaryVoteRegime("iid", 1, 3, 2),)
CORRELATED = (
    BinaryVoteRegime("easy", 1, 9, 1),
    BinaryVoteRegime("hard", 1, 3, 7),
)


@pytest.mark.parametrize(
    ("sample_count", "expected"),
    [
        (1, Fraction(3, 5)),
        (3, Fraction(81, 125)),
        (5, Fraction(2133, 3125)),
        (11, Fraction(36_791_901, 48_828_125)),
    ],
)
def test_independent_majority_exactly_improves(
    sample_count: int,
    expected: Fraction,
) -> None:
    analysis = analyze_latent_regime_binary_majority(
        INDEPENDENT,
        sample_count=sample_count,
    )

    assert analysis.single_sample_success_probability == Fraction(3, 5)
    assert analysis.majority_success_probability == expected
    assert analysis.majority_gain == expected - Fraction(3, 5)
    assert analysis.pairwise_success_covariance == 0
    assert analysis.pairwise_success_correlation == 0
    assert analysis.majority_threshold == sample_count // 2 + 1
    assert analysis.logical_binary_vote_sequences == 2**sample_count


@pytest.mark.parametrize(
    ("sample_count", "expected"),
    [
        (1, Fraction(3, 5)),
        (3, Fraction(297, 500)),
        (5, Fraction(28_863, 50_000)),
        (11, Fraction(13_474_113_561, 25_000_000_000)),
    ],
)
def test_shared_latent_regime_can_make_majority_worse(
    sample_count: int,
    expected: Fraction,
) -> None:
    analysis = analyze_latent_regime_binary_majority(
        CORRELATED,
        sample_count=sample_count,
    )

    assert analysis.single_sample_success_probability == Fraction(3, 5)
    assert analysis.majority_success_probability == expected
    assert analysis.majority_gain == expected - Fraction(3, 5)
    assert analysis.pairwise_success_covariance == Fraction(9, 100)
    assert analysis.pairwise_success_correlation == Fraction(3, 8)


def test_same_marginal_success_does_not_identify_majority_accuracy() -> None:
    independent = analyze_latent_regime_binary_majority(INDEPENDENT, sample_count=11)
    correlated = analyze_latent_regime_binary_majority(CORRELATED, sample_count=11)

    assert (
        independent.single_sample_success_probability
        == correlated.single_sample_success_probability
        == Fraction(3, 5)
    )
    assert independent.majority_success_probability > Fraction(3, 5)
    assert correlated.majority_success_probability < Fraction(3, 5)


def test_closed_form_matches_explicit_binary_sequence_enumeration() -> None:
    analysis = analyze_latent_regime_binary_majority(CORRELATED, sample_count=3)
    enumerated = Fraction(0, 1)
    total_regime_weight = sum(regime.regime_weight for regime in CORRELATED)

    for regime in CORRELATED:
        regime_probability = Fraction(regime.regime_weight, total_regime_weight)
        success_probability = regime.conditional_success_probability
        for sequence in product((False, True), repeat=3):
            successes = sum(sequence)
            if successes < 2:
                continue
            enumerated += (
                regime_probability
                * success_probability**successes
                * (1 - success_probability) ** (3 - successes)
            )

    assert analysis.majority_success_probability == enumerated == Fraction(297, 500)


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        (BinaryVoteRegime("always_wrong", 1, 0, 1), Fraction(0, 1)),
        (BinaryVoteRegime("always_correct", 1, 1, 0), Fraction(1, 1)),
    ],
)
def test_deterministic_boundary_has_undefined_correlation(
    regime: BinaryVoteRegime,
    expected: Fraction,
) -> None:
    analysis = analyze_latent_regime_binary_majority((regime,), sample_count=5)
    payload = analysis.to_dict()

    assert analysis.single_sample_success_probability == expected
    assert analysis.majority_success_probability == expected
    assert analysis.pairwise_success_covariance == 0
    assert analysis.pairwise_success_correlation is None
    assert payload["pairwise_success_correlation"] is None


@pytest.mark.parametrize("sample_count", [0, -1, 2, 256, 257, True])
def test_analysis_rejects_nonpositive_even_unbounded_or_bool_sample_count(
    sample_count: int,
) -> None:
    with pytest.raises(ValueError, match="odd integer"):
        analyze_latent_regime_binary_majority(
            INDEPENDENT,
            sample_count=sample_count,
        )


def test_analysis_rejects_empty_duplicate_and_wrong_regime_types() -> None:
    with pytest.raises(ValueError, match="at least one"):
        analyze_latent_regime_binary_majority((), sample_count=1)
    with pytest.raises(ValueError, match="unique"):
        analyze_latent_regime_binary_majority(
            (
                BinaryVoteRegime("same", 1, 1, 1),
                BinaryVoteRegime("same", 2, 2, 1),
            ),
            sample_count=1,
        )
    with pytest.raises(ValueError, match="BinaryVoteRegime"):
        analyze_latent_regime_binary_majority(
            ("not-a-regime",),  # type: ignore[arg-type]
            sample_count=1,
        )


def test_analysis_rejects_regime_count_above_resource_cap() -> None:
    regimes = tuple(
        BinaryVoteRegime(f"r{index:03d}", 1, 1, 1) for index in range(257)
    )

    with pytest.raises(ValueError, match="cannot exceed 256"):
        analyze_latent_regime_binary_majority(regimes, sample_count=1)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("", 1, 1, 1), "regime_id"),
        (("bad id", 1, 1, 1), "regime_id"),
        (("x", 0, 1, 1), "regime_weight"),
        (("x", True, 1, 1), "regime_weight"),
        (("x", 1_000_001, 1, 1), "regime_weight"),
        (("x", 1, -1, 1), "success_weight"),
        (("x", 1, True, 1), "success_weight"),
        (("x", 1, 1_000_001, 1), "success_weight"),
        (("x", 1, 1, -1), "failure_weight"),
        (("x", 1, 1, True), "failure_weight"),
        (("x", 1, 1, 1_000_001), "failure_weight"),
        (("x", 1, 0, 0), "cannot both be zero"),
    ],
)
def test_regime_rejects_noncanonical_or_unbounded_fields(
    args: tuple[object, object, object, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BinaryVoteRegime(*args)  # type: ignore[arg-type]


