"""Exact binary majority-vote oracle with authored latent regimes."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import comb

MAX_REGIMES = 256
MAX_MAJORITY_SAMPLE_COUNT = 255
MAX_REGIME_WEIGHT = 1_000_000
MAX_OUTCOME_WEIGHT = 1_000_000
_REGIME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _fraction_payload(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


@dataclass(frozen=True, slots=True)
class BinaryVoteRegime:
    """One authored latent regime for a binary correct/incorrect answer."""

    regime_id: str
    regime_weight: int
    success_weight: int
    failure_weight: int

    def __post_init__(self) -> None:
        if not isinstance(self.regime_id, str) or _REGIME_ID.fullmatch(
            self.regime_id
        ) is None:
            raise ValueError(
                "regime_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        if (
            type(self.regime_weight) is not int
            or self.regime_weight <= 0
            or self.regime_weight > MAX_REGIME_WEIGHT
        ):
            raise ValueError(
                f"regime_weight must be an integer in [1, {MAX_REGIME_WEIGHT}]"
            )
        for field_name, value in (
            ("success_weight", self.success_weight),
            ("failure_weight", self.failure_weight),
        ):
            if (
                type(value) is not int
                or value < 0
                or value > MAX_OUTCOME_WEIGHT
            ):
                raise ValueError(
                    f"{field_name} must be an integer in [0, {MAX_OUTCOME_WEIGHT}]"
                )
        if self.success_weight + self.failure_weight == 0:
            raise ValueError("success_weight and failure_weight cannot both be zero")

    @property
    def conditional_success_probability(self) -> Fraction:
        """Return exact candidate success probability conditional on this regime."""

        return Fraction(
            self.success_weight,
            self.success_weight + self.failure_weight,
        )


@dataclass(frozen=True, slots=True)
class RegimeMajorityContribution:
    """Exact conditional majority result for one latent regime."""

    regime: BinaryVoteRegime
    regime_probability: Fraction
    conditional_majority_success_probability: Fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "regime_id": self.regime.regime_id,
            "regime_weight": self.regime.regime_weight,
            "regime_probability": _fraction_payload(self.regime_probability),
            "success_weight": self.regime.success_weight,
            "failure_weight": self.regime.failure_weight,
            "conditional_success_probability": _fraction_payload(
                self.regime.conditional_success_probability
            ),
            "conditional_majority_success_probability": _fraction_payload(
                self.conditional_majority_success_probability
            ),
        }


@dataclass(frozen=True, slots=True)
class BinaryMajorityAnalysis:
    """Exact binary majority accuracy and induced pairwise dependence."""

    sample_count: int
    majority_threshold: int
    logical_binary_vote_sequences: int
    contributions: tuple[RegimeMajorityContribution, ...]
    single_sample_success_probability: Fraction
    majority_success_probability: Fraction
    majority_gain: Fraction
    pairwise_success_covariance: Fraction
    pairwise_success_correlation: Fraction | None

    def to_dict(self) -> dict[str, object]:
        correlation = self.pairwise_success_correlation
        return {
            "sample_count": self.sample_count,
            "logical_candidate_samples": self.sample_count,
            "majority_threshold": self.majority_threshold,
            "logical_binary_vote_sequences": self.logical_binary_vote_sequences,
            "latent_regime_draws_per_question": 1,
            "single_sample_success_probability": _fraction_payload(
                self.single_sample_success_probability
            ),
            "majority_success_probability": _fraction_payload(
                self.majority_success_probability
            ),
            "majority_gain": _fraction_payload(self.majority_gain),
            "pairwise_success_covariance": _fraction_payload(
                self.pairwise_success_covariance
            ),
            "pairwise_success_correlation": (
                None if correlation is None else _fraction_payload(correlation)
            ),
            "regimes": [contribution.to_dict() for contribution in self.contributions],
        }


def _binomial_majority_probability(
    success_probability: Fraction,
    *,
    sample_count: int,
    majority_threshold: int,
) -> Fraction:
    failure_probability = Fraction(1, 1) - success_probability
    return sum(
        (
            comb(sample_count, successes)
            * success_probability**successes
            * failure_probability ** (sample_count - successes)
            for successes in range(majority_threshold, sample_count + 1)
        ),
        start=Fraction(0, 1),
    )


def analyze_latent_regime_binary_majority(
    regimes: Iterable[BinaryVoteRegime],
    *,
    sample_count: int,
) -> BinaryMajorityAnalysis:
    """Analyze odd-N binary majority vote under a shared latent regime.

    One regime is sampled per question. Candidate correctness indicators are
    conditionally i.i.d. Bernoulli draws within that regime, but are generally
    correlated after the shared regime is marginalized out. The oracle models
    exactly two canonical answer labels and does not enumerate the ``2**N``
    logical vote sequences.
    """

    if (
        type(sample_count) is not int
        or sample_count <= 0
        or sample_count > MAX_MAJORITY_SAMPLE_COUNT
        or sample_count % 2 == 0
    ):
        raise ValueError(
            "sample_count must be an odd integer in "
            f"[1, {MAX_MAJORITY_SAMPLE_COUNT}]"
        )
    regime_tuple = tuple(regimes)
    if not regime_tuple:
        raise ValueError("at least one regime is required")
    if len(regime_tuple) > MAX_REGIMES:
        raise ValueError(f"regime count cannot exceed {MAX_REGIMES}")
    if any(not isinstance(regime, BinaryVoteRegime) for regime in regime_tuple):
        raise ValueError("all regimes must be BinaryVoteRegime instances")
    regime_ids = [regime.regime_id for regime in regime_tuple]
    if len(set(regime_ids)) != len(regime_ids):
        raise ValueError("regime_id values must be unique")

    total_regime_weight = sum(regime.regime_weight for regime in regime_tuple)
    majority_threshold = sample_count // 2 + 1
    contributions = tuple(
        RegimeMajorityContribution(
            regime=regime,
            regime_probability=Fraction(regime.regime_weight, total_regime_weight),
            conditional_majority_success_probability=(
                _binomial_majority_probability(
                    regime.conditional_success_probability,
                    sample_count=sample_count,
                    majority_threshold=majority_threshold,
                )
            ),
        )
        for regime in regime_tuple
    )
    single_sample_success = sum(
        (
            contribution.regime_probability
            * contribution.regime.conditional_success_probability
            for contribution in contributions
        ),
        start=Fraction(0, 1),
    )
    majority_success = sum(
        (
            contribution.regime_probability
            * contribution.conditional_majority_success_probability
            for contribution in contributions
        ),
        start=Fraction(0, 1),
    )
    pairwise_joint_success = sum(
        (
            contribution.regime_probability
            * contribution.regime.conditional_success_probability**2
            for contribution in contributions
        ),
        start=Fraction(0, 1),
    )
    pairwise_covariance = pairwise_joint_success - single_sample_success**2
    if pairwise_covariance < 0:
        raise AssertionError("shared-regime covariance cannot be negative")
    bernoulli_variance = single_sample_success * (
        Fraction(1, 1) - single_sample_success
    )
    pairwise_correlation = (
        None
        if bernoulli_variance == 0
        else pairwise_covariance / bernoulli_variance
    )
    return BinaryMajorityAnalysis(
        sample_count=sample_count,
        majority_threshold=majority_threshold,
        logical_binary_vote_sequences=2**sample_count,
        contributions=contributions,
        single_sample_success_probability=single_sample_success,
        majority_success_probability=majority_success,
        majority_gain=majority_success - single_sample_success,
        pairwise_success_covariance=pairwise_covariance,
        pairwise_success_correlation=pairwise_correlation,
    )
