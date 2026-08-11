"""Deterministic NumPy oracle for one documented next-token sampling policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _finite_positive_number(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _logits_vector(logits: NDArray[np.floating] | Sequence[float]) -> FloatArray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("logits must be a non-empty rank-1 vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must contain only finite values")
    return np.array(values, dtype=np.float64, copy=True)


def _stable_softmax(logits: FloatArray, mask: NDArray[np.bool_]) -> FloatArray:
    if logits.shape != mask.shape or not np.any(mask):
        raise RuntimeError("sampling support must contain at least one token")
    probabilities = np.zeros_like(logits, dtype=np.float64)
    maximum = float(np.max(logits[mask]))
    exponentials = np.exp(logits[mask] - maximum)
    denominator = float(np.sum(exponentials))
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("softmax normalization is invalid")
    probabilities[mask] = exponentials / denominator
    return probabilities


def _rank_token_ids(scores: FloatArray, candidates: NDArray[np.bool_]) -> tuple[int, ...]:
    token_ids = np.flatnonzero(candidates)
    order = np.lexsort((token_ids, -scores[token_ids]))
    return tuple(int(token_id) for token_id in token_ids[order])


@dataclass(frozen=True)
class SamplingConfig:
    """One explicit processor order: repetition -> temperature -> top-k -> top-p."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        _finite_positive_number(self.temperature, "temperature")
        _finite_positive_number(self.repetition_penalty, "repetition_penalty")
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None")
        if self.top_p is not None and (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be a finite number in (0, 1] or None")

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass(frozen=True)
class NextTokenSamplingStep:
    """Complete ledger for a single categorical draw from finite logits."""

    config: SamplingConfig
    prior_token_ids: tuple[int, ...]
    input_logits: FloatArray
    repetition_adjusted_logits: FloatArray
    temperature_scaled_logits: FloatArray
    filtered_logits: FloatArray
    ranked_token_ids: tuple[int, ...]
    top_k_token_ids: tuple[int, ...]
    top_p_token_ids: tuple[int, ...]
    support_token_ids: tuple[int, ...]
    probabilities: FloatArray
    uniform: float
    sampled_token_id: int

    @property
    def sampled_probability(self) -> float:
        return float(self.probabilities[self.sampled_token_id])

    @property
    def entropy_nats(self) -> float:
        positive = self.probabilities[self.probabilities > 0]
        return float(-np.sum(positive * np.log(positive)))

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration": self.config.to_dict(),
            "processor_order": (
                "repetition_penalty",
                "temperature",
                "top_k",
                "top_p",
                "renormalize",
                "categorical_inverse_cdf_by_token_id",
            ),
            "prior_token_ids": self.prior_token_ids,
            "input_logits": self.input_logits.tolist(),
            "repetition_adjusted_logits": self.repetition_adjusted_logits.tolist(),
            "temperature_scaled_logits": self.temperature_scaled_logits.tolist(),
            "filtered_logits": [
                float(value) if math.isfinite(float(value)) else None
                for value in self.filtered_logits
            ],
            "ranked_token_ids": self.ranked_token_ids,
            "top_k_token_ids": self.top_k_token_ids,
            "top_p_token_ids": self.top_p_token_ids,
            "support_token_ids": self.support_token_ids,
            "probabilities": self.probabilities.tolist(),
            "uniform": self.uniform,
            "sampled_token_id": self.sampled_token_id,
            "sampled_probability": self.sampled_probability,
            "entropy_nats": self.entropy_nats,
        }


def greedy_next_token(logits: NDArray[np.floating] | Sequence[float]) -> int:
    """Select the lowest token id among exact maximum-logit ties."""

    values = _logits_vector(logits)
    return int(np.argmax(values))


def sample_next_token(
    logits: NDArray[np.floating] | Sequence[float],
    *,
    config: SamplingConfig,
    prior_token_ids: Sequence[int] = (),
    uniform: float,
) -> NextTokenSamplingStep:
    """Execute one transparent next-token sampling step.

    Repetition penalty follows the sign-aware multiplicative convention: for
    each unique prior token, a negative logit is multiplied by the penalty and
    a non-negative logit is divided by it. Temperature scaling follows. This
    oracle then retains exactly ``top_k`` tokens, breaking score ties by token
    id, and applies top-p to probabilities renormalized over that top-k support.
    Top-p retains the first token that reaches or crosses the threshold. The
    final categorical inverse CDF is traversed in ascending token-id order.

    These choices are a documented reference policy, not universal runtime
    defaults and not a multi-token generation or model-quality experiment.
    """

    if not isinstance(config, SamplingConfig):
        raise TypeError("config must be a SamplingConfig")
    values = _logits_vector(logits)
    vocabulary_size = values.size
    if config.top_k is not None and config.top_k > vocabulary_size:
        raise ValueError("top_k cannot exceed the logits vocabulary size")
    if isinstance(prior_token_ids, (str, bytes)) or not isinstance(
        prior_token_ids, Sequence
    ):
        raise TypeError("prior_token_ids must be a sequence of token ids")
    prior = tuple(prior_token_ids)
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < vocabulary_size
        for token_id in prior
    ):
        raise ValueError("prior_token_ids must be integer ids in the vocabulary")
    if (
        isinstance(uniform, bool)
        or not isinstance(uniform, (int, float))
        or not math.isfinite(uniform)
        or not 0 <= uniform < 1
    ):
        raise ValueError("uniform must be a finite number in [0, 1)")

    repetition_adjusted = np.array(values, copy=True)
    for token_id in sorted(set(prior)):
        if repetition_adjusted[token_id] < 0:
            repetition_adjusted[token_id] *= config.repetition_penalty
        else:
            repetition_adjusted[token_id] /= config.repetition_penalty

    temperature_scaled = repetition_adjusted / config.temperature
    full_support = np.ones(vocabulary_size, dtype=np.bool_)
    ranked = _rank_token_ids(temperature_scaled, full_support)

    top_k_count = config.top_k if config.top_k is not None else vocabulary_size
    top_k_token_ids = ranked[:top_k_count]
    top_k_mask = np.zeros(vocabulary_size, dtype=np.bool_)
    top_k_mask[list(top_k_token_ids)] = True
    top_k_probabilities = _stable_softmax(temperature_scaled, top_k_mask)

    if config.top_p is None or config.top_p == 1:
        top_p_token_ids = top_k_token_ids
    else:
        ranked_probabilities = np.array(
            [top_k_probabilities[token_id] for token_id in top_k_token_ids],
            dtype=np.float64,
        )
        cumulative = np.cumsum(ranked_probabilities)
        cutoff = int(np.searchsorted(cumulative, config.top_p, side="left"))
        cutoff = min(cutoff, len(top_k_token_ids) - 1)
        top_p_token_ids = top_k_token_ids[: cutoff + 1]

    final_mask = np.zeros(vocabulary_size, dtype=np.bool_)
    final_mask[list(top_p_token_ids)] = True
    probabilities = _stable_softmax(temperature_scaled, final_mask)
    cumulative_by_token_id = np.cumsum(probabilities)
    cumulative_by_token_id[-1] = 1.0
    sampled_token_id = int(
        np.searchsorted(cumulative_by_token_id, float(uniform), side="right")
    )
    filtered_logits = np.where(final_mask, temperature_scaled, -np.inf)
    support_token_ids = tuple(int(item) for item in np.flatnonzero(final_mask))

    return NextTokenSamplingStep(
        config=config,
        prior_token_ids=prior,
        input_logits=_readonly(values),
        repetition_adjusted_logits=_readonly(repetition_adjusted),
        temperature_scaled_logits=_readonly(temperature_scaled),
        filtered_logits=_readonly(filtered_logits),
        ranked_token_ids=ranked,
        top_k_token_ids=top_k_token_ids,
        top_p_token_ids=top_p_token_ids,
        support_token_ids=support_token_ids,
        probabilities=_readonly(probabilities),
        uniform=float(uniform),
        sampled_token_id=sampled_token_id,
    )
