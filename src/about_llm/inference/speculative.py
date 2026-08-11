"""Probability-level oracle for exact speculative rejection sampling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _probability_vector(values: ArrayLike, label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain numeric probabilities, not booleans")
    probabilities = np.asarray(raw, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError(f"{label} must contain finite non-negative probabilities")
    total = float(np.sum(probabilities))
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{label} must sum to one")
    return probabilities / total


def _uniform(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number in [0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result < 1:
        raise ValueError(f"{label} must be a real number in [0, 1)")
    return result


def _sample(probabilities: NDArray[np.float64], uniform: float) -> int:
    index = int(np.searchsorted(np.cumsum(probabilities), uniform, side="right"))
    return min(index, probabilities.size - 1)


@dataclass(frozen=True)
class SpeculativeDistributionAudit:
    draft_probabilities: tuple[float, ...]
    target_probabilities: tuple[float, ...]
    theoretical_output_probabilities: tuple[float, ...]
    acceptance_probability: float
    rejection_probability: float
    total_variation_distance: float
    maximum_target_difference: float


@dataclass(frozen=True)
class SpeculativeStepResult:
    proposed_token: int
    output_token: int
    accepted: bool
    acceptance_probability: float
    correction_probabilities: tuple[float, ...] | None


@dataclass(frozen=True)
class SpeculativeBlockResult:
    emitted_tokens: tuple[int, ...]
    accepted_draft_tokens: int
    first_rejection_index: int | None
    used_bonus_target_token: bool
    acceptance_probabilities: tuple[float, ...]


def audit_speculative_distribution(
    draft_probabilities: ArrayLike,
    target_probabilities: ArrayLike,
) -> SpeculativeDistributionAudit:
    """Derive the exact one-step output marginal without Monte Carlo."""

    draft = _probability_vector(draft_probabilities, "draft_probabilities")
    target = _probability_vector(target_probabilities, "target_probabilities")
    if draft.shape != target.shape:
        raise ValueError("draft and target probability vectors must have equal shape")

    positive_residual = np.maximum(target - draft, 0)
    rejection = float(np.sum(positive_residual))
    acceptance = float(1 - rejection)
    accepted_mass = np.minimum(draft, target)
    if rejection > 0:
        output = accepted_mass + rejection * (positive_residual / rejection)
    else:
        output = accepted_mass
    total_variation = float(0.5 * np.sum(np.abs(target - draft)))
    return SpeculativeDistributionAudit(
        draft_probabilities=tuple(float(value) for value in draft),
        target_probabilities=tuple(float(value) for value in target),
        theoretical_output_probabilities=tuple(float(value) for value in output),
        acceptance_probability=acceptance,
        rejection_probability=rejection,
        total_variation_distance=total_variation,
        maximum_target_difference=float(np.max(np.abs(output - target))),
    )


def speculative_sample_step(
    draft_probabilities: ArrayLike,
    target_probabilities: ArrayLike,
    *,
    draft_uniform: float,
    acceptance_uniform: float,
    correction_uniform: float,
) -> SpeculativeStepResult:
    """Sample one exact speculative step from supplied deterministic uniforms."""

    draft = _probability_vector(draft_probabilities, "draft_probabilities")
    target = _probability_vector(target_probabilities, "target_probabilities")
    if draft.shape != target.shape:
        raise ValueError("draft and target probability vectors must have equal shape")
    draft_draw = _uniform(draft_uniform, "draft_uniform")
    acceptance_draw = _uniform(acceptance_uniform, "acceptance_uniform")
    correction_draw = _uniform(correction_uniform, "correction_uniform")
    proposal = _sample(draft, draft_draw)
    acceptance = min(1.0, float(target[proposal] / draft[proposal]))
    if acceptance_draw < acceptance:
        return SpeculativeStepResult(
            proposed_token=proposal,
            output_token=proposal,
            accepted=True,
            acceptance_probability=acceptance,
            correction_probabilities=None,
        )

    residual = np.maximum(target - draft, 0)
    residual_total = float(np.sum(residual))
    if residual_total <= 0:
        raise RuntimeError("rejection requires a non-empty positive residual")
    correction = residual / residual_total
    return SpeculativeStepResult(
        proposed_token=proposal,
        output_token=_sample(correction, correction_draw),
        accepted=False,
        acceptance_probability=acceptance,
        correction_probabilities=tuple(float(value) for value in correction),
    )


def verify_speculative_block(
    draft_tokens: Sequence[int],
    draft_probabilities: Sequence[ArrayLike],
    target_probabilities: Sequence[ArrayLike],
    *,
    acceptance_uniforms: Sequence[float],
    correction_uniforms: Sequence[float],
    bonus_target_probabilities: ArrayLike,
    bonus_uniform: float,
) -> SpeculativeBlockResult:
    """Verify a supplied draft block and stop at its first rejection.

    Each probability pair must be conditioned on the same prefix at that
    position. If every draft token is accepted, one bonus target token is
    emitted. This function is a control-flow oracle, not a model forward pass.
    """

    block_length = len(draft_tokens)
    if block_length == 0:
        raise ValueError("draft block cannot be empty")
    if not (
        len(draft_probabilities)
        == len(target_probabilities)
        == len(acceptance_uniforms)
        == len(correction_uniforms)
        == block_length
    ):
        raise ValueError("draft block inputs must have the same length")
    bonus = _probability_vector(
        bonus_target_probabilities, "bonus_target_probabilities"
    )
    bonus_draw = _uniform(bonus_uniform, "bonus_uniform")
    validated_draft: list[NDArray[np.float64]] = []
    validated_target: list[NDArray[np.float64]] = []
    validated_tokens: list[int] = []
    validated_acceptance: list[float] = []
    validated_correction: list[float] = []
    for index, token in enumerate(draft_tokens):
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ValueError("draft tokens must be integer vocabulary indices")
        token_index = int(token)
        draft = _probability_vector(
            draft_probabilities[index], f"draft_probabilities[{index}]"
        )
        target = _probability_vector(
            target_probabilities[index], f"target_probabilities[{index}]"
        )
        if draft.shape != target.shape or draft.shape != bonus.shape:
            raise ValueError("all block probability vectors must share a vocabulary")
        if not 0 <= token_index < draft.size or draft[token_index] <= 0:
            raise ValueError("each supplied draft token must have positive draft mass")
        validated_draft.append(draft)
        validated_target.append(target)
        validated_tokens.append(token_index)
        validated_acceptance.append(
            _uniform(acceptance_uniforms[index], f"acceptance_uniforms[{index}]")
        )
        validated_correction.append(
            _uniform(correction_uniforms[index], f"correction_uniforms[{index}]")
        )

    emitted: list[int] = []
    acceptance_probabilities: list[float] = []

    for index, (token, draft, target, acceptance_draw, correction_draw) in enumerate(
        zip(
            validated_tokens,
            validated_draft,
            validated_target,
            validated_acceptance,
            validated_correction,
            strict=True,
        )
    ):
        acceptance = min(1.0, float(target[token] / draft[token]))
        acceptance_probabilities.append(acceptance)
        if acceptance_draw < acceptance:
            emitted.append(token)
            continue

        residual = np.maximum(target - draft, 0)
        residual_total = float(np.sum(residual))
        if residual_total <= 0:
            raise RuntimeError("rejection requires a non-empty positive residual")
        emitted.append(_sample(residual / residual_total, correction_draw))
        return SpeculativeBlockResult(
            emitted_tokens=tuple(emitted),
            accepted_draft_tokens=index,
            first_rejection_index=index,
            used_bonus_target_token=False,
            acceptance_probabilities=tuple(acceptance_probabilities),
        )

    emitted.append(_sample(bonus, bonus_draw))
    return SpeculativeBlockResult(
        emitted_tokens=tuple(emitted),
        accepted_draft_tokens=block_length,
        first_rejection_index=None,
        used_bonus_target_token=True,
        acceptance_probabilities=tuple(acceptance_probabilities),
    )
