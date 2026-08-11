"""Deterministic table-driven beam-search reference implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

BeamFinishReason = Literal["eos", "length"]


@dataclass(frozen=True)
class BeamPrefix:
    token_ids: tuple[int, ...]
    cumulative_log_probability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "cumulative_log_probability": self.cumulative_log_probability,
        }


@dataclass(frozen=True)
class BeamSequence:
    token_ids: tuple[int, ...]
    cumulative_log_probability: float
    generated_length: int
    normalized_score: float
    finish_reason: BeamFinishReason

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "cumulative_log_probability": self.cumulative_log_probability,
            "generated_length": self.generated_length,
            "normalized_score": self.normalized_score,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class BeamSearchStep:
    step_index: int
    active_before: tuple[BeamPrefix, ...]
    positive_expansions_considered: int
    eos_finished: tuple[BeamSequence, ...]
    active_after: tuple[BeamPrefix, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "active_before": [prefix.to_dict() for prefix in self.active_before],
            "positive_expansions_considered": self.positive_expansions_considered,
            "eos_finished": [sequence.to_dict() for sequence in self.eos_finished],
            "active_after": [prefix.to_dict() for prefix in self.active_after],
        }


@dataclass(frozen=True)
class BeamSearchResult:
    vocabulary_size: int
    eos_token_id: int
    beam_width: int
    max_new_tokens: int
    length_penalty: float
    num_return_sequences: int
    length_definition: str
    steps: tuple[BeamSearchStep, ...]
    ranked_candidates: tuple[BeamSequence, ...]
    returned_sequences: tuple[BeamSequence, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "eos_token_id": self.eos_token_id,
            "beam_width": self.beam_width,
            "max_new_tokens": self.max_new_tokens,
            "length_penalty": self.length_penalty,
            "num_return_sequences": self.num_return_sequences,
            "length_definition": self.length_definition,
            "steps": [step.to_dict() for step in self.steps],
            "ranked_candidates": [
                sequence.to_dict() for sequence in self.ranked_candidates
            ],
            "returned_sequences": [
                sequence.to_dict() for sequence in self.returned_sequences
            ],
        }


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalized_score(
    cumulative_log_probability: float,
    generated_length: int,
    length_penalty: float,
) -> float:
    denominator = float(generated_length**length_penalty)
    return cumulative_log_probability / denominator


def _sequence(
    prefix: BeamPrefix,
    finish_reason: BeamFinishReason,
    length_penalty: float,
) -> BeamSequence:
    generated_length = len(prefix.token_ids)
    return BeamSequence(
        token_ids=prefix.token_ids,
        cumulative_log_probability=prefix.cumulative_log_probability,
        generated_length=generated_length,
        normalized_score=_normalized_score(
            prefix.cumulative_log_probability,
            generated_length,
            length_penalty,
        ),
        finish_reason=finish_reason,
    )


def _prefix_sort_key(prefix: BeamPrefix) -> tuple[float, tuple[int, ...]]:
    return (-prefix.cumulative_log_probability, prefix.token_ids)


def _sequence_sort_key(
    sequence: BeamSequence,
) -> tuple[float, float, tuple[int, ...], int]:
    finish_order = 0 if sequence.finish_reason == "eos" else 1
    return (
        -sequence.normalized_score,
        -sequence.cumulative_log_probability,
        sequence.token_ids,
        finish_order,
    )


def _validated_probability_table(
    probability_table: Mapping[tuple[int, ...], ArrayLike],
    *,
    vocabulary_size: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> dict[tuple[int, ...], np.ndarray]:
    if not isinstance(probability_table, Mapping) or not probability_table:
        raise ValueError("probability_table must be a non-empty mapping")
    validated: dict[tuple[int, ...], np.ndarray] = {}
    for prefix, probabilities in probability_table.items():
        if not isinstance(prefix, tuple) or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocabulary_size
            for token_id in prefix
        ):
            raise ValueError("table prefixes must be tuples of valid integer token ids")
        if eos_token_id in prefix:
            raise ValueError("table prefixes must not contain EOS because EOS is terminal")
        if len(prefix) >= max_new_tokens:
            raise ValueError("table prefixes must be shorter than max_new_tokens")
        values = np.array(probabilities, dtype=np.float64, copy=True)
        if values.ndim != 1 or values.size != vocabulary_size:
            raise ValueError(
                "each probability vector must be one-dimensional with vocabulary_size entries"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("each probability vector must sum to one")
        values.setflags(write=False)
        validated[prefix] = values
    if () not in validated:
        raise ValueError("probability_table must contain the empty root prefix")
    return validated


def beam_search_from_probabilities(
    probability_table: Mapping[tuple[int, ...], ArrayLike],
    *,
    vocabulary_size: int,
    eos_token_id: int,
    beam_width: int,
    max_new_tokens: int,
    length_penalty: float = 0.0,
    num_return_sequences: int = 1,
) -> BeamSearchResult:
    """Run a deterministic synchronous beam search over an authored table.

    At each step every positive-probability token from every active prefix is
    considered. EOS expansions are finalized immediately and never expanded.
    Non-EOS expansions are ranked by cumulative log probability; ties use the
    lexicographically smaller token-id sequence. The best ``beam_width`` remain
    active. Search does not use heuristic early stopping. At
    ``max_new_tokens``, remaining active prefixes become ``length`` candidates.

    Final candidates use ``log_probability / generated_length**length_penalty``.
    Generated length excludes the prompt and includes an emitted EOS token.
    This explicit convention is not claimed to match Transformers, vLLM, or any
    provider's length penalty, EOS handling, candidate cap, or early stopping.
    """

    vocabulary_size = _validate_positive_int(vocabulary_size, "vocabulary_size")
    beam_width = _validate_positive_int(beam_width, "beam_width")
    max_new_tokens = _validate_positive_int(max_new_tokens, "max_new_tokens")
    num_return_sequences = _validate_positive_int(
        num_return_sequences, "num_return_sequences"
    )
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or not 0 <= eos_token_id < vocabulary_size
    ):
        raise ValueError("eos_token_id must be a valid token id")
    if num_return_sequences > beam_width:
        raise ValueError("num_return_sequences must not exceed beam_width")
    if (
        isinstance(length_penalty, bool)
        or not isinstance(length_penalty, (int, float))
        or not math.isfinite(length_penalty)
        or length_penalty < 0
    ):
        raise ValueError("length_penalty must be a finite non-negative number")

    table = _validated_probability_table(
        probability_table,
        vocabulary_size=vocabulary_size,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    active: tuple[BeamPrefix, ...] = (BeamPrefix((), 0.0),)
    finished: list[BeamSequence] = []
    steps: list[BeamSearchStep] = []

    for step_index in range(1, max_new_tokens + 1):
        active_before = active
        next_prefixes: list[BeamPrefix] = []
        step_finished: list[BeamSequence] = []
        expansion_count = 0
        for prefix in active_before:
            probabilities = table.get(prefix.token_ids)
            if probabilities is None:
                raise ValueError(
                    f"probability_table is missing active prefix {prefix.token_ids}"
                )
            for token_id, probability in enumerate(probabilities):
                if probability == 0:
                    continue
                expansion_count += 1
                expanded = BeamPrefix(
                    token_ids=(*prefix.token_ids, token_id),
                    cumulative_log_probability=(
                        prefix.cumulative_log_probability + math.log(float(probability))
                    ),
                )
                if token_id == eos_token_id:
                    sequence = _sequence(expanded, "eos", float(length_penalty))
                    step_finished.append(sequence)
                    finished.append(sequence)
                else:
                    next_prefixes.append(expanded)

        next_prefixes.sort(key=_prefix_sort_key)
        active = tuple(next_prefixes[:beam_width])
        step_finished.sort(key=_sequence_sort_key)
        steps.append(
            BeamSearchStep(
                step_index=step_index,
                active_before=active_before,
                positive_expansions_considered=expansion_count,
                eos_finished=tuple(step_finished),
                active_after=active,
            )
        )
        if not active:
            break

    if active:
        finished.extend(
            _sequence(prefix, "length", float(length_penalty)) for prefix in active
        )
    if not finished:
        raise RuntimeError("beam search produced no terminal or length candidate")
    finished.sort(key=_sequence_sort_key)
    ranked = tuple(finished)
    returned = ranked[:num_return_sequences]
    return BeamSearchResult(
        vocabulary_size=vocabulary_size,
        eos_token_id=eos_token_id,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        length_penalty=float(length_penalty),
        num_return_sequences=num_return_sequences,
        length_definition="generated tokens only; emitted EOS included; prompt excluded",
        steps=tuple(steps),
        ranked_candidates=ranked,
        returned_sequences=returned,
    )
