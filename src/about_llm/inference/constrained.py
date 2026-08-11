"""Token-aware finite-language constrained-decoding reference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ConstraintFinishReason = Literal["eos", "length"]


class ConstraintDeadEndError(RuntimeError):
    """Raised when the grammar has no positive-probability allowed token."""


@dataclass(frozen=True, init=False)
class LiteralSetConstraint:
    """A deterministic character trie accepting an explicit finite language."""

    literals: tuple[str, ...]
    start_state: int
    accepting_states: tuple[int, ...]
    transitions: tuple[tuple[int, str, int], ...]
    state_count: int

    @classmethod
    def from_literals(cls, literals: Sequence[str]) -> LiteralSetConstraint:
        if isinstance(literals, (str, bytes)) or not isinstance(literals, Sequence):
            raise ValueError("literals must be a non-string sequence of strings")
        values = tuple(literals)
        if not values:
            raise ValueError("literals must be non-empty")
        if any(not isinstance(literal, str) for literal in values):
            raise ValueError("every literal must be a string")
        if len(set(values)) != len(values):
            raise ValueError("literals must be unique")

        children: list[dict[str, int]] = [{}]
        accepting: set[int] = set()
        for literal in values:
            state = 0
            for character in literal:
                next_state = children[state].get(character)
                if next_state is None:
                    next_state = len(children)
                    children[state][character] = next_state
                    children.append({})
                state = next_state
            accepting.add(state)
        transitions = tuple(
            (state, character, next_state)
            for state, outgoing in enumerate(children)
            for character, next_state in sorted(outgoing.items())
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "literals", values)
        object.__setattr__(instance, "start_state", 0)
        object.__setattr__(instance, "accepting_states", tuple(sorted(accepting)))
        object.__setattr__(instance, "transitions", transitions)
        object.__setattr__(instance, "state_count", len(children))
        return instance

    def is_accepting(self, state: int) -> bool:
        self._validate_state(state)
        return state in self.accepting_states

    def advance(self, state: int, text: str) -> int | None:
        """Consume every Unicode code point in ``text`` or reject the token."""

        self._validate_state(state)
        if not isinstance(text, str) or not text:
            raise ValueError("token text must be a non-empty string")
        current = state
        for character in text:
            next_state = next(
                (
                    destination
                    for source, label, destination in self.transitions
                    if source == current and label == character
                ),
                None,
            )
            if next_state is None:
                return None
            current = next_state
        return current

    def _validate_state(self, state: int) -> None:
        if (
            isinstance(state, bool)
            or not isinstance(state, int)
            or not 0 <= state < self.state_count
        ):
            raise ValueError("state must be a valid constraint state")

    def to_dict(self) -> dict[str, object]:
        return {
            "literals": list(self.literals),
            "start_state": self.start_state,
            "accepting_states": list(self.accepting_states),
            "transitions": [list(transition) for transition in self.transitions],
            "state_count": self.state_count,
        }


@dataclass(frozen=True)
class ConstrainedDecodingStep:
    step_index: int
    prefix_token_ids: tuple[int, ...]
    state_before: int
    grammar_allowed_token_ids: tuple[int, ...]
    grammar_blocked_token_ids: tuple[int, ...]
    raw_allowed_probability_mass: float
    normalized_probabilities: tuple[float, ...]
    selected_token_id: int
    selected_token_text: str | None
    state_after: int
    selected_is_eos: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "prefix_token_ids": list(self.prefix_token_ids),
            "state_before": self.state_before,
            "grammar_allowed_token_ids": list(self.grammar_allowed_token_ids),
            "grammar_blocked_token_ids": list(self.grammar_blocked_token_ids),
            "raw_allowed_probability_mass": self.raw_allowed_probability_mass,
            "normalized_probabilities": list(self.normalized_probabilities),
            "selected_token_id": self.selected_token_id,
            "selected_token_text": self.selected_token_text,
            "state_after": self.state_after,
            "selected_is_eos": self.selected_is_eos,
        }


@dataclass(frozen=True)
class ConstrainedDecodingResult:
    token_ids: tuple[int, ...]
    decoded_text: str
    finish_reason: ConstraintFinishReason
    final_state: int
    constraint_accepting: bool
    eos_emitted: bool
    steps: tuple[ConstrainedDecodingStep, ...]
    text_unit: str

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "decoded_text": self.decoded_text,
            "finish_reason": self.finish_reason,
            "final_state": self.final_state,
            "constraint_accepting": self.constraint_accepting,
            "eos_emitted": self.eos_emitted,
            "steps": [step.to_dict() for step in self.steps],
            "text_unit": self.text_unit,
        }


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_token_texts(
    token_texts: Sequence[str | None], eos_token_id: int
) -> tuple[str | None, ...]:
    if isinstance(token_texts, (str, bytes)) or not isinstance(token_texts, Sequence):
        raise ValueError("token_texts must be a non-string sequence")
    values = tuple(token_texts)
    if not values:
        raise ValueError("token_texts must be non-empty")
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or not 0 <= eos_token_id < len(values)
    ):
        raise ValueError("eos_token_id must index token_texts")
    for token_id, token_text in enumerate(values):
        if token_id == eos_token_id:
            if token_text is not None:
                raise ValueError("the EOS token text must be None")
        elif not isinstance(token_text, str) or not token_text:
            raise ValueError("every non-EOS token must have non-empty string text")
    return values


def _validated_probability_table(
    probability_table: Mapping[tuple[int, ...], ArrayLike],
    *,
    vocabulary_size: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> dict[tuple[int, ...], NDArray[np.float64]]:
    if not isinstance(probability_table, Mapping) or not probability_table:
        raise ValueError("probability_table must be a non-empty mapping")
    validated: dict[tuple[int, ...], NDArray[np.float64]] = {}
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
                "each probability vector must be one-dimensional with vocabulary-size entries"
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


def constrained_greedy_from_probabilities(
    probability_table: Mapping[tuple[int, ...], ArrayLike],
    *,
    token_texts: Sequence[str | None],
    eos_token_id: int,
    constraint: LiteralSetConstraint,
    max_new_tokens: int,
) -> ConstrainedDecodingResult:
    """Greedily decode after masking tokens with a finite-language trie.

    Every non-EOS token is checked by consuming its complete supplied text,
    one Python Unicode code point at a time. EOS is allowed only in an
    accepting state. Allowed raw probability mass is renormalized, then the
    highest-probability token is selected; ties use the smaller token id.

    The supplied token fragments are assumed to concatenate to decoded text.
    This function does not implement tokenizer byte state, normalization,
    JSON Schema, a context-free grammar, a model forward pass, or a provider
    runtime contract.
    """

    max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
    if not isinstance(constraint, LiteralSetConstraint):
        raise ValueError("constraint must be a LiteralSetConstraint")
    texts = _validated_token_texts(token_texts, eos_token_id)
    table = _validated_probability_table(
        probability_table,
        vocabulary_size=len(texts),
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    state = constraint.start_state
    token_ids: list[int] = []
    decoded_fragments: list[str] = []
    steps: list[ConstrainedDecodingStep] = []

    for step_index in range(1, max_new_tokens + 1):
        prefix = tuple(token_ids)
        probabilities = table.get(prefix)
        if probabilities is None:
            raise ValueError(f"probability_table is missing active prefix {prefix}")

        next_states: dict[int, int] = {}
        allowed: list[int] = []
        for token_id, token_text in enumerate(texts):
            if token_id == eos_token_id:
                if constraint.is_accepting(state):
                    allowed.append(token_id)
                continue
            assert token_text is not None
            next_state = constraint.advance(state, token_text)
            if next_state is not None:
                allowed.append(token_id)
                next_states[token_id] = next_state

        allowed_mass = float(probabilities[allowed].sum())
        if allowed_mass <= 0:
            raise ConstraintDeadEndError(
                "constraint has no positive-probability allowed token at "
                f"prefix {prefix} and state {state}"
            )
        normalized = np.zeros(len(texts), dtype=np.float64)
        normalized[allowed] = probabilities[allowed] / allowed_mass
        selected = min(allowed, key=lambda token_id: (-normalized[token_id], token_id))
        selected_text = texts[selected]
        selected_is_eos = selected == eos_token_id
        state_after = state if selected_is_eos else next_states[selected]
        blocked = tuple(token_id for token_id in range(len(texts)) if token_id not in allowed)
        steps.append(
            ConstrainedDecodingStep(
                step_index=step_index,
                prefix_token_ids=prefix,
                state_before=state,
                grammar_allowed_token_ids=tuple(allowed),
                grammar_blocked_token_ids=blocked,
                raw_allowed_probability_mass=allowed_mass,
                normalized_probabilities=tuple(float(value) for value in normalized),
                selected_token_id=selected,
                selected_token_text=selected_text,
                state_after=state_after,
                selected_is_eos=selected_is_eos,
            )
        )
        token_ids.append(selected)
        state = state_after
        if selected_is_eos:
            return ConstrainedDecodingResult(
                token_ids=tuple(token_ids),
                decoded_text="".join(decoded_fragments),
                finish_reason="eos",
                final_state=state,
                constraint_accepting=True,
                eos_emitted=True,
                steps=tuple(steps),
                text_unit="Python Unicode code point over supplied token text",
            )
        assert selected_text is not None
        decoded_fragments.append(selected_text)

    return ConstrainedDecodingResult(
        token_ids=tuple(token_ids),
        decoded_text="".join(decoded_fragments),
        finish_reason="length",
        final_state=state,
        constraint_accepting=constraint.is_accepting(state),
        eos_emitted=False,
        steps=tuple(steps),
        text_unit="Python Unicode code point over supplied token text",
    )
