from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from about_llm.inference import (
    ConstraintDeadEndError,
    LiteralSetConstraint,
    constrained_greedy_from_probabilities,
)

ROOT = Path(__file__).resolve().parents[1]


def _json_fixture() -> tuple[
    LiteralSetConstraint,
    tuple[str | None, ...],
    dict[tuple[int, ...], list[float]],
]:
    constraint = LiteralSetConstraint.from_literals(('{"x":1}', '{"x":2}'))
    token_texts = ('{"x"', ":", "1}", "1]", "2}", None, "garbage")
    probability_table = {
        (): [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        (0,): [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
        (0, 1): [0.0, 0.0, 0.25, 0.65, 0.10, 0.0, 0.0],
        (0, 1, 2): [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    return constraint, token_texts, probability_table


def test_complete_token_text_is_checked_before_mask_and_renormalization() -> None:
    constraint, token_texts, table = _json_fixture()
    result = constrained_greedy_from_probabilities(
        table,
        token_texts=token_texts,
        eos_token_id=5,
        constraint=constraint,
        max_new_tokens=4,
    )

    assert result.token_ids == (0, 1, 2, 5)
    assert result.decoded_text == '{"x":1}'
    assert result.finish_reason == "eos"
    assert result.constraint_accepting
    assert result.eos_emitted
    critical = result.steps[2]
    assert critical.grammar_allowed_token_ids == (2, 4)
    assert 3 in critical.grammar_blocked_token_ids
    assert token_texts[3] == "1]"
    assert critical.raw_allowed_probability_mass == pytest.approx(0.35)
    assert critical.normalized_probabilities[2] == pytest.approx(5 / 7)
    assert critical.normalized_probabilities[4] == pytest.approx(2 / 7)


def test_eos_is_blocked_until_the_constraint_state_is_accepting() -> None:
    result = constrained_greedy_from_probabilities(
        {(): [0.1, 0.9], (0,): [0.0, 1.0]},
        token_texts=("a", None),
        eos_token_id=1,
        constraint=LiteralSetConstraint.from_literals(("a",)),
        max_new_tokens=2,
    )

    assert result.token_ids == (0, 1)
    assert result.steps[0].grammar_allowed_token_ids == (0,)
    assert result.steps[1].grammar_allowed_token_ids == (1,)
    assert all(
        1 not in step.prefix_token_ids for step in result.steps
    )


def test_equal_allowed_probability_uses_smaller_token_id() -> None:
    result = constrained_greedy_from_probabilities(
        {(): [0.5, 0.5, 0.0]},
        token_texts=("a", "b", None),
        eos_token_id=2,
        constraint=LiteralSetConstraint.from_literals(("a", "b")),
        max_new_tokens=1,
    )

    assert result.token_ids == (0,)
    assert result.finish_reason == "length"
    assert result.constraint_accepting


def test_length_finish_separates_accepting_from_eos_completion() -> None:
    accepting = constrained_greedy_from_probabilities(
        {(): [1.0, 0.0]},
        token_texts=("a", None),
        eos_token_id=1,
        constraint=LiteralSetConstraint.from_literals(("a",)),
        max_new_tokens=1,
    )
    incomplete = constrained_greedy_from_probabilities(
        {(): [1.0, 0.0, 0.0]},
        token_texts=("a", "b", None),
        eos_token_id=2,
        constraint=LiteralSetConstraint.from_literals(("ab",)),
        max_new_tokens=1,
    )

    assert accepting.constraint_accepting
    assert not accepting.eos_emitted
    assert accepting.finish_reason == "length"
    assert not incomplete.constraint_accepting
    assert not incomplete.eos_emitted


def test_unicode_token_consumes_every_python_code_point() -> None:
    constraint = LiteralSetConstraint.from_literals(("甲🙂",))
    result = constrained_greedy_from_probabilities(
        {(): [0.9, 0.0, 0.1], (0,): [0.0, 1.0, 0.0]},
        token_texts=("甲🙂", None, "甲🙃"),
        eos_token_id=1,
        constraint=constraint,
        max_new_tokens=2,
    )

    assert result.decoded_text == "甲🙂"
    assert result.token_ids == (0, 1)
    assert constraint.advance(constraint.start_state, "甲🙃") is None
    assert result.text_unit == "Python Unicode code point over supplied token text"


def test_zero_allowed_probability_mass_fails_as_a_typed_dead_end() -> None:
    with pytest.raises(ConstraintDeadEndError, match="no positive-probability"):
        constrained_greedy_from_probabilities(
            {(): [0.0, 0.0, 1.0]},
            token_texts=("a", None, "b"),
            eos_token_id=1,
            constraint=LiteralSetConstraint.from_literals(("a",)),
            max_new_tokens=1,
        )


def test_probability_arrays_are_copied() -> None:
    root = np.asarray([1.0, 0.0])
    constrained_greedy_from_probabilities(
        {(): root},
        token_texts=("a", None),
        eos_token_id=1,
        constraint=LiteralSetConstraint.from_literals(("a",)),
        max_new_tokens=1,
    )

    assert root.flags.writeable


def test_literal_trie_supports_empty_literal_and_prefix_language() -> None:
    constraint = LiteralSetConstraint.from_literals(("", "a"))

    assert constraint.is_accepting(constraint.start_state)
    after_a = constraint.advance(constraint.start_state, "a")
    assert after_a is not None
    assert constraint.is_accepting(after_a)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: LiteralSetConstraint.from_literals(cast(list[str], [])),
            "non-empty",
        ),
        (
            lambda: LiteralSetConstraint.from_literals(cast(list[str], ["a", "a"])),
            "unique",
        ),
        (
            lambda: LiteralSetConstraint.from_literals(cast(list[str], ["a", 1])),
            "every literal",
        ),
        (
            lambda: LiteralSetConstraint.from_literals(cast(list[str], "a")),
            "non-string sequence",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [1.0]},
                token_texts=("not-none",),
                eos_token_id=0,
                constraint=LiteralSetConstraint.from_literals(("",)),
                max_new_tokens=1,
            ),
            "EOS token text",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [1.0, 0.0]},
                token_texts=(None, None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("a",)),
                max_new_tokens=1,
            ),
            "non-EOS token",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [1.0, 0.0]},
                token_texts=("a", None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("a",)),
                max_new_tokens=0,
            ),
            "max_new_tokens",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {},
                token_texts=("a", None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("a",)),
                max_new_tokens=1,
            ),
            "non-empty mapping",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [0.8, 0.1]},
                token_texts=("a", None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("a",)),
                max_new_tokens=1,
            ),
            "sum to one",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [1.0, float("nan")]},
                token_texts=("a", None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("a",)),
                max_new_tokens=1,
            ),
            "finite",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [1.0, 0.0, 0.0]},
                token_texts=("a", None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("a",)),
                max_new_tokens=1,
            ),
            "vocabulary-size entries",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [1.0, 0.0], (1,): [1.0, 0.0]},
                token_texts=("a", None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("a",)),
                max_new_tokens=2,
            ),
            "must not contain EOS",
        ),
        (
            lambda: constrained_greedy_from_probabilities(
                {(): [1.0, 0.0]},
                token_texts=("a", None),
                eos_token_id=1,
                constraint=LiteralSetConstraint.from_literals(("ab",)),
                max_new_tokens=2,
            ),
            "missing active prefix",
        ),
    ],
)
def test_invalid_constraint_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises((ValueError, ConstraintDeadEndError), match=message):
        operation()


def test_state_and_token_text_validation() -> None:
    constraint = LiteralSetConstraint.from_literals(("a",))

    with pytest.raises(ValueError, match="valid constraint state"):
        constraint.is_accepting(99)
    with pytest.raises(ValueError, match="non-empty string"):
        constraint.advance(0, "")


