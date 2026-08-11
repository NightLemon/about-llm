from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from about_llm.inference import beam_search_from_probabilities

ROOT = Path(__file__).resolve().parents[1]


def test_finite_beam_can_prune_the_globally_better_finished_sequence() -> None:
    table = {
        (): [0.6, 0.4, 0.0],
        (0,): [0.49, 0.0, 0.51],
        (1,): [0.0, 0.0, 1.0],
    }
    narrow = beam_search_from_probabilities(
        table,
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=1,
        max_new_tokens=2,
    )
    wide = beam_search_from_probabilities(
        table,
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=2,
        max_new_tokens=2,
    )

    assert narrow.returned_sequences[0].token_ids == (0, 2)
    assert math.exp(narrow.returned_sequences[0].cumulative_log_probability) == pytest.approx(
        0.6 * 0.51
    )
    assert wide.returned_sequences[0].token_ids == (1, 2)
    assert math.exp(wide.returned_sequences[0].cumulative_log_probability) == pytest.approx(
        0.4
    )
    assert wide.returned_sequences[0].cumulative_log_probability > narrow.returned_sequences[
        0
    ].cumulative_log_probability


def test_explicit_length_penalty_can_reverse_final_ranking() -> None:
    table = {
        (): [0.6, 0.4, 0.0, 0.0],
        (0,): [0.0, 0.0, 0.0, 1.0],
        (1,): [0.0, 0.0, 1.0, 0.0],
        (1, 2): [0.0, 0.0, 0.0, 1.0],
    }
    raw = beam_search_from_probabilities(
        table,
        vocabulary_size=4,
        eos_token_id=3,
        beam_width=2,
        max_new_tokens=3,
        length_penalty=0,
    )
    normalized = beam_search_from_probabilities(
        table,
        vocabulary_size=4,
        eos_token_id=3,
        beam_width=2,
        max_new_tokens=3,
        length_penalty=2,
    )

    assert raw.returned_sequences[0].token_ids == (0, 3)
    assert raw.returned_sequences[0].generated_length == 2
    assert normalized.returned_sequences[0].token_ids == (1, 2, 3)
    assert normalized.returned_sequences[0].generated_length == 3
    assert normalized.length_definition == (
        "generated tokens only; emitted EOS included; prompt excluded"
    )


def test_active_score_tie_uses_lexicographically_smaller_token_ids() -> None:
    result = beam_search_from_probabilities(
        {(): [0.5, 0.5, 0.0]},
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=1,
        max_new_tokens=1,
    )

    assert result.returned_sequences[0].token_ids == (0,)
    assert result.returned_sequences[0].finish_reason == "length"


def test_eos_is_finalized_and_never_requires_a_transition_row() -> None:
    result = beam_search_from_probabilities(
        {
            (): [0.5, 0.0, 0.5],
            (0,): [0.0, 0.0, 1.0],
        },
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=1,
        max_new_tokens=2,
        num_return_sequences=1,
    )

    assert result.steps[0].eos_finished[0].token_ids == (2,)
    assert all(2 not in prefix.token_ids for step in result.steps for prefix in step.active_before)
    assert {sequence.token_ids for sequence in result.ranked_candidates} >= {
        (2,),
        (0, 2),
    }


def test_probability_arrays_are_copied_and_zero_mass_tokens_are_not_expanded() -> None:
    root = np.asarray([1.0, 0.0, 0.0])
    result = beam_search_from_probabilities(
        {(): root},
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=1,
        max_new_tokens=1,
    )

    assert root.flags.writeable
    assert result.steps[0].positive_expansions_considered == 1
    assert result.returned_sequences[0].token_ids == (0,)


def test_multiple_return_sequences_are_finally_ranked_not_input_ordered() -> None:
    result = beam_search_from_probabilities(
        {
            (): [0.55, 0.45, 0.0],
            (0,): [0.0, 0.0, 1.0],
            (1,): [0.0, 0.0, 1.0],
        },
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=2,
        max_new_tokens=2,
        num_return_sequences=2,
    )

    assert [sequence.token_ids for sequence in result.returned_sequences] == [
        (0, 2),
        (1, 2),
    ]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: beam_search_from_probabilities(
                {},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=1,
            ),
            "non-empty mapping",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [0.5, 0.4]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=1,
            ),
            "sum to one",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [1.0, -0.0, 0.0]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=1,
            ),
            "vocabulary_size entries",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [float("nan"), 1.0]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=1,
            ),
            "finite",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [1.0, 0.0], (1,): [1.0, 0.0]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=2,
            ),
            "must not contain EOS",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [1.0, 0.0]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=0,
                max_new_tokens=1,
            ),
            "beam_width",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [1.0, 0.0]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=1,
                length_penalty=-1,
            ),
            "length_penalty",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [1.0, 0.0]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=1,
                num_return_sequences=2,
            ),
            "num_return_sequences",
        ),
        (
            lambda: beam_search_from_probabilities(
                {(): [1.0, 0.0]},
                vocabulary_size=2,
                eos_token_id=1,
                beam_width=1,
                max_new_tokens=2,
            ),
            "missing active prefix",
        ),
    ],
)
def test_invalid_beam_contracts_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_beam_search_toy_records_pruning_length_penalty_and_scope() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "projects" / "inference-serving" / "beam_search_toy.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["pruning_counterexample"]["beam_1"]["returned_sequences"][0][
        "token_ids"
    ] == [0, 2]
    assert artifact["pruning_counterexample"]["beam_2"]["returned_sequences"][0][
        "token_ids"
    ] == [1, 2]
    assert artifact["length_penalty_counterexample"]["alpha_0"][
        "returned_sequences"
    ][0]["token_ids"] == [0, 3]
    assert artifact["length_penalty_counterexample"]["alpha_2"][
        "returned_sequences"
    ][0]["token_ids"] == [1, 2, 3]
    assert artifact["scope"] == {
        "beam_pruning_eos_and_length_finalization_executed": True,
        "global_sequence_optimality_proved": False,
        "length_penalty_includes_eos_and_excludes_prompt": True,
        "model_tokenizer_kv_or_gpu_executed": False,
        "runtime_or_provider_equivalence_claimed": False,
        "text_quality_or_performance_proved": False,
    }
