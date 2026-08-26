from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from about_llm.from_scratch.language_model_sample import (
    build_language_model_sample,
)

pytestmark = pytest.mark.formula

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "trace_language_model_sample.py"


def test_default_sample_connects_bytes_tokens_shift_and_masks() -> None:
    report = build_language_model_sample()

    assert report["counts"] == {
        "unicode_code_points": 4,
        "utf8_bytes": 11,
        "model_text_tokens": 2,
    }
    tokenizer = report["tokenizer"]
    assert tokenizer["text_token_ids"] == [264, 33]
    assert tokenizer["round_trip_matches_input"] is True
    assert tokenizer["pieces"] == [
        {
            "position": 0,
            "token_id": 264,
            "bytes_hex": "e4bda0e5a5bdf09f9982",
            "utf8_preview": "你好🙂",
        },
        {
            "position": 1,
            "token_id": 33,
            "bytes_hex": "21",
            "utf8_preview": "!",
        },
    ]

    model = report["teaching_model"]
    assert model["special_token_ids"] == {"BOS": 265, "EOS": 266, "PAD": 267}
    assert model["full_sequence_ids"] == [265, 264, 33, 266, 267]
    assert model["model_input_ids"] == [265, 264, 33, 266]
    assert model["labels"] == [264, 33, 266, 267]
    assert model["loss_mask"] == [True, True, True, False]
    assert model["effective_target_count"] == 3
    assert model["causal_attention_mask"] == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_position_trace_explains_predictor_target_and_padding() -> None:
    rows = build_language_model_sample()["teaching_model"]["position_trace"]

    assert [row["input_piece"] for row in rows] == ["<BOS>", "你好🙂", "!", "<EOS>"]
    assert [row["target_piece"] for row in rows] == ["你好🙂", "!", "<EOS>", "<PAD>"]
    assert [row["visible_input_positions"] for row in rows] == [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
    ]
    assert [row["included_in_loss"] for row in rows] == [True, True, True, False]


def test_special_ids_follow_the_learned_vocabulary_without_collision() -> None:
    report = build_language_model_sample(
        text="ab",
        training_documents=("abab",),
        vocab_size=258,
        min_pair_frequency=2,
    )
    tokenizer = report["tokenizer"]
    model = report["teaching_model"]

    assert tokenizer["text_token_ids"] == [256]
    assert model["special_token_ids"] == {"BOS": 257, "EOS": 258, "PAD": 259}
    assert model["embedding_row_count_required"] == 260


@pytest.mark.parametrize("text", ["", 7])
def test_sample_requires_nonempty_string(text: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_language_model_sample(text=text)  # type: ignore[arg-type]


def test_training_documents_reject_one_bare_string() -> None:
    with pytest.raises(TypeError, match="iterable of strings"):
        build_language_model_sample(training_documents="abab")


def test_cli_emits_utf8_json_without_running_a_model() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    report = json.loads(completed.stdout.decode("utf-8"))

    assert report["text"] == "你好🙂!"
    assert report["scope"]["byte_bpe_trained_and_executed"] is True
    assert report["scope"]["embedding_or_language_model_executed"] is False
    assert completed.stderr == b""
