from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from about_llm.from_scratch.mini_gpt_training_trace import (  # noqa: E402
    run_minigpt_training_trace,
)

pytestmark = [pytest.mark.formula, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "trace_minigpt_training_step.py"


def test_same_sample_runs_forward_loss_backward_and_one_update() -> None:
    report = run_minigpt_training_trace()

    assert report["sample"] == {
        "text": "你好🙂!",
        "input_ids": [265, 264, 33, 266],
        "original_labels": [264, 33, 266, 267],
        "supervised_labels": [264, 33, 266, -100],
        "effective_target_count": 3,
    }
    model = report["model"]
    assert model["vocab_size"] == 268
    assert model["trainable_parameter_count"] == 6496
    assert model["token_embedding_and_lm_head_are_tied"] is True
    assert model["causal_mask_matches_input_trace"] is True

    forward = report["forward_before_update"]
    assert forward["logits_shape"] == [1, 4, 268]
    assert [row["input_piece"] for row in forward["positions"]] == [
        "<BOS>",
        "你好🙂",
        "!",
        "<EOS>",
    ]
    assert [row["target_piece"] for row in forward["positions"]] == [
        "你好🙂",
        "!",
        "<EOS>",
        "<PAD>",
    ]
    assert [row["negative_log_probability"] is not None for row in forward["positions"]] == [
        True,
        True,
        True,
        False,
    ]
    assert [row["target_probability"] for row in forward["positions"]] == pytest.approx(
        [0.003612, 0.004073, 0.003587, 0.004083], abs=1e-6
    )
    assert [
        row["negative_log_probability"] for row in forward["positions"][:3]
    ] == pytest.approx([5.623527, 5.503419, 5.630446], abs=1e-6)
    assert forward["mean_nll_from_model"] == pytest.approx(5.585798, abs=1e-6)
    assert forward["mean_nll_recomputed_from_positions"] == pytest.approx(
        forward["mean_nll_from_model"], abs=1e-6
    )
    assert forward["perplexity_on_three_targets"] == pytest.approx(
        math.exp(forward["mean_nll_from_model"])
    )

    update = report["backward_and_update"]
    assert math.isfinite(update["gradient_global_l2"])
    assert update["gradient_global_l2"] > 0.0
    assert update["updated_parameter_tensor_count"] > 0
    assert update["updated_parameter_tensor_count"] == update["parameter_tensor_count"] == 12
    assert update["maximum_parameter_change"] > 0.0
    assert update["gradient_global_l2"] == pytest.approx(3.086677, abs=1e-6)
    assert update["mean_nll_after_one_step"] == pytest.approx(5.134247, abs=1e-6)
    assert update["loss_decreased_on_same_sample"] is True
    assert update["mean_nll_after_one_step"] < forward["mean_nll_from_model"]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("seed", True), ("seed", -1), ("learning_rate", 0.0), ("learning_rate", 2.0)],
)
def test_trace_rejects_invalid_training_controls(keyword: str, value: object) -> None:
    with pytest.raises(ValueError):
        run_minigpt_training_trace(**{keyword: value})  # type: ignore[arg-type]


@pytest.mark.slow
def test_cli_has_a_guided_view_and_optional_json() -> None:
    guided = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    guided_text = guided.stdout.decode("utf-8")
    assert "同一个样本现在进入 MiniGPT" in guided_text
    assert "位置 0: <BOS> → 你好🙂" in guided_text
    assert "随机初始化模型的一步 loss 下降" in guided_text
    assert guided.stderr == b""

    machine = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    report = json.loads(machine.stdout.decode("utf-8"))
    assert report["schema_version"] == "about-llm.minigpt-training-trace.v1"
    assert report["scope"]["pretrained_checkpoint_loaded"] is False
    assert machine.stderr == b""
