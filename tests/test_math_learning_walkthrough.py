# ruff: noqa: RUF001 -- Full-width punctuation is intentional in learner output checks.
from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.formula, pytest.mark.smoke]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "math_learning_walkthrough.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("math_learning_walkthrough", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_walkthrough_matches_independent_hand_calculation() -> None:
    walkthrough: dict[str, Any] = _load().build_walkthrough()

    assert walkthrough["matrix_shape"] == [2, 3]
    assert walkthrough["logits"] == (2.0, 1.0, 0.0)

    denominator = 1.0 + math.exp(-1.0) + math.exp(-2.0)
    expected_probabilities = (
        1.0 / denominator,
        math.exp(-1.0) / denominator,
        math.exp(-2.0) / denominator,
    )
    assert walkthrough["probabilities"] == pytest.approx(expected_probabilities)
    assert walkthrough["negative_log_likelihood"] == pytest.approx(
        -math.log(expected_probabilities[1])
    )
    assert walkthrough["logit_gradient"] == pytest.approx(
        (expected_probabilities[0], expected_probabilities[1] - 1.0, expected_probabilities[2])
    )

    finite_difference = walkthrough["finite_difference"]
    assert finite_difference["analytic_derivative"] == pytest.approx(expected_probabilities[0])
    assert finite_difference["numerical_derivative"] == pytest.approx(
        finite_difference["analytic_derivative"], rel=1e-8, abs=1e-8
    )


def test_softmax_is_stable_and_negative_gradient_step_improves_target() -> None:
    module = _load()
    probabilities = module._softmax((10_000.0, 9_999.0, 9_998.0))

    assert all(math.isfinite(value) for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)

    walkthrough: dict[str, Any] = module.build_walkthrough()
    target_index = walkthrough["target_index"]
    assert (
        walkthrough["updated_probabilities"][target_index]
        > walkthrough["probabilities"][target_index]
    )
    assert (
        walkthrough["updated_negative_log_likelihood"]
        < walkthrough["negative_log_likelihood"]
    )


def test_cli_explains_the_numbers_and_uses_utf8() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )

    output = completed.stdout.decode("utf-8")
    assert "从两个数字到一次模型更新" in output
    assert "第一个 logit = 1×1 + 1×1 = 2" in output
    assert "正确 token 概率" in output
    assert "NLL" in output
    assert "只负责把数学关系摊开" in output
