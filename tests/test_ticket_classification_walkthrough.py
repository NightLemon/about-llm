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

pytestmark = [pytest.mark.formula, pytest.mark.contract, pytest.mark.smoke]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "ticket_classification_walkthrough.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ticket_classification_walkthrough", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_walkthrough_connects_split_loss_gradient_and_sliced_metric() -> None:
    walkthrough: dict[str, Any] = _load().build_walkthrough()

    assert walkthrough["task"]["sample_unit"] == "thread_id"
    assert walkthrough["split_audit"]["row_level"]["overlap_threads"] == [
        "thread-100",
        "thread-200",
    ]
    assert walkthrough["split_audit"]["thread_level"]["overlap_threads"] == []

    prediction = walkthrough["prediction"]
    assert sum(prediction["probabilities"]) == pytest.approx(1.0)
    assert prediction["probabilities"][2] == pytest.approx(0.03911257327)
    assert prediction["negative_log_likelihood"] == pytest.approx(3.2413112967)
    assert sum(prediction["logit_gradient"]) == pytest.approx(0.0)
    assert prediction["updated_probabilities"][2] > prediction["probabilities"][2]
    assert prediction["updated_negative_log_likelihood"] < prediction["negative_log_likelihood"]
    assert prediction["updated_logits"][2] == max(prediction["updated_logits"])

    assert walkthrough["metrics"]["accuracy"] == 0.99
    assert walkthrough["metrics"]["fraud_review_recall"] == 0.0
    assert walkthrough["scope"] == {
        "real_ticket_text_or_trained_model_used": False,
        "logits_treated_as_direct_parameters_for_one_local_step": True,
        "generalization_or_business_value_measured": False,
    }


def test_softmax_uses_a_stable_shift_and_step_validates_inputs() -> None:
    module = _load()

    probabilities = module._softmax((10_000.0, 9_999.0, 9_998.0))
    assert all(math.isfinite(value) for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="one value per label"):
        module._classification_step((1.0, 2.0), target_index=0, learning_rate=1.0)
    with pytest.raises(ValueError, match="outside the label set"):
        module._classification_step((1.0, 2.0, 3.0), target_index=3, learning_rate=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        module._classification_step((1.0, 2.0, 3.0), target_index=0, learning_rate=float("nan"))


def test_cli_output_explains_what_each_number_means(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load()

    module.main()

    output = capsys.readouterr().out
    assert "逐行切分的跨集合 thread: ['thread-100', 'thread-200']" in output
    assert "按 thread 切分的跨集合 thread: []" in output
    assert "dL/dlogits = p-y" in output
    assert "99% accuracy 可以与风险类 recall=0 同时出现" in output
    assert "本实验没有训练真实模型" in output


def test_cli_prints_utf8_on_windows_console_encodings() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )

    output = completed.stdout.decode("utf-8")
    assert "工单分类最小学习闭环" in output
    assert "fraud_review recall" in output
