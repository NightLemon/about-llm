from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("torch")

from about_llm.from_scratch.gpt_cross_framework import (
    GPT_CROSS_FRAMEWORK_PARITY_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "projects" / "jax-minigpt" / "cross_framework_parity.py"


def _reject_nonfinite(value: str) -> NoReturn:
    raise AssertionError(f"non-standard JSON number: {value}")


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def test_pytorch_jax_layernorm_forward_backward_and_sgd_parity() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    report = json.loads(
        completed.stdout,
        parse_constant=_reject_nonfinite,
    )

    assert completed.stderr == ""
    assert report["schema_version"] == GPT_CROSS_FRAMEWORK_PARITY_VERSION
    assert report["fixture"]["config"]["normalization"] == (
        "LayerNorm with affine scale/bias"
    )
    assert report["fixture"]["config"]["normalization_epsilon"] == 1e-5
    assert report["fixture"]["targets"] == [
        [1, 2, 3, 4],
        [2, -100, 0, 5],
    ]
    comparison = report["comparison"]
    assert comparison["initial_logits_max_abs_difference"] <= 2e-6
    assert comparison["initial_loss_abs_difference"] <= 2e-6
    assert comparison["gradient_global_max_abs_difference"] <= 2e-6
    assert comparison[
        "post_update_parameter_global_max_abs_difference"
    ] <= 2e-6
    assert comparison["post_update_logits_max_abs_difference"] <= 2e-6
    assert comparison["post_update_loss_abs_difference"] <= 2e-6
    assert comparison[
        "native_rmsnorm_counterfactual_logits_max_abs_difference"
    ] > 1e-3
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "adamw_optimizer_state_or_schedule_compared": False,
        "cuda_tpu_multi_device_or_sharding_executed": False,
        "dropout_rng_or_stochastic_sampling_compared": False,
        "every_unique_parameter_gradient_compared": True,
        "framework_rng_equivalence_claimed": False,
        "jit_compile_or_async_timing_compared": False,
        "jax_cpu_execution_forced": True,
        "large_model_training_convergence_or_performance_proved": False,
        "layernorm_bias_epsilon_gelu_mask_and_tying_aligned": True,
        "masked_cross_entropy_forward_compared": True,
        "native_rmsnorm_architecture_counterfactual_executed": True,
        "plain_sgd_one_step_compared": True,
        "post_update_forward_compared": True,
        "same_initial_parameter_values_compared": True,
    }
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout
