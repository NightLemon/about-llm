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

import jax.numpy as jnp

from about_llm.from_scratch.gpt_cross_framework_training import (
    GPT_CROSS_FRAMEWORK_TRAINING_PARITY_VERSION,
    layernorm_jax_forward_with_embedding_mask,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "projects"
    / "jax-minigpt"
    / "cross_framework_training_parity.py"
)


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


def test_pytorch_jax_adamw_clipping_schedule_and_mask_parity() -> None:
    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    report = json.loads(
        completed.stdout,
        parse_constant=_reject_nonfinite,
    )

    assert completed.stderr == ""
    assert (
        report["schema_version"]
        == GPT_CROSS_FRAMEWORK_TRAINING_PARITY_VERSION
    )
    assert report["runtime"]["jax_backend"] == "cpu"
    assert report["runtime"]["torch_device"] == "cpu"
    assert report["fixture"]["comparison_tolerance"] == 5e-6
    assert report["fixture"]["optimizer"] == {
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "kind": "AdamW",
        "learning_rates": [0.02, 0.01, 0.005],
        "max_grad_norm": 0.08,
        "weight_decay": 0.03,
        "weight_decay_mask": "all parameters",
    }
    assert report["fixture"]["dropout"] == {
        "generator": "NumPy PCG64",
        "kind": "externally materialized inverted dropout",
        "mask_sha256": [
            "sha256:7277dcc5670adf12d8eacb46155845890aa0272e54c51be11df570e6aa40287e",
            "sha256:46a4c1fbd64af64766839d4992e097b78e1eabf2cf4469804a07ba816db025cf",
            "sha256:47b90e2328e71d62e2c3d718b7714109176c7083db9d1e5f040978f3b6dfc67c",
        ],
        "mask_shape": [2, 4, 8],
        "rate": 0.25,
        "seed": 20260814,
        "site": "embedding sum only",
    }
    assert [step["kept_elements"] for step in report["steps"]] == [
        54,
        50,
        45,
    ]
    assert [step["torch_adam_step"] for step in report["steps"]] == [1, 2, 3]
    assert [step["jax_adam_count"] for step in report["steps"]] == [1, 2, 3]
    assert [step["jax_schedule_count"] for step in report["steps"]] == [
        1,
        2,
        3,
    ]
    maxima = report["comparison"]["maximum_difference_across_steps"]
    assert maxima["raw_gradient_global_max_abs_difference"] <= 5e-6
    assert maxima["clipped_gradient_global_max_abs_difference"] <= 5e-6
    assert maxima["parameter_global_max_abs_difference"] <= 5e-6
    assert maxima["first_moment_global_max_abs_difference"] <= 5e-6
    assert maxima["second_moment_global_max_abs_difference"] <= 5e-6
    assert maxima["post_step_logits_max_abs_difference"] <= 5e-6
    assert report["comparison"][
        "wrong_mask_final_parameter_max_abs_difference"
    ] > 0.06
    assert all(report["assertions"].values())
    assert report["scope"] == {
        "adamw_first_second_moments_and_count_compared": True,
        "all_parameter_weight_decay_compared": True,
        "checkpoint_resume_or_artifact_serialization_compared": False,
        "cuda_tpu_multi_device_or_sharding_executed": False,
        "dropout_prng_state_advance_compared": False,
        "framework_native_rng_equivalence_claimed": False,
        "jit_compile_or_async_timing_compared": False,
        "large_model_training_convergence_or_performance_proved": False,
        "learning_rate_schedule_compared": True,
        "norm_or_bias_weight_decay_mask_compared": False,
        "raw_and_global_norm_clipped_gradients_compared": True,
        "same_initial_parameter_values_compared": True,
        "shared_materialized_embedding_dropout_masks_compared": True,
        "three_post_update_forwards_compared": True,
        "wrong_materialized_mask_counterfactual_executed": True,
    }
    fingerprint = report.pop("report_fingerprint")
    assert fingerprint == (
        "sha256:68ffa8093a1f2b986fa7c0c8d9c45075"
        "dcb17c1cf5c0b92d0852e874e175c609"
    )
    assert fingerprint == "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout


def test_masked_forward_rejects_invalid_mask_contract_before_parameter_access() -> None:
    input_ids = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="embedding_mask must have shape"):
        layernorm_jax_forward_with_embedding_mask(
            {},
            input_ids,
            jnp.ones((1, 4, 7), dtype=jnp.float32),
        )
    with pytest.raises(TypeError, match="floating dtype"):
        layernorm_jax_forward_with_embedding_mask(
            {},
            input_ids,
            jnp.ones((1, 4, 8), dtype=jnp.int32),
        )
