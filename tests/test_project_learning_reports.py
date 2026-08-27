from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.smoke, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]

REPORT_CASES = (
    (
        "projects/transformers-basics/smoke_tiny.py",
        (("configuration",), ("fixture", "input_ids"), ("scope",)),
        ("torch", "transformers"),
    ),
    (
        "projects/transformers-basics/moe_routing.py",
        (("fixture", "router_logits_by_token"), ("configuration",), ("scope",)),
        (),
    ),
    (
        "projects/inference-serving/quantized_bundle_toy.py",
        (("configuration", "bit_width"), ("output_error_vs_fp32",), ("scope",)),
        (),
    ),
    (
        "projects/inference-serving/minigpt_checkpoint_toy.py",
        (("configuration", "bit_width"), ("tokenizer", "prompt"), ("scope",)),
        ("torch",),
    ),
    (
        "projects/evaluation-gate/clustered_bootstrap_toy.py",
        (("experiment", "case_weighted_estimand"), ("conclusion",), ("scope",)),
        (),
    ),
    (
        "projects/evaluation-gate/clustered_randomization_toy.py",
        (
            ("experiment", "cluster_sign_flip"),
            ("experiment", "comparison_design"),
            ("conclusion",),
            ("scope",),
        ),
        (),
    ),
    (
        "projects/evaluation-gate/paired_randomization_toy.py",
        (("experiment", "paired_unit"), ("conclusion",), ("scope",)),
        (),
    ),
    (
        "projects/rag-foundations/retriever_learning_toy.py",
        (("fixture", "dense"), ("observations",), ("scope",)),
        (),
    ),
    (
        "projects/cloud-api-contracts/usage_budget_toy.py",
        (("scenario",), ("transitions",), ("conclusion",), ("scope",)),
        (),
    ),
    (
        "projects/cloud-api-contracts/gemini_interactions_replay.py",
        (("input_artifact", "sha256"), ("conclusion",), ("evidence_boundary",)),
        (),
    ),
    (
        "projects/safe-agent/decision_theory_toy.py",
        (("fixture", "action_utilities_by_state"), ("transition_systems",), ("scope",)),
        (),
    ),
    (
        "projects/single-gpu-finetuning/continual_replay_toy.py",
        (("mode",), ("extended_mode",), ("task_contract",), ("scope",)),
        ("torch",),
    ),
    (
        "projects/single-gpu-finetuning/smoke_transformer_ppo.py",
        (("task_contract", "reward_rule"), ("configuration",), ("scope",)),
        ("torch", "transformers"),
    ),
    (
        "projects/single-gpu-finetuning/smoke_transformer_reward_model.py",
        (("task_contract", "loss"), ("outcome",), ("scope",)),
        ("torch", "transformers"),
    ),
    (
        "projects/single-gpu-finetuning/minhash_lsh_toy.py",
        (("authored_relationships",), ("conclusion",), ("scope",)),
        (),
    ),
    (
        "projects/rag-foundations/rag_request_walkthrough.py",
        (("requests",), ("scope",)),
        (),
    ),
)
SLOW_REPORT_SCRIPTS = frozenset(
    {
        "projects/transformers-basics/smoke_tiny.py",
        "projects/inference-serving/minigpt_checkpoint_toy.py",
        "projects/single-gpu-finetuning/continual_replay_toy.py",
        "projects/single-gpu-finetuning/smoke_transformer_ppo.py",
        "projects/single-gpu-finetuning/smoke_transformer_reward_model.py",
    }
)


def _nested_value(payload: object, path: tuple[str, ...]) -> object:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise AssertionError(f"missing report field: {'.'.join(path)}")
        current = current[key]
    return current


def _run_json_report(relative_script: str, *arguments: str) -> dict[str, object]:
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            value for value in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if value
        ),
    }
    completed = subprocess.run(
        [sys.executable, str(ROOT / relative_script), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=environment,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("project report must be a JSON object")
    return payload


@pytest.mark.parametrize(
    ("relative_script", "required_paths", "required_modules"),
    [
        pytest.param(
            relative_script,
            required_paths,
            required_modules,
            id=Path(relative_script).stem,
            marks=(pytest.mark.slow,)
            if relative_script in SLOW_REPORT_SCRIPTS
            else (),
        )
        for relative_script, required_paths, required_modules in REPORT_CASES
    ],
)
def test_project_stdout_includes_learning_context(
    relative_script: str,
    required_paths: tuple[tuple[str, ...], ...],
    required_modules: tuple[str, ...],
) -> None:
    for module_name in required_modules:
        pytest.importorskip(module_name)

    report = _run_json_report(relative_script)

    for path in required_paths:
        assert _nested_value(report, path) is not None

    if relative_script.endswith("rag_request_walkthrough.py"):
        assert report["requests"][0]["packing"]["selected_evidence"]
    if relative_script.endswith("retriever_learning_toy.py"):
        assert all(report["observations"].values())
    if relative_script.endswith("clustered_randomization_toy.py"):
        alternatives = {
            report[name]["alternative"]
            for name in (
                "naive_case_sign_flip",
                "cluster_joint_case_weighted",
                "cluster_joint_equal_weighted",
            )
        }
        assert alternatives == {"greater"}


@pytest.mark.slow
def test_jax_training_report_includes_fixture_when_available() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    report = _run_json_report("projects/jax-minigpt/train_tiny.py", "--steps", "2")

    assert report["fixture"]["input_ids"] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert report["scope"]["fixed_batch_overfit_executed"] is True


@pytest.mark.parametrize(
    "relative_script",
    [
        "projects/single-gpu-finetuning/smoke_trl_sft.py",
        "projects/single-gpu-finetuning/smoke_trl_dpo.py",
    ],
)
@pytest.mark.slow
def test_trl_smoke_stdout_is_one_contextual_json_when_available(
    relative_script: str,
) -> None:
    pytest.importorskip("datasets")
    pytest.importorskip("trl")

    report = _run_json_report(relative_script)

    assert report["task_contract"]
    assert report["outcome"]["training_loss_decreased"] is True
    assert report["trainer_metrics"]


@pytest.mark.parametrize(
    "relative_script",
    [
        "projects/rag-framework-adapters/demo.py",
        "projects/rag-framework-adapters/parity_control.py",
    ],
)
@pytest.mark.slow
def test_framework_report_includes_readable_evidence_when_available(
    relative_script: str,
) -> None:
    pytest.importorskip("langchain_core")
    pytest.importorskip("llama_index")

    report = _run_json_report(relative_script)

    if relative_script.endswith("demo.py"):
        assert report["input"]["documents"]
        assert all(report["conclusion"].values())
    else:
        assert report["cases"]["engineering"]["authorized_evidence"]
        assert report["cases"]["engineering"]["rendered_prompt"]
