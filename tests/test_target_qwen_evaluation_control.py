from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch

from about_llm.evaluation.target_qwen_control import (
    TARGET_QWEN_CHECKPOINT_MANIFEST_FINGERPRINT,
    TargetQwenEvaluationCase,
    TargetQwenEvaluationSpec,
    execute_loaded_target_qwen_evaluation,
    load_target_qwen_evaluation_spec,
    verify_recorded_target_qwen_evaluation_report,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "evaluation-gate"
SUITE = PROJECT / "target-qwen-behavior-suite.control.json"
RECORDED_REPORT = PROJECT / "target-qwen-behavior.recorded-report.json"
SCRIPT = PROJECT / "run_qwen_target_behavior_evaluation.py"


class FixtureTokenizer:
    chat_template = "fixture"
    eos_token_id = 9
    pad_token_id = 0

    def __init__(self) -> None:
        self._case = 0

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        self._case += 1
        return torch.tensor([[1, self._case]], dtype=torch.long)

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return {3: "42", 4: "wrong"}[token_ids[0]]


class FixtureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def generate(self, *, input_ids: torch.Tensor, **_kwargs: Any) -> Any:
        self.calls += 1
        token = 3 if self.calls == 1 else 4
        continuation = torch.tensor([[token, 9]], dtype=torch.long)
        return SimpleNamespace(sequences=torch.cat((input_ids, continuation), dim=1))


def _tiny_spec() -> TargetQwenEvaluationSpec:
    return TargetQwenEvaluationSpec(
        checked_at="2026-08-15",
        suite_id="fixture",
        checkpoint_manifest_fingerprint=TARGET_QWEN_CHECKPOINT_MANIFEST_FINGERPRINT,
        system_prompt="Answer only.",
        max_new_tokens=4,
        metric_revisions={
            "exact_match": "about-llm.normalized-exact-match.v1",
            "literal_exact_match": "about-llm.literal-exact-match.v1",
            "token_f1": "about-llm.token-f1.v1",
        },
        cases=(
            TargetQwenEvaluationCase("first", "one", "42", ("a",)),
            TargetQwenEvaluationCase("second", "two", "42", ("a", "b")),
        ),
    )


def test_suite_manifest_is_closed_and_bound_to_reviewed_checkpoint() -> None:
    spec = load_target_qwen_evaluation_spec(SUITE)

    assert len(spec.cases) == 7
    assert spec.max_new_tokens == 12
    assert spec.checkpoint_manifest_fingerprint == (
        TARGET_QWEN_CHECKPOINT_MANIFEST_FINGERPRINT
    )
    assert spec.suite_fingerprint.startswith("sha256:")
    assert {case.case_id for case in spec.cases} == {
        "zh-arithmetic",
        "en-arithmetic",
        "zh-factual",
        "en-factual",
        "zh-abstention",
        "zh-copy",
        "zh-json",
    }


def test_tiny_loaded_execution_records_failures_without_rewriting_them() -> None:
    model = FixtureModel()
    results, aggregates = execute_loaded_target_qwen_evaluation(
        model, FixtureTokenizer(), _tiny_spec()
    )

    assert model.calls == 2
    assert model.training is False
    assert model.weight.requires_grad is False
    assert [result["output"] for result in results] == ["42", "wrong"]
    assert [result["literal_exact_match"] for result in results] == [1.0, 0.0]
    assert [result["exact_match"] for result in results] == [1.0, 0.0]
    assert aggregates["literal_exact_match_pass_count"] == 1
    assert aggregates["literal_exact_match_mean"] == 0.5
    assert aggregates["exact_match_pass_count"] == 1
    assert aggregates["exact_match_mean"] == 0.5
    assert aggregates["by_slice"] == {
        "a": {
            "case_count": 2,
            "literal_exact_match_pass_count": 1,
            "literal_exact_match_mean": 0.5,
            "exact_match_pass_count": 1,
            "exact_match_mean": 0.5,
            "token_f1_mean": 0.5,
        },
        "b": {
            "case_count": 1,
            "literal_exact_match_pass_count": 0,
            "literal_exact_match_mean": 0.0,
            "exact_match_pass_count": 0,
            "exact_match_mean": 0.0,
            "token_f1_mean": 0.0,
        },
    }


def test_suite_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest = json.loads(SUITE.read_text(encoding="utf-8"))
    manifest["cases"][1]["case_id"] = manifest["cases"][0]["case_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_target_qwen_evaluation_spec(path)


def test_suite_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-key.json"
    path.write_text('{"control_version":"a","control_version":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_target_qwen_evaluation_spec(path)


def test_recorded_report_is_reviewed_and_scope_limited() -> None:
    spec = load_target_qwen_evaluation_spec(SUITE)
    report = verify_recorded_target_qwen_evaluation_report(RECORDED_REPORT, spec)

    assert report["aggregates"]["case_count"] == 7
    assert report["scope"]["target_checkpoint_weights_loaded"] is True
    assert report["scope"]["representative_benchmark_or_quality_proven"] is False
    assert report["scope"]["performance_benchmark_performed"] is False


def test_recorded_report_tamper_is_rejected(tmp_path: Path) -> None:
    spec = load_target_qwen_evaluation_spec(SUITE)
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    report["results"][0]["output"] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_recorded_target_qwen_evaluation_report(path, spec)


def test_cooperatively_rehashed_result_drift_is_not_reviewed(tmp_path: Path) -> None:
    spec = load_target_qwen_evaluation_spec(SUITE)
    report = json.loads(RECORDED_REPORT.read_text(encoding="utf-8"))
    report["results"][0]["prompt_token_count"] += 1
    projection = dict(report)
    del projection["report_fingerprint"]
    report["report_fingerprint"] = "sha256:" + artifact_fingerprint(projection)
    path = tmp_path / "cooperative.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="not the reviewed recording"):
        verify_recorded_target_qwen_evaluation_report(path, spec)


def test_cli_verifies_recorded_report_without_loading_weights() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify", str(RECORDED_REPORT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert report["report_fingerprint"] == (
        "sha256:dd30a278cbc076c973c0b0babc9e752b1063d8bfb114c852b34ea42b2cd85c43"
    )
    assert report["scope"]["performance_benchmark_performed"] is False
