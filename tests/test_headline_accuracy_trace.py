from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from about_llm.evaluation.headline_accuracy_trace import build_headline_accuracy_trace

pytestmark = [pytest.mark.formula, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "evaluation-gate" / "trace_headline_accuracy_trap.py"


def test_headline_gain_is_blocked_by_uncertainty_and_critical_regression() -> None:
    report = build_headline_accuracy_trace()

    assert report["headline"] == {
        "baseline_correct": 22,
        "candidate_correct": 24,
        "baseline_accuracy": pytest.approx(22 / 30),
        "candidate_accuracy": pytest.approx(24 / 30),
        "candidate_minus_baseline_correct": 2,
    }
    changes = report["paired_changes"]
    assert (changes["improved"], changes["regressed"], changes["unchanged"]) == (
        4,
        2,
        24,
    )
    assert [row["case_id"] for row in changes["rows"]] == [
        "routine-19",
        "routine-20",
        "routine-21",
        "routine-22",
        "cross-tenant-03",
        "cross-tenant-04",
    ]
    assert [row["change"] for row in changes["rows"]] == [
        "improved",
        "improved",
        "improved",
        "improved",
        "regressed",
        "regressed",
    ]

    comparison = report["comparison"]
    assert comparison["quality"]["baseline_mean"] == pytest.approx(22 / 30)
    assert comparison["quality"]["candidate_mean"] == pytest.approx(24 / 30)
    assert comparison["quality"]["mean_difference"] == pytest.approx(2 / 30)
    assert comparison["quality"]["confidence_low"] == pytest.approx(-0.1)
    assert comparison["quality"]["confidence_high"] == pytest.approx(7 / 30)
    critical = comparison["protected_slices"]["cross_tenant"]
    assert critical["baseline_mean"] == pytest.approx(4 / 5)
    assert critical["candidate_mean"] == pytest.approx(2 / 5)
    assert critical["mean_difference"] == pytest.approx(-2 / 5)
    assert critical["confidence_low"] == pytest.approx(-4 / 5)
    assert critical["confidence_high"] == pytest.approx(0.0)
    assert comparison["passed"] is False
    assert len(comparison["reasons"]) == 2
    assert report["decision"]["release"] == "block"


@pytest.mark.smoke
def test_cli_explains_the_decision_and_can_emit_json() -> None:
    guided = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    guided_text = guided.stdout.decode("utf-8")
    assert "Baseline: 22 / 30" in guided_text
    assert "Candidate: 24 / 30" in guided_text
    assert "cross-tenant-03 [cross_tenant] 退化" in guided_text
    assert "发布决定: 拦截 Candidate" in guided_text
    assert guided.stderr == b""

    machine = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    payload = json.loads(machine.stdout.decode("utf-8"))
    assert payload["schema_version"] == "about-llm.headline-accuracy-trace.v1"
    assert payload["scope"]["model_or_provider_called"] is False
    assert payload["scope"]["general_safety_metric_executed"] is False
    assert payload["scope"]["teaching_cli_exit_code_represents_execution_not_release"] is True
    assert machine.stderr == b""
