from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from about_llm.evaluation import (
    EVALUATION_COMPARISON_HTML_SCOPE,
    EvaluationCase,
    EvaluationComparisonArtifact,
    EvaluationResult,
    ReleaseGate,
    build_evaluation_comparison_artifact,
    load_evaluation_comparison_artifact,
    render_evaluation_comparison_html,
)
from about_llm.evaluation.cli import compare_results, main

pytestmark = [pytest.mark.contract, pytest.mark.security]

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "evaluation-gate"
COMPARISON = PROJECT / "comparison.example.json"


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.tags.append(tag)


def test_project_comparison_renders_deterministic_script_free_html() -> None:
    artifact = load_evaluation_comparison_artifact(COMPARISON)

    first = render_evaluation_comparison_html(artifact)
    second = render_evaluation_comparison_html(artifact)
    parser = _TagCollector()
    parser.feed(first)

    assert first == second
    assert first.startswith("<!doctype html>\n")
    assert parser.tags.count("html") == 1
    assert "table" in parser.tags
    assert "script" not in parser.tags
    assert "http://" not in first
    assert "https://" not in first
    assert "default-src 'none'" in first
    assert 'content="artifact_only_render"' in first
    assert artifact.comparison_fingerprint in first
    assert "authored-fixture-baseline@v1" in first
    assert "authored-fixture-candidate@v1" in first
    assert EVALUATION_COMPARISON_HTML_SCOPE in first
    assert "未重新评分或重跑统计" in first


def test_renderer_escapes_system_and_slice_text() -> None:
    artifact = load_evaluation_comparison_artifact(COMPARISON)
    content = artifact.identity_dict()
    attack = '</td><script src="https://attacker.invalid/x.js">alert(1)</script>'
    content["run_bindings"]["baseline"]["system_id"] = attack  # type: ignore[index]
    original_slice = next(iter(content["protected_slices"]))  # type: ignore[arg-type]
    result = content["protected_slices"].pop(original_slice)  # type: ignore[union-attr]
    content["protected_slices"][attack] = result  # type: ignore[index]
    content["gate_configuration"]["protected_slices"] = [attack]  # type: ignore[index]
    changed = EvaluationComparisonArtifact(content)

    rendered = render_evaluation_comparison_html(changed)

    assert attack not in rendered
    assert "<script" not in rendered.lower()
    assert "attacker.invalid" in rendered
    assert "&lt;script" in rendered
    assert "&quot;https://attacker.invalid/x.js&quot;" in rendered


def test_cluster_report_exposes_estimand_and_resampling_ledger() -> None:
    cases = [
        EvaluationCase(
            case_id=f"case-{index}",
            input="input",
            expected="expected",
            metadata={"user_id": "A" if index < 2 else "B"},
        )
        for index in range(3)
    ]
    baseline = [
        EvaluationResult(case.case_id, "", {"quality": 0.0}, 0.1)
        for case in cases
    ]
    candidate = [
        EvaluationResult(case.case_id, "", {"quality": 1.0}, 0.11)
        for case in cases
    ]
    comparison = compare_results(
        cases,
        baseline,
        candidate,
        quality_metric="quality",
        safety_metric=None,
        confidence=0.95,
        bootstrap_samples=100,
        seed=7,
        gate=ReleaseGate(),
        protected_slices=(),
        maximum_slice_regression=0.0,
        cluster_metadata_key="user_id",
        cluster_weighting="equal",
        cluster_exact_max=6,
    )
    artifact = build_evaluation_comparison_artifact(
        comparison=comparison,
        gate_configuration={
            "minimum_quality_difference": 0.0,
            "maximum_safety_regression": 0.0,
            "maximum_latency_increase_fraction": 0.1,
            "maximum_slice_regression": 0.0,
            "protected_slices": [],
        },
        run_bindings={
            "baseline": {
                "system_id": "baseline",
                "manifest_fingerprint": "sha256:" + "1" * 64,
            },
            "candidate": {
                "system_id": "candidate",
                "manifest_fingerprint": "sha256:" + "2" * 64,
            },
            "cases_fingerprint": "sha256:" + "3" * 64,
            "metric_revisions": {"quality": "quality.v1"},
        },
    )

    rendered = render_evaluation_comparison_html(artifact)

    assert "Cluster result:" in rendered
    assert "clusters=2" in rendered
    assert "sizes=<code>[2, 1]</code>" in rendered
    assert "weighting=<code>equal</code>" in rendered
    assert "method=<code>exact</code>" in rendered
    assert "resamples=4" in rendered


@pytest.mark.smoke
def test_render_comparison_html_cli_writes_report_and_scope_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "comparison.html"

    assert (
        main(
            [
                "render-comparison-html",
                "--input",
                str(COMPARISON),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    rendered = output.read_text(encoding="utf-8")

    assert receipt["verification_scope"] == "artifact_only_render"
    assert receipt["scripts_included"] is False
    assert receipt["external_resources_included"] is False
    assert receipt["statistics_recomputed"] is False
    assert receipt["artifact_authentication_verified"] is False
    assert receipt["comparison_fingerprint"] in rendered
