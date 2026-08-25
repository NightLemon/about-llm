from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.evaluation import (
    EvaluationCase,
    EvaluationComparisonArtifact,
    EvaluationResult,
    ReleaseGate,
    evaluation_results_fingerprint,
    load_evaluation_comparison_artifact,
    load_evaluation_run_manifest,
    load_results,
    write_evaluation_comparison_artifact,
    write_evaluation_run_manifest,
    write_results,
)
from about_llm.evaluation.cli import (
    compare_results,
    load_answers,
    load_calibration_records,
    main,
)

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "evaluation-gate"
CASES = PROJECT / "cases.example.jsonl"
BASELINE_ANSWERS = PROJECT / "answers.baseline.example.jsonl"
CANDIDATE_ANSWERS = PROJECT / "answers.candidate.example.jsonl"
BASELINE_RESULTS = PROJECT / "results.baseline.example.jsonl"
CANDIDATE_RESULTS = PROJECT / "results.candidate.example.jsonl"
BASELINE_MANIFEST = PROJECT / "run.baseline.manifest.example.json"
CANDIDATE_MANIFEST = PROJECT / "run.candidate.manifest.example.json"
COMPARISON = PROJECT / "comparison.example.json"
CALIBRATION_RECORDS = PROJECT / "calibration.example.jsonl"
CALIBRATION_MANIFEST = PROJECT / "calibration.manifest.example.json"
STRUCTURED_CASES = PROJECT / "structured-metrics.cases.jsonl"
STRUCTURED_ANSWERS = PROJECT / "structured-metrics.answers.jsonl"
CITATION_SPAN_CASES = PROJECT / "citation-evidence-span.cases.jsonl"
CITATION_SPAN_ANSWERS = PROJECT / "citation-evidence-span.answers.jsonl"


@pytest.mark.parametrize(
    ("protected_slices", "maximum_slice_regression", "message"),
    [
        ([], -0.01, "maximum_slice_regression"),
        (["critical", "critical"], 0.0, "must not contain duplicates"),
    ],
)
def test_compare_results_rejects_invalid_slice_gate_configuration(
    protected_slices: list[str],
    maximum_slice_regression: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_results(
            [],
            [],
            [],
            quality_metric="quality",
            safety_metric=None,
            confidence=0.95,
            bootstrap_samples=100,
            seed=0,
            gate=ReleaseGate(),
            protected_slices=protected_slices,
            maximum_slice_regression=maximum_slice_regression,
        )


def test_compare_results_rejects_string_slice_collection() -> None:
    with pytest.raises(ValueError, match="sequence of slice names"):
        compare_results(
            [],
            [],
            [],
            quality_metric="quality",
            safety_metric=None,
            confidence=0.95,
            bootstrap_samples=100,
            seed=0,
            gate=ReleaseGate(),
            protected_slices="critical",  # type: ignore[arg-type]
            maximum_slice_regression=0.0,
        )


def _score(answers: Path, results: Path, report: Path, system_id: str) -> Path:
    manifest = results.with_suffix(".manifest.json")
    assert (
        main(
            [
                "score",
                "--cases",
                str(CASES),
                "--answers",
                str(answers),
                "--results",
                str(results),
                "--report",
                str(report),
                "--manifest",
                str(manifest),
                "--system-id",
                system_id,
            ]
        )
        == 0
    )
    return manifest


def _verify_evidence_args(
    *,
    baseline_answers: Path = BASELINE_ANSWERS,
    baseline_results: Path = BASELINE_RESULTS,
    baseline_manifest: Path = BASELINE_MANIFEST,
    comparison: Path = COMPARISON,
) -> list[str]:
    return [
        "verify-evidence",
        "--cases",
        str(CASES),
        "--baseline-answers",
        str(baseline_answers),
        "--candidate-answers",
        str(CANDIDATE_ANSWERS),
        "--baseline-results",
        str(baseline_results),
        "--candidate-results",
        str(CANDIDATE_RESULTS),
        "--baseline-manifest",
        str(baseline_manifest),
        "--candidate-manifest",
        str(CANDIDATE_MANIFEST),
        "--comparison",
        str(comparison),
    ]


def test_verify_evidence_recomputes_complete_local_graph(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_verify_evidence_args()) == 0

    verification = json.loads(capsys.readouterr().out)
    assert verification["verification_scope"] == "full_local_recomputation"
    assert verification["case_semantics_rehashed"] is True
    assert verification["recorded_answers_rehashed"] is True
    assert verification["scores_recomputed"] is True
    assert verification["run_manifests_revalidated"] is True
    assert verification["statistics_recomputed"] is True
    assert verification["comparison_rebuilt"] is True
    assert verification["artifact_authentication_verified"] is False
    assert verification["model_execution_replayed"] is False
    assert verification["comparison_fingerprint"] == (
        "sha256:999e29b9d9fae5e37a3d8e680711e4cb79be222af859a35e9d41083ba587b18a"
    )


def test_verify_evidence_rejects_recorded_answer_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changed_answers = tmp_path / "answers.jsonl"
    rows = [
        json.loads(line)
        for line in BASELINE_ANSWERS.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["output"] = "rewritten answer"
    changed_answers.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main(_verify_evidence_args(baseline_answers=changed_answers))

    assert error.value.code == 2
    assert "recorded answers do not match" in capsys.readouterr().err


def test_verify_evidence_rejects_self_consistent_manifest_with_wrong_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changed_results = list(load_results(BASELINE_RESULTS))
    changed_results[0] = replace(
        changed_results[0],
        scores={**changed_results[0].scores, "exact_match": 1.0},
    )
    results_path = tmp_path / "results.jsonl"
    write_results(results_path, changed_results)
    changed_manifest = replace(
        load_evaluation_run_manifest(BASELINE_MANIFEST),
        results_fingerprint=evaluation_results_fingerprint(changed_results),
    )
    manifest_path = tmp_path / "manifest.json"
    write_evaluation_run_manifest(manifest_path, changed_manifest)

    with pytest.raises(SystemExit) as error:
        main(
            _verify_evidence_args(
                baseline_results=results_path,
                baseline_manifest=manifest_path,
            )
        )

    assert error.value.code == 2
    assert "recomputed scores do not match" in capsys.readouterr().err


def test_verify_evidence_rejects_self_consistent_comparison_summary_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = load_evaluation_comparison_artifact(COMPARISON).identity_dict()
    content["baseline_mean_latency_seconds"] = 0.12
    changed = EvaluationComparisonArtifact(content)
    comparison_path = tmp_path / "comparison.json"
    write_evaluation_comparison_artifact(comparison_path, changed)

    with pytest.raises(SystemExit) as error:
        main(_verify_evidence_args(comparison=comparison_path))

    assert error.value.code == 2
    assert "recomputed comparison does not match" in capsys.readouterr().err


def test_verify_evidence_requires_current_executable_metric_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = load_evaluation_run_manifest(BASELINE_MANIFEST)
    changed = replace(
        manifest,
        metric_revisions={
            **manifest.metric_revisions,
            "exact_match": "unavailable-exact-match.v999",
        },
    )
    manifest_path = tmp_path / "manifest.json"
    write_evaluation_run_manifest(manifest_path, changed)

    with pytest.raises(SystemExit) as error:
        main(_verify_evidence_args(baseline_manifest=manifest_path))

    assert error.value.code == 2
    assert "metric revision mismatch" in capsys.readouterr().err


def test_verify_evidence_recomputes_cluster_exact_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    comparison_path = tmp_path / "cluster-comparison.json"
    assert (
        main(
            [
                "compare",
                "--cases",
                str(CASES),
                "--baseline-results",
                str(BASELINE_RESULTS),
                "--candidate-results",
                str(CANDIDATE_RESULTS),
                "--baseline-manifest",
                str(BASELINE_MANIFEST),
                "--candidate-manifest",
                str(CANDIDATE_MANIFEST),
                "--quality-metric",
                "exact_match",
                "--cluster-metadata-key",
                "risk",
                "--cluster-weighting",
                "equal",
                "--cluster-exact-max",
                "6",
                "--output",
                str(comparison_path),
            ]
        )
        == 0
    )
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["bootstrap"]["unit"] == "cluster"
    assert comparison["quality"]["method"] == "exact"

    assert main(_verify_evidence_args(comparison=comparison_path)) == 0
    verification = json.loads(capsys.readouterr().out)
    assert verification["statistics_recomputed"] is True
    assert verification["comparison_fingerprint"] == comparison["comparison_fingerprint"]


@pytest.mark.smoke
def test_score_cli_writes_results_and_slice_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "candidate.results.jsonl"
    report = tmp_path / "candidate.report.md"

    manifest_path = _score(
        CANDIDATE_ANSWERS, results, report, "fixture-candidate@v1"
    )

    loaded = load_results(results)
    assert len(loaded) == 2
    assert all(result.scores["exact_match"] == 1 for result in loaded)
    assert "| overall | 2 | 0.000 | 1.000 | 1.000 |" in report.read_text(
        encoding="utf-8"
    )
    assert "# Evaluation report" in capsys.readouterr().out
    manifest = load_evaluation_run_manifest(manifest_path)
    assert manifest.system_id == "fixture-candidate@v1"
    assert set(manifest.metric_revisions) == {"exact_match", "token_f1"}


@pytest.mark.smoke
def test_score_cli_exposes_literal_exact_as_opt_in_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = tmp_path / "literal.cases.jsonl"
    answers = tmp_path / "literal.answers.jsonl"
    results = tmp_path / "literal.results.jsonl"
    report = tmp_path / "literal.report.md"
    manifest = tmp_path / "literal.manifest.json"
    cases.write_text(
        json.dumps(
            {
                "case_id": "case-sensitive-id",
                "input": "copy",
                "expected": "LLM-2026",
                "slices": ["instruction"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps(
            {
                "case_id": "case-sensitive-id",
                "output": "llm-2026",
                "latency_seconds": 0.01,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "score",
                "--cases",
                str(cases),
                "--answers",
                str(answers),
                "--results",
                str(results),
                "--report",
                str(report),
                "--manifest",
                str(manifest),
                "--system-id",
                "literal-fixture@v1",
                "--metric",
                "literal_exact_match",
                "--metric",
                "exact_match",
            ]
        )
        == 0
    )
    capsys.readouterr()

    loaded = load_results(results)
    assert loaded[0].scores == {
        "literal_exact_match": 0.0,
        "exact_match": 1.0,
    }
    run_manifest = load_evaluation_run_manifest(manifest)
    assert run_manifest.metric_revisions == {
        "literal_exact_match": "about-llm.literal-exact-match.v1",
        "exact_match": "about-llm.normalized-exact-match.v1",
    }


@pytest.mark.smoke
def test_score_cli_distinguishes_strict_schema_from_json_value_equality(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "structured.results.jsonl"
    report = tmp_path / "structured.report.md"
    manifest = tmp_path / "structured.manifest.json"

    assert (
        main(
            [
                "score",
                "--cases",
                str(STRUCTURED_CASES),
                "--answers",
                str(STRUCTURED_ANSWERS),
                "--results",
                str(results),
                "--report",
                str(report),
                "--manifest",
                str(manifest),
                "--system-id",
                "authored-structured-fixture@v1",
                "--metric",
                "json_schema",
                "--metric",
                "json_value_exact",
            ]
        )
        == 0
    )
    capsys.readouterr()

    loaded = load_results(results)
    assert [row.scores["json_schema"] for row in loaded] == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert [row.scores["json_value_exact"] for row in loaded] == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    run_manifest = load_evaluation_run_manifest(manifest)
    assert run_manifest.metric_revisions == {
        "json_schema": "about-llm.json-schema-metric.v2",
        "json_value_exact": "about-llm.json-value-exact.v1",
    }


@pytest.mark.smoke
def test_score_cli_keeps_evidence_span_identity_separate_from_entailment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "citation-span.results.jsonl"
    report = tmp_path / "citation-span.report.md"
    manifest = tmp_path / "citation-span.manifest.json"

    assert (
        main(
            [
                "score",
                "--cases",
                str(CITATION_SPAN_CASES),
                "--answers",
                str(CITATION_SPAN_ANSWERS),
                "--results",
                str(results),
                "--report",
                str(report),
                "--manifest",
                str(manifest),
                "--system-id",
                "authored-citation-span-fixture@v1",
                "--metric",
                "citation_evidence_span",
            ]
        )
        == 0
    )
    capsys.readouterr()

    loaded = load_results(results)
    assert [row.scores["citation_evidence_span"] for row in loaded] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    run_manifest = load_evaluation_run_manifest(manifest)
    assert run_manifest.metric_revisions == {
        "citation_evidence_span": (
            "about-llm.citation-evidence-span-metric.v1"
        )
    }


@pytest.mark.smoke
def test_compare_cli_runs_paired_gate_and_protected_slice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline.results.jsonl"
    candidate = tmp_path / "candidate.results.jsonl"
    baseline_manifest = _score(
        BASELINE_ANSWERS, baseline, tmp_path / "baseline.md", "fixture-baseline@v1"
    )
    candidate_manifest = _score(
        CANDIDATE_ANSWERS,
        candidate,
        tmp_path / "candidate.md",
        "fixture-candidate@v1",
    )
    capsys.readouterr()
    gate_path = tmp_path / "gate.json"

    exit_code = main(
        [
            "compare",
            "--cases",
            str(CASES),
            "--baseline-results",
            str(baseline),
            "--candidate-results",
            str(candidate),
            "--baseline-manifest",
            str(baseline_manifest),
            "--candidate-manifest",
            str(candidate_manifest),
            "--quality-metric",
            "exact_match",
            "--protected-slice",
            "zh",
            "--bootstrap-samples",
            "500",
            "--seed",
            "7",
            "--output",
            str(gate_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["quality"]["baseline_mean"] == pytest.approx(0.0)
    assert payload["quality"]["candidate_mean"] == pytest.approx(1.0)
    assert payload["quality"]["confidence_low"] == pytest.approx(1.0)
    assert payload["protected_slices"]["zh"]["mean_difference"] == pytest.approx(1.0)
    assert payload["baseline_mean_latency_seconds"] == pytest.approx(0.11)
    assert payload["candidate_mean_latency_seconds"] == pytest.approx(0.115)
    assert payload["safety_metric"] is None
    assert payload["run_bindings"]["baseline"]["system_id"] == "fixture-baseline@v1"
    assert payload["run_bindings"]["candidate"]["system_id"] == "fixture-candidate@v1"
    assert "does not authenticate" in payload["evidence_boundary"]
    assert payload["comparison_version"] == "about-llm.evaluation-comparison.v2"
    assert payload["bootstrap"]["unit"] == "case"
    assert payload["bootstrap"]["cluster_metadata_key"] is None
    assert payload["comparison_fingerprint"].startswith("sha256:")
    assert payload["gate_configuration"]["protected_slices"] == ["zh"]
    assert json.loads(gate_path.read_text(encoding="utf-8")) == payload

    assert main(["verify-comparison", "--input", str(gate_path)]) == 0
    verification = json.loads(capsys.readouterr().out)
    assert verification["valid"] is True
    assert verification["verification_scope"] == "artifact_only"
    assert verification["referenced_manifests_revalidated"] is False
    assert verification["statistics_recomputed"] is False
    assert verification["comparison_fingerprint"] == payload["comparison_fingerprint"]
    assert verification["baseline_system_id"] == "fixture-baseline@v1"

    cluster_exit = main(
        [
            "compare",
            "--cases",
            str(CASES),
            "--baseline-results",
            str(baseline),
            "--candidate-results",
            str(candidate),
            "--baseline-manifest",
            str(baseline_manifest),
            "--candidate-manifest",
            str(candidate_manifest),
            "--quality-metric",
            "exact_match",
            "--cluster-metadata-key",
            "risk",
            "--cluster-weighting",
            "equal",
            "--cluster-exact-max",
            "6",
        ]
    )
    cluster_payload = json.loads(capsys.readouterr().out)
    assert cluster_exit == 0
    assert cluster_payload["bootstrap"]["unit"] == "cluster"
    assert cluster_payload["bootstrap"]["cluster_metadata_key"] == "risk"
    assert cluster_payload["quality"]["cluster_count"] == 1
    assert cluster_payload["quality"]["cluster_sizes"] == [2]
    assert cluster_payload["quality"]["method"] == "exact"
    assert cluster_payload["quality"]["resamples_evaluated"] == 1

    failed_exit = main(
        [
            "compare",
            "--cases",
            str(CASES),
            "--baseline-results",
            str(candidate),
            "--candidate-results",
            str(baseline),
            "--baseline-manifest",
            str(candidate_manifest),
            "--candidate-manifest",
            str(baseline_manifest),
            "--quality-metric",
            "exact_match",
            "--bootstrap-samples",
            "200",
        ]
    )
    failed_payload = json.loads(capsys.readouterr().out)
    assert failed_exit == 1
    assert failed_payload["passed"] is False
    assert failed_payload["reasons"]


def test_compare_results_integrates_exact_cluster_bootstrap_estimands() -> None:
    cases = [
        EvaluationCase(
            case_id=f"case-{index}",
            input="input",
            expected="expected",
            metadata={"user_id": "A" if index < 5 else "B"},
        )
        for index in range(6)
    ]
    baseline = [
        EvaluationResult(
            case_id=case.case_id,
            output="",
            scores={
                "quality": 0.0 if index < 5 else 1.0,
                "safety": 0.0 if index < 5 else 1.0,
            },
            latency_seconds=0.1,
        )
        for index, case in enumerate(cases)
    ]
    candidate = [
        EvaluationResult(
            case_id=case.case_id,
            output="",
            scores={
                "quality": 1.0 if index < 5 else 0.0,
                "safety": 1.0 if index < 5 else 0.0,
            },
            latency_seconds=0.1,
        )
        for index, case in enumerate(cases)
    ]

    case_weighted = compare_results(
        cases,
        baseline,
        candidate,
        quality_metric="quality",
        safety_metric="safety",
        confidence=0.95,
        bootstrap_samples=1000,
        seed=7,
        gate=ReleaseGate(minimum_quality_difference=-1.0),
        protected_slices=[],
        maximum_slice_regression=0.0,
        cluster_metadata_key="user_id",
        cluster_weighting="case",
        cluster_exact_max=6,
    )
    equal_cluster = compare_results(
        cases,
        baseline,
        candidate,
        quality_metric="quality",
        safety_metric="safety",
        confidence=0.95,
        bootstrap_samples=1000,
        seed=7,
        gate=ReleaseGate(minimum_quality_difference=-1.0),
        protected_slices=[],
        maximum_slice_regression=0.0,
        cluster_metadata_key="user_id",
        cluster_weighting="equal",
        cluster_exact_max=6,
    )

    assert case_weighted["bootstrap"] == {
        "unit": "cluster",
        "confidence": 0.95,
        "samples": 1000,
        "seed": 7,
        "cluster_metadata_key": "user_id",
        "cluster_weighting": "case",
        "exact_max_clusters": 6,
    }
    assert case_weighted["quality"]["cluster_sizes"] == (5, 1)
    assert case_weighted["quality"]["method"] == "exact"
    assert case_weighted["quality"]["resamples_evaluated"] == 4
    assert case_weighted["quality"]["mean_difference"] == pytest.approx(2 / 3)
    assert case_weighted["quality"]["confidence_low"] == pytest.approx(-0.875)
    assert case_weighted["quality"]["confidence_high"] == pytest.approx(0.975)
    assert case_weighted["safety_difference"] == pytest.approx(2 / 3)
    assert equal_cluster["quality"]["mean_difference"] == pytest.approx(0.0)
    assert equal_cluster["quality"]["confidence_low"] == pytest.approx(-0.925)
    assert equal_cluster["quality"]["confidence_high"] == pytest.approx(0.925)
    assert equal_cluster["safety_difference"] == pytest.approx(0.0)


def test_compare_cli_rejects_same_ids_with_changed_case_semantics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline.results.jsonl"
    candidate = tmp_path / "candidate.results.jsonl"
    baseline_manifest = _score(
        BASELINE_ANSWERS, baseline, tmp_path / "baseline.md", "baseline@v1"
    )
    candidate_manifest = _score(
        CANDIDATE_ANSWERS, candidate, tmp_path / "candidate.md", "candidate@v1"
    )
    changed_cases = tmp_path / "changed-cases.jsonl"
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines()]
    rows[0]["expected"] = "silently changed expected answer"
    changed_cases.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        main(
            [
                "compare",
                "--cases",
                str(changed_cases),
                "--baseline-results",
                str(baseline),
                "--candidate-results",
                str(candidate),
                "--baseline-manifest",
                str(baseline_manifest),
                "--candidate-manifest",
                str(candidate_manifest),
                "--quality-metric",
                "exact_match",
            ]
        )

    assert error.value.code == 2
    assert "cases_fingerprint" in capsys.readouterr().err

    incompatible_manifest = tmp_path / "candidate.incompatible-manifest.json"
    candidate_run = load_evaluation_run_manifest(candidate_manifest)
    write_evaluation_run_manifest(
        incompatible_manifest,
        replace(
            candidate_run,
            metric_revisions={
                **candidate_run.metric_revisions,
                "exact_match": "different-exact-match.v2",
            },
        ),
    )
    with pytest.raises(SystemExit) as revision_error:
        main(
            [
                "compare",
                "--cases",
                str(CASES),
                "--baseline-results",
                str(baseline),
                "--candidate-results",
                str(candidate),
                "--baseline-manifest",
                str(baseline_manifest),
                "--candidate-manifest",
                str(incompatible_manifest),
                "--quality-metric",
                "exact_match",
            ]
        )
    assert revision_error.value.code == 2
    assert "metric revision mismatch" in capsys.readouterr().err


def test_load_results_rejects_non_finite_latency(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "bad",
                "output": "x",
                "scores": {"quality": 1},
                "latency_seconds": float("nan"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_results(path)


@pytest.mark.smoke
def test_calibrate_cli_writes_brier_ece_and_risk_coverage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "calibration.json"

    assert (
        main(
            [
                "calibrate",
                "--input",
                str(CALIBRATION_RECORDS),
                "--bins",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["calibration"]["count"] == 8
    assert 0 <= saved["calibration"]["brier_score"] <= 1
    assert 0 <= saved["calibration"]["expected_calibration_error"] <= 1
    assert saved["risk_coverage"][-1]["coverage"] == 1
    assert saved["semantics"]["risk"] == "error rate among records at or above threshold"


def test_calibrate_cli_rejects_duplicate_case_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"case_id":"x","label":1,"probability":0.9}\n'
        '{"case_id":"x","label":0,"probability":0.1}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main(["calibrate", "--input", str(path)])
    assert error.value.code == 2
    assert "duplicate case_id" in capsys.readouterr().err


def test_answer_and_calibration_loaders_reject_strict_json_drift(
    tmp_path: Path,
) -> None:
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        '{"case_id":"q","output":"x","latency_seconds":0.1,"typo":true}\n',
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        '{"case_id":"q","label":1,"probability":0.9,"probability":0.8}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unknown=\['typo'\]"):
        load_answers(answers)
    with pytest.raises(ValueError, match="duplicate JSON object key 'probability'"):
        load_calibration_records(calibration)


def test_calibration_fixture_manifest_matches_input_artifact() -> None:
    manifest = json.loads(CALIBRATION_MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(CALIBRATION_RECORDS.read_bytes()).hexdigest()

    assert manifest["fixture_only"] is True
    assert manifest["input"]["sha256"] == digest
    assert manifest["input"]["case_count"] == len(
        CALIBRATION_RECORDS.read_text(encoding="utf-8").splitlines()
    )
    assert "does not prove" in manifest["evidence_boundary"]
