from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.evaluation import (
    EvaluationCase,
    EvaluationResult,
    EvaluationRunManifest,
    build_evaluation_run_manifest,
    evaluation_cases_fingerprint,
    load_cases,
    load_evaluation_run_manifest,
    load_results,
    validate_evaluation_run_manifest,
    write_evaluation_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "evaluation-gate"


def _fixture() -> tuple[
    tuple[EvaluationCase, ...],
    tuple[EvaluationResult, ...],
]:
    cases = (
        EvaluationCase(
            case_id="q1",
            input="question one",
            expected="answer one",
            slices=("critical",),
            metadata={"risk": "high"},
        ),
        EvaluationCase(
            case_id="q2",
            input="question two",
            expected="answer two",
            metadata={"risk": "low"},
        ),
    )
    results = (
        EvaluationResult("q1", "answer one", {"quality": 1.0}, 0.1),
        EvaluationResult("q2", "wrong", {"quality": 0.0}, 0.2),
    )
    return cases, results


def _manifest() -> tuple[
    tuple[EvaluationCase, ...],
    tuple[EvaluationResult, ...],
    EvaluationRunManifest,
]:
    cases, results = _fixture()
    manifest = build_evaluation_run_manifest(
        system_id="candidate@immutable-revision",
        cases=cases,
        results=results,
        recorded_answers_fingerprint="sha256:" + "1" * 64,
        metric_revisions={"quality": "fixture-quality.v1"},
        metadata={"scorer_revision": "fixture-scorer.v1"},
    )
    return cases, results, manifest


def test_manifest_round_trip_and_current_artifacts_validate(tmp_path: Path) -> None:
    cases, results, manifest = _manifest()
    path = tmp_path / "run.manifest.json"
    write_evaluation_run_manifest(path, manifest)

    loaded = load_evaluation_run_manifest(path)
    validate_evaluation_run_manifest(
        loaded,
        cases=cases,
        results=results,
        required_metrics=("quality",),
    )

    assert loaded.system_id == "candidate@immutable-revision"
    assert loaded.manifest_fingerprint.startswith("sha256:")
    assert loaded.to_dict() == manifest.to_dict()


@pytest.mark.parametrize("name", ["baseline", "candidate"])
def test_project_run_manifest_fixtures_bind_current_cases_and_results(name: str) -> None:
    cases = load_cases(PROJECT / "cases.example.jsonl")
    results = load_results(PROJECT / f"results.{name}.example.jsonl")
    manifest = load_evaluation_run_manifest(
        PROJECT / f"run.{name}.manifest.example.json"
    )

    validate_evaluation_run_manifest(
        manifest,
        cases=cases,
        results=results,
        required_metrics=("exact_match", "token_f1"),
    )
    assert manifest.system_id == f"authored-fixture-{name}@v1"


def test_same_case_ids_with_changed_semantics_fail_closed() -> None:
    cases, results, manifest = _manifest()
    changed_cases = (replace(cases[0], expected="silently changed gold"), cases[1])

    with pytest.raises(ValueError, match="cases_fingerprint"):
        validate_evaluation_run_manifest(
            manifest,
            cases=changed_cases,
            results=results,
            required_metrics=("quality",),
        )
    assert evaluation_cases_fingerprint(changed_cases) != evaluation_cases_fingerprint(cases)


def test_result_or_metric_revision_drift_fails_closed() -> None:
    cases, results, manifest = _manifest()
    changed_results = (replace(results[0], scores={"quality": 0.0}), results[1])

    with pytest.raises(ValueError, match="results_fingerprint"):
        validate_evaluation_run_manifest(
            manifest,
            cases=cases,
            results=changed_results,
            required_metrics=("quality",),
        )
    with pytest.raises(ValueError, match="missing metric revisions"):
        validate_evaluation_run_manifest(
            manifest,
            cases=cases,
            results=results,
            required_metrics=("safety",),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown", r"unknown=\['unexpected'\]"),
        ("version", "expected manifest_version"),
        ("fingerprint", "manifest_fingerprint does not match"),
    ],
)
def test_manifest_loader_rejects_schema_version_and_identity_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _, _, manifest = _manifest()
    payload = manifest.to_dict()
    if mutation == "unknown":
        payload["unexpected"] = True
    elif mutation == "version":
        payload["manifest_version"] = "about-llm.evaluation-run-manifest.v999"
    else:
        payload["system_id"] = "tampered-system"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluation_run_manifest(path)


@pytest.mark.parametrize(
    "bad_json",
    [
        '{"manifest_version":"v1","manifest_version":"v2"}',
        '{"manifest_version":NaN}',
        '{"manifest_version":Infinity}',
    ],
)
def test_manifest_loader_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path, bad_json: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(bad_json, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid strict JSON"):
        load_evaluation_run_manifest(path)
