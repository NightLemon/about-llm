from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.evaluation import (
    EVALUATION_COMPARISON_EVIDENCE_BOUNDARY,
    EvaluationComparisonArtifact,
    build_evaluation_comparison_artifact,
    load_evaluation_comparison_artifact,
    write_evaluation_comparison_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "evaluation-gate"


def _artifact() -> EvaluationComparisonArtifact:
    bootstrap_result = {
        "baseline_mean": 0.5,
        "candidate_mean": 0.75,
        "mean_difference": 0.25,
        "confidence_low": 0.1,
        "confidence_high": 0.4,
        "probability_of_improvement": 0.98,
    }
    comparison = {
        "passed": True,
        "reasons": [],
        "quality_metric": "quality",
        "quality": bootstrap_result,
        "safety_metric": None,
        "safety_difference": 0.0,
        "baseline_mean_latency_seconds": 0.1,
        "candidate_mean_latency_seconds": 0.11,
        "protected_slices": {"critical": bootstrap_result},
        "case_count": 4,
        "bootstrap": {
            "unit": "case",
            "confidence": 0.95,
            "samples": 1000,
            "seed": 7,
            "cluster_metadata_key": None,
            "cluster_weighting": None,
            "exact_max_clusters": None,
        },
    }
    return build_evaluation_comparison_artifact(
        comparison=comparison,
        gate_configuration={
            "minimum_quality_difference": 0.0,
            "maximum_safety_regression": 0.0,
            "maximum_latency_increase_fraction": 0.2,
            "maximum_slice_regression": 0.0,
            "protected_slices": ["critical"],
        },
        run_bindings={
            "baseline": {
                "system_id": "baseline@v1",
                "manifest_fingerprint": "sha256:" + "1" * 64,
            },
            "candidate": {
                "system_id": "candidate@v2",
                "manifest_fingerprint": "sha256:" + "2" * 64,
            },
            "cases_fingerprint": "sha256:" + "3" * 64,
            "metric_revisions": {"quality": "quality-metric.v1"},
        },
    )


def test_comparison_round_trip_binds_gate_configuration(tmp_path: Path) -> None:
    artifact = _artifact()
    path = tmp_path / "comparison.json"
    write_evaluation_comparison_artifact(path, artifact)

    loaded = load_evaluation_comparison_artifact(path)

    assert loaded.to_dict() == artifact.to_dict()
    assert loaded.passed is True
    assert loaded.baseline_system_id == "baseline@v1"
    assert loaded.candidate_system_id == "candidate@v2"
    assert loaded.content["evidence_boundary"] == EVALUATION_COMPARISON_EVIDENCE_BOUNDARY


def test_project_comparison_fixture_is_strictly_reloadable() -> None:
    artifact = load_evaluation_comparison_artifact(PROJECT / "comparison.example.json")

    assert artifact.passed is True
    assert artifact.baseline_system_id == "authored-fixture-baseline@v1"
    assert artifact.candidate_system_id == "authored-fixture-candidate@v1"
    assert artifact.content["gate_configuration"]["protected_slices"] == ("zh",)  # type: ignore[index]


def test_cluster_comparison_round_trip_binds_resampling_contract(
    tmp_path: Path,
) -> None:
    clustered = {
        "case_count": 4,
        "cluster_count": 2,
        "cluster_sizes": [3, 1],
        "cluster_weighting": "case",
        "baseline_estimand": 0.25,
        "candidate_estimand": 0.5,
        "mean_difference": 0.25,
        "confidence": 0.95,
        "confidence_low": -0.5,
        "confidence_high": 1.0,
        "probability_of_improvement": 0.75,
        "method": "exact",
        "resamples_evaluated": 4,
        "quantile_method": "linear",
        "seed": None,
    }
    base = _artifact().identity_dict()
    base["quality"] = clustered
    protected_clustered = {**clustered, "confidence_low": 0.1}
    base["protected_slices"] = {"critical": protected_clustered}
    base["bootstrap"] = {
        "unit": "cluster",
        "confidence": 0.95,
        "samples": 1000,
        "seed": 7,
        "cluster_metadata_key": "user_id",
        "cluster_weighting": "case",
        "exact_max_clusters": 6,
    }
    base["passed"] = False
    base["reasons"] = ["quality confidence lower bound -0.5000 < 0.0000"]
    artifact = EvaluationComparisonArtifact(base)
    path = tmp_path / "clustered.json"
    write_evaluation_comparison_artifact(path, artifact)

    loaded = load_evaluation_comparison_artifact(path)

    assert loaded.to_dict() == artifact.to_dict()
    assert loaded.content["bootstrap"]["unit"] == "cluster"  # type: ignore[index]
    assert loaded.content["quality"]["cluster_sizes"] == (3, 1)  # type: ignore[index]


def test_cluster_comparison_records_monte_carlo_samples_and_seed() -> None:
    clustered = {
        "case_count": 4,
        "cluster_count": 2,
        "cluster_sizes": [3, 1],
        "cluster_weighting": "equal",
        "baseline_estimand": 0.25,
        "candidate_estimand": 0.5,
        "mean_difference": 0.25,
        "confidence": 0.95,
        "confidence_low": -0.5,
        "confidence_high": 1.0,
        "probability_of_improvement": 0.7,
        "method": "monte_carlo",
        "resamples_evaluated": 17,
        "quantile_method": "linear",
        "seed": 7,
    }
    base = _artifact().identity_dict()
    base["quality"] = clustered
    base["protected_slices"] = {
        "critical": {**clustered, "confidence_low": 0.1}
    }
    base["bootstrap"] = {
        "unit": "cluster",
        "confidence": 0.95,
        "samples": 17,
        "seed": 7,
        "cluster_metadata_key": "user_id",
        "cluster_weighting": "equal",
        "exact_max_clusters": 0,
    }
    base["passed"] = False
    base["reasons"] = ["quality confidence lower bound -0.5000 < 0.0000"]

    artifact = EvaluationComparisonArtifact(base)

    assert artifact.content["quality"]["method"] == "monte_carlo"  # type: ignore[index]
    assert artifact.content["quality"]["resamples_evaluated"] == 17  # type: ignore[index]
    assert artifact.content["quality"]["seed"] == 7  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("sizes", "cluster_sizes must sum to case_count"),
        ("weighting", "cluster_weighting must match"),
        ("resamples", r"cluster_count\*\*cluster_count"),
        ("seed", "seed must be null"),
        ("probability", "exact resample fraction"),
    ],
)
def test_cluster_comparison_rejects_internal_resampling_inconsistency(
    mutation: str, message: str
) -> None:
    clustered = {
        "case_count": 4,
        "cluster_count": 2,
        "cluster_sizes": [3, 1],
        "cluster_weighting": "case",
        "baseline_estimand": 0.25,
        "candidate_estimand": 0.5,
        "mean_difference": 0.25,
        "confidence": 0.95,
        "confidence_low": -0.5,
        "confidence_high": 1.0,
        "probability_of_improvement": 0.75,
        "method": "exact",
        "resamples_evaluated": 4,
        "quantile_method": "linear",
        "seed": None,
    }
    base = _artifact().identity_dict()
    base["quality"] = clustered
    protected_clustered = {**clustered, "confidence_low": 0.1}
    base["protected_slices"] = {"critical": protected_clustered}
    base["bootstrap"] = {
        "unit": "cluster",
        "confidence": 0.95,
        "samples": 1000,
        "seed": 7,
        "cluster_metadata_key": "user_id",
        "cluster_weighting": "case",
        "exact_max_clusters": 6,
    }
    base["passed"] = False
    base["reasons"] = ["quality confidence lower bound -0.5000 < 0.0000"]
    target = base["quality"]
    if mutation == "sizes":
        target["cluster_sizes"] = [2, 1]  # type: ignore[index]
    elif mutation == "weighting":
        target["cluster_weighting"] = "equal"  # type: ignore[index]
    elif mutation == "resamples":
        target["resamples_evaluated"] = 3  # type: ignore[index]
    elif mutation == "seed":
        target["seed"] = 7  # type: ignore[index]
    else:
        target["probability_of_improvement"] = 0.7  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        EvaluationComparisonArtifact(base)


def test_cluster_comparison_binds_overall_case_count() -> None:
    clustered = {
        "case_count": 4,
        "cluster_count": 2,
        "cluster_sizes": [3, 1],
        "cluster_weighting": "case",
        "baseline_estimand": 0.25,
        "candidate_estimand": 0.5,
        "mean_difference": 0.25,
        "confidence": 0.95,
        "confidence_low": 0.1,
        "confidence_high": 1.0,
        "probability_of_improvement": 0.75,
        "method": "exact",
        "resamples_evaluated": 4,
        "quantile_method": "linear",
        "seed": None,
    }
    base = _artifact().identity_dict()
    base["quality"] = clustered
    base["protected_slices"] = {"critical": clustered.copy()}
    base["bootstrap"] = {
        "unit": "cluster",
        "confidence": 0.95,
        "samples": 1000,
        "seed": 7,
        "cluster_metadata_key": "user_id",
        "cluster_weighting": "case",
        "exact_max_clusters": 6,
    }
    base["case_count"] = 5

    with pytest.raises(ValueError, match=r"quality\.case_count must match"):
        EvaluationComparisonArtifact(base)


def test_gate_threshold_tampering_invalidates_existing_fingerprint(tmp_path: Path) -> None:
    payload = _artifact().to_dict()
    payload["gate_configuration"]["minimum_quality_difference"] = 0.3  # type: ignore[index]
    payload["passed"] = False
    payload["reasons"] = ["quality confidence lower bound 0.1000 < 0.3000"]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="comparison_fingerprint does not match"):
        load_evaluation_comparison_artifact(path)


def test_self_consistent_rewrite_changes_identity_but_is_not_authenticated() -> None:
    original = _artifact()
    changed_content = original.identity_dict()
    changed_content["gate_configuration"]["minimum_quality_difference"] = 0.3  # type: ignore[index]
    changed_content["passed"] = False
    changed_content["reasons"] = [
        "quality confidence lower bound 0.1000 < 0.3000"
    ]
    rewritten = EvaluationComparisonArtifact(changed_content)

    assert rewritten.comparison_fingerprint != original.comparison_fingerprint
    assert "does not authenticate system_id" in EVALUATION_COMPARISON_EVIDENCE_BOUNDARY


def test_threshold_rewrite_without_matching_decision_is_rejected() -> None:
    changed_content = _artifact().identity_dict()
    changed_content["gate_configuration"]["minimum_quality_difference"] = 0.3  # type: ignore[index]

    with pytest.raises(ValueError, match="comparison reasons do not match"):
        EvaluationComparisonArtifact(changed_content)


def test_comparison_rejects_arithmetically_inconsistent_bootstrap_summary() -> None:
    changed_content = _artifact().identity_dict()
    changed_content["quality"]["mean_difference"] = 0.2  # type: ignore[index]

    with pytest.raises(ValueError, match="must equal candidate_mean - baseline_mean"):
        EvaluationComparisonArtifact(changed_content)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_safety_regression", -0.01),
        ("maximum_slice_regression", -0.01),
        ("maximum_latency_increase_fraction", -1.0),
    ],
)
def test_comparison_rejects_semantically_invalid_gate_thresholds(
    field: str, value: float
) -> None:
    changed_content = _artifact().identity_dict()
    changed_content["gate_configuration"][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=f"gate_configuration.{field}"):
        EvaluationComparisonArtifact(changed_content)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown", r"unknown=\['unexpected'\]"),
        ("version", "comparison_version must equal"),
        ("boundary", "evidence_boundary does not match"),
    ],
)
def test_comparison_loader_rejects_schema_version_and_boundary_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    payload = _artifact().to_dict()
    if mutation == "unknown":
        payload["unexpected"] = True
    elif mutation == "version":
        payload["comparison_version"] = "about-llm.evaluation-comparison.v999"
    else:
        payload["evidence_boundary"] = "overclaimed"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluation_comparison_artifact(path)


@pytest.mark.parametrize(
    "bad_json",
    [
        '{"comparison_version":"v1","comparison_version":"v2"}',
        '{"comparison_version":NaN}',
        '{"comparison_version":Infinity}',
    ],
)
def test_comparison_loader_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path: Path, bad_json: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(bad_json, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid strict JSON"):
        load_evaluation_comparison_artifact(path)


def test_comparison_requires_metric_revision_for_every_compared_metric() -> None:
    content = _artifact().identity_dict()
    content["safety_metric"] = "safety"

    with pytest.raises(ValueError, match="must exactly match compared metrics"):
        EvaluationComparisonArtifact(content)
