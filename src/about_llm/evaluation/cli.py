"""CLI for offline scoring, calibration, paired comparison, and release gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from about_llm.evaluation.calibration import binary_calibration, risk_coverage_curve
from about_llm.evaluation.comparison_artifact import (
    EvaluationComparisonArtifact,
    build_evaluation_comparison_artifact,
    load_evaluation_comparison_artifact,
    write_evaluation_comparison_artifact,
)
from about_llm.evaluation.comparison_html import write_evaluation_comparison_html
from about_llm.evaluation.reporting import render_markdown_report, summarize_by_slice
from about_llm.evaluation.run_manifest import (
    EvaluationRunManifest,
    build_evaluation_run_manifest,
    load_evaluation_run_manifest,
    validate_evaluation_run_manifest,
    write_evaluation_run_manifest,
)
from about_llm.evaluation.runner import (
    EvaluationCase,
    EvaluationResult,
    Metric,
    evaluation_results_fingerprint,
    load_cases,
    load_results,
    read_strict_jsonl_objects,
    write_results,
)
from about_llm.evaluation.statistics import (
    ClusteredPairedBootstrapResult,
    ClusterWeighting,
    PairedBootstrapResult,
    ReleaseGate,
    clustered_paired_bootstrap,
    paired_bootstrap,
)
from about_llm.evaluation.structured import citation_syntax_metric, json_schema_metric
from about_llm.evaluation.text_metrics import normalized_exact_match, token_f1
from about_llm.llmops import artifact_fingerprint

METRICS: Mapping[str, Metric] = {
    "exact_match": normalized_exact_match,
    "token_f1": token_f1,
    "json_schema": json_schema_metric,
    "citation_syntax": citation_syntax_metric,
}
METRIC_REVISIONS: Mapping[str, str] = {
    "exact_match": "about-llm.normalized-exact-match.v1",
    "token_f1": "about-llm.token-f1.v1",
    "json_schema": "about-llm.json-schema-metric.v1",
    "citation_syntax": "about-llm.citation-syntax-metric.v1",
}
SCORER_REVISION = "about-llm.evaluation-cli.score.v1"


@dataclass(frozen=True)
class RecordedAnswer:
    """A system output captured elsewhere and ready for deterministic scoring."""

    case_id: str
    output: str
    latency_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "output": self.output,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }


@dataclass(frozen=True)
class CalibrationRecord:
    """One binary outcome and its pre-outcome probability estimate."""

    case_id: str
    label: int
    probability: float


def load_calibration_records(path: Path) -> list[CalibrationRecord]:
    """Load unique binary calibration records from UTF-8 JSONL."""

    records: list[CalibrationRecord] = []
    seen: set[str] = set()
    for line_number, value in read_strict_jsonl_objects(path):
        required = {"case_id", "label", "probability"}
        missing = required - set(value)
        unknown = set(value) - required
        if missing or unknown:
            raise ValueError(
                f"{path}:{line_number}: calibration field mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        case_id = value.get("case_id")
        label = value.get("label")
        probability = value.get("probability")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number}: case_id must be a non-empty string")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"{path}:{line_number}: label must be integer 0 or 1")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"{path}:{line_number}: probability must be numeric")
        if case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        records.append(CalibrationRecord(case_id, label, float(probability)))
    if not records:
        raise ValueError(f"{path} contains no calibration records")
    # The metric validation supplies finite and [0, 1] checks for probabilities.
    binary_calibration(
        [record.label for record in records],
        [record.probability for record in records],
    )
    return records


def load_answers(path: Path) -> dict[str, RecordedAnswer]:
    """Load one recorded answer per case id from UTF-8 JSONL."""
    answers: dict[str, RecordedAnswer] = {}
    for line_number, record in read_strict_jsonl_objects(path):
        required = {"case_id", "output", "latency_seconds"}
        optional = {"error"}
        missing = required - set(record)
        unknown = set(record) - required - optional
        if missing or unknown:
            raise ValueError(
                f"{path}:{line_number}: answer field mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        case_id = record.get("case_id")
        output = record.get("output")
        latency = record.get("latency_seconds")
        error_value = record.get("error")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number}: case_id must be a non-empty string")
        if not isinstance(output, str):
            raise ValueError(f"{path}:{line_number}: output must be a string")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            raise ValueError(f"{path}:{line_number}: latency_seconds must be numeric")
        if error_value is not None and not isinstance(error_value, str):
            raise ValueError(f"{path}:{line_number}: error must be a string or null")
        if case_id in answers:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id!r}")
        answer = RecordedAnswer(case_id, output, float(latency), error_value)
        # Reuse result validation for finite, non-negative latency.
        EvaluationResult(case_id, output, {}, answer.latency_seconds, error_value)
        answers[case_id] = answer
    if not answers:
        raise ValueError(f"{path} contains no recorded answers")
    return answers


def score_answers(
    cases: Sequence[EvaluationCase],
    answers: Mapping[str, RecordedAnswer],
    metrics: Mapping[str, Metric],
) -> list[EvaluationResult]:
    """Score recorded outputs while preserving failures as first-class rows."""
    case_ids = {case.case_id for case in cases}
    if case_ids != answers.keys():
        missing = sorted(case_ids - answers.keys())
        unknown = sorted(answers.keys() - case_ids)
        raise ValueError(f"case/answer mismatch: missing={missing}, unknown={unknown}")
    if not metrics:
        raise ValueError("at least one metric is required")

    results: list[EvaluationResult] = []
    for case in cases:
        answer = answers[case.case_id]
        scores = (
            {}
            if answer.error is not None
            else {name: float(metric(case, answer.output)) for name, metric in metrics.items()}
        )
        results.append(
            EvaluationResult(
                case_id=case.case_id,
                output=answer.output,
                scores=scores,
                latency_seconds=answer.latency_seconds,
                error=answer.error,
            )
        )
    return results


def recorded_answers_fingerprint(
    cases: Sequence[EvaluationCase], answers: Mapping[str, RecordedAnswer]
) -> str:
    """Bind answers in canonical case order after exact identity validation."""

    case_ids = {case.case_id for case in cases}
    if case_ids != answers.keys():
        missing = sorted(case_ids - answers.keys())
        unknown = sorted(answers.keys() - case_ids)
        raise ValueError(f"case/answer mismatch: missing={missing}, unknown={unknown}")
    ordered = [answers[case.case_id].to_dict() for case in cases]
    return "sha256:" + artifact_fingerprint({"ordered_recorded_answers": ordered})


def compare_results(
    cases: Sequence[EvaluationCase],
    baseline: Sequence[EvaluationResult],
    candidate: Sequence[EvaluationResult],
    *,
    quality_metric: str,
    safety_metric: str | None,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
    gate: ReleaseGate,
    protected_slices: Sequence[str],
    maximum_slice_regression: float,
    cluster_metadata_key: str | None = None,
    cluster_weighting: ClusterWeighting = "case",
    cluster_exact_max: int = 6,
) -> dict[str, Any]:
    """Run paired overall and protected-slice comparisons plus a release gate."""
    if (
        isinstance(maximum_slice_regression, bool)
        or not isinstance(maximum_slice_regression, (int, float))
        or not math.isfinite(maximum_slice_regression)
        or maximum_slice_regression < 0
    ):
        raise ValueError("maximum_slice_regression must be a finite non-negative number")
    if isinstance(protected_slices, (str, bytes)):
        raise ValueError("protected_slices must be a sequence of slice names")
    if len(protected_slices) != len(set(protected_slices)):
        raise ValueError("protected_slices must not contain duplicates")
    if cluster_metadata_key is not None and (
        not isinstance(cluster_metadata_key, str) or not cluster_metadata_key.strip()
    ):
        raise ValueError("cluster_metadata_key must be a non-empty string or null")
    if cluster_weighting not in ("case", "equal"):
        raise ValueError("cluster_weighting must be 'case' or 'equal'")
    if cluster_metadata_key is None and cluster_weighting != "case":
        raise ValueError("cluster_weighting='equal' requires cluster_metadata_key")
    if (
        isinstance(cluster_exact_max, bool)
        or not isinstance(cluster_exact_max, int)
        or not 0 <= cluster_exact_max <= 7
    ):
        raise ValueError("cluster_exact_max must be an integer in [0, 7]")
    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("duplicate case_id in evaluation cases")
    baseline_by_id = {result.case_id: result for result in baseline}
    candidate_by_id = {result.case_id: result for result in candidate}
    expected_ids = set(case_by_id)
    for label, values in (("baseline", baseline_by_id), ("candidate", candidate_by_id)):
        if set(values) != expected_ids:
            missing = sorted(expected_ids - set(values))
            unknown = sorted(set(values) - expected_ids)
            raise ValueError(f"{label} result mismatch: missing={missing}, unknown={unknown}")
    if len(baseline_by_id) != len(baseline) or len(candidate_by_id) != len(candidate):
        raise ValueError("duplicate case_id in result artifacts")

    ordered_ids = [case.case_id for case in cases]
    quality = _bootstrap_metric(
        ordered_ids,
        baseline_by_id,
        candidate_by_id,
        quality_metric,
        case_by_id=case_by_id,
        cluster_metadata_key=cluster_metadata_key,
        cluster_weighting=cluster_weighting,
        cluster_exact_max=cluster_exact_max,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    safety_difference = 0.0
    if safety_metric is not None:
        safety_difference = _metric_difference(
            ordered_ids,
            baseline_by_id,
            candidate_by_id,
            safety_metric,
            case_by_id=case_by_id,
            cluster_metadata_key=cluster_metadata_key,
            cluster_weighting=cluster_weighting,
        )
    baseline_latency = _mean_latency(ordered_ids, baseline_by_id)
    candidate_latency = _mean_latency(ordered_ids, candidate_by_id)
    passed, reasons = gate.evaluate(
        quality=quality,
        safety_difference=safety_difference,
        baseline_latency=baseline_latency,
        candidate_latency=candidate_latency,
    )

    slice_results: dict[str, dict[str, Any]] = {}
    for slice_name in protected_slices:
        ids = [case.case_id for case in cases if slice_name in case.slices]
        if not ids:
            raise ValueError(f"protected slice {slice_name!r} has no cases")
        comparison = _bootstrap_metric(
            ids,
            baseline_by_id,
            candidate_by_id,
            quality_metric,
            case_by_id=case_by_id,
            cluster_metadata_key=cluster_metadata_key,
            cluster_weighting=cluster_weighting,
            cluster_exact_max=cluster_exact_max,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed,
        )
        slice_results[slice_name] = asdict(comparison)
        if comparison.confidence_low < -maximum_slice_regression:
            passed = False
            reasons.append(
                f"protected slice {slice_name!r} quality lower bound "
                f"{comparison.confidence_low:.4f} < {-maximum_slice_regression:.4f}"
            )

    return {
        "passed": passed,
        "reasons": reasons,
        "quality_metric": quality_metric,
        "quality": asdict(quality),
        "safety_metric": safety_metric,
        "safety_difference": safety_difference,
        "baseline_mean_latency_seconds": baseline_latency,
        "candidate_mean_latency_seconds": candidate_latency,
        "protected_slices": slice_results,
        "case_count": len(cases),
        "bootstrap": {
            "unit": "cluster" if cluster_metadata_key is not None else "case",
            "confidence": confidence,
            "samples": bootstrap_samples,
            "seed": seed,
            "cluster_metadata_key": cluster_metadata_key,
            "cluster_weighting": (
                cluster_weighting if cluster_metadata_key is not None else None
            ),
            "exact_max_clusters": (
                cluster_exact_max if cluster_metadata_key is not None else None
            ),
        },
    }


def _bootstrap_metric(
    case_ids: Sequence[str],
    baseline: Mapping[str, EvaluationResult],
    candidate: Mapping[str, EvaluationResult],
    metric: str,
    *,
    case_by_id: Mapping[str, EvaluationCase],
    cluster_metadata_key: str | None,
    cluster_weighting: ClusterWeighting,
    cluster_exact_max: int,
    confidence: float,
    samples: int,
    seed: int,
) -> PairedBootstrapResult | ClusteredPairedBootstrapResult:
    baseline_scores = [_score(baseline[case_id], metric) for case_id in case_ids]
    candidate_scores = [_score(candidate[case_id], metric) for case_id in case_ids]
    if cluster_metadata_key is not None:
        return clustered_paired_bootstrap(
            baseline_scores,
            candidate_scores,
            _cluster_ids(case_ids, case_by_id, cluster_metadata_key),
            cluster_weighting=cluster_weighting,
            confidence=confidence,
            exact_max_clusters=cluster_exact_max,
            monte_carlo_samples=samples,
            seed=seed,
        )
    return paired_bootstrap(
        baseline_scores,
        candidate_scores,
        confidence=confidence,
        samples=samples,
        seed=seed,
    )


def _metric_difference(
    case_ids: Sequence[str],
    baseline: Mapping[str, EvaluationResult],
    candidate: Mapping[str, EvaluationResult],
    metric: str,
    *,
    case_by_id: Mapping[str, EvaluationCase],
    cluster_metadata_key: str | None,
    cluster_weighting: ClusterWeighting,
) -> float:
    differences = [
        _score(candidate[case_id], metric) - _score(baseline[case_id], metric)
        for case_id in case_ids
    ]
    if cluster_metadata_key is not None and cluster_weighting == "equal":
        cluster_ids = _cluster_ids(case_ids, case_by_id, cluster_metadata_key)
        by_cluster: dict[str, list[float]] = {}
        for cluster_id, difference in zip(cluster_ids, differences, strict=True):
            by_cluster.setdefault(cluster_id, []).append(difference)
        return sum(sum(values) / len(values) for values in by_cluster.values()) / len(
            by_cluster
        )
    return sum(differences) / len(differences)


def _cluster_ids(
    case_ids: Sequence[str],
    case_by_id: Mapping[str, EvaluationCase],
    metadata_key: str,
) -> list[str]:
    values: list[str] = []
    for case_id in case_ids:
        cluster_id = case_by_id[case_id].metadata.get(metadata_key)
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError(
                f"case {case_id!r} metadata {metadata_key!r} must be a non-empty string"
            )
        values.append(cluster_id)
    return values


def _score(result: EvaluationResult, metric: str) -> float:
    if result.error is not None:
        raise ValueError(f"case {result.case_id!r} has system error: {result.error}")
    try:
        return result.scores[metric]
    except KeyError as error:
        raise ValueError(f"case {result.case_id!r} is missing metric {metric!r}") from error


def _mean_latency(
    case_ids: Sequence[str], results: Mapping[str, EvaluationResult]
) -> float:
    value = sum(results[case_id].latency_seconds for case_id in case_ids) / len(case_ids)
    if value <= 0:
        raise ValueError("mean latency must be positive for release comparison")
    return value


def _select_metrics(names: Sequence[str] | None) -> dict[str, Metric]:
    selected = tuple(names) if names else ("exact_match", "token_f1")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate metric name")
    return {name: METRICS[name] for name in selected}


def _run_score(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    metrics = _select_metrics(args.metric)
    answers = load_answers(args.answers)
    results = score_answers(cases, answers, metrics)
    answers_fingerprint = recorded_answers_fingerprint(cases, answers)
    manifest = build_evaluation_run_manifest(
        system_id=args.system_id,
        cases=cases,
        results=results,
        recorded_answers_fingerprint=answers_fingerprint,
        metric_revisions={name: METRIC_REVISIONS[name] for name in metrics},
        metadata={"scorer_revision": SCORER_REVISION},
    )
    write_results(args.results, results)
    write_evaluation_run_manifest(args.manifest, manifest)
    summaries = summarize_by_slice(cases, results)
    report = (
        "# Evaluation report\n\n"
        + render_markdown_report(summaries)
        + "\n> 指标只在给定 case、metric 与 artifact 版本边界内成立。\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(report, end="")
    return 0


def _comparison_run_bindings(
    baseline_manifest: EvaluationRunManifest,
    candidate_manifest: EvaluationRunManifest,
    required_metrics: Sequence[str],
) -> dict[str, object]:
    return {
        "baseline": {
            "system_id": baseline_manifest.system_id,
            "manifest_fingerprint": baseline_manifest.manifest_fingerprint,
        },
        "candidate": {
            "system_id": candidate_manifest.system_id,
            "manifest_fingerprint": candidate_manifest.manifest_fingerprint,
        },
        "cases_fingerprint": baseline_manifest.cases_fingerprint,
        "metric_revisions": {
            metric: baseline_manifest.metric_revisions[metric]
            for metric in required_metrics
        },
    }


def _run_compare(args: argparse.Namespace) -> int:
    gate = ReleaseGate(
        minimum_quality_difference=args.minimum_quality_difference,
        maximum_safety_regression=args.maximum_safety_regression,
        maximum_latency_increase_fraction=args.maximum_latency_increase,
    )
    cases = load_cases(args.cases)
    baseline_results = load_results(args.baseline_results)
    candidate_results = load_results(args.candidate_results)
    baseline_manifest = load_evaluation_run_manifest(args.baseline_manifest)
    candidate_manifest = load_evaluation_run_manifest(args.candidate_manifest)
    required_metrics = [args.quality_metric]
    if args.safety_metric is not None:
        required_metrics.append(args.safety_metric)
    validate_evaluation_run_manifest(
        baseline_manifest,
        cases=cases,
        results=baseline_results,
        required_metrics=required_metrics,
    )
    validate_evaluation_run_manifest(
        candidate_manifest,
        cases=cases,
        results=candidate_results,
        required_metrics=required_metrics,
    )
    revision_mismatches = {
        metric: (
            baseline_manifest.metric_revisions[metric],
            candidate_manifest.metric_revisions[metric],
        )
        for metric in required_metrics
        if baseline_manifest.metric_revisions[metric]
        != candidate_manifest.metric_revisions[metric]
    }
    if revision_mismatches:
        raise ValueError(
            f"baseline/candidate metric revision mismatch: {revision_mismatches}"
        )
    comparison = compare_results(
        cases,
        baseline_results,
        candidate_results,
        quality_metric=args.quality_metric,
        safety_metric=args.safety_metric,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        gate=gate,
        protected_slices=args.protected_slice,
        maximum_slice_regression=args.maximum_slice_regression,
        cluster_metadata_key=args.cluster_metadata_key,
        cluster_weighting=args.cluster_weighting,
        cluster_exact_max=args.cluster_exact_max,
    )
    run_bindings = _comparison_run_bindings(
        baseline_manifest, candidate_manifest, required_metrics
    )
    artifact = build_evaluation_comparison_artifact(
        comparison=comparison,
        gate_configuration={
            "minimum_quality_difference": args.minimum_quality_difference,
            "maximum_safety_regression": args.maximum_safety_regression,
            "maximum_latency_increase_fraction": args.maximum_latency_increase,
            "maximum_slice_regression": args.maximum_slice_regression,
            "protected_slices": args.protected_slice,
        },
        run_bindings=run_bindings,
    )
    payload = artifact.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        write_evaluation_comparison_artifact(args.output, artifact)
    print(rendered)
    return 0 if payload["passed"] else 1


def _run_verify_comparison(args: argparse.Namespace) -> int:
    artifact = load_evaluation_comparison_artifact(args.input)
    print(
        json.dumps(
            {
                "valid": True,
                "verification_scope": "artifact_only",
                "referenced_manifests_revalidated": False,
                "statistics_recomputed": False,
                "comparison_version": artifact.content["comparison_version"],
                "comparison_fingerprint": artifact.comparison_fingerprint,
                "passed": artifact.passed,
                "baseline_system_id": artifact.baseline_system_id,
                "candidate_system_id": artifact.candidate_system_id,
                "evidence_boundary": artifact.content["evidence_boundary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_render_comparison_html(args: argparse.Namespace) -> int:
    artifact = load_evaluation_comparison_artifact(args.input)
    write_evaluation_comparison_html(args.output, artifact)
    print(
        json.dumps(
            {
                "written": str(args.output),
                "verification_scope": "artifact_only_render",
                "comparison_fingerprint": artifact.comparison_fingerprint,
                "passed": artifact.passed,
                "scripts_included": False,
                "external_resources_included": False,
                "statistics_recomputed": False,
                "artifact_authentication_verified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _verify_run_evidence(
    *,
    label: str,
    cases: Sequence[EvaluationCase],
    answers: Mapping[str, RecordedAnswer],
    results: Sequence[EvaluationResult],
    manifest: EvaluationRunManifest,
) -> None:
    metric_names = tuple(manifest.metric_revisions)
    validate_evaluation_run_manifest(
        manifest,
        cases=cases,
        results=results,
        required_metrics=metric_names,
    )
    if recorded_answers_fingerprint(cases, answers) != (
        manifest.recorded_answers_fingerprint
    ):
        raise ValueError(
            f"{label} recorded answers do not match run manifest fingerprint"
        )
    if manifest.metadata.get("scorer_revision") != SCORER_REVISION:
        raise ValueError(
            f"{label} scorer_revision is not executable by this verifier: "
            f"{manifest.metadata.get('scorer_revision')!r}"
        )
    metrics: dict[str, Metric] = {}
    for metric_name, recorded_revision in manifest.metric_revisions.items():
        current_revision = METRIC_REVISIONS.get(metric_name)
        if current_revision is None:
            raise ValueError(
                f"{label} metric {metric_name!r} is not executable by this verifier"
            )
        if current_revision != recorded_revision:
            raise ValueError(
                f"{label} metric revision mismatch for {metric_name!r}: "
                f"recorded={recorded_revision!r}, executable={current_revision!r}"
            )
        metrics[metric_name] = METRICS[metric_name]
    rescored = score_answers(cases, answers, metrics)
    if evaluation_results_fingerprint(rescored) != evaluation_results_fingerprint(
        results
    ):
        raise ValueError(f"{label} recomputed scores do not match recorded results")


def _recompute_comparison_artifact(
    *,
    artifact: EvaluationComparisonArtifact,
    cases: Sequence[EvaluationCase],
    baseline_results: Sequence[EvaluationResult],
    candidate_results: Sequence[EvaluationResult],
    baseline_manifest: EvaluationRunManifest,
    candidate_manifest: EvaluationRunManifest,
) -> EvaluationComparisonArtifact:
    content = artifact.content
    quality_metric = cast(str, content["quality_metric"])
    safety_metric = cast(str | None, content["safety_metric"])
    required_metrics = [quality_metric]
    if safety_metric is not None:
        required_metrics.append(safety_metric)
    revision_mismatches = {
        metric: (
            baseline_manifest.metric_revisions[metric],
            candidate_manifest.metric_revisions[metric],
        )
        for metric in required_metrics
        if baseline_manifest.metric_revisions[metric]
        != candidate_manifest.metric_revisions[metric]
    }
    if revision_mismatches:
        raise ValueError(
            f"baseline/candidate metric revision mismatch: {revision_mismatches}"
        )
    bootstrap = cast(Mapping[str, Any], content["bootstrap"])
    gate_configuration = cast(Mapping[str, Any], content["gate_configuration"])
    cluster_metadata_key = cast(str | None, bootstrap["cluster_metadata_key"])
    cluster_weighting: ClusterWeighting = (
        "case"
        if cluster_metadata_key is None
        else cast(ClusterWeighting, bootstrap["cluster_weighting"])
    )
    exact_max_value = bootstrap["exact_max_clusters"]
    cluster_exact_max = 6 if exact_max_value is None else cast(int, exact_max_value)
    comparison = compare_results(
        cases,
        baseline_results,
        candidate_results,
        quality_metric=quality_metric,
        safety_metric=safety_metric,
        confidence=cast(float, bootstrap["confidence"]),
        bootstrap_samples=cast(int, bootstrap["samples"]),
        seed=cast(int, bootstrap["seed"]),
        gate=ReleaseGate(
            minimum_quality_difference=cast(
                float, gate_configuration["minimum_quality_difference"]
            ),
            maximum_safety_regression=cast(
                float, gate_configuration["maximum_safety_regression"]
            ),
            maximum_latency_increase_fraction=cast(
                float, gate_configuration["maximum_latency_increase_fraction"]
            ),
        ),
        protected_slices=cast(
            Sequence[str], gate_configuration["protected_slices"]
        ),
        maximum_slice_regression=cast(
            float, gate_configuration["maximum_slice_regression"]
        ),
        cluster_metadata_key=cluster_metadata_key,
        cluster_weighting=cluster_weighting,
        cluster_exact_max=cluster_exact_max,
    )
    return build_evaluation_comparison_artifact(
        comparison=comparison,
        gate_configuration=gate_configuration,
        run_bindings=_comparison_run_bindings(
            baseline_manifest, candidate_manifest, required_metrics
        ),
    )


def _run_verify_evidence(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    baseline_answers = load_answers(args.baseline_answers)
    candidate_answers = load_answers(args.candidate_answers)
    baseline_results = load_results(args.baseline_results)
    candidate_results = load_results(args.candidate_results)
    baseline_manifest = load_evaluation_run_manifest(args.baseline_manifest)
    candidate_manifest = load_evaluation_run_manifest(args.candidate_manifest)
    artifact = load_evaluation_comparison_artifact(args.comparison)
    _verify_run_evidence(
        label="baseline",
        cases=cases,
        answers=baseline_answers,
        results=baseline_results,
        manifest=baseline_manifest,
    )
    _verify_run_evidence(
        label="candidate",
        cases=cases,
        answers=candidate_answers,
        results=candidate_results,
        manifest=candidate_manifest,
    )
    recomputed = _recompute_comparison_artifact(
        artifact=artifact,
        cases=cases,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
    )
    if recomputed.to_dict() != artifact.to_dict():
        raise ValueError(
            "recomputed comparison does not match the supplied comparison artifact"
        )
    print(
        json.dumps(
            {
                "valid": True,
                "verification_scope": "full_local_recomputation",
                "case_semantics_rehashed": True,
                "recorded_answers_rehashed": True,
                "scores_recomputed": True,
                "run_manifests_revalidated": True,
                "statistics_recomputed": True,
                "comparison_rebuilt": True,
                "current_metric_revisions_matched": True,
                "artifact_authentication_verified": False,
                "model_execution_replayed": False,
                "sampling_assumptions_validated": False,
                "construct_validity_validated": False,
                "production_impact_validated": False,
                "comparison_fingerprint": artifact.comparison_fingerprint,
                "passed": artifact.passed,
                "baseline_system_id": artifact.baseline_system_id,
                "candidate_system_id": artifact.candidate_system_id,
                "evidence_boundary": (
                    "This verification reopens local cases, answers, results, run "
                    "manifests, and the comparison; rehashes answer/case identities; "
                    "re-executes the recorded repository metric revisions; and rebuilds "
                    "the paired statistics and gate decision. It does not authenticate "
                    "the files, replay model/provider execution, validate sampling or "
                    "cluster assumptions, establish metric construct validity, or prove "
                    "production impact."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    records = load_calibration_records(args.input)
    labels = [record.label for record in records]
    probabilities = [record.probability for record in records]
    calibration = binary_calibration(labels, probabilities, bins=args.bins)
    curve = risk_coverage_curve(labels, probabilities)
    payload = {
        "semantics": {
            "label": "binary observed correctness/outcome",
            "probability": "pre-outcome probability assigned to label 1",
            "ece": "equal-width, empty bins omitted",
            "risk": "error rate among records at or above threshold",
        },
        "calibration": asdict(calibration),
        "risk_coverage": [asdict(point) for point in curve],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-eval",
        description=(
            "Offline scoring, evidence recomputation, reports, and release gates"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="score recorded outputs")
    score.add_argument("--cases", type=Path, required=True)
    score.add_argument("--answers", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--system-id", required=True)
    score.add_argument("--metric", action="append", choices=sorted(METRICS))
    score.set_defaults(handler=_run_score)

    calibrate = subparsers.add_parser(
        "calibrate", help="measure binary Brier, equal-width ECE, and selective risk"
    )
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--bins", type=int, default=10)
    calibrate.add_argument("--output", type=Path)
    calibrate.set_defaults(handler=_run_calibrate)

    compare = subparsers.add_parser("compare", help="compare paired result artifacts")
    compare.add_argument("--cases", type=Path, required=True)
    compare.add_argument("--baseline-results", type=Path, required=True)
    compare.add_argument("--candidate-results", type=Path, required=True)
    compare.add_argument("--baseline-manifest", type=Path, required=True)
    compare.add_argument("--candidate-manifest", type=Path, required=True)
    compare.add_argument("--quality-metric", required=True)
    compare.add_argument("--safety-metric")
    compare.add_argument("--minimum-quality-difference", type=float, default=0.0)
    compare.add_argument("--maximum-safety-regression", type=float, default=0.0)
    compare.add_argument("--maximum-latency-increase", type=float, default=0.10)
    compare.add_argument("--protected-slice", action="append", default=[])
    compare.add_argument("--maximum-slice-regression", type=float, default=0.0)
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--cluster-metadata-key")
    compare.add_argument(
        "--cluster-weighting", choices=("case", "equal"), default="case"
    )
    compare.add_argument("--cluster-exact-max", type=int, default=6)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=_run_compare)

    verify_comparison = subparsers.add_parser(
        "verify-comparison",
        help="strictly reload and verify a versioned comparison artifact",
    )
    verify_comparison.add_argument("--input", type=Path, required=True)
    verify_comparison.set_defaults(handler=_run_verify_comparison)

    render_html = subparsers.add_parser(
        "render-comparison-html",
        help="render a strict comparison artifact as self-contained HTML",
    )
    render_html.add_argument("--input", type=Path, required=True)
    render_html.add_argument("--output", type=Path, required=True)
    render_html.set_defaults(handler=_run_render_comparison_html)

    verify_evidence = subparsers.add_parser(
        "verify-evidence",
        help="reopen and recompute the complete local comparison evidence graph",
    )
    verify_evidence.add_argument("--cases", type=Path, required=True)
    verify_evidence.add_argument("--baseline-answers", type=Path, required=True)
    verify_evidence.add_argument("--candidate-answers", type=Path, required=True)
    verify_evidence.add_argument("--baseline-results", type=Path, required=True)
    verify_evidence.add_argument("--candidate-results", type=Path, required=True)
    verify_evidence.add_argument("--baseline-manifest", type=Path, required=True)
    verify_evidence.add_argument("--candidate-manifest", type=Path, required=True)
    verify_evidence.add_argument("--comparison", type=Path, required=True)
    verify_evidence.set_defaults(handler=_run_verify_evidence)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    try:
        return handler(args)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
