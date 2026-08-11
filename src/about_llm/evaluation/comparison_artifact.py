"""Strict, reloadable artifacts for paired evaluation release decisions."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

EVALUATION_COMPARISON_VERSION = "about-llm.evaluation-comparison.v2"
EVALUATION_COMPARISON_EVIDENCE_BOUNDARY = (
    "This artifact verifies canonical case/result/metric/configuration bindings and records "
    "case- or cluster-resampling configuration supplied by unsigned manifests. It does not "
    "authenticate system_id, prove outputs came from that system, validate cluster definition, "
    "independence, representative sampling, interval coverage, case or metric construct "
    "validity, or establish production impact."
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_ROOT_FIELDS = {
    "comparison_version",
    "passed",
    "reasons",
    "quality_metric",
    "quality",
    "safety_metric",
    "safety_difference",
    "baseline_mean_latency_seconds",
    "candidate_mean_latency_seconds",
    "protected_slices",
    "case_count",
    "bootstrap",
    "gate_configuration",
    "run_bindings",
    "evidence_boundary",
}
_BOOTSTRAP_RESULT_FIELDS = {
    "baseline_mean",
    "candidate_mean",
    "mean_difference",
    "confidence_low",
    "confidence_high",
    "probability_of_improvement",
}
_CLUSTER_BOOTSTRAP_RESULT_FIELDS = {
    "case_count",
    "cluster_count",
    "cluster_sizes",
    "cluster_weighting",
    "baseline_estimand",
    "candidate_estimand",
    "mean_difference",
    "confidence",
    "confidence_low",
    "confidence_high",
    "probability_of_improvement",
    "method",
    "resamples_evaluated",
    "quantile_method",
    "seed",
}


@dataclass(frozen=True)
class EvaluationComparisonArtifact:
    """Canonical snapshot of one paired comparison and its release-gate inputs."""

    content: Mapping[str, Any]

    def __post_init__(self) -> None:
        snapshot = json.loads(canonical_json_bytes(self.content))
        if not isinstance(snapshot, dict):
            raise ValueError("comparison content must be a JSON object")
        record = cast(dict[str, Any], snapshot)
        _validate_comparison_content(record)
        object.__setattr__(self, "content", _freeze(record))

    @property
    def comparison_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.identity_dict())

    @property
    def passed(self) -> bool:
        return cast(bool, self.content["passed"])

    @property
    def baseline_system_id(self) -> str:
        bindings = cast(Mapping[str, Any], self.content["run_bindings"])
        baseline = cast(Mapping[str, Any], bindings["baseline"])
        return cast(str, baseline["system_id"])

    @property
    def candidate_system_id(self) -> str:
        bindings = cast(Mapping[str, Any], self.content["run_bindings"])
        candidate = cast(Mapping[str, Any], bindings["candidate"])
        return cast(str, candidate["system_id"])

    def identity_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _thaw(self.content))

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_dict(),
            "comparison_fingerprint": self.comparison_fingerprint,
        }


def build_evaluation_comparison_artifact(
    *,
    comparison: Mapping[str, Any],
    gate_configuration: Mapping[str, Any],
    run_bindings: Mapping[str, Any],
) -> EvaluationComparisonArtifact:
    content = {
        **dict(comparison),
        "comparison_version": EVALUATION_COMPARISON_VERSION,
        "gate_configuration": dict(gate_configuration),
        "run_bindings": dict(run_bindings),
        "evidence_boundary": EVALUATION_COMPARISON_EVIDENCE_BOUNDARY,
    }
    return EvaluationComparisonArtifact(content)


def load_evaluation_comparison_artifact(path: Path) -> EvaluationComparisonArtifact:
    try:
        value: Any = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: comparison must be a JSON object")
    record = cast(dict[str, Any], value)
    allowed = _ROOT_FIELDS | {"comparison_fingerprint"}
    _exact_fields(record, allowed, f"{path}: comparison")
    supplied = _string(record["comparison_fingerprint"], "comparison_fingerprint")
    _fingerprint(supplied, "comparison_fingerprint")
    content = {key: item for key, item in record.items() if key != "comparison_fingerprint"}
    artifact = EvaluationComparisonArtifact(content)
    if supplied != artifact.comparison_fingerprint:
        raise ValueError(f"{path}: comparison_fingerprint does not match canonical content")
    return artifact


def write_evaluation_comparison_artifact(
    path: Path, artifact: EvaluationComparisonArtifact
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            artifact.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _validate_comparison_content(value: Mapping[str, Any]) -> None:
    _exact_fields(value, _ROOT_FIELDS, "comparison")
    if value["comparison_version"] != EVALUATION_COMPARISON_VERSION:
        raise ValueError(
            f"comparison_version must equal {EVALUATION_COMPARISON_VERSION!r}"
        )
    if value["evidence_boundary"] != EVALUATION_COMPARISON_EVIDENCE_BOUNDARY:
        raise ValueError("comparison evidence_boundary does not match the schema version")
    if not isinstance(value["passed"], bool):
        raise ValueError("comparison passed must be a boolean")
    reasons = _array(value["reasons"], "reasons")
    if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
        raise ValueError("comparison reasons must contain non-empty strings")
    _string(value["quality_metric"], "quality_metric")
    safety_metric = value["safety_metric"]
    if safety_metric is not None:
        _string(safety_metric, "safety_metric")
    bootstrap = _validate_bootstrap(value["bootstrap"])
    _bootstrap_result(value["quality"], "quality", bootstrap)
    _finite(value["safety_difference"], "safety_difference")
    for name in (
        "baseline_mean_latency_seconds",
        "candidate_mean_latency_seconds",
    ):
        if _finite(value[name], name) <= 0:
            raise ValueError(f"{name} must be positive")
    protected = _object(value["protected_slices"], "protected_slices")
    for slice_name, result in protected.items():
        _string(slice_name, "protected slice name")
        _bootstrap_result(result, f"protected_slices.{slice_name}", bootstrap)
    case_count = _integer(value["case_count"], "case_count")
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    if bootstrap["unit"] == "cluster":
        quality = cast(Mapping[str, Any], value["quality"])
        if quality["case_count"] != case_count:
            raise ValueError("quality.case_count must match comparison case_count")
        for slice_name, result in protected.items():
            slice_result = cast(Mapping[str, Any], result)
            if cast(int, slice_result["case_count"]) > case_count:
                raise ValueError(
                    f"protected_slices.{slice_name}.case_count must not exceed "
                    "comparison case_count"
                )
    _validate_gate_configuration(value["gate_configuration"])
    _validate_run_bindings(value["run_bindings"], value)
    config = cast(Mapping[str, Any], value["gate_configuration"])
    configured_slices = cast(list[str], config["protected_slices"])
    if set(configured_slices) != set(protected):
        raise ValueError(
            "gate_configuration.protected_slices must match protected_slices results"
        )
    if safety_metric is None and value["safety_difference"] != 0:
        raise ValueError("safety_difference must be zero when safety_metric is null")
    expected_reasons = _expected_reasons(value)
    if reasons != expected_reasons:
        raise ValueError(
            "comparison reasons do not match the recorded statistics and gate configuration"
        )
    if value["passed"] is not (not expected_reasons):
        raise ValueError("comparison passed does not match the recorded gate decision")


def _bootstrap_result(
    value: Any, location: str, bootstrap: Mapping[str, Any]
) -> None:
    if bootstrap["unit"] == "cluster":
        _cluster_bootstrap_result(value, location, bootstrap)
        return
    result = _object(value, location)
    _exact_fields(result, _BOOTSTRAP_RESULT_FIELDS, location)
    numbers: dict[str, float] = {}
    for name in _BOOTSTRAP_RESULT_FIELDS:
        number = _finite(result[name], f"{location}.{name}")
        numbers[name] = number
        if name == "probability_of_improvement" and not 0 <= number <= 1:
            raise ValueError(f"{location}.{name} must be in [0, 1]")
    if result["confidence_low"] > result["confidence_high"]:
        raise ValueError(f"{location} confidence_low must not exceed confidence_high")
    expected_difference = numbers["candidate_mean"] - numbers["baseline_mean"]
    if not math.isclose(
        numbers["mean_difference"],
        expected_difference,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{location}.mean_difference must equal candidate_mean - baseline_mean"
        )


def _cluster_bootstrap_result(
    value: Any, location: str, bootstrap: Mapping[str, Any]
) -> None:
    result = _object(value, location)
    _exact_fields(result, _CLUSTER_BOOTSTRAP_RESULT_FIELDS, location)
    case_count = _integer(result["case_count"], f"{location}.case_count")
    cluster_count = _integer(result["cluster_count"], f"{location}.cluster_count")
    if case_count <= 0 or not 0 < cluster_count <= case_count:
        raise ValueError(
            f"{location} case_count and cluster_count must satisfy "
            "0 < cluster_count <= case_count"
        )
    sizes = _array(result["cluster_sizes"], f"{location}.cluster_sizes")
    if len(sizes) != cluster_count or any(
        _integer(size, f"{location}.cluster_sizes") <= 0 for size in sizes
    ):
        raise ValueError(
            f"{location}.cluster_sizes must contain one positive size per cluster"
        )
    if sum(cast(list[int], sizes)) != case_count:
        raise ValueError(f"{location}.cluster_sizes must sum to case_count")
    if result["cluster_weighting"] != bootstrap["cluster_weighting"]:
        raise ValueError(
            f"{location}.cluster_weighting must match bootstrap.cluster_weighting"
        )
    baseline = _finite(result["baseline_estimand"], f"{location}.baseline_estimand")
    candidate = _finite(
        result["candidate_estimand"], f"{location}.candidate_estimand"
    )
    difference = _finite(result["mean_difference"], f"{location}.mean_difference")
    if not math.isclose(
        difference,
        candidate - baseline,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{location}.mean_difference must equal "
            "candidate_estimand - baseline_estimand"
        )
    confidence = _finite(result["confidence"], f"{location}.confidence")
    if confidence != bootstrap["confidence"]:
        raise ValueError(f"{location}.confidence must match bootstrap.confidence")
    low = _finite(result["confidence_low"], f"{location}.confidence_low")
    high = _finite(result["confidence_high"], f"{location}.confidence_high")
    if low > high:
        raise ValueError(f"{location} confidence_low must not exceed confidence_high")
    probability = _finite(
        result["probability_of_improvement"],
        f"{location}.probability_of_improvement",
    )
    if not 0 <= probability <= 1:
        raise ValueError(f"{location}.probability_of_improvement must be in [0, 1]")
    if result["quantile_method"] != "linear":
        raise ValueError(f"{location}.quantile_method must equal 'linear'")
    method = result["method"]
    if method not in ("exact", "monte_carlo"):
        raise ValueError(f"{location}.method must be 'exact' or 'monte_carlo'")
    evaluated = _integer(
        result["resamples_evaluated"], f"{location}.resamples_evaluated"
    )
    exact_max = cast(int, bootstrap["exact_max_clusters"])
    if method == "exact":
        if cluster_count > exact_max:
            raise ValueError(
                f"{location}.method exact exceeds bootstrap.exact_max_clusters"
            )
        if evaluated != cluster_count**cluster_count:
            raise ValueError(
                f"{location}.resamples_evaluated must equal cluster_count**cluster_count"
            )
        if result["seed"] is not None:
            raise ValueError(f"{location}.seed must be null for exact resampling")
        scaled_probability = probability * evaluated
        if not math.isclose(
            scaled_probability,
            round(scaled_probability),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{location}.probability_of_improvement must be an exact "
                "resample fraction"
            )
    else:
        if cluster_count <= exact_max:
            raise ValueError(
                f"{location}.method monte_carlo conflicts with exact cluster threshold"
            )
        if evaluated != bootstrap["samples"]:
            raise ValueError(
                f"{location}.resamples_evaluated must match bootstrap.samples"
            )
        if _integer(result["seed"], f"{location}.seed") != bootstrap["seed"]:
            raise ValueError(f"{location}.seed must match bootstrap.seed")


def _validate_bootstrap(value: Any) -> dict[str, Any]:
    bootstrap = _object(value, "bootstrap")
    _exact_fields(
        bootstrap,
        {
            "unit",
            "confidence",
            "samples",
            "seed",
            "cluster_metadata_key",
            "cluster_weighting",
            "exact_max_clusters",
        },
        "bootstrap",
    )
    if bootstrap["unit"] not in ("case", "cluster"):
        raise ValueError("bootstrap.unit must be 'case' or 'cluster'")
    confidence = _finite(bootstrap["confidence"], "bootstrap.confidence")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap.confidence must be in (0, 1)")
    if _integer(bootstrap["samples"], "bootstrap.samples") <= 0:
        raise ValueError("bootstrap.samples must be positive")
    _integer(bootstrap["seed"], "bootstrap.seed")
    if bootstrap["unit"] == "case":
        if any(
            bootstrap[name] is not None
            for name in (
                "cluster_metadata_key",
                "cluster_weighting",
                "exact_max_clusters",
            )
        ):
            raise ValueError("case bootstrap must have null cluster configuration")
    else:
        _string(bootstrap["cluster_metadata_key"], "bootstrap.cluster_metadata_key")
        if bootstrap["cluster_weighting"] not in ("case", "equal"):
            raise ValueError("bootstrap.cluster_weighting must be 'case' or 'equal'")
        exact_max = _integer(
            bootstrap["exact_max_clusters"], "bootstrap.exact_max_clusters"
        )
        if not 0 <= exact_max <= 7:
            raise ValueError("bootstrap.exact_max_clusters must be in [0, 7]")
    return bootstrap


def _validate_gate_configuration(value: Any) -> None:
    config = _object(value, "gate_configuration")
    fields = {
        "minimum_quality_difference",
        "maximum_safety_regression",
        "maximum_latency_increase_fraction",
        "maximum_slice_regression",
        "protected_slices",
    }
    _exact_fields(config, fields, "gate_configuration")
    numeric = {
        name: _finite(config[name], f"gate_configuration.{name}")
        for name in fields - {"protected_slices"}
    }
    for name in ("maximum_safety_regression", "maximum_slice_regression"):
        if numeric[name] < 0:
            raise ValueError(f"gate_configuration.{name} must be non-negative")
    if numeric["maximum_latency_increase_fraction"] <= -1:
        raise ValueError(
            "gate_configuration.maximum_latency_increase_fraction must be greater "
            "than -1"
        )
    slices = _array(config["protected_slices"], "gate_configuration.protected_slices")
    if any(not isinstance(item, str) or not item.strip() for item in slices):
        raise ValueError("gate_configuration.protected_slices must contain strings")
    if len(slices) != len(set(cast(list[str], slices))):
        raise ValueError("gate_configuration.protected_slices must not contain duplicates")


def _validate_run_bindings(value: Any, comparison: Mapping[str, Any]) -> None:
    bindings = _object(value, "run_bindings")
    _exact_fields(
        bindings,
        {"baseline", "candidate", "cases_fingerprint", "metric_revisions"},
        "run_bindings",
    )
    for side in ("baseline", "candidate"):
        run = _object(bindings[side], f"run_bindings.{side}")
        _exact_fields(
            run,
            {"system_id", "manifest_fingerprint"},
            f"run_bindings.{side}",
        )
        _string(run["system_id"], f"run_bindings.{side}.system_id")
        _fingerprint(
            _string(
                run["manifest_fingerprint"],
                f"run_bindings.{side}.manifest_fingerprint",
            ),
            f"run_bindings.{side}.manifest_fingerprint",
        )
    _fingerprint(
        _string(bindings["cases_fingerprint"], "run_bindings.cases_fingerprint"),
        "run_bindings.cases_fingerprint",
    )
    revisions = _object(bindings["metric_revisions"], "run_bindings.metric_revisions")
    required = {cast(str, comparison["quality_metric"])}
    if comparison["safety_metric"] is not None:
        required.add(cast(str, comparison["safety_metric"]))
    if set(revisions) != required:
        raise ValueError(
            "run_bindings.metric_revisions must exactly match compared metrics"
        )
    for name, revision in revisions.items():
        _string(name, "metric revision name")
        _string(revision, f"metric revision {name!r}")


def _expected_reasons(comparison: Mapping[str, Any]) -> list[str]:
    quality = cast(Mapping[str, Any], comparison["quality"])
    config = cast(Mapping[str, Any], comparison["gate_configuration"])
    reasons: list[str] = []
    quality_low = cast(float, quality["confidence_low"])
    minimum_quality = cast(float, config["minimum_quality_difference"])
    if quality_low < minimum_quality:
        reasons.append(
            "quality confidence lower bound "
            f"{quality_low:.4f} < {minimum_quality:.4f}"
        )
    safety_difference = cast(float, comparison["safety_difference"])
    maximum_safety_regression = cast(float, config["maximum_safety_regression"])
    if safety_difference < -maximum_safety_regression:
        reasons.append(
            f"safety difference {safety_difference:.4f} exceeds allowed regression"
        )
    baseline_latency = cast(float, comparison["baseline_mean_latency_seconds"])
    candidate_latency = cast(float, comparison["candidate_mean_latency_seconds"])
    latency_increase = candidate_latency / baseline_latency - 1
    maximum_latency_increase = cast(
        float, config["maximum_latency_increase_fraction"]
    )
    if latency_increase > maximum_latency_increase:
        reasons.append(
            f"latency increase {latency_increase:.1%} exceeds "
            f"{maximum_latency_increase:.1%}"
        )
    maximum_slice_regression = cast(float, config["maximum_slice_regression"])
    protected = cast(Mapping[str, Mapping[str, Any]], comparison["protected_slices"])
    for slice_name in cast(list[str], config["protected_slices"]):
        confidence_low = cast(float, protected[slice_name]["confidence_low"])
        if confidence_low < -maximum_slice_regression:
            reasons.append(
                f"protected slice {slice_name!r} quality lower bound "
                f"{confidence_low:.4f} < {-maximum_slice_regression:.4f}"
            )
    return reasons


def _exact_fields(value: Mapping[str, Any], fields: set[str], location: str) -> None:
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise ValueError(
            f"{location} field mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return cast(dict[str, Any], value)


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _finite(value: Any, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{location} must be a finite number")
    return float(value)


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return cast(int, value)


def _fingerprint(value: str, location: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{location} must be sha256:<64 lowercase hex>")


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _freeze(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _freeze(cast(dict[str, Any], value))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
