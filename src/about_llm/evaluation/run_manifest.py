"""Strict identities binding evaluation cases, results, metrics, and systems."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.evaluation.runner import (
    EvaluationCase,
    EvaluationResult,
    evaluation_cases_fingerprint,
    evaluation_results_fingerprint,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

EVALUATION_RUN_MANIFEST_VERSION = "about-llm.evaluation-run-manifest.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class EvaluationRunManifest:
    """Canonical identity for one deterministic scoring run.

    This is an unsigned content binding. It does not authenticate who ran the
    system or prove the supplied outputs came from the named system.
    """

    system_id: str
    cases_fingerprint: str
    results_fingerprint: str
    recorded_answers_fingerprint: str
    ordered_case_ids: tuple[str, ...]
    metric_revisions: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.system_id, "system_id")
        for name, value in (
            ("cases_fingerprint", self.cases_fingerprint),
            ("results_fingerprint", self.results_fingerprint),
            ("recorded_answers_fingerprint", self.recorded_answers_fingerprint),
        ):
            _fingerprint(value, name)
        _unique_strings(self.ordered_case_ids, "ordered_case_ids")
        if not self.ordered_case_ids:
            raise ValueError("ordered_case_ids must not be empty")
        revisions = dict(self.metric_revisions)
        if not revisions:
            raise ValueError("metric_revisions must not be empty")
        for name, revision in revisions.items():
            _nonempty(name, "metric name")
            _nonempty(revision, f"metric revision for {name!r}")
        object.__setattr__(
            self,
            "metric_revisions",
            MappingProxyType(dict(sorted(revisions.items()))),
        )
        metadata = json.loads(canonical_json_bytes(self.metadata))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze(cast(dict[str, Any], metadata)))

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.identity_dict())

    def identity_dict(self) -> dict[str, object]:
        return {
            "manifest_version": EVALUATION_RUN_MANIFEST_VERSION,
            "system_id": self.system_id,
            "cases_fingerprint": self.cases_fingerprint,
            "results_fingerprint": self.results_fingerprint,
            "recorded_answers_fingerprint": self.recorded_answers_fingerprint,
            "ordered_case_ids": list(self.ordered_case_ids),
            "metric_revisions": dict(self.metric_revisions),
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_dict(),
            "manifest_fingerprint": self.manifest_fingerprint,
        }


def build_evaluation_run_manifest(
    *,
    system_id: str,
    cases: Sequence[EvaluationCase],
    results: Sequence[EvaluationResult],
    recorded_answers_fingerprint: str,
    metric_revisions: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> EvaluationRunManifest:
    case_ids = tuple(case.case_id for case in cases)
    result_ids = tuple(result.case_id for result in results)
    if case_ids != result_ids:
        raise ValueError(
            "ordered case/result identity mismatch: "
            f"cases={case_ids!r}, results={result_ids!r}"
        )
    return EvaluationRunManifest(
        system_id=system_id,
        cases_fingerprint=evaluation_cases_fingerprint(cases),
        results_fingerprint=evaluation_results_fingerprint(results),
        recorded_answers_fingerprint=recorded_answers_fingerprint,
        ordered_case_ids=case_ids,
        metric_revisions=metric_revisions,
        metadata={} if metadata is None else metadata,
    )


def validate_evaluation_run_manifest(
    manifest: EvaluationRunManifest,
    *,
    cases: Sequence[EvaluationCase],
    results: Sequence[EvaluationResult],
    required_metrics: Sequence[str],
) -> None:
    case_ids = tuple(case.case_id for case in cases)
    result_ids = tuple(result.case_id for result in results)
    findings: list[str] = []
    if manifest.ordered_case_ids != case_ids:
        findings.append("ordered_case_ids do not match current cases")
    if result_ids != case_ids:
        findings.append("ordered result ids do not match current cases")
    if manifest.cases_fingerprint != evaluation_cases_fingerprint(cases):
        findings.append("cases_fingerprint does not match current case semantics")
    if manifest.results_fingerprint != evaluation_results_fingerprint(results):
        findings.append("results_fingerprint does not match current result bytes/values")
    missing_metrics = sorted(set(required_metrics) - set(manifest.metric_revisions))
    if missing_metrics:
        findings.append(f"manifest is missing metric revisions {missing_metrics}")
    if findings:
        raise ValueError("evaluation run manifest mismatch: " + "; ".join(findings))


def load_evaluation_run_manifest(path: Path) -> EvaluationRunManifest:
    try:
        value: Any = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    record = cast(dict[str, Any], value)
    required = {
        "manifest_version",
        "system_id",
        "cases_fingerprint",
        "results_fingerprint",
        "recorded_answers_fingerprint",
        "ordered_case_ids",
        "metric_revisions",
        "metadata",
        "manifest_fingerprint",
    }
    missing = required - set(record)
    unknown = set(record) - required
    if missing or unknown:
        raise ValueError(
            f"{path}: manifest field mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    version = _string(record["manifest_version"], "manifest_version")
    if version != EVALUATION_RUN_MANIFEST_VERSION:
        raise ValueError(
            f"{path}: expected manifest_version "
            f"{EVALUATION_RUN_MANIFEST_VERSION!r}, got {version!r}"
        )
    raw_ids = record["ordered_case_ids"]
    if not isinstance(raw_ids, list):
        raise ValueError(f"{path}: ordered_case_ids must be an array")
    ordered_case_ids = tuple(
        _string(item, f"ordered_case_ids[{index}]")
        for index, item in enumerate(raw_ids)
    )
    raw_revisions = record["metric_revisions"]
    if not isinstance(raw_revisions, dict):
        raise ValueError(f"{path}: metric_revisions must be an object")
    metric_revisions = {
        _string(name, "metric name"): _string(revision, f"metric {name!r}")
        for name, revision in raw_revisions.items()
    }
    raw_metadata = record["metadata"]
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"{path}: metadata must be an object")
    manifest = EvaluationRunManifest(
        system_id=_string(record["system_id"], "system_id"),
        cases_fingerprint=_string(record["cases_fingerprint"], "cases_fingerprint"),
        results_fingerprint=_string(record["results_fingerprint"], "results_fingerprint"),
        recorded_answers_fingerprint=_string(
            record["recorded_answers_fingerprint"], "recorded_answers_fingerprint"
        ),
        ordered_case_ids=ordered_case_ids,
        metric_revisions=metric_revisions,
        metadata=cast(dict[str, Any], raw_metadata),
    )
    supplied_fingerprint = _string(
        record["manifest_fingerprint"], "manifest_fingerprint"
    )
    _fingerprint(supplied_fingerprint, "manifest_fingerprint")
    if supplied_fingerprint != manifest.manifest_fingerprint:
        raise ValueError(f"{path}: manifest_fingerprint does not match canonical content")
    return manifest


def write_evaluation_run_manifest(path: Path, manifest: EvaluationRunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest.to_dict(),
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


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _string(value: Any, name: str) -> str:
    _nonempty(value, name)
    return cast(str, value)


def _unique_strings(values: Sequence[str], name: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def _fingerprint(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")


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
