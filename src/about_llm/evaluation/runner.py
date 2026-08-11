"""A dependency-light evaluation runner with versionable JSONL artifacts."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

Metric = Callable[["EvaluationCase", str], float]
SystemUnderTest = Callable[[str], str]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    input: str
    expected: str
    slices: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not isinstance(self.input, str) or not self.input.strip():
            raise ValueError(f"case {self.case_id!r} input cannot be empty")
        if not isinstance(self.expected, str):
            raise ValueError(f"case {self.case_id!r} expected must be a string")
        if any(
            not isinstance(slice_name, str) or not slice_name.strip()
            for slice_name in self.slices
        ):
            raise ValueError(f"case {self.case_id!r} contains an empty slice")
        if len(self.slices) != len(set(self.slices)):
            raise ValueError(f"case {self.case_id!r} contains duplicate slices")
        metadata = json.loads(canonical_json_bytes(self.metadata))
        if not isinstance(metadata, dict):
            raise ValueError("evaluation case metadata must be an object")
        object.__setattr__(self, "metadata", _freeze(cast(dict[str, Any], metadata)))

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "expected": self.expected,
            "slices": list(self.slices),
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationCase:
        required = {"case_id", "input", "expected"}
        _validate_fields(
            value,
            required=required,
            optional={"slices", "metadata"},
            label="evaluation case",
        )
        case_id, input_text, expected = (value[name] for name in ("case_id", "input", "expected"))
        if not all(isinstance(item, str) for item in (case_id, input_text, expected)):
            raise ValueError("evaluation case id/input/expected must be strings")
        raw_slices = value.get("slices", [])
        if not isinstance(raw_slices, list) or not all(
            isinstance(item, str) for item in raw_slices
        ):
            raise ValueError("evaluation case slices must be a string array")
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping) or not all(
            isinstance(key, str) for key in raw_metadata
        ):
            raise ValueError("evaluation case metadata must be an object")
        return cls(
            case_id=case_id,
            input=input_text,
            expected=expected,
            slices=tuple(raw_slices),
            metadata=dict(raw_metadata),
        )


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    output: str
    scores: Mapping[str, float]
    latency_seconds: float
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("result case_id cannot be empty")
        if not isinstance(self.output, str):
            raise ValueError("result output must be a string")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise ValueError("result error must be a non-empty string or null")
        if (
            isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, (int, float))
            or not math.isfinite(self.latency_seconds)
            or self.latency_seconds < 0
        ):
            raise ValueError("result latency_seconds must be finite and non-negative")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
            for name, score in self.scores.items()
        ):
            raise ValueError("result scores need names and finite values in [0, 1]")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "output": self.output,
            "scores": dict(self.scores),
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationResult:
        required = {"case_id", "output", "scores", "latency_seconds"}
        _validate_fields(
            value,
            required=required,
            optional={"error"},
            label="evaluation result",
        )
        case_id = value["case_id"]
        output = value["output"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("evaluation result case_id must be a non-empty string")
        if not isinstance(output, str):
            raise ValueError("evaluation result output must be a string")
        scores = value["scores"]
        if not isinstance(scores, Mapping) or not all(
            isinstance(name, str)
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            for name, score in scores.items()
        ):
            raise ValueError("evaluation result scores must be a numeric mapping")
        latency = value["latency_seconds"]
        if not isinstance(latency, (int, float)) or isinstance(latency, bool):
            raise ValueError("evaluation result latency_seconds must be numeric")
        error = value.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("evaluation result error must be a string or null")
        return cls(
            case_id=case_id,
            output=output,
            scores={str(name): float(score) for name, score in scores.items()},
            latency_seconds=float(latency),
            error=error,
        )


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load JSONL cases and reject malformed or duplicate ids."""
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for line_number, value in read_strict_jsonl_objects(path):
        case = EvaluationCase.from_mapping(value)
        if case.case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    return cases


def load_results(path: Path) -> list[EvaluationResult]:
    """Load versionable JSONL results and reject malformed or duplicate ids."""
    results: list[EvaluationResult] = []
    seen: set[str] = set()
    for line_number, value in read_strict_jsonl_objects(path):
        result = EvaluationResult.from_mapping(value)
        if result.case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {result.case_id!r}")
        seen.add(result.case_id)
        results.append(result)
    if not results:
        raise ValueError(f"{path} contains no evaluation results")
    return results


def run_evaluation(
    cases: Iterable[EvaluationCase],
    system: SystemUnderTest,
    metrics: Mapping[str, Metric],
    *,
    fail_fast: bool = False,
) -> list[EvaluationResult]:
    """Run all cases in stable order and retain failures as first-class results."""
    if not metrics:
        raise ValueError("at least one metric is required")
    results: list[EvaluationResult] = []
    for case in cases:
        started_at = time.perf_counter()
        try:
            output = system(case.input)
            if not isinstance(output, str):
                raise TypeError(f"system output must be str, got {type(output).__name__}")
            scores = {name: float(metric(case, output)) for name, metric in metrics.items()}
            if any(not 0 <= score <= 1 for score in scores.values()):
                raise ValueError("metric scores must be in [0, 1]")
            error = None
        except Exception as exception:
            if fail_fast:
                raise
            output = ""
            scores = {}
            error = f"{type(exception).__name__}: {exception}"
        results.append(
            EvaluationResult(
                case_id=case.case_id,
                output=output,
                scores=scores,
                latency_seconds=time.perf_counter() - started_at,
                error=error,
            )
        )
    return results


def write_results(path: Path, results: Iterable[EvaluationResult]) -> None:
    """Atomically write JSONL results so interrupted runs do not look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


def evaluation_cases_fingerprint(cases: Iterable[EvaluationCase]) -> str:
    """Identify ordered case semantics, including input, expected text, slices, and metadata."""
    snapshot = tuple(cases)
    if not snapshot:
        raise ValueError("evaluation cases must not be empty")
    return "sha256:" + artifact_fingerprint(
        {"ordered_cases": [case.to_dict() for case in snapshot]}
    )


def evaluation_results_fingerprint(results: Iterable[EvaluationResult]) -> str:
    """Identify ordered outputs, scores, latency values, and error terminals."""
    snapshot = tuple(results)
    if not snapshot:
        raise ValueError("evaluation results must not be empty")
    return "sha256:" + artifact_fingerprint(
        {"ordered_results": [result.to_dict() for result in snapshot]}
    )


def read_strict_jsonl_objects(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read standard JSON objects, rejecting duplicate keys and non-finite constants."""
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: invalid strict JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record must be a JSON object")
        records.append((line_number, cast(dict[str, Any], value)))
    return records


def _validate_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise ValueError(
            f"{label} field mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


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
