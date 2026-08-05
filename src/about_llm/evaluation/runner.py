"""A dependency-light evaluation runner with versionable JSONL artifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not self.input.strip():
            raise ValueError(f"case {self.case_id!r} input cannot be empty")
        if len(self.slices) != len(set(self.slices)):
            raise ValueError(f"case {self.case_id!r} contains duplicate slices")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationCase:
        required = {"case_id", "input", "expected"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"evaluation case missing fields: {sorted(missing)}")
        return cls(
            case_id=str(value["case_id"]),
            input=str(value["input"]),
            expected=str(value["expected"]),
            slices=tuple(str(item) for item in value.get("slices", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    output: str
    scores: Mapping[str, float]
    latency_seconds: float
    error: str | None = None


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load JSONL cases and reject malformed or duplicate ids."""
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: case must be a JSON object")
        case = EvaluationCase.from_mapping(value)
        if case.case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    return cases


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
            handle.write(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
