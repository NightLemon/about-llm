"""Offline inference-attempt analysis and transparent SLO gate."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.inference import (
    InferenceAttempt,
    RequestOutcome,
    WorkloadSLO,
    summarize_attempts,
)


def _number(record: dict[str, Any], name: str, *, context: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: {name} must be numeric")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{context}: {name} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{context}: {name} must be finite")
    return converted


def _optional_number(record: dict[str, Any], name: str, *, context: str) -> float | None:
    value = record.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: {name} must be numeric or null")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{context}: {name} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{context}: {name} must be finite")
    return converted


def _optional_integer(record: dict[str, Any], name: str, *, context: str) -> int | None:
    value = record.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {name} must be an integer or null")
    return cast(int, value)


def load_attempts(path: Path) -> list[InferenceAttempt]:
    """Load one terminal request attempt per JSONL row."""

    attempts: list[InferenceAttempt] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        context = f"{path}:{line_number}"
        try:
            value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{context}: invalid strict JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{context}: attempt must be a JSON object")
        record = cast(dict[str, Any], value)
        required = {"request_id", "outcome", "started_at", "completed_at"}
        allowed = required | {
            "offered_at",
            "first_token_at",
            "prompt_tokens",
            "output_tokens",
        }
        missing = required - set(record)
        unknown = set(record) - allowed
        if missing or unknown:
            raise ValueError(
                f"{context}: field mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        request_id = record.get("request_id")
        outcome_value = record.get("outcome")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"{context}: request_id must be a non-empty string")
        if request_id in seen:
            raise ValueError(f"{context}: duplicate request_id {request_id!r}")
        if not isinstance(outcome_value, str):
            raise ValueError(f"{context}: outcome must be a string")
        try:
            outcome = RequestOutcome(outcome_value)
        except ValueError as error:
            raise ValueError(f"{context}: unknown outcome {outcome_value!r}") from error
        seen.add(request_id)
        attempts.append(
            InferenceAttempt(
                request_id=request_id,
                outcome=outcome,
                started_at=_number(record, "started_at", context=context),
                completed_at=_number(record, "completed_at", context=context),
                first_token_at=_optional_number(record, "first_token_at", context=context),
                prompt_tokens=_optional_integer(record, "prompt_tokens", context=context),
                output_tokens=_optional_integer(record, "output_tokens", context=context),
                offered_at=_optional_number(record, "offered_at", context=context),
            )
        )
    if not attempts:
        raise ValueError(f"{path} contains no attempts")
    return attempts


def _run_analyze(args: argparse.Namespace) -> int:
    summary = summarize_attempts(
        load_attempts(args.attempts),
        benchmark_started_at=args.benchmark_started_at,
        benchmark_completed_at=args.benchmark_completed_at,
    )
    slo = WorkloadSLO(
        minimum_success_rate=args.minimum_success_rate,
        maximum_ttft_p95_seconds=args.maximum_ttft_p95,
        maximum_e2e_p95_seconds=args.maximum_e2e_p95,
        maximum_tpot_p95_seconds=args.maximum_tpot_p95,
        maximum_client_queue_p95_seconds=args.maximum_client_queue_p95,
        maximum_successful_offered_ttft_p95_seconds=(
            args.maximum_successful_offered_ttft_p95
        ),
        maximum_offered_to_terminal_p95_seconds=(
            args.maximum_offered_to_terminal_p95
        ),
    )
    passed, reasons = slo.evaluate(summary)
    payload = {
        "evidence_boundary": (
            "This report analyzes recorded client attempts only. It does not identify GPU "
            "capacity, server-side queue time, or production SLO compliance without a "
            "representative workload and target-environment trace. Client queue and "
            "offered-to-terminal metrics exist only when every row records offered_at."
        ),
        "passed": passed,
        "reasons": reasons,
        "slo": asdict(slo),
        "summary": asdict(summary),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-inference-analyze",
        description="Analyze recorded inference attempts without hiding failures",
    )
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--benchmark-started-at", type=float, required=True)
    parser.add_argument("--benchmark-completed-at", type=float, required=True)
    parser.add_argument("--minimum-success-rate", type=float, required=True)
    parser.add_argument("--maximum-ttft-p95", type=float)
    parser.add_argument("--maximum-e2e-p95", type=float)
    parser.add_argument("--maximum-tpot-p95", type=float)
    parser.add_argument("--maximum-client-queue-p95", type=float)
    parser.add_argument("--maximum-successful-offered-ttft-p95", type=float)
    parser.add_argument("--maximum-offered-to-terminal-p95", type=float)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(handler=_run_analyze)
    return parser


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    try:
        return handler(args)
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
