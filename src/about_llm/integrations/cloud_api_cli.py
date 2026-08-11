"""Offline fixture verifier for provider-specific cloud chat contracts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from about_llm.integrations.cloud_api import (
    ChatMessage,
    ChatResponse,
    RequestSpec,
    Role,
    build_anthropic_request,
    build_gemini_request,
    build_openai_compatible_request,
    parse_anthropic_response,
    parse_gemini_response,
    parse_openai_compatible_response,
)
from about_llm.integrations.retry import RetryPolicy, decide_retry

Provider = Literal[
    "openai-compatible",
    "anthropic-messages",
    "gemini-generate-content",
]
PROVIDERS: tuple[Provider, ...] = (
    "openai-compatible",
    "anthropic-messages",
    "gemini-generate-content",
)
_TOP_LEVEL_FIELDS = frozenset(
    {"case_id", "provider", "config", "messages", "response", "expected"}
)
_OFFLINE_SECRET = "offline-contract-secret-never-sent"


@dataclass(frozen=True)
class ContractCase:
    case_id: str
    provider: Provider
    config: Mapping[str, Any]
    messages: tuple[ChatMessage, ...]
    response: Mapping[str, Any]
    expected: Mapping[str, Any]


def load_contracts(path: Path) -> tuple[ContractCase, ...]:
    """Load strict JSONL contract fixtures for the supported teaching adapters."""
    cases: list[ContractCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: contract must be a JSON object")
        record = cast(dict[str, Any], value)
        unknown = sorted(set(record) - _TOP_LEVEL_FIELDS)
        missing = sorted(_TOP_LEVEL_FIELDS - set(record))
        if unknown or missing:
            raise ValueError(
                f"{path}:{line_number}: contract fields differ from schema; "
                f"unknown={unknown}, missing={missing}"
            )
        case_id = _required_string(record, "case_id", path, line_number)
        if case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        provider_value = record.get("provider")
        if provider_value not in PROVIDERS:
            raise ValueError(f"{path}:{line_number}: unsupported provider {provider_value!r}")
        config = _required_object(record, "config", path, line_number)
        response = _required_object(record, "response", path, line_number)
        expected = _required_object(record, "expected", path, line_number)
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(f"{path}:{line_number}: messages must be a non-empty array")
        messages = tuple(
            _parse_message(item, path=path, line_number=line_number) for item in raw_messages
        )
        cases.append(
            ContractCase(
                case_id=case_id,
                provider=cast(Provider, provider_value),
                config=config,
                messages=messages,
                response=response,
                expected=expected,
            )
        )
    if not cases:
        raise ValueError(f"{path} contains no contract cases")
    return tuple(cases)


def verify_contracts(cases: Sequence[ContractCase]) -> dict[str, Any]:
    """Build and parse all fixtures without importing an HTTP client."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        request = _build_request(case)
        parsed = _parse_response(case)
        actual = asdict(parsed)
        unknown_expected = sorted(set(case.expected) - set(actual))
        mismatches = [
            f"{name}: expected {expected!r}, got {actual.get(name)!r}"
            for name, expected in case.expected.items()
            if name in actual and actual[name] != expected
        ]
        if unknown_expected:
            mismatches.append(f"unknown expected field(s): {unknown_expected}")
        rows.append(
            {
                "case_id": case.case_id,
                "provider": case.provider,
                "passed": not mismatches,
                "mismatches": mismatches,
                "request": {
                    "url": request.url,
                    "headers": request.sanitized_headers(),
                    "body": dict(request.body),
                },
                "parsed_response": actual,
            }
        )
    payload = {
        "passed": all(row["passed"] for row in rows),
        "network_performed": False,
        "real_credentials_used": False,
        "case_count": len(rows),
        "cases": rows,
    }
    if _OFFLINE_SECRET in json.dumps(payload, ensure_ascii=False):
        raise RuntimeError("credential redaction invariant failed")
    return payload


def build_retry_matrix() -> dict[str, Any]:
    """Build a deterministic offline decision table for the teaching retry policy."""
    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=1,
        multiplier=2,
        max_backoff_seconds=4,
        max_retry_after_seconds=10,
    )
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cases: tuple[tuple[str, str, dict[str, Any]], ...] = (
        (
            "rate-limit-with-retry-after",
            "retryable_status",
            {"status_code": 429, "response_headers": {"Retry-After": "3"}},
        ),
        ("bad-request", "not_retryable", {"status_code": 400}),
        ("not-implemented", "not_retryable", {"status_code": 501}),
        (
            "unsafe-replay",
            "replay_unsafe",
            {"status_code": 503, "replay_safe": False},
        ),
        (
            "uncertain-timeout",
            "outcome_uncertain",
            {"error_category": "timeout", "outcome_uncertain": True},
        ),
        (
            "retry-after-over-budget",
            "retry_after_too_long",
            {"status_code": 429, "response_headers": {"Retry-After": "11"}},
        ),
        (
            "malformed-retry-after-fallback",
            "retryable_status",
            {"status_code": 503, "response_headers": {"Retry-After": "soon"}},
        ),
        (
            "deadline-too-short",
            "deadline_exhausted",
            {"status_code": 502, "remaining_seconds": 1},
        ),
        (
            "attempts-exhausted",
            "attempts_exhausted",
            {"status_code": 500, "attempt": 3},
        ),
    )
    rows: list[dict[str, Any]] = []
    for case_id, expected_reason, overrides in cases:
        inputs: dict[str, Any] = {
            "policy": policy,
            "attempt": 1,
            "replay_safe": True,
            "outcome_uncertain": False,
            "now": now,
            "jitter_fraction": 1.0,
        }
        inputs.update(overrides)
        decision = decide_retry(**inputs)
        rows.append(
            {
                "case_id": case_id,
                "expected_reason": expected_reason,
                "passed": decision.reason == expected_reason,
                "decision": asdict(decision),
            }
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "network_performed": False,
        "provider_semantics_assumed": False,
        "case_count": len(rows),
        "policy": {
            "max_attempts": policy.max_attempts,
            "base_delay_seconds": policy.base_delay_seconds,
            "multiplier": policy.multiplier,
            "max_backoff_seconds": policy.max_backoff_seconds,
            "max_retry_after_seconds": policy.max_retry_after_seconds,
            "retryable_statuses": sorted(policy.retryable_statuses),
            "retryable_errors": sorted(policy.retryable_errors),
        },
        "cases": rows,
    }


def _build_request(case: ContractCase) -> RequestSpec:
    base_url = _config_string(case, "base_url")
    model = _config_string(case, "model")
    max_tokens = _config_int(case, "max_tokens")
    temperature = _config_float(case, "temperature", default=0.0)
    if case.provider == "openai-compatible":
        return build_openai_compatible_request(
            base_url=base_url,
            api_key=_OFFLINE_SECRET,
            model=model,
            messages=case.messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if case.provider == "anthropic-messages":
        return build_anthropic_request(
            base_url=base_url,
            api_key=_OFFLINE_SECRET,
            api_version=_config_string(case, "api_version"),
            model=model,
            messages=case.messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return build_gemini_request(
        base_url=base_url,
        api_key=_OFFLINE_SECRET,
        model=model,
        messages=case.messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _parse_response(case: ContractCase) -> ChatResponse:
    if case.provider == "openai-compatible":
        return parse_openai_compatible_response(case.response)
    if case.provider == "anthropic-messages":
        return parse_anthropic_response(case.response)
    return parse_gemini_response(case.response)


def _parse_message(value: Any, *, path: Path, line_number: int) -> ChatMessage:
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: each message must be an object")
    if set(value) != {"role", "content"}:
        raise ValueError(
            f"{path}:{line_number}: message fields must be exactly role and content"
        )
    role = value.get("role")
    content = value.get("content")
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"{path}:{line_number}: unsupported message role {role!r}")
    if not isinstance(content, str):
        raise ValueError(f"{path}:{line_number}: message content must be a string")
    return ChatMessage(cast(Role, role), content)


def _required_string(
    record: Mapping[str, Any], key: str, path: Path, line_number: int
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_number}: {key} must be a non-empty string")
    return value


def _required_object(
    record: Mapping[str, Any], key: str, path: Path, line_number: int
) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        raise ValueError(f"{path}:{line_number}: {key} must be a JSON object")
    return cast(dict[str, Any], value)


def _config_string(case: ContractCase, key: str) -> str:
    value = case.config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"case {case.case_id!r}: config.{key} must be a non-empty string")
    return value


def _config_int(case: ContractCase, key: str) -> int:
    value = case.config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"case {case.case_id!r}: config.{key} must be an integer")
    return value


def _config_float(case: ContractCase, key: str, *, default: float) -> float:
    value = case.config.get(key, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"case {case.case_id!r}: config.{key} must be numeric")
    return float(value)


def _strict_json_loads(line: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard numeric constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate object key {name!r}")
            result[name] = value
        return result

    return json.loads(
        line,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _run_verify(args: argparse.Namespace) -> int:
    payload = verify_contracts(load_contracts(args.contracts))
    return _render_payload(payload, output=args.output)


def _run_retry_matrix(args: argparse.Namespace) -> int:
    return _render_payload(build_retry_matrix(), output=args.output)


def _render_payload(payload: Mapping[str, Any], *, output: Path | None) -> int:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if payload["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-cloud-contract",
        description="Build and parse cloud API fixtures without network access or credentials",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="verify strict provider fixtures")
    verify.add_argument("--contracts", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    verify.set_defaults(handler=_run_verify)
    retry = commands.add_parser(
        "retry-matrix", help="render the deterministic offline retry decision table"
    )
    retry.add_argument("--output", type=Path)
    retry.set_defaults(handler=_run_retry_matrix)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    try:
        return handler(args)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
