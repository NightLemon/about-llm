"""Offline CLI for synthetic-data lineage, verifier, duplicate, and mixture audits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from about_llm.synthetic_data import (
    FingerprintProfile,
    MixtureComponent,
    SourceKind,
    SyntheticRecord,
    VerificationResult,
    audit_synthetic_records,
    plan_mixture,
)


def _string(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{context}: {field} must be a string")
    return value


def _integer(record: Mapping[str, Any], field: str, *, context: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {field} must be an integer")
    return value


def load_synthetic_records(path: Path) -> list[SyntheticRecord]:
    """Load versioned candidate records from JSONL."""

    records: list[SyntheticRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        context = f"{path}:{line_number}"
        try:
            value: Any = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{context}: record must be a JSON object")
        record = cast(dict[str, Any], value)
        raw_parents = record.get("parent_ids")
        if not isinstance(raw_parents, list) or not all(
            isinstance(parent, str) for parent in raw_parents
        ):
            raise ValueError(f"{context}: parent_ids must be a string array")
        raw_verifications = record.get("verifications")
        if not isinstance(raw_verifications, list):
            raise ValueError(f"{context}: verifications must be an array")
        verifications: list[VerificationResult] = []
        for index, raw_verification in enumerate(raw_verifications):
            verification_context = f"{context}: verifications[{index}]"
            if not isinstance(raw_verification, dict):
                raise ValueError(f"{verification_context} must be an object")
            passed = raw_verification.get("passed")
            if not isinstance(passed, bool):
                raise ValueError(f"{verification_context}: passed must be a boolean")
            verifications.append(
                VerificationResult(
                    verifier_id=_string(
                        raw_verification, "verifier_id", context=verification_context
                    ),
                    revision=_string(raw_verification, "revision", context=verification_context),
                    passed=passed,
                )
            )
        human_reviewed = record.get("human_reviewed", False)
        if not isinstance(human_reviewed, bool):
            raise ValueError(f"{context}: human_reviewed must be a boolean")
        records.append(
            SyntheticRecord(
                record_id=_string(record, "record_id", context=context),
                content=_string(record, "content", context=context),
                parent_ids=tuple(raw_parents),
                generator_revision=_string(record, "generator_revision", context=context),
                prompt_revision=_string(record, "prompt_revision", context=context),
                generation_round=_integer(record, "generation_round", context=context),
                verifications=tuple(verifications),
                human_reviewed=human_reviewed,
            )
        )
    if not records:
        raise ValueError(f"{path} contains no synthetic records")
    return records


def load_mixture(path: Path) -> tuple[list[MixtureComponent], int]:
    """Load a token-mixture plan from JSON."""

    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: mixture must be a JSON object")
    payload = cast(dict[str, Any], value)
    total = _integer(payload, "total_consumed_tokens", context=str(path))
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        raise ValueError(f"{path}: components must be an array")
    components: list[MixtureComponent] = []
    for index, raw_component in enumerate(raw_components):
        context = f"{path}: components[{index}]"
        if not isinstance(raw_component, dict):
            raise ValueError(f"{context} must be an object")
        source_kind_value = _string(raw_component, "source_kind", context=context)
        try:
            source_kind = SourceKind(source_kind_value)
        except ValueError as error:
            raise ValueError(f"{context}: unknown source_kind {source_kind_value!r}") from error
        weight = raw_component.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{context}: weight must be numeric")
        generation_round = raw_component.get("generation_round")
        if generation_round is not None and (
            isinstance(generation_round, bool) or not isinstance(generation_round, int)
        ):
            raise ValueError(f"{context}: generation_round must be an integer or null")
        components.append(
            MixtureComponent(
                name=_string(raw_component, "name", context=context),
                source_kind=source_kind,
                unique_tokens=_integer(raw_component, "unique_tokens", context=context),
                weight=float(weight),
                generation_round=generation_round,
            )
        )
    return components, total


def _run_audit(args: argparse.Namespace) -> int:
    report = audit_synthetic_records(
        load_synthetic_records(args.records),
        required_verifiers=args.required_verifier,
        known_parent_ids=args.known_parent_id,
        fingerprint_profile=FingerprintProfile(args.fingerprint_profile),
    )
    payload: dict[str, Any] = {
        "evidence_boundary": (
            "Eligibility means only that declared verifier gates passed. It does not prove "
            "semantic correctness, diversity, safety, licensing, or absence of collapse."
        ),
        "audit": {**asdict(report), "eligibility_rate": report.eligibility_rate},
        "mixture": None,
    }
    if args.mixture is not None:
        components, total = load_mixture(args.mixture)
        payload["mixture"] = asdict(
            plan_mixture(components, total_consumed_tokens=total)
        )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-synthetic-audit",
        description="Audit offline synthetic-data lineage, verifier gates, and mixture exposure",
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--required-verifier", action="append", required=True)
    parser.add_argument("--known-parent-id", action="append", default=[])
    parser.add_argument(
        "--fingerprint-profile",
        choices=[profile.value for profile in FingerprintProfile],
        default=FingerprintProfile.BYTE_EXACT.value,
    )
    parser.add_argument("--mixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(handler=_run_audit)
    return parser


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
