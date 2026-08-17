"""Offline CLI for synthetic-data lineage, verifier, duplicate, and mixture audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.synthetic_data import (
    FingerprintProfile,
    MixtureComponent,
    SourceKind,
    SyntheticRecord,
    VerificationResult,
    audit_synthetic_records,
    plan_mixture,
)

SYNTHETIC_AUDIT_ARTIFACT_VERSION = "about-llm.synthetic-data-audit.v2"
MAX_RECORDS_BYTES = 64 * 1024 * 1024
MAX_MIXTURE_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024 * 1024

_RECORD_FIELDS = {
    "record_id",
    "content",
    "parent_ids",
    "generator_revision",
    "prompt_revision",
    "generation_round",
    "verifications",
    "human_reviewed",
}
_VERIFICATION_FIELDS = {"verifier_id", "revision", "passed"}
_MIXTURE_FIELDS = {"total_consumed_tokens", "components"}
_COMPONENT_FIELDS = {
    "name",
    "source_kind",
    "unique_tokens",
    "weight",
    "generation_round",
}


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_utf8(raw: bytes, *, context: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context}: invalid UTF-8") from error


def _read_input_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{label} exceeds byte limit {max_bytes}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{label} changed while being read")
    return raw


def _strict_json(text: str, *, context: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context}: invalid strict JSON: {error}") from error


def _expect_fields(
    record: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    context: str,
) -> None:
    fields = set(record)
    missing = sorted(required - fields)
    unknown = sorted(fields - allowed)
    if missing or unknown:
        raise ValueError(
            f"{context}: field set mismatch; missing={missing}, unknown={unknown}"
        )


def _input_identity(raw: bytes) -> dict[str, object]:
    return {
        "size_bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _string(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{context}: {field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"{context}: {field} contains an unpaired Unicode surrogate"
        ) from error
    return value


def _integer(record: Mapping[str, Any], field: str, *, context: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {field} must be an integer")
    return value


def _load_synthetic_records_bytes(
    raw: bytes, *, context: str
) -> list[SyntheticRecord]:
    records: list[SyntheticRecord] = []
    text = _decode_utf8(raw, context=context)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        line_context = f"{context}:{line_number}"
        value = _strict_json(line, context=line_context)
        if not isinstance(value, dict):
            raise ValueError(f"{line_context}: record must be a JSON object")
        record = cast(dict[str, Any], value)
        _expect_fields(
            record,
            required=_RECORD_FIELDS - {"human_reviewed"},
            allowed=_RECORD_FIELDS,
            context=line_context,
        )
        raw_parents = record.get("parent_ids")
        if not isinstance(raw_parents, list) or not all(
            isinstance(parent, str) for parent in raw_parents
        ):
            raise ValueError(f"{line_context}: parent_ids must be a string array")
        for index, parent in enumerate(raw_parents):
            try:
                parent.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(
                    f"{line_context}: parent_ids[{index}] contains an "
                    "unpaired Unicode surrogate"
                ) from error
        raw_verifications = record.get("verifications")
        if not isinstance(raw_verifications, list):
            raise ValueError(f"{line_context}: verifications must be an array")
        verifications: list[VerificationResult] = []
        for index, raw_verification in enumerate(raw_verifications):
            verification_context = f"{line_context}: verifications[{index}]"
            if not isinstance(raw_verification, dict):
                raise ValueError(f"{verification_context} must be an object")
            _expect_fields(
                raw_verification,
                required=_VERIFICATION_FIELDS,
                allowed=_VERIFICATION_FIELDS,
                context=verification_context,
            )
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
            raise ValueError(f"{line_context}: human_reviewed must be a boolean")
        records.append(
            SyntheticRecord(
                record_id=_string(record, "record_id", context=line_context),
                content=_string(record, "content", context=line_context),
                parent_ids=tuple(raw_parents),
                generator_revision=_string(
                    record, "generator_revision", context=line_context
                ),
                prompt_revision=_string(
                    record, "prompt_revision", context=line_context
                ),
                generation_round=_integer(
                    record, "generation_round", context=line_context
                ),
                verifications=tuple(verifications),
                human_reviewed=human_reviewed,
            )
        )
    if not records:
        raise ValueError(f"{context} contains no synthetic records")
    return records


def load_synthetic_records(path: Path) -> list[SyntheticRecord]:
    """Load a size-bounded strict-JSONL synthetic candidate set."""

    raw = _read_input_bytes(
        path,
        max_bytes=MAX_RECORDS_BYTES,
        label="synthetic records",
    )
    return _load_synthetic_records_bytes(raw, context=str(path))


def _load_mixture_bytes(
    raw: bytes, *, context: str
) -> tuple[list[MixtureComponent], int]:
    value = _strict_json(_decode_utf8(raw, context=context), context=context)
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mixture must be a JSON object")
    payload = cast(dict[str, Any], value)
    _expect_fields(
        payload,
        required=_MIXTURE_FIELDS,
        allowed=_MIXTURE_FIELDS,
        context=context,
    )
    total = _integer(payload, "total_consumed_tokens", context=context)
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        raise ValueError(f"{context}: components must be an array")
    components: list[MixtureComponent] = []
    for index, raw_component in enumerate(raw_components):
        component_context = f"{context}: components[{index}]"
        if not isinstance(raw_component, dict):
            raise ValueError(f"{component_context} must be an object")
        _expect_fields(
            raw_component,
            required=_COMPONENT_FIELDS - {"generation_round"},
            allowed=_COMPONENT_FIELDS,
            context=component_context,
        )
        source_kind_value = _string(
            raw_component, "source_kind", context=component_context
        )
        try:
            source_kind = SourceKind(source_kind_value)
        except ValueError as error:
            raise ValueError(
                f"{component_context}: unknown source_kind {source_kind_value!r}"
            ) from error
        weight = raw_component.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{component_context}: weight must be numeric")
        generation_round = raw_component.get("generation_round")
        if generation_round is not None and (
            isinstance(generation_round, bool) or not isinstance(generation_round, int)
        ):
            raise ValueError(
                f"{component_context}: generation_round must be an integer or null"
            )
        components.append(
            MixtureComponent(
                name=_string(raw_component, "name", context=component_context),
                source_kind=source_kind,
                unique_tokens=_integer(
                    raw_component, "unique_tokens", context=component_context
                ),
                weight=float(weight),
                generation_round=generation_round,
            )
        )
    return components, total


def load_mixture(path: Path) -> tuple[list[MixtureComponent], int]:
    """Load a size-bounded strict-JSON token-mixture plan."""

    raw = _read_input_bytes(
        path,
        max_bytes=MAX_MIXTURE_BYTES,
        label="mixture plan",
    )
    return _load_mixture_bytes(raw, context=str(path))


def _build_synthetic_audit_artifact(
    *,
    records_raw: bytes,
    records_context: str,
    required_verifiers: Sequence[str],
    known_parent_ids: Sequence[str],
    fingerprint_profile: FingerprintProfile,
    mixture_raw: bytes | None,
    mixture_context: str,
) -> dict[str, Any]:
    required = tuple(sorted(required_verifiers))
    known = tuple(sorted(known_parent_ids))
    report = audit_synthetic_records(
        _load_synthetic_records_bytes(records_raw, context=records_context),
        required_verifiers=required,
        known_parent_ids=known,
        fingerprint_profile=fingerprint_profile,
    )
    payload: dict[str, Any] = {
        "schema_version": SYNTHETIC_AUDIT_ARTIFACT_VERSION,
        "evidence_boundary": (
            "Eligibility means only that declared verifier gates passed. It does not prove "
            "semantic correctness, verifier independence or calibration, diversity, safety, "
            "licensing, privacy, absence of leakage or collapse, observed training exposure, "
            "or downstream training benefit. Unkeyed hashes do not authenticate provenance."
        ),
        "inputs": {
            "records": _input_identity(records_raw),
            "mixture": None if mixture_raw is None else _input_identity(mixture_raw),
        },
        "policy": {
            "required_verifiers": list(required),
            "known_parent_ids": list(known),
            "fingerprint_profile": fingerprint_profile.value,
            "unresolved_lineage_affects_verifier_eligibility": False,
            "duplicate_content_affects_verifier_eligibility": False,
            "revision_overlap_affects_verifier_eligibility": False,
        },
        "audit": {**asdict(report), "eligibility_rate": report.eligibility_rate},
        "mixture": None,
        "scope": {
            "strict_json_duplicate_nonfinite_unknown_field_rejection": True,
            "input_bytes_and_external_policy_bound": True,
            "full_local_recomputation_verifier_available": True,
            "teacher_student_or_verifier_model_executed": False,
            "semantic_near_duplicate_detector_executed": False,
            "license_consent_privacy_or_secret_review_executed": False,
            "training_or_observed_token_ledger_executed": False,
            "quality_diversity_collapse_or_downstream_benefit_proved": False,
            "source_authentication_or_signature_verified": False,
            "atomic_directory_publication_or_verify_use_toctou_prevented": False,
        },
    }
    if mixture_raw is not None:
        components, total = _load_mixture_bytes(
            mixture_raw,
            context=mixture_context,
        )
        payload["mixture"] = asdict(
            plan_mixture(components, total_consumed_tokens=total)
        )
    payload["report_fingerprint"] = "sha256:" + artifact_fingerprint(payload)
    return payload


def verify_synthetic_audit_artifact(
    report_path: Path,
    *,
    records_path: Path,
    required_verifiers: Sequence[str],
    known_parent_ids: Sequence[str] = (),
    fingerprint_profile: FingerprintProfile = FingerprintProfile.BYTE_EXACT,
    mixture_path: Path | None = None,
) -> dict[str, Any]:
    """Strictly reload and fully recompute an audit from caller-supplied inputs."""

    report_raw = _read_input_bytes(
        report_path,
        max_bytes=MAX_REPORT_BYTES,
        label="synthetic audit report",
    )
    value = _strict_json(
        _decode_utf8(report_raw, context=str(report_path)),
        context=str(report_path),
    )
    if not isinstance(value, dict):
        raise ValueError("synthetic audit report must be a JSON object")
    records_raw = _read_input_bytes(
        records_path,
        max_bytes=MAX_RECORDS_BYTES,
        label="synthetic records",
    )
    mixture_raw = (
        None
        if mixture_path is None
        else _read_input_bytes(
            mixture_path,
            max_bytes=MAX_MIXTURE_BYTES,
            label="mixture plan",
        )
    )
    expected = _build_synthetic_audit_artifact(
        records_raw=records_raw,
        records_context=str(records_path),
        required_verifiers=required_verifiers,
        known_parent_ids=known_parent_ids,
        fingerprint_profile=fingerprint_profile,
        mixture_raw=mixture_raw,
        mixture_context="<no-mixture>" if mixture_path is None else str(mixture_path),
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError(
            "synthetic audit report does not match full local recomputation "
            "from caller-supplied inputs and policy"
        )
    return cast(dict[str, Any], value)


def _write_artifact(
    path: Path, rendered: str, *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_audit(args: argparse.Namespace) -> int:
    fingerprint_profile = FingerprintProfile(args.fingerprint_profile)
    if args.verify_report is not None:
        if args.output is not None or args.overwrite:
            raise ValueError(
                "--verify-report cannot be combined with --output or --overwrite"
            )
        report = verify_synthetic_audit_artifact(
            args.verify_report,
            records_path=args.records,
            required_verifiers=args.required_verifier,
            known_parent_ids=args.known_parent_id,
            fingerprint_profile=fingerprint_profile,
            mixture_path=args.mixture,
        )
        print(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "report_fingerprint": report["report_fingerprint"],
                    "verification_scope": "full_local_recomputation",
                    "verified": True,
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    records_raw = _read_input_bytes(
        args.records,
        max_bytes=MAX_RECORDS_BYTES,
        label="synthetic records",
    )
    mixture_raw = (
        None
        if args.mixture is None
        else _read_input_bytes(
            args.mixture,
            max_bytes=MAX_MIXTURE_BYTES,
            label="mixture plan",
        )
    )
    payload = _build_synthetic_audit_artifact(
        records_raw=records_raw,
        records_context=str(args.records),
        required_verifiers=args.required_verifier,
        known_parent_ids=args.known_parent_id,
        fingerprint_profile=fingerprint_profile,
        mixture_raw=mixture_raw,
        mixture_context="<no-mixture>" if args.mixture is None else str(args.mixture),
    )
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    if args.output is not None:
        _write_artifact(args.output, rendered, overwrite=args.overwrite)
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
    parser.add_argument(
        "--verify-report",
        type=Path,
        help="strictly reload and fully recompute an existing report",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow --output to replace an existing file; default is exclusive create",
    )
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
