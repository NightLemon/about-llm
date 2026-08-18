"""Authenticated, no-overwrite snapshots for evaluation release decisions.

The ledger uses a domain-separated HMAC chain.  A valid chain authenticates
records relative to caller-supplied secret keys; it is not a public signature,
timestamp authority, key-custody proof, or append-only storage by itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, TypeAlias, cast

from about_llm.llmops import canonical_json_bytes

EVALUATION_RELEASE_LEDGER_VERSION = "about-llm.evaluation-release-ledger.v1"
EVALUATION_RELEASE_LEDGER_EVIDENCE_BOUNDARY = (
    "A successful verification authenticates the complete record chain relative to the "
    "caller-supplied HMAC keys. Artifact bytes are checked only when an exact artifact-path "
    "mapping is supplied, and tail truncation or rollback is checked only when an externally "
    "trusted head is supplied. It does not prove key custody, artifact provenance, actual "
    "execution, recorded wall-clock time, semantic validity, directory-level atomic publish, "
    "or that files remain unchanged after verification."
)

_MAC_DOMAIN = b"about-llm:evaluation-release-ledger:record:v1\x00"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_HMAC_SHA256 = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_DECISIONS = frozenset({"recorded", "approved", "rejected"})
_ROOT_FIELDS = frozenset({"ledger_version", "records", "head", "evidence_boundary"})
_HEAD_FIELDS = frozenset({"sequence", "record_mac"})
_RECORD_FIELDS = frozenset(
    {
        "sequence",
        "release_id",
        "artifact_id",
        "artifact_kind",
        "artifact_size_bytes",
        "artifact_sha256",
        "decision",
        "recorded_at",
        "key_id",
        "previous_record_mac",
        "record_mac",
    }
)

KeyResolver: TypeAlias = Mapping[str, bytes] | Callable[[str], bytes]


@dataclass(frozen=True)
class EvaluationReleaseHead:
    """Sequence and MAC anchored outside a ledger snapshot."""

    sequence: int
    record_mac: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("head sequence must be an integer")
        if self.sequence <= 0:
            raise ValueError("head sequence must be positive")
        _mac(self.record_mac, "head record_mac")

    def to_dict(self) -> dict[str, object]:
        return {"sequence": self.sequence, "record_mac": self.record_mac}


@dataclass(frozen=True)
class EvaluationReleaseRecord:
    """One artifact decision and its link to the preceding authenticated record."""

    sequence: int
    release_id: str
    artifact_id: str
    artifact_kind: str
    artifact_size_bytes: int
    artifact_sha256: str
    decision: str
    recorded_at: str
    key_id: str
    previous_record_mac: str | None
    record_mac: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("record sequence must be an integer")
        if self.sequence <= 0:
            raise ValueError("record sequence must be positive")
        for name, value in (
            ("release_id", self.release_id),
            ("artifact_id", self.artifact_id),
            ("artifact_kind", self.artifact_kind),
            ("key_id", self.key_id),
        ):
            _nonempty(value, name)
        if (
            isinstance(self.artifact_size_bytes, bool)
            or not isinstance(self.artifact_size_bytes, int)
        ):
            raise TypeError("artifact_size_bytes must be an integer")
        if self.artifact_size_bytes < 0:
            raise ValueError("artifact_size_bytes must be non-negative")
        _sha256(self.artifact_sha256, "artifact_sha256")
        if self.decision not in _DECISIONS:
            raise ValueError(f"decision must be one of {sorted(_DECISIONS)}")
        _timestamp(self.recorded_at)
        if self.previous_record_mac is not None:
            _mac(self.previous_record_mac, "previous_record_mac")
        _mac(self.record_mac, "record_mac")

    def unsigned_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "release_id": self.release_id,
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_sha256": self.artifact_sha256,
            "decision": self.decision,
            "recorded_at": self.recorded_at,
            "key_id": self.key_id,
            "previous_record_mac": self.previous_record_mac,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.unsigned_dict(), "record_mac": self.record_mac}


@dataclass(frozen=True)
class EvaluationReleaseLedger:
    """A non-empty authenticated-chain snapshot."""

    records: tuple[EvaluationReleaseRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise ValueError("release ledger must contain at least one record")
        if any(not isinstance(record, EvaluationReleaseRecord) for record in records):
            raise TypeError("release ledger records must be EvaluationReleaseRecord values")
        object.__setattr__(self, "records", records)
        release_ids: set[str] = set()
        artifact_ids: set[str] = set()
        previous: str | None = None
        for expected_sequence, record in enumerate(records, start=1):
            if record.sequence != expected_sequence:
                raise ValueError(
                    "record sequence must be contiguous from 1: "
                    f"expected {expected_sequence}, got {record.sequence}"
                )
            if record.previous_record_mac != previous:
                raise ValueError(
                    f"record {record.sequence} previous_record_mac does not match chain"
                )
            if record.release_id in release_ids:
                raise ValueError(f"duplicate release_id {record.release_id!r}")
            if record.artifact_id in artifact_ids:
                raise ValueError(f"duplicate artifact_id {record.artifact_id!r}")
            release_ids.add(record.release_id)
            artifact_ids.add(record.artifact_id)
            previous = record.record_mac

    @property
    def head(self) -> EvaluationReleaseHead:
        final = self.records[-1]
        return EvaluationReleaseHead(final.sequence, final.record_mac)

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_version": EVALUATION_RELEASE_LEDGER_VERSION,
            "records": [record.to_dict() for record in self.records],
            "head": self.head.to_dict(),
            "evidence_boundary": EVALUATION_RELEASE_LEDGER_EVIDENCE_BOUNDARY,
        }


@dataclass(frozen=True)
class EvaluationReleaseVerification:
    """Machine-readable statement of exactly which checks were executed."""

    record_count: int
    head: EvaluationReleaseHead
    referenced_artifacts_rehashed: bool
    trusted_head_matched: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "record_count": self.record_count,
            "head": self.head.to_dict(),
            "authenticated_chain": True,
            "recorded_at_bytes_mac_bound": True,
            "referenced_artifacts_rehashed": self.referenced_artifacts_rehashed,
            "trusted_head_matched": self.trusted_head_matched,
            "key_custody_verified": False,
            "timestamp_authority_verified": False,
            "artifact_provenance_verified": False,
            "directory_atomic_publish_verified": False,
            "post_verification_immutability_verified": False,
            "evidence_boundary": EVALUATION_RELEASE_LEDGER_EVIDENCE_BOUNDARY,
        }


def append_evaluation_release_record(
    ledger: EvaluationReleaseLedger | None,
    *,
    release_id: str,
    artifact_id: str,
    artifact_kind: str,
    artifact_bytes: bytes,
    decision: str,
    recorded_at: str,
    key_id: str,
    secret_key: bytes,
) -> EvaluationReleaseLedger:
    """Return a new snapshot with one record appended; never persist the key."""

    if not isinstance(artifact_bytes, bytes):
        raise TypeError("artifact_bytes must be bytes")
    key = _secret_key(secret_key, key_id)
    records = () if ledger is None else ledger.records
    sequence = len(records) + 1
    previous_record_mac = None if not records else records[-1].record_mac
    unsigned: dict[str, object] = {
        "sequence": sequence,
        "release_id": release_id,
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "artifact_size_bytes": len(artifact_bytes),
        "artifact_sha256": "sha256:" + hashlib.sha256(artifact_bytes).hexdigest(),
        "decision": decision,
        "recorded_at": recorded_at,
        "key_id": key_id,
        "previous_record_mac": previous_record_mac,
    }
    record = EvaluationReleaseRecord(
        sequence=sequence,
        release_id=release_id,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        artifact_size_bytes=len(artifact_bytes),
        artifact_sha256=cast(str, unsigned["artifact_sha256"]),
        decision=decision,
        recorded_at=recorded_at,
        key_id=key_id,
        previous_record_mac=previous_record_mac,
        record_mac=_record_mac(unsigned, key),
    )
    return EvaluationReleaseLedger((*records, record))


def append_evaluation_release_artifact(
    ledger: EvaluationReleaseLedger | None,
    *,
    release_id: str,
    artifact_id: str,
    artifact_kind: str,
    artifact_path: Path,
    decision: str,
    recorded_at: str,
    key_id: str,
    secret_key: bytes,
) -> EvaluationReleaseLedger:
    """Read one artifact as bytes and append its exact byte identity."""

    return append_evaluation_release_record(
        ledger,
        release_id=release_id,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        artifact_bytes=artifact_path.read_bytes(),
        decision=decision,
        recorded_at=recorded_at,
        key_id=key_id,
        secret_key=secret_key,
    )


def verify_evaluation_release_ledger(
    ledger: EvaluationReleaseLedger,
    *,
    key_resolver: KeyResolver,
    artifact_paths: Mapping[str, Path] | None = None,
    trusted_head: EvaluationReleaseHead | None = None,
) -> EvaluationReleaseVerification:
    """Verify MACs and optionally exact artifact bytes and an external head."""

    for record in ledger.records:
        key = _resolve_key(key_resolver, record.key_id)
        expected = _record_mac(record.unsigned_dict(), key)
        if not hmac.compare_digest(record.record_mac, expected):
            raise ValueError(f"record {record.sequence} MAC verification failed")

    if artifact_paths is not None:
        expected_ids = {record.artifact_id for record in ledger.records}
        supplied_ids = set(artifact_paths)
        if supplied_ids != expected_ids:
            raise ValueError(
                "artifact path mapping must exactly match ledger artifact ids: "
                f"missing={sorted(expected_ids - supplied_ids)}, "
                f"unknown={sorted(supplied_ids - expected_ids)}"
            )
        for record in ledger.records:
            artifact_bytes = artifact_paths[record.artifact_id].read_bytes()
            actual_size = len(artifact_bytes)
            actual_sha256 = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
            if actual_size != record.artifact_size_bytes:
                raise ValueError(
                    f"artifact {record.artifact_id!r} size does not match release record"
                )
            if not hmac.compare_digest(actual_sha256, record.artifact_sha256):
                raise ValueError(
                    f"artifact {record.artifact_id!r} SHA-256 does not match release record"
                )

    if trusted_head is not None and ledger.head != trusted_head:
        raise ValueError(
            "ledger head does not match externally trusted head: "
            f"expected={trusted_head.to_dict()}, actual={ledger.head.to_dict()}"
        )
    return EvaluationReleaseVerification(
        record_count=len(ledger.records),
        head=ledger.head,
        referenced_artifacts_rehashed=artifact_paths is not None,
        trusted_head_matched=trusted_head is not None,
    )


def load_evaluation_release_ledger(path: Path) -> EvaluationReleaseLedger:
    """Load exact canonical JSON (one trailing LF) and reject ambiguous JSON."""

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: ledger must be UTF-8") from error
    try:
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: ledger must be a JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{path}: ledger is not canonical JSON with one trailing LF")
    root = cast(dict[str, Any], value)
    _exact_fields(root, _ROOT_FIELDS, f"{path}: ledger")
    if root["ledger_version"] != EVALUATION_RELEASE_LEDGER_VERSION:
        raise ValueError(
            f"{path}: ledger_version must equal {EVALUATION_RELEASE_LEDGER_VERSION!r}"
        )
    if root["evidence_boundary"] != EVALUATION_RELEASE_LEDGER_EVIDENCE_BOUNDARY:
        raise ValueError(f"{path}: evidence_boundary does not match ledger version")
    raw_records = root["records"]
    if not isinstance(raw_records, list):
        raise ValueError(f"{path}: records must be an array")
    records = tuple(
        _parse_record(item, index=index, path=path)
        for index, item in enumerate(raw_records)
    )
    ledger = EvaluationReleaseLedger(records)
    raw_head = root["head"]
    if not isinstance(raw_head, dict):
        raise ValueError(f"{path}: head must be an object")
    _exact_fields(raw_head, _HEAD_FIELDS, f"{path}: head")
    supplied_head = EvaluationReleaseHead(
        sequence=_integer(raw_head["sequence"], "head.sequence"),
        record_mac=_string(raw_head["record_mac"], "head.record_mac"),
    )
    if supplied_head != ledger.head:
        raise ValueError(f"{path}: head does not match final record")
    return ledger


def write_evaluation_release_ledger(path: Path, ledger: EvaluationReleaseLedger) -> None:
    """Exclusive-create and file-fsync one canonical snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(ledger.to_dict()) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _parse_record(value: Any, *, index: int, path: Path) -> EvaluationReleaseRecord:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: records[{index}] must be an object")
    _exact_fields(value, _RECORD_FIELDS, f"{path}: records[{index}]")
    previous = value["previous_record_mac"]
    if previous is not None:
        previous = _string(previous, f"records[{index}].previous_record_mac")
    return EvaluationReleaseRecord(
        sequence=_integer(value["sequence"], f"records[{index}].sequence"),
        release_id=_string(value["release_id"], f"records[{index}].release_id"),
        artifact_id=_string(value["artifact_id"], f"records[{index}].artifact_id"),
        artifact_kind=_string(
            value["artifact_kind"], f"records[{index}].artifact_kind"
        ),
        artifact_size_bytes=_integer(
            value["artifact_size_bytes"], f"records[{index}].artifact_size_bytes"
        ),
        artifact_sha256=_string(
            value["artifact_sha256"], f"records[{index}].artifact_sha256"
        ),
        decision=_string(value["decision"], f"records[{index}].decision"),
        recorded_at=_string(value["recorded_at"], f"records[{index}].recorded_at"),
        key_id=_string(value["key_id"], f"records[{index}].key_id"),
        previous_record_mac=cast(str | None, previous),
        record_mac=_string(value["record_mac"], f"records[{index}].record_mac"),
    )


def _record_mac(unsigned: Mapping[str, object], key: bytes) -> str:
    digest = hmac.new(key, _MAC_DOMAIN + canonical_json_bytes(unsigned), hashlib.sha256)
    return "hmac-sha256:" + digest.hexdigest()


def _resolve_key(resolver: KeyResolver, key_id: str) -> bytes:
    try:
        key = resolver[key_id] if isinstance(resolver, Mapping) else resolver(key_id)
    except KeyError as error:
        raise ValueError(f"no verification key for key_id {key_id!r}") from error
    return _secret_key(key, key_id)


def _secret_key(value: Any, key_id: str) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"secret key {key_id!r} must be bytes")
    if len(value) < 32:
        raise ValueError(f"secret key {key_id!r} must contain at least 32 bytes")
    return value


def _timestamp(value: Any) -> None:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise ValueError("recorded_at must be an RFC 3339 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("recorded_at must be a valid RFC 3339 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("recorded_at must include a timezone")


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _string(value: Any, name: str) -> str:
    _nonempty(value, name)
    return cast(str, value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return cast(int, value)


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256:<64 lowercase hex>")


def _mac(value: Any, name: str) -> None:
    if not isinstance(value, str) or _HMAC_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be hmac-sha256:<64 lowercase hex>")


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{location} field mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
