"""Hash-pinned, conservative evidence for selected open-model releases."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlsplit

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.model_config import (
    estimate_standard_kv_cache,
    inspect_decoder_config,
)

MODEL_RELEASE_EVIDENCE_VERSION = "about-llm.model-release-evidence.v1"
MODEL_RELEASE_EVIDENCE_REPORT_VERSION = "about-llm.model-release-evidence-report.v1"
MODEL_RELEASE_EVIDENCE_BOUNDARY = (
    "The manifest binds exact bytes at immutable upstream revisions to local "
    "semantic snapshots and conservative projections. Vendor model-card claims "
    "are not independent measurements. Config fields do not prove weight identity, "
    "executed architecture, effective context, quality, license compatibility, "
    "runtime support, memory peak, throughput, or production safety."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_RECORD_ID = re.compile(r"[a-z0-9][a-z0-9.-]{2,79}\Z")
_ALLOWED_SOURCE_HOSTS = frozenset({"huggingface.co", "raw.githubusercontent.com"})
_MAX_ARTIFACT_BYTES = 1_000_000
_MANIFEST_FIELDS = {
    "manifest_version",
    "checked_at",
    "evidence_boundary",
    "records",
}
_COMMON_RECORD_FIELDS = {
    "record_id",
    "family",
    "artifact_kind",
    "revision",
    "source_url",
    "upstream_sha256",
    "upstream_size_bytes",
    "local_snapshot_path",
    "local_snapshot_sha256",
}
_CONFIG_RECORD_FIELDS = _COMMON_RECORD_FIELDS | {
    "expected_contract",
    "kv_scenarios",
}
_CONTRACT_FIELDS = {
    "architectures",
    "attention_kind",
    "head_dim",
    "known_mla_markers_present",
    "known_moe_markers_present",
    "model_type",
    "query_heads_per_kv_head",
    "standard_kv_applicable",
}
_SCENARIO_FIELDS = {
    "batch_size",
    "element_bytes",
    "token_count",
    "expected_bytes_per_token_per_layer",
    "expected_total_bytes",
}
_PROJECTION_FIELDS = {"projection_version", "source_fragments", "vendor_reported"}
_VENDOR_FIELDS = {
    "evidence_type",
    "family",
    "gqa",
    "knowledge_cutoff",
    "parameters",
    "pretraining_tokens",
    "publisher",
    "reported_context_length",
    "shared_embeddings",
}

ArtifactFetcher = Callable[[str], bytes]


@dataclass(frozen=True)
class ReleaseEvidenceReport:
    """Deterministic verification result without embedding upstream raw content."""

    manifest_checked_at: str
    manifest_fingerprint: str
    upstream_verified: bool
    records: tuple[Mapping[str, object], ...]
    projection_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "report_version": MODEL_RELEASE_EVIDENCE_REPORT_VERSION,
            "manifest_checked_at": self.manifest_checked_at,
            "manifest_fingerprint": self.manifest_fingerprint,
            "upstream_verified": self.upstream_verified,
            "records": [dict(record) for record in self.records],
            "projection_fingerprint": self.projection_fingerprint,
            "evidence_boundary": MODEL_RELEASE_EVIDENCE_BOUNDARY,
        }


def fetch_release_artifact(url: str, *, timeout_seconds: float = 30.0) -> bytes:
    """Fetch one public, allowlisted HTTPS evidence artifact with a hard size cap."""

    _validate_source_url(url, revision=None)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json,text/markdown,text/plain;q=0.9,*/*;q=0.1",
            "User-Agent": "about-llm-release-evidence/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        _validate_source_url(final_url, revision=None)
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise ValueError("upstream Content-Length is not an integer") from error
            if declared_length < 0 or declared_length > _MAX_ARTIFACT_BYTES:
                raise ValueError("upstream artifact exceeds the download size limit")
        raw = cast(bytes, response.read(_MAX_ARTIFACT_BYTES + 1))
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError("upstream artifact exceeds the download size limit")
    return raw


def verify_model_release_evidence(
    manifest_path: Path,
    *,
    verify_upstream: bool = False,
    fetcher: ArtifactFetcher | None = None,
) -> ReleaseEvidenceReport:
    """Verify a strict local manifest and optionally exact immutable upstream bytes."""

    manifest = _load_json_file(manifest_path)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    if manifest.get("manifest_version") != MODEL_RELEASE_EVIDENCE_VERSION:
        raise ValueError("manifest: unsupported manifest_version")
    if manifest.get("evidence_boundary") != MODEL_RELEASE_EVIDENCE_BOUNDARY:
        raise ValueError("manifest: evidence_boundary drift")
    checked_at = _required_string(manifest, "checked_at", "manifest")
    try:
        date.fromisoformat(checked_at)
    except ValueError as error:
        raise ValueError("manifest.checked_at must be an ISO date") from error
    records_value = manifest.get("records")
    if not isinstance(records_value, list) or not records_value:
        raise ValueError("manifest.records must be a non-empty array")
    if len(records_value) > 100:
        raise ValueError("manifest.records exceeds the bounded record count")

    root = manifest_path.resolve().parent
    seen_ids: set[str] = set()
    record_reports: list[Mapping[str, object]] = []
    effective_fetcher = fetch_release_artifact if fetcher is None else fetcher
    for index, raw_record in enumerate(records_value):
        location = f"manifest.records[{index}]"
        record = _mapping(raw_record, location)
        record_id = _required_string(record, "record_id", location)
        if _RECORD_ID.fullmatch(record_id) is None:
            raise ValueError(f"{location}.record_id has an invalid format")
        if record_id in seen_ids:
            raise ValueError(f"{location}.record_id is duplicated")
        seen_ids.add(record_id)
        artifact_kind = _required_string(record, "artifact_kind", location)
        if artifact_kind == "transformers_config_json":
            _require_exact_fields(record, _CONFIG_RECORD_FIELDS, location)
        elif artifact_kind == "vendor_markdown_projection":
            _require_exact_fields(record, _COMMON_RECORD_FIELDS, location)
        else:
            raise ValueError(f"{location}.artifact_kind is unsupported")
        family = _required_string(record, "family", location)
        revision = _required_string(record, "revision", location)
        if _REVISION.fullmatch(revision) is None:
            raise ValueError(f"{location}.revision must be a 40-character commit id")
        source_url = _required_string(record, "source_url", location)
        _validate_source_url(source_url, revision=revision)
        upstream_sha256 = _sha256(record.get("upstream_sha256"), location)
        upstream_size_bytes = _positive_integer(
            record.get("upstream_size_bytes"), f"{location}.upstream_size_bytes"
        )
        if upstream_size_bytes > _MAX_ARTIFACT_BYTES:
            raise ValueError(f"{location}.upstream_size_bytes exceeds the size limit")
        local_snapshot_sha256 = _sha256(
            record.get("local_snapshot_sha256"), location
        )
        local_path = _contained_local_path(
            root,
            _required_string(record, "local_snapshot_path", location),
            location,
        )
        local_value = _load_json_file(local_path)
        actual_local_sha256 = _canonical_sha256(local_value)
        if not hmac.compare_digest(actual_local_sha256, local_snapshot_sha256):
            raise ValueError(f"{location}: local semantic snapshot hash mismatch")

        upstream_raw: bytes | None = None
        if verify_upstream:
            upstream_raw = effective_fetcher(source_url)
            _verify_upstream_identity(
                upstream_raw,
                expected_sha256=upstream_sha256,
                expected_size=upstream_size_bytes,
                location=location,
            )

        common_report: dict[str, object] = {
            "record_id": record_id,
            "family": family,
            "artifact_kind": artifact_kind,
            "revision": revision,
            "source_url": source_url,
            "upstream_sha256": upstream_sha256,
            "upstream_size_bytes": upstream_size_bytes,
            "local_snapshot_sha256": local_snapshot_sha256,
            "upstream_verified": verify_upstream,
        }
        if artifact_kind == "transformers_config_json":
            record_report = _verify_config_record(
                record,
                local_value,
                upstream_raw=upstream_raw,
                location=location,
                common_report=common_report,
            )
        else:
            record_report = _verify_vendor_projection(
                local_value,
                upstream_raw=upstream_raw,
                location=location,
                common_report=common_report,
            )
        record_reports.append(record_report)

    manifest_fingerprint = _canonical_sha256(manifest)
    projection = {
        "report_version": MODEL_RELEASE_EVIDENCE_REPORT_VERSION,
        "manifest_checked_at": checked_at,
        "manifest_fingerprint": manifest_fingerprint,
        "upstream_verified": verify_upstream,
        "records": record_reports,
        "evidence_boundary": MODEL_RELEASE_EVIDENCE_BOUNDARY,
    }
    projection_fingerprint = _canonical_sha256(projection)
    return ReleaseEvidenceReport(
        manifest_checked_at=checked_at,
        manifest_fingerprint=manifest_fingerprint,
        upstream_verified=verify_upstream,
        records=tuple(record_reports),
        projection_fingerprint=projection_fingerprint,
    )


def _verify_config_record(
    record: Mapping[str, Any],
    local_value: Mapping[str, Any],
    *,
    upstream_raw: bytes | None,
    location: str,
    common_report: dict[str, object],
) -> Mapping[str, object]:
    if upstream_raw is not None:
        upstream_value = _load_json_bytes(upstream_raw, f"{location}.upstream")
        if canonical_json_bytes(upstream_value) != canonical_json_bytes(local_value):
            raise ValueError(f"{location}: upstream JSON differs from local snapshot")
    inspection = inspect_decoder_config(local_value)
    layout = inspection.standard_kv_layout
    actual_contract: dict[str, object] = {
        "architectures": list(inspection.architectures),
        "attention_kind": layout.attention_kind,
        "head_dim": layout.head_dim,
        "known_mla_markers_present": bool(inspection.mla_marker_fields),
        "known_moe_markers_present": bool(inspection.moe_marker_fields),
        "model_type": inspection.model_type,
        "query_heads_per_kv_head": layout.query_heads_per_kv_head,
        "standard_kv_applicable": layout.applicable,
    }
    expected_contract = _mapping(record.get("expected_contract"), location)
    _require_exact_fields(expected_contract, _CONTRACT_FIELDS, location)
    if canonical_json_bytes(expected_contract) != canonical_json_bytes(actual_contract):
        raise ValueError(f"{location}: expected_contract does not match inspection")
    scenario_values = record.get("kv_scenarios")
    if not isinstance(scenario_values, list):
        raise ValueError(f"{location}.kv_scenarios must be an array")
    if scenario_values and not layout.applicable:
        raise ValueError(f"{location}: KV scenarios require an applicable standard layout")
    scenario_reports: list[dict[str, object]] = []
    for scenario_index, raw_scenario in enumerate(scenario_values):
        scenario_location = f"{location}.kv_scenarios[{scenario_index}]"
        scenario = _mapping(raw_scenario, scenario_location)
        _require_exact_fields(scenario, _SCENARIO_FIELDS, scenario_location)
        token_count = _positive_integer(
            scenario.get("token_count"), f"{scenario_location}.token_count"
        )
        batch_size = _positive_integer(
            scenario.get("batch_size"), f"{scenario_location}.batch_size"
        )
        element_bytes = _positive_integer(
            scenario.get("element_bytes"), f"{scenario_location}.element_bytes"
        )
        estimate = estimate_standard_kv_cache(
            inspection,
            token_count=token_count,
            batch_size=batch_size,
            element_bytes=element_bytes,
        )
        expected_per_layer = _positive_integer(
            scenario.get("expected_bytes_per_token_per_layer"),
            f"{scenario_location}.expected_bytes_per_token_per_layer",
        )
        expected_total = _positive_integer(
            scenario.get("expected_total_bytes"),
            f"{scenario_location}.expected_total_bytes",
        )
        if estimate.bytes_per_token_per_layer != expected_per_layer:
            raise ValueError(f"{scenario_location}: per-layer KV estimate drift")
        if estimate.total_bytes != expected_total:
            raise ValueError(f"{scenario_location}: total KV estimate drift")
        scenario_reports.append(estimate.to_dict())
    return {
        **common_report,
        "config_fingerprint": inspection.config_fingerprint,
        "contract": actual_contract,
        "mla_marker_fields": dict(inspection.mla_marker_fields),
        "moe_marker_fields": dict(inspection.moe_marker_fields),
        "standard_kv_estimates": scenario_reports,
        "estimate_refused": not layout.applicable,
        "estimate_refusal_reason": None if layout.applicable else layout.reason,
    }


def _verify_vendor_projection(
    local_value: Mapping[str, Any],
    *,
    upstream_raw: bytes | None,
    location: str,
    common_report: dict[str, object],
) -> Mapping[str, object]:
    _require_exact_fields(local_value, _PROJECTION_FIELDS, f"{location}.projection")
    if local_value.get("projection_version") != (
        "about-llm.vendor-model-card-projection.v1"
    ):
        raise ValueError(f"{location}: unsupported vendor projection version")
    raw_fragments = local_value.get("source_fragments")
    if not isinstance(raw_fragments, list) or not raw_fragments:
        raise ValueError(f"{location}: source_fragments must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in raw_fragments):
        raise ValueError(f"{location}: source_fragments must contain non-empty strings")
    fragments = cast(list[str], raw_fragments)
    if len(set(fragments)) != len(fragments):
        raise ValueError(f"{location}: source_fragments must be unique")
    vendor_reported = _mapping(local_value.get("vendor_reported"), location)
    _require_exact_fields(vendor_reported, _VENDOR_FIELDS, f"{location}.vendor_reported")
    _validate_vendor_reported(vendor_reported, location)
    if upstream_raw is not None:
        try:
            upstream_text = upstream_raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{location}: upstream model card is not UTF-8") from error
        missing = [fragment for fragment in fragments if fragment not in upstream_text]
        if missing:
            raise ValueError(f"{location}: source fragment is absent from upstream bytes")
    return {
        **common_report,
        "source_fragment_count": len(fragments),
        "source_fragments_verified": upstream_raw is not None,
        "vendor_reported": dict(vendor_reported),
    }


def _validate_vendor_reported(value: Mapping[str, Any], location: str) -> None:
    for name in (
        "evidence_type",
        "family",
        "knowledge_cutoff",
        "pretraining_tokens",
        "publisher",
        "reported_context_length",
    ):
        _required_string(value, name, f"{location}.vendor_reported")
    for name in ("gqa", "shared_embeddings"):
        if not isinstance(value.get(name), bool):
            raise ValueError(f"{location}.vendor_reported.{name} must be a boolean")
    parameters = value.get("parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or any(not isinstance(item, str) or not item for item in parameters)
    ):
        raise ValueError(
            f"{location}.vendor_reported.parameters must contain non-empty strings"
        )


def _verify_upstream_identity(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_size: int,
    location: str,
) -> None:
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"{location}: fetched artifact exceeds the size limit")
    if len(raw) != expected_size:
        raise ValueError(f"{location}: upstream byte length mismatch")
    actual = "sha256:" + hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected_sha256):
        raise ValueError(f"{location}: upstream SHA-256 mismatch")


def _validate_source_url(url: str, *, revision: str | None) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.hostname not in _ALLOWED_SOURCE_HOSTS
        or parsed.fragment
    ):
        raise ValueError("source URL must use an allowlisted public HTTPS origin")
    if revision is not None and revision not in parsed.path:
        raise ValueError("source URL must contain its immutable revision")


def _contained_local_path(root: Path, relative: str, location: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not relative or ".." in candidate.parts:
        raise ValueError(f"{location}.local_snapshot_path must be contained")
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{location}.local_snapshot_path escapes the manifest directory")
    if not resolved.is_file():
        raise ValueError(f"{location}.local_snapshot_path is not a file")
    return resolved


def _load_json_file(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: cannot read JSON: {error}") from error
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"{path}: JSON exceeds the size limit")
    return _load_json_bytes(raw, str(path))


def _load_json_bytes(raw: bytes, location: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{location}: JSON is not valid UTF-8") from error
    try:
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{location}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{location}: JSON root must be an object")
    canonical_json_bytes(value)
    return cast(dict[str, Any], value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{location}: field set mismatch; missing={missing}, extra={extra}"
        )


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} must use string keys")
    return cast(Mapping[str, Any], value)


def _required_string(value: Mapping[str, Any], name: str, location: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{location}.{name} must be a non-empty string")
    return result


def _positive_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return cast(int, value)


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{location}: expected canonical sha256 fingerprint")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + artifact_fingerprint(cast(Mapping[str, object], value))


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
