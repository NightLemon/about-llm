"""Verifiable, no-overwrite backup and restore for the SQLite RAG chunk store."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.rag.sqlite_store import _SCHEMA_SQL

SQLITE_BACKUP_MANIFEST_VERSION = "about-llm.rag-sqlite-backup.v1"
SQLITE_LOGICAL_FINGERPRINT_REVISION = "about-llm.rag-sqlite-logical.v1"
_SUPPORTED_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class SQLiteChunkBackupManifest:
    """Unsigned content binding for one physically and logically checked snapshot."""

    created_at_utc: str
    backup_sha256: str
    backup_size_bytes: int
    sqlite_schema_version: int
    source_count: int
    chunk_count: int
    logical_fingerprint: str

    def __post_init__(self) -> None:
        _validate_utc_timestamp(self.created_at_utc)
        _validate_sha256(self.backup_sha256, "backup_sha256")
        _validate_sha256(self.logical_fingerprint, "logical_fingerprint")
        for name, value in (
            ("backup_size_bytes", self.backup_size_bytes),
            ("sqlite_schema_version", self.sqlite_schema_version),
            ("source_count", self.source_count),
            ("chunk_count", self.chunk_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.backup_size_bytes == 0:
            raise ValueError("backup_size_bytes must be positive")
        if self.sqlite_schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported SQLite chunk schema version {self.sqlite_schema_version}"
            )

    def identity_dict(self) -> dict[str, object]:
        return {
            "manifest_version": SQLITE_BACKUP_MANIFEST_VERSION,
            "created_at_utc": self.created_at_utc,
            "backup_sha256": self.backup_sha256,
            "backup_size_bytes": self.backup_size_bytes,
            "sqlite_schema_version": self.sqlite_schema_version,
            "source_count": self.source_count,
            "chunk_count": self.chunk_count,
            "logical_fingerprint_revision": SQLITE_LOGICAL_FINGERPRINT_REVISION,
            "logical_fingerprint": self.logical_fingerprint,
        }

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.identity_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "manifest_fingerprint": self.manifest_fingerprint}


@dataclass(frozen=True)
class _DatabaseInspection:
    schema_version: int
    source_count: int
    chunk_count: int
    logical_fingerprint: str


def create_sqlite_chunk_backup(
    source_path: Path,
    backup_path: Path,
    manifest_path: Path,
    *,
    created_at_utc: str | None = None,
) -> SQLiteChunkBackupManifest:
    """Create a consistent snapshot and manifest without replacing any output path."""

    _validate_distinct_paths(source_path, backup_path, manifest_path)
    if not source_path.is_file():
        raise ValueError(f"source SQLite database does not exist: {source_path}")
    _require_new_file(backup_path, "backup")
    _require_new_file(manifest_path, "manifest")
    _require_existing_parent(backup_path)
    _require_existing_parent(manifest_path)

    backup_created = False
    manifest_created = False
    try:
        _reserve_private_file(backup_path)
        backup_created = True
        with closing(_open_read_only(source_path)) as source, closing(
            sqlite3.connect(backup_path, isolation_level=None)
        ) as destination:
            _inspect_connection(source)
            source.backup(destination)
            inspection = _inspect_connection(destination)
        _fsync_file(backup_path)
        manifest = SQLiteChunkBackupManifest(
            created_at_utc=(
                _current_utc_timestamp()
                if created_at_utc is None
                else created_at_utc
            ),
            backup_sha256=_file_sha256(backup_path),
            backup_size_bytes=backup_path.stat().st_size,
            sqlite_schema_version=inspection.schema_version,
            source_count=inspection.source_count,
            chunk_count=inspection.chunk_count,
            logical_fingerprint=inspection.logical_fingerprint,
        )
        _write_manifest_exclusive(manifest_path, manifest)
        manifest_created = True
        return manifest
    except BaseException:
        if manifest_created:
            manifest_path.unlink(missing_ok=True)
        if backup_created:
            backup_path.unlink(missing_ok=True)
        raise


def verify_sqlite_chunk_backup(
    backup_path: Path, manifest_path: Path
) -> SQLiteChunkBackupManifest:
    """Verify strict manifest, physical bytes, SQLite integrity, and logical rows."""

    manifest = load_sqlite_chunk_backup_manifest(manifest_path)
    if not backup_path.is_file():
        raise ValueError(f"backup SQLite database does not exist: {backup_path}")
    _verify_physical_file(backup_path, manifest)
    with closing(_open_read_only(backup_path)) as connection:
        inspection = _inspect_connection(connection)
    _verify_physical_file(backup_path, manifest)
    _match_inspection(inspection, manifest)
    return manifest


def restore_sqlite_chunk_backup(
    backup_path: Path,
    manifest_path: Path,
    target_path: Path,
) -> SQLiteChunkBackupManifest:
    """Restore a verified snapshot to a new path; never replace an existing target."""

    _validate_distinct_paths(backup_path, manifest_path, target_path)
    _require_new_file(target_path, "restore target")
    _require_existing_parent(target_path)
    manifest = verify_sqlite_chunk_backup(backup_path, manifest_path)
    target_created = False
    try:
        _reserve_private_file(target_path)
        target_created = True
        with closing(_open_read_only(backup_path)) as source, closing(
            sqlite3.connect(target_path, isolation_level=None)
        ) as destination:
            source.backup(destination)
            inspection = _inspect_connection(destination)
        _fsync_file(target_path)
        _verify_physical_file(backup_path, manifest)
        _match_inspection(inspection, manifest)
        return manifest
    except BaseException:
        if target_created:
            target_path.unlink(missing_ok=True)
        raise


def load_sqlite_chunk_backup_manifest(path: Path) -> SQLiteChunkBackupManifest:
    try:
        value: Any = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, OSError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict backup manifest: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: backup manifest must be a JSON object")
    record = cast(dict[str, Any], value)
    fields = {
        "manifest_version",
        "created_at_utc",
        "backup_sha256",
        "backup_size_bytes",
        "sqlite_schema_version",
        "source_count",
        "chunk_count",
        "logical_fingerprint_revision",
        "logical_fingerprint",
        "manifest_fingerprint",
    }
    missing = fields - set(record)
    unknown = set(record) - fields
    if missing or unknown:
        raise ValueError(
            f"{path}: backup manifest field mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    version = _required_string(record["manifest_version"], "manifest_version")
    if version != SQLITE_BACKUP_MANIFEST_VERSION:
        raise ValueError(
            f"{path}: expected manifest_version {SQLITE_BACKUP_MANIFEST_VERSION!r}, "
            f"got {version!r}"
        )
    logical_revision = _required_string(
        record["logical_fingerprint_revision"], "logical_fingerprint_revision"
    )
    if logical_revision != SQLITE_LOGICAL_FINGERPRINT_REVISION:
        raise ValueError(
            f"{path}: unsupported logical_fingerprint_revision {logical_revision!r}"
        )
    manifest = SQLiteChunkBackupManifest(
        created_at_utc=_required_string(record["created_at_utc"], "created_at_utc"),
        backup_sha256=_required_string(record["backup_sha256"], "backup_sha256"),
        backup_size_bytes=_required_int(
            record["backup_size_bytes"], "backup_size_bytes"
        ),
        sqlite_schema_version=_required_int(
            record["sqlite_schema_version"], "sqlite_schema_version"
        ),
        source_count=_required_int(record["source_count"], "source_count"),
        chunk_count=_required_int(record["chunk_count"], "chunk_count"),
        logical_fingerprint=_required_string(
            record["logical_fingerprint"], "logical_fingerprint"
        ),
    )
    supplied = _required_string(record["manifest_fingerprint"], "manifest_fingerprint")
    _validate_sha256(supplied, "manifest_fingerprint")
    if supplied != manifest.manifest_fingerprint:
        raise ValueError(f"{path}: manifest_fingerprint does not match canonical content")
    return manifest


def _inspect_connection(connection: sqlite3.Connection) -> _DatabaseInspection:
    quick_check = tuple(row[0] for row in connection.execute("PRAGMA quick_check"))
    if quick_check != ("ok",):
        raise ValueError(f"SQLite quick_check failed: {quick_check!r}")
    foreign_key_findings = tuple(connection.execute("PRAGMA foreign_key_check"))
    if foreign_key_findings:
        raise ValueError("SQLite foreign_key_check failed")
    schema_version = cast(int, connection.execute("PRAGMA user_version").fetchone()[0])
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported SQLite chunk schema version {schema_version}")
    if _schema_rows(connection) != _expected_schema_rows():
        raise ValueError("SQLite schema objects do not match chunk schema version 1")

    sources = tuple(
        connection.execute(
            "SELECT tenant_id, source_id, source_version, source_fingerprint "
            "FROM sources ORDER BY tenant_id, source_id"
        )
    )
    chunks = tuple(
        connection.execute(
            "SELECT chunk_id, tenant_id, source_id, source_version, ordinal, "
            "content_hash, text, heading_path_json, acl_json, metadata_json "
            "FROM chunks ORDER BY tenant_id, source_id, ordinal, chunk_id"
        )
    )
    _validate_rows(sources, chunks)
    logical = _logical_fingerprint(schema_version, sources, chunks)
    return _DatabaseInspection(schema_version, len(sources), len(chunks), logical)


def _schema_rows(connection: sqlite3.Connection) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    )


@lru_cache(maxsize=1)
def _expected_schema_rows() -> tuple[tuple[Any, ...], ...]:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(_SCHEMA_SQL)
        return _schema_rows(connection)


def _validate_rows(
    sources: tuple[tuple[Any, ...], ...], chunks: tuple[tuple[Any, ...], ...]
) -> None:
    source_versions: dict[tuple[str, str], str] = {}
    for row in sources:
        tenant_id = _stored_nonempty_string(row[0], "source tenant_id")
        source_id = _stored_nonempty_string(row[1], "source source_id")
        version = _stored_nonempty_string(row[2], "source version")
        fingerprint = _stored_nonempty_string(row[3], "source fingerprint")
        _validate_sha256(fingerprint, "stored source fingerprint")
        source_versions[(tenant_id, source_id)] = version

    ordinal_by_source: dict[tuple[str, str], int] = {}
    occurrence_by_source: dict[tuple[str, str, tuple[str, ...], str], int] = {}
    chunk_count_by_source: dict[tuple[str, str], int] = {}
    source_security: dict[tuple[str, str], tuple[tuple[str, ...], bytes]] = {}
    for row in chunks:
        chunk_id = _stored_nonempty_string(row[0], "chunk_id")
        tenant_id = _stored_nonempty_string(row[1], "chunk tenant_id")
        source_id = _stored_nonempty_string(row[2], "chunk source_id")
        version = _stored_nonempty_string(row[3], "chunk source_version")
        identity = tenant_id, source_id
        if source_versions.get(identity) != version:
            raise ValueError(f"chunk {chunk_id!r} has a source version mismatch")
        ordinal = row[4]
        expected_ordinal = ordinal_by_source.get(identity, 0)
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal != expected_ordinal
        ):
            raise ValueError(f"source {identity!r} has non-contiguous chunk ordinals")
        ordinal_by_source[identity] = expected_ordinal + 1
        content_hash = _stored_nonempty_string(row[5], "chunk content_hash")
        text = _stored_nonempty_string(row[6], "chunk text")
        expected_content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash != expected_content_hash:
            raise ValueError(f"chunk {chunk_id!r} content_hash does not match text")
        heading = _strict_string_array(
            row[7], "heading_path_json", require_unique=False
        )
        acl = _strict_string_array(row[8], "acl_json", require_unique=True)
        metadata = _strict_json_object(row[9], "metadata_json")
        security = acl, canonical_json_bytes(metadata)
        prior_security = source_security.setdefault(identity, security)
        if prior_security != security:
            raise ValueError(f"source {identity!r} has inconsistent ACL/metadata across chunks")
        occurrence_key = tenant_id, source_id, heading, content_hash
        occurrence = occurrence_by_source.get(occurrence_key, 0)
        occurrence_by_source[occurrence_key] = occurrence + 1
        chunk_identity = "\x1f".join(
            (tenant_id, source_id, " / ".join(heading), content_hash, str(occurrence))
        )
        expected_chunk_id = "chk_" + hashlib.sha256(
            chunk_identity.encode("utf-8")
        ).hexdigest()[:24]
        if chunk_id != expected_chunk_id:
            raise ValueError(f"stored chunk_id {chunk_id!r} does not match chunk identity")
        chunk_count_by_source[identity] = chunk_count_by_source.get(identity, 0) + 1
    empty_sources = sorted(set(source_versions) - set(chunk_count_by_source))
    if empty_sources:
        raise ValueError(f"stored sources have no chunks: {empty_sources!r}")


def _logical_fingerprint(
    schema_version: int,
    sources: tuple[tuple[Any, ...], ...],
    chunks: tuple[tuple[Any, ...], ...],
) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, canonical_json_bytes({"revision": SQLITE_LOGICAL_FINGERPRINT_REVISION}))
    _update_digest(digest, canonical_json_bytes({"schema_version": schema_version}))
    for table, rows in (("sources", sources), ("chunks", chunks)):
        _update_digest(digest, table.encode("ascii"))
        for row in rows:
            _update_digest(digest, canonical_json_bytes(list(row)))
    return "sha256:" + digest.hexdigest()


def _update_digest(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _match_inspection(
    inspection: _DatabaseInspection, manifest: SQLiteChunkBackupManifest
) -> None:
    actual = (
        inspection.schema_version,
        inspection.source_count,
        inspection.chunk_count,
        inspection.logical_fingerprint,
    )
    expected = (
        manifest.sqlite_schema_version,
        manifest.source_count,
        manifest.chunk_count,
        manifest.logical_fingerprint,
    )
    if actual != expected:
        raise ValueError("backup logical content does not match its manifest")


def _verify_physical_file(
    backup_path: Path, manifest: SQLiteChunkBackupManifest
) -> None:
    size = backup_path.stat().st_size
    if size != manifest.backup_size_bytes:
        raise ValueError(
            f"backup size mismatch: expected {manifest.backup_size_bytes}, found {size}"
        )
    fingerprint = _file_sha256(backup_path)
    if fingerprint != manifest.backup_sha256:
        raise ValueError("backup SHA-256 does not match its manifest")


def _write_manifest_exclusive(
    path: Path, manifest: SQLiteChunkBackupManifest
) -> None:
    payload = (
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _reserve_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)


def _open_read_only(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, isolation_level=None)


def _fsync_file(path: Path) -> None:
    # Windows' CRT may reject fsync on a read-only descriptor even though POSIX
    # accepts it. The file is ours and already complete, so reopen read/write.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _strict_json_loads(value: Any, label: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"stored {label} must be TEXT")
    try:
        return json.loads(
            value,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"stored {label} is invalid strict JSON: {error}") from error


def _strict_string_array(
    value: Any, label: str, *, require_unique: bool
) -> tuple[str, ...]:
    parsed = _strict_json_loads(value, label)
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) and item.strip() for item in parsed
    ):
        raise ValueError(f"stored {label} must be an array of non-empty strings")
    if require_unique and len(parsed) != len(set(parsed)):
        raise ValueError(f"stored {label} must not contain duplicates")
    return tuple(cast(list[str], parsed))


def _strict_json_object(value: Any, label: str) -> dict[str, Any]:
    parsed = _strict_json_loads(value, label)
    if not isinstance(parsed, dict):
        raise ValueError(f"stored {label} must be an object")
    return cast(dict[str, Any], parsed)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _validate_distinct_paths(*paths: Path) -> None:
    if any(not isinstance(path, Path) for path in paths):
        raise TypeError("backup paths must be pathlib.Path values")
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("source, backup, manifest, and target paths must be distinct")


def _require_new_file(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} path already exists: {path}")


def _require_existing_parent(path: Path) -> None:
    if not path.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {path.parent}")


def _stored_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"stored {label} must be a non-empty string")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be sha256:<64 lowercase hex>")


def _validate_utc_timestamp(value: str) -> None:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError("created_at_utc must use YYYY-MM-DDTHH:MM:SSZ")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
