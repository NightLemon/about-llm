"""Transactional SQLite persistence for deterministic RAG source chunks."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from about_llm.rag.ingestion import (
    IngestionPlan,
    SourceChunk,
    SourceDocument,
    plan_incremental_update,
    split_markdown,
)

_SCHEMA_VERSION = 1
_CHUNKING_REVISION = "markdown-v1"


class SQLiteChunkStore:
    """One-process connection with transactional, version-checked source updates."""

    def __init__(self, path: Path, *, busy_timeout_ms: int = 5000) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
        if (
            not isinstance(busy_timeout_ms, int)
            or isinstance(busy_timeout_ms, bool)
            or busy_timeout_ms < 0
        ):
            raise ValueError("busy_timeout_ms must be a non-negative integer")
        self.path = path
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self._closed = False
        self._initialize_schema()

    def __enter__(self) -> SQLiteChunkStore:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def current_version(self, *, tenant_id: str, source_id: str) -> str | None:
        self._validate_identity(tenant_id, source_id)
        self._require_open()
        row = self._connection.execute(
            "SELECT source_version FROM sources WHERE tenant_id = ? AND source_id = ?",
            (tenant_id, source_id),
        ).fetchone()
        return None if row is None else cast(str, row[0])

    def upsert_source(
        self,
        source: SourceDocument,
        *,
        expected_current_version: str | None,
        max_chars: int = 1200,
    ) -> IngestionPlan:
        """Atomically replace one source; ``None`` means the source must be absent."""
        self._require_open()
        desired = tuple(split_markdown(source, max_chars=max_chars))
        if not desired:
            raise ValueError("source produced no chunks; use delete_source explicitly")
        fingerprint = _source_fingerprint(source, max_chars=max_chars)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT source_version, source_fingerprint FROM sources "
                "WHERE tenant_id = ? AND source_id = ?",
                (source.tenant_id, source.source_id),
            ).fetchone()
            actual_version = None if row is None else cast(str, row[0])
            if actual_version != expected_current_version:
                raise ValueError(
                    "source version conflict: expected "
                    f"{expected_current_version!r}, found {actual_version!r}"
                )
            if row is not None and actual_version == source.version and row[1] != fingerprint:
                raise ValueError("source version cannot be reused for different content")
            existing = self._load_source_chunks(source.tenant_id, source.source_id)
            plan = plan_incremental_update(existing, desired)
            self._connection.executemany(
                "DELETE FROM chunks WHERE chunk_id = ?",
                ((chunk_id,) for chunk_id in plan.delete_chunk_ids),
            )
            self._connection.execute(
                "INSERT INTO sources(tenant_id, source_id, source_version, source_fingerprint) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(tenant_id, source_id) DO UPDATE SET "
                "source_version=excluded.source_version, "
                "source_fingerprint=excluded.source_fingerprint",
                (source.tenant_id, source.source_id, source.version, fingerprint),
            )
            self._connection.executemany(
                _UPSERT_CHUNK_SQL,
                (_chunk_row(chunk) for chunk in plan.upsert),
            )
            self._connection.execute("COMMIT")
            return plan
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def delete_source(
        self, *, tenant_id: str, source_id: str, expected_current_version: str
    ) -> tuple[str, ...]:
        """Explicitly delete one source and all chunks with optimistic concurrency."""
        self._validate_identity(tenant_id, source_id)
        if not isinstance(expected_current_version, str) or not expected_current_version:
            raise ValueError("expected_current_version must be non-empty")
        self._require_open()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            actual = self.current_version(tenant_id=tenant_id, source_id=source_id)
            if actual != expected_current_version:
                raise ValueError(
                    f"source version conflict: expected {expected_current_version!r}, "
                    f"found {actual!r}"
                )
            chunk_ids = tuple(
                row[0]
                for row in self._connection.execute(
                    "SELECT chunk_id FROM chunks WHERE tenant_id=? AND source_id=? "
                    "ORDER BY chunk_id",
                    (tenant_id, source_id),
                )
            )
            self._connection.execute(
                "DELETE FROM sources WHERE tenant_id=? AND source_id=?",
                (tenant_id, source_id),
            )
            self._connection.execute("COMMIT")
            return cast(tuple[str, ...], chunk_ids)
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def visible_chunks(
        self, *, tenant_id: str, principals: Iterable[str] = ()
    ) -> tuple[SourceChunk, ...]:
        """Load only tenant/ACL-visible chunks before any retrieval scoring."""
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")
        principal_set = set(principals)
        if any(not isinstance(value, str) or not value.strip() for value in principal_set):
            raise ValueError("principals must contain non-empty strings")
        self._require_open()
        chunks = tuple(
            _row_to_chunk(row)
            for row in self._connection.execute(
                _SELECT_CHUNKS_SQL + " WHERE tenant_id = ? ORDER BY source_id, ordinal",
                (tenant_id,),
            )
        )
        return tuple(
            chunk
            for chunk in chunks
            if not chunk.acl or not principal_set.isdisjoint(chunk.acl)
        )

    def _load_source_chunks(self, tenant_id: str, source_id: str) -> tuple[SourceChunk, ...]:
        return tuple(
            _row_to_chunk(row)
            for row in self._connection.execute(
                _SELECT_CHUNKS_SQL
                + " WHERE tenant_id = ? AND source_id = ? ORDER BY ordinal",
                (tenant_id, source_id),
            )
        )

    def _initialize_schema(self) -> None:
        version = cast(int, self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, _SCHEMA_VERSION}:
            self.close()
            raise ValueError(f"unsupported SQLite chunk schema version {version}")
        self._connection.executescript(_SCHEMA_SQL)
        if version == 0:
            self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteChunkStore is closed")

    @staticmethod
    def _validate_identity(tenant_id: str, source_id: str) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id cannot be empty")


def _source_fingerprint(source: SourceDocument, *, max_chars: int) -> str:
    payload = {
        "acl": list(source.acl),
        "chunking": {
            "max_chars": max_chars,
            "revision": _CHUNKING_REVISION,
        },
        "metadata": _strict_json_value(source.metadata),
        "source_id": source.source_id,
        "tenant_id": source.tenant_id,
        "text": source.text,
        "version": source.version,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _strict_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata floats must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata object keys must be strings")
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    raise ValueError("metadata must contain strict JSON values")


def _chunk_row(chunk: SourceChunk) -> tuple[Any, ...]:
    return (
        chunk.chunk_id,
        chunk.tenant_id,
        chunk.source_id,
        chunk.source_version,
        chunk.ordinal,
        chunk.content_hash,
        chunk.text,
        json.dumps(list(chunk.heading_path), ensure_ascii=False, separators=(",", ":")),
        json.dumps(list(chunk.acl), ensure_ascii=False, separators=(",", ":")),
        json.dumps(
            _strict_json_value(chunk.metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _row_to_chunk(row: tuple[Any, ...]) -> SourceChunk:
    heading = _strict_string_array(row[7], "heading_path_json")
    acl = _strict_string_array(row[8], "acl_json")
    metadata = _strict_json_object(row[9], "metadata_json")
    return SourceChunk(
        chunk_id=cast(str, row[0]), tenant_id=cast(str, row[1]),
        source_id=cast(str, row[2]), source_version=cast(str, row[3]),
        ordinal=cast(int, row[4]), content_hash=cast(str, row[5]), text=cast(str, row[6]),
        heading_path=heading, acl=acl, metadata=metadata,
    )


def _strict_json_object(value: str, label: str) -> dict[str, Any]:
    parsed = _strict_json_loads(value, label)
    if not isinstance(parsed, dict):
        raise ValueError(f"stored {label} must be an object")
    return cast(dict[str, Any], parsed)


def _strict_string_array(value: str, label: str) -> tuple[str, ...]:
    parsed = _strict_json_loads(value, label)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"stored {label} must be a string array")
    return tuple(parsed)


def _strict_json_loads(value: str, label: str) -> Any:
    def reject(_: str) -> None:
        raise ValueError(f"stored {label} contains a non-finite constant")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"stored {label} contains a duplicate key")
            result[key] = item
        return result

    def finite_float(raw: str) -> float:
        parsed = float(raw)
        if not math.isfinite(parsed):
            raise ValueError(f"stored {label} contains a non-finite number")
        return parsed

    try:
        return json.loads(
            value,
            parse_constant=reject,
            parse_float=finite_float,
            object_pairs_hook=pairs,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"stored {label} is invalid JSON") from error


_SELECT_CHUNKS_SQL = (
    "SELECT chunk_id, tenant_id, source_id, source_version, ordinal, content_hash, "
    "text, heading_path_json, acl_json, metadata_json FROM chunks"
)
_UPSERT_CHUNK_SQL = (
    "INSERT INTO chunks(chunk_id,tenant_id,source_id,source_version,ordinal,content_hash,"
    "text,heading_path_json,acl_json,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?) "
    "ON CONFLICT(chunk_id) DO UPDATE SET source_version=excluded.source_version, "
    "ordinal=excluded.ordinal, text=excluded.text, heading_path_json=excluded.heading_path_json, "
    "acl_json=excluded.acl_json, metadata_json=excluded.metadata_json"
)
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources(
  tenant_id TEXT NOT NULL, source_id TEXT NOT NULL, source_version TEXT NOT NULL,
  source_fingerprint TEXT NOT NULL, PRIMARY KEY(tenant_id, source_id)
);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, source_id TEXT NOT NULL,
  source_version TEXT NOT NULL, ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  content_hash TEXT NOT NULL, text TEXT NOT NULL CHECK(length(text) > 0),
  heading_path_json TEXT NOT NULL, acl_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
  FOREIGN KEY(tenant_id, source_id) REFERENCES sources(tenant_id, source_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_source ON chunks(tenant_id, source_id, ordinal);
"""
