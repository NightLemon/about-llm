from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from about_llm.llmops import artifact_fingerprint
from about_llm.rag import (
    SourceDocument,
    SQLiteChunkStore,
    create_sqlite_chunk_backup,
    load_sqlite_chunk_backup_manifest,
    restore_sqlite_chunk_backup,
    verify_sqlite_chunk_backup,
)
from about_llm.rag.cli import main

pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]


def _source(text: str, *, version: str = "v1") -> SourceDocument:
    return SourceDocument(
        source_id="guide",
        tenant_id="tenant-a",
        version=version,
        text=text,
        acl=("engineering",),
        metadata={"uri": "kb://guide", "quality": 1.0},
    )


def _create_store(path: Path) -> None:
    with SQLiteChunkStore(path) as store:
        store.upsert_source(
            _source("# Repeated\n\n## Repeated\n\nOriginal snapshot."),
            expected_current_version=None,
        )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rebind_physical_manifest(backup: Path, manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["backup_sha256"] = _sha256(backup)
    payload["backup_size_bytes"] = backup.stat().st_size
    identity = {
        key: value for key, value in payload.items() if key != "manifest_fingerprint"
    }
    payload["manifest_fingerprint"] = "sha256:" + artifact_fingerprint(identity)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_backup_restore_round_trip_is_snapshot_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rag.db"
    backup = tmp_path / "rag.backup.db"
    manifest_path = tmp_path / "rag.backup.manifest.json"
    restored = tmp_path / "rag.restored.db"
    _create_store(database)

    manifest = create_sqlite_chunk_backup(
        database,
        backup,
        manifest_path,
        created_at_utc="2026-08-06T12:00:00Z",
    )
    backup_bytes = backup.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    with SQLiteChunkStore(database) as store:
        store.upsert_source(
            _source("Changed after snapshot.", version="v2"),
            expected_current_version="v1",
        )

    verified = verify_sqlite_chunk_backup(backup, manifest_path)
    restored_manifest = restore_sqlite_chunk_backup(backup, manifest_path, restored)
    assert verified == manifest == restored_manifest
    assert manifest.source_count == 1
    assert manifest.chunk_count == 1
    assert manifest.backup_sha256 == _sha256(backup)
    assert manifest.logical_fingerprint.startswith("sha256:")
    assert load_sqlite_chunk_backup_manifest(manifest_path) == manifest

    with SQLiteChunkStore(restored) as store:
        chunks = store.visible_chunks(
            tenant_id="tenant-a", principals=("engineering",)
        )
        assert store.current_version(tenant_id="tenant-a", source_id="guide") == "v1"
        assert [chunk.text for chunk in chunks] == ["Original snapshot."]

    with pytest.raises(FileExistsError, match="backup path already exists"):
        create_sqlite_chunk_backup(database, backup, manifest_path)
    assert backup.read_bytes() == backup_bytes
    assert manifest_path.read_bytes() == manifest_bytes

    sentinel = restored.read_bytes()
    with pytest.raises(FileExistsError, match="restore target path already exists"):
        restore_sqlite_chunk_backup(backup, manifest_path, restored)
    assert restored.read_bytes() == sentinel


def test_backup_verification_detects_physical_tampering(tmp_path: Path) -> None:
    database = tmp_path / "rag.db"
    backup = tmp_path / "backup.db"
    manifest = tmp_path / "backup.json"
    _create_store(database)
    create_sqlite_chunk_backup(database, backup, manifest)

    with backup.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="backup size mismatch"):
        verify_sqlite_chunk_backup(backup, manifest)


def test_backup_verify_restore_cli_reports_narrow_evidence_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "rag.db"
    backup = tmp_path / "backup.db"
    manifest = tmp_path / "backup.json"
    restored = tmp_path / "restored.db"
    _create_store(database)

    assert main(
        [
            "store-backup",
            "--database",
            str(database),
            "--backup",
            str(backup),
            "--manifest",
            str(manifest),
        ]
    ) == 0
    backup_payload = json.loads(capsys.readouterr().out)
    assert backup_payload["operation"] == "backup"
    assert backup_payload["manifest"]["source_count"] == 1
    assert backup_payload["scope"]["logical_row_fingerprint_checked"] is True
    assert backup_payload["scope"]["manifest_signature_verified"] is False
    assert backup_payload["scope"]["remote_vector_store_included"] is False

    assert main(
        [
            "store-verify-backup",
            "--backup",
            str(backup),
            "--manifest",
            str(manifest),
        ]
    ) == 0
    verified_payload = json.loads(capsys.readouterr().out)
    assert verified_payload["operation"] == "verify-backup"
    assert verified_payload["verified"] is True

    assert main(
        [
            "store-restore",
            "--backup",
            str(backup),
            "--manifest",
            str(manifest),
            "--database",
            str(restored),
        ]
    ) == 0
    restored_payload = json.loads(capsys.readouterr().out)
    assert restored_payload["operation"] == "restore"
    assert restored_payload["restored"] is True
    with SQLiteChunkStore(restored) as store:
        assert store.current_version(tenant_id="tenant-a", source_id="guide") == "v1"


def test_backup_verification_rejects_semantically_corrupted_rows_even_if_rehashed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rag.db"
    backup = tmp_path / "backup.db"
    manifest_path = tmp_path / "backup.json"
    _create_store(database)
    create_sqlite_chunk_backup(database, backup, manifest_path)

    with sqlite3.connect(backup) as connection:
        connection.execute("UPDATE chunks SET content_hash = ?", ("0" * 64,))
    _rebind_physical_manifest(backup, manifest_path)

    with pytest.raises(ValueError, match="content_hash does not match text"):
        verify_sqlite_chunk_backup(backup, manifest_path)


def test_backup_verification_rejects_unversioned_schema_objects_even_if_rehashed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rag.db"
    backup = tmp_path / "backup.db"
    manifest_path = tmp_path / "backup.json"
    _create_store(database)
    create_sqlite_chunk_backup(database, backup, manifest_path)

    with sqlite3.connect(backup) as connection:
        connection.execute(
            "CREATE TRIGGER unexpected_trigger BEFORE DELETE ON chunks "
            "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
        )
    _rebind_physical_manifest(backup, manifest_path)

    with pytest.raises(ValueError, match="schema objects do not match"):
        verify_sqlite_chunk_backup(backup, manifest_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"manifest_version":"a","manifest_version":"b"}', "duplicate JSON"),
        ('{"manifest_version":NaN}', "non-standard JSON constant"),
        ('{"manifest_version":1e999}', "non-finite JSON number"),
        ('{"manifest_version":"unknown","extra":true}', "field mismatch"),
    ],
)
def test_backup_manifest_loader_is_strict(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_sqlite_chunk_backup_manifest(path)
