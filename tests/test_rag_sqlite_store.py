from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from about_llm.rag import SourceDocument, SQLiteChunkStore

pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]


def _source(
    text: str,
    *,
    version: str = "v1",
    tenant: str = "tenant-a",
    source_id: str = "guide",
    acl: tuple[str, ...] = ("engineering",),
    metadata: dict[str, object] | None = None,
) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        tenant_id=tenant,
        version=version,
        text=text,
        acl=acl,
        metadata=metadata or {"uri": f"kb://{source_id}"},
    )


def test_store_persists_chunks_and_filters_acl_before_return(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    with SQLiteChunkStore(path) as store:
        store.upsert_source(
            _source("# Guide\n\nPublic to engineering."),
            expected_current_version=None,
        )
        store.upsert_source(
            _source(
                "# Public\n\nTenant public.",
                source_id="public",
                acl=(),
            ),
            expected_current_version=None,
        )
        store.upsert_source(
            _source("# Secret\n\nOther tenant.", tenant="tenant-b"),
            expected_current_version=None,
        )

    with SQLiteChunkStore(path) as reopened:
        anonymous = reopened.visible_chunks(tenant_id="tenant-a")
        engineering = reopened.visible_chunks(
            tenant_id="tenant-a", principals=("engineering",)
        )
        assert {chunk.source_id for chunk in anonymous} == {"public"}
        assert {chunk.source_id for chunk in engineering} == {"guide", "public"}
        assert all(chunk.tenant_id == "tenant-a" for chunk in engineering)


def test_incremental_update_is_version_checked_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    first = _source("# Guide\n\nKeep.\n\nRemove.")
    second = _source("# Guide\n\nKeep.\n\nAdded.", version="v2")
    with SQLiteChunkStore(path) as store:
        initial = store.upsert_source(first, expected_current_version=None)
        repeat = store.upsert_source(first, expected_current_version="v1")
        updated = store.upsert_source(second, expected_current_version="v1")
        assert len(initial.upsert) == 2
        assert not repeat.upsert and len(repeat.unchanged_chunk_ids) == 2
        assert {chunk.text for chunk in updated.upsert} == {"Keep.", "Added."}
        assert len(updated.delete_chunk_ids) == 1
        assert store.current_version(tenant_id="tenant-a", source_id="guide") == "v2"
        with pytest.raises(ValueError, match="version conflict"):
            store.upsert_source(first, expected_current_version="v1")


def test_same_version_cannot_name_different_content(tmp_path: Path) -> None:
    with SQLiteChunkStore(tmp_path / "rag.db") as store:
        store.upsert_source(_source("Original."), expected_current_version=None)
        with pytest.raises(ValueError, match="cannot be reused"):
            store.upsert_source(
                _source("Changed without version bump."),
                expected_current_version="v1",
            )
        assert [chunk.text for chunk in store.visible_chunks(
            tenant_id="tenant-a", principals=("engineering",)
        )] == ["Original."]


def test_same_version_cannot_change_chunking_configuration(tmp_path: Path) -> None:
    text = "A" * 160
    with SQLiteChunkStore(tmp_path / "rag.db") as store:
        store.upsert_source(
            _source(text),
            expected_current_version=None,
            max_chars=80,
        )
        with pytest.raises(ValueError, match="cannot be reused"):
            store.upsert_source(
                _source(text),
                expected_current_version="v1",
                max_chars=120,
            )
        chunks = store.visible_chunks(
            tenant_id="tenant-a", principals=("engineering",)
        )
        assert len(chunks) == 2
        assert all(len(chunk.text) == 80 for chunk in chunks)


def test_explicit_delete_requires_current_version(tmp_path: Path) -> None:
    with SQLiteChunkStore(tmp_path / "rag.db") as store:
        store.upsert_source(_source("Delete me."), expected_current_version=None)
        with pytest.raises(ValueError, match="version conflict"):
            store.delete_source(
                tenant_id="tenant-a", source_id="guide", expected_current_version="stale"
            )
        deleted = store.delete_source(
            tenant_id="tenant-a", source_id="guide", expected_current_version="v1"
        )
        assert deleted
        assert store.visible_chunks(tenant_id="tenant-a", principals=("engineering",)) == ()


def test_zero_chunk_update_cannot_implicitly_delete_existing_source(tmp_path: Path) -> None:
    with SQLiteChunkStore(tmp_path / "rag.db") as store:
        store.upsert_source(_source("Keep me."), expected_current_version=None)
        with pytest.raises(ValueError, match="produced no chunks"):
            store.upsert_source(
                _source("# Heading only", version="v2"),
                expected_current_version="v1",
            )
        assert store.current_version(tenant_id="tenant-a", source_id="guide") == "v1"
        assert [
            chunk.text
            for chunk in store.visible_chunks(
                tenant_id="tenant-a", principals=("engineering",)
            )
        ] == ["Keep me."]


def test_database_trigger_failure_rolls_back_delete_and_version(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    with SQLiteChunkStore(path) as store:
        store.upsert_source(_source("Old chunk."), expected_current_version=None)
        with sqlite3.connect(path) as admin:
            admin.execute(
                "CREATE TRIGGER injected_failure BEFORE INSERT ON chunks "
                "WHEN NEW.source_version = 'v2' BEGIN "
                "SELECT RAISE(ABORT, 'injected'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            store.upsert_source(
                _source("New chunk.", version="v2"),
                expected_current_version="v1",
            )
        assert store.current_version(tenant_id="tenant-a", source_id="guide") == "v1"
        chunks = store.visible_chunks(
            tenant_id="tenant-a", principals=("engineering",)
        )
        assert [chunk.text for chunk in chunks] == ["Old chunk."]


def test_strict_metadata_rejects_nonfinite_and_corrupted_json(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    with SQLiteChunkStore(path) as store:
        with pytest.raises(ValueError, match="finite"):
            store.upsert_source(
                _source("Text.", metadata={"score": float("nan")}),
                expected_current_version=None,
            )
        store.upsert_source(_source("Text."), expected_current_version=None)
        with sqlite3.connect(path) as admin:
            admin.execute("UPDATE chunks SET metadata_json = '{\"x\":1,\"x\":2}'")
        with pytest.raises(ValueError, match="duplicate key"):
            store.visible_chunks(tenant_id="tenant-a", principals=("engineering",))
