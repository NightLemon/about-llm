from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.content_quality import check_encoding, check_ledger


def _write_registry(path: Path, sources: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "sources": sources}),
        encoding="utf-8",
    )


def test_source_registry_accepts_current_complete_review(tmp_path: Path) -> None:
    url = "https://example.com/official"
    ledger = tmp_path / "accuracy.md"
    registry = tmp_path / "sources.json"
    ledger.write_text(f"# Accuracy\n\n{url}\n", encoding="utf-8")
    _write_registry(
        registry,
        [{"url": url, "checked_at": "2026-08-01", "scope": "test contract"}],
    )

    assert check_ledger(
        accuracy_page=ledger,
        source_registry=registry,
        as_of=date(2026, 8, 11),
        expected_urls={url},
    ) == []


def test_source_registry_rejects_stale_duplicate_review(tmp_path: Path) -> None:
    url = "https://example.com/official"
    ledger = tmp_path / "accuracy.md"
    registry = tmp_path / "sources.json"
    ledger.write_text(f"# Accuracy\n\n{url}\n", encoding="utf-8")
    source = {"url": url, "checked_at": "2026-01-01", "scope": "test contract"}
    _write_registry(registry, [source, source])

    errors = check_ledger(
        accuracy_page=ledger,
        source_registry=registry,
        as_of=date(2026, 8, 11),
        expected_urls={url},
    )

    assert any("stale" in error for error in errors)
    assert any("duplicate URL" in error for error in errors)


def test_encoding_rejects_replacement_character(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("broken: \ufffd", encoding="utf-8")

    assert check_encoding([page], root=tmp_path) == [
        "page.md: contains Unicode replacement character"
    ]