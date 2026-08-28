from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scripts.content_quality import check_encoding, check_ledger, effective_source_status

pytestmark = [pytest.mark.contract, pytest.mark.security]


def _source(**overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "id": "example-official",
        "url": "https://example.com/official",
        "scope": "test contract",
        "status": "verified",
        "volatility": "high",
        "checked_at": "2026-08-01",
        "next_review_at": "2026-09-01",
        "used_by": ["page.md"],
    }
    source.update(overrides)
    return source


def _write_registry(path: Path, sources: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 2, "sources": sources}),
        encoding="utf-8",
    )


def test_source_registry_accepts_current_complete_review(tmp_path: Path) -> None:
    source = _source()
    ledger = tmp_path / "accuracy.md"
    registry = tmp_path / "sources.json"
    (tmp_path / "page.md").write_text("[SOURCE:example-official]\n", encoding="utf-8")
    ledger.write_text(f"# Accuracy\n\n{source['url']}\n", encoding="utf-8")
    _write_registry(registry, [source])

    assert check_ledger(
        accuracy_page=ledger,
        source_registry=registry,
        as_of=date(2026, 8, 11),
        expected_urls={str(source["url"])},
        docs_root=tmp_path,
    ) == []
    assert effective_source_status(source, as_of=date(2026, 8, 11)) == "verified"


def test_source_registry_reports_stale_without_failing_structure(tmp_path: Path) -> None:
    source = _source(next_review_at="2026-08-10")
    ledger = tmp_path / "accuracy.md"
    registry = tmp_path / "sources.json"
    (tmp_path / "page.md").write_text("[SOURCE:example-official]\n", encoding="utf-8")
    ledger.write_text(f"# Accuracy\n\n{source['url']}\n", encoding="utf-8")
    _write_registry(registry, [source])

    assert check_ledger(
        accuracy_page=ledger,
        source_registry=registry,
        as_of=date(2026, 8, 11),
        expected_urls={str(source["url"])},
        docs_root=tmp_path,
    ) == []
    assert effective_source_status(source, as_of=date(2026, 8, 11)) == "stale"


def test_source_registry_rejects_duplicate_and_unknown_marker(tmp_path: Path) -> None:
    source = _source()
    ledger = tmp_path / "accuracy.md"
    registry = tmp_path / "sources.json"
    (tmp_path / "page.md").write_text("[SOURCE:not-registered]\n", encoding="utf-8")
    ledger.write_text(f"# Accuracy\n\n{source['url']}\n", encoding="utf-8")
    _write_registry(registry, [source, source])

    errors = check_ledger(
        accuracy_page=ledger,
        source_registry=registry,
        as_of=date(2026, 8, 11),
        expected_urls={str(source["url"])},
        docs_root=tmp_path,
    )

    assert any("duplicate URL" in error for error in errors)
    assert any("duplicate id" in error for error in errors)
    assert any("unknown source marker" in error for error in errors)


def test_network_probe_states_override_reviewed_status() -> None:
    source = _source()

    assert effective_source_status(
        source,
        as_of=date(2026, 8, 11),
        probe={"status": "unknown"},
    ) == "unknown"
    assert effective_source_status(
        source,
        as_of=date(2026, 8, 11),
        probe={"status": "pending-review"},
    ) == "pending-review"


def test_encoding_rejects_replacement_character(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("broken: \ufffd", encoding="utf-8")

    assert check_encoding([page], root=tmp_path) == [
        "page.md: contains Unicode replacement character"
    ]
