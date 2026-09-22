from __future__ import annotations

from pathlib import Path

import pytest

from scripts.report_content_structure import (
    ROOT,
    advisory_findings,
    analyze_page,
    canonical_nav_pages,
    classify_page,
    content_files,
    measure_text,
)

pytestmark = pytest.mark.contract


def test_page_roles_cover_entry_body_practice_readme_reference_and_evidence(
    tmp_path: Path,
) -> None:
    paths = [
        tmp_path / "README.md",
        tmp_path / "docs" / "core" / "topic.md",
        tmp_path / "docs" / "practice" / "labs" / "lab.md",
        tmp_path / "projects" / "demo" / "README.md",
        tmp_path / "docs" / "reference" / "lookup.md",
        tmp_path / "docs" / "evidence" / "ledger.md",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Page\n", encoding="utf-8")

    assert [classify_page(path, root=tmp_path) for path in paths] == [
        "entry",
        "body",
        "practice",
        "readme",
        "reference",
        "evidence",
    ]


def test_metrics_ignore_fence_contents_for_headings_and_tables() -> None:
    text = """# Page

## Section

正文 [链接](target.md) 和 `inline`。

| A | B |
|---|---|
| 1 | 2 |

```text
## Not a heading
| not | a table |
code
```
"""

    metrics = measure_text(text)

    assert metrics["lines"] == 15
    assert metrics["headings"] == 1
    assert metrics["table_lines"] == 3
    assert metrics["code_lines"] == 3
    assert metrics["prose_chars"] > 0


def test_reference_is_exempt_and_evidence_only_reports_role_drift() -> None:
    metrics = {
        "lines": 2_000,
        "headings": 100,
        "prose_chars": 100_000,
        "code_lines": 0,
        "table_lines": 0,
    }

    assert advisory_findings("reference", metrics, "# Lookup") == ()
    assert advisory_findings("evidence", metrics, "# Ledger\n\nOnly rows.") == ()

    findings = advisory_findings(
        "evidence",
        metrics,
        "# Ledger\n\n<!-- learning-contract -->\n\n## 自测\n",
    )
    assert [finding.kind for finding in findings] == ["evidence-role-drift"]


def test_body_findings_are_advisory_metrics() -> None:
    metrics = {
        "lines": 521,
        "headings": 41,
        "prose_chars": 30_001,
        "code_lines": 0,
        "table_lines": 0,
    }

    assert {finding.kind for finding in advisory_findings("body", metrics, "")} == {
        "page-length",
        "section-count",
        "prose-density",
    }


def test_reader_scope_uses_canonical_nav_and_reader_readmes(tmp_path: Path) -> None:
    (tmp_path / "docs" / "core").mkdir(parents=True)
    (tmp_path / "docs" / "reference").mkdir(parents=True)
    (tmp_path / "projects" / "demo").mkdir(parents=True)
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
    (tmp_path / "docs" / "core" / "topic.md").write_text("# Topic\n", encoding="utf-8")
    (tmp_path / "docs" / "reference" / "lookup.md").write_text("# Lookup\n", encoding="utf-8")
    (tmp_path / "docs" / "orphan.md").write_text("# Orphan\n", encoding="utf-8")
    (tmp_path / "projects" / "demo" / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "notebooks" / "README.md").write_text("# Notebooks\n", encoding="utf-8")
    (tmp_path / "mkdocs.yml").write_text(
        "site_name: Test\n"
        "nav:\n"
        "  - Topic: core/topic.md\n"
        "  - Lookup: reference/lookup.md\n"
        "exclude_docs: |\n"
        "  _templates/\n",
        encoding="utf-8",
    )

    relative = {
        path.relative_to(tmp_path.resolve()).as_posix()
        for path in content_files("reader", root=tmp_path)
    }

    assert relative == {
        "README.md",
        "docs/core/topic.md",
        "docs/reference/lookup.md",
        "notebooks/README.md",
        "projects/demo/README.md",
    }
    assert analyze_page(
        tmp_path / "docs" / "core" / "topic.md", root=tmp_path
    ).role == "body"


def test_repository_has_expected_canonical_pages_and_redirects() -> None:
    pages = canonical_nav_pages()
    relative = [path.relative_to(ROOT / "docs").as_posix() for path in pages]
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert len(relative) == 172
    assert len(relative) == len(set(relative))
    assert "models/gemini-production.md" not in relative
    assert "quality/governance-templates.md" not in relative
    assert "'models/gemini-production.md': 'models/gemini.md'" in config
    assert (
        "'quality/governance-templates.md': 'reference/governance-templates.md'"
        in config
    )
