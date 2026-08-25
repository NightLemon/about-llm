from __future__ import annotations

from pathlib import Path

import pytest

from scripts.report_readability import analyze_text, markdown_files, prose_blocks

pytestmark = pytest.mark.contract


def test_prose_blocks_ignore_code_tables_math_and_inline_code() -> None:
    text = """# 页面

正文中的 fixed fixture 应该改写。

```text
authored fixture and strict control
```

| 字段 | 值 |
|---|---|
| policy | authored fixture |

\\[
oracle = control
\\]

运行 `strict control` 后查看结果。
"""

    blocks = list(prose_blocks(text))
    combined = "\n".join(block.text for block in blocks)

    assert "正文中的 fixed fixture" in combined
    assert "authored fixture and strict control" not in combined
    assert "oracle = control" not in combined
    assert "strict control" not in combined


def test_analyzer_reports_prose_without_turning_findings_into_failures() -> None:
    text = (
        "# 页面\n\n"
        "这个 authored fixture 由 strict verifier 检查\uff0c不能证明来源\uff0c"
        "不代表质量\uff0c也不能说明线上性能。\n"
    )

    findings = analyze_text(text, display_path="docs/page.md")

    assert {finding.kind for finding in findings} >= {
        "internal-jargon",
        "negation-chain",
    }
    assert all(finding.path == "docs/page.md" for finding in findings)


def test_heading_anchor_metadata_is_not_treated_as_reader_prose() -> None:
    text = (
        "# 页面\n\n"
        "## 用固定模型核对最终 labels { #target-label-control }\n\n"
        "自然语言正文。\n"
    )

    blocks = list(prose_blocks(text))
    combined = "\n".join(block.text for block in blocks)
    findings = analyze_text(text, display_path="docs/page.md")

    assert "target-label-control" not in combined
    assert all(finding.kind != "internal-jargon" for finding in findings)


def test_scope_includes_reader_pages_and_only_primary_project_readmes(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs" / "core").mkdir(parents=True)
    (tmp_path / "docs" / "evidence").mkdir(parents=True)
    (tmp_path / "projects" / "demo" / "nested").mkdir(parents=True)
    for relative in (
        "README.md",
        "CONTRIBUTING.md",
        "docs/core/page.md",
        "docs/evidence/ledger.md",
        "projects/demo/README.md",
        "projects/demo/nested/README.md",
    ):
        path = tmp_path / relative
        path.write_text("# Page\n", encoding="utf-8")

    reader = {
        path.relative_to(tmp_path).as_posix()
        for path in markdown_files("reader", root=tmp_path)
    }
    all_pages = {
        path.relative_to(tmp_path).as_posix()
        for path in markdown_files("all", root=tmp_path)
    }

    assert "docs/core/page.md" in reader
    assert "projects/demo/README.md" in reader
    assert "docs/evidence/ledger.md" not in reader
    assert "projects/demo/nested/README.md" not in reader
    assert "docs/evidence/ledger.md" in all_pages
