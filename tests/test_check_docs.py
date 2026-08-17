from __future__ import annotations

from pathlib import Path

from scripts.check_docs import (
    check_internal_links,
    check_learning_contracts,
    check_math_delimiters,
    check_test_references,
    parse_document,
)


def test_parser_extracts_inline_reference_and_image_targets() -> None:
    parsed = parse_document(
        "# Page\n\n[inline](guide.md)\n\n[reference][guide]\n\n"
        "![diagram](diagram.png)\n\n[guide]: <guide(a).md#section>\n"
    )

    assert parsed.targets == ["guide.md", "guide(a).md#section", "diagram.png"]


def test_link_check_accepts_parentheses_and_explicit_anchor(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "index.md"
    target = docs / "guide(a).md"
    source.write_text("# Index\n\n[Guide](<guide(a).md#stable>)\n", encoding="utf-8")
    target.write_text("# Guide\n\n## 中文标题 { #stable }\n", encoding="utf-8")

    assert check_internal_links([source, target], root=tmp_path, docs=docs) == []


def test_link_check_rejects_missing_anchor_and_image(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "index.md"
    target = docs / "guide.md"
    source.write_text(
        "# Index\n\n[Guide](guide.md#missing)\n\n![Image](missing.png)\n",
        encoding="utf-8",
    )
    target.write_text("# Guide\n\n## Existing { #existing }\n", encoding="utf-8")

    errors = check_internal_links([source, target], root=tmp_path, docs=docs)

    assert any("missing anchor '#missing'" in error for error in errors)
    assert any("missing target: missing.png" in error for error in errors)


def test_learning_contract_accepts_complete_navigation(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "core" / "page.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# Page\n\n"
        "<!-- learning-contract -->\n"
        '<div class="learning-contract" markdown="1">\n\n'
        "**学习导航**\n\n"
        "- **适合读者**: 读者。\n"
        "- **先修**: 先修。\n"
        "- **首次阅读**: 路径。\n"
        "- **完成信号**: 信号。\n"
        "- **卡住时**: 回退。\n\n"
        "</div>\n",
        encoding="utf-8",
    )

    assert check_learning_contracts([page], root=tmp_path) == []


def test_learning_contract_rejects_missing_marker_and_field(tmp_path: Path) -> None:
    missing_marker = tmp_path / "docs" / "core" / "missing-marker.md"
    missing_field = tmp_path / "docs" / "core" / "missing-field.md"
    missing_marker.parent.mkdir(parents=True)
    missing_marker.write_text("# Missing\n", encoding="utf-8")
    missing_field.write_text(
        "# Incomplete\n\n"
        "<!-- learning-contract -->\n"
        '<div class="learning-contract" markdown="1">\n\n'
        "- **适合读者**: 读者。\n"
        "- **先修**: 先修。\n"
        "- **首次阅读**: 路径。\n"
        "- **卡住时**: 回退。\n\n"
        "</div>\n",
        encoding="utf-8",
    )

    errors = check_learning_contracts([missing_marker, missing_field], root=tmp_path)

    assert any("expected exactly one learning contract marker" in error for error in errors)
    assert any("learning contract missing **完成信号**" in error for error in errors)


def test_math_delimiters_ignore_code_and_accept_balanced_prose(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "page.md"
    page.parent.mkdir()
    page.write_text(
        "# Page\n\nInline \\(x\\) and display:\n\n\\[\ny = x\n\\]\n\n"
        "`literal \\(`\n\n```text\nunclosed \\[\n```\n",
        encoding="utf-8",
    )

    assert check_math_delimiters([page], root=tmp_path) == []


def test_math_delimiters_reject_unclosed_and_mismatched_pairs(tmp_path: Path) -> None:
    unclosed = tmp_path / "docs" / "unclosed.md"
    mismatched = tmp_path / "docs" / "mismatched.md"
    unclosed.parent.mkdir()
    unclosed.write_text("# Unclosed\n\n\\[\nx + y\n", encoding="utf-8")
    mismatched.write_text("# Mismatched\n\n\\(x\\]\n", encoding="utf-8")

    errors = check_math_delimiters([unclosed, mismatched], root=tmp_path)

    assert any("unclosed.md:3: unclosed math delimiter \\[" in error for error in errors)
    assert any("mismatched.md:3: unexpected math delimiter \\]" in error for error in errors)


def test_test_references_reject_missing_files(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "page.md"
    existing = tmp_path / "tests" / "test_existing.py"
    page.parent.mkdir()
    existing.parent.mkdir()
    existing.write_text("", encoding="utf-8")
    page.write_text(
        "# Page\n\n"
        "python -m pytest tests/test_existing.py tests/test_missing.py -q\n",
        encoding="utf-8",
    )

    assert check_test_references([page], root=tmp_path) == [
        f"{page.relative_to(tmp_path)}: missing test file: tests/test_missing.py"
    ]
