from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_docs import (
    READABILITY_DEFAULTS,
    check_evidence_entrypoints,
    check_glossary_graph,
    check_internal_links,
    check_learning_contracts,
    check_math_delimiters,
    check_readability,
    check_test_references,
    parse_document,
)

pytestmark = pytest.mark.contract


def _write_readability_baseline(path: Path, pages: dict[str, dict[str, int]]) -> None:
    path.write_text(
        json.dumps(
            {"schema_version": 1, "defaults": READABILITY_DEFAULTS, "pages": pages}
        ),
        encoding="utf-8",
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


def test_glossary_graph_accepts_complete_linked_terms(tmp_path: Path) -> None:
    glossary = tmp_path / "docs" / "reference" / "glossary.md"
    glossary.parent.mkdir(parents=True)
    glossary.write_text(
        "# 术语表\n\n"
        "| 术语 | 定义 | 分类 | 先修 | 易混淆 | 正文 | 实验 |\n"
        "|---|---|---|---|---|---|---|\n"
        "| <a id=\"term-logit\"></a>Logit | 归一化为概率以前的模型分数。 | 基础 |"
        " [Token](#term-token) | [概率](#term-token) | [正文](../../README.md) |"
        " [实验](../../CONTRIBUTING.md) |\n"
        "| <a id=\"term-token\"></a>Token | tokenizer 词表中的离散单位。 | 基础 |"
        " 根概念 | — | [正文](../../README.md) | [实验](../../CONTRIBUTING.md) |\n",
        encoding="utf-8",
    )

    assert check_glossary_graph(
        glossary, root=tmp_path, minimum_terms=2
    ) == []


def test_glossary_graph_rejects_unknown_dependency_and_missing_binding(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "docs" / "reference" / "glossary.md"
    glossary.parent.mkdir(parents=True)
    glossary.write_text(
        "# 术语表\n\n"
        "| 术语 | 定义 | 分类 | 先修 | 易混淆 | 正文 | 实验 |\n"
        "|---|---|---|---|---|---|---|\n"
        "| <a id=\"term-logit\"></a>Logit | 归一化为概率以前的模型分数。 | 基础 |"
        " [Missing](#term-missing) | — | 无 | [实验](../../CONTRIBUTING.md) |\n",
        encoding="utf-8",
    )

    errors = check_glossary_graph(glossary, root=tmp_path, minimum_terms=1)

    assert any("canonical needs a local link" in error for error in errors)
    assert any("unknown term 'missing'" in error for error in errors)


def test_readability_uses_strict_defaults_for_new_pages(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "core" / "page.md"
    baseline = tmp_path / "baseline.json"
    page.parent.mkdir(parents=True)
    page.write_text("# Page\n\n" + "x" * 201 + "\n", encoding="utf-8")
    _write_readability_baseline(baseline, {})

    errors = check_readability([page], baseline_path=baseline, root=tmp_path)

    assert any("overlong_prose_line_count 1 exceeds budget 0" in error for error in errors)
    assert any("max_prose_line_length 201 exceeds budget 200" in error for error in errors)


def test_readability_debt_is_an_exact_downward_ratchet(tmp_path: Path) -> None:
    page = tmp_path / "docs" / "models" / "legacy.md"
    baseline = tmp_path / "baseline.json"
    page.parent.mkdir(parents=True)
    page.write_text("# Legacy\n\n" + "x" * 220 + "\n", encoding="utf-8")
    relative = page.relative_to(tmp_path).as_posix()
    _write_readability_baseline(
        baseline,
        {relative: {"overlong_prose_line_count": 1, "max_prose_line_length": 220}},
    )
    assert check_readability([page], baseline_path=baseline, root=tmp_path) == []

    page.write_text("# Legacy\n\nshort\n", encoding="utf-8")
    errors = check_readability([page], baseline_path=baseline, root=tmp_path)

    assert any("improved" in error and "tighten stale budget" in error for error in errors)


def test_evidence_ledgers_must_redirect_first_time_readers(tmp_path: Path) -> None:
    evidence = tmp_path / "docs" / "evidence"
    evidence.mkdir(parents=True)
    good = evidence / "good.md"
    bad = evidence / "bad.md"
    good.write_text(
        "# Ledger\n\n第一次学习请从[教程](../guide/start.md)开始。\n",
        encoding="utf-8",
    )
    bad.write_text("# Ledger\n\nOnly internal fixtures.\n", encoding="utf-8")

    errors = check_evidence_entrypoints(tmp_path / "docs", root=tmp_path)

    assert not any("good.md" in error for error in errors)
    assert any("bad.md" in error and "first-time readers" in error for error in errors)
    assert any("bad.md" in error and "reader-facing entry link" in error for error in errors)
