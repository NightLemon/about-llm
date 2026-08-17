"""Structured checks for the Markdown knowledge base."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from functools import cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
READABILITY_BASELINE = DOCS / "reference" / "readability-baseline.json"
READABILITY_DEFAULTS = {
    "line_count": 600,
    "heading_count": 45,
    "overlong_prose_line_count": 0,
    "max_prose_line_length": 200,
}
NON_READER_DIRECTORIES = {"assets", "evidence", "papers", "reference"}
CURRICULUM_DIRECTORIES = (
    "models",
    "foundations",
    "core",
    "training",
    "systems",
    "applications",
    "quality",
    "frontier",
)
LEARNING_CONTRACT_MARKER = "<!-- learning-contract -->"
LEARNING_CONTRACT_FIELDS = (
    "**适合读者**",
    "**先修**",
    "**首次阅读**",
    "**完成信号**",
    "**卡住时**",
)
MARKDOWN_EXTENSIONS = ["extra", "toc"]
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
ATTR_LIST_RE = re.compile(r"\s*\{[^{}]*\}\s*$")
INLINE_MARKUP_RE = re.compile(r"[`*_~]|\[([^]]+)\]\([^)]+\)")
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1")
MATH_DELIMITER_RE = re.compile(r"(?<!\\)(\\\(|\\\)|\\\[|\\\])")
TEST_FILE_RE = re.compile(r"\btests/test_[A-Za-z0-9_]+\.py\b")


@dataclass
class ParsedDocument:
    targets: list[str] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.document = ParsedDocument()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.document.ids.add(element_id)
        target = attributes.get("href") if tag == "a" else attributes.get("src")
        if target and tag in {"a", "img"}:
            self.document.targets.append(target)


def parse_document(text: str) -> ParsedDocument:
    rendered = markdown.markdown(text, extensions=MARKDOWN_EXTENSIONS)
    parser = _DocumentParser()
    parser.feed(rendered)
    return parser.document


def _github_heading_ids(text: str) -> set[str]:
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for match in HEADING_RE.finditer(text):
        heading = ATTR_LIST_RE.sub("", match.group(1))
        heading = INLINE_MARKUP_RE.sub(lambda item: item.group(1) or "", heading)
        slug = "".join(
            character
            for character in heading.casefold()
            if character.isalnum() or character in {" ", "-", "_"}
        )
        slug = re.sub(r"[\s-]+", "-", slug).strip("-")
        duplicate_index = counts.get(slug, 0)
        counts[slug] = duplicate_index + 1
        ids.add(slug if duplicate_index == 0 else f"{slug}-{duplicate_index}")
    return ids


@cache
def _target_ids(path: Path, github_style: bool) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return _github_heading_ids(text) if github_style else parse_document(text).ids


def markdown_files(root: Path = ROOT, docs: Path = DOCS) -> list[Path]:
    return sorted([root / "README.md", root / "CONTRIBUTING.md", *docs.rglob("*.md")])


def curriculum_markdown_files(docs: Path = DOCS) -> list[Path]:
    return sorted(
        path
        for directory in CURRICULUM_DIRECTORIES
        for path in (docs / directory).glob("*.md")
    )


def reader_markdown_files(docs: Path = DOCS) -> list[Path]:
    return sorted(
        path
        for path in docs.rglob("*.md")
        if path.relative_to(docs).parts[0] not in NON_READER_DIRECTORIES
    )


def project_markdown_files(root: Path = ROOT) -> list[Path]:
    projects = root / "projects"
    return sorted(projects.rglob("*.md")) if projects.exists() else []


def check_internal_links(
    files: list[Path], *, root: Path = ROOT, docs: Path = DOCS
) -> list[str]:
    errors: list[str] = []
    for source in files:
        parsed = parse_document(source.read_text(encoding="utf-8"))
        source_uses_github_anchors = not source.is_relative_to(docs)
        for raw_target in parsed.targets:
            split = urlsplit(raw_target)
            if split.scheme or split.netloc:
                continue
            path_part = unquote(split.path)
            fragment = unquote(split.fragment)
            target = source if not path_part else (source.parent / path_part).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"{source.relative_to(root)}: link escapes repo: {raw_target}")
                continue
            if not target.exists():
                errors.append(f"{source.relative_to(root)}: missing target: {raw_target}")
                continue
            if fragment and target.suffix.lower() == ".md":
                ids = _target_ids(target, source_uses_github_anchors)
                if fragment not in ids:
                    errors.append(
                        f"{source.relative_to(root)}: missing anchor '#{fragment}' "
                        f"in {target.relative_to(root)}"
                    )
    return errors


def check_basics(files: list[Path], *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for source in files:
        text = source.read_text(encoding="utf-8")
        if not text.startswith("# "):
            errors.append(f"{source.relative_to(root)}: missing H1 at first line")
        if "\ufffd" in text:
            errors.append(f"{source.relative_to(root)}: contains Unicode replacement character")
        if "TODO" in text or "TBD" in text:
            errors.append(f"{source.relative_to(root)}: unresolved TODO/TBD")
    return errors


def _prose_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    fence: tuple[str, int] | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = FENCE_RE.match(line)
        if fence is not None:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= fence[1]:
                fence = None
            continue
        if match:
            marker = match.group(1)
            fence = (marker[0], len(marker))
            continue
        lines.append((line_number, INLINE_CODE_RE.sub("", line)))
    return lines


def check_math_delimiters(files: list[Path], *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    opening = {r"\(": r"\)", r"\[": r"\]"}
    closing = {r"\)": r"\(", r"\]": r"\["}
    for source in files:
        stack: list[tuple[str, int]] = []
        for line_number, line in _prose_lines(source.read_text(encoding="utf-8")):
            for match in MATH_DELIMITER_RE.finditer(line):
                token = match.group(1)
                if token in opening:
                    stack.append((token, line_number))
                    continue
                if not stack or stack[-1][0] != closing[token]:
                    errors.append(
                        f"{source.relative_to(root)}:{line_number}: "
                        f"unexpected math delimiter {token}"
                    )
                    continue
                stack.pop()
        errors.extend(
            f"{source.relative_to(root)}:{line_number}: "
            f"unclosed math delimiter {token}"
            for token, line_number in stack
        )
    return errors


def check_test_references(files: list[Path], *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for source in files:
        references = set(TEST_FILE_RE.findall(source.read_text(encoding="utf-8")))
        errors.extend(
            f"{source.relative_to(root)}: missing test file: {reference}"
            for reference in sorted(references)
            if not (root / reference).is_file()
        )
    return errors


def check_learning_contracts(files: list[Path], *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for source in files:
        text = source.read_text(encoding="utf-8")
        display_path = source.relative_to(root)
        marker_count = text.count(LEARNING_CONTRACT_MARKER)
        if marker_count != 1:
            errors.append(
                f"{display_path}: expected exactly one learning contract marker, "
                f"found {marker_count}"
            )
            continue
        marker_index = text.index(LEARNING_CONTRACT_MARKER)
        if marker_index > 2000:
            errors.append(f"{display_path}: learning contract must appear near the page start")
        contract_end = text.find("</div>", marker_index)
        if contract_end == -1:
            errors.append(f"{display_path}: learning contract is missing closing </div>")
            continue
        contract = text[marker_index:contract_end]
        for field_name in LEARNING_CONTRACT_FIELDS:
            if field_name not in contract:
                errors.append(f"{display_path}: learning contract missing {field_name}")
    return errors


def _readability_metrics(text: str) -> dict[str, int]:
    prose_lengths = [len(line) for _, line in _prose_lines(text)]
    return {
        "line_count": len(text.splitlines()),
        "heading_count": len(HEADING_RE.findall(text)) - 1,
        "overlong_prose_line_count": sum(
            length > READABILITY_DEFAULTS["max_prose_line_length"]
            for length in prose_lengths
        ),
        "max_prose_line_length": max(prose_lengths, default=0),
    }


def check_readability(
    files: list[Path],
    *,
    baseline_path: Path = READABILITY_BASELINE,
    root: Path = ROOT,
) -> list[str]:
    """Enforce strict defaults while ratcheting down explicitly recorded debt."""

    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"{baseline_path}: invalid readability baseline: {error}"]
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return [f"{baseline_path}: expected schema_version 1 object"]
    if payload.get("defaults") != READABILITY_DEFAULTS:
        return [f"{baseline_path}: defaults drifted from the checker contract"]
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        return [f"{baseline_path}: pages must be an object"]

    errors: list[str] = []
    seen: set[str] = set()
    for source in files:
        relative = source.relative_to(root).as_posix()
        seen.add(relative)
        overrides = pages.get(relative, {})
        if not isinstance(overrides, dict) or any(
            key not in READABILITY_DEFAULTS
            or isinstance(value, bool)
            or not isinstance(value, int)
            for key, value in overrides.items()
        ):
            errors.append(f"{relative}: invalid readability baseline entry")
            continue
        metrics = _readability_metrics(source.read_text(encoding="utf-8"))
        for key, actual in metrics.items():
            default = READABILITY_DEFAULTS[key]
            budget = overrides.get(key, default)
            if budget < default:
                errors.append(f"{relative}: {key} baseline cannot be below strict default")
            elif actual > budget:
                errors.append(f"{relative}: {key} {actual} exceeds budget {budget}")
            elif key in overrides and actual < budget:
                errors.append(
                    f"{relative}: {key} improved to {actual}; tighten stale budget {budget}"
                )
        if overrides and not any(
            value > READABILITY_DEFAULTS[key] for key, value in overrides.items()
        ):
            errors.append(f"{relative}: baseline entry records no readability debt")

    for stale in sorted(set(pages) - seen):
        errors.append(f"{baseline_path}: stale readability page: {stale}")
    return errors


def check_evidence_entrypoints(docs: Path = DOCS, *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for source in sorted((docs / "evidence").glob("*.md")):
        introduction = source.read_text(encoding="utf-8")[:1200]
        relative = source.relative_to(root)
        if "第一次" not in introduction:
            errors.append(f"{relative}: evidence ledger must redirect first-time readers")
        targets = parse_document(introduction).targets
        if not any("../" in target and "../evidence/" not in target for target in targets):
            errors.append(f"{relative}: evidence ledger needs a reader-facing entry link")
    return errors


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    files = markdown_files()
    curriculum_files = curriculum_markdown_files()
    reader_files = reader_markdown_files()
    reference_files = sorted({*files, *project_markdown_files()})
    errors = (
        check_basics(files)
        + check_internal_links(files)
        + check_math_delimiters(files)
        + check_test_references(reference_files)
        + check_learning_contracts(curriculum_files)
        + check_readability(reader_files)
        + check_evidence_entrypoints()
    )
    if errors:
        print("Documentation checks failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(
        f"OK: checked {len(files)} Markdown files, assets, local anchors, "
        f"test references, and {len(curriculum_files)} learning contracts"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
