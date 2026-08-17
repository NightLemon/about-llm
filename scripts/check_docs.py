"""Structured checks for the Markdown knowledge base."""

from __future__ import annotations

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


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    files = markdown_files()
    curriculum_files = curriculum_markdown_files()
    errors = (
        check_basics(files)
        + check_internal_links(files)
        + check_learning_contracts(curriculum_files)
    )
    if errors:
        print("Documentation checks failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(
        f"OK: checked {len(files)} Markdown files, assets, local anchors, "
        f"and {len(curriculum_files)} learning contracts"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
