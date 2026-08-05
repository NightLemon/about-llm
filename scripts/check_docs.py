"""Small dependency-free checks for the Markdown knowledge base."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return sorted([ROOT / "README.md", ROOT / "CONTRIBUTING.md", *DOCS.rglob("*.md")])


def check_internal_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for source in files:
        text = source.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            raw = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = unquote(raw.split("#", 1)[0])
            if not path_part:
                continue
            target = (source.parent / path_part).resolve()
            try:
                target.relative_to(ROOT)
            except ValueError:
                errors.append(f"{source.relative_to(ROOT)}: link escapes repo: {raw}")
                continue
            if not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{source.relative_to(ROOT)}:{line}: missing target: {raw}")
    return errors


def check_basics(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for source in files:
        text = source.read_text(encoding="utf-8")
        if not text.startswith("# "):
            errors.append(f"{source.relative_to(ROOT)}: missing H1 at first line")
        if "\ufffd" in text:
            errors.append(f"{source.relative_to(ROOT)}: contains Unicode replacement character")
        if "TODO" in text or "TBD" in text:
            errors.append(f"{source.relative_to(ROOT)}: unresolved TODO/TBD")
    return errors


def main() -> int:
    files = markdown_files()
    errors = check_basics(files) + check_internal_links(files)
    if errors:
        print("Documentation checks failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"OK: checked {len(files)} Markdown files and their local links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
