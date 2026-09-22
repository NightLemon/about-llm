"""Advisory structural metrics for reader-facing repository content.

The report is deliberately non-blocking: findings describe pages that deserve a
human review, while process failures are reserved for invalid input,
configuration, or unreadable files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FENCE_RE = re.compile(r"^[ \t]*((?:`{3,})|(?:~{3,}))")
HEADING_RE = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.+?)\s*$")
NAV_PAGE_RE = re.compile(
    r"^[ \t]*-[^:\r\n]+:[ \t]*['\"]?([^'\"#\r\n]+?\.md)['\"]?[ \t]*$",
    re.MULTILINE,
)
HTML_RE = re.compile(r"<[^>]+>")
LINK_RE = re.compile(r"!?\[([^]]*)\]\([^)]+\)")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1")
EVIDENCE_TUTORIAL_RE = re.compile(
    r"^(?:#{1,6}\s+)?(?:学习导航|学习目标|首次阅读|自测|实践任务|面试追问)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Threshold:
    max_lines: int
    max_headings: int
    max_prose_chars: int


THRESHOLDS = {
    "entry": Threshold(max_lines=450, max_headings=32, max_prose_chars=24_000),
    "body": Threshold(max_lines=520, max_headings=40, max_prose_chars=30_000),
    "practice": Threshold(max_lines=500, max_headings=36, max_prose_chars=26_000),
    "readme": Threshold(max_lines=450, max_headings=32, max_prose_chars=22_000),
}


@dataclass(frozen=True)
class Finding:
    kind: str
    message: str
    actual: int | None = None
    limit: int | None = None


@dataclass(frozen=True)
class PageReport:
    path: str
    role: str
    lines: int
    headings: int
    prose_chars: int
    code_lines: int
    table_lines: int
    findings: tuple[Finding, ...]


class ReportInputError(RuntimeError):
    """Raised when the requested repository inputs cannot be inspected."""


def classify_page(path: Path, *, root: Path = ROOT) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    if relative == "README.md" or relative.startswith("docs/guide/") or relative == "docs/index.md":
        return "entry"
    if relative.startswith("docs/reference/") or relative.startswith("docs/decisions/"):
        return "reference"
    if relative.startswith("docs/evidence/"):
        return "evidence"
    if relative.startswith("docs/practice/") or relative.startswith("notebooks/"):
        return "practice"
    if relative.startswith("projects/") and path.name.lower() == "readme.md":
        return "readme"
    if path.name.lower() == "readme.md":
        return "readme"
    return "body"


def canonical_nav_pages(root: Path = ROOT) -> list[Path]:
    config = root / "mkdocs.yml"
    try:
        text = config.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReportInputError(f"cannot read {config}: {error}") from error

    nav_match = re.search(
        r"(?ms)^nav:\s*\n(?P<nav>.*?)(?=^[A-Za-z_][A-Za-z0-9_]*:\s*(?:\n|\|))",
        text,
    )
    if nav_match is None:
        raise ReportInputError(f"{config}: missing top-level nav")

    relative_paths = [
        match.group(1).strip()
        for match in NAV_PAGE_RE.finditer(nav_match.group("nav"))
    ]
    if not relative_paths:
        raise ReportInputError(f"{config}: nav does not contain Markdown pages")

    pages: list[Path] = []
    seen: set[str] = set()
    for relative in relative_paths:
        if relative in seen:
            raise ReportInputError(f"{config}: duplicate nav page: {relative}")
        seen.add(relative)
        page = root / "docs" / relative
        if not page.is_file():
            raise ReportInputError(f"{config}: nav page does not exist: {relative}")
        pages.append(page)
    return pages


def content_files(scope: str, *, root: Path = ROOT) -> list[Path]:
    if scope not in {"reader", "all"}:
        raise ValueError(f"unsupported scope: {scope}")

    docs = root / "docs"
    projects = root / "projects"
    notebooks = root / "notebooks"
    if not docs.is_dir():
        raise ReportInputError(f"missing docs directory: {docs}")

    project_readmes = list(projects.glob("*/README.md")) if projects.is_dir() else []
    notebook_pages: list[Path] = []
    if notebooks.is_dir():
        notebook_pages.extend(notebooks.glob("*.ipynb"))
        notebook_readme = notebooks / "README.md"
        if notebook_readme.is_file():
            notebook_pages.append(notebook_readme)

    if scope == "reader":
        pages = canonical_nav_pages(root)
        root_readme = root / "README.md"
        if not root_readme.is_file():
            raise ReportInputError(f"missing reader entry: {root_readme}")
        pages.extend([root_readme, *project_readmes, *notebook_pages])
    else:
        pages = [*docs.rglob("*.md"), *project_readmes, *notebook_pages]
        pages.extend(
            path
            for name in ("README.md", "CONTRIBUTING.md")
            if (path := root / name).is_file()
        )
    return sorted({path.resolve() for path in pages})


def _notebook_text(path: Path) -> tuple[str, set[int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportInputError(f"cannot read notebook {path}: {error}") from error
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ReportInputError(f"{path}: notebook cells must be an array")

    lines: list[str] = []
    forced_code_lines: set[int] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ReportInputError(f"{path}: notebook cell must be an object")
        source = cell.get("source", [])
        if isinstance(source, str):
            cell_lines = source.splitlines()
        elif isinstance(source, list) and all(isinstance(item, str) for item in source):
            cell_lines = "".join(source).splitlines()
        else:
            raise ReportInputError(f"{path}: notebook cell source must be text")
        start = len(lines) + 1
        lines.extend(cell_lines)
        if cell.get("cell_type") == "code":
            forced_code_lines.update(range(start, len(lines) + 1))
        lines.append("")
    return "\n".join(lines), forced_code_lines


def _read_page(path: Path) -> tuple[str, set[int]]:
    if path.suffix.lower() == ".ipynb":
        return _notebook_text(path)
    try:
        return path.read_text(encoding="utf-8"), set()
    except (OSError, UnicodeError) as error:
        raise ReportInputError(f"cannot read {path}: {error}") from error


def measure_text(text: str, *, forced_code_lines: set[int] | None = None) -> dict[str, int]:
    forced_code_lines = forced_code_lines or set()
    lines = text.splitlines()
    headings = 0
    code_lines = 0
    table_lines = 0
    prose_parts: list[str] = []
    fence: tuple[str, int] | None = None
    in_math = False

    for line_number, raw in enumerate(lines, start=1):
        marker = FENCE_RE.match(raw)
        if line_number in forced_code_lines:
            code_lines += 1
            continue
        if fence is not None:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= fence[1]:
                fence = None
            else:
                code_lines += 1
            continue
        if marker:
            token = marker.group(1)
            fence = (token[0], len(token))
            continue

        stripped = raw.strip()
        if in_math:
            if stripped in {r"\]", "$$"}:
                in_math = False
            continue
        if stripped in {r"\[", "$$"}:
            in_math = True
            continue
        if stripped.startswith("|"):
            table_lines += 1
            continue
        heading = HEADING_RE.match(raw)
        if heading:
            if len(heading.group(1)) >= 2:
                headings += 1
            prose_parts.append(heading.group(2))
            continue
        if not stripped or stripped.startswith(("<!--", "</", "<div", "{ .")):
            continue
        if raw.startswith(("    ", "\t")):
            code_lines += 1
            continue
        cleaned = INLINE_CODE_RE.sub("", stripped)
        cleaned = LINK_RE.sub(lambda match: match.group(1), cleaned)
        cleaned = HTML_RE.sub("", cleaned)
        cleaned = re.sub(r"^[-*+]\s+", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned)
        prose_parts.append(cleaned)

    prose_chars = sum(len(re.sub(r"\s+", "", part)) for part in prose_parts)
    return {
        "lines": len(lines),
        "headings": headings,
        "prose_chars": prose_chars,
        "code_lines": code_lines,
        "table_lines": table_lines,
    }


def advisory_findings(role: str, metrics: dict[str, int], text: str) -> tuple[Finding, ...]:
    if role == "reference":
        return ()
    if role == "evidence":
        signals = len(EVIDENCE_TUTORIAL_RE.findall(text))
        if "<!-- learning-contract -->" in text or signals >= 2:
            return (
                Finding(
                    kind="evidence-role-drift",
                    message="证据页出现线性教程信号; 保留台账, 并把首次学习路线指回正文",
                    actual=signals,
                ),
            )
        return ()

    threshold = THRESHOLDS[role]
    findings: list[Finding] = []
    checks = (
        ("page-length", "页面行数", "lines", threshold.max_lines),
        ("section-count", "小节数量", "headings", threshold.max_headings),
        ("prose-density", "正文字符数", "prose_chars", threshold.max_prose_chars),
    )
    for kind, label, field, limit in checks:
        actual = metrics[field]
        if actual > limit:
            findings.append(
                Finding(
                    kind=kind,
                    message=f"{label} {actual} 超过 {role} 页提示阈值 {limit}",
                    actual=actual,
                    limit=limit,
                )
            )
    return tuple(findings)


def analyze_page(path: Path, *, root: Path = ROOT) -> PageReport:
    text, forced_code = _read_page(path)
    metrics = measure_text(text, forced_code_lines=forced_code)
    role = classify_page(path, root=root)
    return PageReport(
        path=path.resolve().relative_to(root.resolve()).as_posix(),
        role=role,
        findings=advisory_findings(role, metrics, text),
        **metrics,
    )


def build_report(scope: str, *, root: Path = ROOT) -> dict[str, Any]:
    pages = [analyze_page(path, root=root) for path in content_files(scope, root=root)]
    return {
        "schema_version": 1,
        "scope": scope,
        "page_count": len(pages),
        "finding_count": sum(len(page.findings) for page in pages),
        "pages": [asdict(page) for page in pages],
    }


def text_report(payload: dict[str, Any]) -> str:
    lines = [
        (
            f"Content structure report: {payload['page_count']} pages, "
            f"{payload['finding_count']} advisory findings"
        )
    ]
    for page in payload["pages"]:
        metrics = (
            f"lines={page['lines']} headings={page['headings']} "
            f"prose={page['prose_chars']} code={page['code_lines']} tables={page['table_lines']}"
        )
        lines.append(f"- {page['path']} [{page['role']}] {metrics}")
        for finding in page["findings"]:
            lines.append(f"  - {finding['kind']}: {finding['message']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("reader", "all"), default="reader")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        payload = build_report(args.scope)
    except (OSError, UnicodeError, ValueError, ReportInputError) as error:
        print(f"content structure report failed: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
