"""Advisory readability report for reader-facing Markdown.

The report deliberately does not fail on style findings.  It points reviewers to
prose that deserves a human pass while leaving technical terms, code, tables,
and formulas alone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROJECTS = ROOT / "projects"
NON_READER_DIRECTORIES = {"assets", "evidence", "papers", "reference", "_templates"}

FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1")
LINK_RE = re.compile(r"!?\[([^]]*)\]\([^)]+\)")
URL_RE = re.compile(r"https?://\S+")
ATTR_LIST_RE = re.compile(r"\s*\{\s*[#.][^}]*\}\s*$")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
CHINESE_RE = re.compile(r"[\u3400-\u9fff]")
NEGATION_RE = re.compile(r"不能|不证明|不等于|并不|并非|不是|不代表|不要|没有|未能")
JARGON_PATTERNS = (
    re.compile(r"\bauthored\s+(?:fixture|control|contract|policy|metadata)\b", re.I),
    re.compile(
        r"\bstrict\s+(?:JSON|loader|parser|verifier|control|artifact|report|schema)\b",
        re.I,
    ),
    re.compile(r"\b(?:identity|revision|fingerprint)\s+(?:drift|boundary|binding)\b", re.I),
    re.compile(r"\b(?:fixture|control|oracle)\b", re.I),
)


@dataclass(frozen=True)
class ProseBlock:
    line: int
    text: str


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    message: str
    excerpt: str


def markdown_files(scope: str, *, root: Path = ROOT) -> list[Path]:
    docs = root / "docs"
    projects = root / "projects"
    reader_docs = [
        path
        for path in docs.rglob("*.md")
        if path.relative_to(docs).parts[0] not in NON_READER_DIRECTORIES
    ]
    project_readmes = list(projects.glob("*/README.md")) if projects.exists() else []
    root_pages = [
        path
        for name in ("README.md", "CONTRIBUTING.md")
        if (path := root / name).is_file()
    ]

    if scope == "reader":
        return sorted({*root_pages, *reader_docs, *project_readmes})
    if scope == "all":
        return sorted({*root_pages, *docs.rglob("*.md"), *project_readmes})
    raise ValueError(f"unsupported scope: {scope}")


def _clean_markup(text: str) -> str:
    text = INLINE_CODE_RE.sub("", text)
    text = LINK_RE.sub(lambda match: match.group(1), text)
    text = URL_RE.sub("", text)
    text = ATTR_LIST_RE.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", text)
    return " ".join(text.split())


def prose_blocks(text: str) -> Iterator[ProseBlock]:
    """Yield prose paragraphs while skipping code, tables, and display math."""

    fence: tuple[str, int] | None = None
    in_math = False
    buffered: list[str] = []
    start_line = 0

    def flush() -> Iterator[ProseBlock]:
        nonlocal buffered, start_line
        cleaned = _clean_markup(" ".join(buffered))
        buffered = []
        if cleaned:
            yield ProseBlock(start_line, cleaned)

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fence_match = FENCE_RE.match(raw_line)
        if fence is not None:
            if (
                fence_match
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= fence[1]
            ):
                fence = None
            continue
        if fence_match:
            yield from flush()
            marker = fence_match.group(1)
            fence = (marker[0], len(marker))
            continue

        stripped = raw_line.strip()
        if in_math:
            if stripped in {r"\]", "$$"}:
                in_math = False
            continue
        if stripped in {r"\[", "$$"}:
            yield from flush()
            in_math = True
            continue
        if not stripped:
            yield from flush()
            continue
        if stripped.startswith("|") or raw_line.startswith(("    ", "\t")):
            yield from flush()
            continue
        if stripped.startswith(("<!--", "</", "<div", "{ .")):
            yield from flush()
            continue

        is_heading_or_list = bool(
            re.match(r"^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)", stripped)
        )
        if is_heading_or_list:
            yield from flush()
            start_line = line_number
            buffered = [stripped]
            yield from flush()
            continue
        if not buffered:
            start_line = line_number
        buffered.append(stripped)

    yield from flush()


def analyze_text(text: str, *, display_path: str) -> list[Finding]:
    findings: list[Finding] = []
    for block in prose_blocks(text):
        excerpt = block.text[:180] + ("…" if len(block.text) > 180 else "")
        if any(pattern.search(block.text) for pattern in JARGON_PATTERNS):
            findings.append(
                Finding(
                    path=display_path,
                    line=block.line,
                    kind="internal-jargon",
                    message="内部验证术语需要解释, 或改成具体的中文说法",
                    excerpt=excerpt,
                )
            )

        negations = len(NEGATION_RE.findall(block.text))
        if negations >= 3:
            findings.append(
                Finding(
                    path=display_path,
                    line=block.line,
                    kind="negation-chain",
                    message=f"同一段出现 {negations} 个否定表达, 可集中说明适用范围",
                    excerpt=excerpt,
                )
            )

        punctuation = sum(block.text.count(mark) for mark in "\uff0c\uff1b\u3001")
        if len(block.text) >= 150 or (len(block.text) >= 90 and punctuation >= 8):
            findings.append(
                Finding(
                    path=display_path,
                    line=block.line,
                    kind="dense-sentence",
                    message="句子承载的条件过多, 建议拆出主语、因果和先后顺序",
                    excerpt=excerpt,
                )
            )

        english_words = ENGLISH_WORD_RE.findall(block.text)
        if CHINESE_RE.search(block.text) and len(english_words) >= 8:
            findings.append(
                Finding(
                    path=display_path,
                    line=block.line,
                    kind="english-stack",
                    message=(
                        f"一段中连续承载 {len(english_words)} 个英文词, "
                        "请确认中文已经解释关系"
                    ),
                    excerpt=excerpt,
                )
            )
    return findings


def collect_findings(files: Iterable[Path], *, root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for path in files:
        display_path = path.relative_to(root).as_posix()
        findings.extend(
            analyze_text(path.read_text(encoding="utf-8"), display_path=display_path)
        )
    return findings


def _text_report(findings: list[Finding], *, files_scanned: int, top: int) -> str:
    counts = Counter(finding.kind for finding in findings)
    weights = {
        "internal-jargon": 4,
        "negation-chain": 3,
        "dense-sentence": 2,
        "english-stack": 1,
    }
    page_scores: Counter[str] = Counter()
    page_counts: Counter[str] = Counter()
    for finding in findings:
        page_scores[finding.path] += weights[finding.kind]
        page_counts[finding.path] += 1
    ranked_pages = sorted(page_scores, key=lambda path: (-page_scores[path], path))
    ranked_findings = sorted(
        findings,
        key=lambda finding: (
            -page_scores[finding.path],
            -weights[finding.kind],
            finding.path,
            finding.line,
        ),
    )
    lines = [
        f"Readability report: {files_scanned} files, {len(findings)} advisory findings",
        "Categories: "
        + (", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())) or "none"),
    ]
    if ranked_pages:
        hotspots = ", ".join(
            f"{path}={page_counts[path]}" for path in ranked_pages[:5]
        )
        lines.append(f"Highest-priority pages: {hotspots}")
    for finding in ranked_findings[:top]:
        lines.extend(
            [
                f"- {finding.path}:{finding.line} [{finding.kind}] {finding.message}",
                f"  {finding.excerpt}",
            ]
        )
    if len(ranked_findings) > top:
        lines.append(
            f"… {len(ranked_findings) - top} more findings; "
            "use --format json for the full report"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("reader", "all"), default="reader")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--top", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")

    try:
        files = markdown_files(args.scope)
        findings = collect_findings(files)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"readability report failed: {error}", file=sys.stderr)
        return 2

    if args.format == "json":
        payload = {
            "schema_version": 1,
            "scope": args.scope,
            "files_scanned": len(files),
            "finding_count": len(findings),
            "findings": [asdict(finding) for finding in findings],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_text_report(findings, files_scanned=len(files), top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
