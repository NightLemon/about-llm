"""Probe registered sources without turning network health into a build gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

try:
    from scripts.content_quality import SOURCE_REGISTRY, load_registry
except ModuleNotFoundError:  # Direct `python scripts/probe_sources.py` execution.
    from content_quality import SOURCE_REGISTRY, load_registry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts" / "source-health.json"
MAX_FINGERPRINT_BYTES = 512 * 1024
WHITESPACE_RE = re.compile(r"\s+")


class _VisibleTextParser(HTMLParser):
    """Extract reader-visible text while ignoring volatile scripts and styling."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self.suppressed_depth or tag in {"script", "style", "noscript", "svg"}:
            self.suppressed_depth += 1

    def handle_endtag(self, tag: str) -> None:
        del tag
        if self.suppressed_depth:
            self.suppressed_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.suppressed_depth:
            self.parts.append(data)


def content_fingerprint(content: bytes, content_type: str) -> tuple[str, str]:
    """Hash normalized visible HTML text or the bounded raw payload for other media."""
    if "html" in content_type.casefold():
        parser = _VisibleTextParser()
        parser.feed(content.decode("utf-8", errors="replace"))
        normalized = WHITESPACE_RE.sub(" ", " ".join(parser.parts)).strip().encode("utf-8")
        return f"sha256:{hashlib.sha256(normalized).hexdigest()}", "visible-text-v1"
    return f"sha256:{hashlib.sha256(content).hexdigest()}", "raw-prefix-v1"


def fetch_source(url: str, *, timeout: float = 20.0) -> dict[str, Any]:
    """Fetch a bounded prefix and return metadata suitable for change detection."""
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.1",
            "Range": f"bytes=0-{MAX_FINGERPRINT_BYTES - 1}",
            "User-Agent": "about-llm-source-probe/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content = response.read(MAX_FINGERPRINT_BYTES + 1)
        bounded_content = content[:MAX_FINGERPRINT_BYTES]
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        fingerprint, fingerprint_kind = content_fingerprint(bounded_content, content_type)
        return {
            "http_status": response.status,
            "final_url": response.geturl(),
            "fingerprint": fingerprint,
            "fingerprint_kind": fingerprint_kind,
            "fingerprint_bytes": len(bounded_content),
            "truncated": len(content) >= MAX_FINGERPRINT_BYTES,
            "content_type": content_type,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def probe_source(
    source: dict[str, Any],
    *,
    fetcher: Callable[[str], dict[str, Any]] = fetch_source,
) -> dict[str, Any]:
    """Return verified, unknown, or pending-review while preserving the reviewed registry."""
    result: dict[str, Any] = {"id": source["id"], "url": source["url"]}
    try:
        observation = fetcher(str(source["url"]))
    except Exception as error:  # Network failures are data, not a failed documentation build.
        result.update(
            {
                "status": "unknown",
                "error": f"{type(error).__name__}: {error}"[:500],
            }
        )
        return result

    expected_fingerprint = source.get("fingerprint")
    observed_fingerprint = observation.get("fingerprint")
    expected_kind = source.get("fingerprint_kind")
    observed_kind = observation.get("fingerprint_kind")
    status = (
        "pending-review"
        if expected_fingerprint is not None
        and (observed_fingerprint != expected_fingerprint or observed_kind != expected_kind)
        else "verified"
    )
    result.update(observation)
    result["status"] = status
    if status == "pending-review":
        result["expected_fingerprint"] = expected_fingerprint
        result["expected_fingerprint_kind"] = expected_kind
    return result


def build_report(
    sources: list[dict[str, Any]],
    *,
    workers: int = 8,
    fetcher: Callable[[str], dict[str, Any]] = fetch_source,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda source: probe_source(source, fetcher=fetcher), sources))
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=SOURCE_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    sources, errors = load_registry(args.registry)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2
    report = build_report(sources, workers=max(1, args.workers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for result in report["sources"]:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(f"Source probe report: {args.output} ({counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
