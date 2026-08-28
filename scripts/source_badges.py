"""MkDocs hook that renders registered source markers and visible degradation notices."""

from __future__ import annotations

import html
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

try:
    from scripts.content_quality import SOURCE_MARKER_RE, SOURCE_REGISTRY, effective_source_status
except ModuleNotFoundError:  # MkDocs loads local hooks from the scripts directory.
    from content_quality import SOURCE_MARKER_RE, SOURCE_REGISTRY, effective_source_status

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts" / "source-health.json"
STATUS_LABELS = {
    "verified": "已核验",
    "stale": "已过复核期",
    "unknown": "暂时无法访问",
    "pending-review": "内容变化 / 待人工复核",
    "rejected": "已拒绝",
}

_sources: dict[str, dict[str, Any]] = {}
_probes: dict[str, dict[str, Any]] = {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_views(
    *,
    registry_path: Path = SOURCE_REGISTRY,
    report_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    registry = _load_json(registry_path)
    sources = {source["id"]: source for source in registry["sources"]}
    probes: dict[str, dict[str, Any]] = {}
    if report_path is not None and report_path.is_file():
        report = _load_json(report_path)
        probes = {source["id"]: source for source in report.get("sources", [])}
    return sources, probes


def render_source_markers(
    markdown: str,
    *,
    sources: dict[str, dict[str, Any]],
    probes: dict[str, dict[str, Any]] | None = None,
    as_of: date | None = None,
) -> str:
    effective_probes = probes or {}
    degraded: dict[str, str] = {}

    def replace_marker(match: Any) -> str:
        source_id = match.group(1)
        source = sources[source_id]
        status = effective_source_status(
            source,
            as_of=as_of,
            probe=effective_probes.get(source_id),
        )
        if status != "verified":
            degraded[source_id] = status
        label = STATUS_LABELS.get(status, status)
        return (
            f'<a class="source-badge source-status-{html.escape(status)}" '
            f'data-source-id="{html.escape(source_id)}" '
            f'href="{html.escape(str(source["url"]), quote=True)}" '
            f'title="{html.escape(str(source["scope"]), quote=True)}">'
            f"来源 {html.escape(source_id)} · {html.escape(label)}</a>"
        )

    rendered = SOURCE_MARKER_RE.sub(replace_marker, markdown)
    if not degraded:
        return rendered
    detail = " / ".join(
        f"`{source_id}`: {STATUS_LABELS[status]}" for source_id, status in degraded.items()
    )
    warning = (
        '\n!!! warning "本页来源状态已降级"\n'
        f"    {detail}。自动探测只能发现过期、不可访问或内容变化, "
        "不能判断正文结论是否仍成立; 请等待人工复核。\n"
    )
    first_line, separator, remainder = rendered.partition("\n")
    return f"{first_line}\n{warning}{remainder}" if separator else f"{rendered}{warning}"


def on_config(config: Any) -> Any:
    global _sources, _probes
    configured_report = os.environ.get("ABOUT_LLM_SOURCE_HEALTH_REPORT")
    report_path = Path(configured_report) if configured_report else DEFAULT_REPORT
    _sources, _probes = load_source_views(report_path=report_path)
    return config


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    del page, config, files
    return render_source_markers(markdown, sources=_sources, probes=_probes)
