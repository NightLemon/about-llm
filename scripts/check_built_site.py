"""Validate the static site artifact before GitHub Pages upload."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

SITEMAP_NAMESPACE = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
REQUIRED_INDEX_REFERENCES = (
    "mathjax@4.1.3/tex-mml-chtml.js",
    "mermaid@11.16.0/dist/mermaid.min.js",
    "javascripts/mermaid.js",
    "stylesheets/extra.css",
)
REQUIRED_PAGE_CONTENT: Mapping[str, tuple[str, ...]] = {
    "index.html": ("建议的第一小时", "新手知识地图"),
    "guide/beginner-map/index.html": ("新手知识地图", "30 分钟最小成功"),
    "practice/labs/lab-0a-sampling/index.html": ("实验 0A", "最低通过"),
    "core/generation-basics/index.html": ("生成与解码入门", "最小生成循环"),
    "training/alignment-basics/index.html": ("对齐与偏好优化入门", "最小离线审计"),
    "papers/index.html": ("近期论文解读", "热门不等于重要"),
    "papers/2026-08/index.html": ("近期热门论文解读：2026 年 8 月", "六篇榜单入选"),  # noqa: RUF001
    "applications/agent-interoperability/index.html": (
        "Agent 互操作：MCP、A2A 与内部契约",  # noqa: RUF001
        "可运行的官方 MCP SDK stdio control",
        "可运行的官方 MCP SDK Streamable HTTP control",
        "可运行的自写 strict MCP stdio control",
        "可运行的 A2A 1.0 loopback control",
    ),
}


def check_site(
    site_dir: Path,
    site_url: str,
    minimum_urls: int,
    required_page_content: Mapping[str, tuple[str, ...]] = REQUIRED_PAGE_CONTENT,
) -> list[str]:
    errors: list[str] = []
    index = site_dir / "index.html"
    sitemap = site_dir / "sitemap.xml"
    if not index.is_file():
        errors.append(f"missing site index: {index}")
    else:
        index_html = index.read_text(encoding="utf-8")
        for reference in REQUIRED_INDEX_REFERENCES:
            if reference not in index_html:
                errors.append(f"site index is missing required asset reference: {reference}")
    for relative_path, tokens in required_page_content.items():
        page = site_dir / relative_path
        if not page.is_file():
            errors.append(f"missing required built page: {relative_path}")
            continue
        page_html = page.read_text(encoding="utf-8")
        for token in tokens:
            if token not in page_html:
                errors.append(f"{relative_path} is missing current content token: {token}")
    if (site_dir / "overrides").exists():
        errors.append("theme override sources must not be published under site/overrides")
    if not sitemap.is_file():
        return [*errors, f"missing sitemap: {sitemap}"]
    try:
        root = ET.parse(sitemap).getroot()
    except ET.ParseError as error:
        return [*errors, f"invalid sitemap XML: {error}"]
    locations = [
        element.text or "" for element in root.findall("sitemap:url/sitemap:loc", SITEMAP_NAMESPACE)
    ]
    if len(locations) < minimum_urls:
        errors.append(f"sitemap has {len(locations)} URLs; expected at least {minimum_urls}")
    invalid_locations = [location for location in locations if not location.startswith(site_url)]
    if invalid_locations:
        errors.append(f"sitemap URLs do not use {site_url}: {invalid_locations[:3]}")
    return errors


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--site-url", default="https://nightlemon.github.io/about-llm/")
    parser.add_argument("--minimum-urls", type=int, default=70)
    args = parser.parse_args()
    errors = check_site(args.site_dir, args.site_url, args.minimum_urls)
    if errors:
        print("Built site checks failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"OK: validated {args.site_dir} for {args.site_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
