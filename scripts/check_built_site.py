"""Validate the static site artifact before GitHub Pages upload."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SITEMAP_NAMESPACE = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
REQUIRED_INDEX_REFERENCES = (
    "mathjax@4.1.3/tex-mml-chtml.js",
    "mermaid@11.16.0/dist/mermaid.min.js",
    "javascripts/mermaid.js",
    "stylesheets/extra.css",
)


def check_site(
    site_dir: Path,
    site_url: str,
    minimum_urls: int,
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
