from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.check_built_site import REQUIRED_INDEX_REFERENCES, REQUIRED_PAGE_CONTENT, check_site


def _write_site(site: Path, *, url_count: int = 2) -> None:
    site.mkdir()
    references = "\n".join(REQUIRED_INDEX_REFERENCES)
    for relative_path, tokens in REQUIRED_PAGE_CONTENT.items():
        page = site / relative_path
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("\n".join((references, *tokens)), encoding="utf-8")
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    root = ET.Element(f"{{{namespace}}}urlset")
    for index in range(url_count):
        url = ET.SubElement(root, f"{{{namespace}}}url")
        ET.SubElement(url, f"{{{namespace}}}loc").text = (
            f"https://example.test/project/page-{index}/"
        )
    ET.ElementTree(root).write(site / "sitemap.xml", encoding="utf-8", xml_declaration=True)


def test_check_site_accepts_complete_artifact(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)

    assert check_site(site, "https://example.test/project/", minimum_urls=2) == []


def test_check_site_rejects_missing_asset_and_published_override(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "index.html").write_text("stylesheets/extra.css", encoding="utf-8")
    (site / "overrides").mkdir()

    errors = check_site(site, "https://example.test/project/", minimum_urls=2)

    assert any("missing required asset reference" in error for error in errors)
    assert any("override sources" in error for error in errors)


def test_check_site_rejects_stale_or_missing_required_page(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    stale_page = site / "core" / "generation-basics" / "index.html"
    stale_page.write_text("生成与解码入门", encoding="utf-8")
    missing_page = site / "training" / "alignment-basics" / "index.html"
    missing_page.unlink()

    errors = check_site(site, "https://example.test/project/", minimum_urls=2)

    assert any("is missing current content token: 最小生成循环" in error for error in errors)
    assert any(
        "missing required built page: training/alignment-basics" in error
        for error in errors
    )


def test_check_site_rejects_stale_paper_snapshot(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    snapshot = site / "papers" / "2026-08" / "index.html"
    snapshot.write_text("近期热门论文解读：2026 年 8 月", encoding="utf-8")  # noqa: RUF001

    errors = check_site(site, "https://example.test/project/", minimum_urls=2)

    assert any("is missing current content token: 六篇榜单入选" in error for error in errors)


def test_cli_reports_unicode_error_under_non_utf8_stdout(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    stale_page = site / "applications" / "agent-interoperability" / "index.html"
    stale_page.write_text("Agent 互操作", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "check_built_site.py"
    environment = {**os.environ, "PYTHONIOENCODING": "cp1252"}

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--site-dir",
            str(site),
            "--site-url",
            "https://example.test/project/",
            "--minimum-urls",
            "2",
        ],
        capture_output=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 1
    assert b"Built site checks failed" in completed.stdout
    assert b"UnicodeEncodeError" not in completed.stderr
