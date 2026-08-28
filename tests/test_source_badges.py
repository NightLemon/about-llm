from __future__ import annotations

from datetime import date

import pytest

from scripts.source_badges import render_source_markers

pytestmark = [pytest.mark.contract]


def _source(**overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "id": "example-official",
        "url": "https://example.com/official",
        "scope": "example contract",
        "status": "verified",
        "next_review_at": "2026-09-01",
    }
    source.update(overrides)
    return source


def test_badge_renders_without_page_warning_for_current_source() -> None:
    rendered = render_source_markers(
        "# Page\n\nClaim [SOURCE:example-official]",
        sources={"example-official": _source()},
        as_of=date(2026, 8, 28),
    )

    assert "source-status-verified" in rendered
    assert "本页来源状态已降级" not in rendered


def test_badge_and_page_warning_render_for_all_degraded_states() -> None:
    for status in ("stale", "unknown", "pending-review"):
        source = _source(next_review_at="2026-08-01" if status == "stale" else "2026-09-01")
        probe = None if status == "stale" else {"example-official": {"status": status}}
        rendered = render_source_markers(
            "# Page\n\nClaim [SOURCE:example-official]",
            sources={"example-official": source},
            probes=probe,
            as_of=date(2026, 8, 28),
        )

        assert f"source-status-{status}" in rendered
        assert "本页来源状态已降级" in rendered
        assert "不能判断正文结论是否仍成立" in rendered
