from __future__ import annotations

import pytest

from scripts.probe_sources import build_report, content_fingerprint, probe_source

pytestmark = [pytest.mark.contract, pytest.mark.security]


def _source(**overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "id": "example-official",
        "url": "https://example.com/official",
        "status": "verified",
    }
    source.update(overrides)
    return source


def test_probe_marks_reachable_source_verified() -> None:
    result = probe_source(
        _source(),
        fetcher=lambda url: {"http_status": 200, "fingerprint": "sha256:current"},
    )

    assert result["status"] == "verified"


def test_probe_marks_network_failure_unknown() -> None:
    def fail(url: str) -> dict[str, object]:
        raise TimeoutError(url)

    result = probe_source(_source(), fetcher=fail)

    assert result["status"] == "unknown"
    assert "TimeoutError" in result["error"]


def test_probe_marks_fingerprint_drift_pending_review() -> None:
    result = probe_source(
        _source(fingerprint="sha256:reviewed", fingerprint_kind="visible-text-v1"),
        fetcher=lambda url: {
            "http_status": 200,
            "fingerprint": "sha256:changed",
            "fingerprint_kind": "visible-text-v1",
        },
    )

    assert result["status"] == "pending-review"
    assert result["expected_fingerprint"] == "sha256:reviewed"


def test_html_fingerprint_ignores_script_nonce_but_detects_visible_change() -> None:
    first, kind = content_fingerprint(
        b"<html><body><h1>Contract</h1><script>nonce=one</script></body></html>",
        "text/html; charset=utf-8",
    )
    second, _ = content_fingerprint(
        b"<html><body><h1>Contract</h1><script>nonce=two</script></body></html>",
        "text/html; charset=utf-8",
    )
    changed, _ = content_fingerprint(
        b"<html><body><h1>Changed</h1><script>nonce=two</script></body></html>",
        "text/html; charset=utf-8",
    )

    assert kind == "visible-text-v1"
    assert first == second
    assert first != changed


def test_report_keeps_failures_as_data() -> None:
    def mixed(url: str) -> dict[str, object]:
        if url.endswith("/bad"):
            raise OSError("offline")
        return {"http_status": 200, "fingerprint": "sha256:current"}

    report = build_report(
        [_source(), _source(id="second", url="https://example.com/bad")],
        workers=2,
        fetcher=mixed,
    )

    assert [item["status"] for item in report["sources"]] == ["verified", "unknown"]
