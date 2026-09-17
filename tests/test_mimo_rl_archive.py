"""Supplied two-step fixtures check archive contracts, not MiMo training truth."""

from __future__ import annotations

import ast
import copy
import gzip
import json
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import mimo_rl_archive as archive

pytestmark = pytest.mark.contract


def status(version: str = "v1") -> dict:
    return {"version": version, "run": {"key": "pro", "start": 100},
            "step": {"last": 2}, "totals": {"tokens_cum": 30}, "clock": {"now": 120},
            "headline": {"last": 0.75}, "events": [], "cost": {"so_far": 2}}


def series() -> dict:
    return {"run": "pro", "version": "v1", "steps": [1, 2], "walls": [110, 120],
            "run_start": 100, "series": {"dynsam/avg@n": [0.5, 0.75], "empty": [None, None]}}


def run_data() -> dict:
    result = {"key": "pro", "version": "v1", "tags": ["dynsam/avg@n", "empty"],
              "series": {}, "status_before": archive.project_status(status()),
              "status_after": archive.project_status(status())}
    archive.merge_series(result, result["tags"], series())
    result["coverage"] = archive.coverage(result)
    return result


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, Path]:
    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir()
    private.mkdir()
    run = run_data()
    facts = {"schema": archive.SCHEMA, "source": archive.BASE,
             "runs": {"pro": run}, "benchmarks": []}
    raw = b'{"number": 1}\r\n'
    (private / "response-0001.bin").write_bytes(raw)
    manifest = {"schema": archive.SCHEMA, "source": archive.BASE, "snapshot_id": "fixture",
                "status": "complete", "issues": [], "coverage": {"pro": run["coverage"]},
                "requests": [{"url": archive.BASE + "api/runs", "ok": True,
                              "started_at": "2026-09-17T01:00:00+00:00",
                              "finished_at": "2026-09-17T01:00:01+00:00",
                              "raw_file": "response-0001.bin", "bytes": len(raw),
                              "sha256": archive.digest(raw)}]}
    archive.write_snapshot(public, private, facts, manifest)
    return public, private


def test_projection_preserves_unknown_null_and_zero_separately() -> None:
    assert archive.numeric_tree({"missing": None, "zero": 0, "false": False,
                                 "notice": "unreviewed prose", "nested": [1, None]}) == {
        "missing": None, "zero": 0, "false": False, "nested": [1, None]}
    assert archive.digest(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    with pytest.raises(ValueError, match="schema"):
        archive.project_status(status() | {"unreviewed": {"data": 1}})


@pytest.mark.parametrize("change", [
    "version", "run", "axis", "length", "bool", "nan", "extra", "start", "start-type",
])
def test_series_rejects_incompatible_batches(change: str) -> None:
    run = run_data()
    data = series()
    if change in {"version", "run"}:
        data[change] = "other"
    elif change == "axis":
        data["walls"][0] = 111
    elif change == "length":
        data["series"]["empty"] = [None]
    elif change == "extra":
        data["series"]["unknown"] = [1, 2]
    elif change == "start":
        data["run_start"] = 101
    elif change == "start-type":
        data["run_start"] = None
    else:
        data["series"]["empty"][0] = True if change == "bool" else float("nan")
    before = copy.deepcopy(run)
    with pytest.raises(ValueError):
        archive.merge_series(run, run["tags"], data)
    assert run == before


def test_coverage_distinguishes_missing_empty_and_version_drift() -> None:
    run = run_data()
    run["tags"].append("missing")
    run["status_after"]["version"] = "v2"
    assert archive.coverage(run) == {
        "expected_tags": 3, "received_tags": 2, "missing_tags": ["missing"],
        "empty_tags": ["empty"], "steps": 2, "version_consistent": False}


@pytest.mark.parametrize("target", ["facts.json.gz", "viewer.html", "raw", "unlisted"])
def test_verify_detects_tampering(bundle: tuple[Path, Path], target: str) -> None:
    public, private = bundle
    assert archive.verify(public, private)["raw_verified"] is True
    path = private / "response-0001.bin" if target == "raw" else public / target
    path.write_bytes(path.read_bytes() + b"x" if path.exists() else b"x")
    with pytest.raises(ValueError):
        archive.verify(public, private)


def test_rehashed_viewer_must_still_match_facts(bundle: tuple[Path, Path]) -> None:
    public, _ = bundle
    path = public / "viewer.html"
    body = path.read_bytes().replace(b'"snapshot_id":"fixture"', b'"snapshot_id":"changed"')
    path.write_bytes(body)
    manifest = json.loads((public / "manifest.json").read_bytes())
    manifest["files"]["viewer.html"] = {"bytes": len(body), "sha256": archive.digest(body)}
    (public / "manifest.json").write_bytes(archive.encode(manifest))
    with pytest.raises(ValueError, match="disagree"):
        archive.verify(public)


@pytest.mark.smoke
def test_offline_cli_export_and_no_overwrite(bundle: tuple[Path, Path], tmp_path: Path) -> None:
    public, private = bundle
    output = tmp_path / "export.json"
    assert archive.main(["verify", str(public), "--private-root", str(private)]) == 0
    assert archive.main(["export", str(public), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["runs"]["pro"]["series"]["dynsam/avg@n"] == [0.5, 0.75]
    assert archive.main(["export", str(public), "--output", str(output)]) == 1
    with pytest.raises(ValueError, match="never overwrite"):
        archive.write_snapshot(public, private, {}, {})


def test_decompression_budget(bundle: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    public, _ = bundle
    monkeypatch.setattr(archive, "MAX_FACTS_BYTES", 10)
    with pytest.raises(ValueError, match="budget"):
        archive.read_facts(public / "facts.json.gz")
    assert gzip.decompress((public / "facts.json.gz").read_bytes()).endswith(b"\n")


def test_fake_capture_covers_all_tags_and_omits_original_text(tmp_path: Path) -> None:
    class FakeCollector(archive.Collector):
        def get(self, endpoint: str, *, parse: bool = True) -> object:
            if not parse:
                return b"original site code"
            path = urlsplit(endpoint).path
            params = parse_qs(urlsplit(endpoint).query)
            if path == "api/runs":
                return {"runs": [{"key": "pro", "label": "not redistributable text"}]}
            if path == "api/status":
                return status()
            if path == "api/tags":
                return {"run": "pro", "version": "v1", "tags": [f"tag/{i}" for i in range(100)]}
            if path == "api/series":
                return series() | {"series": {t: [0, None] for t in params["tags"][0].split(",")}}
            if path == "api/live":
                return {"log_time": 120, "latest": {"t": 120, "step": 2, "accept": 1,
                                                    "target": 2, "judged": None}, "entries": []}
            if path == "api/benchmarks":
                return {"benchmarks": [{"key": "deepswe", "title": "DeepSWE v1.1",
                                        "note": "mini-swe-agent, avg@3",
                                        "results": {"pro": {"1": 58}}}]}
            if path == "api/notices":
                return {"notices": [{"t": 110, "text": "private incident narrative"}]}
            raise AssertionError(endpoint)

    collector = FakeCollector(tmp_path, interval=0)
    facts = {"runs": {}, "benchmarks": []}
    archive.capture(collector, facts)
    assert collector.issues == []
    assert facts["runs"]["pro"]["coverage"]["received_tags"] == 100
    assert facts["runs"]["pro"]["series"]["tag/99"] == [0, None]
    text = json.dumps(facts)
    assert "not redistributable" not in text and "private incident narrative" not in text


@pytest.mark.parametrize("mode", ["timeout", "oversize", "html"])
def test_http_failures_are_bounded_and_not_zero_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    class Response:
        status = 200
        url = archive.BASE + "api/runs"
        headers: ClassVar = {
            "Content-Type": "text/html" if mode == "html" else "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, limit):
            return b"x" * limit if mode == "oversize" else b"{}"

    class Opener:
        def open(self, *_args, **_kwargs):
            if mode == "timeout":
                raise TimeoutError("private diagnostic")
            return Response()

    monkeypatch.setattr(archive, "MAX_BYTES", 10)
    collector = archive.Collector(tmp_path, interval=0, max_requests=1)
    collector.opener = Opener()
    assert collector.get("api/runs") is None
    assert len(collector.requests) == 1 and collector.requests[0]["ok"] is False
    assert "private diagnostic" not in json.dumps(collector.requests)
    with pytest.raises(ValueError, match="budget"):
        collector.get("api/runs")


def test_committed_snapshots_are_offline_verifiable() -> None:
    root = Path(__file__).resolve().parents[1] / "docs/evidence/mimo-rl/snapshots"
    snapshots = sorted(root.glob("*/manifest.json"))
    assert snapshots, "The teaching entry needs a committed baseline snapshot"
    statuses = []
    for manifest in snapshots:
        result = archive.verify(manifest.parent)
        statuses.append(result["status"])
        facts = archive.read_facts(manifest.parent / "facts.json.gz")
        for run in facts["runs"].values():
            assert all(archive.IDENTIFIER.fullmatch(tag) for tag in run["tags"])
    assert "complete" in statuses


@pytest.mark.parametrize("change", ["empty-version", "totals-text", "clock-object", "bad-key"])
def test_changed_status_fails_closed(change: str) -> None:
    data = status()
    if change == "empty-version":
        data["version"] = ""
    elif change == "totals-text":
        data["totals"]["tokens_cum"] = "not a number"
    elif change == "clock-object":
        data["clock"] = "schema drift"
    else:
        data["totals"]["private prose with spaces"] = 1
    with pytest.raises(ValueError):
        archive.project_status(data)


def test_unknown_live_and_benchmark_shapes_fail_closed() -> None:
    with pytest.raises(ValueError):
        archive.project_live({"log_time": 1, "latest": "broken", "entries": []})
    with pytest.raises(ValueError):
        archive.project_benchmark({"key": "deepswe", "title": "DeepSWE v1.1",
                                   "note": "mini-swe-agent, avg@3",
                                   "results": {"pro": {"arbitrary prose": 12}}}, {"pro": {}})


def test_total_network_failure_keeps_a_partial_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_capture(collector, facts):
        collector.issues.append("network unavailable")

    monkeypatch.setattr(archive, "capture", failed_capture)
    snapshot = archive.collect(tmp_path / "public", tmp_path / "private")
    result = archive.verify(snapshot)
    assert result["status"] == "partial" and result["coverage"] == {}


@pytest.mark.parametrize("limit", [{"max_seconds": 0}, {"max_total_bytes": 0}])
def test_global_capture_budget_stops_before_network(tmp_path: Path, limit: dict) -> None:
    collector = archive.Collector(tmp_path, **limit)
    with pytest.raises(ValueError, match="budget"):
        collector.get("api/runs")
    assert collector.requests == []


def test_public_private_roots_cannot_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        archive.collect(tmp_path, tmp_path / "private")


def test_workflow_uses_explicit_network_and_valid_python() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/mimo-rl-archive.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["jobs"]["archive"]["permissions"] == {"contents": "write"}
    for step in workflow["jobs"]["archive"]["steps"]:
        script = step.get("run", "")
        marker = "python - <<'PY'\n"
        if marker in script:
            ast.parse(script.split(marker, 1)[1].split("\nPY", 1)[0])