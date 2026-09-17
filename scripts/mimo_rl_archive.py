"""Archive public MiMo RL measurements; network collection is explicit, verification offline.

Original response bodies stay in a private directory. The public projection contains
measurements and identifiers, not the original site's prose, images or application code.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = "https://mimo.xiaomi.com/rl/"
SCHEMA = "mimo-rl-archive/v1"
MAX_BYTES = 16 * 1024 * 1024
MAX_FACTS_BYTES = 128 * 1024 * 1024
TEMPLATE = Path(__file__).with_name("mimo_rl_viewer.html")
USER_AGENT = "about-llm-educational-archive/1.0 (bounded public dashboard snapshot)"
BENCHMARKS = {"deepswe": ("DeepSWE v1.1", "mini-swe-agent, avg@3")}
STATUS_KEYS = {"run", "cost", "clock", "version", "step", "totals", "events", "headline"}
LIVE_KEYS = {"log_time", "latest", "entries"}
IDENTIFIER = re.compile(r"[A-Za-z0-9_@./()=-]{1,512}")
LIVE_FIELDS = {
    "t", "step", "accept", "target", "judged", "judged_of", "ds", "passrate", "n",
    "pr0", "pr1", "partial", "working", "remain", "remain_partial", "remain_seq",
    "prewarm", "prewarm_wait",
}
_DROP = object()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def encode(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def load_json(body: bytes) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(body, parse_constant=reject)


def numeric_tree(value: Any, *, strict: bool = False) -> Any:
    """Keep null and numerical observations; omit unreviewed third-party prose."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError("non-finite measurement")
        return value
    if isinstance(value, list):
        return [None if (v := numeric_tree(item, strict=strict)) is _DROP else v for item in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) and IDENTIFIER.fullmatch(k) for k in value):
            raise ValueError("unreviewed measurement key")
        return {key: v for key, item in value.items()
                if (v := numeric_tree(item, strict=strict)) is not _DROP}
    if strict:
        raise ValueError("expected numerical measurement, not text")
    return _DROP


def project_status(data: dict[str, Any]) -> dict[str, Any]:
    required = {"version", "step", "totals", "clock", "run", "headline"}
    if (not required <= data.keys() or data.keys() - STATUS_KEYS
            or not isinstance(data["version"], str) or not data["version"]
            or not IDENTIFIER.fullmatch(data["version"])):
        raise ValueError("status schema changed")
    for name in ("run", "step", "totals", "clock", "headline", "cost"):
        if name not in data or not isinstance(data[name], dict):
            raise ValueError("status section type changed")
    if not {"last"} <= data["step"].keys() or "now" not in data["clock"]:
        raise ValueError("missing status progress/clock")
    for name in ("totals", "clock", "cost"):
        if not all(v is None or (type(v) in (int, float) and math.isfinite(v))
               for v in data[name].values()):
            raise ValueError("status scalar type changed")
    for name, text_fields in (("run", {"key", "label", "mode"}),
                              ("step", {"phase"}), ("headline", {"tag"})):
        for field, value in data[name].items():
            if field in text_fields:
                if not isinstance(value, str):
                    raise ValueError("status identifier type changed")
            else:
                numeric_tree(value, strict=True)
    if not isinstance(data.get("events"), list):
        raise ValueError("status events type changed")
    for event in data["events"]:
        if not isinstance(event, dict) or event.get("kind") not in {"restart", "step"}:
            raise ValueError("status event schema changed")
        numeric_tree({k: v for k, v in event.items() if k != "kind"}, strict=True)
    return {"version": data["version"], "observations": numeric_tree(data)}


def project_live(data: dict[str, Any]) -> dict[str, Any]:
    if data.keys() != LIVE_KEYS or not isinstance(data["entries"], list):
        raise ValueError("live schema changed")
    records = data["entries"] + ([data["latest"]] if data["latest"] is not None else [])
    for record in records:
        if (not isinstance(record, dict) or record.keys() - LIVE_FIELDS
                or not {"t", "step", "accept", "target"} <= record.keys()):
            raise ValueError("live record schema changed")
    return numeric_tree(data, strict=True)


def project_benchmark(item: dict[str, Any], runs: dict[str, Any]) -> dict[str, Any]:
    protocol = BENCHMARKS.get(item.get("key"))
    results = item.get("results")
    if (protocol is None or not isinstance(results, dict) or set(results) - runs.keys()
            or (item.get("title"), item.get("note")) != protocol):
        raise ValueError("benchmark schema/protocol changed")
    for scores in results.values():
        if not isinstance(scores, dict):
            raise ValueError("benchmark results schema changed")
        for step, score in scores.items():
            if (not re.fullmatch(r"[0-9]{1,12}", step) or
                    (score is not None and (type(score) not in (int, float)
                                           or not math.isfinite(score)))):
                raise ValueError("benchmark checkpoint/score changed")
    return {"key": item["key"], "title": protocol[0], "note": protocol[1], "results": results}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise ValueError("redirect refused; review source endpoint")


class Collector:
    def __init__(self, private: Path, *, interval: float = 0.25,
                 timeout: float = 20, max_requests: int = 256,
                 max_total_bytes: int = 64 * 1024 * 1024, max_seconds: float = 600) -> None:
        self.private = private
        self.interval = interval
        self.timeout = timeout
        self.max_requests = max_requests
        self.max_total_bytes = max_total_bytes
        self.total_bytes = 0
        self.deadline = time.monotonic() + max_seconds
        self.requests: list[dict[str, Any]] = []
        self.issues: list[str] = []
        self.opener = urllib.request.build_opener(NoRedirect())
        self.last_request = 0.0

    def get(self, endpoint: str, *, parse: bool = True) -> Any:
        if (len(self.requests) >= self.max_requests or time.monotonic() >= self.deadline
                or self.total_bytes >= self.max_total_bytes):
            raise ValueError("request budget exhausted")
        if endpoint.startswith(("/", ".")) or ":" in endpoint.split("?")[0]:
            raise ValueError("invalid relative endpoint")
        delay = self.interval - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)  # Rate limit actual requests, never poll a running job.
        url = BASE + endpoint
        record: dict[str, Any] = {"url": url, "started_at": utc_now(), "ok": False}
        self.requests.append(record)
        self.last_request = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT, "Accept-Encoding": "identity",
                "Accept": "application/json" if parse else "*/*",
            })
            try:
                response = self.opener.open(
                    request, timeout=min(self.timeout, max(0.1, self.deadline - time.monotonic())))
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                record.update(status=response.status, final_url=response.url,
                              content_type=response.headers.get("Content-Type"),
                              etag=response.headers.get("ETag"),
                              last_modified=response.headers.get("Last-Modified"),
                              server_date=response.headers.get("Date"))
                limit = min(MAX_BYTES, self.max_total_bytes - self.total_bytes)
                body = response.read(limit + 1)
            self.total_bytes += len(body)
            if len(body) > limit:
                raise ValueError("response exceeds byte budget")
            name = f"response-{len(self.requests):04d}.bin"
            (self.private / name).write_bytes(body)
            record.update(raw_file=name, bytes=len(body), sha256=digest(body))
            if record["status"] != 200:
                raise ValueError(f"HTTP {record['status']}; no automatic retry")
            if parse and "json" not in (record["content_type"] or "").lower():
                raise ValueError("expected JSON Content-Type")
            result = load_json(body) if parse else body
            if parse and not isinstance(result, dict):
                raise ValueError("expected a JSON object")
            record["ok"] = True
            return result
        except (OSError, ValueError) as exc:
            # Do not publish upstream error prose or local exception paths.
            record["error"] = type(exc).__name__
            self.issues.append(f"request {len(self.requests)}: {record['error']}")
            return None
        finally:
            record["finished_at"] = utc_now()


def query(endpoint: str, **params: str) -> str:
    return "api/" + endpoint + "?" + urllib.parse.urlencode(params)


def merge_series(run: dict[str, Any], chunk: list[str], data: dict[str, Any]) -> None:
    """Only combine responses with the same run, version and axes."""
    if data.get("run") != run["key"] or data.get("version") != run["version"]:
        raise ValueError("series run/version changed during collection")
    run_start = data.get("run_start")
    if type(run_start) not in (int, float) or not math.isfinite(run_start):
        raise ValueError("invalid run start")
    if "run_start" in run and run["run_start"] != run_start:
        raise ValueError("run start changed during collection")
    steps, walls, series = data.get("steps"), data.get("walls"), data.get("series")
    if not isinstance(steps, list) or not isinstance(walls, list) or len(steps) != len(walls):
        raise ValueError("series axes malformed")
    if not all(type(s) is int for s in steps):
        raise ValueError("non-integer step")
    if not all(type(w) in (int, float) and math.isfinite(w) for w in walls):
        raise ValueError("invalid wall time")
    if not isinstance(series, dict) or set(series) - set(chunk):
        raise ValueError("unexpected series tags/schema")
    if "steps" in run and (run["steps"] != steps or run["walls"] != walls):
        raise ValueError("series axes changed during collection")
    checked = {}
    for tag in chunk:
        values = series.get(tag)
        if values is None:
            continue
        if not isinstance(values, list) or len(values) != len(steps):
            raise ValueError(f"series length mismatch: {tag}")
        if not all(v is None or (type(v) in (int, float) and math.isfinite(v)) for v in values):
            raise ValueError(f"invalid series values: {tag}")
        checked[tag] = values
    run.update(steps=steps, walls=walls, run_start=data.get("run_start"))
    run["series"].update(checked)


def coverage(run: dict[str, Any]) -> dict[str, Any]:
    tags, series = run["tags"], run["series"]
    versions = [run.get("version"), run.get("status_before", {}).get("version"),
                run.get("status_after", {}).get("version")]
    missing = sorted(set(tags) - series.keys())
    empty = sorted(t for t, values in series.items() if not any(v is not None for v in values))
    return {"expected_tags": len(tags), "received_tags": len(series), "missing_tags": missing,
            "empty_tags": empty, "steps": len(run.get("steps", [])),
            "version_consistent": None not in versions and len(set(versions)) == 1}


def capture(collector: Collector, facts: dict[str, Any]) -> None:
    config = collector.get("api/runs")
    if not config or not isinstance(config.get("runs"), list) or not 1 <= len(config["runs"]) <= 8:
        raise ValueError("runs catalogue unavailable or changed")
    for item in config["runs"]:
        key = item.get("key", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", key) or key in facts["runs"]:
            raise ValueError("invalid or duplicate run identifier")
        run: dict[str, Any] = {"key": key, "tags": [], "series": {}}
        facts["runs"][key] = run
        before = collector.get(query("status", run=key))
        if before:
            if before.get("run", {}).get("key") != key:
                raise ValueError("status run identity changed")
            run["status_before"] = project_status(before)
        version = before.get("version", "") if before else ""
        tags_doc = collector.get(query("tags", run=key, v=version))
        if tags_doc:
            tags = tags_doc.get("tags")
            if (not isinstance(tags, list) or not 1 <= len(tags) <= 10000
                    or not all(isinstance(t, str) and IDENTIFIER.fullmatch(t) for t in tags)
                    or len(set(tags)) != len(tags) or tags_doc.get("run") != key
                    or not isinstance(tags_doc.get("version"), str)
                    or not IDENTIFIER.fullmatch(tags_doc["version"])):
                raise ValueError("tag catalogue schema changed")
            run.update(tags=tags, version=tags_doc["version"])
            # Below the site's own 96-tag batch. Sequential, bounded requests.
            for start in range(0, len(tags), 48):
                chunk = tags[start:start + 48]
                endpoint = query("series", run=key, v=run["version"], tags=",".join(chunk))
                data = collector.get(endpoint)
                if data is not None:
                    try:
                        merge_series(run, chunk, data)
                    except ValueError as exc:
                        collector.issues.append(f"{key}: {exc}")
        live = collector.get(query("live", run=key))
        if live is not None:
            run["live"] = project_live(live)
        after = collector.get(query("status", run=key))
        if after:
            if after.get("run", {}).get("key") != key:
                raise ValueError("status run identity changed")
            run["status_after"] = project_status(after)
        run["coverage"] = coverage(run)
        if run["coverage"]["missing_tags"] or not run["coverage"]["version_consistent"]:
            collector.issues.append(f"{key}: incomplete catalogue or version changed")
    benchmarks = collector.get("api/benchmarks")
    if benchmarks is not None:
        if not isinstance(benchmarks.get("benchmarks"), list):
            collector.issues.append("benchmark schema changed")
        else:
            for item in benchmarks["benchmarks"]:
                facts["benchmarks"].append(project_benchmark(item, facts["runs"]))
    notices = collector.get("api/notices")
    if notices is not None:
        if not isinstance(notices.get("notices"), list):
            collector.issues.append("notices schema changed")
        else:
            for item in notices["notices"]:
                if not isinstance(item, dict) or item.keys() - {"id", "t", "text", "run"}:
                    raise ValueError("notice schema changed")
                numeric_tree(item.get("t"), strict=True)
            facts["notices"] = {"count": len(notices["notices"]),
                                "observations": [{"t": item.get("t")}
                                                 for item in notices["notices"]],
                                "text_policy": "original prose retained in private raw only"}
    # Preserve the observed entry page and its known first-party assets privately.
    # No recursive crawl, fonts, third-party resources or active page execution.
    for asset in ("", "style.css", "favicon.svg", "js/theme.js", "js/format.js",
                  "js/charts.js", "js/app.js"):
        collector.get(asset, parse=False)


def viewer(facts: dict[str, Any], manifest: dict[str, Any]) -> bytes:
    payload = json.dumps({"facts": facts, "manifest": manifest}, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return TEMPLATE.read_text(encoding="utf-8").replace("__ARCHIVE_DATA__", payload).encode("utf-8")


def write_snapshot(output: Path, private: Path, facts: dict[str, Any],
                   manifest: dict[str, Any]) -> None:
    if any(output.iterdir()) or (private / "manifest.json").exists():
        raise ValueError("snapshot already exists; never overwrite historical files")
    facts_body = gzip.compress(encode(facts), mtime=0)
    # Embed metadata before file hashes: avoids circular viewer/manifest hashing.
    html = viewer(facts, manifest)
    files = {"facts.json.gz": facts_body, "viewer.html": html}
    manifest["files"] = {name: {"bytes": len(body), "sha256": digest(body)}
                         for name, body in files.items()}
    for name, body in files.items():
        (output / name).write_bytes(body)
    body = encode(manifest)
    (output / "manifest.json").write_bytes(body)  # Commit marker written last.
    (private / "manifest.json").write_bytes(body)


def collect(output_root: Path, private_root: Path) -> Path:
    public_path, private_path = output_root.resolve(), private_root.resolve()
    if (public_path == private_path or public_path in private_path.parents
            or private_path in public_path.parents):
        raise ValueError("public and private roots must not overlap")
    started = utc_now()
    snapshot_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    output, private = output_root / snapshot_id, private_root / snapshot_id
    output.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    collector = Collector(private)
    facts: dict[str, Any] = {"schema": SCHEMA, "source": BASE, "runs": {}, "benchmarks": []}
    try:
        capture(collector, facts)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        collector.issues.append(f"schema/collection failure: {type(exc).__name__}")
        (private / "collection-error.txt").write_text(str(exc), encoding="utf-8")
    for run in facts["runs"].values():
        run["coverage"] = coverage(run)
    manifest = {
        "schema": SCHEMA, "snapshot_id": snapshot_id, "source": BASE,
        "started_at": started, "finished_at": utc_now(),
        "status": "partial" if collector.issues else "complete",
        "issues": collector.issues, "requests": collector.requests,
        "collector_sha256": digest(Path(__file__).read_bytes()),
        "viewer_template_sha256": digest(TEMPLATE.read_bytes()),
        "limits": {"requests": 256, "response_bytes": MAX_BYTES,
               "total_response_bytes": collector.max_total_bytes,
               "collection_seconds": 600, "retries_per_request": 0},
        "python": sys.version.split()[0], "user_agent": USER_AGENT,
        "policy": {"public": "numerical observations, metric identifiers, original viewer",
                   "private": "original response bodies, site code, notices and descriptions",
                   "redistribution": "third-party materials excluded; capture grants no rights",
                   "scope": "all tags discoverable now; no guarantee of deleted or hidden history",
                   "time": "per-response capture window; server clocks are untrusted observations",
                   "integrity": "SHA-256 binds bytes, not authenticity or training truth"},
        "coverage": {key: run["coverage"] for key, run in facts["runs"].items()},
    }
    write_snapshot(output, private, facts, manifest)
    return output


def read_facts(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        body = stream.read(MAX_FACTS_BYTES + 1)
    if len(body) > MAX_FACTS_BYTES:
        raise ValueError("facts exceed decompression budget")
    facts = load_json(body)
    if not isinstance(facts, dict) or facts.get("schema") != SCHEMA:
        raise ValueError("unknown facts schema")
    return facts


def verify(snapshot: Path, private_root: Path | None = None) -> dict[str, Any]:
    manifest = load_json((snapshot / "manifest.json").read_bytes())
    if manifest.get("schema") != SCHEMA or manifest.get("status") not in {"complete", "partial"}:
        raise ValueError("unknown manifest schema/status")
    expected_files = {"facts.json.gz", "viewer.html"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("unexpected archive inventory")
    if {p.name for p in snapshot.iterdir()} != expected_files | {"manifest.json"}:
        raise ValueError("missing or unlisted archive files")
    for name, expected in manifest["files"].items():
        path = snapshot / name
        if path.is_symlink():
            raise ValueError("archive symlink refused")
        body = path.read_bytes()
        if {"bytes": len(body), "sha256": digest(body)} != expected:
            raise ValueError(f"integrity mismatch: {name}")
    facts = read_facts(snapshot / "facts.json.gz")
    if facts.get("source") != BASE or manifest.get("source") != BASE:
        raise ValueError("source mismatch")
    calculated = {}
    for key, run in facts["runs"].items():
        if set(run) - {"key", "tags", "series", "status_before", "status_after", "version",
                   "steps", "walls", "run_start", "live", "coverage"}:
            raise ValueError("unreviewed public run fields")
        if run["key"] != key or len(set(run["tags"])) != len(run["tags"]):
            raise ValueError("invalid run/catalogue")
        if set(run["series"]) - set(run["tags"]):
            raise ValueError("unlisted metric")
        if run["series"]:
            check = {"key": key, "version": run["version"], "series": {}}
            merge_series(check, list(run["series"]), {
                "run": key, "version": run["version"], "steps": run["steps"],
                "walls": run["walls"], "series": run["series"], "run_start": run["run_start"],
            })
        calculated[key] = coverage(run)
        if run.get("coverage") != calculated[key]:
            raise ValueError("run coverage mismatch")
    if calculated != manifest["coverage"]:
        raise ValueError("manifest coverage mismatch")
    for benchmark in facts["benchmarks"]:
        if (set(benchmark) != {"key", "title", "note", "results"}
            or project_benchmark(benchmark, facts["runs"]) != benchmark):
            raise ValueError("unreviewed benchmark protocol")
    # Viewer is self-contained and must embed exactly the verified facts and metadata.
    html = (snapshot / "viewer.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="archive-data" type="application/json">(.*?)</script>',
                      html, re.DOTALL)
    embedded = load_json(match.group(1).encode("utf-8")) if match else None
    embedded_manifest = {k: v for k, v in manifest.items() if k != "files"}
    if embedded != {"facts": facts, "manifest": embedded_manifest}:
        raise ValueError("viewer and facts disagree")
    if manifest["status"] == "complete" and (
        not calculated or manifest["issues"]
        or any(not r["ok"] for r in manifest["requests"])
        or any(c["missing_tags"] or not c["version_consistent"] for c in calculated.values())
    ):
        raise ValueError("incomplete capture advertised as complete")
    for record in manifest["requests"]:
        if not record["url"].startswith(BASE):
            raise ValueError("unexpected endpoint")
        start = datetime.fromisoformat(record["started_at"])
        end = datetime.fromisoformat(record["finished_at"])
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("invalid request time window")
        if private_root is not None and "raw_file" in record:
            name = record["raw_file"]
            if not re.fullmatch(r"response-\d{4}\.bin", name):
                raise ValueError("unsafe raw path")
            path = private_root / name
            if path.is_symlink():
                raise ValueError("raw symlink refused")
            body = path.read_bytes()
            if len(body) != record["bytes"] or digest(body) != record["sha256"]:
                raise ValueError(f"raw integrity mismatch: {name}")
    return {"snapshot_id": manifest["snapshot_id"], "status": manifest["status"],
            "coverage": calculated, "raw_verified": private_root is not None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("collect", help="explicit bounded network capture")
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--private-output", type=Path, required=True)
    verify_parser = sub.add_parser("verify", help="offline byte and coverage checks")
    verify_parser.add_argument("snapshot", type=Path)
    verify_parser.add_argument("--private-root", type=Path)
    export_parser = sub.add_parser("export", help="verify then decompress numerical facts")
    export_parser.add_argument("snapshot", type=Path)
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            path = collect(args.output, args.private_output)
            result = verify(path, args.private_output / path.name)
            print(json.dumps({"path": str(path), **result}, ensure_ascii=True))
            return 0 if result["status"] == "complete" else 2
        result = verify(args.snapshot, getattr(args, "private_root", None))
        if args.command == "export":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                stream.write(encode(read_facts(args.snapshot / "facts.json.gz")))
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Archive error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())