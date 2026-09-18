"""Frozen figure/data contracts and lesson arithmetic, not training-truth evidence."""

from __future__ import annotations

import gzip
import json
import math
import struct
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/evidence/mimo-rl/figures/2026-09-17-series"


@pytest.mark.contract
def test_teaching_figures_have_bounded_metadata_and_reader_references() -> None:
    index = json.loads((FIGURES / "figure-index.json").read_text(encoding="utf-8"))
    assert set(index) == {"schema", "figures"}
    assert index["schema"] == 1
    names = [row["file"] for row in index["figures"]]
    assert len(names) == len(set(names)) == 21
    assert set(names) == {path.name for path in FIGURES.glob("*.png")}
    lessons = list((ROOT / "docs/models").glob("mimo-rl*.md"))
    assert len(lessons) == 12  # Eleven lessons plus the series overview.
    text = "\n".join(path.read_text(encoding="utf-8") for path in lessons)
    for row in index["figures"]:
        assert set(row) == {"file", "source", "started_at", "finished_at"}
        assert Path(row["file"]).name == row["file"]
        assert row["source"].startswith("https://mimo.xiaomi.com/rl/#")
        start = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(row["finished_at"].replace("Z", "+00:00"))
        assert start <= end
        assert start.date().isoformat() == "2026-09-17"
        png = (FIGURES / row["file"]).read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", png[16:24])
        assert width >= 460 and height >= 130
        assert "2026-09-17-series/" + row["file"] in text


@pytest.mark.contract
def test_frozen_curves_and_benchmarks_keep_their_own_steps() -> None:
    facts = json.loads((FIGURES / "selected-series.json").read_text(encoding="utf-8"))
    observations = json.loads((FIGURES / "observations.json").read_text(encoding="utf-8"))
    assert set(facts) == {"runs", "requestedAt", "completedAt"}
    assert facts["requestedAt"] <= facts["completedAt"]
    assert set(facts["runs"]) == {"pro", "flash"}
    versions = {"pro": "3-5513.14.6.11", "flash": "3-5513.19.3.16"}
    expected = {"pro": (14, 11, 65.78), "flash": (17, 12, 60.77)}
    for run, (train_step, eval_step, score) in expected.items():
        data = facts["runs"][run]
        assert set(data) == {"version", "steps", "walls", "series"}
        assert data["version"] == versions[run]
        assert data["steps"] == list(range(1, train_step + 1))
        assert len(data["walls"]) == train_step
        assert len(data["series"]) == 30
        for values in [data["walls"], *data["series"].values()]:
            assert len(values) == train_step
            assert all(v is None or (type(v) in (int, float) and math.isfinite(v)) for v in values)
        results = observations["benchmarks"]["results"][run]
        assert max(map(int, results)) == eval_step
        assert str(train_step) not in results  # No fill from a different checkpoint.
        assert results[str(eval_step)] == score
    for value in observations.values():
        if isinstance(value, dict) and "figure" in value:
            assert (FIGURES / value["figure"]).is_file()


@pytest.mark.formula
def test_lesson_arithmetic_uses_frozen_units_and_denominators() -> None:
    facts = json.loads((FIGURES / "selected-series.json").read_text(encoding="utf-8"),
                       parse_float=Decimal, parse_int=Decimal)
    observed = json.loads((FIGURES / "observations.json").read_text(encoding="utf-8"),
                          parse_float=Decimal, parse_int=Decimal)
    status = observed["status"]
    nominal_batch = status["target_prompts"] * status["sequences_per_prompt"]
    assert nominal_batch == 25088
    assert nominal_batch * status["reported_step"] == 351232
    assert nominal_batch != observed["harness"]["rollouts"]
    categories = observed["composition"]["categories"]
    assert sum(row["prompts"] for row in categories.values()) == 1568
    assert sum(row["sources"] for row in categories.values()) == 25
    for row in categories.values():
        share = (row["prompts"] / 1568 * 100).quantize(Decimal("0.1"))
        assert share == row["display_share_percent"]
    for run, rounded_rate in (("pro", 252578), ("flash", 332808)):
        series = facts["runs"][run]["series"]
        rate = series["perf/total_num_tokens"][-1] / series["timing_s/step"][-1]
        assert rate.quantize(Decimal("1")) == rounded_rate
    pro = facts["runs"]["pro"]["series"]
    mixed = (1 - pro["dynsam/passrate/zero"][-1] - pro["dynsam/passrate/one"][-1]) * 100
    assert mixed.quantize(Decimal("0.1")) == Decimal("59.3")
    # Lesson explicitly uses rounded figure labels for its approximately 59.4%.
    assert 100 - Decimal("15.0") - Decimal("25.6") == Decimal("59.4")
    benchmark = observed["benchmarks"]["results"]
    assert benchmark["pro"]["11"] - benchmark["pro"]["1"] == Decimal("7.37")
    assert benchmark["flash"]["12"] - benchmark["flash"]["11"] == Decimal("6.64")


@pytest.mark.contract
def test_supervised_review_keeps_figures_and_later_evaluation_separate() -> None:
    """Check frozen input relationships, not the publisher's training/score claims."""
    review = FIGURES.parent / "2026-09-18-review"
    snapshot = FIGURES.parents[1] / "snapshots/2026-09-18T06-07-28-636152Z"
    index = json.loads((review / "figure-index.json").read_text(encoding="utf-8"))
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    facts = json.loads(gzip.decompress((snapshot / "facts.json.gz").read_bytes()))
    follow = json.loads((review / "benchmark-followup.json").read_text(encoding="utf-8"))
    text = (FIGURES.parents[1] / "reviews/2026-09-18.md").read_text(encoding="utf-8")

    def timestamp(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    assert set(index) == {"figures"}
    names = {row["file"] for row in index["figures"]}
    assert len(index["figures"]) == len(names) == 4
    assert names == {path.name for path in review.glob("*.png")}
    for row in index["figures"]:
        assert set(row) == {"file", "source", "started_at", "finished_at"}
        assert Path(row["file"]).name == row["file"]
        assert row["source"].startswith("https://mimo.xiaomi.com/rl/#")
        assert timestamp(manifest["finished_at"]) < timestamp(row["started_at"])
        assert timestamp(row["started_at"]) <= timestamp(row["finished_at"])
        assert timestamp(row["finished_at"]) < timestamp(follow["started_at"])
        png = (review / row["file"]).read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", png[16:24])
        assert width >= 700 and height >= 250
        assert "2026-09-18-review/" + row["file"] in text
    assert set(follow) == {"source", "started_at", "finished_at", "benchmarks"}
    assert follow["source"] == "https://mimo.xiaomi.com/rl/api/benchmarks"
    assert timestamp(follow["started_at"]) <= timestamp(follow["finished_at"])
    older = {b["key"]: b["results"] for b in facts["benchmarks"]}
    newer = {b["key"]: b["results"] for b in follow["benchmarks"]}
    assert older.keys() == newer.keys() == {"deepswe", "inhouse-coding", "automation"}
    for key, results in older.items():
        for run, scores in results.items():
            assert all(newer[key][run][step] == value for step, value in scores.items())
    assert max(map(int, older["deepswe"]["flash"])) == 16
    assert max(map(int, newer["deepswe"]["flash"])) == 18
    assert newer["deepswe"]["flash"]["18"] == 64.01
    assert newer["deepswe"]["pro"]["14"] == 64.60
    assert facts["runs"]["pro"]["steps"][-1] == 17
    assert facts["runs"]["flash"]["steps"][-1] == 24


@pytest.mark.contract
def test_later_protocol_record_does_not_replace_earlier_checkpoint_scores() -> None:
    """Two supplied observations demonstrate a revision, not its cause."""
    snapshots = FIGURES.parents[1] / "snapshots"
    records = []
    for identity in ("2026-09-18T06-07-28-636152Z", "2026-09-18T09-19-30-038024Z"):
        data = json.loads(gzip.decompress((snapshots / identity / "facts.json.gz").read_bytes()))
        records.append(next(b for b in data["benchmarks"] if b["key"] == "automation"))
    earlier, later = records
    assert earlier["note"] == ""
    assert later["note"] == "avg@3"
    assert earlier["results"]["pro"]["12"] == 48.7
    assert later["results"]["pro"]["12"] == 48.9
    assert earlier["results"]["flash"]["16"] == 49.3
    assert later["results"]["flash"]["16"] == 49.4
    figures = FIGURES.parent / "2026-09-18-publish-review"
    index = json.loads((figures / "figure-index.json").read_text(encoding="utf-8"))
    assert len(index["figures"]) == 1
    row = index["figures"][0]
    assert set(row) == {"file", "source", "started_at", "finished_at"}
    assert row["file"] == "benchmarks.png"
    assert (figures / row["file"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    text = (FIGURES.parents[1] / "reviews/2026-09-18.md").read_text(encoding="utf-8")
    assert "2026-09-18-publish-review/" + row["file"] in text