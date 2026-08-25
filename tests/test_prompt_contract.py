from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from about_llm.prompt_contract import validate_contract_extraction

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH = (
    ROOT / "projects" / "cloud-api-contracts" / "prompt_contract_walkthrough.py"
)
SOURCE = (
    "甲方为海云科技有限公司。本合同由双方于 2026 年 8 月 1 日签署，"  # noqa: RUF001
    "结算币种见附件 A。"
)


def _span(source: str, field: str, quote: str, occurrence: int = 0) -> dict[str, object]:
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(quote, start + 1)
    return {
        "field": field,
        "quote": quote,
        "start_char": start,
        "end_char": start + len(quote),
    }


def _output(**changes: object) -> str:
    value: dict[str, object] = {
        "status": "insufficient_evidence",
        "party": "海云科技有限公司",
        "signed_on": "2026-08-01",
        "currency": None,
        "evidence": [
            _span(SOURCE, "party", "海云科技有限公司"),
            _span(SOURCE, "signed_on", "2026 年 8 月 1 日"),
        ],
    }
    value.update(changes)
    return json.dumps(value, ensure_ascii=False)


def test_missing_currency_evidence_rejects_fabricated_value() -> None:
    report = validate_contract_extraction(SOURCE, _output(status="complete", currency="CNY"))

    assert report.decision == "reject"
    assert report.exact_spans_valid is True
    assert report.field_semantics_valid is False
    assert "semantic:evidence_count:currency" in report.errors


def test_exact_span_does_not_make_an_unsupported_quote_valid() -> None:
    evidence = [
        _span(SOURCE, "party", "海云科技有限公司"),
        _span(SOURCE, "signed_on", "2026 年 8 月 1 日"),
        _span(SOURCE, "currency", "结算币种见附件 A"),
    ]

    report = validate_contract_extraction(
        SOURCE,
        _output(status="complete", currency="CNY", evidence=evidence),
    )

    assert report.exact_spans_valid is True
    assert report.field_semantics_valid is False
    assert report.errors == ("semantic:unsupported_value:currency",)


def test_duplicate_json_key_is_rejected_before_shape_validation() -> None:
    raw_output = _output().replace(
        '"status": "insufficient_evidence"',
        '"status": "complete", "status": "insufficient_evidence"',
        1,
    )

    report = validate_contract_extraction(SOURCE, raw_output)

    assert report.strict_json_valid is False
    assert report.decision == "reject"
    assert "duplicate object key 'status'" in report.errors[0]


def test_corrected_insufficient_evidence_output_is_accepted() -> None:
    report = validate_contract_extraction(SOURCE, _output())

    assert report.decision == "accept"
    assert report.errors == ()


def test_conflict_keeps_two_distinct_candidate_spans() -> None:
    source = "正文写结算币种为 CNY；附件写结算币种为 USD。"  # noqa: RUF001
    output = {
        "status": "conflict",
        "party": None,
        "signed_on": None,
        "currency": None,
        "evidence": [
            _span(source, "currency", "CNY"),
            _span(source, "currency", "USD"),
        ],
    }

    report = validate_contract_extraction(
        source,
        json.dumps(output, ensure_ascii=False),
    )

    assert report.decision == "accept"
    assert report.exact_spans_valid is True


def test_repeated_same_value_is_not_a_conflict() -> None:
    source = "正文写 CNY，附件也写 CNY。"  # noqa: RUF001
    output = {
        "status": "conflict",
        "party": None,
        "signed_on": None,
        "currency": None,
        "evidence": [
            _span(source, "currency", "CNY", occurrence=0),
            _span(source, "currency", "CNY", occurrence=1),
        ],
    }

    report = validate_contract_extraction(
        source,
        json.dumps(output, ensure_ascii=False),
    )

    assert report.decision == "reject"
    assert "semantic:conflict_requires_distinct_candidates" in report.errors


def test_invalid_calendar_date_is_rejected() -> None:
    report = validate_contract_extraction(
        SOURCE,
        _output(signed_on="2026-13-40"),
    )

    assert report.closed_shape_valid is False
    assert "shape:signed_on_format" in report.errors


@pytest.mark.smoke
def test_documented_walkthrough_runs_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(WALKTHROUGH)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert report["implementation"] == "about-llm.prompt-contract-walkthrough.v1"
    assert report["cases"]["corrected_insufficient_evidence"]["decision"] == "accept"
    assert report["scope"]["language_model_or_provider_executed"] is False
