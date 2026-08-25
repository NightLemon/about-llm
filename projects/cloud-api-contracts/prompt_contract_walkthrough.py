"""Trace prepared contract-extraction outputs through independent validation layers."""

from __future__ import annotations

import json
from typing import Any

from about_llm.prompt_contract import validate_contract_extraction

DOCUMENT_ID = "contract-017"
SOURCE_TEXT = (
    "甲方为海云科技有限公司。本合同由双方于 2026 年 8 月 1 日签署，"  # noqa: RUF001
    "结算币种见附件 A。"
)


def _evidence(field: str, quote: str) -> dict[str, object]:
    start = SOURCE_TEXT.index(quote)
    return {
        "field": field,
        "quote": quote,
        "start_char": start,
        "end_char": start + len(quote),
    }


def _base_output() -> dict[str, Any]:
    return {
        "status": "complete",
        "party": "海云科技有限公司",
        "signed_on": "2026-08-01",
        "currency": "CNY",
        "evidence": [
            _evidence("party", "海云科技有限公司"),
            _evidence("signed_on", "2026 年 8 月 1 日"),
        ],
    }


def _prepared_outputs() -> dict[str, str]:
    fabricated = _base_output()

    unsupported_quote = _base_output()
    unsupported_quote["evidence"] = [
        *unsupported_quote["evidence"],
        _evidence("currency", "结算币种见附件 A"),
    ]

    missing_status = _base_output()
    missing_status.pop("status")

    corrected = _base_output()
    corrected["status"] = "insufficient_evidence"
    corrected["currency"] = None

    return {
        "fabricated_currency_without_evidence": json.dumps(
            fabricated, ensure_ascii=False
        ),
        "fabricated_currency_with_exact_but_unsupported_quote": json.dumps(
            unsupported_quote,
            ensure_ascii=False,
        ),
        "missing_required_status": json.dumps(
            missing_status,
            ensure_ascii=False,
        ),
        "duplicate_status_key": (
            '{"status":"complete","status":"insufficient_evidence",'
            '"party":"海云科技有限公司","signed_on":"2026-08-01",'
            '"currency":null,"evidence":[]}'
        ),
        "corrected_insufficient_evidence": json.dumps(
            corrected,
            ensure_ascii=False,
        ),
    }


def run_walkthrough() -> dict[str, object]:
    expected = {
        "fabricated_currency_without_evidence": "reject",
        "fabricated_currency_with_exact_but_unsupported_quote": "reject",
        "missing_required_status": "repair_or_reject",
        "duplicate_status_key": "reject",
        "corrected_insufficient_evidence": "accept",
    }
    cases: dict[str, dict[str, object]] = {}
    for name, raw_output in _prepared_outputs().items():
        result = validate_contract_extraction(SOURCE_TEXT, raw_output)
        if result.decision != expected[name]:
            raise AssertionError(
                f"{name} decision changed: expected {expected[name]!r}, "
                f"got {result.decision!r}"
            )
        cases[name] = result.to_dict()

    return {
        "implementation": "about-llm.prompt-contract-walkthrough.v1",
        "input": {
            "document_id": DOCUMENT_ID,
            "source_characters": len(SOURCE_TEXT),
            "raw_source_saved_in_report": False,
        },
        "cases": cases,
        "scope": {
            "prepared_outputs_used": True,
            "language_model_or_provider_executed": False,
            "strict_json_and_closed_shape_checked": True,
            "exact_character_spans_checked": True,
            "narrow_field_semantics_checked": True,
            "general_legal_semantics_or_model_quality_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_walkthrough(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
