import json

import pytest

from about_llm.evaluation import (
    EvaluationCase,
    citation_evidence_span_metric,
    citation_syntax_metric,
    json_schema_metric,
    json_value_exact_metric,
)

pytestmark = pytest.mark.contract


def test_json_schema_metric_checks_parse_types_and_required_fields() -> None:
    case = EvaluationCase(
        "json-1",
        "return an object",
        "",
        metadata={
            "output_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        },
    )

    assert json_schema_metric(case, '{"answer": "ok"}') == 1
    assert json_schema_metric(case, '{"answer": 7}') == 0
    assert json_schema_metric(case, '{"answer": 7, "answer": "ok"}') == 0
    assert json_schema_metric(case, "not json") == 0


@pytest.mark.parametrize("output", ['{"value":NaN}', '{"value":Infinity}'])
def test_json_schema_metric_rejects_nonstandard_constants(output: str) -> None:
    case = EvaluationCase(
        "strict-json",
        "return strict JSON",
        "",
        metadata={"output_schema": {}},
    )

    assert json_schema_metric(case, output) == 0


def test_json_schema_metric_allows_local_refs_and_rejects_nonlocal_resolution() -> None:
    local_case = EvaluationCase(
        "local-ref",
        "return an integer answer",
        "",
        metadata={
            "output_schema": {
                "$defs": {"answer": {"type": "integer"}},
                "type": "object",
                "properties": {"answer": {"$ref": "#/$defs/answer"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        },
    )
    remote_case = EvaluationCase(
        "remote-ref",
        "return anything",
        "",
        metadata={"output_schema": {"$ref": "https://example.invalid/schema.json"}},
    )

    assert json_schema_metric(local_case, '{"answer":42}') == 1
    with pytest.raises(ValueError, match="local fragment reference"):
        json_schema_metric(remote_case, "{}")


def test_json_schema_metric_treats_invalid_schema_as_case_error() -> None:
    case = EvaluationCase(
        "invalid-schema",
        "return anything",
        "",
        metadata={"output_schema": {"type": "not-a-json-schema-type"}},
    )

    with pytest.raises(ValueError, match="invalid output_schema"):
        json_schema_metric(case, "{}")


def test_json_value_exact_ignores_object_order_and_json_whitespace_only() -> None:
    case = EvaluationCase(
        "json-value",
        "return the object",
        '{"answer":42,"ok":true,"items":[1,2]}',
    )

    assert (
        json_value_exact_metric(
            case,
            '{ "items": [1, 2], "ok": true, "answer": 42 }',
        )
        == 1
    )
    assert json_value_exact_metric(case, '{"answer":43,"ok":true,"items":[1,2]}') == 0
    assert json_value_exact_metric(case, '{"answer":42,"ok":true,"items":[2,1]}') == 0
    assert (
        json_value_exact_metric(
            case,
            '{"answer":0,"answer":42,"ok":true,"items":[1,2]}',
        )
        == 0
    )


def test_json_value_exact_preserves_scalar_type_and_numeric_parse_class() -> None:
    integer_case = EvaluationCase("integer", "return one", "1")

    assert json_value_exact_metric(integer_case, "1") == 1
    assert json_value_exact_metric(integer_case, "true") == 0
    assert json_value_exact_metric(integer_case, "1.0") == 0
    assert json_value_exact_metric(integer_case, "NaN") == 0


def test_json_value_exact_rejects_invalid_expected_as_case_error() -> None:
    case = EvaluationCase("bad-gold", "return JSON", '{"answer":1,"answer":2}')

    with pytest.raises(ValueError, match="expected must be strict JSON"):
        json_value_exact_metric(case, '{"answer":2}')


def test_citation_evidence_span_accepts_exact_authorized_unicode_binding() -> None:
    case = EvaluationCase(
        "span-valid",
        "return claims with exact evidence spans",
        "",
        metadata={"citation_sources": {"S1": "地球围绕太阳运行。"}},
    )
    output = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "地球围绕太阳运行。",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "start_char": 0,
                            "end_char": 9,
                            "quote": "地球围绕太阳运行。",
                        }
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )

    assert citation_evidence_span_metric(case, output) == 1


@pytest.mark.parametrize(
    "span",
    [
        {"source_id": "S9", "start_char": 0, "end_char": 3, "quote": "abc"},
        {"source_id": "S1", "start_char": 0, "end_char": 3, "quote": "bcd"},
        {"source_id": "S1", "start_char": True, "end_char": 3, "quote": "abc"},
        {"source_id": "S1", "start_char": 0, "end_char": 7, "quote": "abcdef"},
    ],
)
def test_citation_evidence_span_rejects_unbound_spans(
    span: dict[str, object],
) -> None:
    case = EvaluationCase(
        "span-invalid",
        "return claims with exact evidence spans",
        "",
        metadata={"citation_sources": {"S1": "abcdef"}},
    )
    output = json.dumps(
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "a claim",
                    "evidence": [span],
                }
            ]
        }
    )

    assert citation_evidence_span_metric(case, output) == 0


def test_citation_evidence_span_is_strict_but_does_not_infer_entailment() -> None:
    case = EvaluationCase(
        "span-boundary",
        "return claims with exact evidence spans",
        "",
        metadata={"citation_sources": {"S1": "Earth is round."}},
    )
    unrelated_claim = (
        '{"claims":[{"claim_id":"c1","text":"The moon is cheese.",'
        '"evidence":[{"source_id":"S1","start_char":0,"end_char":5,'
        '"quote":"Earth"}]}]}'
    )
    duplicate_key = (
        '{"claims":[{"claim_id":"c1","claim_id":"c2",'
        '"text":"Earth is round.","evidence":[{"source_id":"S1",'
        '"start_char":0,"end_char":5,"quote":"Earth"}]}]}'
    )

    assert citation_evidence_span_metric(case, unrelated_claim) == 1
    assert citation_evidence_span_metric(case, duplicate_key) == 0


def _exact_span() -> dict[str, object]:
    return {"source_id": "S1", "start_char": 0, "end_char": 3, "quote": "abc"}


@pytest.mark.parametrize(
    "payload",
    [
        {"claims": []},
        {"claims": [], "extra": True},
        {
            "claims": [
                {"claim_id": "c1", "text": "claim", "evidence": [], "extra": 1}
            ]
        },
        {"claims": [{"claim_id": "c1", "text": "claim", "evidence": []}]},
        {
            "claims": [
                {"claim_id": "c1", "text": "", "evidence": [_exact_span()]}
            ]
        },
        {
            "claims": [
                {"claim_id": "c1", "text": "first", "evidence": [_exact_span()]},
                {"claim_id": "c1", "text": "second", "evidence": [_exact_span()]},
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "claim",
                    "evidence": [_exact_span(), _exact_span()],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "claim",
                    "evidence": [{**_exact_span(), "extra": True}],
                }
            ]
        },
    ],
)
def test_citation_evidence_span_rejects_open_or_ambiguous_structures(
    payload: dict[str, object],
) -> None:
    case = EvaluationCase(
        "span-structure",
        "return claims with exact evidence spans",
        "",
        metadata={"citation_sources": {"S1": "abcdef"}},
    )

    assert citation_evidence_span_metric(case, json.dumps(payload)) == 0


def test_citation_evidence_span_rejects_invalid_case_configuration() -> None:
    missing = EvaluationCase("missing", "return spans", "")
    wrong_text = EvaluationCase(
        "wrong-text",
        "return spans",
        "",
        metadata={"citation_sources": {"S1": 7}},
    )

    with pytest.raises(ValueError, match="citation_sources"):
        citation_evidence_span_metric(missing, "{}")
    with pytest.raises(ValueError, match="string text"):
        citation_evidence_span_metric(wrong_text, "{}")


def test_citation_metric_combines_known_id_precision_and_coverage() -> None:
    case = EvaluationCase(
        "citation-1",
        "answer with sources",
        "",
        metadata={"valid_source_ids": ["S1", "S2"]},
    )

    assert citation_syntax_metric(case, "Claim. [S1]") == 1
    assert citation_syntax_metric(case, "Claim. [S9]") == 0.5
    assert citation_syntax_metric(case, "Claim without citation.") == 0
