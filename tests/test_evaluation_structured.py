from about_llm.evaluation import EvaluationCase, citation_syntax_metric, json_schema_metric


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
    assert json_schema_metric(case, "not json") == 0


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
