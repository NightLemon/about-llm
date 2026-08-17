from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.evaluation import EvaluationCase, load_cases, run_evaluation, write_results
from about_llm.evaluation.text_metrics import (
    literal_exact_match,
    normalized_exact_match,
    token_f1,
)


def test_load_run_and_write_evaluation(tmp_path: Path) -> None:
    case_path = tmp_path / "cases.jsonl"
    case_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "q1",
                        "input": "capital",
                        "expected": "Beijing",
                        "slices": ["fact"],
                    }
                ),
                json.dumps(
                    {
                        "case_id": "q2",
                        "input": "中文",
                        "expected": "检索增强生成",
                        "slices": ["zh"],
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(case_path)
    answers = {"capital": "  BEIJING ", "中文": "检索增强"}
    results = run_evaluation(
        cases,
        answers.__getitem__,
        {
            "literal_exact_match": literal_exact_match,
            "exact_match": normalized_exact_match,
            "token_f1": token_f1,
        },
    )
    result_path = tmp_path / "nested" / "results.jsonl"
    write_results(result_path, results)

    assert results[0].scores == {
        "literal_exact_match": 0.0,
        "exact_match": 1.0,
        "token_f1": 1.0,
    }
    assert results[1].scores["literal_exact_match"] == 0.0
    assert results[1].scores["exact_match"] == 0.0
    assert 0 < results[1].scores["token_f1"] < 1
    assert len(result_path.read_text(encoding="utf-8").splitlines()) == 2
    assert not result_path.with_suffix(".jsonl.tmp").exists()


def test_runner_records_system_failures_without_losing_case() -> None:
    cases = [EvaluationCase("broken", "input", "expected")]

    def broken_system(_: str) -> str:
        raise TimeoutError("deadline")

    results = run_evaluation(cases, broken_system, {"exact": normalized_exact_match})
    assert results[0].case_id == "broken"
    assert results[0].scores == {}
    assert results[0].error == "TimeoutError: deadline"


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    value = json.dumps({"case_id": "same", "input": "x", "expected": "y"})
    path.write_text(value + "\n" + value, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_cases(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            '{"case_id":"q","case_id":"changed","input":"x","expected":"y"}',
            "duplicate JSON object key",
        ),
        (
            '{"case_id":"q","input":"x","expected":"y","metadata":{"x":NaN}}',
            "non-standard JSON constant",
        ),
        (
            '{"case_id":"q","input":"x","expected":"y","typo":true}',
            r"unknown=\['typo'\]",
        ),
    ],
)
def test_case_loader_rejects_nonstandard_or_drifting_json(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(payload + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cases(path)
