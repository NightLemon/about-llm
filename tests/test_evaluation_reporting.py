import pytest

from about_llm.evaluation import (
    EvaluationCase,
    EvaluationResult,
    render_markdown_report,
    summarize_by_slice,
)

pytestmark = pytest.mark.formula


def test_slice_summary_keeps_overall_and_exposes_regressions() -> None:
    cases = [
        EvaluationCase("a", "A", "", slices=("zh", "critical")),
        EvaluationCase("b", "B", "", slices=("en",)),
    ]
    results = [
        EvaluationResult("a", "", {"quality": 0.4}, 0.1),
        EvaluationResult("b", "", {"quality": 1.0}, 0.2),
    ]
    summaries = summarize_by_slice(cases, results)

    assert [summary.name for summary in summaries] == ["overall", "critical", "en", "zh"]
    assert summaries[0].mean_scores["quality"] == pytest.approx(0.7)
    report = render_markdown_report(summaries)
    assert "| critical | 1 | 0.000 | 0.400 |" in report


def test_slice_summary_rejects_incomplete_case_result_join() -> None:
    with pytest.raises(ValueError, match="missing_results"):
        summarize_by_slice([EvaluationCase("a", "A", "")], [])
