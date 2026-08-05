"""Deterministic overall and slice-level evaluation summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from about_llm.evaluation.runner import EvaluationCase, EvaluationResult


@dataclass(frozen=True)
class SliceSummary:
    name: str
    case_count: int
    error_rate: float
    mean_scores: Mapping[str, float]


def summarize_by_slice(
    cases: Iterable[EvaluationCase], results: Iterable[EvaluationResult]
) -> tuple[SliceSummary, ...]:
    """Summarize overall plus declared slices and reject incomplete joins."""
    case_list, result_list = list(cases), list(results)
    case_by_id = {case.case_id: case for case in case_list}
    result_by_id = {result.case_id: result for result in result_list}
    if len(case_by_id) != len(case_list) or len(result_by_id) != len(result_list):
        raise ValueError("duplicate case_id in cases or results")
    if case_by_id.keys() != result_by_id.keys():
        missing_results = sorted(case_by_id.keys() - result_by_id.keys())
        unknown_results = sorted(result_by_id.keys() - case_by_id.keys())
        raise ValueError(
            f"case/result mismatch: missing_results={missing_results}, "
            f"unknown_results={unknown_results}"
        )

    groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    for case_id, case in case_by_id.items():
        groups["overall"].append(result_by_id[case_id])
        for slice_name in case.slices:
            groups[slice_name].append(result_by_id[case_id])

    summaries: list[SliceSummary] = []
    for name in ["overall", *sorted(set(groups) - {"overall"})]:
        group = groups[name]
        metric_names = sorted({metric for result in group for metric in result.scores})
        means = {
            metric: sum(result.scores[metric] for result in group if metric in result.scores)
            / sum(metric in result.scores for result in group)
            for metric in metric_names
        }
        summaries.append(
            SliceSummary(
                name=name,
                case_count=len(group),
                error_rate=sum(result.error is not None for result in group) / len(group),
                mean_scores=means,
            )
        )
    return tuple(summaries)


def render_markdown_report(summaries: Iterable[SliceSummary]) -> str:
    summary_list = list(summaries)
    metrics = sorted({metric for summary in summary_list for metric in summary.mean_scores})
    header = ["Slice", "Cases", "Error rate", *metrics]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for summary in summary_list:
        cells = [summary.name, str(summary.case_count), f"{summary.error_rate:.3f}"]
        cells.extend(
            f"{summary.mean_scores[metric]:.3f}" if metric in summary.mean_scores else "—"
            for metric in metrics
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
