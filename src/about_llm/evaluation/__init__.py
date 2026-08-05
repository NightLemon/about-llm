"""Evaluation primitives shared by projects."""

from about_llm.evaluation.reporting import (
    SliceSummary,
    render_markdown_report,
    summarize_by_slice,
)
from about_llm.evaluation.retrieval import (
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    recall_at_k,
)
from about_llm.evaluation.runner import (
    EvaluationCase,
    EvaluationResult,
    load_cases,
    run_evaluation,
    write_results,
)
from about_llm.evaluation.statistics import (
    PairedBootstrapResult,
    ReleaseGate,
    paired_bootstrap,
)
from about_llm.evaluation.structured import citation_syntax_metric, json_schema_metric

__all__ = [
    "EvaluationCase",
    "EvaluationResult",
    "PairedBootstrapResult",
    "ReleaseGate",
    "SliceSummary",
    "citation_syntax_metric",
    "json_schema_metric",
    "load_cases",
    "mean_reciprocal_rank",
    "normalized_discounted_cumulative_gain",
    "paired_bootstrap",
    "recall_at_k",
    "render_markdown_report",
    "run_evaluation",
    "summarize_by_slice",
    "write_results",
]
