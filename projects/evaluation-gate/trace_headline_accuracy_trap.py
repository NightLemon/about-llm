"""Explain why a 24/30 candidate can still fail a release gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from about_llm.evaluation.headline_accuracy_trace import build_headline_accuracy_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable trace instead of the guided view",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    report = build_headline_accuracy_trace()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_guided_view(report))
    return 0


def _guided_view(report: dict[str, Any]) -> str:
    headline = report["headline"]
    changes = report["paired_changes"]
    comparison = report["comparison"]
    quality = comparison["quality"]
    critical = comparison["protected_slices"]["cross_tenant"]
    lines = [
        "先看总分",
        f"  Baseline: {headline['baseline_correct']} / 30 "
        f"({headline['baseline_accuracy']:.1%})",
        f"  Candidate: {headline['candidate_correct']} / 30 "
        f"({headline['candidate_accuracy']:.1%})",
        "  只看这一行。Candidate 多答对了 2 题。",
        "",
        "再看真正发生变化的 6 条 case",
    ]
    for row in changes["rows"]:
        direction = "改善" if row["change"] == "improved" else "退化"
        lines.append(
            f"  {row['case_id']} [{row['slice']}] {direction}: "
            f"{row['baseline_output']} → {row['candidate_output']}"
        )
    lines.extend(
        [
            "",
            "最后做发布判断",
            f"  总体差值: {quality['mean_difference']:+.3f}; "
            f"95% paired bootstrap interval "
            f"[{quality['confidence_low']:+.3f}, {quality['confidence_high']:+.3f}]",
            "  跨租户拒绝: Baseline 4/5 → Candidate 2/5",
            f"  该切片差值: {critical['mean_difference']:+.3f}; "
            f"interval [{critical['confidence_low']:+.3f}, "
            f"{critical['confidence_high']:+.3f}]",
            "",
            "发布决定: 拦截 Candidate",
            "  原因一: 30 条样例还不足以确认总体提升大于零。",
            "  原因二: 跨租户拒绝退化。该区间下界低于门槛。",
            "",
            "这个实验使用仓库准备的输出。它不会调用真实模型。",
            "它教你读取配对差异和切片。这些 30 条样例并不代表真实流量。",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
