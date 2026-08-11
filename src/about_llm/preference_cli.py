"""Offline CLI for strict pairwise-preference JSONL auditing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from about_llm.finetuning.data import DataSplit
from about_llm.finetuning.governance import (
    load_sft_governance_policy,
    parse_utc_timestamp,
)
from about_llm.finetuning.near_duplicate import NearDuplicateProfile
from about_llm.finetuning.preference_data import (
    audit_preference_records,
    load_preference_records,
    validate_dpo_training_subset,
)
from about_llm.finetuning.preference_evaluation import (
    audit_preference_judgments,
    load_preference_judgments,
    summarize_preference_judgments,
)
from about_llm.finetuning.preference_governance import (
    audit_preference_governance,
)
from about_llm.finetuning.preference_near_duplicate import (
    PreferenceNearDuplicateView,
    audit_preference_near_duplicates,
)
from about_llm.finetuning.preference_readiness import (
    PreferenceTrainingReadinessReport,
)


def _split_list(value: str) -> tuple[DataSplit, ...]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("require-splits must not be empty")
    try:
        splits = tuple(DataSplit(name) for name in names)
    except ValueError as error:
        choices = ", ".join(split.value for split in DataSplit)
        raise argparse.ArgumentTypeError(
            f"unknown split; choose comma-separated values from: {choices}"
        ) from error
    if len(splits) != len(set(splits)):
        raise argparse.ArgumentTypeError("require-splits must not contain duplicates")
    return splits


def _utc_timestamp(value: str) -> datetime:
    try:
        return parse_utc_timestamp(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _run_audit(args: argparse.Namespace) -> int:
    report = audit_preference_records(
        load_preference_records(args.jsonl), required_splits=args.require_splits
    )
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if report.gate_passed else 1


def _run_prepare_training(args: argparse.Namespace) -> int:
    """Bind train-only pairs without passing held-out plaintext to the trainer."""

    training = load_preference_records(args.train_jsonl)
    combined = load_preference_records(args.audit_jsonl)
    binding = validate_dpo_training_subset(training, combined)
    selected_views = (
        tuple(PreferenceNearDuplicateView(value) for value in args.view)
        if args.view
        else tuple(PreferenceNearDuplicateView)
    )
    near_report = audit_preference_near_duplicates(
        combined,
        profile=NearDuplicateProfile(args.profile),
        ngram_size=args.ngram_size,
        threshold=args.threshold,
        views=selected_views,
        cross_split_only=not args.include_within_split,
    )
    governance_report = audit_preference_governance(
        combined,
        policy=load_sft_governance_policy(args.governance_policy),
        evaluated_at=args.governance_evaluated_at,
    )
    readiness = PreferenceTrainingReadinessReport.from_reports(
        binding, near_report, governance_report
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "preference-train-audit.json": binding.training_report.to_dict(),
        "preference-split-audit.json": binding.split_report.to_dict(),
        "preference-data-binding.json": binding.to_dict(),
        "preference-near-duplicate-audit.json": near_report.to_dict(),
        "preference-governance-audit.json": governance_report.to_dict(),
        "preference-training-readiness.json": readiness.to_dict(),
    }
    for name, payload in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2))
    return 0 if readiness.gate_passed else 1


def _run_evaluate_judgments(args: argparse.Namespace) -> int:
    if not 0 < args.confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    cases = load_preference_records(args.cases_jsonl)
    judgments = load_preference_judgments(args.judgments_jsonl)
    audit = audit_preference_judgments(
        cases,
        judgments,
        selected_splits=args.case_splits,
        judgments_per_pair=args.judgments_per_pair,
        minimum_judgments_per_order=args.minimum_per_order,
    )
    evaluation = (
        summarize_preference_judgments(
            judgments,
            audit,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ).to_dict()
        if audit.gate_passed
        else None
    )
    payload = {
        "gate_passed": audit.gate_passed,
        "audit": audit.to_dict(),
        "evaluation": evaluation,
        "evidence_boundary": (
            "A passing artifact proves declared case binding, coverage, order, "
            "rubric, blind/independent flags, and reproducible descriptive statistics. "
            "It does not prove the annotators are human, identities are pseudonymous, "
            "assignment was randomized, blindness was operationally enforced, the "
            "rubric is valid, or the observed position effect is causal."
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if audit.gate_passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-preference-data",
        description="Validate pairwise preference annotation and split identities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--jsonl", type=Path, required=True)
    audit.add_argument(
        "--require-splits",
        type=_split_list,
        default=(DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST),
    )
    audit.add_argument("--output", type=Path)
    audit.set_defaults(handler=_run_audit)
    prepare = subparsers.add_parser(
        "prepare-training",
        help="Bind binary train pairs to a passing combined split audit",
    )
    prepare.add_argument("--train-jsonl", type=Path, required=True)
    prepare.add_argument("--audit-jsonl", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--governance-policy", type=Path, required=True)
    prepare.add_argument(
        "--governance-evaluated-at",
        type=_utc_timestamp,
        required=True,
        help="Explicit UTC decision time, YYYY-MM-DDTHH:MM:SSZ",
    )
    prepare.add_argument(
        "--profile",
        choices=[profile.value for profile in NearDuplicateProfile],
        required=True,
    )
    prepare.add_argument("--ngram-size", type=int, default=5)
    prepare.add_argument("--threshold", type=float, default=0.85)
    prepare.add_argument(
        "--view",
        action="append",
        choices=[view.value for view in PreferenceNearDuplicateView],
    )
    prepare.add_argument("--include-within-split", action="store_true")
    prepare.set_defaults(handler=_run_prepare_training)
    evaluate = subparsers.add_parser(
        "evaluate-judgments",
        help="Audit raw pairwise judgments and report agreement/position diagnostics",
    )
    evaluate.add_argument("--cases-jsonl", type=Path, required=True)
    evaluate.add_argument("--judgments-jsonl", type=Path, required=True)
    evaluate.add_argument(
        "--case-splits",
        type=_split_list,
        default=(DataSplit.VALIDATION, DataSplit.TEST),
    )
    evaluate.add_argument("--judgments-per-pair", type=int, required=True)
    evaluate.add_argument("--minimum-per-order", type=int, required=True)
    evaluate.add_argument("--confidence", type=float, default=0.95)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=0)
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=_run_evaluate_judgments)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    try:
        return handler(args)
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
