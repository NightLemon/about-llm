"""Offline CLI for strict SFT JSONL and transparent leakage auditing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from about_llm.finetuning.data import (
    DataSplit,
    audit_sft_records,
    load_sft_records,
    validate_training_subset,
)
from about_llm.finetuning.governance import (
    audit_sft_governance,
    load_sft_governance_policy,
    parse_utc_timestamp,
)
from about_llm.finetuning.near_duplicate import (
    NearDuplicateProfile,
    NearDuplicateView,
    audit_sft_near_duplicates,
)
from about_llm.finetuning.readiness import SFTTrainingReadinessReport


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
    report = audit_sft_records(
        load_sft_records(args.jsonl), required_splits=args.require_splits
    )
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if report.gate_passed else 1


def _run_near_audit(args: argparse.Namespace) -> int:
    records = load_sft_records(args.jsonl)
    exact_report = audit_sft_records(records, required_splits=args.require_splits)
    selected_views = (
        tuple(NearDuplicateView(value) for value in args.view)
        if args.view
        else tuple(NearDuplicateView)
    )
    near_report = audit_sft_near_duplicates(
        records,
        profile=NearDuplicateProfile(args.profile),
        ngram_size=args.ngram_size,
        threshold=args.threshold,
        views=selected_views,
        cross_split_only=not args.include_within_split,
    )
    passed = exact_report.gate_passed and near_report.gate_passed
    payload = {
        "gate_passed": passed,
        "exact_audit": exact_report.to_dict(),
        "near_duplicate_audit": near_report.to_dict(),
        "evidence_boundary": (
            "A finding is a lexical character-ngram candidate under the declared "
            "lossy normalization profile and threshold. It is not a verified semantic "
            "duplicate; no finding is not proof that split contamination is absent."
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if passed else 1


def _run_governance_audit(args: argparse.Namespace) -> int:
    records = load_sft_records(args.jsonl)
    policy = load_sft_governance_policy(args.policy)
    report = audit_sft_governance(
        records,
        policy=policy,
        evaluated_at=args.evaluated_at,
    )
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if report.gate_passed else 1


def _run_prepare_training(args: argparse.Namespace) -> int:
    """Audit with held-out access and emit a plaintext-free trainer envelope."""

    training_records = load_sft_records(args.train_jsonl)
    combined_records = load_sft_records(args.audit_jsonl)
    binding = validate_training_subset(training_records, combined_records)
    selected_views = (
        tuple(NearDuplicateView(value) for value in args.view)
        if args.view
        else tuple(NearDuplicateView)
    )
    near_report = audit_sft_near_duplicates(
        combined_records,
        profile=NearDuplicateProfile(args.profile),
        ngram_size=args.ngram_size,
        threshold=args.threshold,
        views=selected_views,
        cross_split_only=not args.include_within_split,
    )
    governance_report = audit_sft_governance(
        combined_records,
        policy=load_sft_governance_policy(args.governance_policy),
        evaluated_at=args.governance_evaluated_at,
    )
    readiness = SFTTrainingReadinessReport.from_reports(
        binding, near_report, governance_report
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "sft-data-audit.json": binding.training_report.to_dict(),
        "sft-split-audit.json": binding.split_report.to_dict(),
        "sft-data-binding.json": binding.to_dict(),
        "sft-near-duplicate-audit.json": near_report.to_dict(),
        "sft-governance-audit.json": governance_report.to_dict(),
        "sft-training-readiness.json": readiness.to_dict(),
    }
    for name, payload in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2))
    return 0 if readiness.gate_passed else 1


def _add_common_audit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument(
        "--require-splits",
        type=_split_list,
        default=(DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST),
        help="Comma-separated required splits (default: train,validation,test)",
    )
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-sft-data",
        description=(
            "Validate strict SFT data and audit exact/group/lexical split leakage"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Audit one JSONL artifact")
    _add_common_audit_arguments(audit)
    audit.set_defaults(handler=_run_audit)
    near = subparsers.add_parser(
        "near-audit",
        help="Run exact gates plus exhaustive lexical character-ngram candidates",
    )
    _add_common_audit_arguments(near)
    near.add_argument(
        "--profile",
        choices=[profile.value for profile in NearDuplicateProfile],
        required=True,
        help="Lossy normalization profile; choose deliberately for the task",
    )
    near.add_argument("--ngram-size", type=int, default=5)
    near.add_argument("--threshold", type=float, default=0.85)
    near.add_argument(
        "--view",
        action="append",
        choices=[view.value for view in NearDuplicateView],
        help="Repeat to select views; default checks full/user/assistant independently",
    )
    near.add_argument(
        "--include-within-split",
        action="store_true",
        help="Compare same-split pairs too; default is cross-split leakage only",
    )
    near.set_defaults(handler=_run_near_audit)
    governance = subparsers.add_parser(
        "governance-audit",
        help="Apply exact source policy and limited sensitive-content candidates",
    )
    governance.add_argument("--jsonl", type=Path, required=True)
    governance.add_argument("--policy", type=Path, required=True)
    governance.add_argument(
        "--evaluated-at",
        type=_utc_timestamp,
        required=True,
        help="Explicit UTC decision time, YYYY-MM-DDTHH:MM:SSZ",
    )
    governance.add_argument("--output", type=Path)
    governance.set_defaults(handler=_run_governance_audit)
    prepare = subparsers.add_parser(
        "prepare-training",
        help="Audit train+combined data and emit a held-out-plaintext-free readiness artifact",
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
        choices=[view.value for view in NearDuplicateView],
    )
    prepare.add_argument("--include-within-split", action="store_true")
    prepare.set_defaults(handler=_run_prepare_training)
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
