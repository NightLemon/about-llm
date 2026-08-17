from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from about_llm.finetuning.target_dpo_control import (
    load_target_dpo_control_spec,
    run_target_dpo_control,
    verify_recorded_target_dpo_report,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)

PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parents[1]
DEFAULT_CONTROL = PROJECT / "qwen2.5-0.5b-dpo.control.json"
DEFAULT_CHECKPOINT_CONTROL = (
    REPOSITORY
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
DEFAULT_CHECKPOINT_REPORT = DEFAULT_CHECKPOINT_CONTROL.with_name(
    "qwen2.5-0.5b-instruct.recorded-report.json"
)
DEFAULT_TRAINING = PROJECT / "preference.train.example.jsonl"
DEFAULT_READINESS = PROJECT / "preference-training-readiness.example.json"


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or verify the reviewed Qwen2.5-0.5B CPU FP32 TRL DPO control"
    )
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument(
        "--checkpoint-control", type=Path, default=DEFAULT_CHECKPOINT_CONTROL
    )
    parser.add_argument(
        "--checkpoint-report", type=Path, default=DEFAULT_CHECKPOINT_REPORT
    )
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed checkpoint files to exist in the local Hugging Face cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_spec = load_checkpoint_control_spec(args.checkpoint_control)
    spec = load_target_dpo_control_spec(
        args.control, checkpoint_spec=checkpoint_spec
    )
    if args.verify is not None:
        report = verify_recorded_target_dpo_report(
            args.verify,
            spec=spec,
            checkpoint_spec=checkpoint_spec,
            checkpoint_report_path=args.checkpoint_report,
            training_path=args.training,
            readiness_path=args.readiness,
        )
    else:
        report = run_target_dpo_control(
            spec,
            checkpoint_spec=checkpoint_spec,
            checkpoint_report_path=args.checkpoint_report,
            training_path=args.training,
            readiness_path=args.readiness,
            local_files_only=args.local_files_only,
        )
        if args.output_report is not None:
            _write_json_new(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
