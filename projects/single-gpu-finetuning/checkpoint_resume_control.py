"""Run the cross-process CPU AMP training-checkpoint resume control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from about_llm.finetuning.training_resume import (
    WorkerMode,
    run_training_resume_process_control,
    run_training_resume_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-mode",
        choices=(
            "baseline",
            "phase1",
            "resume",
            "omit-scheduler",
            "omit-scaler",
            "omit-rng",
            "omit-data",
            "wrong-scheduler",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkpoint-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_mode is None:
        if args.checkpoint_path is not None:
            raise SystemExit("--checkpoint-path is reserved for internal workers")
        payload = run_training_resume_process_control(Path(__file__))
    else:
        if args.checkpoint_path is None:
            raise SystemExit("internal worker requires --checkpoint-path")
        payload = run_training_resume_worker(
            cast(WorkerMode, args.worker_mode), args.checkpoint_path
        )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.buffer.write((encoded + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
