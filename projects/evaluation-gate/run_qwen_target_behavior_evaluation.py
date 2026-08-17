from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.evaluation.target_qwen_control import (
    load_target_qwen_evaluation_spec,
    run_target_qwen_evaluation_control,
    verify_recorded_target_qwen_evaluation_report,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
DEFAULT_SUITE = PROJECT / "target-qwen-behavior-suite.control.json"
DEFAULT_CHECKPOINT_MANIFEST = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser(
        description="Run or verify the fixed target-Qwen behavior evaluation"
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="verify a recorded report without loading model weights",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed checkpoint files in the local Hugging Face cache",
    )
    args = parser.parse_args()
    suite = load_target_qwen_evaluation_spec(args.suite)
    if args.verify is not None:
        report = verify_recorded_target_qwen_evaluation_report(args.verify, suite)
    else:
        checkpoint = load_checkpoint_control_spec(args.checkpoint_manifest)
        report = run_target_qwen_evaluation_control(
            checkpoint, suite, local_files_only=args.local_files_only
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
