from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
    run_checkpoint_control,
)

DEFAULT_MANIFEST = (
    Path(__file__).with_name("target-checkpoints")
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and execute the reviewed Qwen2.5-0.5B-Instruct CPU FP32 "
            "checkpoint control"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require all reviewed files to already exist in the Hugging Face cache",
    )
    args = parser.parse_args()
    spec = load_checkpoint_control_spec(args.manifest)
    report = run_checkpoint_control(spec, local_files_only=args.local_files_only)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
