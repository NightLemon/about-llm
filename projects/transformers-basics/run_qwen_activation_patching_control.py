from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.integrations.transformers_activation_patching_control import (
    run_target_activation_patching_control,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)

DEFAULT_CHECKPOINT_MANIFEST = (
    Path(__file__).with_name("target-checkpoints")
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the reviewed Qwen2.5-0.5B-Instruct CPU FP32 "
            "activation-patching control"
        )
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINT_MANIFEST,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require every selected checkpoint file to exist in the local HF cache",
    )
    args = parser.parse_args()
    spec = load_checkpoint_control_spec(args.checkpoint_manifest)
    report = run_target_activation_patching_control(
        spec, local_files_only=args.local_files_only
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
