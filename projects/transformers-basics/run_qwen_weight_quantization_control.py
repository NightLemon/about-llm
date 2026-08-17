from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.integrations.transformers_weight_quantization_control import (
    run_target_weight_quantization_control,
    verify_recorded_target_weight_quantization_report,
)

TARGET_DIRECTORY = Path(__file__).with_name("target-checkpoints")
DEFAULT_MANIFEST = TARGET_DIRECTORY / "qwen2.5-0.5b-instruct.control.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one selected-matrix packed INT4 control on the reviewed Qwen "
            "CPU FP32 checkpoint, or verify a recorded closed report"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed snapshot to already exist in the local cache",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="verify a recorded report without loading the target checkpoint",
    )
    args = parser.parse_args()
    spec = load_checkpoint_control_spec(args.manifest)
    if args.verify is not None:
        report = verify_recorded_target_weight_quantization_report(
            args.verify,
            expected_manifest_fingerprint=spec.manifest_fingerprint,
        )
    else:
        report = run_target_weight_quantization_control(
            spec,
            local_files_only=args.local_files_only,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
