from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.rag.guarded_transformers_control import (
    load_guarded_rag_transformers_control_spec,
    run_guarded_rag_transformers_control,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT / "qwen2.5-0.5b-rag.guarded.control.json"
DEFAULT_CHECKPOINT_MANIFEST = (
    PROJECT.parent
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Run the fail-closed policy around one real Qwen GenerationMixin.generate "
            "invocation while suppressing generation for empty authorized evidence"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed checkpoint files to exist in the local cache",
    )
    args = parser.parse_args()
    checkpoint_spec = load_checkpoint_control_spec(args.checkpoint_manifest)
    spec = load_guarded_rag_transformers_control_spec(args.manifest)
    report = run_guarded_rag_transformers_control(
        spec,
        checkpoint_spec=checkpoint_spec,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
