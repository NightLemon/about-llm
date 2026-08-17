from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.rag.generation_policy import (
    build_publication_policy_replay_report,
    verify_publication_policy_replay_report,
)
from about_llm.rag.transformers_control import (
    load_rag_transformers_control_spec,
    verify_recorded_rag_transformers_report,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT / "qwen2.5-0.5b-rag.control.json"
DEFAULT_SOURCE_REPORT = PROJECT / "qwen2.5-0.5b-rag.recorded-report.json"
DEFAULT_REPLAY_REPORT = (
    PROJECT / "qwen2.5-0.5b-rag.publication-policy-replay.json"
)
DEFAULT_CHECKPOINT_MANIFEST = (
    PROJECT.parent
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the counterfactual fail-closed publication-policy "
            "replay over the separately verified Qwen RAG attempt"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINT_MANIFEST,
    )
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="REPLAY_REPORT",
        help=(
            "strictly verify an existing replay report; defaults are not inferred "
            "so build and verify remain explicit operations"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write a newly built report without overwriting an existing path",
    )
    args = parser.parse_args()
    if args.verify is not None and args.output is not None:
        parser.error("--verify and --output are mutually exclusive")

    checkpoint_spec = load_checkpoint_control_spec(args.checkpoint_manifest)
    spec = load_rag_transformers_control_spec(args.manifest)
    source_report = verify_recorded_rag_transformers_report(
        args.source_report,
        spec=spec,
        checkpoint_spec=checkpoint_spec,
    )
    if args.verify is not None:
        report = verify_publication_policy_replay_report(
            args.verify,
            spec=spec,
            source_report=source_report,
        )
    else:
        report = build_publication_policy_replay_report(
            spec=spec,
            source_report=source_report,
        )

    rendered = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered + "\n")
        except OSError as error:
            raise SystemExit(f"cannot create {args.output}: {error}") from error
    print(rendered)


if __name__ == "__main__":
    main()
