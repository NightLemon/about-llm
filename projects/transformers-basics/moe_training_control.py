"""Run the deterministic trainable top-k MoE gradient control."""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_training import run_trainable_moe_control


def main() -> None:
    report = run_trainable_moe_control()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
