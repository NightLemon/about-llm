"""Run the two-process CPU/Gloo MoE all-to-all training control."""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_all_to_all_training import (
    run_moe_all_to_all_training_control,
)


def main() -> None:
    print(
        json.dumps(
            run_moe_all_to_all_training_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
