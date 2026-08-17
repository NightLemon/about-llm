"""Run the two-process CPU/Gloo MoE capacity-group control."""

from __future__ import annotations

import json

from about_llm.from_scratch.moe_distributed_capacity import (
    run_distributed_moe_capacity_control,
)


def main() -> None:
    print(
        json.dumps(
            run_distributed_moe_capacity_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
