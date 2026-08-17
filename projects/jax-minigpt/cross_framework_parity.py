"""Run the PyTorch/JAX MiniGPT numerical parity control."""

from __future__ import annotations

import json

from about_llm.from_scratch.gpt_cross_framework import (
    run_gpt_cross_framework_parity_control,
)


def main() -> None:
    print(
        json.dumps(
            run_gpt_cross_framework_parity_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
