"""Run the strict cross-process JAX/Optax checkpoint-resume control."""

from __future__ import annotations

import json

from about_llm.from_scratch.jax_training_resume import (
    run_jax_training_resume_control,
)


def main() -> None:
    print(
        json.dumps(
            run_jax_training_resume_control(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
