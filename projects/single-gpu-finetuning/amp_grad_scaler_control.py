"""Run the real CPU AMP/GradScaler sequencing and resume control."""

from __future__ import annotations

import json
import sys

from about_llm.finetuning.amp_scaler import run_cpu_amp_grad_scaler_control


def main() -> None:
    payload = run_cpu_amp_grad_scaler_control().to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.buffer.write((encoded + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
