"""Print a safe, compact environment report for reproducible experiments."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from typing import Any

PACKAGES = ("numpy", "torch", "jax", "transformers", "vllm", "langchain", "llama-index-core")
SECRET_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_device_report() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        report["devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return report


def main() -> None:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: package_version(name) for name in PACKAGES},
        "torch": torch_device_report(),
        "credentials_configured": {name: bool(os.getenv(name)) for name in SECRET_NAMES},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Credential values are intentionally never printed.")


if __name__ == "__main__":
    main()
