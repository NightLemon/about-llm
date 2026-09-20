"""Print a safe environment report and optionally validate a learning profile."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 13)

REPORT_PACKAGES = (
    "about-llm",
    "numpy",
    "torch",
    "jax",
    "transformers",
    "vllm",
    "langchain",
    "llama-index-core",
    "mcp",
    "mkdocs",
    "mkdocs-material",
    "nbclient",
    "nbformat",
    "ipykernel",
)
PROFILE_REQUIREMENTS = {
    "docs": ("about-llm", "Markdown", "mkdocs", "mkdocs-material"),
    "cpu-starter": ("about-llm", "numpy"),
    "notebooks": (
        "about-llm",
        "numpy",
        "torch",
        "jax",
        "nbclient",
        "nbformat",
        "ipykernel",
    ),
    "full-ci": (
        "about-llm",
        "Markdown",
        "mkdocs",
        "mkdocs-material",
        "build",
        "pytest",
        "pytest-cov",
        "ruff",
        "mypy",
        "nbclient",
        "nbformat",
        "ipykernel",
        "numpy",
        "torch",
        "jax",
        "flax",
        "optax",
        "transformers",
        "datasets",
        "accelerate",
        "sentencepiece",
        "peft",
        "trl",
        "safetensors",
        "sentence-transformers",
        "rank-bm25",
        "langchain",
        "langchain-community",
        "llama-index-core",
        "a2a-sdk",
        "mcp",
        "httpx",
        "pydantic",
        "python-dotenv",
        "fastapi",
        "starlette",
        "uvicorn",
        "jsonschema",
        "sse-starlette",
    ),
}
PROFILE_INSTALL_COMMANDS = {
    "docs": 'python -m pip install -c constraints/ci.txt -e ".[docs]"',
    "cpu-starter": "python -m pip install -c constraints/ci.txt -e .",
    "notebooks": 'python -m pip install -c constraints/ci.txt -e ".[dev,torch,jax]"',
    "full-ci": (
        'python -m pip install -c constraints/ci.txt -e '
        '".[docs,dev,torch,jax,transformers,finetune,rag,langchain,'
        'llamaindex,api,evaluation,agents]"'
    ),
}
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


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "packages": {name: package_version(name) for name in REPORT_PACKAGES},
        "torch": torch_device_report(),
        "credentials_configured": {name: bool(os.getenv(name)) for name in SECRET_NAMES},
    }


NOTEBOOK_KERNEL_NAME = "about-llm"
NOTEBOOK_KERNEL_INSTALL_COMMAND = (
    'python -m ipykernel install --sys-prefix --name about-llm '
    '--display-name "Python (about-llm)"'
)


def notebook_kernel_check(
    *,
    kernel_specs: Mapping[str, str] | None = None,
    kernel_spec_loader: Callable[[str], Any] | None = None,
    executable: str | None = None,
) -> tuple[bool, str]:
    """Check that the documented kernel starts this virtual environment.

    Compare normalized paths without resolving symlinks.  Resolving them would
    incorrectly accept two virtual environments whose Python launchers share
    the same underlying interpreter.
    """
    if kernel_specs is None or kernel_spec_loader is None:
        try:
            from jupyter_client.kernelspec import find_kernel_specs, get_kernel_spec
        except ImportError:
            return False, "Jupyter kernelspec support is unavailable"
        kernel_specs = find_kernel_specs()
        kernel_spec_loader = get_kernel_spec

    if NOTEBOOK_KERNEL_NAME not in kernel_specs:
        return False, f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel is unavailable"

    try:
        kernel_spec = kernel_spec_loader(NOTEBOOK_KERNEL_NAME)
        argv = kernel_spec.argv
    except Exception:
        return False, f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel could not be read"
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        return False, f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel has no interpreter command"

    registered_executable = argv[0]
    expected_executable = executable or sys.executable
    if not os.path.isabs(registered_executable):
        return False, (
            f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel does not use an absolute interpreter path"
        )
    if os.path.normcase(os.path.abspath(registered_executable)) != os.path.normcase(
        os.path.abspath(expected_executable)
    ):
        return False, (
            f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel uses a different interpreter: "
            f"{registered_executable}"
        )
    return True, f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel uses the current interpreter"


def notebook_kernel_available() -> bool:
    return notebook_kernel_check()[0]


def evaluate_profile(
    profile: str,
    *,
    versions: Mapping[str, str | None] | None = None,
    python_version: tuple[int, int] | None = None,
    virtual_environment: bool | None = None,
    package_importable: bool | None = None,
    kernel_available: bool | None = None,
    root_writable: bool | None = None,
) -> dict[str, Any]:
    if profile not in PROFILE_REQUIREMENTS:
        raise ValueError(f"Unknown profile: {profile}")

    effective_python = python_version or (sys.version_info.major, sys.version_info.minor)
    effective_virtual_environment = (
        sys.prefix != sys.base_prefix
        if virtual_environment is None
        else virtual_environment
    )
    effective_package_importable = (
        importlib.util.find_spec("about_llm") is not None
        if package_importable is None
        else package_importable
    )
    effective_root_writable = os.access(ROOT, os.W_OK) if root_writable is None else root_writable
    effective_versions = (
        {name: package_version(name) for name in PROFILE_REQUIREMENTS[profile]}
        if versions is None
        else dict(versions)
    )

    checks: list[dict[str, str]] = []

    def add_check(name: str, status: str, detail: str, remediation: str = "") -> None:
        check = {"name": name, "status": status, "detail": detail}
        if remediation:
            check["remediation"] = remediation
        checks.append(check)

    python_supported = (
        SUPPORTED_PYTHON_MIN
        <= effective_python
        < SUPPORTED_PYTHON_MAX_EXCLUSIVE
    )
    add_check(
        "python_version",
        "pass" if python_supported else "fail",
        f"Python {effective_python[0]}.{effective_python[1]}",
        "Create a Python 3.10-3.12 virtual environment." if not python_supported else "",
    )
    add_check(
        "virtual_environment",
        "pass" if effective_virtual_environment else "warn",
        "isolated virtual environment"
        if effective_virtual_environment
        else "current interpreter is not inside a virtual environment",
        "python -m venv .venv" if not effective_virtual_environment else "",
    )
    add_check(
        "about_llm_import",
        "pass" if effective_package_importable else "fail",
        "about_llm is importable"
        if effective_package_importable
        else "about_llm is not importable from this interpreter",
        PROFILE_INSTALL_COMMANDS[profile] if not effective_package_importable else "",
    )

    missing_packages = [
        name for name in PROFILE_REQUIREMENTS[profile] if effective_versions.get(name) is None
    ]
    add_check(
        "required_packages",
        "pass" if not missing_packages else "fail",
        "all required packages are installed"
        if not missing_packages
        else f"missing packages: {', '.join(missing_packages)}",
        PROFILE_INSTALL_COMMANDS[profile] if missing_packages else "",
    )
    add_check(
        "workspace_writable",
        "pass" if effective_root_writable else "fail",
        "workspace can create local experiment outputs"
        if effective_root_writable
        else "workspace is not writable",
        "Use a writable clone or adjust directory permissions."
        if not effective_root_writable
        else "",
    )

    if profile in {"notebooks", "full-ci"}:
        kernel_detail = ""
        if kernel_available is None:
            effective_kernel_available, kernel_detail = notebook_kernel_check()
        else:
            effective_kernel_available = kernel_available
            kernel_detail = (
                f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel is available"
                if effective_kernel_available
                else f"{NOTEBOOK_KERNEL_NAME} Jupyter kernel is unavailable"
            )
        add_check(
            "about_llm_kernel",
            "pass" if effective_kernel_available else "fail",
            kernel_detail,
            NOTEBOOK_KERNEL_INSTALL_COMMAND
            if not effective_kernel_available
            else "",
        )

    statuses = {check["status"] for check in checks}
    status = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return {
        "profile": profile,
        "status": status,
        "install_command": PROFILE_INSTALL_COMMANDS[profile],
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_REQUIREMENTS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {"environment": environment_report()}
    if args.profile is not None:
        report["readiness"] = evaluate_profile(args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Credential values are intentionally never printed.")
    readiness = report.get("readiness")
    return 1 if isinstance(readiness, dict) and readiness["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
