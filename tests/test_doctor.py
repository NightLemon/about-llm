from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import doctor

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = tuple(sorted((ROOT / "notebooks").glob("*.ipynb")))
EXPECTED_KERNEL_NAME = "about-llm"
EXPECTED_INSTALL_COMMAND = (
    'python -m ipykernel install --sys-prefix --name about-llm '
    '--display-name "Python (about-llm)"'
)


def _kernel(argv: list[str]) -> SimpleNamespace:
    return SimpleNamespace(argv=argv)


def _notebook_kernel_names() -> set[str]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["metadata"]["kernelspec"]["name"]
        for path in NOTEBOOKS
    }


def test_notebook_metadata_and_readme_define_the_kernel_contract() -> None:
    assert NOTEBOOKS
    assert _notebook_kernel_names() == {EXPECTED_KERNEL_NAME}
    readme = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    assert EXPECTED_INSTALL_COMMAND in readme


def test_kernel_check_rejects_python3_when_documented_kernel_is_missing(tmp_path: Path) -> None:
    executable = str(tmp_path / "current-env" / "python")
    available, detail = doctor.notebook_kernel_check(
        kernel_specs={"python3": str(tmp_path / "kernels" / "python3")},
        kernel_spec_loader=lambda _: _kernel([executable]),
        executable=executable,
    )

    assert not available
    assert EXPECTED_KERNEL_NAME in detail


def test_kernel_check_accepts_documented_kernel_for_current_interpreter(tmp_path: Path) -> None:
    executable = str(tmp_path / "current-env" / "python")
    available, detail = doctor.notebook_kernel_check(
        kernel_specs={EXPECTED_KERNEL_NAME: str(tmp_path / "kernels" / "about-llm")},
        kernel_spec_loader=lambda _: _kernel([executable, "-m"]),
        executable=executable,
    )

    assert available
    assert "current interpreter" in detail


def test_kernel_check_rejects_documented_name_with_other_environment(tmp_path: Path) -> None:
    available, detail = doctor.notebook_kernel_check(
        kernel_specs={EXPECTED_KERNEL_NAME: str(tmp_path / "kernels" / "about-llm")},
        kernel_spec_loader=lambda _: _kernel([str(tmp_path / "old-env" / "python")]),
        executable=str(tmp_path / "current-env" / "python"),
    )

    assert not available
    assert "different interpreter" in detail


@pytest.mark.parametrize("profile", ["notebooks", "full-ci"])
def test_notebook_profiles_report_readme_kernel_remediation(profile: str) -> None:
    readiness = doctor.evaluate_profile(
        profile,
        versions={name: "installed" for name in doctor.PROFILE_REQUIREMENTS[profile]},
        python_version=(3, 12),
        virtual_environment=True,
        package_importable=True,
        kernel_available=False,
        root_writable=True,
    )

    kernel_check = next(
        check for check in readiness["checks"] if check["name"] == "about_llm_kernel"
    )
    assert kernel_check["status"] == "fail"
    assert kernel_check["detail"]
    assert kernel_check["remediation"] == EXPECTED_INSTALL_COMMAND
