from __future__ import annotations

from scripts import doctor


def test_full_ci_profile_requires_both_interoperability_sdks() -> None:
    requirements = doctor.PROFILE_REQUIREMENTS["full-ci"]

    assert "mcp" in requirements
    assert "a2a-sdk" in requirements


def test_cpu_starter_profile_reports_missing_package() -> None:
    result = doctor.evaluate_profile(
        "cpu-starter",
        versions={"about-llm": "0.1.0", "numpy": None},
        python_version=(3, 12),
        virtual_environment=True,
        package_importable=True,
        root_writable=True,
    )

    assert result["status"] == "fail"
    package_check = next(
        check for check in result["checks"] if check["name"] == "required_packages"
    )
    assert package_check["detail"] == "missing packages: numpy"
    assert "pip install" in package_check["remediation"]


def test_docs_profile_warns_outside_virtual_environment() -> None:
    versions = {name: "1.0" for name in doctor.PROFILE_REQUIREMENTS["docs"]}

    result = doctor.evaluate_profile(
        "docs",
        versions=versions,
        python_version=(3, 10),
        virtual_environment=False,
        package_importable=True,
        root_writable=True,
    )

    assert result["status"] == "warn"
    virtual_environment_check = next(
        check for check in result["checks"] if check["name"] == "virtual_environment"
    )
    assert virtual_environment_check["status"] == "warn"
    assert virtual_environment_check["remediation"] == "python -m venv .venv"


def test_notebook_profile_requires_python3_kernel() -> None:
    versions = {name: "1.0" for name in doctor.PROFILE_REQUIREMENTS["notebooks"]}

    result = doctor.evaluate_profile(
        "notebooks",
        versions=versions,
        python_version=(3, 12),
        virtual_environment=True,
        package_importable=True,
        kernel_available=False,
        root_writable=True,
    )

    assert result["status"] == "fail"
    kernel_check = next(
        check for check in result["checks"] if check["name"] == "python3_kernel"
    )
    assert kernel_check["status"] == "fail"
    assert "ipykernel install" in kernel_check["remediation"]


def test_default_report_never_prints_secret_value(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "top-secret-value")

    assert doctor.main([]) == 0

    output = capsys.readouterr().out
    assert "top-secret-value" not in output
    assert '"OPENAI_API_KEY": true' in output
