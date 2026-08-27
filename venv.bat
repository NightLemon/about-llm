@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "ACTIVATE_SCRIPT=%REPO_ROOT%.venv\Scripts\Activate.ps1"

if not exist "%ACTIVATE_SCRIPT%" (
    echo [ERROR] Virtual environment not found: "%ACTIVATE_SCRIPT%"
    echo Create it first with: python -m venv .venv
    exit /b 1
)

where pwsh.exe >nul 2>&1
if not errorlevel 1 (
    set "POWERSHELL_EXE=pwsh.exe"
) else (
    set "POWERSHELL_EXE=powershell.exe"
)

if /i "%~1"=="--check" (
    "%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -Command "Set-Location -LiteralPath $env:REPO_ROOT; & $env:ACTIVATE_SCRIPT; Write-Output ('VIRTUAL_ENV=' + $env:VIRTUAL_ENV); Write-Output ('PYTHON=' + (Get-Command python).Source)"
    exit /b %errorlevel%
)

echo Starting an activated PowerShell. Run "exit" to return.
"%POWERSHELL_EXE%" -NoLogo -NoExit -ExecutionPolicy RemoteSigned -Command "Set-Location -LiteralPath $env:REPO_ROOT; & $env:ACTIVATE_SCRIPT"