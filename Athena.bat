@echo off
REM ============================================================
REM  Athena Launcher
REM
REM  Double-click to start Athena. Works from any location:
REM  it always runs from its own folder (no fixed paths), so you
REM  can move or rename the Athena folder freely.
REM ============================================================
setlocal

REM Run from the folder this launcher lives in.
cd /d "%~dp0"

REM Prefer a project virtual environment when one exists.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

REM Make sure Python is actually available.
"%PYTHON%" --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python was not found on this computer.
    echo Please install Python 3.10 or newer from https://www.python.org/downloads/
    echo ^(during installation, tick "Add Python to PATH"^).
    echo.
    pause
    exit /b 1
)

"%PYTHON%" -m athena.terminal_chat

REM If Athena exited with an error, keep the window open so the
REM message can be read before the console closes.
if errorlevel 1 (
    echo.
    echo Athena closed with an error ^(see message above^).
    pause
)

endlocal
