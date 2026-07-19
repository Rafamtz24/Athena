@echo off
REM ============================================================
REM  Athena Launcher
REM
REM  Double-click to start Athena. Works from any location:
REM  it always runs from its own folder (no fixed paths), so you
REM  can move or rename the Athena folder freely.
REM
REM  First time here? Run setup.bat instead - it installs
REM  everything Athena needs.
REM ============================================================
setlocal

REM Run from the folder this launcher lives in.
cd /d "%~dp0"

REM Prefer a project virtual environment when one exists.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :run
)
if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
    goto :run
)

REM No virtual environment: this looks like a fresh clone that has not
REM been set up yet. Offer to do it now rather than failing later with
REM a confusing import error.
echo.
echo Athena has not been set up on this computer yet.
echo.
echo Setup installs Python ^(if needed^), creates a virtual environment,
echo and builds the inference backend for your GPU.
echo.
set /p RUNSETUP="Run setup now? [Y/n] "
if /i "%RUNSETUP%"=="n" goto :cancelled
if /i "%RUNSETUP%"=="no" goto :cancelled

call "%~dp0setup.bat"
if errorlevel 1 exit /b 1

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    goto :run
)

echo.
echo Setup finished but no virtual environment was found.
echo See docs\SETUP.md for help.
pause
exit /b 1

:cancelled
echo.
echo Setup skipped. Run setup.bat when you are ready.
pause
exit /b 1

:run
REM Make sure the interpreter is actually usable.
"%PYTHON%" --version >nul 2>nul
if errorlevel 1 (
    echo.
    echo The Python in this project's virtual environment is not working.
    echo Re-run setup.bat to rebuild it.
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
