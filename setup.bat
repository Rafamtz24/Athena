@echo off
REM ============================================================
REM  Athena Setup
REM
REM  Double-click to prepare a freshly cloned Athena for use:
REM  installs Python if needed, creates the virtual environment,
REM  installs dependencies and builds the inference backend for
REM  whatever GPU this machine has.
REM
REM  Safe to run more than once - it repairs a partial install
REM  rather than starting over.
REM
REM  Advanced (from a terminal):
REM      setup.bat -Backend cpu     build without GPU offload
REM      setup.bat -Backend vulkan  force Vulkan (AMD / Intel)
REM      setup.bat -Backend cuda    force CUDA (NVIDIA)
REM      setup.bat -Backend none    no compiler; use LM Studio
REM      setup.bat -Yes             never pause for confirmation
REM ============================================================
setlocal

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"

if not "%SETUP_EXIT%"=="0" (
    echo.
    echo Setup did not finish successfully ^(see the message above^).
    echo For help, see docs\SETUP.md
)

echo.
pause
endlocal
exit /b %SETUP_EXIT%
