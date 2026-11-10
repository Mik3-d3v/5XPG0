@echo off
rem start_demo.bat - double-click launcher for the live ECG demo.
rem Runs start_demo.sh with Git Bash: uploads to the PYNQ-Z2, starts the
rem board and PC servers and opens http://localhost:8001.
rem Keep this window open while presenting; close it to stop the demo.
rem Arguments are passed on, e.g.  start_demo.bat --offline

title ECG monitor demo
cd /d "%~dp0"

set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%ProgramFiles(x86)%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%LocalAppData%\Programs\Git\bin\bash.exe"
if not exist "%BASH%" (
    echo Git Bash was not found. Install Git for Windows: https://git-scm.com
    pause
    exit /b 1
)

"%BASH%" ./start_demo.sh %*
echo.
echo The demo has stopped (see the messages above).
pause
