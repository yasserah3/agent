@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 or newer is required. Install it from python.org with the Python launcher.
        pause
        exit /b 1
    )
)
if not exist ".venv\ready_v1_6.txt" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check the internet connection and retry.
        pause
        exit /b 1
    )
    echo ready>".venv\ready_v1_6.txt"
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
