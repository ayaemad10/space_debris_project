@echo off
chcp 65001 >nul
title TAVRA Mission Control Launcher
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo   TAVRA - Space Debris Collision Prediction System
echo ============================================================
echo.

set "PYTHON311="
set "VENV_DIR=venv_py311"

where py >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%p in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON311=%%p"
)

if not defined PYTHON311 (
    set "PYTHON311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

if not exist "%PYTHON311%" (
    echo [ERROR] Python 3.11 was not found.
    echo.
    echo Install Python 3.11, then tick:
    echo   Add python.exe to PATH
    echo.
    echo Download:
    echo   https://www.python.org/downloads/release/python-3119/
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.11:
echo      "%PYTHON311%"
echo.

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    "%PYTHON311%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists.
)
echo.

echo [2/4] Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)
echo.

echo [3/4] Installing requirements...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)
echo.

echo [4/4] Starting Backend and Frontend...
start "TAVRA Backend (FastAPI)" cmd /k "cd /d "%~dp0" && "%VENV_DIR%\Scripts\python.exe" src\backend\app.py"
timeout /t 4 /nobreak >nul
start "TAVRA Frontend (Streamlit)" cmd /k "cd /d "%~dp0" && "%VENV_DIR%\Scripts\python.exe" -m streamlit run src\frontend\mission_control.py --server.port 8501"

echo.
echo ============================================================
echo   Project is running.
echo.
echo   Dashboard : http://localhost:8501
echo   API       : http://localhost:8000
echo   API Docs  : http://localhost:8000/docs
echo.
echo   To stop it, close the Backend and Frontend windows.
echo ============================================================
echo.

timeout /t 5 /nobreak >nul
start http://localhost:8501
pause
