@echo off
REM Batch launcher for Retrochat
REM This script will create venv if needed, install requirements, and run retrochat.py

cd /d "%~dp0"

set VENV_PATH=%~dp0venv
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
set REQUIREMENTS_PATH=%~dp0requirements.txt

REM Check if venv exists and has required packages
if exist "%PYTHON_EXE%" (
    REM Quick check if rich is installed
    "%PYTHON_EXE%" -c "import rich; print('OK')" >nul 2>&1
    if %errorlevel% equ 0 (
        REM Environment is ready, run the app
        "%PYTHON_EXE%" retrochat.py %*
        goto :end
    ) else (
        echo Virtual environment exists but dependencies are missing.
        goto :setup
    )
) else (
    echo No virtual environment found. Setting up for first time...
    goto :setup
)

:setup
echo Setting up Retrochat environment...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python and add it to your PATH.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if not exist "%PYTHON_EXE%" (
    echo Error: Failed to create virtual environment.
    pause
    exit /b 1
)

REM Install requirements
if exist "%REQUIREMENTS_PATH%" (
    echo Installing requirements...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_PATH%"
    
    if %errorlevel% equ 0 (
        echo Environment setup complete!
        "%PYTHON_EXE%" retrochat.py %*
    ) else (
        echo Error: Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    echo Warning: requirements.txt not found.
    "%PYTHON_EXE%" retrochat.py %*
)

:end