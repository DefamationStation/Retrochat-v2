# RetroChat v2 - Installed Launcher
# This script runs RetroChat from the installed location in ~/.retrochat/

$INSTALL_DIR = Join-Path $env:USERPROFILE ".retrochat"
$SOURCE_DIR = Join-Path $INSTALL_DIR "source"
$VENV_DIR = Join-Path $SOURCE_DIR "venv"
$PYTHON_EXE = Join-Path $VENV_DIR "Scripts\python.exe"
$SCRIPT_PATH = Join-Path $SOURCE_DIR "retrochat.py"

# Check if installation exists
if (-not (Test-Path $SOURCE_DIR)) {
    Write-Host "❌ RetroChat installation not found at $SOURCE_DIR" -ForegroundColor Red
    Write-Host "Please run the installer first:" -ForegroundColor Yellow
    Write-Host "powershell -ExecutionPolicy Bypass -Command `"iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex`"" -ForegroundColor Cyan
    exit 1
}

# Check if Python environment exists
if (-not (Test-Path $PYTHON_EXE)) {
    Write-Host "❌ Python environment not found. Please reinstall RetroChat." -ForegroundColor Red
    exit 1
}

# Change to source directory and run RetroChat
Set-Location $SOURCE_DIR
& $PYTHON_EXE $SCRIPT_PATH @args
