# PowerShell launcher for Retrochat
# This script will activate the venv (if present) and run retrochat.py using the venv's Python

$venvPath = Join-Path $PSScriptRoot "venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (Test-Path $pythonExe) {
    & $pythonExe retrochat.py
} else {
    Write-Host "No venv found. Running with system Python."
    python retrochat.py
}
