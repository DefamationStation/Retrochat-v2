
# PowerShell launcher for Retrochat
# This script will activate the venv (if present) and run retrochat.py from the original project directory

# Set the path to the original project directory (edit this if your project moves)
$projectDir = "C:\Users\frenz\Documents\devspace\Retrochat-v2"
$venvPath = Join-Path $projectDir "venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$scriptPath = Join-Path $projectDir "retrochat.py"

if (Test-Path $pythonExe) {
    & $pythonExe $scriptPath @args
} else {
    Write-Host "No venv found. Running with system Python."
    python $scriptPath @args
}
