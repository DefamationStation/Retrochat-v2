
# PowerShell launcher for Retrochat
# This script will activate the venv (if present) and run retrochat.py from the original project directory

# Dynamically determine the project directory
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "retrochat.py"))) {
    # If running from the project directory, use the script's location
    $projectDir = $PSScriptRoot
} else {
    # If running from installed location, use the placeholder that gets replaced during setup
    $projectDir = "{{PROJECT_DIR_PLACEHOLDER}}"
}

$venvPath = Join-Path $projectDir "venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$scriptPath = Join-Path $projectDir "retrochat.py"

# Change to the project directory to ensure relative imports work
Set-Location $projectDir

if (Test-Path $pythonExe) {
    & $pythonExe $scriptPath @args
} else {
    Write-Host "No venv found. Running with system Python."
    python $scriptPath @args
}
