
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
$requirementsPath = Join-Path $projectDir "requirements.txt"

# Change to the project directory to ensure relative imports work
Set-Location $projectDir

# Function to create virtual environment and install requirements
function Setup-Environment {
    Write-Host "Setting up Retrochat environment..." -ForegroundColor Yellow
    
    # Check if Python is available
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "Error: Python not found. Please install Python and add it to your PATH." -ForegroundColor Red
        exit 1
    }
    
    # Create virtual environment
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    if (-not (Test-Path $pythonExe)) {
        Write-Host "Error: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    
    # Install requirements
    if (Test-Path $requirementsPath) {
        Write-Host "Installing requirements..." -ForegroundColor Yellow
        & $pythonExe -m pip install --upgrade pip
        & $pythonExe -m pip install -r $requirementsPath
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Environment setup complete!" -ForegroundColor Green
        } else {
            Write-Host "Error: Failed to install requirements." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Warning: requirements.txt not found." -ForegroundColor Yellow
    }
}

# Check if venv exists and has required packages
if (Test-Path $pythonExe) {
    # Quick check if rich is installed (main dependency that was missing)
    $richCheck = & $pythonExe -c "import rich; print('OK')" 2>$null
    if ($richCheck -eq "OK") {
        # Environment is ready, run the app
        & $pythonExe $scriptPath @args
    } else {
        Write-Host "Virtual environment exists but dependencies are missing." -ForegroundColor Yellow
        Setup-Environment
        & $pythonExe $scriptPath @args
    }
} else {
    # No venv found, set it up
    Write-Host "No virtual environment found. Setting up for first time..." -ForegroundColor Yellow
    Setup-Environment
    & $pythonExe $scriptPath @args
}
