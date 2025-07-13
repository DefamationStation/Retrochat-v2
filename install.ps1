# RetroChat v2 - One-Click Installer
# This script downloads, installs, and sets up RetroChat in ~/.retrochat/
# Usage: powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex"

param(
    [switch]$Force,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

# Configuration
$REPO_OWNER = "DefamationStation"
$REPO_NAME = "Retrochat-v2"
$INSTALL_DIR = Join-Path $env:USERPROFILE ".retrochat"
$SOURCE_DIR = Join-Path $INSTALL_DIR "source"
$VENV_DIR = Join-Path $SOURCE_DIR "venv"
$PYTHON_EXE = Join-Path $VENV_DIR "Scripts\python.exe"

Write-Host ">> RetroChat v2 Installer" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Function to check if command exists
function Test-Command($cmd) {
    try {
        Get-Command $cmd -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to download and extract ZIP
function Install-FromZip {
    Write-Host "[*] Downloading latest version from GitHub..." -ForegroundColor Yellow
    
    $zipUrl = "https://github.com/$REPO_OWNER/$REPO_NAME/archive/refs/heads/$Branch.zip"
    $zipPath = Join-Path $env:TEMP "retrochat-v2.zip"
    $extractPath = Join-Path $env:TEMP "retrochat-extract"
    
    try {
        # Download ZIP
        Write-Host "[*] Please wait, downloading may take a moment..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        Write-Host "[OK] Download completed" -ForegroundColor Green
        
        # Extract ZIP
        Write-Host "[*] Extracting files..." -ForegroundColor Yellow
        if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        
        # Copy to source directory
        $extractedDir = Get-ChildItem $extractPath | Select-Object -First 1
        if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
        Move-Item $extractedDir.FullName $SOURCE_DIR
        
        # Cleanup
        Remove-Item $zipPath -Force
        Remove-Item $extractPath -Recurse -Force
        
        Write-Host "[OK] Installation completed via ZIP download" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[ERROR] ZIP download failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install using Git
function Install-FromGit {
    Write-Host "[*] Cloning repository with Git..." -ForegroundColor Yellow
    
    try {
        if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
        
        $gitArgs = @("clone", "https://github.com/$REPO_OWNER/$REPO_NAME.git", $SOURCE_DIR)
        if ($Branch -ne "main") {
            $gitArgs += @("--branch", $Branch)
        }
        
        Write-Host "[*] Please wait, cloning repository..." -ForegroundColor Gray
        & git @gitArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Repository cloned successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[ERROR] Git clone failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "[ERROR] Git clone failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to offer Git installation
function Install-Git {
    Write-Host "[*] Git is not installed. Would you like to install it automatically?" -ForegroundColor Yellow
    $response = Read-Host "This will download and install Git for Windows. Continue? (y/N)"
    
    if ($response -match "^[Yy]") {
        Write-Host "[*] Downloading Git for Windows..." -ForegroundColor Yellow
        
        try {
            # Get latest Git release
            Write-Host "[*] Finding latest Git version..." -ForegroundColor Gray
            $gitRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest"
            $gitInstaller = $gitRelease.assets | Where-Object { $_.name -match "Git-.*-64-bit\.exe$" } | Select-Object -First 1
            
            if (-not $gitInstaller) {
                Write-Host "[ERROR] Could not find Git installer" -ForegroundColor Red
                return $false
            }
            
            $installerPath = Join-Path $env:TEMP $gitInstaller.name
            Write-Host "[*] Downloading Git installer (this may take a few minutes)..." -ForegroundColor Gray
            Invoke-WebRequest -Uri $gitInstaller.browser_download_url -OutFile $installerPath
            
            Write-Host "[*] Installing Git (this may take a few minutes)..." -ForegroundColor Yellow
            Write-Host "[*] Please wait, the installer may appear to freeze but is working..." -ForegroundColor Gray
            Start-Process -FilePath $installerPath -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
            
            # Refresh PATH
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
            
            Remove-Item $installerPath -Force
            
            if (Test-Command "git") {
                Write-Host "[OK] Git installed successfully" -ForegroundColor Green
                return $true
            } else {
                Write-Host "[ERROR] Git installation may have failed. Please restart your terminal." -ForegroundColor Red
                return $false
            }
        } catch {
            Write-Host "[ERROR] Git installation failed: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "[INFO] Continuing with ZIP download method" -ForegroundColor Blue
        return $false
    }
}

# Function to setup Python environment
function Initialize-PythonEnvironment {
    Write-Host "[*] Setting up Python environment..." -ForegroundColor Yellow
    
    # Check Python
    if (-not (Test-Command "python")) {
        Write-Host "[ERROR] Python not found. Please install Python and add it to PATH." -ForegroundColor Red
        Write-Host "[INFO] Download from: https://www.python.org/downloads/" -ForegroundColor Blue
        exit 1
    }
    
    $pythonVersion = python --version
    Write-Host "[OK] Found Python: $pythonVersion" -ForegroundColor Green
    
    # Create virtual environment
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Yellow
    Write-Host "[*] Please wait, this may take a moment..." -ForegroundColor Gray
    Set-Location $SOURCE_DIR
    python -m venv venv
    
    if (-not (Test-Path $PYTHON_EXE)) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
    
    # Install requirements
    $requirementsPath = Join-Path $SOURCE_DIR "requirements.txt"
    if (Test-Path $requirementsPath) {
        Write-Host "[*] Installing Python packages..." -ForegroundColor Yellow
        Write-Host "[*] This may take several minutes and may appear frozen - please wait..." -ForegroundColor Gray
        Write-Host "[*] Installing pip updates..." -ForegroundColor Cyan
        
        # Upgrade pip with better error handling
        try {
            $pipUpgrade = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "--upgrade", "pip", "--quiet" -NoNewWindow -PassThru -Wait
            
            if ($pipUpgrade.ExitCode -eq 0) {
                Write-Host "[OK] Pip updated successfully" -ForegroundColor Green
            } else {
                Write-Host "[WARNING] Pip upgrade had issues, continuing anyway..." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[WARNING] Pip upgrade failed, continuing with existing version..." -ForegroundColor Yellow
        }
        
        Write-Host "[*] Installing packages from requirements.txt..." -ForegroundColor Cyan
        Write-Host "[*] This step takes the longest - installing AI and document processing libraries..." -ForegroundColor Gray
        
        # First try with --quiet flag
        try {
            $reqInstall = Start-Process -FilePath $PYTHON_EXE -ArgumentList "-m", "pip", "install", "-r", $requirementsPath, "--quiet" -NoNewWindow -PassThru
            
            # Show progress with package names
            $progressCount = 0
            $packages = @("anthropic", "chromadb", "langchain", "rich", "requests", "other packages")
            
            while (-not $reqInstall.HasExited) {
                $currentPackage = $packages[$progressCount % $packages.Length]
                Write-Host "[*] Installing $currentPackage" -ForegroundColor Cyan
                Start-Sleep -Seconds 3
                $progressCount++
            }
            
            $reqInstall.WaitForExit()
            
            if ($reqInstall.ExitCode -eq 0) {
                Write-Host "[OK] Python packages installed successfully" -ForegroundColor Green
            } else {
                throw "Pip installation failed with exit code $($reqInstall.ExitCode)"
            }
            
        } catch {
            Write-Host "[WARNING] Quiet installation failed, trying with verbose output..." -ForegroundColor Yellow
            Write-Host "[*] Installing packages (with output)..." -ForegroundColor Cyan
            
            # Try again without --quiet to see actual error
            & $PYTHON_EXE -m pip install -r $requirementsPath
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[ERROR] Package installation failed." -ForegroundColor Red
                Write-Host "[INFO] This could be due to:" -ForegroundColor Yellow
                Write-Host "  - Poor internet connection" -ForegroundColor Yellow
                Write-Host "  - Corporate firewall blocking downloads" -ForegroundColor Yellow
                Write-Host "  - Python/pip configuration issues" -ForegroundColor Yellow
                Write-Host "[INFO] Try running the installer again, or install manually with:" -ForegroundColor Blue
                Write-Host "  cd $SOURCE_DIR" -ForegroundColor Blue
                Write-Host "  venv\Scripts\python -m pip install -r requirements.txt" -ForegroundColor Blue
                exit 1
            } else {
                Write-Host "[OK] Packages installed successfully on retry" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "[WARNING] requirements.txt not found, skipping package installation" -ForegroundColor Yellow
    }
}

# Function to setup launcher
function Initialize-Launcher {
    Write-Host "[*] Setting up global launcher..." -ForegroundColor Yellow
    
    # Create launcher script
    $launcherPath = Join-Path $INSTALL_DIR "rchat.ps1"
    $launcherBatPath = Join-Path $INSTALL_DIR "rchat.bat"
    
    # PowerShell launcher
    $launcherContent = @"
# RetroChat v2 Launcher - Installed Version
Set-Location "$SOURCE_DIR"
& "$PYTHON_EXE" "retrochat.py" @args
"@
    
    Set-Content -Path $launcherPath -Value $launcherContent
    
    # Batch launcher
    $batContent = @"
@echo off
powershell -ExecutionPolicy Bypass -File "$launcherPath" %*
"@
    
    Set-Content -Path $launcherBatPath -Value $batContent
    
    # Add to PATH
    Write-Host "[*] Adding to system PATH..." -ForegroundColor Yellow
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$INSTALL_DIR*") {
        Write-Host "[*] Updating PATH environment variable..." -ForegroundColor Gray
        [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$INSTALL_DIR", "User")
        $env:PATH += ";$INSTALL_DIR"
        Write-Host "[OK] Added to PATH (restart terminal for system-wide effect)" -ForegroundColor Green
    } else {
        Write-Host "[OK] Already in PATH" -ForegroundColor Green
    }
}

# Main installation process
try {
    Write-Host "[INFO] Installation directory: $INSTALL_DIR" -ForegroundColor Blue
    
    # Create installation directory
    if (-not (Test-Path $INSTALL_DIR)) {
        New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
    }
    
    # Check if already installed
    if ((Test-Path $SOURCE_DIR) -and -not $Force) {
        Write-Host "[WARNING] RetroChat appears to be already installed." -ForegroundColor Yellow
        $response = Read-Host "Reinstall? This will overwrite existing installation. (y/N)"
        if ($response -notmatch "^[Yy]") {
            Write-Host "Installation cancelled." -ForegroundColor Yellow
            exit 0
        }
    }
    
    # Download source code
    $success = $false
    
    # Try ZIP download first (faster, no dependencies)
    $success = Install-FromZip
    
    # Fallback to Git if ZIP failed
    if (-not $success) {
        Write-Host "[*] Trying Git clone method..." -ForegroundColor Yellow
        
        if (Test-Command "git") {
            $success = Install-FromGit
        } else {
            # Offer to install Git
            if (Install-Git) {
                $success = Install-FromGit
            }
        }
    }
    
    if (-not $success) {
        Write-Host "[ERROR] Failed to download RetroChat. Please check your internet connection." -ForegroundColor Red
        exit 1
    }
    
    # Setup environment
    Initialize-PythonEnvironment
    Initialize-Launcher
    
    Write-Host ""
    Write-Host "[SUCCESS] RetroChat v2 installed successfully!" -ForegroundColor Green
    Write-Host "[INFO] Installed to: $INSTALL_DIR" -ForegroundColor Blue
    Write-Host "[INFO] Run 'rchat' from anywhere to start!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Note: If 'rchat' command is not found, restart your terminal." -ForegroundColor Yellow
    
} catch {
    Write-Host "[ERROR] Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please report this error at: https://github.com/$REPO_OWNER/$REPO_NAME/issues" -ForegroundColor Blue
    exit 1
}
