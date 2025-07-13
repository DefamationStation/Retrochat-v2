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

Write-Host "🚀 RetroChat v2 Installer" -ForegroundColor Cyan
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
    Write-Host "📦 Downloading latest version from GitHub..." -ForegroundColor Yellow
    
    $zipUrl = "https://github.com/$REPO_OWNER/$REPO_NAME/archive/refs/heads/$Branch.zip"
    $zipPath = Join-Path $env:TEMP "retrochat-v2.zip"
    $extractPath = Join-Path $env:TEMP "retrochat-extract"
    
    try {
        # Download ZIP
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        Write-Host "✅ Download completed" -ForegroundColor Green
        
        # Extract ZIP
        if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        
        # Copy to source directory
        $extractedDir = Get-ChildItem $extractPath | Select-Object -First 1
        if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
        Move-Item $extractedDir.FullName $SOURCE_DIR
        
        # Cleanup
        Remove-Item $zipPath -Force
        Remove-Item $extractPath -Recurse -Force
        
        Write-Host "✅ Installation completed via ZIP download" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ ZIP download failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install using Git
function Install-FromGit {
    Write-Host "📦 Cloning repository with Git..." -ForegroundColor Yellow
    
    try {
        if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
        
        $gitArgs = @("clone", "https://github.com/$REPO_OWNER/$REPO_NAME.git", $SOURCE_DIR)
        if ($Branch -ne "main") {
            $gitArgs += @("--branch", $Branch)
        }
        
        & git @gitArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Repository cloned successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Git clone failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ Git clone failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to offer Git installation
function Install-Git {
    Write-Host "🔧 Git is not installed. Would you like to install it automatically?" -ForegroundColor Yellow
    $response = Read-Host "This will download and install Git for Windows. Continue? (y/N)"
    
    if ($response -match "^[Yy]") {
        Write-Host "📦 Downloading Git for Windows..." -ForegroundColor Yellow
        
        try {
            # Get latest Git release
            $gitRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest"
            $gitInstaller = $gitRelease.assets | Where-Object { $_.name -match "Git-.*-64-bit\.exe$" } | Select-Object -First 1
            
            if (-not $gitInstaller) {
                Write-Host "❌ Could not find Git installer" -ForegroundColor Red
                return $false
            }
            
            $installerPath = Join-Path $env:TEMP $gitInstaller.name
            Invoke-WebRequest -Uri $gitInstaller.browser_download_url -OutFile $installerPath
            
            Write-Host "🔧 Installing Git (this may take a few minutes)..." -ForegroundColor Yellow
            Start-Process -FilePath $installerPath -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
            
            # Refresh PATH
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
            
            Remove-Item $installerPath -Force
            
            if (Test-Command "git") {
                Write-Host "✅ Git installed successfully" -ForegroundColor Green
                return $true
            } else {
                Write-Host "❌ Git installation may have failed. Please restart your terminal." -ForegroundColor Red
                return $false
            }
        } catch {
            Write-Host "❌ Git installation failed: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "ℹ️ Continuing with ZIP download method" -ForegroundColor Blue
        return $false
    }
}

# Function to setup Python environment
function Setup-PythonEnvironment {
    Write-Host "🐍 Setting up Python environment..." -ForegroundColor Yellow
    
    # Check Python
    if (-not (Test-Command "python")) {
        Write-Host "❌ Python not found. Please install Python and add it to PATH." -ForegroundColor Red
        Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Blue
        exit 1
    }
    
    $pythonVersion = python --version
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
    
    # Create virtual environment
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    Set-Location $SOURCE_DIR
    python -m venv venv
    
    if (-not (Test-Path $PYTHON_EXE)) {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    
    # Install requirements
    $requirementsPath = Join-Path $SOURCE_DIR "requirements.txt"
    if (Test-Path $requirementsPath) {
        Write-Host "📦 Installing Python packages..." -ForegroundColor Yellow
        & $PYTHON_EXE -m pip install --upgrade pip --quiet
        & $PYTHON_EXE -m pip install -r $requirementsPath --quiet
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python packages installed successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to install Python packages" -ForegroundColor Red
            exit 1
        }
    }
}

# Function to setup launcher
function Setup-Launcher {
    Write-Host "🔧 Setting up global launcher..." -ForegroundColor Yellow
    
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
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$INSTALL_DIR*") {
        Write-Host "🔧 Adding to PATH..." -ForegroundColor Yellow
        [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$INSTALL_DIR", "User")
        $env:PATH += ";$INSTALL_DIR"
        Write-Host "✅ Added to PATH (restart terminal for system-wide effect)" -ForegroundColor Green
    } else {
        Write-Host "✅ Already in PATH" -ForegroundColor Green
    }
}

# Main installation process
try {
    Write-Host "📁 Installation directory: $INSTALL_DIR" -ForegroundColor Blue
    
    # Create installation directory
    if (-not (Test-Path $INSTALL_DIR)) {
        New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
    }
    
    # Check if already installed
    if ((Test-Path $SOURCE_DIR) -and -not $Force) {
        Write-Host "⚠️ RetroChat appears to be already installed." -ForegroundColor Yellow
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
        Write-Host "📦 Trying Git clone method..." -ForegroundColor Yellow
        
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
        Write-Host "❌ Failed to download RetroChat. Please check your internet connection." -ForegroundColor Red
        exit 1
    }
    
    # Setup environment
    Setup-PythonEnvironment
    Setup-Launcher
    
    Write-Host ""
    Write-Host "🎉 RetroChat v2 installed successfully!" -ForegroundColor Green
    Write-Host "📁 Installed to: $INSTALL_DIR" -ForegroundColor Blue
    Write-Host "🚀 Run 'rchat' from anywhere to start!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Note: If 'rchat' command is not found, restart your terminal." -ForegroundColor Yellow
    
} catch {
    Write-Host "❌ Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please report this error at: https://github.com/$REPO_OWNER/$REPO_NAME/issues" -ForegroundColor Blue
    exit 1
}
