#!/usr/bin/env pwsh
<#
.SYNOPSIS
    RetroChat v2 - Consolidated Universal Installer

.DESCRIPTION
    This script works on Windows, Linux, and macOS. It automatically detects the platform
    and installs RetroChat v2 with all dependencies.

.PARAMETER Force
    Force reinstallation even if RetroChat is already installed

.PARAMETER Branch
    Git branch to install (default: main)

.PARAMETER CustomPath
    Custom installation path (default: ~/.retrochat)

.PARAMETER Interactive
    Run in interactive mode with menu options

.EXAMPLE
    # Quick install
    ./install-universal.ps1
    
    # Force reinstall
    ./install-universal.ps1 -Force
    
    # Install specific branch
    ./install-universal.ps1 -Branch development
    
    # Interactive mode
    ./install-universal.ps1 -Interactive

.EXAMPLE
    # One-liner remote install (Windows)
    powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install-universal.ps1 | iex"
    
    # One-liner remote install (Linux/macOS)
    curl -fsSL https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install-universal.ps1 | pwsh -
#>

param(
    [switch]$Force,
    [string]$Branch = "main",
    [string]$CustomPath = "",
    [switch]$Interactive
)

$ErrorActionPreference = "Stop"

# Platform detection
$IsLinuxOrMac = $PSVersionTable.Platform -eq "Unix"
$IsWindowsPlatform = -not $IsLinuxOrMac

# Configuration
$REPO_OWNER = "DefamationStation"
$REPO_NAME = "Retrochat-v2"

# Platform-specific paths
if (-not $IsLinuxOrMac) {
    $DEFAULT_INSTALL_DIR = Join-Path $env:USERPROFILE ".retrochat"
    $PYTHON_CMD = "python"
    $VENV_PYTHON = "Scripts\python.exe"
} else {
    $DEFAULT_INSTALL_DIR = Join-Path $env:HOME ".retrochat"
    $PYTHON_CMD = "python3"
    $VENV_PYTHON = "bin/python"
}

$INSTALL_DIR = if ($CustomPath) { $CustomPath } else { $DEFAULT_INSTALL_DIR }
$SOURCE_DIR = Join-Path $INSTALL_DIR "source"
$VENV_DIR = Join-Path $SOURCE_DIR "venv"
$PYTHON_EXE = Join-Path $VENV_DIR $VENV_PYTHON

# ASCII Art
function Show-Banner {
    Write-Host ""
    Write-Host "  ____      _             ____ _           _   " -ForegroundColor Cyan
    Write-Host " |  _ \ ___| |_ _ __ ___  / ___| |__   __ _| |_ " -ForegroundColor Cyan
    Write-Host " | |_) / _ \ __| '__/ _ \| |   | '_ \ / _` | __|" -ForegroundColor Cyan
    Write-Host " |  _ <  __/ |_| |  | (_) | |___| | | | (_| | |_" -ForegroundColor Cyan
    Write-Host " |_| \_\___|\__|_|   \___/ \____|_| |_|\__,_|\__|" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "           Universal Cross-Platform Installer" -ForegroundColor Yellow
    Write-Host "           =================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Platform: $($PSVersionTable.Platform) | OS: $($PSVersionTable.OS)" -ForegroundColor Gray
    Write-Host ""
}

# Interactive menu
function Show-InteractiveMenu {
    Show-Banner
    Write-Host "What would you like to do?" -ForegroundColor White
    Write-Host ""
    Write-Host "[1] Quick Install (recommended)" -ForegroundColor Green
    Write-Host "[2] Install to custom location" -ForegroundColor Yellow
    Write-Host "[3] Force reinstall" -ForegroundColor Red
    Write-Host "[4] Install development branch" -ForegroundColor Magenta
    Write-Host "[5] Show system requirements" -ForegroundColor Blue
    Write-Host "[6] Cancel" -ForegroundColor Gray
    Write-Host ""
    
    do {
        $choice = Read-Host "Enter choice (1-6)"
        switch ($choice) {
            "1" { return @{} }
            "2" { 
                $path = Read-Host "Enter installation path (or press Enter for default)"
                if ($path) { return @{ CustomPath = $path } }
                return @{}
            }
            "3" { return @{ Force = $true } }
            "4" {
                $branch = Read-Host "Enter branch name (default: development)"
                if (-not $branch) { $branch = "development" }
                return @{ Branch = $branch }
            }
            "5" {
                Show-SystemRequirements
                Write-Host ""
                Write-Host "Press Enter to continue..."
                Read-Host
            }
            "6" { 
                Write-Host "Installation cancelled." -ForegroundColor Yellow
                exit 0
            }
            default {
                Write-Host "Invalid choice. Please enter 1-6." -ForegroundColor Red
            }
        }
    } while ($true)
}

function Show-SystemRequirements {
    Write-Host ""
    Write-Host "System Requirements:" -ForegroundColor Yellow
    Write-Host "===================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Required:" -ForegroundColor White
    Write-Host "- Python 3.8 or higher" -ForegroundColor Green
    Write-Host "- Internet connection" -ForegroundColor Green
    Write-Host ""
    Write-Host "Optional (for better updates):" -ForegroundColor White
    Write-Host "- Git" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Supported Platforms:" -ForegroundColor White
    Write-Host "- Windows 10/11" -ForegroundColor Green
    Write-Host "- Windows Server 2019+" -ForegroundColor Green
    Write-Host "- Linux (Ubuntu, Debian, CentOS, etc.)" -ForegroundColor Green
    Write-Host "- macOS 10.15+" -ForegroundColor Green
    Write-Host ""
}

# Function to check if command exists
function Test-Command($cmd) {
    try {
        if (Get-Command $cmd -ErrorAction Stop) { return $true }
    } catch {
        return $false
    }
}

# Progress bar helper
function Write-Progress-Custom($Activity, $Status, $PercentComplete) {
    Write-Progress -Activity $Activity -Status $Status -PercentComplete $PercentComplete
    Write-Host "[$PercentComplete%] $Status" -ForegroundColor Yellow
}

# Function to install Git (Windows only)
function Install-Git {
    if ($IsLinuxOrMac) { return $false }
    
    Write-Host "[?] Git not found. Git enables better updates and faster installs." -ForegroundColor Yellow
    $install = Read-Host "Would you like to install Git? (y/N)"
    
    if ($install -match '^[Yy]') {
        try {
            Write-Host "[*] Installing Git via winget..." -ForegroundColor Yellow
            & winget install --id Git.Git --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Git installed successfully!" -ForegroundColor Green
                # Refresh PATH
                $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
                return Test-Command "git"
            }
        } catch {
            Write-Host "[!] Winget installation failed, trying Chocolatey..." -ForegroundColor Yellow
            try {
                if (Test-Command "choco") {
                    & choco install git -y
                    if ($LASTEXITCODE -eq 0) {
                        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
                        return Test-Command "git"
                    }
                }
            } catch {}
        }
    }
    return $false
}

# Function to download and extract ZIP
function Install-FromZip {
    Write-Progress-Custom "Installing RetroChat" "Downloading from GitHub..." 20
    
    $zipUrl = "https://github.com/$REPO_OWNER/$REPO_NAME/archive/refs/heads/$Branch.zip"
    $zipPath = Join-Path ([System.IO.Path]::GetTempPath()) "retrochat-v2.zip"
    $extractPath = Join-Path ([System.IO.Path]::GetTempPath()) "retrochat-extract"
    
    try {
        # Download ZIP
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        Write-Progress-Custom "Installing RetroChat" "Download completed, extracting..." 40
        
        # Extract ZIP
        if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        
        # Move source files
        $sourcePath = Join-Path $extractPath "$REPO_NAME-$Branch"
        if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
        New-Item -ItemType Directory -Path $SOURCE_DIR -Force | Out-Null
        
        Get-ChildItem $sourcePath | Move-Item -Destination $SOURCE_DIR -Force
        
        # Cleanup
        Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
        Remove-Item $extractPath -Recurse -Force -ErrorAction SilentlyContinue
        
        Write-Host "[OK] Source code extracted successfully" -ForegroundColor Green
        return $true
        
    } catch {
        Write-Host "[ERROR] ZIP download failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to clone with Git
function Install-FromGit {
    Write-Progress-Custom "Installing RetroChat" "Cloning with Git..." 20
    
    try {
        if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
        
        $repoUrl = "https://github.com/$REPO_OWNER/$REPO_NAME.git"
        $gitArgs = @("clone", "--branch", $Branch, "--depth", "1", $repoUrl, $SOURCE_DIR)
        
        & git @gitArgs 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Repository cloned successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[!] Git clone failed, trying ZIP..." -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "[!] Git clone failed: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

# Function to setup Python environment
function Initialize-PythonEnvironment {
    Write-Progress-Custom "Installing RetroChat" "Setting up Python environment..." 60
    
    # Check Python
    if (-not (Test-Command $PYTHON_CMD)) {
        throw "Python is required but not found. Please install Python 3.8+ and try again."
    }
    
    # Verify Python version
    $pythonVersion = & $PYTHON_CMD --version 2>&1
    Write-Host "[*] Found: $pythonVersion" -ForegroundColor Green
    
    # Create virtual environment
    if (Test-Path $VENV_DIR) { Remove-Item $VENV_DIR -Recurse -Force }
    & $PYTHON_CMD -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
    
    # Install requirements with retries
    Write-Host "[*] Installing Python packages..." -ForegroundColor Yellow
    $maxRetries = 3
    $retryCount = 0
    
    do {
        $retryCount++
        try {
            $requirementsPath = Join-Path $SOURCE_DIR "requirements.txt"
            & $PYTHON_EXE -m pip install --upgrade pip
            & $PYTHON_EXE -m pip install -r $requirementsPath
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Python packages installed successfully" -ForegroundColor Green
                break
            } else {
                throw "pip install failed"
            }
        } catch {
            if ($retryCount -ge $maxRetries) {
                throw "Failed to install requirements after $maxRetries attempts: $($_.Exception.Message)"
            }
            Write-Host "[!] Attempt $retryCount failed, retrying... ($($_.Exception.Message))" -ForegroundColor Yellow
            Start-Sleep 2
        }
    } while ($retryCount -lt $maxRetries)
}

# Function to create launcher
function Initialize-Launcher {
    Write-Progress-Custom "Installing RetroChat" "Creating launcher..." 80
    
    if (-not $IsLinuxOrMac) {
        # Windows launchers
        $batchContent = @"
@echo off
"$PYTHON_EXE" "$SOURCE_DIR\retrochat.py" %*
"@
        $batchPath = Join-Path $INSTALL_DIR "rchat.bat"
        $batchContent | Out-File -FilePath $batchPath -Encoding ASCII
        
        # Add to PATH
        $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($userPath -notlike "*$INSTALL_DIR*") {
            [Environment]::SetEnvironmentVariable("PATH", "$userPath;$INSTALL_DIR", "User")
            Write-Host "[OK] Added to PATH" -ForegroundColor Green
        }
    } else {
        # Unix launcher
        $shellContent = @"
#!/usr/bin/env bash
"$PYTHON_EXE" "$SOURCE_DIR/retrochat.py" `$@
"@
        $shellPath = Join-Path $INSTALL_DIR "rchat"
        $shellContent | Out-File -FilePath $shellPath -Encoding UTF8
        & chmod +x $shellPath
        
        # Add to PATH via shell profile
        $profileFiles = @("~/.bashrc", "~/.zshrc", "~/.profile")
        $pathLine = "export PATH=`"${INSTALL_DIR}:`$PATH`""
        
        foreach ($profileFile in $profileFiles) {
            $profilePath = [System.IO.Path]::GetFullPath($profileFile.Replace("~", $env:HOME))
            if (Test-Path $profilePath) {
                $content = Get-Content $profilePath -Raw
                if ($content -notlike "*$INSTALL_DIR*") {
                    Add-Content $profilePath "`n# RetroChat v2`n$pathLine"
                }
            }
        }
    }
}

# Main installation function
function Start-Installation {
    try {
        Show-Banner
        
        # Check for existing installation
        if ((Test-Path $INSTALL_DIR) -and (-not $Force)) {
            Write-Host "[!] RetroChat appears to be already installed at: $INSTALL_DIR" -ForegroundColor Yellow
            $overwrite = Read-Host "Overwrite existing installation? (y/N)"
            if ($overwrite -notmatch '^[Yy]') {
                Write-Host "Installation cancelled." -ForegroundColor Yellow
                return
            }
        }
        
        # Create installation directory
        Write-Progress-Custom "Installing RetroChat" "Preparing installation..." 10
        New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
        
        # Download source code
        $success = $false
        
        # Try Git first (preferred for updates)
        if (Test-Command "git") {
            $success = Install-FromGit
        } elseif (-not $IsLinuxOrMac) {
            # Offer Git installation on Windows
            if (Install-Git) {
                $success = Install-FromGit
            }
        }
        
        # Fallback to ZIP
        if (-not $success) {
            Write-Host "[*] Using ZIP download method..." -ForegroundColor Yellow
            $success = Install-FromZip
        }
        
        if (-not $success) {
            throw "Failed to download RetroChat source code"
        }
        
        # Setup environment and launcher
        Initialize-PythonEnvironment
        Initialize-Launcher
        
        Write-Progress -Activity "Installing RetroChat" -Completed
        Write-Host ""
        Write-Host "[SUCCESS] RetroChat v2 installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Installation Details:" -ForegroundColor Cyan
        Write-Host "  Location: $INSTALL_DIR" -ForegroundColor White
        Write-Host "  Command:  rchat" -ForegroundColor White
        Write-Host "  Branch:   $Branch" -ForegroundColor White
        Write-Host ""
        
        if (-not $IsLinuxOrMac) {
            Write-Host "You can now run 'rchat' from any command prompt!" -ForegroundColor Green
            Write-Host "Note: Restart your terminal if the command is not found." -ForegroundColor Yellow
        } else {
            Write-Host "You can now run 'rchat' from any terminal!" -ForegroundColor Green
            Write-Host "Note: Run 'source ~/.bashrc' or restart your terminal if needed." -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host ""
        Write-Host "[ERROR] Installation failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please report this issue at: https://github.com/$REPO_OWNER/$REPO_NAME/issues" -ForegroundColor Blue
        exit 1
    }
}

# Main execution
if ($Interactive) {
    $params = Show-InteractiveMenu
    foreach ($key in $params.Keys) {
        Set-Variable -Name $key -Value $params[$key]
    }
}

Start-Installation
