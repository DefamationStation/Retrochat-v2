#!/usr/bin/env pwsh
<#
.SYNOPSIS
    RetroChat v2 - One Universal Installer
    
.DESCRIPTION
    Single file that installs RetroChat on Windows, Linux, or macOS.
    Works as PowerShell script, batch file, or shell script.
    
.USAGE
    # Quick install (one command):
    powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex"
    
    # Or download and run:
    ./install.ps1
    ./install.ps1 -Force
    ./install.ps1 -Interactive
    
.NOTES
    This file replaces all other installers and works everywhere.
#>
<# :BATCH_START
@echo off
title RetroChat v2 - Universal Installer
echo.
echo  ____      _             ____ _           _   
echo ^|  _ \ ___^| ^|_ _ __ ___  / ___^| ^|__   __ _^| ^|_ 
echo ^| ^|_) / _ \ __^| '__/ _ \^| ^|   ^| '_ \ / _` ^| __^|
echo ^|  _ ^<  __/ ^|_^| ^|  ^| ^(_) ^| ^|___^| ^| ^| ^| ^(_^| ^| ^|_ 
echo ^|_^| \_\___^|\__^|_^|   \___/ \____^|_^| ^|_^|\__,_^|\__^|
echo.
echo           Universal Installer
echo        ======================
echo.

REM Check if PowerShell is available
powershell -Command "exit 0" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is required but not available.
    echo Please install PowerShell or use Windows 7+
    pause
    exit /b 1
)

echo [*] Starting PowerShell installer...
powershell -ExecutionPolicy Bypass -File "%~f0" %*
if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Installation completed!
    echo [INFO] You can now run 'rchat' from anywhere
) else (
    echo.
    echo [ERROR] Installation failed
)
echo.
pause
exit /b %errorlevel%
:BATCH_END #>

param(
    [switch]$Force,
    [string]$Branch = "main",
    [string]$CustomPath = "",
    [switch]$Interactive
)

$ErrorActionPreference = "Stop"

# Platform detection
$IsLinuxOrMac = $PSVersionTable.Platform -eq "Unix"

# Configuration
$REPO_OWNER = "DefamationStation"
$REPO_NAME = "Retrochat-v2"

# Platform-specific settings
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

# Functions
function Show-Banner {
    Write-Host ""
    Write-Host "  ____      _             ____ _           _   " -ForegroundColor Cyan
    Write-Host " |  _ \ ___| |_ _ __ ___  / ___| |__   __ _| |_ " -ForegroundColor Cyan
    Write-Host " | |_) / _ \ __| '__/ _ \| |   | '_ \ / _` | __|" -ForegroundColor Cyan
    Write-Host " |  _ <  __/ |_| |  | (_) | |___| | | | (_| | |_" -ForegroundColor Cyan
    Write-Host " |_| \_\___|\__|_|   \___/ \____|_| |_|\__,_|\__|" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "           Universal One-File Installer" -ForegroundColor Yellow
    Write-Host "           ==========================" -ForegroundColor Yellow
    Write-Host ""
}

function Show-Menu {
    Show-Banner
    Write-Host "What would you like to do?" -ForegroundColor White
    Write-Host ""
    Write-Host "[1] Quick Install (recommended)" -ForegroundColor Green
    Write-Host "[2] Custom location" -ForegroundColor Yellow
    Write-Host "[3] Force reinstall" -ForegroundColor Red
    Write-Host "[4] Development branch" -ForegroundColor Magenta
    Write-Host "[5] Cancel" -ForegroundColor Gray
    Write-Host ""
    
    do {
        $choice = Read-Host "Choice (1-5)"
        switch ($choice) {
            "1" { return @{} }
            "2" { 
                $path = Read-Host "Installation path (Enter for default)"
                return if ($path) { @{ CustomPath = $path } } else { @{} }
            }
            "3" { return @{ Force = $true } }
            "4" { return @{ Branch = "development" } }
            "5" { Write-Host "Cancelled." -ForegroundColor Yellow; exit 0 }
            default { Write-Host "Invalid choice." -ForegroundColor Red }
        }
    } while ($true)
}

function Test-Command($cmd) {
    try { return [bool](Get-Command $cmd -ErrorAction Stop) }
    catch { return $false }
}

function Install-Git {
    if ($IsLinuxOrMac) { return $false }
    
    $install = Read-Host "[?] Install Git for better updates? (y/N)"
    if ($install -match '^[Yy]') {
        try {
            & winget install --id Git.Git --silent 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")
                return Test-Command "git"
            }
        } catch {}
    }
    return $false
}

function Install-Source {
    Write-Host "[*] Downloading RetroChat..." -ForegroundColor Yellow
    
    # Try Git first
    if (Test-Command "git") {
        try {
            if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
            & git clone --depth 1 --branch $Branch "https://github.com/$REPO_OWNER/$REPO_NAME.git" $SOURCE_DIR 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Downloaded with Git" -ForegroundColor Green
                return $true
            }
        } catch {}
    } elseif (-not $IsLinuxOrMac -and (Install-Git)) {
        return Install-Source  # Retry with Git
    }
    
    # Fallback to ZIP
    try {
        $zipUrl = "https://github.com/$REPO_OWNER/$REPO_NAME/archive/refs/heads/$Branch.zip"
        $zipPath = Join-Path ([System.IO.Path]::GetTempPath()) "retrochat.zip"
        $extractPath = Join-Path ([System.IO.Path]::GetTempPath()) "retrochat-extract"
        
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        
        if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
        Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
        
        if (Test-Path $SOURCE_DIR) { Remove-Item $SOURCE_DIR -Recurse -Force }
        New-Item -ItemType Directory -Path $SOURCE_DIR -Force | Out-Null
        
        $sourcePath = Join-Path $extractPath "$REPO_NAME-$Branch"
        Get-ChildItem $sourcePath | Move-Item -Destination $SOURCE_DIR -Force
        
        Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
        Remove-Item $extractPath -Recurse -Force -ErrorAction SilentlyContinue
        
        Write-Host "[OK] Downloaded with ZIP" -ForegroundColor Green
        return $true
        
    } catch {
        Write-Host "[ERROR] Download failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Install-Python {
    Write-Host "[*] Setting up Python environment..." -ForegroundColor Yellow
    
    if (-not (Test-Command $PYTHON_CMD)) {
        throw "Python 3.8+ is required. Please install Python and try again."
    }
    
    # Create virtual environment
    if (Test-Path $VENV_DIR) { Remove-Item $VENV_DIR -Recurse -Force }
    & $PYTHON_CMD -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment" }
    
    # Install packages with retry
    $maxRetries = 3
    for ($i = 1; $i -le $maxRetries; $i++) {
        try {
            $requirementsPath = Join-Path $SOURCE_DIR "requirements.txt"
            & $PYTHON_EXE -m pip install --upgrade pip --quiet
            & $PYTHON_EXE -m pip install -r $requirementsPath --quiet
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Python packages installed" -ForegroundColor Green
                return
            }
        } catch {}
        
        if ($i -lt $maxRetries) {
            Write-Host "[!] Retry $i..." -ForegroundColor Yellow
            Start-Sleep 2
        }
    }
    throw "Failed to install Python packages after $maxRetries attempts"
}

function Install-Launcher {
    Write-Host "[*] Creating launcher..." -ForegroundColor Yellow
    
    if (-not $IsLinuxOrMac) {
        # Windows batch launcher
        $launcherContent = @"
@echo off
"$PYTHON_EXE" "$SOURCE_DIR\retrochat.py" %*
"@
        $launcherPath = Join-Path $INSTALL_DIR "rchat.bat"
        $launcherContent | Out-File -FilePath $launcherPath -Encoding ASCII
        
        # Add to PATH
        $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($userPath -notlike "*$INSTALL_DIR*") {
            [Environment]::SetEnvironmentVariable("PATH", "$userPath;$INSTALL_DIR", "User")
        }
    } else {
        # Unix shell launcher
        $launcherContent = @"
#!/usr/bin/env bash
"$PYTHON_EXE" "$SOURCE_DIR/retrochat.py" `$@
"@
        $launcherPath = Join-Path $INSTALL_DIR "rchat"
        $launcherContent | Out-File -FilePath $launcherPath -Encoding UTF8
        & chmod +x $launcherPath
        
        # Add to shell profiles
        $pathLine = "export PATH=`"${INSTALL_DIR}:`$PATH`""
        foreach ($profileFile in @("~/.bashrc", "~/.zshrc", "~/.profile")) {
            $profilePath = $profileFile.Replace("~", $env:HOME)
            if (Test-Path $profilePath) {
                $content = Get-Content $profilePath -Raw -ErrorAction SilentlyContinue
                if ($content -notlike "*$INSTALL_DIR*") {
                    Add-Content $profilePath "`n# RetroChat v2`n$pathLine" -ErrorAction SilentlyContinue
                }
            }
        }
    }
    
    Write-Host "[OK] Launcher created" -ForegroundColor Green
}

# Main installation
try {
    if ($Interactive) {
        $params = Show-Menu
        foreach ($key in $params.Keys) {
            Set-Variable -Name $key -Value $params[$key] -Force
        }
    } else {
        Show-Banner
    }
    
    # Check existing installation
    if ((Test-Path $INSTALL_DIR) -and (-not $Force)) {
        Write-Host "[!] RetroChat already installed at: $INSTALL_DIR" -ForegroundColor Yellow
        $overwrite = Read-Host "Overwrite? (y/N)"
        if ($overwrite -notmatch '^[Yy]') {
            Write-Host "Installation cancelled." -ForegroundColor Yellow
            exit 0
        }
    }
    
    # Create installation directory
    New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
    
    # Install components
    if (-not (Install-Source)) { throw "Failed to download source code" }
    Install-Python
    Install-Launcher
    
    # Success message
    Write-Host ""
    Write-Host "[SUCCESS] RetroChat v2 installed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Location: $INSTALL_DIR" -ForegroundColor Cyan
    Write-Host "Command:  rchat" -ForegroundColor Cyan
    Write-Host "Branch:   $Branch" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not $IsLinuxOrMac) {
        Write-Host "Run 'rchat' from any command prompt!" -ForegroundColor Green
        Write-Host "Note: Restart terminal if command not found." -ForegroundColor Yellow
    } else {
        Write-Host "Run 'rchat' from any terminal!" -ForegroundColor Green
        Write-Host "Note: Restart terminal or run 'source ~/.bashrc'" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host ""
    Write-Host "[ERROR] Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Report issues: https://github.com/$REPO_OWNER/$REPO_NAME/issues" -ForegroundColor Blue
    exit 1
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
    
    # Try Git clone first (preferred for future updates)
    if (Test-Command "git") {
        $success = Install-FromGit
    } else {
        # Offer to install Git for better update experience
        if (Install-Git) {
            $success = Install-FromGit
        }
    }
    
    # Fallback to ZIP if Git failed
    if (-not $success) {
        Write-Host "[*] Falling back to ZIP download..." -ForegroundColor Yellow
        $success = Install-FromZip
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
