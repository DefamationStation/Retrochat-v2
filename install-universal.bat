@echo off
REM RetroChat v2 - Universal Installer
REM This script works on any Windows system and automatically handles installation

title RetroChat v2 - Universal Installer
echo.
echo  ____      _             ____ _           _   
echo ^|  _ \ ___^| ^|_ _ __ ___  / ___^| ^|__   __ _^| ^|_ 
echo ^| ^|_) / _ \ __^| '__/ _ \^| ^|   ^| '_ \ / _` ^| __^|
echo ^|  _ ^<  __/ ^|_^| ^|  ^| ^(_) ^| ^|___^| ^| ^| ^| ^(_^| ^| ^|_ 
echo ^|_^| \_\___^|\__^|_^|   \___/ \____^|_^| ^|_^|\__,_^|\__^|
echo.
echo           Universal One-Click Installer
echo ==========================================
echo.

REM Check if PowerShell is available
powershell -Command "exit 0" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is required but not available.
    echo Please install PowerShell or run this on Windows 7+ / Windows Server 2008 R2+
    pause
    exit /b 1
)

echo [*] Starting installation process...
echo [*] This will download and install RetroChat v2 to your system
echo.
echo What would you like to do?
echo [1] Install RetroChat (recommended)
echo [2] Install to custom location
echo [3] Force reinstall
echo [4] Cancel
echo.
set /p choice="Enter choice (1-4): "

set "ps_args=-ExecutionPolicy Bypass -File ""%~dp0install.ps1"""
if "%choice%"=="2" (
    set /p custom_path="Enter installation path (or press Enter for default): "
    if not "%custom_path%"=="" (
        set "ps_args=%ps_args% -CustomPath ""%custom_path%"""
    )
) else if "%choice%"=="3" (
    set "ps_args=%ps_args% -Force"
) else if "%choice%"=="4" (
    echo Installation cancelled.
    pause
    exit /b 0
) else if not "%choice%"=="1" (
    echo Invalid choice, using default installation...
)

echo.
echo [*] Running PowerShell installer...
powershell %ps_args%

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Installation completed!
    echo [INFO] You can now run 'rchat' from anywhere
    echo.
) else (
    echo.
    echo [ERROR] Installation failed with code %errorlevel%
    echo Please check the error messages above.
    echo.
)

echo Press any key to exit...
pause >nul
