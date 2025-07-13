@echo off
echo 🚀 RetroChat v2 - One-Click Installer
echo =====================================
echo.
echo This will install RetroChat v2 to your system.
echo.
pause

powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex"

echo.
echo Installation complete! You can now run 'rchat' from anywhere.
pause
