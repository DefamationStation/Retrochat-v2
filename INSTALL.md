# RetroChat v2 - Installation

## Quick Install (One Command)

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex"
```

**Linux/macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | pwsh -
```

That's it! This will:
- 📦 Download RetroChat to `~/.retrochat/`
- 🐍 Create a Python virtual environment  
- 📋 Install all dependencies
- 🔧 Set up the global `rchat` command
- ✅ Add to your system PATH

## Interactive Install

For more options, download and run the installer:

```bash
# Download
curl -o install.ps1 https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1

# Run with menu
pwsh ./install.ps1 -Interactive
```

**Menu Options:**
- Quick install (default)
- Custom installation path
- Force reinstall
- Install development branch
- Show system requirements

## Usage

After installation:
```bash
rchat
```

## Auto-Updates

RetroChat checks for updates on startup and shows:
- 📋 Recent commits with descriptions
- 🕒 Timestamps and authors  
- ❓ Update prompt (yes/no)

## System Requirements

- **Python 3.8+** (required)
- **Git** (optional, enables faster updates)
- **PowerShell** (Linux/macOS: install `powershell`)

## Troubleshooting

**Command not found after install:**
1. Restart your terminal
2. Or run: `~/.retrochat/rchat` (Linux/macOS) or `%USERPROFILE%\.retrochat\rchat.bat` (Windows)

**Manual installation:**
```bash
git clone https://github.com/DefamationStation/Retrochat-v2.git
cd Retrochat-v2
python retrochat.py
```
