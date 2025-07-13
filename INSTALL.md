# RetroChat v2 - One-Command Installation

## Quick Install

Run this single command to install RetroChat v2:

```powershell
powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex"
```

That's it! This command will:

- 📦 Download the latest RetroChat to `~/.retrochat/`
- 🐍 Create a Python virtual environment
- 📋 Install all required dependencies
- 🔧 Set up the global `rchat` command
- ✅ Add to your system PATH

## Usage

After installation, simply run:

```bash
rchat
```

From anywhere on your system!

## Auto-Updates

RetroChat automatically checks for updates every time you start it. You'll see:

- 📋 List of recent commits/changes
- 🕒 Timestamps and authors
- ❓ Option to update or skip

Updates are seamless and maintain your settings.

## Features

- **Zero Dependencies**: Installer handles everything
- **Self-Contained**: Everything lives in `~/.retrochat/`
- **Auto-Updates**: Always get the latest features
- **Smart Fallbacks**: Works with or without Git
- **Clean Installation**: No system pollution

## Troubleshooting

If `rchat` command isn't found after installation:
1. Restart your terminal/command prompt
2. Or run the full path: `~/.retrochat/rchat.bat`

## Manual Installation

If you prefer manual setup:
1. Clone this repository
2. Run `python retrochat.py` from the project directory

---

*RetroChat v2 - Modern AI chat with automatic updates*
