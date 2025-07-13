# RetroChat v2 - Consolidated Installation

## Quick Start

**Single Command Install:**

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install-universal.ps1 | iex"

# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install-universal.ps1 | pwsh -
```

## Installation Options

### 1. Universal Installer (Recommended)
- **File**: `install-universal.ps1`
- **Platforms**: Windows, Linux, macOS
- **Features**: Cross-platform, interactive menu, Git detection, automatic fallbacks

```bash
# Download and run
./install-universal.ps1

# Interactive mode
./install-universal.ps1 -Interactive

# Force reinstall
./install-universal.ps1 -Force

# Custom location
./install-universal.ps1 -CustomPath "C:\MyApps\RetroChat"
```

### 2. Windows Batch Wrapper
- **File**: `install-universal.bat`
- **Platform**: Windows only
- **Features**: Simple double-click installation, menu options

Double-click `install-universal.bat` or run from command prompt.

### 3. Legacy Installers
- **Files**: `install.ps1`, `install.bat`
- **Status**: Maintained for compatibility
- **Recommendation**: Use universal installer instead

## File Comparison

| File | Platform | Features | Recommended |
|------|----------|----------|-------------|
| `install-universal.ps1` | All | Cross-platform, interactive, full-featured | ✅ **Yes** |
| `install-universal.bat` | Windows | Simple menu wrapper for universal installer | ✅ **Yes** |
| `install.ps1` | Windows | Original Windows installer | ⚠️ Legacy |
| `install.bat` | Windows | Simple batch wrapper | ⚠️ Legacy |

## Consolidation Benefits

1. **Single Source of Truth**: All logic in one universal file
2. **Cross-Platform**: Works on Windows, Linux, and macOS
3. **Interactive**: Optional menu-driven installation
4. **Smart Fallbacks**: Git → ZIP download, winget → chocolatey
5. **Consistent Experience**: Same features across all platforms
6. **Easy Maintenance**: One file to update instead of multiple

## Migration Plan

**Phase 1** (Current): Both systems coexist
- Universal installer is recommended
- Legacy installers remain functional

**Phase 2** (Future): Deprecate legacy files
- Update documentation to use universal installer
- Add deprecation warnings to legacy files

**Phase 3** (Later): Remove legacy files
- Keep only universal installer
- Redirect old URLs to new installer

## Usage Examples

```bash
# Quick install (one-liner)
powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install-universal.ps1 | iex"

# Download first, then run
curl -o install-universal.ps1 https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install-universal.ps1
pwsh ./install-universal.ps1 -Interactive

# Automated deployment
pwsh ./install-universal.ps1 -Force -CustomPath "/opt/retrochat"

# Development branch
pwsh ./install-universal.ps1 -Branch development
```
