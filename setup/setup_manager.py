import os
import sys
from config import Config
from utils.console import console


class SetupManager:
    def __init__(self):
        pass

    def check_and_setup(self):
        """Check if RetroChat is properly set up and guide user if needed."""
        # Check if we're running from the installed location
        expected_installed_path = os.path.join(Config.RETROCHAT_DIR, "source")
        current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        is_installed = os.path.normpath(current_script_dir) == os.path.normpath(expected_installed_path)
        
        if is_installed:
            # We're running from installed location, all good
            return
            
        # We're running from development/manual clone
        console.print("🔧 Development Mode Detected", style="bold cyan")
        console.print("You're running RetroChat from a development directory.", style="cyan")
        console.print("For the best experience, consider using the installer:", style="cyan")
        console.print("")
        console.print("PowerShell command:", style="bold blue")
        console.print("powershell -ExecutionPolicy Bypass -Command \"iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex\"", style="green")
        console.print("")
        console.print("This will install RetroChat to ~/.retrochat/ with auto-updates.", style="cyan")
        console.print("Continuing with current setup...", style="yellow")
        console.print("")

    def setup_rchat(self):
        """Legacy setup method - now directs to installer."""
        console.print("🔧 Setup Method Changed", style="bold cyan")
        console.print("The setup process has been simplified!", style="cyan")
        console.print("")
        console.print("Please use the new one-command installer:", style="bold blue")
        console.print("powershell -ExecutionPolicy Bypass -Command \"iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex\"", style="green")
        console.print("")
        console.print("This new installer will:", style="cyan")
        console.print("• Download the latest version to ~/.retrochat/", style="cyan")
        console.print("• Set up Python environment automatically", style="cyan")
        console.print("• Create global 'rchat' command", style="cyan")
        console.print("• Enable automatic updates", style="cyan")
