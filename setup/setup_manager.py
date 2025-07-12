import os
import sys
import shutil
import platform
from config import Config
from utils.console import console


class SetupManager:
    def __init__(self):
        pass

    def check_and_setup(self):
        rchat_bat_path = os.path.join(Config.RETROCHAT_DIR, "rchat.bat")
        if not os.path.exists(rchat_bat_path) or not os.path.exists(Config.RETROCHAT_SCRIPT):
            console.print("RetroChat Setup", style="bold cyan")
            console.print("This setup will do the following:", style="cyan")
            console.print("1. Create a '.retrochat' folder in your home directory", style="cyan")
            console.print("2. Copy the RetroChat script to the '.retrochat' folder", style="cyan")
            console.print("3. Create an 'rchat.bat' file in the '.retrochat' folder", style="cyan")
            console.print("4. Add the '.retrochat' folder to your system PATH", style="cyan")
            console.print("\nThis will allow you to run RetroChat from anywhere using the 'rchat' command.", style="cyan")
            
            response = console.ask("Do you want to proceed with the setup?", choices=["yes", "no"])
            if response.lower() == "yes":
                self.setup_rchat()
            else:
                console.print("Setup cancelled. You can run the setup later by using the --setup flag.", style="yellow")

    def setup_rchat(self):
        os.makedirs(Config.RETROCHAT_DIR, exist_ok=True)
        
        current_script = sys.argv[0]
        shutil.copy2(current_script, Config.RETROCHAT_SCRIPT)
        console.print(f"Copied RetroChat script to {Config.RETROCHAT_SCRIPT}", style="cyan")

        # Also copy rchat.ps1 to the .retrochat directory
        rchat_ps1_src = os.path.join(os.path.dirname(current_script), "rchat.ps1")
        rchat_ps1_dst = os.path.join(Config.RETROCHAT_DIR, "rchat.ps1")
        if os.path.exists(rchat_ps1_src):
            shutil.copy2(rchat_ps1_src, rchat_ps1_dst)
            console.print(f"Copied rchat.ps1 to {rchat_ps1_dst}", style="cyan")
        else:
            console.print(f"Warning: rchat.ps1 not found at {rchat_ps1_src}. Batch launcher may not work.", style="yellow")
        
        if sys.platform.startswith('win'):
            rchat_bat_path = os.path.join(Config.RETROCHAT_DIR, "rchat.bat")
            rchat_ps1_path = os.path.join(os.path.dirname(Config.RETROCHAT_SCRIPT), "rchat.ps1")
            # The batch file will call the PowerShell script, passing all arguments
            with open(rchat_bat_path, "w") as f:
                f.write(f"@echo off\n"
                        f"powershell -ExecutionPolicy Bypass -File \"{rchat_ps1_path}\" %*\n")
            console.print(f"Created rchat.bat at {rchat_bat_path}", style="cyan")
        else:  # Mac or Linux
            rchat_sh_path = os.path.join(Config.RETROCHAT_DIR, "rchat")
            with open(rchat_sh_path, "w") as f:
                f.write(f'#!/bin/bash\npython3 "{Config.RETROCHAT_SCRIPT}" "$@"')
            os.chmod(rchat_sh_path, 0o755)  # Make the script executable
            console.print(f"Created rchat shell script at {rchat_sh_path}", style="cyan")
        
        if not os.path.exists(Config.ENV_FILE):
            with open(Config.ENV_FILE, "w") as f:
                f.write(f"{Config.ANTHROPIC_API_KEY_NAME}=\n")
                f.write(f"{Config.OPENAI_API_KEY_NAME}=\n")
                f.write(f"{Config.GOOGLE_API_KEY_NAME}=\n")
                f.write(f"{Config.OPENROUTER_API_KEY_NAME}=\n")
                f.write(f"{Config.LAST_CHAT_NAME_KEY}=default\n")
                f.write(f"{Config.OLLAMA_IP_KEY}=localhost\n")
                f.write(f"{Config.OLLAMA_PORT_KEY}=11434\n")
                f.write(f"{Config.LAST_PROVIDER_KEY}=\n")
                f.write(f"{Config.LAST_MODEL_KEY}=\n")
            console.print(f"Created .env file at {Config.ENV_FILE}", style="cyan")
        
        console.print("Setup complete. You can now use the 'rchat' command from anywhere.", style="green")
        
        if sys.platform.startswith('win'):
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
            try:
                path, _ = winreg.QueryValueEx(key, "Path")
                if Config.RETROCHAT_DIR not in path:
                    new_path = f"{path};{Config.RETROCHAT_DIR}"
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                    console.print(f"Added {Config.RETROCHAT_DIR} to PATH.", style="cyan")
                else:
                    console.print(f"{Config.RETROCHAT_DIR} is already in PATH.", style="cyan")
            except WindowsError:
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, Config.RETROCHAT_DIR)
                console.print(f"Created PATH and added {Config.RETROCHAT_DIR}.", style="cyan")
            finally:
                winreg.CloseKey(key)
        else:  # Mac or Linux
            shell = os.environ.get("SHELL", "").split("/")[-1]
            rc_file = f".{shell}rc" if shell in ['bash', 'zsh'] else ".profile"
            rc_path = os.path.join(Config.USER_HOME, rc_file)
            
            with open(rc_path, "a") as f:
                f.write(f'\nexport PATH="$PATH:{Config.RETROCHAT_DIR}"')
            
            console.print(f"Added {Config.RETROCHAT_DIR} to PATH in {rc_path}", style="cyan")
            console.print(f"Please run 'source ~/{rc_file}' or restart your terminal for the changes to take effect.", style="cyan")

        console.print("Setup complete. You can now use the 'rchat' command from anywhere.", style="green")