import os
import sys
import subprocess
import asyncio
import json
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Optional
import requests
from utils.console import console
from utils.env_manager import EnvManager
from config import Config


class UpdateManager:
    def __init__(self):
        self.repo_owner = "DefamationStation"
        self.repo_name = "Retrochat-v2"
        self.github_api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        self.repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}.git"
        
        # Check if we're running from installed location
        self.is_installed = self._check_if_installed()
        if self.is_installed:
            self.current_dir = os.path.join(Config.RETROCHAT_DIR, "source")
        else:
            self.current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _check_if_installed(self) -> bool:
        """Check if we're running from the installed location in ~/.retrochat/"""
        expected_installed_path = os.path.join(Config.RETROCHAT_DIR, "source")
        # update_manager.py is in app/ subdirectory, so go up one level to get to source/
        current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(current_script_dir) == os.path.normpath(expected_installed_path)

    def _has_git(self) -> bool:
        """Check if git is available."""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        return os.path.exists(os.path.join(self.current_dir, '.git'))

    async def get_remote_commits(self, limit: int = 10) -> List[Dict]:
        """Get recent commits from GitHub API."""
        try:
            response = requests.get(f"{self.github_api_url}/commits", 
                                  params={"per_page": limit}, timeout=10)
            response.raise_for_status()
            
            commits = []
            for commit_data in response.json():
                commit_info = {
                    "sha": commit_data["sha"][:8],
                    "full_sha": commit_data["sha"],
                    "message": commit_data["commit"]["message"].split('\n')[0],
                    "author": commit_data["commit"]["author"]["name"],
                    "date": commit_data["commit"]["author"]["date"],
                    "url": commit_data["html_url"]
                }
                commits.append(commit_info)
            
            return commits
        except Exception as e:
            console.print(f"Failed to fetch remote commits: {str(e)}", style="yellow")
            return []

    def get_current_commit(self) -> Optional[str]:
        """Get current commit hash."""
        if not self._is_git_repo():
            return None
            
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  cwd=self.current_dir, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    async def check_for_updates(self) -> bool:
        """Check for updates and handle the update process."""
        try:
            console.print("[*] Checking for updates...", style="cyan")
            
            # Get remote commits
            remote_commits = await self.get_remote_commits(15)
            if not remote_commits:
                console.print("Unable to check for updates. Continuing...", style="yellow")
                return False

            current_commit = self.get_current_commit()
            
            # Find updates
            available_updates = []
            if current_commit:
                current_found = False
                for commit in remote_commits:
                    if commit["full_sha"].startswith(current_commit):
                        current_found = True
                        break
                    available_updates.append(commit)
                
                if not current_found:
                    # Current commit not in recent history, assume updates available
                    available_updates = remote_commits[:5]
            else:
                # No git repo - check if this is a fresh install
                # If installed via ZIP, we likely have the latest version
                if self.is_installed:
                    console.print("[OK] RetroChat is up to date! (installed from latest source)", style="green")
                    return False
                else:
                    # Development/manual clone without git - show recent commits
                    available_updates = remote_commits[:3]  # Show fewer since it's probably recent

            if not available_updates:
                console.print("[OK] RetroChat is up to date!", style="green")
                return False

            # Display available updates
            console.print(f"[UPDATE] {len(available_updates)} update(s) available!", style="bold cyan")
            console.print("\nRecent changes:", style="cyan")
            
            for i, commit in enumerate(available_updates, 1):
                try:
                    date_obj = datetime.fromisoformat(commit["date"].replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime("%m/%d %H:%M")
                    # Ensure we have a SHA, use truncated version or 'unknown'
                    sha_display = commit.get('sha', 'unknown')
                    if not sha_display or sha_display == 'unknown':
                        sha_display = commit.get('full_sha', 'unknown')[:8] if commit.get('full_sha') else 'unknown'
                    
                    console.print(f"  {i}. [{sha_display}] {commit['message']}", style="yellow")
                    console.print(f"     By {commit['author']} on {formatted_date}", style="dim")
                except Exception as e:
                    # Fallback for malformed commit data
                    console.print(f"  {i}. {commit.get('message', 'Unknown commit')}", style="yellow")
                    console.print(f"     By {commit.get('author', 'Unknown')} on {commit.get('date', 'Unknown date')}", style="dim")

            # Ask user if they want to update
            console.print("")
            response = console.ask("Do you want to update now?", choices=["yes", "no"])
            
            if response.lower() == "yes":
                return await self.perform_update()
            else:
                console.print("Update skipped. You can update later by restarting the application.", style="yellow")
                return False

        except Exception as e:
            console.print(f"Error checking for updates: {str(e)}", style="yellow")
            return False

    async def perform_update(self) -> bool:
        """Perform the actual update."""
        try:
            console.print("[*] Updating RetroChat...", style="cyan")
            
            # Always prefer git if available
            if self._has_git() and self._is_git_repo():
                console.print("[*] Using Git for fast update...", style="dim")
                return await self._update_with_git()
            else:
                console.print("[*] Using ZIP download method...", style="dim")
                return await self._update_with_zip()

        except Exception as e:
            console.print(f"Error during update: {str(e)}", style="bold red")
            return False

    async def _update_with_git(self) -> bool:
        """Update using git pull."""
        try:
            console.print("[*] Fetching latest changes...", style="yellow")
            
            # Get current commit before update
            old_commit = self.get_current_commit() or "unknown"

            # Ensure we're running git from repo root, not venv
            # Fetch first to check for changes
            result = subprocess.run(['git', 'fetch'], cwd=self.current_dir, 
                                  capture_output=True, text=True, env=dict(os.environ))
            if result.returncode != 0:
                console.print(f"Git fetch failed: {result.stderr}", style="yellow")
                console.print("Falling back to ZIP download...", style="yellow")
                return await self._update_with_zip()

            # Check if there are actually new commits
            result = subprocess.run(['git', 'rev-list', 'HEAD..origin/main', '--count'], 
                                  cwd=self.current_dir, capture_output=True, text=True, env=dict(os.environ))
            if result.returncode == 0:
                commit_count = int(result.stdout.strip())
                if commit_count == 0:
                    console.print("[OK] Already up to date via Git!", style="green")
                    return True

            # Pull latest changes - run from repo root, not venv
            console.print("[*] Pulling latest changes...", style="yellow")
            result = subprocess.run(['git', 'pull'], cwd=self.current_dir, 
                                  capture_output=True, text=True, env=dict(os.environ))
            if result.returncode != 0:
                console.print(f"Git pull failed: {result.stderr}", style="bold red")
                console.print("Falling back to ZIP download...", style="yellow")
                return await self._update_with_zip()

            # Get new commit hash
            new_commit = self.get_current_commit() or "unknown"

            # Update environment and requirements
            await self._post_update_setup(old_commit, new_commit)
            return True

        except Exception as e:
            console.print(f"Git update failed: {str(e)}", style="yellow")
            console.print("Falling back to ZIP download...", style="yellow")
            return await self._update_with_zip()

    async def _update_with_zip(self) -> bool:
        """Update by downloading ZIP from GitHub."""
        try:
            console.print("[*] Downloading latest version...", style="yellow")
            console.print("[*] Please wait, this may take a moment...", style="dim")
            
            # Download ZIP
            zip_url = f"https://github.com/{self.repo_owner}/{self.repo_name}/archive/refs/heads/main.zip"
            response = requests.get(zip_url, timeout=30)
            response.raise_for_status()

            # Save and extract
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_zip_path = temp_file.name

            console.print("[*] Extracting files...", style="yellow")
            
            # Create a delayed update script since we can't overwrite running files
            update_script_path = os.path.join(Config.RETROCHAT_DIR, "update_script.ps1")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                shutil.unpack_archive(temp_zip_path, temp_dir)
                
                # Find extracted directory
                extracted_items = os.listdir(temp_dir)
                source_dir = os.path.join(temp_dir, extracted_items[0])
                
                # Create update script that will run after we exit
                update_script_content = f"""
# RetroChat Update Script - Auto-generated
Start-Sleep -Seconds 3

$sourceDir = "{self.current_dir}"
$backupDir = $sourceDir + "_backup"
$newDir = "{source_dir}"

# Backup current installation
if (Test-Path $backupDir) {{
    Remove-Item $backupDir -Recurse -Force
}}

if (Test-Path $sourceDir) {{
    Move-Item $sourceDir $backupDir
}}

# Move new files
Move-Item $newDir $sourceDir

# Update Python packages
$pythonExe = Join-Path $sourceDir "venv\\Scripts\\python.exe"
$requirementsFile = Join-Path $sourceDir "requirements.txt"

if ((Test-Path $pythonExe) -and (Test-Path $requirementsFile)) {{
    Set-Location $sourceDir
    & $pythonExe -m pip install -r $requirementsFile --quiet
}}

# Clean up
Remove-Item $backupDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "{temp_zip_path}" -Force -ErrorAction SilentlyContinue
Remove-Item $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue

Write-Host "[OK] Update completed! RetroChat has been updated to the latest version." -ForegroundColor Green
Write-Host "You can now run 'rchat' again." -ForegroundColor Cyan
"""
                
                with open(update_script_path, 'w', encoding='utf-8') as f:
                    f.write(update_script_content)

            os.unlink(temp_zip_path)
            
            console.print("[OK] Update prepared. Restarting to complete update...", style="green")
            
            # Start the update script and exit
            subprocess.Popen(['powershell', '-ExecutionPolicy', 'Bypass', '-File', update_script_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
            
            # Save update info and exit
            EnvManager.set_env_variable("UPDATED", "true")
            console.print("Exiting to complete update...", style="cyan")
            sys.exit(0)

        except Exception as e:
            console.print(f"ZIP update failed: {str(e)}", style="bold red")
            return False

    async def _post_update_setup(self, old_commit: str, new_commit: str):
        """Perform post-update setup tasks."""
        # Update Python packages (this needs venv)
        console.print("[*] Updating Python packages...", style="yellow")
        console.print("[*] Please wait, this may take a moment...", style="dim")
        
        venv_python = os.path.join(self.current_dir, "venv", "Scripts", "python.exe")
        requirements_file = os.path.join(self.current_dir, "requirements.txt")
        
        if os.path.exists(venv_python) and os.path.exists(requirements_file):
            # Run pip install from the repo root, but using venv python
            result = subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file, "--quiet"],
                         cwd=self.current_dir)
            if result.returncode == 0:
                console.print("[OK] Python packages updated", style="green")
            else:
                console.print("[WARNING] Some packages may not have updated properly", style="yellow")

        # Save update information
        EnvManager.set_env_variable("LAST_COMMIT_HASH", new_commit)
        EnvManager.set_env_variable("UPDATED", "true")

        console.print("[OK] Update completed successfully!", style="bold green")
        console.print("Please restart RetroChat to use the updated version.", style="cyan")
        
        # Exit to force restart
        sys.exit(0)

    def display_update_message(self):
        """Display message if app was recently updated."""
        updated = str(EnvManager.get_env_variable("UPDATED", "false")).lower() == "true"
        if updated:
            console.print("[SUCCESS] RetroChat has been updated to the latest version!", style="bold green")
            EnvManager.set_env_variable("UPDATED", "false")