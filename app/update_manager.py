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
                # No current commit info, show recent commits
                available_updates = remote_commits[:5]

            if not available_updates:
                console.print("[OK] RetroChat is up to date!", style="green")
                return False

            # Display available updates
            console.print(f"[UPDATE] {len(available_updates)} update(s) available!", style="bold cyan")
            console.print("\nRecent changes:", style="cyan")
            
            for i, commit in enumerate(available_updates, 1):
                date_obj = datetime.fromisoformat(commit["date"].replace('Z', '+00:00'))
                formatted_date = date_obj.strftime("%m/%d %H:%M")
                console.print(f"  {i}. [{commit['sha']}] {commit['message']}", style="yellow")
                console.print(f"     By {commit['author']} on {formatted_date}", style="dim")

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
            
            if self._has_git() and self._is_git_repo():
                return await self._update_with_git()
            else:
                return await self._update_with_zip()

        except Exception as e:
            console.print(f"Error during update: {str(e)}", style="bold red")
            return False

    async def _update_with_git(self) -> bool:
        """Update using git pull."""
        try:
            console.print("[*] Updating via Git...", style="yellow")
            
            # Get current commit before update
            old_commit = self.get_current_commit() or "unknown"

            # Pull latest changes
            result = subprocess.run(['git', 'pull'], cwd=self.current_dir, 
                                  capture_output=True, text=True)
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
            with tempfile.TemporaryDirectory() as temp_dir:
                shutil.unpack_archive(temp_zip_path, temp_dir)
                
                # Find extracted directory
                extracted_items = os.listdir(temp_dir)
                source_dir = os.path.join(temp_dir, extracted_items[0])
                
                # Backup current installation
                backup_dir = self.current_dir + "_backup"
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                
                if os.path.exists(self.current_dir):
                    shutil.move(self.current_dir, backup_dir)
                
                # Move new files
                shutil.move(source_dir, self.current_dir)
                
                # Remove backup on success
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)

            os.unlink(temp_zip_path)
            
            await self._post_update_setup("zip_update", "zip_update")
            return True

        except Exception as e:
            console.print(f"ZIP update failed: {str(e)}", style="bold red")
            return False

    async def _post_update_setup(self, old_commit: str, new_commit: str):
        """Perform post-update setup tasks."""
        # Update Python packages
        console.print("[*] Updating Python packages...", style="yellow")
        console.print("[*] Please wait, this may take a moment...", style="dim")
        
        venv_python = os.path.join(self.current_dir, "venv", "Scripts", "python.exe")
        requirements_file = os.path.join(self.current_dir, "requirements.txt")
        
        if os.path.exists(venv_python) and os.path.exists(requirements_file):
            subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file, "--quiet"],
                         cwd=self.current_dir)

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