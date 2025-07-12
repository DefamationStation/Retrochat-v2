import asyncio
import aiohttp
import hashlib
import requests
from utils.console import console
from utils.env_manager import EnvManager


class UpdateManager:
    def __init__(self):
        pass

    def display_update_message(self):
        updated = str(EnvManager.get_env_variable("UPDATED", "false")).lower() == "true"
        if updated:
            last_commit_hash = EnvManager.get_env_variable("LAST_COMMIT_HASH")
            missed_commits = self._get_missed_commits("DefamationStation", "Retrochat-v2", "retrochat.py", last_commit_hash)
            for i, commit_message in enumerate(missed_commits, 1):
                console.print(f"{i}. {commit_message}", style="yellow")
            EnvManager.set_env_variable("UPDATED", "false")

    async def check_for_updates(self):
        repo_owner = "DefamationStation"
        repo_name = "Retrochat-v2"
        file_path = "retrochat.py"

        EnvManager.load_env_variables()
        last_commit_hash = EnvManager.get_env_variable("LAST_COMMIT_HASH", "")

        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&page=1&per_page=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        latest_commit = (await response.json())[0]
                        latest_commit_hash = latest_commit['sha']

                        if latest_commit_hash == last_commit_hash:
                            console.print("You're running the latest version.", style="green")
                            return False

                        missed_commits = self._get_missed_commits(repo_owner, repo_name, file_path, last_commit_hash)

                        url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{latest_commit_hash}/{file_path}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                latest_content = await response.text()

                                # Get current file content
                                import sys
                                with open(sys.argv[0], 'r') as f:
                                    current_content = f.read()

                                if hashlib.sha256(current_content.encode()).hexdigest() != hashlib.sha256(latest_content.encode()).hexdigest():
                                    console.print("Updates are available:", style="bold yellow")
                                    for i, commit_message in enumerate(missed_commits, 1):
                                        console.print(f"{i}. {commit_message}", style="yellow")

                                    console.print("\nDo you want to update?\n\n1. Yes\n2. No")
                                    choice = console.ask("", choices=["1", "2"])

                                    if choice == "1":
                                        console.print("Updating...", style="cyan")
                                        import sys
                                        with open(sys.argv[0], 'w') as f:
                                            f.write(latest_content)
                                        EnvManager.set_env_variable("LAST_COMMIT_HASH", latest_commit_hash)
                                        EnvManager.set_env_variable("UPDATED", "true")
                                        console.print("Update complete. Please restart the script.", style="bold green")
                                        return True
                                    else:
                                        console.print("Update skipped. Running current version.", style="yellow")
                                        return False
                                else:
                                    console.print("You're running the latest version.", style="green")
                                    EnvManager.set_env_variable("LAST_COMMIT_HASH", latest_commit_hash)
                                    EnvManager.set_env_variable("UPDATED", "false")
                                    return False
                            else:
                                console.print(f"Failed to fetch the latest version: {response.status}", style="bold red")
                                return False
                    else:
                        console.print(f"Failed to check for updates: {response.status}", style="bold red")
                        return False
        except (aiohttp.ClientError, asyncio.TimeoutError):
            console.print("Unable to connect to GitHub. Skipping update check.", style="yellow")
            return False

    def _get_missed_commits(self, repo_owner, repo_name, file_path, last_commit_hash):
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}"
        response = requests.get(url)
        if response.status_code != 200:
            console.print(f"Failed to fetch commits: {response.status_code}", style="bold red")
            return []

        commits = response.json()
        missed_commits = []
        for commit in commits:
            if commit['sha'] == last_commit_hash:
                break
            missed_commits.append(commit['commit']['message'])
        
        return missed_commits