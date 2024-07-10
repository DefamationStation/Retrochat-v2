import os
import platform
import sys
import asyncio
import aiohttp
import requests
import sqlite3
import base64
import json
import logging
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from abc import ABC, abstractmethod
from rich.console import Console
from rich.prompt import Prompt
from dotenv import load_dotenv, set_key
import shutil

# Constants
USER_HOME = os.path.expanduser('~')
RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
DB_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.db')
SETTINGS_FILE = os.path.join(RETROCHAT_DIR, 'settings.json')
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
LAST_CHAT_NAME_KEY = "LAST_CHAT_NAME"
OLLAMA_IP_KEY = "OLLAMA_IP"
OLLAMA_PORT_KEY = "OLLAMA_PORT"
RETROCHAT_SCRIPT = os.path.join(RETROCHAT_DIR, 'retrochat.py')

os.makedirs(RETROCHAT_DIR, exist_ok=True)

# Initialize rich console
console = Console()

# Self-setup functionality
def setup_rchat():
    os.makedirs(RETROCHAT_DIR, exist_ok=True)
    
    # Copy the current script to RETROCHAT_DIR
    current_script = sys.argv[0]
    shutil.copy2(current_script, RETROCHAT_SCRIPT)
    console.print(f"Copied RetroChat script to {RETROCHAT_SCRIPT}", style="cyan")
    
    # Create rchat.bat
    rchat_bat_path = os.path.join(RETROCHAT_DIR, "rchat.bat")
    with open(rchat_bat_path, "w") as f:
        f.write(f'@echo off\npython "{RETROCHAT_SCRIPT}" %*')
    console.print(f"Created rchat.bat at {rchat_bat_path}", style="cyan")
    
    # Create .env file
    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w") as f:
            f.write(f"{ANTHROPIC_API_KEY_NAME}=\n")
            f.write(f"{OPENAI_API_KEY_NAME}=\n")
            f.write(f"{LAST_CHAT_NAME_KEY}=default\n")
            f.write(f"{OLLAMA_IP_KEY}=localhost\n")
            f.write(f"{OLLAMA_PORT_KEY}=11434\n")
        console.print(f"Created .env file at {ENV_FILE}", style="cyan")
    
    # Add RETROCHAT_DIR to PATH
    if sys.platform.startswith('win'):
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
        try:
            path, _ = winreg.QueryValueEx(key, "Path")
            if RETROCHAT_DIR not in path:
                new_path = f"{path};{RETROCHAT_DIR}"
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                console.print(f"Added {RETROCHAT_DIR} to PATH.", style="cyan")
            else:
                console.print(f"{RETROCHAT_DIR} is already in PATH.", style="cyan")
        except WindowsError:
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, RETROCHAT_DIR)
            console.print(f"Created PATH and added {RETROCHAT_DIR}.", style="cyan")
        finally:
            winreg.CloseKey(key)
    else:
        # For Unix-like systems
        shell = os.environ.get("SHELL", "").split("/")[-1]
        rc_file = f".{shell}rc"
        rc_path = os.path.join(USER_HOME, rc_file)
        
        with open(rc_path, "a") as f:
            f.write(f'\nexport PATH="$PATH:{RETROCHAT_DIR}"')
        
        console.print(f"Added {RETROCHAT_DIR} to PATH in {rc_path}", style="cyan")
        console.print(f"Please run 'source ~/{rc_file}' or restart your terminal for the changes to take effect.", style="cyan")

    console.print("Setup complete. You can now use the 'rchat' command from anywhere.", style="green")

def check_and_setup():
    rchat_bat_path = os.path.join(RETROCHAT_DIR, "rchat.bat")
    if not os.path.exists(rchat_bat_path) or not os.path.exists(RETROCHAT_SCRIPT):
        console.print("RetroChat Setup", style="bold cyan")
        console.print("This setup will do the following:", style="cyan")
        console.print("1. Create a '.retrochat' folder in your home directory", style="cyan")
        console.print("2. Copy the RetroChat script to the '.retrochat' folder", style="cyan")
        console.print("3. Create an 'rchat.bat' file in the '.retrochat' folder", style="cyan")
        console.print("4. Add the '.retrochat' folder to your system PATH", style="cyan")
        console.print("\nThis will allow you to run RetroChat from anywhere using the 'rchat' command.", style="cyan")
        
        response = Prompt.ask("Do you want to proceed with the setup?", choices=["yes", "no"])
        if response.lower() == "yes":
            setup_rchat()
        else:
            console.print("Setup cancelled. You can run the setup later by using the --setup flag.", style="yellow")

@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        return cls(
            role=data["role"],
            content=data["content"]
        )

class ChatHistoryManager:
    def __init__(self, db_file: str, chat_name: str = 'default'):
        self.db_file = db_file
        self.chat_name = chat_name
        self.conn = sqlite3.connect(self.db_file)
        self._create_tables()
        self._update_schema()

    def _create_tables(self):
        with self.conn:
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_name TEXT NOT NULL UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                );
            ''')

    def _update_schema(self):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA table_info(chat_messages)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'role' not in columns:
                    self.conn.execute('ALTER TABLE chat_messages ADD COLUMN role TEXT NOT NULL DEFAULT "user"')
                    logging.info("Database schema updated successfully.")
        except Exception as e:
            logging.error(f"Error updating database schema: {e}")

    def _get_session_id(self, chat_name: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM chat_sessions WHERE chat_name = ?', (chat_name,))
        session = cursor.fetchone()
        if session:
            return session[0]
        cursor.execute('INSERT INTO chat_sessions (chat_name) VALUES (?)', (chat_name,))
        return cursor.lastrowid

    def set_chat_name(self, chat_name: str):
        self.chat_name = chat_name

    def save_history(self, history: List[ChatMessage]):
        session_id = self._get_session_id(self.chat_name)
        try:
            with self.conn:
                self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
                self.conn.executemany('''
                    INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)
                ''', [(session_id, msg.role, msg.content) for msg in history])
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")

    def load_history(self) -> List[ChatMessage]:
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
        try:
            return [ChatMessage(role=row[0], content=row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Error loading chat history: {e}")
            return []

    def clear_history(self):
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))

    def rename_history(self, new_name: str):
        session_id = self._get_session_id(self.chat_name)
        try:
            with self.conn:
                self.conn.execute('UPDATE chat_sessions SET chat_name = ? WHERE id = ?', (new_name, session_id))
            self.set_chat_name(new_name)
        except sqlite3.IntegrityError:
            console.print(f"Error: A chat with the name '{new_name}' already exists.", style="bold red")

    def delete_history(self):
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
            self.conn.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        self.set_chat_name('default')

    def list_chats(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT chat_name FROM chat_sessions')
        return [row[0] for row in cursor.fetchall()]

class ChatProvider(ABC):
    def __init__(self, history_manager: ChatHistoryManager):
        self.history_manager = history_manager
        self.chat_history = self.load_history()
        self.system_message = None

    def load_history(self) -> List[ChatMessage]:
        try:
            return self.history_manager.load_history()
        except Exception as e:
            logging.error(f"Error loading chat history: {e}")
            return []

    @abstractmethod
    async def send_message(self, message: str):
        pass

    def add_to_history(self, role: str, content: str):
        self.chat_history.append(ChatMessage(role, content))
        self.save_history()

    def save_history(self):
        try:
            self.history_manager.save_history(self.chat_history)
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")

    def display_history(self):
        if not self.chat_history:
            console.print("No previous chat history.", style="cyan")
        else:
            console.print("Chat history loaded from previous session:", style="cyan")
            for entry in self.chat_history:
                if entry.role == "user":
                    console.print(entry.content, style="green")
                else:
                    console.print(entry.content, style="yellow")

    def set_system_message(self, message: str):
        self.system_message = message
        console.print(f"System message set for the current session: {message}", style="cyan")

    def format_message(self, message: str) -> str:
        message = message.strip()
        paragraphs = message.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        formatted_message = '\n\n'.join(paragraphs)
        return formatted_message

class OllamaChatSession(ChatProvider):
    def __init__(self, model_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model
        self.parameters = {}
        self.default_parameters = {
            "num_predict": 128,
            "top_k": 40,
            "top_p": 0.95,
            "temperature": 0.8,
            "repeat_penalty": 0.95,
            "repeat_last_n": 64,
            "num_ctx": 8192,
            "stop": None,
        }

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {k: v for k, v in self.parameters.items() if v is not None}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.model_url, json=data) as response:
                if response.status == 200:
                    complete_message = ""
                    async for line in response.content:
                        if line:
                            response_json = json.loads(line)
                            message_content = response_json.get('message', {}).get('content', '')
                            complete_message += message_content
                            console.print(message_content, end="", style="yellow")
                            if response_json.get('done', False):
                                break
                    console.print()
                    formatted_message = self.format_message(complete_message)
                    self.add_to_history("assistant", formatted_message)
                    self.save_history()
                    return formatted_message
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")
                    return None

    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters:
            if param in ["num_predict", "top_k", "repeat_last_n", "num_ctx"]:
                value = int(value)
            elif param in ["top_p", "temperature", "repeat_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split()  # Split into a list
            self.parameters[param] = value
            console.print(f"Parameter '{param}' set to {value}", style="cyan")
        else:
            console.print(f"Invalid parameter: {param}", style="bold red")

    def show_parameters(self):
        console.print("Current Parameters:", style="cyan")
        for param, default_value in self.default_parameters.items():
            current_value = self.parameters.get(param, default_value)
            console.print(f"{param}: {current_value}", style="green")
        console.print(f"system: {self.system_message if self.system_message else 'Not set'}", style="green")

    def get_parameter_description(self, param: str) -> str:
        descriptions = {
            "num_predict": "Max number of tokens to predict",
            "top_k": "Pick from top k num of tokens",
            "top_p": "Nucleus sampling probability threshold",
            "temperature": "Temperature for sampling",
            "repeat_penalty": "Repetition penalty for sampling",
            "repeat_last_n": "Last n tokens to consider for repetition penalty",
            "num_ctx": "Context window size",
            "stop": "Stop sequences for text generation",
            "system": "System message for chat",
        }
        return descriptions.get(param, "")

class AnthropicChatSession(ChatProvider):
    def __init__(self, api_key: str, model_url: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model_url = model_url

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 4096,
            "temperature": 0.0,
            "messages": messages
        }
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.model_url, json=data, headers=headers) as response:
                if response.status == 200:
                    response_json = await response.json()
                    assistant_message = response_json.get('content', [{}])[0].get('text', '')
                    if assistant_message:
                        formatted_message = self.format_message(assistant_message)
                        self.add_to_history("assistant", formatted_message)
                        self.save_history()
                        console.print(formatted_message, style="yellow")
                        return formatted_message
                    else:
                        console.print("No response content received.", style="bold red")
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")
                return None

class OpenAIChatSession(ChatProvider):
    def __init__(self, api_key: str, base_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    complete_message = ""
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: "):
                                if line == "data: [DONE]":
                                    break
                                json_str = line[6:]
                                try:
                                    response_json = json.loads(json_str)
                                    content = response_json['choices'][0]['delta'].get('content', '')
                                    if content:
                                        complete_message += content
                                        console.print(content, end="", style="yellow")
                                except json.JSONDecodeError:
                                    continue
                    console.print()
                    formatted_message = self.format_message(complete_message)
                    self.add_to_history("assistant", formatted_message)
                    self.save_history()
                    return formatted_message
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")
                    return None

class ChatProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, *args, **kwargs) -> ChatProvider:
        providers = {
            'Ollama': OllamaChatSession,
            'Anthropic': AnthropicChatSession,
            'OpenAI': OpenAIChatSession
        }
        provider_class = providers.get(provider_type)
        if provider_class:
            return provider_class(*args, **kwargs)
        raise ValueError(f"Unsupported provider type: {provider_type}")

class CommandHandler:
    def __init__(self, history_manager: ChatHistoryManager, chat_app):
        self.history_manager = history_manager
        self.chat_app = chat_app

    async def handle_command(self, command: str, session: ChatProvider):
        cmd_parts = command.split(maxsplit=2)
        cmd, *args = cmd_parts + ['', '']

        if cmd == '/chat':
            method_name = f"handle_{args[0]}"
            method = getattr(self, method_name, None)
            if method:
                await method(args[1], session)
            else:
                self.display_help()
        elif cmd == '/set':
            if args[0] == 'system':
                self.handle_set_system(args[1])
            elif isinstance(session, OllamaChatSession):
                if not args[0]:
                    session.show_parameters()
                else:
                    self.handle_set(args[0], args[1], session)
            else:
                console.print("The /set command is only available for Ollama sessions, except for /set system.", style="bold red")
        elif cmd == '/edit':
            await self.chat_app.edit_conversation(session)
        elif cmd == '/help':
            self.display_help()
        else:
            console.print("Unknown command. Type /help for available commands.", style="bold red")

    def handle_set(self, param: str, value: str, session: OllamaChatSession):
        if not param:
            session.show_parameters()
        else:
            session.set_parameter(param, value)

    def handle_set_system(self, message: str):
        self.chat_app.set_global_system_message(message)
        console.print(f"Global system message set to: {message}", style="cyan")
        if self.chat_app.current_session:
            self.chat_app.current_session.set_system_message(message)
            console.print("System message updated for the current session.", style="cyan")

    async def handle_rename(self, new_name: str, session: ChatProvider):
        if new_name:
            self.history_manager.rename_history(new_name)
            console.print(f"Chat renamed to '{new_name}'", style="cyan")
            self.chat_app.save_last_chat_name(new_name)
        else:
            console.print("Please provide a new name for the chat. Usage: /chat rename <new_name>", style="bold red")

    async def handle_delete(self, _, session: ChatProvider):
        self.history_manager.delete_history()
        console.print("Current chat history deleted.", style="cyan")

    async def handle_new(self, new_name: str, session: ChatProvider):
        if new_name:
            self.history_manager.set_chat_name(new_name)
            self.history_manager.save_history([])
            console.print(f"New chat '{new_name}' created.", style="cyan")
            self.chat_app.save_last_chat_name(new_name)
        else:
            console.print("Please provide a name for the new chat. Usage: /chat new <chat_name>", style="bold red")

    async def handle_reset(self, _, session: ChatProvider):
        self.history_manager.clear_history()
        session.chat_history = []
        console.print("Chat history has been reset.", style="cyan")

    async def handle_list(self, _, session: ChatProvider):
        chats = self.history_manager.list_chats()
        if chats:
            console.print("Available chats:", style="cyan")
            for chat in chats:
                console.print(chat, style="green")
        else:
            console.print("No available chats.", style="bold red")

    async def handle_open(self, chat_name: str, session: ChatProvider):
        if chat_name:
            if chat_name in self.history_manager.list_chats():
                self.history_manager.set_chat_name(chat_name)
                session.chat_history = self.history_manager.load_history()
                console.print(f"Chat '{chat_name}' opened.", style="cyan")
                session.display_history()
                self.chat_app.save_last_chat_name(chat_name)
            else:
                console.print(f"Chat '{chat_name}' does not exist.", style="bold red")
        else:
            console.print("Please provide the name of the chat to open. Usage: /chat open <chat_name>", style="bold red")

    def display_help(self):
        console.print("Available commands:", style="cyan")
        console.print("/chat rename <new_name> - Rename the current chat", style="green")
        console.print("/chat delete - Delete the current chat", style="green")
        console.print("/chat new <chat_name> - Create a new chat", style="green")
        console.print("/chat reset - Reset the current chat history", style="green")
        console.print("/chat list - List all available chats", style="green")
        console.print("/chat open <chat_name> - Open a specific chat", style="green")
        console.print("/set system <message> - Set the global system message", style="green")
        console.print("/set - Show available parameters and their current values (Ollama only)", style="green")
        console.print("/set <parameter> <value> - Set a parameter (Ollama only)", style="green")
        console.print("/edit - Edit the entire conversation", style="green")
        console.print("/help - Display this help message", style="green")
        console.print("/exit - Exit the program", style="green")

class ChatApp:
    def __init__(self):
        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(DB_FILE)
        self.command_handler = CommandHandler(self.history_manager, self)
        self.provider_factory = ChatProviderFactory()
        self.openai_api_key = None
        self.anthropic_api_key = None
        self.global_system_message = None
        self.ollama_ip = None
        self.ollama_port = None

        self.load_env_variables()
        self.save_last_chat_name(self.chat_name)

    def load_env_variables(self):
        if os.path.exists(ENV_FILE):
            load_dotenv(ENV_FILE)
            self.openai_api_key = os.getenv(OPENAI_API_KEY_NAME)
            self.anthropic_api_key = os.getenv(ANTHROPIC_API_KEY_NAME)
            self.chat_name = os.getenv(LAST_CHAT_NAME_KEY, 'default')
            self.ollama_ip = os.getenv(OLLAMA_IP_KEY, 'localhost')
            self.ollama_port = os.getenv(OLLAMA_PORT_KEY, '11434')
            self.history_manager.set_chat_name(self.chat_name)

    def ensure_api_key(self, key_name: str, env_var: str):
        if not getattr(self, key_name):
            console.print(f"{env_var} is not set. Please enter your API key.", style="cyan")
            api_key = Prompt.ask(f"Enter your {env_var}")
            if api_key:
                set_key(ENV_FILE, env_var, api_key)
                load_dotenv(ENV_FILE)
                setattr(self, key_name, api_key)
                console.print(f"{env_var} has been set and saved in the .env file.", style="cyan")
                return True
            else:
                console.print(f"No API key provided. {key_name.replace('_', ' ').title()} mode cannot be used.", style="bold red")
                return False
        return True

    def ensure_ollama_connection(self):
        url = f"http://{self.ollama_ip}:{self.ollama_port}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return True
        except requests.RequestException:
            console.print(f"Unable to connect to Ollama at {self.ollama_ip}:{self.ollama_port}", style="bold red")
            new_ip = Prompt.ask("Enter Ollama IP (press Enter for localhost)")
            new_port = Prompt.ask("Enter Ollama port (press Enter for 11434)")
            
            self.ollama_ip = new_ip or 'localhost'
            self.ollama_port = new_port or '11434'
            
            set_key(ENV_FILE, OLLAMA_IP_KEY, self.ollama_ip)
            set_key(ENV_FILE, OLLAMA_PORT_KEY, self.ollama_port)
            load_dotenv(ENV_FILE)
            
            console.print(f"Ollama connection details updated and saved in the .env file.", style="cyan")
            return self.ensure_ollama_connection()

    async def select_ollama_model(self) -> str:
        url = f"http://{self.ollama_ip}:{self.ollama_port}/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_info = await response.json()
                    if isinstance(models_info, dict) and 'models' in models_info:
                        model_names = [model['name'] for model in models_info['models']]
                        console.print("Available Ollama models:", style="cyan")
                        for idx, model in enumerate(model_names):
                            console.print(f"{idx + 1}. {model}", style="green")
                        choice = Prompt.ask("Select a model number", choices=[str(i) for i in range(1, len(model_names) + 1)])
                        return model_names[int(choice) - 1]
                    else:
                        console.print("Unexpected API response structure.", style="bold red")
                else:
                    console.print(f"Error fetching Ollama models: {response.status} - {await response.text()}", style="bold red")
        return None

    def save_last_chat_name(self, chat_name: str):
        set_key(ENV_FILE, LAST_CHAT_NAME_KEY, chat_name)

    def set_global_system_message(self, message: str):
        self.global_system_message = message
        if self.current_session:
            self.current_session.set_system_message(message)

    async def edit_conversation(self, session: ChatProvider):
        # Convert chat history to a string
        chat_text = ""
        for msg in session.chat_history:
            chat_text += f"{msg.role.upper()}:\n{msg.content}\n\n"

        # Create a temporary file with the chat history
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
            temp_file.write(chat_text)
            temp_file_path = temp_file.name

        # Determine the appropriate editor command
        if platform.system() == 'Windows':
            editor_cmd = ['notepad.exe', temp_file_path]
        else:
            editor = os.environ.get('EDITOR', 'nano')
            editor_cmd = [editor, temp_file_path]

        # Open the text editor
        try:
            subprocess.run(editor_cmd, check=True)
        except subprocess.CalledProcessError:
            console.print(f"Error: Unable to open the default editor.", style="bold red")
            console.print("You can manually edit the file at:", style="cyan")
            console.print(temp_file_path, style="yellow")
            console.print("After editing, press Enter to continue.", style="cyan")
            input()

        # Read the edited content
        with open(temp_file_path, 'r') as file:
            edited_content = file.read()

        # Remove the temporary file
        os.unlink(temp_file_path)

        # Parse the edited content back into chat history
        new_history = []
        current_role = None
        current_content = []

        for line in edited_content.split('\n'):
            line = line.strip()
            if line.upper() in ['USER:', 'ASSISTANT:']:
                if current_role is not None:
                    new_history.append(ChatMessage(role=current_role, content='\n'.join(current_content).strip()))
                current_role = line[:-1].lower()
                current_content = []
            elif line:
                current_content.append(line)

        if current_role is not None:
            new_history.append(ChatMessage(role=current_role, content='\n'.join(current_content).strip()))

        # Update the session's chat history
        session.chat_history = new_history
        session.save_history()
        console.print("Chat history updated successfully.", style="cyan")
        session.display_history()

    async def start(self):
        try:
            console.clear()
            console.print("Welcome to Retrochat!", style="bold green")
            
            # Check for setup
            check_and_setup()
            
            console.print("Select the mode:\n1. Ollama\n2. Anthropic\n3. OpenAI", style="cyan")

            mode = Prompt.ask("Enter your choice", choices=["1", "2", "3"])

            if mode == '1':
                if not self.ensure_ollama_connection():
                    return
                selected_model = await self.select_ollama_model()
                if not selected_model:
                    return
                model_url = f"http://{self.ollama_ip}:{self.ollama_port}/api/chat"
                session = self.provider_factory.create_provider('Ollama', model_url, selected_model, self.history_manager)
            elif mode == '2':
                if not self.ensure_api_key('anthropic_api_key', ANTHROPIC_API_KEY_NAME):
                    return
                session = self.provider_factory.create_provider('Anthropic', self.anthropic_api_key, self.model_url_anthropic, self.history_manager)
            elif mode == '3':
                if not self.ensure_api_key('openai_api_key', OPENAI_API_KEY_NAME):
                    return
                session = self.provider_factory.create_provider('OpenAI', self.openai_api_key, self.openai_base_url, "gpt-4", self.history_manager)

            session.set_system_message(self.global_system_message)
            session.display_history()

            while True:
                try:
                    user_input = await self.get_multiline_input()

                    if user_input.lower() == '/exit':
                        console.print("Thank you for chatting. Goodbye!", style="cyan")
                        break
                    elif user_input.startswith('/'):
                        await self.command_handler.handle_command(user_input, session)
                    elif user_input:  # Only send non-empty messages
                        await session.send_message(user_input)
                        session.save_history()  # Save history after each message
                except KeyboardInterrupt:
                    continue  # Allow Ctrl+C to clear the current input
                except EOFError:
                    break  # Exit on Ctrl+D

        except Exception as e:
            console.print(f"\nAn unexpected error occurred: {e}", style="bold red")
        finally:
            session.save_history()  # Ensure history is saved when exiting

    async def get_multiline_input(self) -> str:
        lines = []
        prompt_session = PromptSession()
        
        while True:
            if not lines:
                prompt = "> "
            else:
                prompt = "... "
            
            with patch_stdout():
                line = await prompt_session.prompt_async(prompt, multiline=False)
            
            if not line and lines:  # Empty line finishes input if there's already content
                break
            elif line.endswith('...'):
                lines.append(line[:-3])  # Remove the '...' and add to lines
            else:
                lines.append(line)
                if not line.endswith('...'):
                    break
        
        return '\n'.join(lines)

async def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_rchat()
    else:
        app = ChatApp()
        await app.start()

if __name__ == "__main__":
    asyncio.run(main())