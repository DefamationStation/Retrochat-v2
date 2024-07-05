import os
import sys
import asyncio
import aiohttp
import sqlite3
import base64
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rich.console import Console
from rich.prompt import Prompt
from dotenv import load_dotenv, set_key

# Constants
USER_HOME = os.path.expanduser('~')
RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
DB_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.db')
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
LAST_CHAT_NAME_KEY = "LAST_CHAT_NAME"

os.makedirs(RETROCHAT_DIR, exist_ok=True)

# Initialize rich console
console = Console()

@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content  # Remove base64 encoding
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        return cls(
            role=data["role"],
            content=data["content"]  # Remove base64 decoding
        )

class ChatHistoryManager:
    def __init__(self, db_file: str, chat_name: str = 'default'):
        self.db_file = db_file
        self.chat_name = chat_name
        self.conn = sqlite3.connect(self.db_file)
        self._create_tables()
        self._update_schema()
        logging.basicConfig(level=logging.DEBUG)

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

class OllamaChatSession(ChatProvider):
    def __init__(self, model_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in self.chat_history],
            "stream": True
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
                    self.add_to_history("assistant", complete_message)
                    self.save_history()
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")

class AnthropicChatSession(ChatProvider):
    def __init__(self, api_key: str, model_url: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model_url = model_url

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 4096,
            "temperature": 0.0,
            "system": "Keep your answers short and to the point.",
            "messages": [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
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
                        self.add_to_history("assistant", assistant_message)
                        self.save_history()
                        console.print(assistant_message, style="yellow")
                    else:
                        console.print("No response content received.", style="bold red")
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")

class OpenAIChatSession(ChatProvider):
    def __init__(self, api_key: str, base_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in self.chat_history],
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
                    self.add_to_history("assistant", complete_message)
                    self.save_history()
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")

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
    def __init__(self, history_manager: ChatHistoryManager):
        self.history_manager = history_manager

    async def handle_command(self, command: str, session: ChatProvider):
        cmd_parts = command.split(maxsplit=2)
        if len(cmd_parts) < 2:
            self.display_help()
            return

        cmd, sub_cmd, *args = cmd_parts + ['']

        if cmd == '/chat':
            method_name = f"handle_{sub_cmd}"
            method = getattr(self, method_name, None)
            if method:
                await method(args[0], session)
            else:
                self.display_help()
        else:
            self.display_help()

    async def handle_rename(self, new_name: str, session: ChatProvider):
        if new_name:
            self.history_manager.rename_history(new_name)
            console.print(f"Chat renamed to '{new_name}'", style="cyan")
            self.save_last_chat_name(new_name)
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
            self.save_last_chat_name(new_name)
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
                self.save_last_chat_name(chat_name)
            else:
                console.print(f"Chat '{chat_name}' does not exist.", style="bold red")
        else:
            console.print("Please provide the name of the chat to open. Usage: /chat open <chat_name>", style="bold red")

    def save_last_chat_name(self, chat_name: str):
        set_key(ENV_FILE, LAST_CHAT_NAME_KEY, chat_name)

    def display_help(self):
        console.print("Available commands:", style="cyan")
        console.print("/chat rename <new_name> - Rename the current chat", style="green")
        console.print("/chat delete - Delete the current chat", style="green")
        console.print("/chat new <chat_name> - Create a new chat", style="green")
        console.print("/chat reset - Reset the current chat history", style="green")
        console.print("/chat list - List all available chats", style="green")
        console.print("/chat open <chat_name> - Open a specific chat", style="green")
        console.print("/help - Display this help message", style="green")
        console.print("/exit - Exit the program", style="green")

class ChatApp:
    def __init__(self):
        self.model_url_ollama = "http://192.168.1.82:11434/api/chat"
        self.model_url_anthropic = "https://api.anthropic.com/v1/messages"
        self.openai_base_url = "https://api.openai.com/v1/chat/completions"
        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(DB_FILE)
        self.command_handler = CommandHandler(self.history_manager)
        self.provider_factory = ChatProviderFactory()
        self.openai_api_key = None
        self.anthropic_api_key = None

        self.load_env_variables()
        self.save_last_chat_name(self.chat_name)

    def load_env_variables(self):
        if os.path.exists(ENV_FILE):
            load_dotenv(ENV_FILE)
            self.openai_api_key = os.getenv(OPENAI_API_KEY_NAME)
            self.anthropic_api_key = os.getenv(ANTHROPIC_API_KEY_NAME)
            self.chat_name = os.getenv(LAST_CHAT_NAME_KEY, 'default')
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

    async def select_ollama_model(self) -> str:
        url = "http://192.168.1.82:11434/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_info = await response.json()
                    if isinstance(models_info, dict) and 'models' in models_info:
                        model_names = [model['name'] for model in models_info['models']]
                        console.print("Available Ollama models:", style="cyan")
                        for idx, model in enumerate(model_names):
                            console.print(f"{idx + 1}. {model}", style="cyan")
                        choice = Prompt.ask("Select a model number", choices=[str(i) for i in range(1, len(model_names) + 1)])
                        return model_names[int(choice) - 1]
                    else:
                        console.print("Unexpected API response structure.", style="bold red")
                else:
                    console.print(f"Error fetching Ollama models: {response.status} - {await response.text()}", style="bold red")
        return None

    def save_last_chat_name(self, chat_name: str):
        set_key(ENV_FILE, LAST_CHAT_NAME_KEY, chat_name)

    async def start(self):
        try:
            console.clear()
            console.print("Welcome to Retrochat!", style="bold green")
            console.print("Select the mode:\n1. Ollama\n2. Anthropic\n3. OpenAI", style="cyan")

            mode = Prompt.ask("Enter your choice", choices=["1", "2", "3"])

            if mode == '1':
                selected_model = await self.select_ollama_model()
                if not selected_model:
                    return
                session = self.provider_factory.create_provider('Ollama', self.model_url_ollama, selected_model, self.history_manager)
            elif mode == '2':
                if not self.ensure_api_key('anthropic_api_key', ANTHROPIC_API_KEY_NAME):
                    return
                session = self.provider_factory.create_provider('Anthropic', self.anthropic_api_key, self.model_url_anthropic, self.history_manager)
            elif mode == '3':
                if not self.ensure_api_key('openai_api_key', OPENAI_API_KEY_NAME):
                    return
                session = self.provider_factory.create_provider('OpenAI', self.openai_api_key, self.openai_base_url, "gpt-4", self.history_manager)

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
    app = ChatApp()
    await app.start()

if __name__ == "__main__":
    asyncio.run(main())