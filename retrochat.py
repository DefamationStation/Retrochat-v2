import os
import sys
import asyncio
import aiohttp
import requests
import sqlite3
import json
import logging
import tempfile
import subprocess
import hashlib
import platform
import google.generativeai as genai
import shutil
import tiktoken
import warnings
import contextlib
import io
from google.api_core import client_options as client_options_lib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from prompt_toolkit import PromptSession
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.prompt import Prompt
from dotenv import load_dotenv, set_key
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings, OllamaEmbeddings

class SuppressLogging:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)

USER_HOME = os.path.expanduser('~')
RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
DB_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.db')
SETTINGS_FILE = os.path.join(RETROCHAT_DIR, 'settings.json')
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
GOOGLE_API_KEY_NAME = "GOOGLE_API_KEY"
LAST_CHAT_NAME_KEY = "LAST_CHAT_NAME"
OLLAMA_IP_KEY = "OLLAMA_IP"
OLLAMA_PORT_KEY = "OLLAMA_PORT"
RETROCHAT_SCRIPT = os.path.join(RETROCHAT_DIR, 'retrochat.py')
LAST_PROVIDER_KEY = "LAST_PROVIDER"
LAST_MODEL_KEY = "LAST_MODEL"
CHROMA_PATH = os.path.join(RETROCHAT_DIR, "chroma")

os.makedirs(RETROCHAT_DIR, exist_ok=True)

console = Console()

tokenizer = tiktoken.get_encoding("cl100k_base")

def check_and_fix_env_file():
    required_keys = [
        ANTHROPIC_API_KEY_NAME,
        OPENAI_API_KEY_NAME,
        GOOGLE_API_KEY_NAME,
        LAST_CHAT_NAME_KEY,
        OLLAMA_IP_KEY,
        OLLAMA_PORT_KEY,
        LAST_PROVIDER_KEY,
        LAST_MODEL_KEY
    ]
    
    if not os.path.exists(ENV_FILE):
        setup_rchat()
        return

    with open(ENV_FILE, 'r') as f:
        env_contents = f.read()

    missing_keys = [key for key in required_keys if key not in env_contents]

    if missing_keys:
        console.print("Updating .env file with missing keys...", style="cyan")
        with open(ENV_FILE, 'a') as f:
            for key in missing_keys:
                f.write(f"{key}=\n")
        console.print(".env file updated.", style="green")

def get_embedding_function():
    load_dotenv(ENV_FILE)
    ollama_ip = os.getenv(OLLAMA_IP_KEY, 'localhost')
    ollama_port = os.getenv(OLLAMA_PORT_KEY, '11434')
    return OllamaEmbeddings(
        base_url=f"http://{ollama_ip}:{ollama_port}",
        model="nomic-embed-text"
    )

def setup_rchat():
    os.makedirs(RETROCHAT_DIR, exist_ok=True)
    
    current_script = sys.argv[0]
    shutil.copy2(current_script, RETROCHAT_SCRIPT)
    console.print(f"Copied RetroChat script to {RETROCHAT_SCRIPT}", style="cyan")
    
    if sys.platform.startswith('win'):
        rchat_bat_path = os.path.join(RETROCHAT_DIR, "rchat.bat")
        with open(rchat_bat_path, "w") as f:
            f.write(f'@echo off\npython "{RETROCHAT_SCRIPT}" %*')
        console.print(f"Created rchat.bat at {rchat_bat_path}", style="cyan")
    else:  # Mac or Linux
        rchat_sh_path = os.path.join(RETROCHAT_DIR, "rchat")
        with open(rchat_sh_path, "w") as f:
            f.write(f'#!/bin/bash\npython3 "{RETROCHAT_SCRIPT}" "$@"')
        os.chmod(rchat_sh_path, 0o755)  # Make the script executable
        console.print(f"Created rchat shell script at {rchat_sh_path}", style="cyan")
    
    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w") as f:
            f.write(f"{ANTHROPIC_API_KEY_NAME}=\n")
            f.write(f"{OPENAI_API_KEY_NAME}=\n")
            f.write(f"{GOOGLE_API_KEY_NAME}=\n")
            f.write(f"{LAST_CHAT_NAME_KEY}=default\n")
            f.write(f"{OLLAMA_IP_KEY}=localhost\n")
            f.write(f"{OLLAMA_PORT_KEY}=11434\n")
            f.write(f"{LAST_PROVIDER_KEY}=\n")
            f.write(f"{LAST_MODEL_KEY}=\n")
        console.print(f"Created .env file at {ENV_FILE}", style="cyan")
    
    console.print("Setup complete. You can now use the 'rchat' command from anywhere.", style="green")
    
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
    else:  # Mac or Linux
        shell = os.environ.get("SHELL", "").split("/")[-1]
        rc_file = f".{shell}rc" if shell in ['bash', 'zsh'] else ".profile"
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
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    system_message TEXT,
                    parameters TEXT
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
                cursor.execute("PRAGMA table_info(chat_sessions)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'system_message' not in columns:
                    self.conn.execute('ALTER TABLE chat_sessions ADD COLUMN system_message TEXT')
                if 'parameters' not in columns:
                    self.conn.execute('ALTER TABLE chat_sessions ADD COLUMN parameters TEXT')
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

    def save_system_message(self, system_message: str):
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('UPDATE chat_sessions SET system_message = ? WHERE id = ?', (system_message, session_id))

    def load_system_message(self) -> Optional[str]:
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT system_message FROM chat_sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def save_parameters(self, parameters: Dict[str, Any]):
        session_id = self._get_session_id(self.chat_name)
        parameters_json = json.dumps(parameters)
        with self.conn:
            self.conn.execute('UPDATE chat_sessions SET parameters = ? WHERE id = ?', (parameters_json, session_id))

    def load_parameters(self) -> Dict[str, Any]:
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT parameters FROM chat_sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        return json.loads(result[0]) if result and result[0] else {}

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
        self.system_message = self.history_manager.load_system_message()
        self.parameters = self.history_manager.load_parameters()
        self.default_parameters = {
            "temperature": 0.8,
            "max_tokens": 8192,
            "verbose": False,
            "frequency_penalty": 1.1,
            "repeat_penalty": 1.1,
        }

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
        if self.parameters.get("verbose", False) and role == "user":
            tokens = self.calculate_tokens(content)
            total_tokens = self.calculate_total_tokens()
            console.print(f"Message tokens: {tokens}", style="cyan")
            console.print(f"Total conversation tokens: {total_tokens}", style="cyan")

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
        self.history_manager.save_system_message(message)

    def format_message(self, message: str) -> str:
        message = message.strip()
        paragraphs = message.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        formatted_message = '\n\n'.join(paragraphs)
        return formatted_message

    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters or param in ["repeat_penalty", "frequency_penalty"]:
            if param in ["num_predict", "top_k", "repeat_last_n", "num_ctx", "candidate_count", "max_tokens"]:
                value = int(value)
            elif param in ["top_p", "temperature", "repeat_penalty", "frequency_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split() if isinstance(value, str) else value
            elif param == "verbose":
                value = str(value).lower() == "true"
            
            if param in ["repeat_penalty", "frequency_penalty"]:
                self.parameters["repeat_penalty"] = value
                self.parameters["frequency_penalty"] = value
            else:
                self.parameters[param] = value
            
            self.history_manager.save_parameters(self.parameters)
            
            if param != "verbose" or value:
                console.print(f"Parameter '{param}' set to {value}", style="cyan")
                if param == "max_tokens":
                    console.print("(This will be sent as max_output_tokens to the Gemini API)", style="yellow")
        else:
            console.print(f"Invalid parameter: {param}", style="bold red")

    def show_parameters(self):
        console.print("Current Parameters:", style="cyan")
        for param, default_value in self.default_parameters.items():
            current_value = self.parameters.get(param, default_value)
            if param == "max_tokens":
                console.print(f"max_tokens: {current_value} (sent as max_output_tokens to Gemini API)", style="green")
            else:
                console.print(f"{param}: {current_value}", style="green")
        
        if "frequency_penalty" not in self.default_parameters:
            console.print(f"frequency_penalty: {self.parameters.get('frequency_penalty', self.parameters.get('repeat_penalty', 1.1))}", style="green")
        
        if self.system_message:
            console.print(f"system: {self.system_message}", style="green")

    def calculate_tokens(self, text: str) -> int:
        return len(tokenizer.encode(text))

    def calculate_total_tokens(self) -> int:
        total_tokens = 0
        if self.system_message:
            total_tokens += self.calculate_tokens(self.system_message)
        for msg in self.chat_history:
            total_tokens += self.calculate_tokens(msg.content)
        return total_tokens

class OllamaChatSession(ChatProvider):
    def __init__(self, model_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model
        self.default_parameters.update({
            "num_predict": 128,
            "top_k": 40,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "num_ctx": 8192,
            "stop": None,
        })
    
    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters or param in ["repeat_penalty", "frequency_penalty"]:
            if param in ["num_predict", "top_k", "repeat_last_n", "num_ctx"]:
                value = int(value)
            elif param in ["top_p", "temperature", "repeat_penalty", "frequency_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split() if isinstance(value, str) else value
            elif param == "verbose":
                value = str(value).lower() == "true"
            
            if param in ["repeat_penalty", "frequency_penalty"]:
                self.parameters["repeat_penalty"] = value
                self.parameters["frequency_penalty"] = value
            else:
                self.parameters[param] = value
            
            self.history_manager.save_parameters(self.parameters)
            
            if param != "verbose" or value:
                console.print(f"Parameter '{param}' set to {value}", style="cyan")
        else:
            console.print(f"Invalid parameter: {param}", style="bold red")

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
                    if self.parameters.get("verbose", False):
                        tokens = self.calculate_tokens(formatted_message)
                        total_tokens = self.calculate_total_tokens()
                        console.print(f"Response tokens: {tokens}", style="cyan")
                        console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
                    return formatted_message
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")
                    return None

    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters or param == "repeat_penalty":
            if param in ["num_predict", "top_k", "repeat_last_n", "num_ctx"]:
                value = int(value)
            elif param in ["top_p", "temperature", "repeat_penalty", "frequency_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split() if isinstance(value, str) else value
            elif param == "verbose":
                value = str(value).lower() == "true"
            
            if param == "repeat_penalty" and isinstance(self, OpenAIChatSession):
                param = "frequency_penalty"
            
            self.parameters[param] = value
            self.history_manager.save_parameters(self.parameters)
            
            if param != "verbose" or value:
                console.print(f"Parameter '{param}' set to {value}", style="cyan")
        else:
            console.print(f"Invalid parameter: {param}", style="bold red")

class AnthropicChatSession(ChatProvider):
    def __init__(self, api_key: str, model_url: str, history_manager: ChatHistoryManager, model: str):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model_url = model_url
        self.model = model

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = self.prepare_messages()
        
        data = {
            "model": self.model,
            "max_tokens": self.parameters.get("max_tokens", 8192),
            "temperature": self.parameters.get("temperature", 0.8),
            "messages": messages
        }

        if self.system_message:
            data["system"] = self.system_message

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.model_url, json=data, headers=headers) as response:
                if response.status == 200:
                    response_json = await response.json()
                    assistant_message = response_json.get('content', [{}])[0].get('text', '')
                    if assistant_message:
                        formatted_message = self.format_message(assistant_message)
                        self.add_to_history("assistant", formatted_message)
                        console.print(formatted_message, style="yellow")
                        if self.parameters.get("verbose", False):
                            tokens = self.calculate_tokens(formatted_message)
                            total_tokens = self.calculate_total_tokens()
                            console.print(f"Response tokens: {tokens}", style="cyan")
                            console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
                        return formatted_message
                    else:
                        console.print("No response content received.", style="bold red")
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")
                return None

    def prepare_messages(self):
        messages = []
        for msg in self.chat_history:
            if msg.role != "system":
                if not messages or messages[-1]["role"] != msg.role:
                    messages.append({"role": msg.role, "content": msg.content})
                else:
                    messages[-1]["content"] += "\n" + msg.content
        return messages

class OpenAIChatSession(ChatProvider):
    def __init__(self, api_key: str, base_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_parameters.update({
            "frequency_penalty": 1.1,
        })

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.parameters.get("temperature", 0.8),
            "max_tokens": self.parameters.get("max_tokens", 8192),
            "frequency_penalty": self.parameters.get("frequency_penalty", 0.0),
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
                    if self.parameters.get("verbose", False):
                        tokens = self.calculate_tokens(formatted_message)
                        total_tokens = self.calculate_total_tokens()
                        console.print(f"Response tokens: {tokens}", style="cyan")
                        console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
                    return formatted_message
                else:
                    console.print(f"Error: {response.status} - {await response.text()}", style="bold red")
                    return None
                
class GoogleChatSession(ChatProvider):
    def __init__(self, api_key: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model = model
        
        # Suppress warnings and logging
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("google.generativeai").setLevel(logging.CRITICAL)
        os.environ["GRPC_PYTHON_LOG_LEVEL"] = "error"
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        
        # Configure the Google AI library
        with SuppressLogging():
            client_options = client_options_lib.ClientOptions(
                api_endpoint="generativelanguage.googleapis.com"
            )
            genai.configure(api_key=self.api_key, client_options=client_options)
            
            self.genai_model = genai.GenerativeModel(self.model)
            self.chat = self.genai_model.start_chat(history=[])

        self.default_parameters.update({
            "candidate_count": 1,
            "max_tokens": 8192,
            "temperature": 0.8,
        })

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        
        generation_config = genai.types.GenerationConfig(
            candidate_count=self.parameters.get("candidate_count", 1),
            max_output_tokens=self.parameters.get("max_tokens", 8192),
            temperature=self.parameters.get("temperature", 0.8),
        )

        with SuppressLogging():
            response = await asyncio.to_thread(
                self.chat.send_message,
                message,
                generation_config=generation_config,
                stream=True
            )

            complete_message = ""
            for chunk in response:
                chunk_text = chunk.text
                complete_message += chunk_text
                console.print(chunk_text, end="", style="yellow")
                await asyncio.sleep(0)  # Yield control to allow for responsiveness

        formatted_message = self.format_message(complete_message)
        self.add_to_history("assistant", formatted_message)
        
        if self.parameters.get("verbose", False):
            tokens = self.calculate_tokens(formatted_message)
            total_tokens = self.calculate_total_tokens()
            console.print(f"Response tokens: {tokens}", style="cyan")
            console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
        
        return formatted_message

    def set_system_message(self, message: str):
        super().set_system_message(message)
        self.chat = self.genai_model.start_chat(history=[{"role": "system", "parts": [message]}])

    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters:
            if param in ["candidate_count", "max_tokens", "top_k"]:
                value = int(value)
            elif param in ["temperature", "top_p"]:
                value = float(value)
            elif param == "verbose":
                value = str(value).lower() == "true"
            
            self.parameters[param] = value
            self.history_manager.save_parameters(self.parameters)
            
            if param != "verbose" or value:
                console.print(f"Parameter '{param}' set to {value}", style="cyan")
        else:
            console.print(f"Invalid parameter: {param}", style="bold red")

    def show_parameters(self):
        console.print("Current Parameters:", style="cyan")
        for param, default_value in self.default_parameters.items():
            current_value = self.parameters.get(param, default_value)
            if param == "max_tokens":
                console.print(f"max_tokens (max_output_tokens): {current_value}", style="green")
            else:
                console.print(f"{param}: {current_value}", style="green")
        
        if self.system_message:
            console.print(f"system: {self.system_message}", style="green")

class ChatProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, *args, **kwargs) -> ChatProvider:
        providers = {
            'Ollama': OllamaChatSession,
            'Anthropic': AnthropicChatSession,
            'OpenAI': OpenAIChatSession,
            'Google': GoogleChatSession
        }
        provider_class = providers.get(provider_type)
        if provider_class:
            return provider_class(*args, **kwargs)
        raise ValueError(f"Unsupported provider type: {provider_type}")

class CommandHandler:
    def __init__(self, history_manager: ChatHistoryManager, chat_app):
        self.history_manager = history_manager
        self.chat_app = chat_app

    async def handle_show_length(self, session: ChatProvider):
        total_tokens = session.calculate_total_tokens()
        console.print(f"Total conversation tokens: {total_tokens}", style="cyan")

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
                self.handle_set_system(args[1], session)
            elif not args[0]:
                session.show_parameters()
            else:
                self.handle_set(args[0], args[1], session)
        elif cmd == '/edit':
            try:
                await self.chat_app.edit_conversation(session)
            except Exception as e:
                console.print(f"An error occurred while editing the conversation: {str(e)}", style="bold red")
                console.print("Your original conversation has not been modified.", style="yellow")
        elif cmd == '/show' and args[0] == 'length':
            await self.handle_show_length(session)
        elif cmd == '/show' and args[0] == 'context':
            await self.chat_app.handle_show_context()
        elif cmd == '/switch':
            return await self.handle_switch(args[0], session)
        elif cmd == '/help':
            self.display_help()
        else:
            console.print("Unknown command. Type /help for available commands.", style="bold red")

    def handle_set(self, param: str, value: str, session: ChatProvider):
        if not param:
            session.show_parameters()
        else:
            session.set_parameter(param, value)

    def handle_set_system(self, message: str, session: ChatProvider):
        session.set_system_message(message)
        console.print(f"System message set to: {message}", style="cyan")

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
                session.system_message = self.history_manager.load_system_message()
                session.parameters = self.history_manager.load_parameters()
                console.print(f"Chat '{chat_name}' opened.", style="cyan")
                session.display_history()
                self.chat_app.save_last_chat_name(chat_name)
            else:
                console.print(f"Chat '{chat_name}' does not exist.", style="bold red")
        else:
            console.print("Please provide the name of the chat to open. Usage: /chat open <chat_name>", style="bold red")

    async def handle_switch(self, _, session: ChatProvider):
        new_session = await self.chat_app.switch_provider()
        if new_session:
            return new_session
        return session

    def display_help(self):
        console.print("Available commands:", style="cyan")
        console.print("/chat rename <new_name> - Rename the current chat", style="green")
        console.print("/chat delete - Delete the current chat", style="green")
        console.print("/chat new <chat_name> - Create a new chat", style="green")
        console.print("/chat reset - Reset the current chat history", style="green")
        console.print("/chat list - List all available chats", style="green")
        console.print("/chat open <chat_name> - Open a specific chat", style="green")
        console.print("/set system <message> - Set the system message", style="green")
        console.print("/set - Show available parameters and their current values", style="green")
        console.print("/set <parameter> <value> - Set a parameter", style="green")
        console.print("/edit - Edit the entire conversation", style="green")
        console.print("/show length - Display the total conversation tokens", style="green")
        console.print("/show context - Display the context of the last query", style="green")
        console.print("/switch - Switch to a different provider or model", style="green")
        console.print("/help - Display this help message", style="green")
        console.print("/exit - Exit the program", style="green")

class DocumentManager:
    def __init__(self):
        self.chroma_path = CHROMA_PATH
        self.embedding_function = get_embedding_function()

    def load_documents(self, folder_name: str) -> bool:
        data_path = os.path.join(RETROCHAT_DIR, folder_name)
        console.print(f"Attempting to load documents from: {data_path}", style="cyan")
        
        if not os.path.exists(data_path):
            console.print(f"Folder '{data_path}' does not exist.", style="bold red")
            return False
        
        try:
            documents = []
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if os.path.isfile(file_path):
                    loader = self.get_loader_for_file(file_path)
                    if loader:
                        documents.extend(loader.load())
            
            console.print(f"Loaded {len(documents)} documents", style="green")
            
            chunks = self.split_documents(documents)
            console.print(f"Split into {len(chunks)} chunks", style="green")
            
            self.add_to_chroma(chunks, folder_name)
            console.print("Added chunks to Chroma database", style="green")
            
            return True
        except Exception as e:
            console.print(f"Error loading documents: {str(e)}", style="bold red")
            return False

    def get_loader_for_file(self, file_path: str) -> BaseLoader:
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.pdf':
            return PyPDFDirectoryLoader(os.path.dirname(file_path))
        elif ext == '.txt':
            return TextLoader(file_path)
        elif ext in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            console.print(f"Unsupported file type: {file_path}", style="yellow")
            return None

    def split_documents(self, documents: List[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_to_chroma(self, chunks: List[Document], folder_name: str):
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)

        chunks_with_ids = self.calculate_chunk_ids(chunks, folder_name)

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if new_chunks:
            console.print(f"-> Adding new documents: {len(new_chunks)}", style="cyan")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            console.print("[OK] No new documents to add", style="green")

    def calculate_chunk_ids(self, chunks: List[Document], folder_name: str):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{folder_name}/{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks

    def query_documents(self, folder_name: str, query: str) -> List[Document]:
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        all_docs = db.get()
        
        folder_docs_ids = [doc_id for doc_id in all_docs['ids'] if doc_id.startswith(f"{folder_name}/")]
        
        if not folder_docs_ids:
            console.print(f"No documents found for folder '{folder_name}'", style="yellow")
            return []
        
        results = db.similarity_search(
            query,
            k=min(5, len(folder_docs_ids)),
            filter={"id": {"$in": folder_docs_ids}}
        )
        return results

class ChatApp:
    def __init__(self):
        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(DB_FILE)
        self.command_handler = CommandHandler(self.history_manager, self)
        self.provider_factory = ChatProviderFactory()
        self.openai_api_key = None
        self.anthropic_api_key = None
        self.google_api_key = None
        self.ollama_ip = None
        self.ollama_port = None
        self.current_session = None
        self.last_commit_hash = None
        self.updated = False
        self.last_provider = None
        self.last_model = None
        self.ENV_FILE = ENV_FILE
        self.document_manager = DocumentManager()

        check_and_fix_env_file()  # Add this line
        self.load_env_variables()
        self.load_last_chat()

    def display_non_default_parameters(self):
        non_default_params = {}
        for param, value in self.current_session.parameters.items():
            if param in self.current_session.default_parameters:
                if value != self.current_session.default_parameters[param]:
                    non_default_params[param] = value
            else:
                non_default_params[param] = value
        
        if non_default_params:
            for param, value in non_default_params.items():
                console.print(f"{param}: {value}", style="green")

    def load_env_variables(self):
        if os.path.exists(ENV_FILE):
            load_dotenv(ENV_FILE)
            self.openai_api_key = os.getenv(OPENAI_API_KEY_NAME)
            self.anthropic_api_key = os.getenv(ANTHROPIC_API_KEY_NAME)
            self.google_api_key = os.getenv(GOOGLE_API_KEY_NAME)
            self.chat_name = os.getenv(LAST_CHAT_NAME_KEY, 'default')
            self.ollama_ip = os.getenv(OLLAMA_IP_KEY, 'localhost')
            self.ollama_port = os.getenv(OLLAMA_PORT_KEY, '11434')
            self.last_commit_hash = os.getenv("LAST_COMMIT_HASH")
            self.updated = os.getenv("UPDATED", "false").lower() == "true"
            self.last_provider = os.getenv(LAST_PROVIDER_KEY)
            self.last_model = os.getenv(LAST_MODEL_KEY)
            self.history_manager.set_chat_name(self.chat_name)

            # Debug print
            console.print("Loaded environment variables:", style="cyan")
            console.print(f"Anthropic API key: {'*****' if self.anthropic_api_key else 'Not set'}", style="yellow")
            console.print(f"OpenAI API key: {'*****' if self.openai_api_key else 'Not set'}", style="yellow")
            console.print(f"Google API key: {'*****' if self.google_api_key else 'Not set'}", style="yellow")
            console.print(f"Chat name: {self.chat_name}", style="yellow")
            console.print(f"Ollama IP: {self.ollama_ip}", style="yellow")
            console.print(f"Ollama Port: {self.ollama_port}", style="yellow")
            console.print(f"Last provider: {self.last_provider}", style="yellow")
            console.print(f"Last model: {self.last_model}", style="yellow")

    def save_last_provider_and_model(self, provider: str, model: str):
        set_key(ENV_FILE, LAST_PROVIDER_KEY, provider)
        set_key(ENV_FILE, LAST_MODEL_KEY, model)
        self.last_provider = provider
        self.last_model = model

    def load_last_chat(self):
        self.history_manager.set_chat_name(self.chat_name)
        chat_history = self.history_manager.load_history()
        system_message = self.history_manager.load_system_message()
        parameters = self.history_manager.load_parameters()
        return chat_history, system_message, parameters
    
    def display_update_message(self):
        if self.updated:
            missed_commits = get_missed_commits("DefamationStation", "Retrochat-v2", "retrochat.py", self.last_commit_hash)
            for i, commit_message in enumerate(missed_commits, 1):
                console.print(f"{i}. {commit_message}", style="yellow")
            set_key(self.ENV_FILE, "UPDATED", "false")
            self.updated = False

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
            response = requests.get(url, timeout=5)
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
                        choice = Prompt.ask("Select a model")
                        return model_names[int(choice) - 1]
                    else:
                        console.print("Unexpected API response structure.", style="bold red")
                else:
                    console.print(f"Error fetching Ollama models: {response.status} - {await response.text()}", style="bold red")
        return None

    async def select_anthropic_model(self) -> str:
        models = ["claude-3-5-sonnet-20240620"]
        console.print("Available Anthropic models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = Prompt.ask("Select a model number")
        try:
            return models[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None

    async def select_openai_model(self) -> str:
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4o-64k-output-alpha"]
        console.print("Available OpenAI models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = Prompt.ask("Select a model number")
        return models[int(choice) - 1]
    
    async def select_google_model(self) -> str:
        models = ["gemini-1.5-flash", "gemini-1.5-pro-exp-0801"]
        console.print("Available Google Gemini models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = Prompt.ask("Select a model number")
        return models[int(choice) - 1]

    def save_last_chat_name(self, chat_name: str):
        set_key(ENV_FILE, LAST_CHAT_NAME_KEY, chat_name)

    async def edit_conversation(self, session: ChatProvider):
        chat_text = ""
        for msg in session.chat_history:
            chat_text += f"{msg.role.upper()}:\n{msg.content}\n\n"

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(chat_text)
            temp_file_path = temp_file.name

        if platform.system() == 'Windows':
            editor_cmd = ['notepad.exe', temp_file_path]
        elif platform.system() == 'Darwin':  # macOS
            editor = os.environ.get('EDITOR', 'open -t')
            editor_cmd = editor.split() + [temp_file_path]
        else:  # Linux or other Unix-like systems
            editor = os.environ.get('EDITOR', 'nano')
            editor_cmd = [editor, temp_file_path]

        try:
            subprocess.run(editor_cmd, check=True)
        except subprocess.CalledProcessError:
            console.print(f"Error: Unable to open the default editor.", style="bold red")
            console.print("You can manually edit the file at:", style="cyan")
            console.print(temp_file_path, style="yellow")
            console.print("After editing, press Enter to continue.", style="cyan")
            input()

        try:
            with open(temp_file_path, 'r', encoding='utf-8') as file:
                edited_content = file.read()
        except UnicodeDecodeError:
            with open(temp_file_path, 'r') as file:
                edited_content = file.read()

        os.unlink(temp_file_path)

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

        session.chat_history = new_history
        session.save_history()
        console.print("Chat history updated successfully.", style="cyan")
        session.display_history()

    async def switch_provider(self):
        console.print("Select provider:\n1. Ollama\n2. Anthropic\n3. OpenAI\n4. Google", style="cyan")
        mode = Prompt.ask("Enter your choice")

        if mode == '1':
            if not self.ensure_ollama_connection():
                return None
            selected_model = await self.select_ollama_model()
            if not selected_model:
                return None
            model_url = f"http://{self.ollama_ip}:{self.ollama_port}/api/chat"
            new_session = self.provider_factory.create_provider('Ollama', model_url, selected_model, self.history_manager)
            provider = 'Ollama'
        elif mode == '2':
            if not self.ensure_api_key('anthropic_api_key', ANTHROPIC_API_KEY_NAME):
                return None
            selected_model = await self.select_anthropic_model()
            if not selected_model:
                return None
            new_session = self.provider_factory.create_provider('Anthropic', self.anthropic_api_key,
    "https://api.anthropic.com/v1/messages", self.history_manager, selected_model)
            provider = 'Anthropic'
        elif mode == '3':
            if not self.ensure_api_key('openai_api_key', OPENAI_API_KEY_NAME):
                return None
            selected_model = await self.select_openai_model()
            if not selected_model:
                return None
            new_session = self.provider_factory.create_provider('OpenAI', self.openai_api_key,
    "https://api.openai.com/v1/chat/completions", selected_model, self.history_manager)
            provider = 'OpenAI'
        elif mode == '4':
            if not self.ensure_api_key('google_api_key', GOOGLE_API_KEY_NAME):
                return None
            selected_model = await self.select_google_model()
            if not selected_model:
                return None
            with SuppressLogging():
                new_session = self.provider_factory.create_provider('Google', self.google_api_key, selected_model, self.history_manager)
            provider = 'Google'
        else:
            console.print("Invalid choice. Please select 1, 2, 3, or 4.", style="bold red")
            return None

        if new_session:
            self.apply_saved_parameters(new_session)
            self.save_last_provider_and_model(provider, selected_model)
            return new_session
        return None
    
    async def create_session_from_last(self):
        new_session = None
        
        if self.last_provider == 'Google':
            if self.last_provider == 'Google':
                if not self.ensure_api_key('google_api_key', GOOGLE_API_KEY_NAME):
                    return None
                with SuppressLogging():
                    new_session = self.provider_factory.create_provider('Google', self.google_api_key, self.last_model, self.history_manager)
        elif self.last_provider == 'Ollama':
            if not self.ensure_ollama_connection():
                return None
            model_url = f"http://{self.ollama_ip}:{self.ollama_port}/api/chat"
            with contextlib.redirect_stderr(io.StringIO()):
                new_session = self.provider_factory.create_provider('Ollama', model_url, self.last_model, self.history_manager)
        elif self.last_provider == 'Anthropic':
            if not self.ensure_api_key('anthropic_api_key', ANTHROPIC_API_KEY_NAME):
                return None
            with contextlib.redirect_stderr(io.StringIO()):
                new_session = self.provider_factory.create_provider('Anthropic', self.anthropic_api_key, "https://api.anthropic.com/v1/messages", self.history_manager, self.last_model)
        elif self.last_provider == 'OpenAI':
            if not self.ensure_api_key('openai_api_key', OPENAI_API_KEY_NAME):
                return None
            with contextlib.redirect_stderr(io.StringIO()):
                new_session = self.provider_factory.create_provider('OpenAI', self.openai_api_key, "https://api.openai.com/v1/chat/completions", self.last_model, self.history_manager)

        if new_session:
            self.apply_saved_parameters(new_session)
        return new_session

    def apply_saved_parameters(self, session):
        saved_params = self.history_manager.load_parameters()
        for param, value in saved_params.items():
            session.parameters[param] = value
        if self.current_session:
            session.chat_history = self.current_session.chat_history
            session.system_message = self.current_session.system_message
        else:
            session.chat_history = []
            session.system_message = None

    async def handle_load_command(self, folder_name: str):
        console.print(f"RETROCHAT_DIR: {RETROCHAT_DIR}", style="cyan")
        success = self.document_manager.load_documents(folder_name)
        if success:
            console.print(f"Documents from '{folder_name}' loaded successfully.", style="green")
        else:
            console.print(f"Failed to load documents from '{folder_name}'.", style="bold red")

    async def handle_query_command(self, folder_name: str, query: str):
        results = self.document_manager.query_documents(folder_name, query)
        if not results:
            console.print(f"No results found for query in folder '{folder_name}'", style="yellow")
            return
        
        context = "\n\n".join([doc.page_content for doc in results])
        
        prompt = f"""
        Based on the following context, answer the question: {query}

        Context:
        {context}

        Answer:
        """

        response = await self.current_session.send_message(prompt)
        
        self.last_query_context = context
        self.last_query_prompt = prompt

        return response
    
    async def handle_show_context(self):
        if hasattr(self, 'last_query_context') and hasattr(self, 'last_query_prompt'):
            console.print("Last query context:", style="cyan")
            console.print(self.last_query_context, style="yellow")
            console.print("\nLast query prompt:", style="cyan")
            console.print(self.last_query_prompt, style="yellow")
        else:
            console.print("No context available. Please run a query first.", style="bold red")

    async def start(self):
        try:
            console.clear()
            console.print("Welcome to Retrochat! [bold green]v1.1.1[/bold green]", style="bold green")
            
            check_and_setup()
            
            # Perform update check after displaying welcome message
            if await self.check_for_updates():
                return

            self.display_update_message()

            self.current_session = await self.create_session_from_last()
            
            if not self.current_session:
                self.current_session = await self.switch_provider()
            
            if not self.current_session:
                return

            chat_history, system_message, parameters = self.load_last_chat()
            self.current_session.chat_history = chat_history
            self.current_session.system_message = system_message
            self.apply_saved_parameters(self.current_session)
            
            self.display_non_default_parameters()

            if not chat_history:
                console.print("No previous chat history.", style="cyan")
            else:
                self.current_session.display_history()

            provider_name = type(self.current_session).__name__.replace('ChatSession', '')
            model_name = getattr(self.current_session, 'model', 'Unknown')
            console.print(f"Current provider: [blue]{provider_name}[/blue]", style="cyan")
            console.print(f"Current model: [blue]{model_name}[/blue]", style="cyan")

            while True:
                try:
                    user_input = await self.get_multiline_input()

                    if user_input.lower() == '/exit':
                        console.print("Thank you for chatting. Goodbye!", style="cyan")
                        break
                    elif user_input.startswith('/load '):
                        folder_name = user_input.split(' ', 1)[1]
                        await self.handle_load_command(folder_name)
                    elif user_input.startswith('@'):
                        parts = user_input[1:].split(' ', 1)
                        if len(parts) == 2:
                            folder_name, query = parts
                            await self.handle_query_command(folder_name, query)
                        else:
                            console.print("Invalid query format. Use @<foldername> <question>", style="bold red")
                    elif user_input.startswith('/'):
                        result = await self.command_handler.handle_command(user_input, self.current_session)
                        if isinstance(result, ChatProvider):
                            self.current_session = result
                    elif user_input:
                        await self.current_session.send_message(user_input)
                        self.current_session.save_history()
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break

        except Exception as e:
            console.print(f"\nAn unexpected error occurred: {e}", style="bold red")
        finally:
            if self.current_session:
                self.current_session.save_history()

    async def get_multiline_input(self) -> str:
        kb = KeyBindings()

        @kb.add('c-j')  # Ctrl+J as a workaround for Ctrl+Enter
        def _(event):
            event.current_buffer.insert_text('\n')

        @kb.add('enter')  # Enter key
        def _(event):
            if event.current_buffer.document.is_cursor_at_the_end:
                event.current_buffer.validate_and_handle()
            else:
                event.current_buffer.insert_text('\n')

        prompt_session = PromptSession(multiline=True, key_bindings=kb)

        try:
            with patch_stdout():
                user_input = await prompt_session.prompt_async(
                    "",  # No prompt message
                    bottom_toolbar=None  # No bottom toolbar message
                )
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return ""
    
    async def check_for_updates(self):
        repo_owner = "DefamationStation"
        repo_name = "Retrochat-v2"
        file_path = "retrochat.py"

        load_dotenv(self.ENV_FILE)
        last_commit_hash = os.getenv("LAST_COMMIT_HASH", "")

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&page=1&per_page=1"
        response = requests.get(url)
        if response.status_code != 200:
            console.print(f"Failed to check for updates: {response.status_code}", style="bold red")
            return False

        latest_commit = response.json()[0]
        latest_commit_hash = latest_commit['sha']

        if latest_commit_hash == last_commit_hash:
            console.print("You're running the latest version.", style="green")
            return False

        missed_commits = get_missed_commits(repo_owner, repo_name, file_path, last_commit_hash)

        url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{latest_commit_hash}/{file_path}"
        response = requests.get(url)
        if response.status_code != 200:
            console.print(f"Failed to fetch the latest version: {response.status_code}", style="bold red")
            return False

        latest_content = response.text

        with open(__file__, 'r') as f:
            current_content = f.read()

        if hashlib.sha256(current_content.encode()).hexdigest() != hashlib.sha256(latest_content.encode()).hexdigest():
            console.print("Updates are available:", style="bold yellow")
            for i, commit_message in enumerate(missed_commits, 1):
                console.print(f"{i}. {commit_message}", style="yellow")
            
            console.print("\nDo you want to update?\n\n1. Yes\n2. No")
            choice = Prompt.ask("", choices=["1", "2"])

            if choice == "1":
                console.print("Updating...", style="cyan")
                with open(__file__, 'w') as f:
                    f.write(latest_content)
                set_key(self.ENV_FILE, "LAST_COMMIT_HASH", latest_commit_hash)
                set_key(self.ENV_FILE, "UPDATED", "true")
                console.print("Update complete. Please restart the script.", style="bold green")
                return True
            else:
                console.print("Update skipped. Running current version.", style="yellow")
                return False
        else:
            console.print("You're running the latest version.", style="green")
            set_key(self.ENV_FILE, "LAST_COMMIT_HASH", latest_commit_hash)
            set_key(self.ENV_FILE, "UPDATED", "false")
            return False

def get_missed_commits(repo_owner, repo_name, file_path, last_commit_hash):
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

async def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_rchat()
    else:
        app = ChatApp()
        await app.start()

if __name__ == "__main__":
    asyncio.run(main())
