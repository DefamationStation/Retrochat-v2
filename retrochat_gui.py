import sys
import os
import platform
import asyncio
import aiohttp
import requests
import sqlite3
import base64
import json
import logging
import tempfile
import subprocess
import hashlib
import traceback
import qasync
from threading import Lock
from qasync import QApplication, QEventLoop, asyncSlot
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv, set_key
import shutil

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QComboBox, QLabel, QStackedWidget, QMessageBox,
                             QDialog, QDialogButtonBox, QInputDialog)
from PyQt6.QtGui import QIcon, QFont, QTextCursor, QColor, QPalette
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QRunnable, QThreadPool

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
RETROCHAT_SCRIPT = os.path.join(RETROCHAT_DIR, 'retrochat_gui.py')

os.makedirs(RETROCHAT_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.lock = Lock()
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
                logger.info("Database schema updated successfully.")
        except Exception as e:
            logger.error(f"Error updating database schema: {e}")

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
            with self.lock:
                with self.conn:
                    self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
                    self.conn.executemany('''
                        INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)
                    ''', [(session_id, msg.role, msg.content) for msg in history])
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")

    def load_history(self) -> List[ChatMessage]:
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
        try:
            return [ChatMessage(role=row[0], content=row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
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
            logger.error(f"Error: A chat with the name '{new_name}' already exists.")

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

class ChatProvider:
    def __init__(self, history_manager: ChatHistoryManager):
        self.history_manager = history_manager
        self.chat_history = self.load_history()
        self.system_message = self.history_manager.load_system_message()
        self.parameters = self.history_manager.load_parameters()
        self.default_parameters = {
            "temperature": 0.8,
            "max_tokens": 8192,
        }

    def load_history(self) -> List[ChatMessage]:
        try:
            return self.history_manager.load_history()
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            return []

    async def send_message(self, message: str):
        raise NotImplementedError

    def add_to_history(self, role: str, content: str):
        self.chat_history.append(ChatMessage(role, content))
        self.save_history()

    def save_history(self):
        try:
            self.history_manager.save_history(self.chat_history)
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")

    def display_history(self):
        return self.chat_history

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
        if param in self.default_parameters:
            if param == "max_tokens":
                value = int(value)
            elif param == "temperature":
                value = float(value)
            self.parameters[param] = value
            self.history_manager.save_parameters(self.parameters)
            logger.info(f"Parameter '{param}' set to {value}")
        else:
            logger.error(f"Invalid parameter: {param}")

    def show_parameters(self):
        params = []
        for param, default_value in self.default_parameters.items():
            current_value = self.parameters.get(param, default_value)
            params.append(f"{param}: {current_value}")
        params.append(f"system: {self.system_message if self.system_message else 'Not set'}")
        return params

class OllamaChatSession(ChatProvider):
    def __init__(self, model_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model
        self.default_parameters.update({
            "num_predict": 128,
            "top_k": 40,
            "top_p": 0.95,
            "repeat_penalty": 0.95,
            "repeat_last_n": 64,
            "num_ctx": 8192,
            "stop": None,
        })

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
                            yield message_content
                            if response_json.get('done', False):
                                break
                    formatted_message = self.format_message(complete_message)
                    self.add_to_history("assistant", formatted_message)
                    self.save_history()
                else:
                    error_text = await response.text()
                    raise Exception(f"Error: {response.status} - {error_text}")

    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters:
            if param in ["num_predict", "top_k", "repeat_last_n", "num_ctx"]:
                value = int(value)
            elif param in ["top_p", "temperature", "repeat_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split()  # Split into a list
            self.parameters[param] = value
            self.history_manager.save_parameters(self.parameters)
            logger.info(f"Parameter '{param}' set to {value}")
        else:
            logger.error(f"Invalid parameter: {param}")

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
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
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
                        yield formatted_message
                    else:
                        raise Exception("No response content received.")
                else:
                    error_text = await response.text()
                    raise Exception(f"Error: {response.status} - {error_text}")

    def prepare_messages(self):
        messages = []
        for msg in self.chat_history:
            if msg.role != "system":
                if not messages or messages[-1]["role"] != msg.role:
                    messages.append({"role": msg.role, "content": msg.content})
                else:
                    # If the roles are the same, combine the contents
                    messages[-1]["content"] += "\n" + msg.content
        return messages

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
            "temperature": self.parameters.get("temperature", 0.8),
            "max_tokens": self.parameters.get("max_tokens", 8192),
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
                                        yield content
                                except json.JSONDecodeError:
                                    continue
                    formatted_message = self.format_message(complete_message)
                    self.add_to_history("assistant", formatted_message)
                    self.save_history()
                else:
                    error_text = await response.text()
                    raise Exception(f"Error: {response.status} - {error_text}")

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

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(tuple)
    finished = pyqtSignal()

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class RetrochatGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RetroChat GUI")
        self.setGeometry(100, 100, 1000, 700)

        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(DB_FILE)
        self.provider_factory = ChatProviderFactory()
        self.openai_api_key = None
        self.anthropic_api_key = None
        self.ollama_ip = None
        self.ollama_port = None
        self.current_session = None

        self.load_env_variables()
        self.save_last_chat_name(self.chat_name)

        self.init_ui()
        self.threadpool = QThreadPool()
        self.provider_combo.setCurrentIndex(0)  # Set default provider
        self.change_provider(0)

    def load_env_variables(self):
        if os.path.exists(ENV_FILE):
            load_dotenv(ENV_FILE)
            self.openai_api_key = os.getenv(OPENAI_API_KEY_NAME)
            self.anthropic_api_key = os.getenv(ANTHROPIC_API_KEY_NAME)
            self.chat_name = os.getenv(LAST_CHAT_NAME_KEY, 'default')
            self.ollama_ip = os.getenv(OLLAMA_IP_KEY, 'localhost')
            self.ollama_port = os.getenv(OLLAMA_PORT_KEY, '11434')
            self.history_manager.set_chat_name(self.chat_name)

    def save_last_chat_name(self, chat_name: str):
        set_key(ENV_FILE, LAST_CHAT_NAME_KEY, chat_name)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)

        # Provider and Model selection
        selection_layout = QHBoxLayout()
        provider_label = QLabel("Provider:")
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Ollama", "Anthropic", "OpenAI"])
        self.provider_combo.currentIndexChanged.connect(self.change_provider)
        
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setEnabled(False)
        self.model_combo.currentIndexChanged.connect(self.select_model)
        
        selection_layout.addWidget(provider_label)
        selection_layout.addWidget(self.provider_combo)
        selection_layout.addWidget(model_label)
        selection_layout.addWidget(self.model_combo)
        main_layout.addLayout(selection_layout)

        # Command buttons
        command_layout = QHBoxLayout()
        self.new_chat_button = QPushButton("New Chat")
        self.new_chat_button.clicked.connect(self.new_chat)
        self.open_chat_button = QPushButton("Open Chat")
        self.open_chat_button.clicked.connect(self.open_chat)
        self.rename_chat_button = QPushButton("Rename Chat")
        self.rename_chat_button.clicked.connect(self.rename_chat)
        self.delete_chat_button = QPushButton("Delete Chat")
        self.delete_chat_button.clicked.connect(self.delete_chat)
        self.reset_chat_button = QPushButton("Reset Chat")
        self.reset_chat_button.clicked.connect(self.reset_chat)
        command_layout.addWidget(self.new_chat_button)
        command_layout.addWidget(self.open_chat_button)
        command_layout.addWidget(self.rename_chat_button)
        command_layout.addWidget(self.delete_chat_button)
        command_layout.addWidget(self.reset_chat_button)
        main_layout.addLayout(command_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F4F8;
            }
            QTextEdit, QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #B0BEC5;
                border-radius: 5px;
                padding: 5px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12pt;
                color: #37474F;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #B0BEC5;
                border-radius: 5px;
                padding: 5px;
                font-size: 10pt;
                color: #37474F;
            }
            QLabel {
                font-size: 10pt;
                color: #37474F;
            }
        """)

    def send_message(self):
        message = self.chat_input.text()
        if message and self.current_session:
            self.chat_display.append(f"<b>You:</b> {message}")
            self.chat_input.clear()
            worker = Worker(asyncio.run, self.process_message(message))
            worker.signals.result.connect(self.update_chat_display)
            worker.signals.error.connect(self.handle_error)
            self.threadpool.start(worker)

    async def process_message(self, message):
        response = ""
        async for chunk in self.current_session.send_message(message):
            response += chunk
        return response

    def update_chat_display(self, response):
        self.chat_display.append(f"<b>AI:</b> {response}")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def handle_error(self, error_tuple):
        exctype, value, _ = error_tuple
        error_message = f"An error occurred: {exctype.__name__}: {value}"
        QMessageBox.critical(self, "Error", error_message)

    def change_provider(self, index):
        provider = self.provider_combo.itemText(index)
        self.model_combo.clear()
        self.model_combo.setEnabled(True)

        if provider == "Ollama":
            if not self.ensure_ollama_connection():
                return
            self.load_ollama_models()
        elif provider == "Anthropic":
            if not self.ensure_api_key('anthropic_api_key', ANTHROPIC_API_KEY_NAME):
                return
            self.model_combo.addItem("claude-3-5-sonnet-20240620")
        elif provider == "OpenAI":
            if not self.ensure_api_key('openai_api_key', OPENAI_API_KEY_NAME):
                return
            self.model_combo.addItems(["gpt-4o-mini", "gpt-4o"])
        
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)  # Select first model by default

    def load_ollama_models(self):
        url = f"http://{self.ollama_ip}:{self.ollama_port}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()
            models_info = response.json()
            if isinstance(models_info, dict) and 'models' in models_info:
                model_names = [model['name'] for model in models_info['models']]
                self.model_combo.addItems(model_names)
            else:
                QMessageBox.warning(self, "Error", "Unexpected API response structure.")
        except requests.RequestException as e:
            QMessageBox.critical(self, "Error", f"Error fetching Ollama models: {str(e)}")

    def select_model(self):
        if self.model_combo.currentText():  # Ensure a model is selected
            provider = self.provider_combo.currentText()
            model = self.model_combo.currentText()

            if provider == "Ollama":
                model_url = f"http://{self.ollama_ip}:{self.ollama_port}/api/chat"
                self.current_session = self.provider_factory.create_provider('Ollama', model_url, model, self.history_manager)
            elif provider == "Anthropic":
                self.current_session = self.provider_factory.create_provider('Anthropic', self.anthropic_api_key, "https://api.anthropic.com/v1/messages", self.history_manager, model)
            elif provider == "OpenAI":
                self.current_session = self.provider_factory.create_provider('OpenAI', self.openai_api_key, "https://api.openai.com/v1/chat/completions", model, self.history_manager)

            self.chat_display.append(f"Switched to {provider} mode using {model} model")

    def ensure_api_key(self, key_name: str, env_var: str):
        if not getattr(self, key_name):
            api_key, ok = QInputDialog.getText(self, f"Enter {env_var}", f"{env_var} is not set. Please enter your API key:", QLineEdit.EchoMode.Password)
            if ok and api_key:
                set_key(ENV_FILE, env_var, api_key)
                load_dotenv(ENV_FILE)
                setattr(self, key_name, api_key)
                QMessageBox.information(self, "API Key Set", f"{env_var} has been set and saved in the .env file.")
                return True
            else:
                QMessageBox.warning(self, "No API Key", f"No API key provided. {key_name.replace('_', ' ').title()} mode cannot be used.")
                return False
        return True

    def ensure_ollama_connection(self):
        url = f"http://{self.ollama_ip}:{self.ollama_port}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            QMessageBox.warning(self, "Connection Error", f"Unable to connect to Ollama at {self.ollama_ip}:{self.ollama_port}")
            new_ip, ok1 = QInputDialog.getText(self, "Ollama IP", "Enter Ollama IP (press Enter for localhost):")
            new_port, ok2 = QInputDialog.getText(self, "Ollama Port", "Enter Ollama port (press Enter for 11434):")
            
            if ok1 and ok2:
                self.ollama_ip = new_ip or 'localhost'
                self.ollama_port = new_port or '11434'
                
                set_key(ENV_FILE, OLLAMA_IP_KEY, self.ollama_ip)
                set_key(ENV_FILE, OLLAMA_PORT_KEY, self.ollama_port)
                load_dotenv(ENV_FILE)
                
                QMessageBox.information(self, "Connection Updated", "Ollama connection details updated and saved in the .env file.")
                return self.ensure_ollama_connection()
            else:
                return False

    def new_chat(self):
        name, ok = QInputDialog.getText(self, "New Chat", "Enter a name for the new chat:")
        if ok and name:
            self.history_manager.set_chat_name(name)
            self.history_manager.save_history([])
            self.chat_display.clear()
            self.chat_display.append(f"New chat '{name}' created.")
            self.save_last_chat_name(name)

    def open_chat(self):
        chats = self.history_manager.list_chats()
        if chats:
            chat, ok = QInputDialog.getItem(self, "Open Chat", "Select a chat to open:", chats, 0, False)
            if ok and chat:
                self.history_manager.set_chat_name(chat)
                if self.current_session:
                    self.current_session.chat_history = self.history_manager.load_history()
                    self.current_session.system_message = self.history_manager.load_system_message()
                    self.current_session.parameters = self.history_manager.load_parameters()
                self.chat_display.clear()
                self.chat_display.append(f"Chat '{chat}' opened.")
                self.display_history()
                self.save_last_chat_name(chat)
        else:
            QMessageBox.information(self, "No Chats", "No available chats.")

    def rename_chat(self):
        new_name, ok = QInputDialog.getText(self, "Rename Chat", "Enter a new name for the current chat:")
        if ok and new_name:
            try:
                self.history_manager.rename_history(new_name)
                self.chat_display.append(f"Chat renamed to '{new_name}'")
                self.save_last_chat_name(new_name)
            except sqlite3.IntegrityError:
                QMessageBox.warning(self, "Error", f"A chat with the name '{new_name}' already exists.")

    def delete_chat(self):
        reply = QMessageBox.question(self, "Delete Chat", "Are you sure you want to delete the current chat?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.history_manager.delete_history()
            self.chat_display.clear()
            self.chat_display.append("Current chat history deleted.")

    def reset_chat(self):
        reply = QMessageBox.question(self, "Reset Chat", "Are you sure you want to reset the current chat history?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.history_manager.clear_history()
            if self.current_session:
                self.current_session.chat_history = []
            self.chat_display.clear()
            self.chat_display.append("Chat history has been reset.")

    def display_history(self):
        self.chat_display.clear()
        if self.current_session:
            for message in self.current_session.display_history():
                if message.role == "user":
                    self.chat_display.append(f"<b>You:</b> {message.content}")
                else:
                    self.chat_display.append(f"<b>AI:</b> {message.content}")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

def check_for_updates():
    repo_owner = "DefamationStation"
    repo_name = "Retrochat-v2"
    file_path = "retrochat_gui.py"  # Updated to check for retrochat_gui.py

    # Get the latest commit hash
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&page=1&per_page=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        latest_commit = response.json()[0]
        latest_commit_hash = latest_commit['sha']

        # Get the content of the file in the latest commit
        url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{latest_commit_hash}/{file_path}"
        response = requests.get(url)
        response.raise_for_status()
        latest_content = response.text

        # Compare the content
        with open(__file__, 'r') as f:
            current_content = f.read()

        if hashlib.sha256(current_content.encode()).hexdigest() != hashlib.sha256(latest_content.encode()).hexdigest():
            return True, latest_content
        else:
            return False, None
    except requests.RequestException as e:
        logger.error(f"Failed to check for updates: {e}")
        return False, None

def main():
    app = QApplication(sys.argv)
    
    update_available, latest_content = check_for_updates()
    if update_available:
        reply = QMessageBox.question(None, "Update Available", 
                                     "An update is available. Do you want to update?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            with open(__file__, 'w') as f:
                f.write(latest_content)
            QMessageBox.information(None, "Update Complete", "Update complete. Please restart the application.")
            return

    window = RetrochatGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()