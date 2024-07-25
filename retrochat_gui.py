import sys
import os
import asyncio
import aiohttp
import requests
import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv, set_key
import tiktoken
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QComboBox, QLabel, QTabWidget,
    QDialog, QInputDialog, QDialogButtonBox, QFormLayout, QCheckBox, QMessageBox,
    QListWidget, QListWidgetItem, QScrollArea, QSizePolicy, QSpacerItem, QFrame
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QThreadPool, QRunnable, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QTextCursor, QFont, QPalette, QColor, QIcon, QPixmap

# Constants
USER_HOME = os.path.expanduser('~')
RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
DB_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.db')
SETTINGS_FILE = os.path.join(RETROCHAT_DIR, 'settings.json')
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
LAST_CHAT_NAME_KEY = "LAST_CHAT_NAME"
LAST_PROVIDER_KEY = "LAST_PROVIDER"
LAST_MODEL_KEY = "LAST_MODEL"
OLLAMA_IP_KEY = "OLLAMA_IP"
OLLAMA_PORT_KEY = "OLLAMA_PORT"

os.makedirs(RETROCHAT_DIR, exist_ok=True)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

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
        self._create_tables()
        self._update_schema()

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_name TEXT NOT NULL UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    system_message TEXT,
                    parameters TEXT
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                )
            ''')

    def _update_schema(self):
        try:
            cursor = self.conn.execute("PRAGMA table_info(chat_sessions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'system_message' not in columns:
                self.conn.execute('ALTER TABLE chat_sessions ADD COLUMN system_message TEXT')
            if 'parameters' not in columns:
                self.conn.execute('ALTER TABLE chat_sessions ADD COLUMN parameters TEXT')
            self.conn.commit()
            logging.info("Database schema updated successfully.")
        except Exception as e:
            logging.error(f"Error updating database schema: {e}")

    def _get_session_id(self, chat_name: str) -> int:
        cursor = self.conn.execute('SELECT id FROM chat_sessions WHERE chat_name = ?', (chat_name,))
        session = cursor.fetchone()
        if session:
            return session[0]
        cursor = self.conn.execute('INSERT INTO chat_sessions (chat_name) VALUES (?)', (chat_name,))
        self.conn.commit()
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
        try:
            cursor = self.conn.execute('SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
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
        cursor = self.conn.execute('SELECT system_message FROM chat_sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def save_parameters(self, parameters: Dict[str, Any]):
        session_id = self._get_session_id(self.chat_name)
        parameters_json = json.dumps(parameters)
        with self.conn:
            self.conn.execute('UPDATE chat_sessions SET parameters = ? WHERE id = ?', (parameters_json, session_id))

    def load_parameters(self) -> Dict[str, Any]:
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.execute('SELECT parameters FROM chat_sessions WHERE id = ?', (session_id,))
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
            raise ValueError(f"A chat with the name '{new_name}' already exists.")

    def delete_history(self):
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
            self.conn.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        self.set_chat_name('default')

    def list_chats(self) -> List[str]:
        cursor = self.conn.execute('SELECT chat_name FROM chat_sessions')
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
            "verbose": False,
        }

    def load_history(self) -> List[ChatMessage]:
        try:
            return self.history_manager.load_history()
        except Exception as e:
            logging.error(f"Error loading chat history: {e}")
            return []

    async def send_message(self, message: str):
        raise NotImplementedError("Subclasses must implement this method")

    def add_to_history(self, role: str, content: str):
        self.chat_history.append(ChatMessage(role, content))
        self.save_history()

    def save_history(self):
        try:
            self.history_manager.save_history(self.chat_history)
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")

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
            if param in ["num_predict", "top_k", "repeat_last_n", "num_ctx"]:
                value = int(value)
            elif param in ["top_p", "temperature", "repeat_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split() if isinstance(value, str) else value
            elif param == "verbose":
                value = str(value).lower() == "true"
            self.parameters[param] = value
            self.history_manager.save_parameters(self.parameters)

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
                            if message_content:
                                complete_message += message_content
                                yield message_content
                            if response_json.get('done', False):
                                break
                    formatted_message = self.format_message(complete_message)
                    self.add_to_history("assistant", formatted_message)
                else:
                    error_text = await response.text()
                    raise Exception(f"Error: {response.status} - {error_text}")

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
                        yield formatted_message
                        if self.parameters.get("verbose", False):
                            tokens = self.calculate_tokens(formatted_message)
                            total_tokens = self.calculate_total_tokens()
                            print(f"Response tokens: {tokens}")
                            print(f"Total conversation tokens: {total_tokens}")
                    else:
                        print("No response content received.")
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

class StyleSheet:
    @staticmethod
    def get_style():
        return """
        QMainWindow, QWidget {
            background-color: #f5f5f5;
            color: #333333;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
        QLineEdit, QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 8px;
            background-color: white;
        }
        QComboBox {
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 8px;
            background-color: white;
        }
        QLabel {
            color: #333333;
            font-weight: bold;
        }
        """

class ModernScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f9f9f9;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

class ChatBubble(QFrame):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setObjectName("chatBubble")
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)

        if is_user:
            self.setStyleSheet("""
                #chatBubble {
                    background-color: #e6f3ff;
                    border: 1px solid #b3d9ff;
                    border-radius: 10px;
                    padding: 10px;
                    margin-right: 50px;
                }
            """)
        else:
            self.setStyleSheet("""
                #chatBubble {
                    background-color: #f0f0f0;
                    border: 1px solid #d9d9d9;
                    border-radius: 10px;
                    padding: 10px;
                    margin-left: 50px;
                }
            """)

class ChatArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(15)
        self.layout.addStretch()
        self.setStyleSheet("""
            background-color: #ffffff;
        """)

    def add_message(self, text, is_user=True):
        message = QLabel(text)
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        message.setCursor(Qt.CursorShape.IBeamCursor)
        message.setStyleSheet(f"""
            background-color: {'#e6f3ff' if is_user else '#f0f0f0'};
            border-radius: 15px;
            padding: 10px 15px;
            margin: {'0 50px 0 10px' if is_user else '0 10px 0 50px'};
            font-size: 14px;
        """)
        self.layout.insertWidget(self.layout.count() - 1, message)

    def clear(self):
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def update_last_message(self, text):
        if self.layout.count() > 1:
            last_item = self.layout.itemAt(self.layout.count() - 2).widget()
            if isinstance(last_item, QLabel):
                last_item.setText(text)
                
class ChatBubble(QFrame):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setObjectName("chatBubble")
        self.layout = QVBoxLayout(self)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.layout.addWidget(self.label)

        if is_user:
            self.setStyleSheet("""
                #chatBubble {
                    background-color: #e6f3ff;
                    border: 1px solid #b3d9ff;
                    border-radius: 10px;
                    padding: 10px;
                    margin-right: 50px;
                }
            """)
        else:
            self.setStyleSheet("""
                #chatBubble {
                    background-color: #f0f0f0;
                    border: 1px solid #d9d9d9;
                    border-radius: 10px;
                    padding: 10px;
                    margin-left: 50px;
                }
            """)

    def update_text(self, text):
        self.label.setText(text)

class ModernSidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(15)

        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                color: white;
            }
            QPushButton {
                background-color: #34495e;
                border: none;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a6785;
            }
            QComboBox {
                background-color: #34495e;
                border: none;
                padding: 8px;
                border-radius: 5px;
            }
            QLabel {
                font-size: 16px;
                font-weight: bold;
                margin-top: 10px;
            }
        """)

        self.create_provider_section()
        self.create_model_section()
        self.create_chat_controls()
        self.layout.addStretch()
        self.create_settings_button()

    def create_provider_section(self):
        provider_label = QLabel("Provider:")
        self.layout.addWidget(provider_label)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Ollama", "Anthropic", "OpenAI"])
        self.layout.addWidget(self.provider_combo)

    def create_model_section(self):
        model_label = QLabel("Model:")
        self.layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.layout.addWidget(self.model_combo)

    def create_chat_controls(self):
        self.new_chat_btn = QPushButton("New Chat")
        self.layout.addWidget(self.new_chat_btn)

        self.open_chat_btn = QPushButton("Open Chat")
        self.layout.addWidget(self.open_chat_btn)

    def create_settings_button(self):
        self.settings_btn = QPushButton("Settings")
        self.layout.addWidget(self.settings_btn)

    def set_provider_changed_callback(self, callback):
        self.provider_combo.currentTextChanged.connect(callback)

    def set_model_changed_callback(self, callback):
        self.model_combo.currentTextChanged.connect(callback)

    def set_new_chat_callback(self, callback):
        self.new_chat_btn.clicked.connect(callback)

    def set_open_chat_callback(self, callback):
        self.open_chat_btn.clicked.connect(callback)

    def set_settings_callback(self, callback):
        self.settings_btn.clicked.connect(callback)

    def update_models(self, models):
        self.model_combo.clear()
        self.model_combo.addItems(models)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RetroChat v2")
        self.setGeometry(100, 100, 1200, 800)
        self.threadpool = QThreadPool()

        self.chat_provider = None
        self.history_manager = ChatHistoryManager(DB_FILE)
        self.load_env_variables()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.sidebar = ModernSidebar()
        self.main_layout.addWidget(self.sidebar)

        self.chat_layout = QVBoxLayout()
        self.main_layout.addLayout(self.chat_layout)

        self.create_chat_interface()
        self.create_input_area()
        self.setup_sidebar_connections()

        self.setStyleSheet(StyleSheet.get_style())
        self.load_last_session()

    def load_env_variables(self):
        if os.path.exists(ENV_FILE):
            load_dotenv(ENV_FILE)
            self.openai_api_key = os.getenv(OPENAI_API_KEY_NAME)
            self.anthropic_api_key = os.getenv(ANTHROPIC_API_KEY_NAME)
            self.chat_name = os.getenv(LAST_CHAT_NAME_KEY, 'default')
            self.ollama_ip = os.getenv(OLLAMA_IP_KEY, 'localhost')
            self.ollama_port = os.getenv(OLLAMA_PORT_KEY, '11434')
            self.last_provider = os.getenv(LAST_PROVIDER_KEY)
            self.last_model = os.getenv(LAST_MODEL_KEY)
            self.history_manager.set_chat_name(self.chat_name)

    def create_chat_interface(self):
        self.chat_area = ChatArea()
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setWidget(self.chat_area)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #ffffff;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.chat_layout.addWidget(self.chat_scroll)

    def create_input_area(self):
        input_layout = QHBoxLayout()
        self.message_input = QTextEdit()
        self.message_input.setFixedHeight(60)
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setStyleSheet("""
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        """)
        input_layout.addWidget(self.message_input)

        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        input_layout.addWidget(send_button)

        self.chat_layout.addLayout(input_layout)

    def setup_sidebar_connections(self):
        self.sidebar.set_provider_changed_callback(self.change_provider)
        self.sidebar.set_model_changed_callback(self.change_model)
        self.sidebar.set_new_chat_callback(self.new_chat)
        self.sidebar.set_open_chat_callback(self.open_chat)
        self.sidebar.set_settings_callback(self.open_settings)

    def load_last_session(self):
        if self.last_provider and self.last_model:
            self.sidebar.provider_combo.setCurrentText(self.last_provider)
            self.change_provider(self.last_provider)
            self.sidebar.model_combo.setCurrentText(self.last_model)
            self.change_model(self.last_model)
        self.display_chat_history()

    def change_provider(self, provider):
        models = []
        if provider == "Ollama":
            models = self.get_ollama_models()
        elif provider == "Anthropic":
            models = ["claude-3-5-sonnet-20240620"]
        elif provider == "OpenAI":
            models = ["gpt-4", "gpt-3.5-turbo"]
        
        self.sidebar.update_models(models)
        set_key(ENV_FILE, LAST_PROVIDER_KEY, provider)
        load_dotenv(ENV_FILE)

    def change_model(self, model):
        if not model:
            return
        provider = self.sidebar.provider_combo.currentText()
        self.initialize_provider(provider, model)
        set_key(ENV_FILE, LAST_MODEL_KEY, model)
        load_dotenv(ENV_FILE)

    def get_ollama_models(self):
        url = f"http://{self.ollama_ip}:{self.ollama_port}/api/tags"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                models_info = response.json()
                if isinstance(models_info, dict) and 'models' in models_info:
                    return [model['name'] for model in models_info['models']]
            return []
        except requests.RequestException:
            return []

    def initialize_provider(self, provider_type, model):
        if provider_type == "Ollama":
            if not self.ensure_ollama_connection():
                return
            model_url = f"http://{self.ollama_ip}:{self.ollama_port}/api/chat"
            self.chat_provider = ChatProviderFactory.create_provider('Ollama', model_url, model, self.history_manager)
        elif provider_type == "Anthropic":
            if not self.ensure_api_key('anthropic_api_key', ANTHROPIC_API_KEY_NAME):
                return
            self.chat_provider = ChatProviderFactory.create_provider('Anthropic', self.anthropic_api_key, "https://api.anthropic.com/v1/messages", self.history_manager, model)
        elif provider_type == "OpenAI":
            if not self.ensure_api_key('openai_api_key', OPENAI_API_KEY_NAME):
                return
            self.chat_provider = ChatProviderFactory.create_provider('OpenAI', self.openai_api_key, "https://api.openai.com/v1/chat/completions", model, self.history_manager)

        if self.chat_provider:
            saved_params = self.history_manager.load_parameters()
            for param, value in saved_params.items():
                self.chat_provider.set_parameter(param, value)
            self.display_chat_history()

    def ensure_api_key(self, key_name: str, env_var: str):
        if not getattr(self, key_name):
            api_key, ok = QInputDialog.getText(self, f"Enter {env_var}", f"{env_var} is not set. Please enter your API key:", QLineEdit.EchoMode.Password)
            if ok and api_key:
                set_key(ENV_FILE, env_var, api_key)
                load_dotenv(ENV_FILE)
                setattr(self, key_name, api_key)
                return True
            else:
                QMessageBox.warning(self, "Error", f"No API key provided. {key_name.replace('_', ' ').title()} mode cannot be used.")
                return False
        return True

    def ensure_ollama_connection(self):
        url = f"http://{self.ollama_ip}:{self.ollama_port}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            QMessageBox.warning(self, "Error", f"Unable to connect to Ollama at {self.ollama_ip}:{self.ollama_port}")
            new_ip, ok1 = QInputDialog.getText(self, "Ollama IP", "Enter Ollama IP:", QLineEdit.EchoMode.Normal, self.ollama_ip)
            new_port, ok2 = QInputDialog.getText(self, "Ollama Port", "Enter Ollama port:", QLineEdit.EchoMode.Normal, self.ollama_port)
            
            if ok1 and ok2:
                self.ollama_ip = new_ip
                self.ollama_port = new_port
                
                set_key(ENV_FILE, OLLAMA_IP_KEY, self.ollama_ip)
                set_key(ENV_FILE, OLLAMA_PORT_KEY, self.ollama_port)
                load_dotenv(ENV_FILE)
                
                return self.ensure_ollama_connection()
            else:
                return False

    def new_chat(self):
        name, ok = QInputDialog.getText(self, "New Chat", "Enter name for the new chat:")
        if ok and name:
            self.history_manager.set_chat_name(name)
            self.history_manager.save_history([])
            self.chat_area.clear()
            self.save_last_chat_name(name)
            self.display_chat_history()

    def open_chat(self):
        chats = self.history_manager.list_chats()
        chat, ok = QInputDialog.getItem(self, "Open Chat", "Select a chat to open:", chats, 0, False)
        if ok and chat:
            self.history_manager.set_chat_name(chat)
            if self.chat_provider:
                self.chat_provider.chat_history = self.history_manager.load_history()
                self.chat_provider.system_message = self.history_manager.load_system_message()
                self.chat_provider.parameters = self.history_manager.load_parameters()
            self.display_chat_history()
            self.save_last_chat_name(chat)

    def save_last_chat_name(self, chat_name: str):
        set_key(ENV_FILE, LAST_CHAT_NAME_KEY, chat_name)

    def display_chat_history(self):
        if self.chat_provider:
            self.chat_area.clear()
            for message in self.chat_provider.chat_history:
                self.chat_area.add_message(message.content, message.role == "user")

    def send_message(self):
        message = self.message_input.toPlainText().strip()
        if message and self.chat_provider:
            self.message_input.clear()
            self.chat_area.add_message(message, True)
            
            worker = MessageWorker(self.chat_provider, message)
            worker.signals.response_chunk.connect(self.update_assistant_response)
            worker.signals.error.connect(self.show_error_message)
            worker.signals.finished.connect(self.on_worker_finished)
            
            self.threadpool.start(worker)

    def update_assistant_response(self, chunk):
        if not hasattr(self, 'current_response'):
            self.current_response = ""
            self.chat_area.add_message("", False)
        self.current_response += chunk
        self.chat_area.update_last_message(self.current_response)
        self.chat_scroll.ensureWidgetVisible(self.chat_area.layout.itemAt(self.chat_area.layout.count() - 1).widget())

    def on_worker_finished(self):
        if hasattr(self, 'current_response'):
            del self.current_response

    def show_error_message(self, error):
        QMessageBox.critical(self, "Error", f"An error occurred: {error}")

    def open_settings(self):
        settings_dialog = SettingsDialog(self.chat_provider, self)
        if settings_dialog.exec():
            self.display_chat_history()  # Refresh the chat display in case system message changed

class SettingsDialog(QDialog):
    def __init__(self, chat_provider, parent=None):
        super().__init__(parent)
        self.chat_provider = chat_provider
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        
        self.create_settings()

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def create_settings(self):
        layout = QFormLayout()
        self.setLayout(layout)

        # System Message
        self.system_message_input = QTextEdit()
        self.system_message_input.setPlainText(self.chat_provider.system_message if self.chat_provider else "")
        layout.addRow("System Message:", self.system_message_input)

        # Parameters (with temperature, num ctx, and repeat penalty at the top)
        self.temperature_input = QLineEdit(str(self.chat_provider.parameters.get("temperature", 0.8)))
        layout.addRow("Temperature:", self.temperature_input)

        self.num_ctx_input = QLineEdit(str(self.chat_provider.parameters.get("num_ctx", 8192)))
        layout.addRow("Num CTX:", self.num_ctx_input)

        if isinstance(self.chat_provider, OpenAIChatSession):
            self.frequency_penalty_input = QLineEdit(str(self.chat_provider.parameters.get("frequency_penalty", 0.0)))
            layout.addRow("Frequency Penalty:", self.frequency_penalty_input)
        else:
            self.repeat_penalty_input = QLineEdit(str(self.chat_provider.parameters.get("repeat_penalty", 0.95)))
            layout.addRow("Repeat Penalty:", self.repeat_penalty_input)

        self.max_tokens_input = QLineEdit(str(self.chat_provider.parameters.get("max_tokens", 8192)))
        layout.addRow("Max Tokens:", self.max_tokens_input)

        self.num_predict_input = QLineEdit(str(self.chat_provider.parameters.get("num_predict", 128)))
        layout.addRow("Num Predict:", self.num_predict_input)

        self.top_k_input = QLineEdit(str(self.chat_provider.parameters.get("top_k", 40)))
        layout.addRow("Top K:", self.top_k_input)

        self.top_p_input = QLineEdit(str(self.chat_provider.parameters.get("top_p", 0.95)))
        layout.addRow("Top P:", self.top_p_input)

        self.repeat_last_n_input = QLineEdit(str(self.chat_provider.parameters.get("repeat_last_n", 64)))
        layout.addRow("Repeat Last N:", self.repeat_last_n_input)

    def accept(self):
        if self.chat_provider:
            new_system_message = self.system_message_input.toPlainText()
            self.chat_provider.set_system_message(new_system_message)

            self.chat_provider.set_parameter("temperature", float(self.temperature_input.text()))
            self.chat_provider.set_parameter("num_ctx", int(self.num_ctx_input.text()))
            if isinstance(self.chat_provider, OpenAIChatSession):
                self.chat_provider.set_parameter("frequency_penalty", float(self.frequency_penalty_input.text()))
            else:
                self.chat_provider.set_parameter("repeat_penalty", float(self.repeat_penalty_input.text()))
            self.chat_provider.set_parameter("max_tokens", int(self.max_tokens_input.text()))
            self.chat_provider.set_parameter("num_predict", int(self.num_predict_input.text()))
            self.chat_provider.set_parameter("top_k", int(self.top_k_input.text()))
            self.chat_provider.set_parameter("top_p", float(self.top_p_input.text()))
            self.chat_provider.set_parameter("repeat_last_n", int(self.repeat_last_n_input.text()))

        super().accept()

class WorkerSignals(QObject):
    finished = pyqtSignal()
    response_chunk = pyqtSignal(str)
    error = pyqtSignal(str)

class MessageWorker(QRunnable):
    def __init__(self, chat_provider, message):
        super().__init__()
        self.chat_provider = chat_provider
        self.message = message
        self.signals = WorkerSignals()

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_generator = self.chat_provider.send_message(self.message)
            while True:
                try:
                    chunk = loop.run_until_complete(response_generator.__anext__())
                    self.signals.response_chunk.emit(chunk)
                except StopAsyncIteration:
                    break
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
