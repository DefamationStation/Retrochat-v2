import sys
import os
import asyncio
import aiohttp
import sqlite3
import json
import requests
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel, QMessageBox,
                             QInputDialog, QDialog, QFormLayout, QDialogButtonBox)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QTextCursor
from dotenv import load_dotenv, set_key

# Constants
USER_HOME = os.path.expanduser('~')
RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
DB_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.db')
SETTINGS_FILE = os.path.join(RETROCHAT_DIR, 'settings.json')
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
LAST_CHAT_NAME_KEY = "LAST_CHAT_NAME"

os.makedirs(RETROCHAT_DIR, exist_ok=True)

# Load environment variables
load_dotenv(ENV_FILE)

class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        return cls(role=data["role"], content=data["content"])

class ChatHistoryManager:
    def __init__(self, db_file: str, chat_name: str = 'default'):
        self.db_file = db_file
        self.chat_name = chat_name
        self.local = threading.local()
        self._create_tables()

    def _get_connection(self):
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_file)
        return self.local.conn

    def _create_tables(self):
        with self._get_connection() as conn:
            conn.executescript('''
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

    def _get_session_id(self, chat_name: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM chat_sessions WHERE chat_name = ?', (chat_name,))
            session = cursor.fetchone()
            if session:
                return session[0]
            cursor.execute('INSERT INTO chat_sessions (chat_name) VALUES (?)', (chat_name,))
            return cursor.lastrowid

    def set_chat_name(self, chat_name: str):
        self.chat_name = chat_name

    def save_history(self, history: list):
        session_id = self._get_session_id(self.chat_name)
        with self._get_connection() as conn:
            conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
            conn.executemany('''
                INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)
            ''', [(session_id, msg.role, msg.content) for msg in history])

    def load_history(self) -> list:
        session_id = self._get_session_id(self.chat_name)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
            return [ChatMessage(role=row[0], content=row[1]) for row in cursor.fetchall()]

    def clear_history(self):
        session_id = self._get_session_id(self.chat_name)
        with self._get_connection() as conn:
            conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))

    def rename_history(self, new_name: str):
        session_id = self._get_session_id(self.chat_name)
        with self._get_connection() as conn:
            conn.execute('UPDATE chat_sessions SET chat_name = ? WHERE id = ?', (new_name, session_id))
        self.set_chat_name(new_name)

    def delete_history(self):
        session_id = self._get_session_id(self.chat_name)
        with self._get_connection() as conn:
            conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
            conn.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        self.set_chat_name('default')

    def list_chats(self) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT chat_name FROM chat_sessions')
            return [row[0] for row in cursor.fetchall()]

class ChatProvider:
    def __init__(self, history_manager: ChatHistoryManager):
        self.history_manager = history_manager
        self.chat_history = self.load_history()
        self.system_message = None

    def load_history(self) -> list:
        return self.history_manager.load_history()

    async def send_message(self, message: str):
        pass

    def add_to_history(self, role: str, content: str):
        self.chat_history.append(ChatMessage(role, content))
        self.save_history()

    def save_history(self):
        self.history_manager.save_history(self.chat_history)

    def set_system_message(self, message: str):
        self.system_message = message

class OllamaChatSession(ChatProvider):
    def __init__(self, model_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model
        self.parameters = {
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
        messages = [msg.to_dict() for msg in self.chat_history]
        
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
                    self.add_to_history("assistant", complete_message)
                    self.save_history()
                else:
                    yield f"Error: {response.status} - {await response.text()}"

    def set_parameter(self, param: str, value):
        if param in self.parameters:
            if param in ["num_predict", "top_k", "repeat_last_n", "num_ctx"]:
                value = int(value)
            elif param in ["top_p", "temperature", "repeat_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split() if value else None
            self.parameters[param] = value

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
        }
        return descriptions.get(param, "")

class AnthropicChatSession(ChatProvider):
    def __init__(self, api_key: str, model_url: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model_url = model_url

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [msg.to_dict() for msg in self.chat_history]
        
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
                        self.add_to_history("assistant", assistant_message)
                        self.save_history()
                        yield assistant_message
                    else:
                        yield "No response content received."
                else:
                    yield f"Error: {response.status} - {await response.text()}"

class OpenAIChatSession(ChatProvider):
    def __init__(self, api_key: str, base_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [msg.to_dict() for msg in self.chat_history]
        
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
                                        yield content
                                except json.JSONDecodeError:
                                    continue
                    self.add_to_history("assistant", complete_message)
                    self.save_history()
                else:
                    yield f"Error: {response.status} - {await response.text()}"

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

class ResponseWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    db_operation = pyqtSignal(object, str, str)

    def __init__(self, chat_session, message):
        super().__init__()
        self.chat_session = chat_session
        self.message = message
        self.is_running = True

    @pyqtSlot()
    def run(self):
        asyncio.run(self._run_async())

    async def _run_async(self):
        try:
            self.db_operation.emit(self.chat_session, "user", self.message)
            async for response_chunk in self.chat_session.send_message(self.message):
                if not self.is_running:
                    break
                self.progress.emit(response_chunk)
        finally:
            self.finished.emit()

    def stop(self):
        self.is_running = False

class ParameterDialog(QDialog):
    def __init__(self, parent=None, current_params=None):
        super().__init__(parent)
        self.setWindowTitle("Set Ollama Parameters")
        self.layout = QFormLayout(self)
        self.params = {}
        for param, value in current_params.items():
            if param != "stop":
                self.params[param] = QLineEdit(str(value), self)
            else:
                self.params[param] = QLineEdit(" ".join(value) if value else "", self)
            self.layout.addRow(QLabel(f"{param}:"), self.params[param])
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal, self)
        self.layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_parameters(self):
        return {param: widget.text() for param, widget in self.params.items()}

class ChatApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_url_ollama = "http://192.168.1.82:11434/api/chat"
        self.model_url_anthropic = "https://api.anthropic.com/v1/messages"
        self.openai_base_url = "https://api.openai.com/v1/chat/completions"
        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(DB_FILE)
        self.provider_factory = ChatProviderFactory()
        self.openai_api_key = os.getenv(OPENAI_API_KEY_NAME)
        self.anthropic_api_key = os.getenv(ANTHROPIC_API_KEY_NAME)
        self.global_system_message = None
        self.chat_session = None
        self.response_thread = None
        self.response_worker = None
        self.current_message = None

        self.init_ui()
        self.load_last_chat()

    def init_ui(self):
        self.setWindowTitle('Retrochat GUI')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Mode selection
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Ollama', 'Anthropic', 'OpenAI'])
        mode_layout.addWidget(QLabel('Mode:'))
        mode_layout.addWidget(self.mode_combo)
        self.connect_button = QPushButton('Connect')
        self.connect_button.clicked.connect(self.connect_to_service)
        mode_layout.addWidget(self.connect_button)
        main_layout.addLayout(mode_layout)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setFixedHeight(50)
        input_layout.addWidget(self.input_field)
        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)

        # Command buttons
        command_layout = QHBoxLayout()
        commands = ['New Chat', 'Rename Chat', 'Delete Chat', 'List Chats', 'Open Chat', 'Reset Chat', 'Set System', 'Set Parameters']
        for command in commands:
            button = QPushButton(command)
            button.clicked.connect(lambda checked, cmd=command: self.handle_command(cmd))
            command_layout.addWidget(button)
        main_layout.addLayout(command_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def connect_to_service(self):
        mode = self.mode_combo.currentText()
        if mode == 'Ollama':
            self.connect_to_ollama()
        elif mode == 'Anthropic':
            self.connect_to_anthropic()
        elif mode == 'OpenAI':
            self.connect_to_openai()

    def connect_to_ollama(self):
        model = self.select_ollama_model()
        if model:
            self.chat_session = self.provider_factory.create_provider('Ollama', self.model_url_ollama, model, self.history_manager)
            self.chat_display.append("Connected to Ollama service.")
            self.load_chat_history()

    def connect_to_anthropic(self):
        if self.ensure_api_key('anthropic_api_key', ANTHROPIC_API_KEY_NAME):
            self.chat_session = self.provider_factory.create_provider('Anthropic', self.anthropic_api_key, self.model_url_anthropic, self.history_manager)
            self.chat_display.append("Connected to Anthropic service.")
            self.load_chat_history()

    def connect_to_openai(self):
        if self.ensure_api_key('openai_api_key', OPENAI_API_KEY_NAME):
            self.chat_session = self.provider_factory.create_provider('OpenAI', self.openai_api_key, self.openai_base_url, "gpt-4", self.history_manager)
            self.chat_display.append("Connected to OpenAI service.")
            self.load_chat_history()

    def ensure_api_key(self, key_name: str, env_var: str):
        if not getattr(self, key_name):
            api_key, ok = QInputDialog.getText(self, f"Enter {env_var}", f"{env_var} is not set. Please enter your API key:")
            if ok and api_key:
                set_key(ENV_FILE, env_var, api_key)
                load_dotenv(ENV_FILE)
                setattr(self, key_name, api_key)
                self.chat_display.append(f"{env_var} has been set and saved in the .env file.")
                return True
            else:
                self.chat_display.append(f"No API key provided. {key_name.replace('_', ' ').title()} mode cannot be used.")
                return False
        return True

    def select_ollama_model(self):
        models = self.get_ollama_models()
        if models:
            model, ok = QInputDialog.getItem(self, "Select Ollama Model", "Choose a model:", models, 0, False)
            if ok and model:
                return model
        return None

    def get_ollama_models(self):
        url = "http://192.168.1.82:11434/api/tags"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                models_info = response.json()
                if isinstance(models_info, dict) and 'models' in models_info:
                    return [model['name'] for model in models_info['models']]
            self.chat_display.append(f"Error fetching Ollama models: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            self.chat_display.append(f"Error connecting to Ollama service: {e}")
        return None

    def load_chat_history(self):
        self.chat_display.clear()
        for message in self.chat_session.chat_history:
            self.display_message(message.role, message.content)

    def display_message(self, role, content):
        if role == "user":
            self.chat_display.append(f"<b>You:</b> {content}")
        else:
            self.chat_display.append(f"<b>Assistant:</b> {content}")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def send_message(self):
        message = self.input_field.toPlainText().strip()
        if message and self.chat_session:
            self.input_field.clear()
            self.display_message("user", message)
            self.chat_display.append("")  # Add a new line after user message
            self.current_message = message
            self.process_response()

    def process_response(self):
        if not hasattr(self, 'response_thread') or self.response_thread is None:
            self.response_thread = QThread()
            self.response_worker = None

        if self.response_worker is not None:
            self.response_worker.stop()
            self.response_worker.deleteLater()

        self.response_worker = ResponseWorker(self.chat_session, self.current_message)
        self.response_worker.moveToThread(self.response_thread)

        # Connect signals
        self.response_worker.finished.connect(self.on_response_finished)
        self.response_worker.progress.connect(self.update_response)
        self.response_worker.db_operation.connect(self.handle_db_operation)
        
        # Start the thread
        self.response_thread.started.connect(self.response_worker.run)
        self.response_thread.start()

    @pyqtSlot(object, str, str)
    def handle_db_operation(self, chat_session, role, content):
        chat_session.add_to_history(role, content)

    def on_response_finished(self):
        if self.response_thread and self.response_thread.isRunning():
            self.response_thread.quit()
            self.response_thread.wait()

    def update_response(self, response_chunk):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if cursor.position() == 0 or self.chat_display.toPlainText()[-1] == '\n':
            self.chat_display.insertHtml("<b>Assistant:</b> ")
        self.chat_display.insertPlainText(response_chunk)
        self.chat_display.ensureCursorVisible()

    def handle_command(self, command):
        if command in ['Rename Chat', 'Delete Chat', 'Open Chat', 'Reset Chat', 'Set Parameters']:
            if self.chat_session is None:
                self.chat_display.append("Please connect to a service before performing this action.")
                return
        
        if command == 'New Chat':
            self.new_chat()
        elif command == 'Rename Chat':
            self.rename_chat()
        elif command == 'Delete Chat':
            self.delete_chat()
        elif command == 'List Chats':
            self.list_chats()
        elif command == 'Open Chat':
            self.open_chat()
        elif command == 'Reset Chat':
            self.reset_chat()
        elif command == 'Set System':
            self.set_system_message()
        elif command == 'Set Parameters':
            self.set_parameters()

    def new_chat(self):
        name, ok = QInputDialog.getText(self, 'New Chat', 'Enter name for the new chat:')
        if ok and name:
            self.chat_name = name
            self.history_manager.set_chat_name(name)
            self.history_manager.save_history([])
            self.chat_display.clear()
            self.chat_display.append(f"New chat '{name}' created.")
            self.save_last_chat_name(name)

    def rename_chat(self):
        new_name, ok = QInputDialog.getText(self, 'Rename Chat', 'Enter new name for the chat:')
        if ok and new_name:
            self.history_manager.rename_history(new_name)
            self.chat_display.append(f"Chat renamed to '{new_name}'")
            self.save_last_chat_name(new_name)

    def delete_chat(self):
        reply = QMessageBox.question(self, 'Delete Chat', 'Are you sure you want to delete the current chat?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.history_manager.delete_history()
            self.chat_display.clear()
            self.chat_display.append("Current chat history deleted.")

    def list_chats(self):
        chats = self.history_manager.list_chats()
        if chats:
            self.chat_display.append("Available chats:")
            for chat in chats:
                self.chat_display.append(chat)
        else:
            self.chat_display.append("No available chats.")

    def open_chat(self):
        chats = self.history_manager.list_chats()
        if chats:
            chat_name, ok = QInputDialog.getItem(self, "Open Chat", "Select a chat to open:", chats, 0, False)
            if ok and chat_name:
                self.history_manager.set_chat_name(chat_name)
                loaded_history = self.history_manager.load_history()
                
                if self.chat_session is None:
                    self.chat_display.append("Please connect to a service before opening a chat.")
                    return
                
                self.chat_session.chat_history = loaded_history
                self.chat_display.clear()
                self.chat_display.append(f"Chat '{chat_name}' opened.")
                self.load_chat_history()
                self.save_last_chat_name(chat_name)
        else:
            self.chat_display.append("No available chats to open.")

    def reset_chat(self):
        reply = QMessageBox.question(self, 'Reset Chat', 'Are you sure you want to reset the current chat?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.history_manager.clear_history()
            self.chat_session.chat_history = []
            self.chat_display.clear()
            self.chat_display.append("Chat history has been reset.")

    def set_system_message(self):
        message, ok = QInputDialog.getText(self, 'Set System Message', 'Enter the system message:')
        if ok:
            self.global_system_message = message
            if self.chat_session:
                self.chat_session.set_system_message(message)
            self.chat_display.append(f"Global system message set to: {message}")

    def set_parameters(self):
        if isinstance(self.chat_session, OllamaChatSession):
            dialog = ParameterDialog(self, self.chat_session.parameters)
            if dialog.exec():
                params = dialog.get_parameters()
                for param, value in params.items():
                    if value:
                        self.chat_session.set_parameter(param, value)
                self.chat_display.append("Ollama parameters updated.")
        else:
            self.chat_display.append("Parameter setting is only available for Ollama sessions.")

    def save_last_chat_name(self, chat_name: str):
        set_key(ENV_FILE, LAST_CHAT_NAME_KEY, chat_name)

    def load_last_chat(self):
        last_chat_name = os.getenv(LAST_CHAT_NAME_KEY, 'default')
        self.history_manager.set_chat_name(last_chat_name)
        self.chat_display.append(f"Loaded last chat: {last_chat_name}")

    def closeEvent(self, event):
        if hasattr(self, 'response_thread') and self.response_thread is not None:
            if self.response_thread.isRunning():
                self.response_worker.stop()
                self.response_thread.quit()
                self.response_thread.wait()
            
            # Disconnect all signals
            try:
                self.response_worker.finished.disconnect()
                self.response_worker.progress.disconnect()
                self.response_worker.db_operation.disconnect()
            except TypeError:
                # Ignore errors if signals were not connected
                pass
            
            try:
                self.response_thread.finished.disconnect()
            except TypeError:
                # Ignore errors if signals were not connected
                pass
            
            # Delete the worker and thread
            if self.response_worker:
                self.response_worker.deleteLater()
            if self.response_thread:
                self.response_thread.deleteLater()
        
        # Accept the close event
        event.accept()

def main():
    app = QApplication(sys.argv)
    chat_app = ChatApp()
    chat_app.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()