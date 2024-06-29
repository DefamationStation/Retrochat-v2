import os
import sys
import json
import requests
import sqlite3
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QHBoxLayout, QPushButton, QDialog, QFormLayout, QComboBox, QSpinBox, QFileDialog, QInputDialog
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor, QFont

class ConfigManager:
    CONFIG_FILENAME = "config.json"
    DEFAULT_CONFIG = {
        "baseurl": "http://",
        "ollamahost": "192.168.1.82:11434",
        "llamacpphost": "192.168.1.82:8080",
        "path": "/v1/chat/completions",
        "user_color": "#00FF00",
        "assistant_color": "#FFBF00",
        "fontsize": 18,
        "current_chat_name": "chat_1",
        "selected_model": "",
        "current_mode": "ollama",
        "openaiapikey": "",
        "window_geometry": None,
        "window_state": "normal"
    }

    @classmethod
    def load_config(cls):
        if not os.path.exists(cls.CONFIG_FILENAME):
            cls.save_config(cls.DEFAULT_CONFIG)
            return cls.DEFAULT_CONFIG
        try:
            with open(cls.CONFIG_FILENAME, 'r') as f:
                config = json.load(f)
            for key, value in cls.DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        except json.JSONDecodeError:
            print(f"Error reading {cls.CONFIG_FILENAME}. Using default configuration.")
            return cls.DEFAULT_CONFIG

    @classmethod
    def save_config(cls, config):
        with open(cls.CONFIG_FILENAME, 'w') as f:
            json.dump(config, f, indent=4)

class ChatManager:
    def __init__(self, db_name="chats.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    system_prompt TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (id)
                )
            """)

    def save_chat(self, chat_name, messages, system_prompt):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO chats (name, system_prompt) VALUES (?, ?)", (chat_name, system_prompt))
            chat_id = cursor.lastrowid

            cursor.executemany("""
                INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)
            """, [(chat_id, msg['role'], msg['content']) for msg in messages])

    def load_chat(self, chat_name):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, system_prompt FROM chats WHERE name = ?", (chat_name,))
            result = cursor.fetchone()
            
            if result:
                chat_id, system_prompt = result
                cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,))
                messages = [{'role': role, 'content': content} for role, content in cursor.fetchall()]
                return messages, system_prompt
            else:
                return [], ""

    def get_chat_list(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM chats")
            return [row[0] for row in cursor.fetchall()]

    def delete_chat(self, chat_name):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM chats WHERE name = ?", (chat_name,))
            result = cursor.fetchone()
            if result:
                chat_id = result[0]
                cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
                cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def close(self):
        self.conn.close()

class NetworkWorker(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, url, data, headers):
        super().__init__()
        self.url = url
        self.data = data
        self.headers = headers

    def run(self):
        try:
            response = requests.post(self.url, headers=self.headers, json=self.data, timeout=30)
            response.raise_for_status()
            self.response_received.emit(response.json()["choices"][0]["message"]["content"])
        except requests.RequestException as e:
            self.error_occurred.emit(f"Network error: {str(e)}")
        except KeyError as e:
            self.error_occurred.emit(f"Unexpected response format: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"An unexpected error occurred: {str(e)}")

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Settings")
        layout = QFormLayout()

        self.openaiapikey_input = QLineEdit(self.parent.config["openaiapikey"])
        layout.addRow("OpenAI API Key:", self.openaiapikey_input)

        self.baseurl_input = QLineEdit(self.parent.config["baseurl"])
        layout.addRow("Base URL:", self.baseurl_input)

        self.ollamahost_input = QLineEdit(self.parent.config["ollamahost"])
        layout.addRow("Ollama Host:", self.ollamahost_input)

        self.llamacpphost_input = QLineEdit(self.parent.config["llamacpphost"])
        layout.addRow("Llama.cpp Host:", self.llamacpphost_input)

        self.user_color_input = QLineEdit(self.parent.config["user_color"])
        layout.addRow("User Color:", self.user_color_input)

        self.assistant_color_input = QLineEdit(self.parent.config["assistant_color"])
        layout.addRow("Assistant Color:", self.assistant_color_input)

        self.font_size_input = QSpinBox()
        self.font_size_input.setRange(2, 40)
        self.font_size_input.setValue(self.parent.config["fontsize"])
        layout.addRow("Font Size:", self.font_size_input)

        self.provider_select = QComboBox()
        self.provider_select.addItems(["OpenAI", "Ollama", "Llama.cpp"])
        self.provider_select.setCurrentText(self.parent.config["current_mode"].capitalize())
        self.provider_select.currentTextChanged.connect(self.update_model_list)
        layout.addRow("Provider:", self.provider_select)

        self.model_select = QComboBox()
        layout.addRow("Model:", self.model_select)

        self.system_prompt_input = QTextEdit(self.parent.system_prompt)
        self.system_prompt_input.setFixedHeight(100)
        layout.addRow("System Prompt:", self.system_prompt_input)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        layout.addRow(self.save_button)

        self.setLayout(layout)
        self.update_model_list(self.provider_select.currentText())

    def update_model_list(self, provider):
        self.model_select.clear()
        if provider == "OpenAI":
            self.model_select.addItems(["gpt4-o"])
        elif provider == "Ollama":
            self.model_select.addItems(self.parent.ollama_models)
        elif provider == "Llama.cpp":
            self.model_select.addItem("default")  # Add more models if available

        if self.parent.config["selected_model"] in [self.model_select.itemText(i) for i in range(self.model_select.count())]:
            self.model_select.setCurrentText(self.parent.config["selected_model"])

    def save_settings(self):
        self.parent.config["openaiapikey"] = self.openaiapikey_input.text()
        self.parent.config["baseurl"] = self.baseurl_input.text()
        self.parent.config["ollamahost"] = self.ollamahost_input.text()
        self.parent.config["llamacpphost"] = self.llamacpphost_input.text()
        self.parent.config["user_color"] = self.user_color_input.text()
        self.parent.config["assistant_color"] = self.assistant_color_input.text()
        self.parent.config["fontsize"] = self.font_size_input.value()
        self.parent.config["current_mode"] = self.provider_select.currentText().lower()
        self.parent.config["selected_model"] = self.model_select.currentText()
        self.parent.system_prompt = self.system_prompt_input.toPlainText()

        ConfigManager.save_config(self.parent.config)
        self.parent.apply_settings()
        self.parent.chat_manager.save_chat(self.parent.current_chat_name, self.parent.messages, self.parent.system_prompt)
        self.parent.ollama_models = self.parent.get_ollama_models()  # Refresh Ollama models
        self.close()

class Chatbox(QWidget):
    def __init__(self):
        super().__init__()
        self.config = ConfigManager.load_config()
        self.chat_manager = ChatManager()
        self.current_chat_name = self.config.get("current_chat_name", "chat_1")
        self.messages, self.system_prompt = self.chat_manager.load_chat(self.current_chat_name)
        self.ollama_models = self.get_ollama_models()
        self.initUI()
        self.apply_settings()

    def initUI(self):
        layout = QVBoxLayout()

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.user_input)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.show_settings)
        input_layout.addWidget(self.settings_button)

        self.new_chat_button = QPushButton("New Chat")
        self.new_chat_button.clicked.connect(self.new_chat)
        input_layout.addWidget(self.new_chat_button)

        self.load_chat_button = QPushButton("Load Chat")
        self.load_chat_button.clicked.connect(self.load_chat)
        input_layout.addWidget(self.load_chat_button)

        self.delete_chat_button = QPushButton("Delete Chat")
        self.delete_chat_button.clicked.connect(self.delete_chat)
        input_layout.addWidget(self.delete_chat_button)

        layout.addLayout(input_layout)

        self.setLayout(layout)
        self.setWindowTitle("AI Chat")
        self.resize(800, 600)

    def apply_settings(self):
        self.chat_history.setStyleSheet(f"background-color: black; color: {self.config['user_color']};")
        self.chat_history.setFont(QFont("Courier", self.config["fontsize"]))
        self.user_input.setFont(QFont("Courier", self.config["fontsize"]))
        self.messages, self.system_prompt = self.chat_manager.load_chat(self.current_chat_name)
        self.load_chat_history()

    def show_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec()

    def send_message(self):
        user_message = self.user_input.text()
        if not user_message:
            return

        self.messages.append({"role": "user", "content": user_message})
        self.chat_history.append(f"<font color='{self.config['user_color']}'>User: {user_message}</font>")
        self.user_input.clear()

        self.chat_manager.save_chat(self.current_chat_name, self.messages, self.system_prompt)

        if self.config["current_mode"] == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['openaiapikey']}"
            }
        elif self.config["current_mode"] == "ollama":
            url = f"{self.config['baseurl']}{self.config['ollamahost']}{self.config['path']}"
            headers = {"Content-Type": "application/json"}
        elif self.config["current_mode"] == "llama.cpp":
            url = f"{self.config['baseurl']}{self.config['llamacpphost']}{self.config['path']}"
            headers = {"Content-Type": "application/json"}
        else:
            self.handle_error("Invalid mode selected")
            return

        data = {
            "model": self.config["selected_model"],
            "messages": [{"role": "system", "content": self.system_prompt}] + self.messages
        }

        self.worker = NetworkWorker(url, data, headers)
        self.worker.response_received.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def get_ollama_models(self):
        try:
            url = f"{self.config['baseurl']}{self.config['ollamahost']}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.RequestException as e:
            print(f"Error fetching Ollama models: {e}")
            return ["default"]

    def handle_response(self, response):
        self.messages.append({"role": "assistant", "content": response})
        self.chat_history.append(f"<font color='{self.config['assistant_color']}'>Assistant: {response}</font>")
        self.chat_manager.save_chat(self.current_chat_name, self.messages, self.system_prompt)

    def handle_error(self, error):
        self.chat_history.append(f"<font color='red'>Error: {error}</font>")

    def load_chat_history(self):
        self.chat_history.clear()
        for message in self.messages:
            color = self.config['user_color'] if message['role'] == 'user' else self.config['assistant_color']
            self.chat_history.append(f"<font color='{color}'>{message['role'].capitalize()}: {message['content']}</font>")

    def new_chat(self):
        self.messages = []
        self.chat_history.clear()
        chat_list = self.chat_manager.get_chat_list()
        new_chat_name = f"chat_{len(chat_list) + 1}"
        self.current_chat_name = new_chat_name
        self.config["current_chat_name"] = new_chat_name
        ConfigManager.save_config(self.config)
        self.chat_manager.save_chat(new_chat_name, self.messages, self.system_prompt)

    def load_chat(self):
        chat_list = self.chat_manager.get_chat_list()
        chat_name, ok = QInputDialog.getItem(self, "Load Chat", "Select a chat to load:", chat_list, 0, False)
        if ok and chat_name:
            self.current_chat_name = chat_name
            self.config["current_chat_name"] = chat_name
            self.messages, self.system_prompt = self.chat_manager.load_chat(chat_name)
            ConfigManager.save_config(self.config)
            self.load_chat_history()

    def delete_chat(self):
        chat_list = self.chat_manager.get_chat_list()
        chat_name, ok = QInputDialog.getItem(self, "Delete Chat", "Select a chat to delete:", chat_list, 0, False)
        if ok and chat_name:
            self.chat_manager.delete_chat(chat_name)
            if chat_name == self.current_chat_name:
                self.new_chat()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        chat = Chatbox()
        chat.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)