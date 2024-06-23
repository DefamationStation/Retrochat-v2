import os
import sys
import time
import json
import requests
import sqlite3
from colorama import init, Fore
from dotenv import load_dotenv, set_key
from abc import ABC, abstractmethod
from typing import List, Dict

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Constants for user-specific storage
USER_HOME = os.path.expanduser('~')
RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
os.makedirs(RETROCHAT_DIR, exist_ok=True)

ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
DB_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.db')
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
OPENAI_API_KEY_NAME = "OPENAI_API_KEY"

# Utility functions
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text, delay=0.001, color=Fore.WHITE):
    for char in text:
        sys.stdout.write(color + char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_streaming(text, color=Fore.WHITE):
    sys.stdout.write(color + text)
    sys.stdout.flush()

# Chat History Manager using SQLite
class ChatHistoryManager:
    def __init__(self, db_file: str, chat_name: str = 'default'):
        self.db_file = db_file
        self.chat_name = chat_name
        self.conn = sqlite3.connect(self.db_file)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_name TEXT NOT NULL UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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

    def _get_session_id(self, chat_name: str):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM chat_sessions WHERE chat_name = ?', (chat_name,))
        session = cursor.fetchone()
        if session:
            return session[0]
        else:
            cursor.execute('INSERT INTO chat_sessions (chat_name) VALUES (?)', (chat_name,))
            return cursor.lastrowid

    def set_chat_name(self, chat_name: str):
        self.chat_name = chat_name

    def save_history(self, history: List[Dict[str, str]]):
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
            for message in history:
                self.conn.execute('''
                    INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)
                ''', (session_id, message['role'], message['content']))

    def load_history(self) -> List[Dict[str, str]]:
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
        messages = cursor.fetchall()
        return [{'role': role, 'content': content} for role, content in messages]

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
            print_slow(f"Error: A chat with the name '{new_name}' already exists.", color=Fore.RED)

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

# Abstract base class for chat providers
class ChatProvider(ABC):
    def __init__(self, history_manager: ChatHistoryManager):
        self.history_manager = history_manager
        self.chat_history = history_manager.load_history()

    @abstractmethod
    def send_message(self, message: str):
        pass

    def add_to_history(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def save_history(self):
        self.history_manager.save_history(self.chat_history)

    def display_history(self):
        if not self.chat_history:
            print_slow("No previous chat history.", color=Fore.CYAN)
        else:
            print_slow("Chat history loaded from previous session:", color=Fore.CYAN)
            for entry in self.chat_history:
                role_color = Fore.GREEN if entry['role'] == 'user' else Fore.YELLOW
                print_slow(f"{entry['content']}", color=role_color)

# Ollama chat session implementation with streaming
class OllamaChatSession(ChatProvider):
    def __init__(self, model_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model

    def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": self.model,
            "messages": self.chat_history,
            "stream": True  # Enable streaming
        }
        response = requests.post(self.model_url, json=data, stream=True)
        if response.status_code == 200:
            complete_message = ""
            try:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        response_json = json.loads(line)
                        message_content = response_json.get('message', {}).get('content', '')
                        complete_message += message_content
                        print_streaming(message_content, color=Fore.YELLOW)
                        if response_json.get('done', False):
                            break
                self.add_to_history("assistant", complete_message)
                self.save_history()
                print()  # Print new line after the complete message
            except Exception as e:
                print_slow(f"Error processing response: {str(e)}", color=Fore.RED)
        else:
            print_slow(f"Error: {response.status_code} - {response.text}", color=Fore.RED)

# Anthropic chat session implementation
class AnthropicChatSession(ChatProvider):
    def __init__(self, api_key: str, model_url: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model_url = model_url

    def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 4096,
            "temperature": 0.0,
            "system": "Keep your answers short and to the point.",
            "messages": self.chat_history
        }
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        response = requests.post(self.model_url, json=data, headers=headers)
        if response.status_code == 200:
            try:
                response_json = response.json()
                assistant_message = response_json.get('content', [{}])[0].get('text', '')
                if assistant_message:
                    self.add_to_history("assistant", assistant_message)
                    self.save_history()
                    return assistant_message
                else:
                    return "No response content received."
            except Exception as e:
                return f"Error processing response: {str(e)}"
        else:
            return f"Error: {response.status_code} - {response.text}"

# OpenAI chat session implementation
class OpenAIChatSession(ChatProvider):
    def __init__(self, api_key: str, base_url: str, model: str, history_manager: ChatHistoryManager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": self.model,
            "messages": [{"role": msg['role'], "content": msg['content']} for msg in self.chat_history],
            "stream": True  # Enable streaming
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            complete_message = ""
            buffer = ""

            for chunk in response.iter_lines(decode_unicode=True):
                if chunk:
                    if chunk.strip() == "data: [DONE]":
                        break  # Skip processing the "[DONE]" message

                    try:
                        buffer += chunk
                        if chunk.startswith("data: "):
                            json_str = buffer[6:]  # Strip the "data: " prefix
                            buffer = ""  # Clear the buffer

                            if json_str.strip():  # Ensure it's not just whitespace
                                response_json = json.loads(json_str)  # Parse the JSON
                                
                                # Extract content if available
                                content = response_json['choices'][0]['delta'].get('content', '')
                                if content:
                                    complete_message += content
                                    print_streaming(content, color=Fore.YELLOW)
                    except json.JSONDecodeError as e:
                        print_slow(f"Error decoding JSON: {chunk} - {str(e)}", color=Fore.RED)
                        continue

            self.add_to_history("assistant", complete_message)
            self.save_history()
            print()  # Print new line after the complete message
        except Exception as e:
            print_slow(f"Error processing response: {str(e)}", color=Fore.RED)

# Provider Registry for dynamic provider management
class ProviderRegistry:
    def __init__(self):
        self.providers = {}

    def register_provider(self, name: str, provider_class: type):
        self.providers[name] = provider_class

    def get_provider(self, name: str, *args, **kwargs) -> ChatProvider:
        provider_class = self.providers.get(name)
        if not provider_class:
            raise ValueError(f"Provider '{name}' not registered.")
        return provider_class(*args, **kwargs)

# Initialize the provider registry and register the providers
provider_registry = ProviderRegistry()
provider_registry.register_provider('Ollama', OllamaChatSession)
provider_registry.register_provider('Anthropic', AnthropicChatSession)
provider_registry.register_provider('OpenAI', OpenAIChatSession)

# Command handler for the application
class CommandHandler:
    def __init__(self, history_manager: ChatHistoryManager):
        self.history_manager = history_manager

    def handle_command(self, command: str, session: ChatProvider):
        command_parts = command.split(' ', 2)
        if len(command_parts) < 2:
            print_slow("Command requires a value. Usage examples:\n"
                       "/chat rename <new_name>\n"
                       "/chat delete\n"
                       "/chat new <chat_name>\n"
                       "/chat list\n"
                       "/chat open <chat_name>", color=Fore.RED)
            return

        cmd = command_parts[0]
        sub_cmd = command_parts[1]

        if cmd == '/chat':
            if sub_cmd == 'rename':
                if len(command_parts) == 3:
                    new_name = command_parts[2].strip()
                    self.history_manager.rename_history(new_name)
                    print_slow(f"Chat renamed to '{new_name}'", color=Fore.CYAN)
                else:
                    print_slow("Please provide a new name for the chat. Usage: /chat rename <new_name>", color=Fore.RED)

            elif sub_cmd == 'delete':
                self.history_manager.delete_history()
                print_slow("Current chat history deleted.", color=Fore.CYAN)

            elif sub_cmd == 'new':
                if len(command_parts) == 3:
                    new_name = command_parts[2].strip()
                    self.history_manager.set_chat_name(new_name)
                    self.history_manager.save_history([])  # Start with an empty history
                    print_slow(f"New chat '{new_name}' created.", color=Fore.CYAN)
                else:
                    print_slow("Please provide a name for the new chat. Usage: /chat new <chat_name>", color=Fore.RED)

            elif sub_cmd == 'reset':
                self.history_manager.clear_history()
                session.chat_history = []
                print_slow("Chat history has been reset.", color=Fore.CYAN)

            elif sub_cmd == 'list':
                chats = self.history_manager.list_chats()
                if chats:
                    print_slow("Available chats:", color=Fore.CYAN)
                    for chat in chats:
                        print_slow(chat, color=Fore.GREEN)
                else:
                    print_slow("No available chats.", color=Fore.RED)

            elif sub_cmd == 'open':
                if len(command_parts) == 3:
                    chat_name = command_parts[2].strip()
                    if chat_name in self.history_manager.list_chats():
                        self.history_manager.set_chat_name(chat_name)
                        session.chat_history = self.history_manager.load_history()
                        print_slow(f"Chat '{chat_name}' opened.", color=Fore.CYAN)
                        session.display_history()
                    else:
                        print_slow(f"Chat '{chat_name}' does not exist.", color=Fore.RED)
                else:
                    print_slow("Please provide the name of the chat to open. Usage: /chat open <chat_name>", color=Fore.RED)

            else:
                print_slow("Unknown command. Available commands are:\n"
                           "/chat rename <new_name>\n"
                           "/chat delete\n"
                           "/chat new <chat_name>\n"
                           "/chat reset\n"
                           "/chat list\n"
                           "/chat open <chat_name>", color=Fore.RED)

        else:
            print_slow("Unknown command.", color=Fore.RED)

# Main application class
class ChatApp:
    def __init__(self):
        self.model_url_ollama = "http://192.168.1.82:11434/api/chat"
        self.model_url_anthropic = "https://api.anthropic.com/v1/messages"
        self.openai_base_url = "https://api.openai.com/v1/chat/completions"
        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(DB_FILE, self.chat_name)
        self.command_handler = CommandHandler(self.history_manager)
        self.openai_api_key = None
        self.anthropic_api_key = None

        # Load environment variables and initialize API keys if present
        self.load_env_variables()

    def load_env_variables(self):
        if os.path.exists(ENV_FILE):
            load_dotenv(ENV_FILE)
            self.openai_api_key = os.getenv(OPENAI_API_KEY_NAME)
            self.anthropic_api_key = os.getenv(ANTHROPIC_API_KEY_NAME)

    def ensure_openai_api_key(self):
        if not self.openai_api_key:
            print_slow("OpenAI API key is not set. Please enter your OpenAI API key.", color=Fore.CYAN)
            self.openai_api_key = input(Fore.GREEN + "Enter your OpenAI API key: ").strip()
            if self.openai_api_key:
                set_key(ENV_FILE, OPENAI_API_KEY_NAME, self.openai_api_key)
                load_dotenv(ENV_FILE)  # Reload .env to update environment with new key
                print_slow("OpenAI API key has been set and saved in the .env file.", color=Fore.CYAN)
            else:
                print_slow("No API key provided. OpenAI mode cannot be used.", color=Fore.RED)
                return False
        return True

    def ensure_anthropic_api_key(self):
        if not self.anthropic_api_key:
            print_slow(f"{ANTHROPIC_API_KEY_NAME} is not set. Please enter your Anthropic API key.", color=Fore.CYAN)
            self.anthropic_api_key = input(Fore.GREEN + "Enter your Anthropic API key: ").strip()
            if self.anthropic_api_key:
                set_key(ENV_FILE, ANTHROPIC_API_KEY_NAME, self.anthropic_api_key)
                load_dotenv(ENV_FILE)  # Reload .env to update environment with new key
                print_slow(f"{ANTHROPIC_API_KEY_NAME} has been set and saved in {ENV_FILE}.", color=Fore.CYAN)
            else:
                print_slow("No API key provided. Anthropic mode cannot be used.", color=Fore.RED)
                return False
        return True

    def select_ollama_model(self) -> List[str]:
        url = "http://192.168.1.82:11434/api/tags"
        response = requests.get(url)
        if response.status_code == 200:
            models_info = response.json()
            if isinstance(models_info, dict) and 'models' in models_info:
                model_names = [model['name'] for model in models_info['models']]
                return model_names
            else:
                print_slow("Unexpected API response structure. Expected a dictionary with 'models'.", color=Fore.RED)
                return []
        else:
            print_slow(f"Error fetching Ollama models: {response.status_code} - {response.text}", color=Fore.RED)
            return []
        
    def start(self):
        try:
            clear_screen()
            print_slow("Welcome to the Chat Program!", color=Fore.GREEN)
            print_slow("Select the mode:\n1. Ollama\n2. Anthropic\n3. OpenAI", color=Fore.CYAN)

            mode = input(Fore.GREEN + "Enter 1, 2, or 3: ").strip()

            if mode == '1':
                models = self.select_ollama_model()
                if not models:
                    return
                print_slow("Available Ollama models:", color=Fore.GREEN)
                for idx, model in enumerate(models):
                    print_slow(f"{idx + 1}. {model}", color=Fore.CYAN)
                choice = int(input(Fore.GREEN + "Select a model number: ")) - 1
                if 0 <= choice < len(models):
                    selected_model = models[choice]
                else:
                    print_slow("Invalid choice. Please restart and try again.", color=Fore.RED)
                    return
                session = provider_registry.get_provider('Ollama', self.model_url_ollama, selected_model, self.history_manager)

            elif mode == '2':
                if not self.ensure_anthropic_api_key():
                    return
                session = provider_registry.get_provider('Anthropic', self.anthropic_api_key, self.model_url_anthropic, self.history_manager)

            elif mode == '3':
                if not self.ensure_openai_api_key():
                    return
                session = provider_registry.get_provider('OpenAI', self.openai_api_key, self.openai_base_url, "gpt-4o", self.history_manager)

            else:
                print_slow("Invalid selection. Please restart the program and choose a valid mode.", color=Fore.RED)
                return

            session.display_history()  # Display the loaded chat history

            while True:
                try:
                    user_input = input(Fore.GREEN).strip()
                    if user_input.lower() == '/exit':
                        print_slow("Thank you for chatting. Goodbye!", color=Fore.CYAN)
                        session.save_history()
                        break
                    elif user_input.startswith('/'):
                        self.command_handler.handle_command(user_input, session)
                        continue

                    response = session.send_message(user_input)
                    if response:  # Ensure response is not None before printing
                        print_slow(response, color=Fore.YELLOW)
                except KeyboardInterrupt:
                    print_slow("\nInterrupted by user. Exiting...", color=Fore.CYAN)
                    session.save_history()  # Save history before exiting
                    break

        except KeyboardInterrupt:
            print_slow("\nProgram interrupted by user. Exiting...", color=Fore.CYAN)
        except Exception as e:
            print_slow(f"\nAn unexpected error occurred: {e}", color=Fore.RED)

# Main function
if __name__ == "__main__":
    app = ChatApp()
    app.start()
