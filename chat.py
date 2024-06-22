import os
import sys
import time
import json
import requests
from colorama import init, Fore, Style
from dotenv import load_dotenv, set_key
from abc import ABC, abstractmethod

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Constants for user-specific storage
USER_HOME = os.path.expanduser('~')
RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
if not os.path.exists(RETROCHAT_DIR):
    os.makedirs(RETROCHAT_DIR)

ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
CHAT_HISTORY_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.json')
API_KEY_NAME = "ANTHROPIC_API_KEY"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text, delay=0.01, color=Fore.WHITE):
    if text:
        for char in text:
            sys.stdout.write(color + char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

# Print directly to the UI without delay
def print_streaming(text, color=Fore.WHITE):
    sys.stdout.write(color + text)
    sys.stdout.flush()

# Chat History Manager
class ChatHistoryManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_history(self, history):
        with open(self.file_path, 'w') as f:
            json.dump(history, f, indent=4)

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return []

    def clear_history(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# Abstract base class for chat sessions
class ChatSession(ABC):
    def __init__(self, history_manager):
        self.chat_history = history_manager.load_history()
        self.history_manager = history_manager

    @abstractmethod
    def send_message(self, message):
        pass

    def add_to_history(self, role, content):
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
class OllamaChatSession(ChatSession):
    def __init__(self, model_url, model, history_manager):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model

    def send_message(self, message):
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
                        print_streaming(message_content, color=Fore.YELLOW)  # Print each chunk as it comes in
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
class AnthropicChatSession(ChatSession):
    def __init__(self, api_key, model_url, history_manager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model_url = model_url

    def send_message(self, message):
        self.add_to_history("user", message)
        data = {
            "model": "claude-3-5-sonnet-20240620",  # Use the appropriate model
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

# Main application class
class ChatApp:
    def __init__(self):
        self.model_url_ollama = "http://192.168.1.82:11434/api/chat"
        self.model_url_anthropic = "https://api.anthropic.com/v1/messages"
        self.history_manager = ChatHistoryManager(CHAT_HISTORY_FILE)

    def get_ollama_models(self):
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

    def select_ollama_model(self):
        models = self.get_ollama_models()
        if not models:
            print_slow("No Ollama models available.", color=Fore.RED)
            return None

        print_slow("Available Ollama models:", color=Fore.GREEN)
        for idx, model in enumerate(models):
            print_slow(f"{idx + 1}. {model}", color=Fore.CYAN)

        try:
            choice = int(input(Fore.GREEN + "Select a model number: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print_slow("Invalid choice. Please try again.", color=Fore.RED)
                return self.select_ollama_model()
        except ValueError:
            print_slow("Invalid input. Please enter a number.", color=Fore.RED)
            return self.select_ollama_model()

    def ensure_anthropic_api_key(self):
        if not os.path.exists(ENV_FILE):
            print_slow(f"{ENV_FILE} not found. Creating a new one...", color=Fore.CYAN)
            open(ENV_FILE, 'a').close()

        load_dotenv(ENV_FILE)
        api_key = os.getenv(API_KEY_NAME)

        if not api_key:
            print_slow(f"{API_KEY_NAME} is not set. Please enter your Anthropic API key.", color=Fore.CYAN)
            api_key = input(Fore.GREEN + "Enter your Anthropic API key: ").strip()
            if api_key:
                set_key(ENV_FILE, API_KEY_NAME, api_key)
                load_dotenv(ENV_FILE)  # Reload .env to update environment with new key
                print_slow(f"{API_KEY_NAME} has been set and saved in {ENV_FILE}.", color=Fore.CYAN)
            else:
                print_slow("No API key provided. Anthropic mode cannot be used.", color=Fore.RED)
                return False
        return True

    def start(self):
        clear_screen()
        print_slow("Welcome to the Chat Program!", color=Fore.GREEN)
        print_slow("Select the mode:\n1. Ollama\n2. Anthropic", color=Fore.CYAN)
        
        mode = input(Fore.GREEN + "Enter 1 or 2: ").strip()

        if mode == '1':
            model = self.select_ollama_model()
            if not model:
                return
            session = OllamaChatSession(self.model_url_ollama, model, self.history_manager)

        elif mode == '2':
            if not self.ensure_anthropic_api_key():
                return
            session = AnthropicChatSession(os.getenv(API_KEY_NAME), self.model_url_anthropic, self.history_manager)

        else:
            print_slow("Invalid selection. Please restart the program and choose a valid mode.", color=Fore.RED)
            return

        session.display_history()  # Display the loaded chat history

        while True:
            user_input = input(Fore.GREEN).strip()
            if user_input.lower() == '/exit':
                print_slow("Thank you for chatting. Goodbye!", color=Fore.CYAN)
                session.save_history()  # Save history before exiting
                break
            elif user_input.lower() == '/chat reset':
                self.history_manager.clear_history()
                session.chat_history = []
                print_slow("Chat history has been reset.", color=Fore.CYAN)
                continue

            response = session.send_message(user_input)
            if response:  # Ensure response is not None before printing
                print_slow(response, color=Fore.YELLOW)

if __name__ == "__main__":
    app = ChatApp()
    app.start()
