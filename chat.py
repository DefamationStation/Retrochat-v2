import os
import sys
import time
import json
import requests
from colorama import init, Fore, Style
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("ANTHROPIC_API_KEY")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text, delay=0.01, color=Fore.WHITE):
    for char in text:
        sys.stdout.write(color + char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Abstract base class for chat sessions
class ChatSession(ABC):
    def __init__(self):
        self.chat_history = []

    @abstractmethod
    def send_message(self, message):
        pass

    def add_to_history(self, role, content):
        self.chat_history.append({"role": role, "content": content})

# Ollama chat session implementation
class OllamaChatSession(ChatSession):
    def __init__(self, model_url, model):
        super().__init__()
        self.model_url = model_url
        self.model = model

    def send_message(self, message):
        self.add_to_history("user", message)
        data = {
            "model": self.model,
            "messages": self.chat_history
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
                        if response_json.get('done', False):
                            break
                self.add_to_history("assistant", complete_message)
                return complete_message
            except Exception as e:
                return f"Error processing response: {str(e)}"
        else:
            return f"Error: {response.status_code} - {response.text}"

# Anthropic chat session implementation
class AnthropicChatSession(ChatSession):
    def __init__(self, api_key, model_url):
        super().__init__()
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
            assistant_message = response.json()['content'][0]['text']
            self.add_to_history("assistant", assistant_message)
            return assistant_message
        else:
            return f"Error: {response.status_code} - {response.text}"

# Main application class
class ChatApp:
    def __init__(self):
        self.model_url_ollama = "http://192.168.1.82:11434/api/chat"
        self.model_url_anthropic = "https://api.anthropic.com/v1/messages"

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

    def start(self):
        clear_screen()
        print_slow("Welcome to the Chat Program!", color=Fore.GREEN)
        print_slow("Select the mode:\n1. Ollama\n2. Anthropic", color=Fore.CYAN)
        
        mode = input(Fore.GREEN + "Enter 1 or 2: ").strip()

        if mode == '1':
            model = self.select_ollama_model()
            if not model:
                return
            session = OllamaChatSession(self.model_url_ollama, model)

        elif mode == '2':
            if not API_KEY:
                print_slow("Error: ANTHROPIC_API_KEY not found in environment variables.", color=Fore.RED)
                print_slow("Please set your API key as an environment variable named ANTHROPIC_API_KEY.", color=Fore.YELLOW)
                return
            session = AnthropicChatSession(API_KEY, self.model_url_anthropic)

        else:
            print_slow("Invalid selection. Please restart the program and choose a valid mode.", color=Fore.RED)
            return

        while True:
            user_input = input(Fore.GREEN).strip()
            if user_input.lower() == '/exit':
                print_slow("Thank you for chatting. Goodbye!", color=Fore.CYAN)
                break
            
            response = session.send_message(user_input)
            print_slow(response, color=Fore.YELLOW)

if __name__ == "__main__":
    app = ChatApp()
    app.start()
