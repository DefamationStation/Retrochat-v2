import os
import sys
import time
import requests
from colorama import init, Fore
from dotenv import load_dotenv, set_key
from config import ConfigManager
from database import ChatHistoryManager
from providers import DynamicProviderRegistry
from commands.chat_commands import CommandHandler
from utils import clear_screen, print_slow

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

class ChatApp:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(self.config_manager.get_db_file(), self.chat_name)

        providers_path = os.path.join(os.path.dirname(__file__), 'providers')
        self.provider_registry = DynamicProviderRegistry(providers_path)
        
        self.command_handler = CommandHandler(self.history_manager, self)

    def select_provider(self):
        print_slow("Select the chat provider:\n1. Ollama\n2. Anthropic\n3. OpenAI GPT-4o", color=Fore.CYAN)
        mode = input(Fore.GREEN + "Enter 1, 2 or 3: ").strip()

        if mode == '1':
            provider_name = 'OllamaProvider'
            models = self.get_ollama_models()
            if not models:
                return None, None
            print_slow("Available Ollama models:", color=Fore.GREEN)
            for idx, model in enumerate(models):
                print_slow(f"{idx + 1}. {model}", color=Fore.CYAN)
            try:
                choice = int(input(Fore.GREEN + "Select a model number: ")) - 1
                if 0 <= choice < len(models):
                    model = models[choice]
                else:
                    print_slow("Invalid choice.", color=Fore.RED)
                    return None, None
            except ValueError:
                print_slow("Invalid input. Please enter a number.", color=Fore.RED)
                return None, None

            model_url = self.config_manager.get_model_url('ollama_chat')
            return provider_name, {"model_url": model_url, "model": model}

        elif mode == '2':
            provider_name = 'AnthropicProvider'
            if not self.ensure_anthropic_api_key():
                return None, None
            model_url = self.config_manager.get_model_url('anthropic_chat')
            return provider_name, {"api_key": self.config_manager.get_api_key('ANTHROPIC_API_KEY'), "model_url": model_url}

        elif mode == '3':
            provider_name = 'OpenaiGpt4oProvider'
            api_key = self.config_manager.get_api_key('OPENAI_API_KEY')
            if not api_key:
                print_slow(f"API key for OpenAI not found. Please enter your OpenAI API key.", color=Fore.CYAN)
                api_key = input(Fore.GREEN + "Enter your OpenAI API key: ").strip()
                if api_key:
                    self.config_manager.set_api_key('OPENAI_API_KEY', api_key)
                else:
                    print_slow("No API key provided. OpenAI GPT-4o mode cannot be used.", color=Fore.RED)
                    return None, None
            return provider_name, {"api_key": api_key}
        
        else:
            print_slow("Invalid selection. Please restart the program and choose a valid mode.", color=Fore.RED)
            return None, None

    def get_ollama_models(self):
        url = self.config_manager.get_model_url('ollama_models')
        response = requests.get(url)
        if response.status_code == 200:
            models_info = response.json()
            return [model['name'] for model in models_info.get('models', [])]
        else:
            print_slow(f"Error fetching Ollama models: {response.status_code} - {response.text}", color=Fore.RED)
            return []

    def ensure_anthropic_api_key(self):
        api_key = self.config_manager.get_api_key('ANTHROPIC_API_KEY')
        if not api_key:
            print_slow(f"API key for Anthropic not found. Please enter your Anthropic API key.", color=Fore.CYAN)
            api_key = input(Fore.GREEN + "Enter your Anthropic API key: ").strip()
            if api_key:
                self.config_manager.set_api_key('ANTHROPIC_API_KEY', api_key)
                return True
            else:
                print_slow("No API key provided. Anthropic mode cannot be used.", color=Fore.RED)
                return False
        return True

    def start(self):
        try:
            clear_screen()
            print_slow("Welcome to the Chat Program!", color=Fore.GREEN)

            self.load_last_chat()  # Load last used chat session

            provider_name, provider_kwargs = self.select_provider()
            if not provider_name:
                return

            session = self.provider_registry.get_provider(provider_name, self.history_manager, **provider_kwargs)
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

    def load_last_chat(self):
        last_chat_name = self.config_manager.get_last_chat_name()
        if last_chat_name:
            self.chat_name = last_chat_name
            self.history_manager.set_chat_name(last_chat_name)
            print_slow(f"Loaded last chat session: {last_chat_name}", color=Fore.CYAN)

    def set_chat_name(self, chat_name):
        self.chat_name = chat_name
        self.history_manager.set_chat_name(chat_name)
        self.config_manager.set_last_chat_name(chat_name)

if __name__ == "__main__":
    app = ChatApp()
    app.start()
