import contextlib
import io
import requests
import aiohttp
from typing import Optional
from utils.console import console
from utils.env_manager import EnvManager
from utils.suppress_logging import SuppressLogging
from config import Config
from providers.openrouter import OpenRouterChatSession
from providers.anthropic import AnthropicChatSession
from providers.openai import OpenAIChatSession
from providers.google import GoogleChatSession
from providers.ollama import OllamaChatSession
from providers.oobabooga import OobaboogaChatSession
from providers.lmstudio import LMStudioChatSession
from providers.unified_session_manager import UnifiedSessionManager


class SessionManager:
    def __init__(self, chat_app):
        self.chat_app = chat_app
        self.unified_manager = UnifiedSessionManager(chat_app)

    async def ensure_api_key(self, key_name: str, env_var: str):
        api_key = EnvManager.get_env_variable(env_var)
        if not api_key:
            console.print(f"{env_var} is not set. Please enter your API key.", style="cyan")
            api_key = await self.chat_app.input_handler.get_single_input(f"Enter your {env_var}")
            if api_key:
                EnvManager.set_env_variable(env_var, api_key)
                console.print(f"{env_var} has been set and saved in the .env file.", style="cyan")
                return True
            else:
                console.print(f"No API key provided. {key_name.replace('_', ' ').title()} mode cannot be used.", style="bold red")
                return False
        return True

    async def ensure_ollama_connection(self):
        ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
        ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
        url = f"http://{ollama_ip}:{ollama_port}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            console.print(f"Unable to connect to Ollama at {ollama_ip}:{ollama_port}", style="bold red")
            new_ip = await self.chat_app.input_handler.get_single_input("Enter Ollama IP (press Enter for localhost)")
            new_port = await self.chat_app.input_handler.get_single_input("Enter Ollama port (press Enter for 11434)")
            
            ollama_ip = new_ip or 'localhost'
            ollama_port = new_port or '11434'
            
            EnvManager.set_env_variable(Config.OLLAMA_IP_KEY, ollama_ip)
            EnvManager.set_env_variable(Config.OLLAMA_PORT_KEY, ollama_port)
            
            console.print(f"Ollama connection details updated and saved in the .env file.", style="cyan")
            return await self.ensure_ollama_connection()

    async def select_openrouter_model(self) -> Optional[str]:
        models = OpenRouterChatSession.get_available_models()
        console.print("Available OpenRouter models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = await self.chat_app.input_handler.get_single_input("Select a model number")
        try:
            return models[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None

    async def select_ollama_model(self) -> Optional[str]:
        ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
        ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
        url = f"http://{ollama_ip}:{ollama_port}/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_info = await response.json()
                    if isinstance(models_info, dict) and 'models' in models_info:
                        model_names = [model['name'] for model in models_info['models']]
                        console.print("Available Ollama models:", style="cyan")
                        for idx, model in enumerate(model_names):
                            console.print(f"{idx + 1}. {model}", style="green")
                        choice = await self.chat_app.input_handler.get_single_input("Select a model")
                        try:
                            return model_names[int(choice) - 1]
                        except (ValueError, IndexError):
                            console.print("Invalid selection. Please try again.", style="bold red")
                            return None
                    else:
                        console.print("Unexpected API response structure.", style="bold red")
                else:
                    console.print(f"Error fetching Ollama models: {response.status} - {await response.text()}", style="bold red")
        return None

    async def select_anthropic_model(self) -> Optional[str]:
        models = ["claude-3-5-sonnet-20241022"]
        console.print("Available Anthropic models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = await self.chat_app.input_handler.get_single_input("Select a model number")
        try:
            return models[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None

    async def select_openai_model(self) -> Optional[str]:
        models = ["gpt-4o-mini", "chatgpt-4o-latest", "gpt-4o", "o1-preview", "o1-mini"]
        console.print("Available OpenAI models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = await self.chat_app.input_handler.get_single_input("Select a model number")
        try:
            return models[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None
    
    async def select_google_model(self) -> Optional[str]:
        models = ["gemini-2.0-flash-exp", "gemini-1.5-flash-8b"]
        console.print("Available Google Gemini models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = await self.chat_app.input_handler.get_single_input("Select a model number")
        try:
            return models[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None

    async def select_oobabooga_character(self) -> Optional[str]:
        characters = ["Example", "Assistant", "Chatbot", "Custom"]  # Add more characters as needed
        console.print("Available Oobabooga characters:", style="cyan")
        for idx, character in enumerate(characters):
            console.print(f"{idx + 1}. {character}", style="green")
        choice = await self.chat_app.input_handler.get_single_input("Select a character number")
        try:
            selected = characters[int(choice) - 1]
            if selected == "Custom":
                return await self.chat_app.input_handler.get_single_input("Enter custom character name")
            return selected
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None

    async def select_lmstudio_model(self, base_url: str) -> Optional[str]:
        """Select an LM Studio model from available models."""
        models = await LMStudioChatSession.get_available_models(base_url)
        
        if not models:
            console.print("No models available from LM Studio. Make sure LM Studio is running and has models loaded.", style="bold red")
            return None
            
        console.print("Available LM Studio models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        
        choice = await self.chat_app.input_handler.get_single_input("Select a model number")
        try:
            model_index = int(choice) - 1
            if 0 <= model_index < len(models):
                return models[model_index]
            else:
                console.print("Invalid selection. Please try again.", style="bold red")
                return None
        except ValueError:
            console.print("Invalid input. Please enter a number.", style="bold red")
            return None

    async def switch_provider(self):
        """Switch to a different provider using the unified session manager."""
        # Get available providers dynamically from factory
        available_providers = self.chat_app.provider_factory.get_providers()
        
        # Display provider options
        console.print("Select provider:", style="cyan")
        for idx, (name, _) in available_providers.items():
            console.print(f"{idx}. {name}", style="cyan")
        
        mode = await self.chat_app.input_handler.get_single_input("Enter your choice")
        
        try:
            choice = int(mode)
            if choice not in available_providers:
                console.print("Invalid choice.", style="bold red")
                return None
                
            provider_name, provider_class = available_providers[choice]
            
        except ValueError:
            console.print("Invalid choice. Please enter a number.", style="bold red")
            return None

        # Use unified session creation
        return await self.unified_manager.create_session_for_provider(provider_name)

    async def create_session_from_last(self):
        """Create a session from the last used provider using unified logic."""
        return await self.unified_manager.create_session_from_last_provider()

    def apply_saved_parameters(self, session):
        saved_params = self.chat_app.history_manager.load_parameters()
        for param, value in saved_params.items():
            if param in session.default_parameters and session.default_parameters[param] != value:
                session.set_parameter(param, value)
        if self.chat_app.current_session:
            session.chat_history = self.chat_app.current_session.chat_history
            session.system_message = self.chat_app.current_session.system_message
        else:
            session.chat_history = []
            session.system_message = None