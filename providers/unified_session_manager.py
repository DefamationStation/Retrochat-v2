"""Unified session creation pipeline for all providers."""

from typing import Optional
from utils.console import console
from utils.suppress_logging import SuppressLogging
from providers.provider_config import ProviderRegistry
from providers.base import ChatProvider
import contextlib
import io


class UnifiedSessionManager:
    """Unified session manager that eliminates provider-specific if-elif chains."""
    
    def __init__(self, chat_app):
        self.chat_app = chat_app
        self.registry = ProviderRegistry
    
    async def create_session_for_provider(self, provider_name: str) -> Optional[ChatProvider]:
        """Create a session for any provider using unified logic."""
        
        # Get provider configuration
        config = self.registry.get_provider_config(provider_name)
        if not config:
            console.print(f"Provider '{provider_name}' is not registered.", style="bold red")
            return None
        
        # Get provider initializer
        initializer = self.registry.get_provider_initializer(provider_name)
        if not initializer:
            console.print(f"No initializer found for provider '{provider_name}'.", style="bold red")
            return None
        
        try:
            # Step 1: Validate requirements (API keys, connections, etc.)
            if not await initializer.validate_requirements(self.chat_app):
                return None
            
            # Step 2: Get model selection
            selected_model = await initializer.get_model(self.chat_app)
            if not selected_model:
                return None
            
            # Step 3: Create session with proper context management
            session_args = initializer.create_session_args(self.chat_app, selected_model)
            
            # Use context suppression for providers that need it
            if provider_name in ['Google']:
                with SuppressLogging():
                    new_session = self.chat_app.provider_factory.create_provider(
                        config.class_name, *session_args
                    )
            elif provider_name in ['Anthropic', 'OpenAI', 'OpenRouter', 'Ollama', 'Oobabooga', 'LM Studio']:
                with contextlib.redirect_stderr(io.StringIO()):
                    new_session = self.chat_app.provider_factory.create_provider(
                        config.class_name, *session_args
                    )
            else:
                new_session = self.chat_app.provider_factory.create_provider(
                    config.class_name, *session_args
                )
            
            # Step 4: Apply saved parameters and save provider info
            if new_session:
                self.chat_app.session_manager.apply_saved_parameters(new_session)
                self.chat_app.save_last_provider_and_model(provider_name, selected_model)
            
            return new_session
            
        except Exception as e:
            console.print(f"Error creating session for {provider_name}: {str(e)}", style="bold red")
            return None
    
    async def create_session_from_last_provider(self) -> Optional[ChatProvider]:
        """Create a session from the last used provider using unified logic."""
        
        if not self.chat_app.last_provider or not self.chat_app.last_model:
            return None
        
        provider_name = self.chat_app.last_provider
        config = self.registry.get_provider_config(provider_name)
        if not config:
            return None
        
        # For last provider recreation, we skip model selection and use stored model
        initializer = self.registry.get_provider_initializer(provider_name)
        if not initializer:
            return None
        
        try:
            # Validate requirements
            if not await initializer.validate_requirements(self.chat_app):
                return None
            
            # Use stored model instead of asking for selection
            selected_model = self.chat_app.last_model
            
            # Handle special cases for providers that need user input
            if provider_name == 'Oobabooga':
                base_url = await self.chat_app.input_handler.get_single_input("Enter Oobabooga base URL (default: http://127.0.0.1:5000)")
                base_url = base_url or "http://127.0.0.1:5000"
                session_args = (base_url, selected_model, self.chat_app.history_manager)
            elif provider_name == 'LM Studio':
                from utils.env_manager import EnvManager
                from config import Config
                base_url = EnvManager.get_env_variable(Config.LMSTUDIO_BASE_URL_KEY)
                if not base_url:
                    base_url = await self.chat_app.input_handler.get_single_input("Enter LM Studio base URL (default: http://localhost:1234)")
                    base_url = base_url or "http://localhost:1234"
                    EnvManager.set_env_variable(Config.LMSTUDIO_BASE_URL_KEY, base_url)
                    console.print("LM Studio base URL saved in .env file.", style="cyan")
                session_args = (base_url, selected_model, self.chat_app.history_manager)
            else:
                session_args = initializer.create_session_args(self.chat_app, selected_model)
            
            # Create session with context management
            if provider_name in ['Google']:
                with SuppressLogging():
                    new_session = self.chat_app.provider_factory.create_provider(
                        config.class_name, *session_args
                    )
            else:
                with contextlib.redirect_stderr(io.StringIO()):
                    new_session = self.chat_app.provider_factory.create_provider(
                        config.class_name, *session_args
                    )
            
            if new_session:
                self.chat_app.session_manager.apply_saved_parameters(new_session)
            
            return new_session
            
        except Exception as e:
            console.print(f"Error recreating session for {provider_name}: {str(e)}", style="bold red")
            return None
