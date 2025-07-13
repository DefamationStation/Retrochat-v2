"""Provider configuration registry for centralized provider metadata and initialization."""

from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from utils.env_manager import EnvManager
from config import Config


@dataclass
class ProviderConfig:
    """Configuration for a chat provider."""
    name: str
    class_name: str
    requires_api_key: bool = False
    api_key_env_var: Optional[str] = None
    requires_base_url: bool = False
    default_base_url: Optional[str] = None
    base_url_env_var: Optional[str] = None
    default_models: Optional[list] = None
    supports_model_selection: bool = True
    requires_connection_test: bool = False
    custom_initialization: bool = False
    
    def __post_init__(self):
        if self.default_models is None:
            self.default_models = []


class ProviderInitializer(ABC):
    """Base class for provider-specific initialization logic."""
    
    @abstractmethod
    async def validate_requirements(self, chat_app) -> bool:
        """Validate provider requirements (API keys, connections, etc.)."""
        pass
    
    @abstractmethod
    async def get_model(self, chat_app) -> Optional[str]:
        """Get the selected model for this provider."""
        pass
    
    @abstractmethod
    def create_session_args(self, chat_app, model: str) -> tuple:
        """Create the arguments for provider session creation."""
        pass


class OllamaInitializer(ProviderInitializer):
    """Initializer for Ollama provider."""
    
    async def validate_requirements(self, chat_app) -> bool:
        return await chat_app.session_manager.ensure_ollama_connection()
    
    async def get_model(self, chat_app) -> Optional[str]:
        return await chat_app.session_manager.select_ollama_model()
    
    def create_session_args(self, chat_app, model: str) -> tuple:
        ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
        ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
        model_url = f"http://{ollama_ip}:{ollama_port}/api/chat"
        return (model_url, model, chat_app.history_manager)


class AnthropicInitializer(ProviderInitializer):
    """Initializer for Anthropic provider."""
    
    async def validate_requirements(self, chat_app) -> bool:
        return await chat_app.session_manager.ensure_api_key('anthropic_api_key', Config.ANTHROPIC_API_KEY_NAME)
    
    async def get_model(self, chat_app) -> Optional[str]:
        return await chat_app.session_manager.select_anthropic_model()
    
    def create_session_args(self, chat_app, model: str) -> tuple:
        api_key = EnvManager.get_env_variable(Config.ANTHROPIC_API_KEY_NAME)
        return (api_key, "https://api.anthropic.com/v1/messages", chat_app.history_manager, model)


class OpenAIInitializer(ProviderInitializer):
    """Initializer for OpenAI provider."""
    
    async def validate_requirements(self, chat_app) -> bool:
        return await chat_app.session_manager.ensure_api_key('openai_api_key', Config.OPENAI_API_KEY_NAME)
    
    async def get_model(self, chat_app) -> Optional[str]:
        return await chat_app.session_manager.select_openai_model()
    
    def create_session_args(self, chat_app, model: str) -> tuple:
        api_key = EnvManager.get_env_variable(Config.OPENAI_API_KEY_NAME)
        return (api_key, "https://api.openai.com/v1/chat/completions", model, chat_app.history_manager)


class GoogleInitializer(ProviderInitializer):
    """Initializer for Google provider."""
    
    async def validate_requirements(self, chat_app) -> bool:
        return await chat_app.session_manager.ensure_api_key('google_api_key', Config.GOOGLE_API_KEY_NAME)
    
    async def get_model(self, chat_app) -> Optional[str]:
        return await chat_app.session_manager.select_google_model()
    
    def create_session_args(self, chat_app, model: str) -> tuple:
        api_key = EnvManager.get_env_variable(Config.GOOGLE_API_KEY_NAME)
        return (api_key, model, chat_app.history_manager)


class OpenRouterInitializer(ProviderInitializer):
    """Initializer for OpenRouter provider."""
    
    async def validate_requirements(self, chat_app) -> bool:
        return await chat_app.session_manager.ensure_api_key('openrouter_api_key', Config.OPENROUTER_API_KEY_NAME)
    
    async def get_model(self, chat_app) -> Optional[str]:
        return await chat_app.session_manager.select_openrouter_model()
    
    def create_session_args(self, chat_app, model: str) -> tuple:
        api_key = EnvManager.get_env_variable(Config.OPENROUTER_API_KEY_NAME)
        return (api_key, model, chat_app.history_manager)


class OobaboogaInitializer(ProviderInitializer):
    """Initializer for Oobabooga provider."""
    
    async def validate_requirements(self, chat_app) -> bool:
        # Oobabooga doesn't require API key validation
        return True
    
    async def get_model(self, chat_app) -> Optional[str]:
        base_url = await chat_app.input_handler.get_single_input("Enter Oobabooga base URL (default: http://127.0.0.1:5000)")
        base_url = base_url or "http://127.0.0.1:5000"
        character = await chat_app.session_manager.select_oobabooga_character()
        # Store base_url for create_session_args
        self._base_url = base_url
        return character
    
    def create_session_args(self, chat_app, model: str) -> tuple:
        return (self._base_url, model, chat_app.history_manager)


class LMStudioInitializer(ProviderInitializer):
    """Initializer for LM Studio provider."""
    
    async def validate_requirements(self, chat_app) -> bool:
        # LM Studio doesn't require API key validation
        return True
    
    async def get_model(self, chat_app) -> Optional[str]:
        base_url = EnvManager.get_env_variable(Config.LMSTUDIO_BASE_URL_KEY)
        if not base_url:
            base_url = await chat_app.input_handler.get_single_input("Enter LM Studio base URL (default: http://localhost:1234)")
            base_url = base_url or "http://localhost:1234"
            EnvManager.set_env_variable(Config.LMSTUDIO_BASE_URL_KEY, base_url)
            from utils.console import console
            console.print("LM Studio base URL saved in .env file.", style="cyan")
        
        model = await chat_app.session_manager.select_lmstudio_model(base_url)
        # Store base_url for create_session_args
        self._base_url = base_url
        return model
    
    def create_session_args(self, chat_app, model: str) -> tuple:
        return (self._base_url, model, chat_app.history_manager)


class ProviderRegistry:
    """Registry for provider configurations and initialization logic."""
    
    _providers: Dict[str, ProviderConfig] = {
        'Ollama': ProviderConfig(
            name='Ollama',
            class_name='Ollama',
            requires_connection_test=True,
            supports_model_selection=True
        ),
        'Anthropic': ProviderConfig(
            name='Anthropic',
            class_name='Anthropic',
            requires_api_key=True,
            api_key_env_var=Config.ANTHROPIC_API_KEY_NAME,
            default_models=['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229']
        ),
        'OpenAI': ProviderConfig(
            name='OpenAI',
            class_name='OpenAI',
            requires_api_key=True,
            api_key_env_var=Config.OPENAI_API_KEY_NAME,
            default_models=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']
        ),
        'Google': ProviderConfig(
            name='Google',
            class_name='Google',
            requires_api_key=True,
            api_key_env_var=Config.GOOGLE_API_KEY_NAME,
            default_models=['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
        ),
        'OpenRouter': ProviderConfig(
            name='OpenRouter',
            class_name='OpenRouter',
            requires_api_key=True,
            api_key_env_var=Config.OPENROUTER_API_KEY_NAME,
            supports_model_selection=True
        ),
        'Oobabooga': ProviderConfig(
            name='Oobabooga',
            class_name='Oobabooga',
            requires_base_url=True,
            default_base_url="http://127.0.0.1:5000",
            supports_model_selection=True,
            custom_initialization=True
        ),
        'LM Studio': ProviderConfig(
            name='LM Studio',
            class_name='LM Studio',
            requires_base_url=True,
            default_base_url="http://localhost:1234",
            base_url_env_var=Config.LMSTUDIO_BASE_URL_KEY,
            supports_model_selection=True,
            custom_initialization=True
        )
    }
    
    _initializers: Dict[str, ProviderInitializer] = {
        'Ollama': OllamaInitializer(),
        'Anthropic': AnthropicInitializer(),
        'OpenAI': OpenAIInitializer(),
        'Google': GoogleInitializer(),
        'OpenRouter': OpenRouterInitializer(),
        'Oobabooga': OobaboogaInitializer(),
        'LM Studio': LMStudioInitializer()
    }
    
    @classmethod
    def get_provider_config(cls, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a provider."""
        return cls._providers.get(provider_name)
    
    @classmethod
    def get_provider_initializer(cls, provider_name: str) -> Optional[ProviderInitializer]:
        """Get initializer for a provider."""
        return cls._initializers.get(provider_name)
    
    @classmethod
    def get_all_providers(cls) -> Dict[str, ProviderConfig]:
        """Get all registered providers."""
        return cls._providers.copy()
    
    @classmethod
    def register_provider(cls, config: ProviderConfig, initializer: ProviderInitializer):
        """Register a new provider (for extensions/plugins)."""
        cls._providers[config.name] = config
        cls._initializers[config.name] = initializer
