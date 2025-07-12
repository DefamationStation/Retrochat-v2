"""Chat provider factory for creating provider instances."""

from providers.base import ChatProvider
from providers.openrouter import OpenRouterChatSession
from providers.anthropic import AnthropicChatSession
from providers.openai import OpenAIChatSession
from providers.google import GoogleChatSession
from providers.ollama import OllamaChatSession
from providers.oobabooga import OobaboogaChatSession
from providers.lmstudio import LMStudioChatSession


class ChatProviderFactory:
    """Factory for creating chat provider instances."""
    
    @staticmethod
    def create_provider(provider_type: str, *args, **kwargs) -> ChatProvider:
        """Create a chat provider instance based on the provider type.
        
        Args:
            provider_type: The type of provider to create
            *args: Positional arguments to pass to the provider constructor
            **kwargs: Keyword arguments to pass to the provider constructor
            
        Returns:
            ChatProvider: The created provider instance
            
        Raises:
            ValueError: If the provider type is not supported
        """
        providers = {
            'Ollama': OllamaChatSession,
            'Anthropic': AnthropicChatSession,
            'OpenAI': OpenAIChatSession,
            'Google': GoogleChatSession,
            'OpenRouter': OpenRouterChatSession,
            'Oobabooga': OobaboogaChatSession,
            'LMStudio': LMStudioChatSession
        }
        provider_class = providers.get(provider_type)
        if provider_class:
            return provider_class(*args, **kwargs)
        raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_providers():
        """Get list of available providers."""
        return {
            1: ("Ollama", OllamaChatSession),
            2: ("Anthropic", AnthropicChatSession), 
            3: ("OpenAI", OpenAIChatSession),
            4: ("Google", GoogleChatSession),
            5: ("OpenRouter", OpenRouterChatSession),
            6: ("Oobabooga", OobaboogaChatSession),
            7: ("LM Studio", LMStudioChatSession)
        }
