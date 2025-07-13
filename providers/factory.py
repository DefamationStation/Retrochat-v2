"""Chat provider factory for creating provider instances."""

import os
import importlib
import inspect
from typing import Dict, Type, Tuple
from providers.base import ChatProvider


class ChatProviderFactory:
    """Factory for creating chat provider instances."""
    
    @staticmethod
    def _discover_providers() -> Dict[str, Type[ChatProvider]]:
        """Dynamically discover provider classes from the providers directory."""
        providers = {}
        
        # Get the providers directory path
        providers_dir = os.path.dirname(__file__)
        
        # Iterate through all Python files in the providers directory
        for filename in os.listdir(providers_dir):
            if filename.endswith('.py') and filename not in ['__init__.py', 'base.py', 'factory.py']:
                module_name = filename[:-3]  # Remove .py extension
                
                try:
                    # Import the module
                    module = importlib.import_module(f'providers.{module_name}')
                    
                    # Find classes that inherit from ChatProvider
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, ChatProvider) and 
                            obj != ChatProvider and 
                            obj.__module__ == module.__name__):
                            
                            # Extract provider name from class name
                            # e.g., OllamaChatSession -> Ollama
                            if name.endswith('ChatSession'):
                                provider_name = name[:-11]  # Remove 'ChatSession'
                            elif name.endswith('Session'):
                                provider_name = name[:-7]   # Remove 'Session'
                            else:
                                provider_name = name
                            
                            # Special case for LMStudio to display as "LM Studio"
                            if provider_name == 'LMStudio':
                                provider_name = 'LM Studio'
                                
                            providers[provider_name] = obj
                            break
                            
                except ImportError:
                    continue
                    
        return providers
    
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
        providers = ChatProviderFactory._discover_providers()
        
        # Handle special case for LM Studio
        if provider_type == 'LMStudio':
            provider_type = 'LM Studio'
            
        provider_class = providers.get(provider_type)
        if provider_class:
            return provider_class(*args, **kwargs)
        raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_providers() -> Dict[int, Tuple[str, Type[ChatProvider]]]:
        """Get list of available providers with numbered index."""
        providers = ChatProviderFactory._discover_providers()
        
        # Sort providers for consistent ordering
        sorted_providers = sorted(providers.items())
        
        return {i + 1: (name, cls) for i, (name, cls) in enumerate(sorted_providers)}
