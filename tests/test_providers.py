import pytest
from providers.ollama import OllamaProvider
from providers.anthropic import AnthropicProvider
from database import ChatHistoryManager

@pytest.fixture
def history_manager():
    return ChatHistoryManager(':memory:')  # Use in-memory database for testing

def test_ollama_provider(history_manager):
    provider = OllamaProvider(history_manager, "http://test-url.com", "test-model")
    provider.send_message("Test message")
    assert len(provider.chat_history) > 0

def test_anthropic_provider(history_manager):
    provider = AnthropicProvider(history_manager, "test-api-key", "http://test-url.com")
    provider.send_message("Test message")
    assert len(provider.chat_history) > 0
