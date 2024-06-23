import pytest
from commands.chat_commands import CommandHandler
from database import ChatHistoryManager
from providers.ollama import OllamaProvider

@pytest.fixture
def history_manager():
    return ChatHistoryManager(':memory:')  # Use in-memory database for testing

@pytest.fixture
def command_handler(history_manager):
    return CommandHandler(history_manager)

def test_chat_commands(command_handler, history_manager):
    provider = OllamaProvider(history_manager, "http://test-url.com", "test-model")
    command_handler.handle_command('/chat new test_chat', provider)
    assert history_manager.chat_name == 'test_chat'
    command_handler.handle_command('/chat delete', provider)
    assert history_manager.load_history() == []
