import pytest
from database import ChatHistoryManager

@pytest.fixture
def history_manager():
    return ChatHistoryManager(':memory:')  # Use in-memory database for testing

def test_save_and_load_history(history_manager):
    history = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}]
    history_manager.save_history(history)
    loaded_history = history_manager.load_history()
    assert loaded_history == history

def test_clear_history(history_manager):
    history = [{'role': 'user', 'content': 'Hello'}]
    history_manager.save_history(history)
    history_manager.clear_history()
    assert history_manager.load_history() == []

def test_rename_history(history_manager):
    history = [{'role': 'user', 'content': 'Hello'}]
    history_manager.save_history(history)
    history_manager.rename_history('new_name')
    assert history_manager.chat_name == 'new_name'

def test_list_chats(history_manager):
    history_manager.set_chat_name('chat1')
    history_manager.save_history([{'role': 'user', 'content': 'Hello'}])
    history_manager.set_chat_name('chat2')
    history_manager.save_history([{'role': 'user', 'content': 'Hi'}])
    assert 'chat1' in history_manager.list_chats()
    assert 'chat2' in history_manager.list_chats()
