from abc import ABC, abstractmethod
from colorama import Fore
from utils import print_slow

class ChatProvider(ABC):
    def __init__(self, history_manager):
        self.history_manager = history_manager
        self.chat_history = history_manager.load_history()

    @abstractmethod
    def send_message(self, message: str):
        pass

    def add_to_history(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def save_history(self):
        self.history_manager.save_history(self.chat_history)

    def display_history(self):
        if not self.chat_history:
            print_slow("No previous chat history.", color=Fore.CYAN)
        else:
            print_slow("Chat history loaded from previous session:", color=Fore.CYAN)
            for entry in self.chat_history:
                role_color = Fore.GREEN if entry['role'] == 'user' else Fore.YELLOW
                print_slow(f"{entry['content']}", color=role_color)
