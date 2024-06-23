from utils import print_slow
from colorama import Fore

class CommandHandler:
    def __init__(self, history_manager, chat_app):
        self.history_manager = history_manager
        self.chat_app = chat_app

    def handle_command(self, command: str, session):
        command_parts = command.split(' ', 2)
        if len(command_parts) < 2:
            print_slow("Command requires a value. Usage examples:\n"
                       "/chat rename <new_name>\n"
                       "/chat delete\n"
                       "/chat new <chat_name>\n"
                       "/chat list\n"
                       "/chat open <chat_name>", color=Fore.RED)
            return

        cmd = command_parts[0]
        sub_cmd = command_parts[1]

        if cmd == '/chat':
            if sub_cmd == 'rename':
                if len(command_parts) == 3:
                    new_name = command_parts[2].strip()
                    self.history_manager.rename_history(new_name)
                    self.chat_app.set_chat_name(new_name)
                    print_slow(f"Chat renamed to '{new_name}'", color=Fore.CYAN)
                else:
                    print_slow("Please provide a new name for the chat. Usage: /chat rename <new_name>", color=Fore.RED)

            elif sub_cmd == 'delete':
                self.history_manager.delete_history()
                print_slow("Current chat history deleted.", color=Fore.CYAN)
                self.chat_app.set_chat_name('default')
                session.chat_history = []

            elif sub_cmd == 'new':
                if len(command_parts) == 3:
                    new_name = command_parts[2].strip()
                    self.history_manager.set_chat_name(new_name)
                    self.history_manager.save_history([])  # Start with an empty history
                    self.chat_app.set_chat_name(new_name)
                    print_slow(f"New chat '{new_name}' created.", color=Fore.CYAN)
                else:
                    print_slow("Please provide a name for the new chat. Usage: /chat new <chat_name>", color=Fore.RED)

            elif sub_cmd == 'reset':
                self.history_manager.clear_history()
                session.chat_history = []
                print_slow("Chat history has been reset.", color=Fore.CYAN)

            elif sub_cmd == 'list':
                chats = self.history_manager.list_chats()
                if chats:
                    print_slow("Available chats:", color=Fore.CYAN)
                    for chat in chats:
                        print_slow(chat, color=Fore.GREEN)
                else:
                    print_slow("No available chats.", color=Fore.RED)

            elif sub_cmd == 'open':
                if len(command_parts) == 3:
                    chat_name = command_parts[2].strip()
                    if chat_name in self.history_manager.list_chats():
                        self.history_manager.set_chat_name(chat_name)
                        session.chat_history = self.history_manager.load_history()
                        self.chat_app.set_chat_name(chat_name)
                        print_slow(f"Chat '{chat_name}' opened.", color=Fore.CYAN)
                        session.display_history()
                    else:
                        print_slow(f"Chat '{chat_name}' does not exist.", color=Fore.RED)
                else:
                    print_slow("Please provide the name of the chat to open. Usage: /chat open <chat_name>", color=Fore.RED)

            else:
                print_slow("Unknown command. Available commands are:\n"
                           "/chat rename <new_name>\n"
                           "/chat delete\n"
                           "/chat new <chat_name>\n"
                           "/chat reset\n"
                           "/chat list\n"
                           "/chat open <chat_name>", color=Fore.RED)

        else:
            print_slow("Unknown command.", color=Fore.RED)