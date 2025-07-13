"""Command handler for processing chat commands."""

from rich.console import Console
from providers.base import ChatProvider
from providers.openrouter import OpenRouterChatSession
from core.history_manager import ChatHistoryManager
from core.code_formatter import copy_code_to_clipboard

console = Console()


class CommandHandler:
    """Handles chat commands and their execution."""
    
    def __init__(self, history_manager: ChatHistoryManager, chat_app):
        self.history_manager = history_manager
        self.chat_app = chat_app

    async def handle_show_length(self, session: ChatProvider):
        """Show the total conversation token count."""
        total_tokens = session.calculate_total_tokens()
        console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
    
    async def handle_command(self, command: str, session: ChatProvider):
        """Process and execute chat commands."""
        cmd_parts = command.split(maxsplit=2)
        cmd, *args = cmd_parts + ['', '']
        
        if cmd == '/copy':
            try:
                block_num = int(args[0])
                if 1 <= block_num <= len(self.chat_app.code_blocks):
                    copy_code_to_clipboard(self.chat_app.code_blocks[block_num - 1])
                    console.print(f"Code block {block_num} copied to clipboard.", style="green")
                else:
                    console.print(f"Invalid code block number. Available blocks: 1-{len(self.chat_app.code_blocks)}", style="bold red")
            except ValueError:
                console.print("Invalid block number. Please use a number.", style="bold red")
        elif cmd == '/chat':
            method_name = f"handle_{args[0]}"
            method = getattr(self, method_name, None)
            if method:
                await method(args[1], session)
            else:
                self.display_help()
        elif cmd == '/set':
            if args[0] == 'system':
                self.handle_set_system(args[1], session)
            elif not args[0]:
                session.show_parameters()
            else:
                self.handle_set(args[0], args[1], session)
        elif cmd == '/markdown':
            self.handle_markdown(args[0], session)
        elif cmd == '/edit':
            try:
                await self.chat_app.edit_conversation(session)
            except Exception as e:
                console.print(f"An error occurred while editing the conversation: {str(e)}", style="bold red")
                console.print("Your original conversation has not been modified.", style="yellow")
        elif cmd == '/openrouter':
            await self.handle_openrouter_command(command)
        elif cmd == '/show' and args[0] == 'length':
            await self.handle_show_length(session)
        elif cmd == '/show' and args[0] == 'context':
            await self.chat_app.handle_show_context()
        elif cmd == '/show' and args[0] == 'thinking':
            self.handle_show_thinking(args[1])
        elif cmd == '/switch':
            return await self.handle_switch(args[0], session)
        elif cmd == '/help':
            self.display_help()
        elif cmd == '/thoughts':
            self.chat_app.display_manager.toggle_thoughts_display()
        else:
            console.print("Unknown command. Type /help for available commands.", style="bold red")

    async def handle_openrouter_command(self, command: str):
        """Handle OpenRouter specific commands."""
        cmd_parts = command.split(maxsplit=2)
        action, model_name = cmd_parts[1], cmd_parts[2] if len(cmd_parts) > 2 else None

        if action == "add" and model_name:
            if OpenRouterChatSession.add_model(model_name):
                console.print(f"Model '{model_name}' added to OpenRouter models.", style="green")
            else:
                console.print(f"Model '{model_name}' already exists in OpenRouter models.", style="yellow")
        elif action == "rm" and model_name:
            if OpenRouterChatSession.remove_model(model_name):
                console.print(f"Model '{model_name}' removed from OpenRouter models.", style="green")
            else:
                console.print(f"Model '{model_name}' not found in OpenRouter models.", style="yellow")
        else:
            console.print("Invalid OpenRouter command. Use '/openrouter add <model_name>' or '/openrouter rm <model_name>'.", style="bold red")

    def handle_show_thinking(self, value: str):
        """Handle the /show thinking command."""
        if value.lower() in ['true', 'false']:
            show_thoughts = value.lower() == 'true'
            self.chat_app.display_manager.show_thoughts = show_thoughts
            status = "enabled" if show_thoughts else "disabled"
            console.print(f"Model thoughts display {status}.", style="cyan")
        else:
            console.print("Invalid value. Use '/show thinking true' or '/show thinking false'.", style="bold red")

    def handle_set(self, param: str, value: str, session: ChatProvider):
        """Handle parameter setting commands."""
        if not param:
            session.show_parameters()
        else:
            session.set_parameter(param, value)

    def handle_set_system(self, message: str, session: ChatProvider):
        """Handle system message setting."""
        session.set_system_message(message)
        console.print(f"System message set to: {message}", style="cyan")

    async def handle_rename(self, new_name: str, session: ChatProvider):
        """Handle chat renaming."""
        if new_name:
            self.history_manager.rename_history(new_name)
            console.print(f"Chat renamed to '{new_name}'", style="cyan")
            self.chat_app.save_last_chat_name(new_name)
        else:
            console.print("Please provide a new name for the chat. Usage: /chat rename <new_name>", style="bold red")

    async def handle_delete(self, _, session: ChatProvider):
        """Handle chat deletion."""
        self.history_manager.delete_history()
        console.print("Current chat history deleted.", style="cyan")

    async def handle_new(self, new_name: str, session: ChatProvider):
        """Handle new chat creation."""
        if new_name:
            self.history_manager.set_chat_name(new_name)
            self.history_manager.save_history([])
            console.print(f"New chat '{new_name}' created.", style="cyan")
            self.chat_app.save_last_chat_name(new_name)
        else:
            console.print("Please provide a name for the new chat. Usage: /chat new <chat_name>", style="bold red")

    async def handle_reset(self, _, session: ChatProvider):
        """Handle chat history reset."""
        self.history_manager.clear_history()
        session.chat_history = []
        console.print("Chat history has been reset.", style="cyan")

    async def handle_list(self, _, session: ChatProvider):
        """Handle chat list display."""
        chats = self.history_manager.list_chats()
        if chats:
            console.print("Available chats:", style="cyan")
            for chat in chats:
                console.print(chat, style="green")
        else:
            console.print("No available chats.", style="bold red")

    async def handle_open(self, chat_name: str, session: ChatProvider):
        """Handle opening a specific chat."""
        if chat_name:
            if chat_name in self.history_manager.list_chats():
                self.history_manager.set_chat_name(chat_name)
                session.chat_history = self.history_manager.load_history()
                session.system_message = self.history_manager.load_system_message()
                session.parameters = self.history_manager.load_parameters()
                console.print(f"Chat '{chat_name}' opened.", style="cyan")
                session.display_history()
                self.chat_app.save_last_chat_name(chat_name)
            else:
                console.print(f"Chat '{chat_name}' does not exist.", style="bold red")
        else:
            console.print("Please provide the name of the chat to open. Usage: /chat open <chat_name>", style="bold red")

    async def handle_switch(self, _, session: ChatProvider):
        """Handle provider switching."""
        new_session = await self.chat_app.switch_provider()
        if new_session:
            return new_session
        return session

    def handle_markdown(self, value: str, session: ChatProvider):
        """Handle markdown formatting toggle."""
        if value.lower() in ['true', 'false']:
            use_markdown = value.lower() == 'true'
            session.set_parameter('use_markdown', use_markdown)
            status = "enabled" if use_markdown else "disabled"
            console.print(f"Markdown formatting {status}.", style="cyan")
        else:
            console.print("Invalid value. Use '/markdown true' or '/markdown false'.", style="bold red")

    def display_help(self):
        """Display help information for available commands."""
        console.print("Available commands:", style="cyan")
        console.print("/openrouter add|rm <model_name> - Add or remove an OpenRouter model", style="green")
        console.print("/copy <code block number> for example '/copy 0' to copy an entire code block.")
        console.print("/markdown true|false - Enable or disable markdown formatting", style="green")
        console.print("/load <folder name> - Load a folder of documents into RAG", style="green")
        console.print("@<folder name> <Your query> - Question your local documents")
        console.print("/set system <message> - Set the system message", style="green")
        console.print("/set - Show available parameters and their current values", style="green")
        console.print("/set <parameter> <value> - Set a parameter", style="green")
        console.print("/edit - Edit the entire conversation", style="green")
        console.print("/show length - Display the total conversation tokens", style="green")
        console.print("/show context - Display the context of the last query", style="green")
        console.print("/switch - Switch to a different provider or model", style="green")
        console.print("/chat rename <new_name> - Rename the current chat", style="green")
        console.print("/chat delete - Delete the current chat", style="green")
        console.print("/chat new <chat_name> - Create a new chat", style="green")
        console.print("/chat reset - Reset the current chat history", style="green")
        console.print("/chat list - List all available chats", style="green")
        console.print("/chat open <chat_name> - Open a specific chat", style="green")
        console.print("/help - Display this help message", style="green")
        console.print("/exit - Exit the program", style="green")
