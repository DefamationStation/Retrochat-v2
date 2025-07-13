import os
import asyncio
import contextlib
import io
from typing import Optional
from rich.panel import Panel
from rich.markdown import Markdown

# Import configuration and utilities
from config import Config
from utils.console import console
from utils.env_manager import EnvManager
from models.chat_message import ChatMessage
from core.history_manager import ChatHistoryManager
from core.document_manager import DocumentManager
from core.code_formatter import CodeBlockFormatter, copy_code_to_clipboard
from providers.base import ChatProvider
from providers.factory import ChatProviderFactory
from handlers.command_handler import CommandHandler
from app.session_manager import SessionManager
from app.update_manager import UpdateManager
from setup.setup_manager import SetupManager
from ui.input_handler import InputHandler
from ui.display_manager import DisplayManager


class ChatApp:
    async def switch_provider(self):
        """Delegate provider switching to the session manager."""
        return await self.session_manager.switch_provider()
    def __init__(self):
        self.chat_name = 'default'
        self.history_manager = ChatHistoryManager(Config.DB_FILE)
        self.command_handler = CommandHandler(self.history_manager, self)
        self.provider_factory = ChatProviderFactory()
        self.current_session = None
        self.last_commit_hash = None
        self.updated = False
        self.last_provider = None
        self.last_model = None
        self.document_manager = DocumentManager()
        self.code_blocks = []
        self.code_block_formatter = CodeBlockFormatter()
        
        # Initialize managers
        self.session_manager = SessionManager(self)
        self.update_manager = UpdateManager()
        self.setup_manager = SetupManager()
        self.input_handler = InputHandler(self)
        self.display_manager = DisplayManager(self)

        self.check_and_fix_env_file()
        self.load_env_variables()
        self.load_last_chat()

    def check_and_fix_env_file(self):
        required_keys = [
            Config.ANTHROPIC_API_KEY_NAME,
            Config.OPENAI_API_KEY_NAME,
            Config.GOOGLE_API_KEY_NAME,
            Config.OPENROUTER_API_KEY_NAME,
            Config.LAST_CHAT_NAME_KEY,
            Config.OLLAMA_IP_KEY,
            Config.OLLAMA_PORT_KEY,
            Config.LAST_PROVIDER_KEY,
            Config.LAST_MODEL_KEY
        ]
        
        if not os.path.exists(Config.ENV_FILE):
            # Only run setup if .env file doesn't exist
            # The setup manager will handle creating the launcher scripts
            return

        with open(Config.ENV_FILE, 'r') as f:
            env_contents = f.read()

        missing_keys = [key for key in required_keys if key not in env_contents]

        if missing_keys:
            console.print("Updating .env file with missing keys...", style="cyan")
            with open(Config.ENV_FILE, 'a') as f:
                for key in missing_keys:
                    f.write(f"{key}=\n")
            console.print(".env file updated.", style="green")

    def process_chat_history_for_code_blocks(self):
        self.code_blocks = []
        self.code_block_formatter.reset()
        if self.current_session and self.current_session.chat_history:
            for message in self.current_session.chat_history:
                if message.role == "assistant":
                    _, new_code_blocks = self.code_block_formatter.format_code_blocks(message.content)
                    self.code_blocks.extend(new_code_blocks)

    def load_env_variables(self):
        EnvManager.load_env_variables()
        self.chat_name = EnvManager.get_env_variable(Config.LAST_CHAT_NAME_KEY, 'default')
        self.last_commit_hash = EnvManager.get_env_variable("LAST_COMMIT_HASH")
        self.updated = str(EnvManager.get_env_variable("UPDATED", "false")).lower() == "true"
        self.last_provider = EnvManager.get_env_variable(Config.LAST_PROVIDER_KEY)
        self.last_model = EnvManager.get_env_variable(Config.LAST_MODEL_KEY)
        if self.chat_name is not None:
            self.history_manager.set_chat_name(str(self.chat_name))
        else:
            self.history_manager.set_chat_name("default")

    def save_last_provider_and_model(self, provider: str, model: str):
        EnvManager.set_env_variable(Config.LAST_PROVIDER_KEY, provider)
        EnvManager.set_env_variable(Config.LAST_MODEL_KEY, model)
        self.last_provider = provider
        self.last_model = model

    def load_last_chat(self):
        self.history_manager.set_chat_name(self.chat_name if self.chat_name is not None else "default")
        chat_history = self.history_manager.load_history()
        system_message = self.history_manager.load_system_message()
        parameters = self.history_manager.load_parameters()
        if self.current_session:
            self.process_chat_history_for_code_blocks()
        return chat_history, system_message, parameters
    
    def save_last_chat_name(self, chat_name: str):
        EnvManager.set_env_variable(Config.LAST_CHAT_NAME_KEY, chat_name)

    def save_code_blocks(self):
        self.history_manager.save_code_blocks(self.code_blocks)

    async def start(self):
        try:
            console.print("Welcome to Retrochat! [bold green]v1.1.2[/bold green]", style="bold green")
            
            self.setup_manager.check_and_setup()
            
            # Perform update check after displaying welcome message
            if await self.update_manager.check_for_updates():
                return

            self.update_manager.display_update_message()

            self.current_session = await self.session_manager.create_session_from_last()
        
            if not self.current_session:
                self.current_session = await self.session_manager.switch_provider()
            
            if not self.current_session:
                return

            chat_history, system_message, parameters = self.load_last_chat()
            self.current_session.chat_history = chat_history
            self.current_session.system_message = system_message
            self.session_manager.apply_saved_parameters(self.current_session)

            # Process chat history for code blocks after setting up the session
            self.process_chat_history_for_code_blocks()

            self.code_block_formatter.reset()

            if not chat_history:
                console.print("No previous chat history.", style="cyan")
            else:
                self.display_manager.display_chat_history()

            provider_name = type(self.current_session).__name__.replace('ChatSession', '')
            model_name = getattr(self.current_session, 'model', 'Unknown')
            console.print(f"Current provider: [blue]{provider_name}[/blue]", style="cyan")
            console.print(f"Current model: [blue]{model_name}[/blue]", style="cyan")

            self.code_blocks = []
            self.code_block_formatter.reset()

            await self._main_chat_loop()

        except Exception as e:
            console.print(f"An unexpected error occurred: {str(e)}", style="bold red")
        finally:
            if self.current_session:
                self.current_session.save_history()

    async def _main_chat_loop(self):
        while True:
            try:
                user_input = await self.input_handler.get_multiline_input()

                if user_input.lower() == '/exit':
                    console.print("Thank you for chatting. Goodbye!", style="cyan")
                    break
                elif user_input.startswith('/load '):
                    folder_name = user_input.split(' ', 1)[1]
                    await self._handle_load_command(folder_name)
                elif user_input.startswith('@'):
                    parts = user_input[1:].split(' ', 1)
                    if len(parts) == 2:
                        folder_name, query = parts
                        await self._handle_query_command(folder_name, query)
                    else:
                        console.print("Invalid query format. Use @<foldername> <question>", style="bold red")
                elif user_input.startswith('/'):
                    if user_input.startswith('/copy '):
                        try:
                            block_num = int(user_input.split(' ')[1])
                            if 1 <= block_num <= len(self.code_blocks):
                                copy_code_to_clipboard(self.code_blocks[block_num - 1])
                                console.print(f"Code block {block_num} copied to clipboard.", style="green")
                            else:
                                console.print(f"Invalid code block number. Available blocks: 1-{len(self.code_blocks)}", style="bold red")
                        except ValueError:
                            console.print("Invalid block number. Please use a number.", style="bold red")
                    else:
                        if self.current_session is not None:
                            result = await self.command_handler.handle_command(user_input, self.current_session)
                            if isinstance(result, ChatProvider):
                                self.current_session = result
                        else:
                            console.print("No active session.", style="bold red")
                elif user_input:
                    await self._handle_chat_message(user_input)

            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                console.print(f"An error occurred: {str(e)}", style="bold red")
                console.print("The application will continue running. You can try another input or exit.", style="yellow")

    async def _handle_chat_message(self, user_input: str):
        if self.current_session is None:
            console.print("No active session.", style="bold red")
            return
            
        use_markdown = self.current_session.parameters.get("use_markdown", True)
        try:
            response_chunks = []
            async for chunk in self.current_session.send_message(user_input):
                if chunk is not None:
                    response_chunks.append(chunk)
            complete_response = "".join(response_chunks)
        except Exception as e:
            console.print(f"An error occurred while processing the response: {str(e)}", style="bold red")
            return
        
        if use_markdown:
            # Format and display the complete response
            formatted_response, new_code_blocks = self.code_block_formatter.format_code_blocks(complete_response)
            self.code_blocks.extend(new_code_blocks)
            for line in formatted_response:
                if isinstance(line, Panel):
                    console.print(line)
                elif isinstance(line, str):
                    console.print(Markdown(line), style="yellow")
                else:
                    console.print(str(line), style="yellow")
        else:
            console.print("")  # Add an empty print to create a new line after streaming

        self.current_session.save_history()
        self.save_code_blocks()

    async def _handle_load_command(self, folder_name: str):
        console.print(f"RETROCHAT_DIR: {Config.RETROCHAT_DIR}", style="cyan")
        success = self.document_manager.load_documents(folder_name)
        if success:
            console.print(f"Documents from '{folder_name}' loaded successfully.", style="green")
        else:
            console.print(f"Failed to load documents from '{folder_name}'.", style="bold red")

    async def _handle_query_command(self, folder_name: str, query: str):
        results = self.document_manager.query_documents(folder_name, query)
        if not results:
            console.print(f"No results found for query in folder '{folder_name}'", style="yellow")
            return

        context = "\n\n".join([doc.page_content for doc in results])
        prompt = f"""
        Based on the following context, answer the question: {query}

        Context:
        {context}

        Answer:
        """

        self.last_query_context = context
        self.last_query_prompt = prompt

        if self.current_session is not None:
            try:
                response_chunks = []
                async for chunk in self.current_session.send_message(prompt):
                    if chunk is not None:
                        response_chunks.append(chunk)
                complete_response = "".join(response_chunks)
                formatted_response, self.code_blocks = self.code_block_formatter.format_code_blocks(complete_response)
                for line in formatted_response:
                    if isinstance(line, Panel):
                        console.print(line)
                    elif isinstance(line, str):
                        console.print(Markdown(line), style="yellow")
                    else:
                        console.print(str(line), style="yellow")
                if hasattr(self.current_session, "save_history"):
                    self.current_session.save_history()
            except Exception as e:
                console.print(f"An error occurred while processing the response: {str(e)}", style="bold red")
        else:
            console.print("No active session.", style="bold red")
    
    async def handle_show_context(self):
        if hasattr(self, 'last_query_context') and hasattr(self, 'last_query_prompt'):
            console.print("Last query context:", style="cyan")
            console.print(self.last_query_context, style="yellow")
            console.print("\nLast query prompt:", style="cyan")
            console.print(self.last_query_prompt, style="yellow")
        else:
            console.print("No context available. Please run a query first.", style="bold red")