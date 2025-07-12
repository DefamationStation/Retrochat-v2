import os
import sys
import asyncio
import aiohttp
import requests
import sqlite3
import json
import logging
import tempfile
import subprocess
import hashlib
import platform
import google.generativeai as genai
import shutil
import warnings
import contextlib
import io
import pyperclip
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaEmbeddings

# Import configuration and utilities
from config import Config
from utils.logger import Logger
from utils.console import console
from utils.env_manager import EnvManager
from utils.tokenizer import TokenizerManager
from utils.suppress_logging import SuppressLogging
from models.chat_message import ChatMessage
from core.history_manager import ChatHistoryManager
from core.document_manager import DocumentManager, get_embedding_function
from core.code_formatter import CodeBlockFormatter, copy_code_to_clipboard
from providers.base import ChatProvider
from providers.openrouter import OpenRouterChatSession
from providers.anthropic import AnthropicChatSession
from providers.openai import OpenAIChatSession
from providers.google import GoogleChatSession
from providers.ollama import OllamaChatSession
from providers.oobabooga import OobaboogaChatSession
from providers.factory import ChatProviderFactory
from handlers.command_handler import CommandHandler






    
class ChatApp:
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
            self.setup_rchat()
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

    def display_chat_history(self):
        if self.current_session and self.current_session.chat_history:
            for message in self.current_session.chat_history:
                if message.role == "user":
                    console.print(Markdown(message.content), style="green")
                else:
                    formatted_content, _ = self.code_block_formatter.format_code_blocks(message.content)
                    for line in formatted_content:
                        if isinstance(line, Panel):
                            console.print(line)
                        elif isinstance(line, str):
                            console.print(Markdown(line), style="yellow")
                        else:
                            console.print(str(line), style="yellow")

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

        # Debug print
        #console.print("Loaded environment variables:", style="cyan")
        #console.print(f"Anthropic API key: {'*****' if EnvManager.get_env_variable(Config.ANTHROPIC_API_KEY_NAME) else 'Not set'}", style="yellow")
        #console.print(f"OpenAI API key: {'*****' if EnvManager.get_env_variable(Config.OPENAI_API_KEY_NAME) else 'Not set'}", style="yellow")
        #console.print(f"Google API key: {'*****' if EnvManager.get_env_variable(Config.GOOGLE_API_KEY_NAME) else 'Not set'}", style="yellow")
        #console.print(f"Chat name: {self.chat_name}", style="yellow")
        #console.print(f"Ollama IP: {EnvManager.get_env_variable(Config.OLLAMA_IP_KEY)}", style="yellow")
        #console.print(f"Ollama Port: {EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY)}", style="yellow")
        #console.print(f"Last provider: {self.last_provider}", style="yellow")
        #console.print(f"Last model: {self.last_model}", style="yellow")

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
    
    def display_update_message(self):
        if self.updated:
            missed_commits = get_missed_commits("DefamationStation", "Retrochat-v2", "retrochat.py", self.last_commit_hash)
            for i, commit_message in enumerate(missed_commits, 1):
                console.print(f"{i}. {commit_message}", style="yellow")
            EnvManager.set_env_variable("UPDATED", "false")
            self.updated = False

    def ensure_api_key(self, key_name: str, env_var: str):
        api_key = EnvManager.get_env_variable(env_var)
        if not api_key:
            console.print(f"{env_var} is not set. Please enter your API key.", style="cyan")
            api_key = console.ask(f"Enter your {env_var}")
            if api_key:
                EnvManager.set_env_variable(env_var, api_key)
                console.print(f"{env_var} has been set and saved in the .env file.", style="cyan")
                return True
            else:
                console.print(f"No API key provided. {key_name.replace('_', ' ').title()} mode cannot be used.", style="bold red")
                return False
        return True
    
    async def select_openrouter_model(self) -> Optional[str]:
        models = OpenRouterChatSession.get_available_models()
        console.print("Available OpenRouter models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = console.ask("Select a model number")
        return models[int(choice) - 1]

    def ensure_ollama_connection(self):
        ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
        ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
        url = f"http://{ollama_ip}:{ollama_port}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            console.print(f"Unable to connect to Ollama at {ollama_ip}:{ollama_port}", style="bold red")
            new_ip = console.ask("Enter Ollama IP (press Enter for localhost)")
            new_port = console.ask("Enter Ollama port (press Enter for 11434)")
            
            ollama_ip = new_ip or 'localhost'
            ollama_port = new_port or '11434'
            
            EnvManager.set_env_variable(Config.OLLAMA_IP_KEY, ollama_ip)
            EnvManager.set_env_variable(Config.OLLAMA_PORT_KEY, ollama_port)
            
            console.print(f"Ollama connection details updated and saved in the .env file.", style="cyan")
            return self.ensure_ollama_connection()

    async def select_ollama_model(self) -> Optional[str]:
        ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
        ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
        url = f"http://{ollama_ip}:{ollama_port}/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    models_info = await response.json()
                    if isinstance(models_info, dict) and 'models' in models_info:
                        model_names = [model['name'] for model in models_info['models']]
                        console.print("Available Ollama models:", style="cyan")
                        for idx, model in enumerate(model_names):
                            console.print(f"{idx + 1}. {model}", style="green")
                        choice = console.ask("Select a model")
                        return model_names[int(choice) - 1]
                    else:
                        console.print("Unexpected API response structure.", style="bold red")
                else:
                    console.print(f"Error fetching Ollama models: {response.status} - {await response.text()}", style="bold red")
        return None

    async def select_anthropic_model(self) -> Optional[str]:
        models = ["claude-3-5-sonnet-20241022"]
        console.print("Available Anthropic models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = console.ask("Select a model number")
        try:
            return models[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None

    async def select_openai_model(self) -> Optional[str]:
        models = ["gpt-4o-mini", "chatgpt-4o-latest", "gpt-4o", "o1-preview", "o1-mini"]
        console.print("Available OpenAI models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = console.ask("Select a model number")
        return models[int(choice) - 1]
    
    async def select_google_model(self) -> Optional[str]:
        models = ["gemini-2.0-flash-exp", "gemini-1.5-flash-8b"]
        console.print("Available Google Gemini models:", style="cyan")
        for idx, model in enumerate(models):
            console.print(f"{idx + 1}. {model}", style="green")
        choice = console.ask("Select a model number")
        return models[int(choice) - 1]

    def save_last_chat_name(self, chat_name: str):
        EnvManager.set_env_variable(Config.LAST_CHAT_NAME_KEY, chat_name)

    async def edit_conversation(self, session: ChatProvider):
        chat_text = ""
        for msg in session.chat_history:
            chat_text += f"{msg.role.upper()}:\n{msg.content}\n\n"

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(chat_text)
            temp_file_path = temp_file.name

        if platform.system() == 'Windows':
            editor_cmd = ['notepad.exe', temp_file_path]
        elif platform.system() == 'Darwin':  # macOS
            editor = os.environ.get('EDITOR', 'open -t')
            editor_cmd = editor.split() + [temp_file_path]
        else:  # Linux or other Unix-like systems
            editor = os.environ.get('EDITOR', 'nano')
            editor_cmd = [editor, temp_file_path]

        try:
            subprocess.run(editor_cmd, check=True)
        except subprocess.CalledProcessError:
            console.print(f"Error: Unable to open the default editor.", style="bold red")
            console.print("You can manually edit the file at:", style="cyan")
            console.print(temp_file_path, style="yellow")
            console.print("After editing, press Enter to continue.", style="cyan")
            input()

        try:
            with open(temp_file_path, 'r', encoding='utf-8') as file:
                edited_content = file.read()
        except UnicodeDecodeError:
            with open(temp_file_path, 'r') as file:
                edited_content = file.read()

        os.unlink(temp_file_path)

        new_history = []
        current_role = None
        current_content = []

        for line in edited_content.split('\n'):
            line = line.strip()
            if line.upper() in ['USER:', 'ASSISTANT:']:
                if current_role is not None:
                    new_history.append(ChatMessage(role=current_role, content='\n'.join(current_content).strip()))
                current_role = line[:-1].lower()
                current_content = []
            elif line:
                current_content.append(line)

        if current_role is not None:
            new_history.append(ChatMessage(role=current_role, content='\n'.join(current_content).strip()))

        session.chat_history = new_history
        session.save_history()
        self.process_chat_history_for_code_blocks()  # Add this line
        console.print("Chat history updated successfully.", style="cyan")
        session.display_history()

    async def switch_provider(self):
        console.print("Select provider:\n1. Ollama\n2. Anthropic\n3. OpenAI\n4. Google\n5. OpenRouter\n6. Oobabooga", style="cyan")
        mode = console.ask("Enter your choice")

        if mode == '1':
            if not self.ensure_ollama_connection():
                return None
            selected_model = await self.select_ollama_model()
            if not selected_model:
                return None
            ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
            ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
            model_url = f"http://{ollama_ip}:{ollama_port}/api/chat"
            new_session = self.provider_factory.create_provider('Ollama', model_url, selected_model, self.history_manager)
            provider = 'Ollama'
        elif mode == '2':
            if not self.ensure_api_key('anthropic_api_key', Config.ANTHROPIC_API_KEY_NAME):
                return None
            selected_model = await self.select_anthropic_model()
            if not selected_model:
                return None
            new_session = self.provider_factory.create_provider('Anthropic', EnvManager.get_env_variable(Config.ANTHROPIC_API_KEY_NAME),
    "https://api.anthropic.com/v1/messages", self.history_manager, selected_model)
            provider = 'Anthropic'
        elif mode == '3':
            if not self.ensure_api_key('openai_api_key', Config.OPENAI_API_KEY_NAME):
                return None
            selected_model = await self.select_openai_model()
            if not selected_model:
                return None
            new_session = self.provider_factory.create_provider('OpenAI', EnvManager.get_env_variable(Config.OPENAI_API_KEY_NAME),
    "https://api.openai.com/v1/chat/completions", selected_model, self.history_manager)
            provider = 'OpenAI'
        elif mode == '4':
            if not self.ensure_api_key('google_api_key', Config.GOOGLE_API_KEY_NAME):
                return None
            selected_model = await self.select_google_model()
            if not selected_model:
                return None
            with SuppressLogging():
                new_session = self.provider_factory.create_provider('Google', EnvManager.get_env_variable(Config.GOOGLE_API_KEY_NAME), selected_model, self.history_manager)
            provider = 'Google'
        elif mode == '5':
            if not self.ensure_api_key('openrouter_api_key', Config.OPENROUTER_API_KEY_NAME):  # Add Config. here
                return None
            selected_model = await self.select_openrouter_model()
            if not selected_model:
                return None
            new_session = self.provider_factory.create_provider('OpenRouter', EnvManager.get_env_variable(Config.OPENROUTER_API_KEY_NAME), selected_model, self.history_manager)  # Add Config. here
            provider = 'OpenRouter'
        elif mode == '6':
            base_url = console.ask("Enter Oobabooga base URL (default: http://127.0.0.1:5000)")
            base_url = base_url or "http://127.0.0.1:5000"
            character = await self.select_oobabooga_character()
            if not character:
                return None
            new_session = self.provider_factory.create_provider('Oobabooga', base_url, character, self.history_manager)
            provider = 'Oobabooga'
        else:
            console.print("Invalid choice.", style="bold red")
            return None

        if new_session:
            self.apply_saved_parameters(new_session)
            self.save_last_provider_and_model(provider, selected_model)
            return new_session
        return None
    
    async def select_oobabooga_character(self) -> Optional[str]:
        characters = ["Example", "Assistant", "Chatbot", "Custom"]  # Add more characters as needed
        console.print("Available Oobabooga characters:", style="cyan")
        for idx, character in enumerate(characters):
            console.print(f"{idx + 1}. {character}", style="green")
        choice = console.ask("Select a character number")
        try:
            selected = characters[int(choice) - 1]
            if selected == "Custom":
                return console.ask("Enter custom character name")
            return selected
        except (ValueError, IndexError):
            console.print("Invalid selection. Please try again.", style="bold red")
            return None

    async def create_session_from_last(self):
        new_session = None
        
        if self.last_provider == 'Google':
            if not self.ensure_api_key('google_api_key', Config.GOOGLE_API_KEY_NAME):
                return None
            with SuppressLogging():
                new_session = self.provider_factory.create_provider('Google', EnvManager.get_env_variable(Config.GOOGLE_API_KEY_NAME), self.last_model, self.history_manager)
        elif self.last_provider == 'Ollama':
            if not self.ensure_ollama_connection():
                return None
            ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
            ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
            model_url = f"http://{ollama_ip}:{ollama_port}/api/chat"
            with contextlib.redirect_stderr(io.StringIO()):
                new_session = self.provider_factory.create_provider('Ollama', model_url, self.last_model, self.history_manager)
        elif self.last_provider == 'Anthropic':
            if not self.ensure_api_key('anthropic_api_key', Config.ANTHROPIC_API_KEY_NAME):
                return None
            with contextlib.redirect_stderr(io.StringIO()):
                new_session = self.provider_factory.create_provider('Anthropic', EnvManager.get_env_variable(Config.ANTHROPIC_API_KEY_NAME), "https://api.anthropic.com/v1/messages", self.history_manager, self.last_model)
        elif self.last_provider == 'OpenAI':
            if not self.ensure_api_key('openai_api_key', Config.OPENAI_API_KEY_NAME):
                return None
        elif self.last_provider == 'OpenRouter':
            if not self.ensure_api_key('openrouter_api_key', Config.OPENROUTER_API_KEY_NAME):  # Add Config. here
                return None
            with contextlib.redirect_stderr(io.StringIO()):
                new_session = self.provider_factory.create_provider('OpenRouter', EnvManager.get_env_variable(Config.OPENROUTER_API_KEY_NAME), self.last_model, self.history_manager)
        elif self.last_provider == 'Oobabooga':
            base_url = console.ask("Enter Oobabooga base URL (default: http://127.0.0.1:5000)")
            base_url = base_url or "http://127.0.0.1:5000"
            with contextlib.redirect_stderr(io.StringIO()):
                new_session = self.provider_factory.create_provider('Oobabooga', base_url, self.last_model, self.history_manager)
                    
            if new_session:
                self.apply_saved_parameters(new_session)
            return new_session
        
        if new_session:
            self.apply_saved_parameters(new_session)
        return new_session

    def apply_saved_parameters(self, session):
        saved_params = self.history_manager.load_parameters()
        for param, value in saved_params.items():
            if param in session.default_parameters and session.default_parameters[param] != value:
                session.set_parameter(param, value)
        if self.current_session:
            session.chat_history = self.current_session.chat_history
            session.system_message = self.current_session.system_message
        else:
            session.chat_history = []
            session.system_message = None

    async def handle_load_command(self, folder_name: str):
        console.print(f"RETROCHAT_DIR: {Config.RETROCHAT_DIR}", style="cyan")
        success = self.document_manager.load_documents(folder_name)
        if success:
            console.print(f"Documents from '{folder_name}' loaded successfully.", style="green")
        else:
            console.print(f"Failed to load documents from '{folder_name}'.", style="bold red")

    async def handle_query_command(self, folder_name: str, query: str):
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

    async def start(self):
        try:
            console.print("Welcome to Retrochat! [bold green]v1.1.2[/bold green]", style="bold green")
            
            self.check_and_setup()
            
            # Perform update check after displaying welcome message
            if await self.check_for_updates():
                return

            self.display_update_message()

            self.current_session = await self.create_session_from_last()
        
            if not self.current_session:
                self.current_session = await self.switch_provider()
            
            if not self.current_session:
                return

            chat_history, system_message, parameters = self.load_last_chat()
            self.current_session.chat_history = chat_history
            self.current_session.system_message = system_message
            self.apply_saved_parameters(self.current_session)

            # Process chat history for code blocks after setting up the session
            self.process_chat_history_for_code_blocks()

            self.code_block_formatter.reset()

            if not chat_history:
                console.print("No previous chat history.", style="cyan")
            else:
                self.display_chat_history()

            provider_name = type(self.current_session).__name__.replace('ChatSession', '')
            model_name = getattr(self.current_session, 'model', 'Unknown')
            console.print(f"Current provider: [blue]{provider_name}[/blue]", style="cyan")
            console.print(f"Current model: [blue]{model_name}[/blue]", style="cyan")

            self.code_blocks = []
            self.code_block_formatter.reset()

            while True:
                try:
                    user_input = await self.get_multiline_input()

                    if user_input.lower() == '/exit':
                        console.print("Thank you for chatting. Goodbye!", style="cyan")
                        break
                    elif user_input.startswith('/load '):
                        folder_name = user_input.split(' ', 1)[1]
                        await self.handle_load_command(folder_name)
                    elif user_input.startswith('@'):
                        parts = user_input[1:].split(' ', 1)
                        if len(parts) == 2:
                            folder_name, query = parts
                            await self.handle_query_command(folder_name, query)
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
                            result = await self.command_handler.handle_command(user_input, self.current_session)
                            if isinstance(result, ChatProvider):
                                self.current_session = result
                    elif user_input:
                        use_markdown = self.current_session.parameters.get("use_markdown", True)
                        try:
                            response_chunks = []
                            async for chunk in self.current_session.send_message(user_input):
                                if chunk is not None:
                                    response_chunks.append(chunk)
                            complete_response = "".join(response_chunks)
                        except Exception as e:
                            console.print(f"An error occurred while processing the response: {str(e)}", style="bold red")
                            continue
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

                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                except Exception as e:
                    console.print(f"An error occurred: {str(e)}", style="bold red")
                    console.print("The application will continue running. You can try another input or exit.", style="yellow")

        except Exception as e:
            console.print(f"An unexpected error occurred: {str(e)}", style="bold red")
        finally:
            if self.current_session:
                self.current_session.save_history()

    async def get_multiline_input(self) -> str:
        kb = KeyBindings()

        @kb.add('c-j')  # Ctrl+J as a workaround for Ctrl+Enter
        def _(event):
            event.current_buffer.insert_text('\n')

        @kb.add('enter')  # Enter key
        def _(event):
            if event.current_buffer.document.is_cursor_at_the_end:
                event.current_buffer.validate_and_handle()
            else:
                event.current_buffer.insert_text('\n')

        prompt_session = PromptSession(multiline=True, key_bindings=kb)

        try:
            with patch_stdout():
                user_input = await prompt_session.prompt_async(
                    "",  # No prompt message
                    bottom_toolbar=None,  # Remove the code blocks counter
                )
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return ""
        
    # Add a method to save code blocks
    def save_code_blocks(self):
        self.history_manager.save_code_blocks(self.code_blocks)
    
    async def check_for_updates(self):
        repo_owner = "DefamationStation"
        repo_name = "Retrochat-v2"
        file_path = "retrochat.py"

        EnvManager.load_env_variables()
        last_commit_hash = EnvManager.get_env_variable("LAST_COMMIT_HASH", "")

        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}&page=1&per_page=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        latest_commit = (await response.json())[0]
                        latest_commit_hash = latest_commit['sha']

                        if latest_commit_hash == last_commit_hash:
                            console.print("You're running the latest version.", style="green")
                            return False

                        missed_commits = get_missed_commits(repo_owner, repo_name, file_path, last_commit_hash)

                        url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{latest_commit_hash}/{file_path}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                latest_content = await response.text()

                                with open(__file__, 'r') as f:
                                    current_content = f.read()

                                if hashlib.sha256(current_content.encode()).hexdigest() != hashlib.sha256(latest_content.encode()).hexdigest():
                                    console.print("Updates are available:", style="bold yellow")
                                    for i, commit_message in enumerate(missed_commits, 1):
                                        console.print(f"{i}. {commit_message}", style="yellow")

                                    console.print("\nDo you want to update?\n\n1. Yes\n2. No")
                                    choice = console.ask("", choices=["1", "2"])

                                    if choice == "1":
                                        console.print("Updating...", style="cyan")
                                        with open(__file__, 'w') as f:
                                            f.write(latest_content)
                                        EnvManager.set_env_variable("LAST_COMMIT_HASH", latest_commit_hash)
                                        EnvManager.set_env_variable("UPDATED", "true")
                                        console.print("Update complete. Please restart the script.", style="bold green")
                                        return True
                                    else:
                                        console.print("Update skipped. Running current version.", style="yellow")
                                        return False
                                else:
                                    console.print("You're running the latest version.", style="green")
                                    EnvManager.set_env_variable("LAST_COMMIT_HASH", latest_commit_hash)
                                    EnvManager.set_env_variable("UPDATED", "false")
                                    return False
                            else:
                                console.print(f"Failed to fetch the latest version: {response.status}", style="bold red")
                                return False
                    else:
                        console.print(f"Failed to check for updates: {response.status}", style="bold red")
                        return False
        except (aiohttp.ClientError, asyncio.TimeoutError):
            console.print("Unable to connect to GitHub. Skipping update check.", style="yellow")
            return False

    def check_and_setup(self):
        rchat_bat_path = os.path.join(Config.RETROCHAT_DIR, "rchat.bat")
        if not os.path.exists(rchat_bat_path) or not os.path.exists(Config.RETROCHAT_SCRIPT):
            console.print("RetroChat Setup", style="bold cyan")
            console.print("This setup will do the following:", style="cyan")
            console.print("1. Create a '.retrochat' folder in your home directory", style="cyan")
            console.print("2. Copy the RetroChat script to the '.retrochat' folder", style="cyan")
            console.print("3. Create an 'rchat.bat' file in the '.retrochat' folder", style="cyan")
            console.print("4. Add the '.retrochat' folder to your system PATH", style="cyan")
            console.print("\nThis will allow you to run RetroChat from anywhere using the 'rchat' command.", style="cyan")
            
            response = console.ask("Do you want to proceed with the setup?", choices=["yes", "no"])
            if response.lower() == "yes":
                self.setup_rchat()
            else:
                console.print("Setup cancelled. You can run the setup later by using the --setup flag.", style="yellow")

    def setup_rchat(self):
        os.makedirs(Config.RETROCHAT_DIR, exist_ok=True)
        
        current_script = sys.argv[0]
        shutil.copy2(current_script, Config.RETROCHAT_SCRIPT)
        console.print(f"Copied RetroChat script to {Config.RETROCHAT_SCRIPT}", style="cyan")

        # Also copy rchat.ps1 to the .retrochat directory
        rchat_ps1_src = os.path.join(os.path.dirname(current_script), "rchat.ps1")
        rchat_ps1_dst = os.path.join(Config.RETROCHAT_DIR, "rchat.ps1")
        if os.path.exists(rchat_ps1_src):
            shutil.copy2(rchat_ps1_src, rchat_ps1_dst)
            console.print(f"Copied rchat.ps1 to {rchat_ps1_dst}", style="cyan")
        else:
            console.print(f"Warning: rchat.ps1 not found at {rchat_ps1_src}. Batch launcher may not work.", style="yellow")
        
        if sys.platform.startswith('win'):
            rchat_bat_path = os.path.join(Config.RETROCHAT_DIR, "rchat.bat")
            rchat_ps1_path = os.path.join(os.path.dirname(Config.RETROCHAT_SCRIPT), "rchat.ps1")
            # The batch file will call the PowerShell script, passing all arguments
            with open(rchat_bat_path, "w") as f:
                f.write(f"@echo off\n"
                        f"powershell -ExecutionPolicy Bypass -File \"{rchat_ps1_path}\" %*\n")
            console.print(f"Created rchat.bat at {rchat_bat_path}", style="cyan")
        else:  # Mac or Linux
            rchat_sh_path = os.path.join(Config.RETROCHAT_DIR, "rchat")
            with open(rchat_sh_path, "w") as f:
                f.write(f'#!/bin/bash\npython3 "{Config.RETROCHAT_SCRIPT}" "$@"')
            os.chmod(rchat_sh_path, 0o755)  # Make the script executable
            console.print(f"Created rchat shell script at {rchat_sh_path}", style="cyan")
        
        if not os.path.exists(Config.ENV_FILE):
            with open(Config.ENV_FILE, "w") as f:
                f.write(f"{Config.ANTHROPIC_API_KEY_NAME}=\n")
                f.write(f"{Config.OPENAI_API_KEY_NAME}=\n")
                f.write(f"{Config.GOOGLE_API_KEY_NAME}=\n")
                f.write(f"{Config.OPENROUTER_API_KEY_NAME}=\n")
                f.write(f"{Config.LAST_CHAT_NAME_KEY}=default\n")
                f.write(f"{Config.OLLAMA_IP_KEY}=localhost\n")
                f.write(f"{Config.OLLAMA_PORT_KEY}=11434\n")
                f.write(f"{Config.LAST_PROVIDER_KEY}=\n")
                f.write(f"{Config.LAST_MODEL_KEY}=\n")
            console.print(f"Created .env file at {Config.ENV_FILE}", style="cyan")
        
        console.print("Setup complete. You can now use the 'rchat' command from anywhere.", style="green")
        
        if sys.platform.startswith('win'):
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
            try:
                path, _ = winreg.QueryValueEx(key, "Path")
                if Config.RETROCHAT_DIR not in path:
                    new_path = f"{path};{Config.RETROCHAT_DIR}"
                    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
                    console.print(f"Added {Config.RETROCHAT_DIR} to PATH.", style="cyan")
                else:
                    console.print(f"{Config.RETROCHAT_DIR} is already in PATH.", style="cyan")
            except WindowsError:
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, Config.RETROCHAT_DIR)
                console.print(f"Created PATH and added {Config.RETROCHAT_DIR}.", style="cyan")
            finally:
                winreg.CloseKey(key)
        else:  # Mac or Linux
            shell = os.environ.get("SHELL", "").split("/")[-1]
            rc_file = f".{shell}rc" if shell in ['bash', 'zsh'] else ".profile"
            rc_path = os.path.join(Config.USER_HOME, rc_file)
            
            with open(rc_path, "a") as f:
                f.write(f'\nexport PATH="$PATH:{Config.RETROCHAT_DIR}"')
            
            console.print(f"Added {Config.RETROCHAT_DIR} to PATH in {rc_path}", style="cyan")
            console.print(f"Please run 'source ~/{rc_file}' or restart your terminal for the changes to take effect.", style="cyan")

        console.print("Setup complete. You can now use the 'rchat' command from anywhere.", style="green")


def get_missed_commits(repo_owner, repo_name, file_path, last_commit_hash):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?path={file_path}"
    response = requests.get(url)
    if response.status_code != 200:
        console.print(f"Failed to fetch commits: {response.status_code}", style="bold red")
        return []

    commits = response.json()
    missed_commits = []
    for commit in commits:
        if commit['sha'] == last_commit_hash:
            break
        missed_commits.append(commit['commit']['message'])
    
    return missed_commits

async def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        ChatApp().setup_rchat()
    else:
        app = ChatApp()
        await app.start()

if __name__ == "__main__":
    asyncio.run(main())
