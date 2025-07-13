"""
Base chat provider for Retrochat-v2

This module contains the abstract base class for all chat providers.
"""

from abc import ABC, abstractmethod
from typing import List, Any

from rich.panel import Panel
from rich.markdown import Markdown

from utils.logger import Logger
from utils.console import console
from utils.tokenizer import TokenizerManager
from core.history_manager import ChatHistoryManager
from core.code_formatter import CodeBlockFormatter
from core.parameter_manager import UnifiedParameterManager
from models.chat_message import ChatMessage


class ChatProvider(ABC):
    """Abstract base class for all chat providers."""
    
    def __init__(self, history_manager: ChatHistoryManager):
        self.history_manager = history_manager
        self.chat_history = self.load_history()
        self.system_message = self.history_manager.load_system_message()
        
        # Initialize unified parameter manager
        self.parameter_manager = UnifiedParameterManager(history_manager)
        self.parameter_manager.load_parameters()
        
        self.code_block_formatter = CodeBlockFormatter()
        
        # Legacy compatibility - maintain parameters dict
        self.parameters = self.parameter_manager.get_all_parameters()
        
        # Default parameters for backward compatibility
        self.default_parameters = {
            "temperature": 0.8,
            "max_tokens": 8192,
            "verbose": False,
            "frequency_penalty": 1.1,
            "repeat_penalty": 1.1,
            "use_markdown": True,
        }
        
        self.tokenizer_manager = TokenizerManager()
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Ensure all default parameters are set in the parameter manager."""
        # Register any provider-specific parameters
        for key, value in self.default_parameters.items():
            if not self.parameter_manager.get_parameter_info(key):
                # This is a custom parameter, ensure it exists
                if key not in self.parameter_manager.parameters:
                    self.parameter_manager.parameters[key] = value
        
        # Sync the legacy parameters dict
        self.parameters = self.parameter_manager.get_all_parameters()
        self.save_parameters()

    def load_history(self) -> List[ChatMessage]:
        """Load chat history from the database."""
        try:
            return self.history_manager.load_history()
        except Exception as e:
            Logger.error(f"Error loading chat history: {e}")
            return []

    @abstractmethod
    async def send_message(self, message: str):
        """
        Async generator that yields response chunks (str) from the provider.
        Always yield at least one value, even in error cases.
        """
        yield NotImplementedError("send_message must be implemented by subclasses.")

    def add_to_history(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append(ChatMessage(role, content))
        self.save_history()
        if self.parameters.get("verbose", False) and role == "user":
            tokens = self.tokenizer_manager.calculate_tokens(content)
            total_tokens = self.calculate_total_tokens()
            console.print(f"Message tokens: {tokens}", style="cyan")
            console.print(f"Total conversation tokens: {total_tokens}", style="cyan")

    def save_history(self):
        """Save chat history to the database."""
        try:
            self.history_manager.save_history(self.chat_history)
        except Exception as e:
            Logger.error(f"Error saving chat history: {e}")

    def display_history(self):
        """Display the chat history with formatting."""
        if not self.chat_history:
            console.print("No previous chat history.", style="cyan")
        else:
            console.print("Chat history loaded from previous session:", style="cyan")
            for entry in self.chat_history:
                if entry.role == "user":
                    console.print(Markdown(entry.content), style="green")
                else:
                    formatted_content = self.format_message(entry.content)
                    formatted_response, code_blocks = self.code_block_formatter.format_code_blocks(formatted_content)
                    for line in formatted_response:
                        if isinstance(line, Panel):
                            console.print(line)
                        elif isinstance(line, str):
                            console.print(Markdown(line), style="yellow")
                        else:
                            console.print(str(line), style="yellow")

    def set_system_message(self, message: str):
        """Set the system message for the chat."""
        self.system_message = message
        self.history_manager.save_system_message(message)

    def format_message(self, message: str) -> str:
        """Format message based on markdown preference."""
        if self.parameters.get("use_markdown", True):
            message = message.strip()
            paragraphs = message.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            formatted_message = '\n\n'.join(paragraphs)
            return formatted_message
        else:
            return message

    def set_parameter(self, param: str, value: Any):
        """Set a chat parameter using the unified parameter manager."""
        success = self.parameter_manager.set_parameter(param, value)
        if success:
            # Sync the legacy parameters dict for backward compatibility
            self.parameters = self.parameter_manager.get_all_parameters()

    def save_parameters(self):
        """Save chat parameters to the database."""
        # Parameters are automatically saved by the parameter manager
        # This method is kept for backward compatibility
        pass

    def show_parameters(self):
        """Display current chat parameters using the unified parameter manager."""
        self.parameter_manager.show_parameters()
        
        # Show system message if set
        if self.system_message:
            console.print(f"system: {self.system_message}", style="green")

    def calculate_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text."""
        return self.tokenizer_manager.calculate_tokens(text)

    def calculate_total_tokens(self) -> int:
        """Calculate the total number of tokens in the conversation."""
        total_tokens = 0
        if self.system_message:
            total_tokens += self.calculate_tokens(self.system_message)
        for msg in self.chat_history:
            total_tokens += self.calculate_tokens(msg.content)
        return total_tokens
