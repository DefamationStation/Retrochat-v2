"""Google (Gemini) chat provider implementation."""

import asyncio
import warnings
import logging
import os
from typing import Any
from rich.console import Console

from .base import ChatProvider
from utils.suppress_logging import SuppressLogging

console = Console()


class GoogleChatSession(ChatProvider):
    def __init__(self, api_key: str, model: str, history_manager):
        super().__init__(history_manager, "google")
        self.api_key = api_key
        self.model = model
        
        # Suppress warnings and logging
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("google.generativeai").setLevel(logging.CRITICAL)
        os.environ["GRPC_PYTHON_LOG_LEVEL"] = "error"
        os.environ["GRPC_VERBOSITY"] = "ERROR"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        
        # Configure the Google AI library
        import google.generativeai as genai
        from google.generativeai.generative_models import GenerativeModel
        from google.generativeai.types import GenerationConfig
        os.environ["GOOGLE_API_KEY"] = self.api_key
        with SuppressLogging():
            self.genai_model = GenerativeModel(self.model)
            self.chat = self.initialize_chat()

        self.default_parameters.update({
            "candidate_count": 1,
            "max_tokens": 8192,
            "temperature": 0.8,
        })

    def initialize_chat(self):
        history = []
        if self.system_message:
            history.append({"role": "user", "parts": ["System: " + self.system_message]})
        for msg in self.chat_history:
            role = "user" if msg.role in ["user", "system"] else "model"
            history.append({"role": role, "parts": [msg.content]})
        return self.genai_model.start_chat(history=history)

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        
        from google.generativeai.types import GenerationConfig
        generation_config = GenerationConfig(
            candidate_count=self.parameters.get("candidate_count", 1),
            max_output_tokens=self.parameters.get("max_tokens", 8192),
            temperature=self.parameters.get("temperature", 0.8),
        )

        try:
            with SuppressLogging():
                response = await asyncio.to_thread(
                    self.chat.send_message,
                    message,
                    generation_config=generation_config,
                    stream=False  # Google API doesn't support streaming
                )

            complete_message = response.text
            formatted_message = self.format_message(complete_message)
            self.add_to_history("assistant", formatted_message)
            
            # Simulate streaming by yielding chunks
            chunk_size = 4  # Adjust this value to control the streaming speed
            for i in range(0, len(complete_message), chunk_size):
                yield complete_message[i:i+chunk_size]
                await asyncio.sleep(0.01)  # Add a small delay between chunks
            
            if self.parameters.get("verbose", False):
                tokens = self.calculate_tokens(formatted_message)
                total_tokens = self.calculate_total_tokens()
                console.print(f"\nResponse tokens: {tokens}", style="cyan")
                console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
            
            yield None  # Signal end of streaming
        except Exception as e:
            error_message = f"Error in Google API: {str(e)}"
            console.print(error_message, style="bold red")
            yield error_message

    def set_system_message(self, message: str):
        super().set_system_message(message)
        self.chat = self.initialize_chat()

    def add_to_history(self, role: str, content: str):
        super().add_to_history(role, content)
        self.chat = self.initialize_chat()

    
