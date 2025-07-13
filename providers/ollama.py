"""Ollama chat provider implementation."""

import json
from typing import Any
import aiohttp
from rich.console import Console

from .base import ChatProvider

console = Console()


class OllamaChatSession(ChatProvider):
    def __init__(self, model_url: str, model: str, history_manager):
        super().__init__(history_manager, "ollama")
        self.model_url = model_url
        self.model = model
        self.default_parameters.update({
            "num_predict": 128,
            "top_k": 40,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "num_ctx": 8192,
            "stop": None,
        })
    
    

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {k: v for k, v in self.parameters.items() if v is not None}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.model_url, json=data) as response:
                if response.status == 200:
                    complete_message = ""
                    async for line in response.content:
                        if line:
                            response_json = json.loads(line)
                            message_content = response_json.get('message', {}).get('content', '')
                            if message_content:
                                complete_message += message_content
                                yield message_content  # Always yield the content
                                if response_json.get('done', False):
                                    break
                    
                    formatted_message = self.format_message(complete_message)
                    self.add_to_history("assistant", formatted_message)
                    
                    if self.parameters.get("verbose", False):
                        tokens = self.calculate_tokens(formatted_message)
                        total_tokens = self.calculate_total_tokens()
                        console.print(f"\nResponse tokens: {tokens}", style="cyan")
                        console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
                    
                    yield None  # Signal end of streaming
                else:
                    error_message = f"Error: {response.status} - {await response.text()}"
                    console.print(error_message, style="bold red")
                    yield error_message
