"""OpenAI chat provider implementation."""

import json
import aiohttp
from rich.console import Console

from .base import ChatProvider
from core.http_handler import HttpHandlerFactory, RequestConfig

console = Console()


class OpenAIChatSession(ChatProvider):
    def __init__(self, api_key: str, base_url: str, model: str, history_manager):
        super().__init__(history_manager, "openai")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        self.default_parameters.update({
            "frequency_penalty": 1.1,
        })

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        # Create request configuration
        config = RequestConfig(
            url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json_data={
                "model": self.model,
                "messages": messages,
                "temperature": self.parameters.get("temperature", 0.8),
                "max_tokens": self.parameters.get("max_tokens", 8192),
                "frequency_penalty": self.parameters.get("frequency_penalty", 0.0),
                "stream": True
            },
            stream=True
        )
        
        # Use the unified streaming request method
        async for chunk in self._send_streaming_request(config):
            yield chunk
