"""LM Studio chat provider implementation."""

import json
from typing import Any, List, Optional
import aiohttp
from rich.console import Console

from .base import ChatProvider
from core.http_handler import HttpHandlerFactory, RequestConfig

console = Console()


class LMStudioChatSession(ChatProvider):
    def __init__(self, base_url: str, model: str, history_manager):
        super().__init__(history_manager, "lmstudio")
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.model = model
        
        self.default_parameters.update({
            "max_tokens": 8192,
            "temperature": 0.8,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None,
        })
    
    @classmethod
    async def get_available_models(cls, base_url: str) -> List[str]:
        """Fetch available models from LM Studio's /v1/models endpoint."""
        base_url = base_url.rstrip('/')
        url = f"{base_url}/v1/models"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        if 'data' in models_data:
                            return [model['id'] for model in models_data['data']]
                        else:
                            console.print("Unexpected API response structure from LM Studio", style="bold red")
                            return []
                    else:
                        console.print(f"Error fetching LM Studio models: {response.status} - {await response.text()}", style="bold red")
                        return []
        except Exception as e:
            console.print(f"Error connecting to LM Studio: {e}", style="bold red")
            return []
    
    def set_parameter(self, param: str, value: Any):
        # Remove the custom set_parameter method to use the unified one from base class
        super().set_parameter(param, value)

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        # Prepare request data
        json_data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.parameters.get("max_tokens", 8192),
            "temperature": self.parameters.get("temperature", 0.8),
            "top_p": self.parameters.get("top_p", 0.95),
            "frequency_penalty": self.parameters.get("frequency_penalty", 0.0),
            "presence_penalty": self.parameters.get("presence_penalty", 0.0),
        }

        # Add stop parameter if it's set
        if self.parameters.get("stop"):
            json_data["stop"] = self.parameters["stop"]

        # Create request configuration
        config = RequestConfig(
            url=f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json_data=json_data,
            stream=True
        )
        
        # Use the unified streaming request method
        async for chunk in self._send_streaming_request(config):
            yield chunk