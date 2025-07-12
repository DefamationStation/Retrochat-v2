"""LM Studio chat provider implementation."""

import json
from typing import Any
import aiohttp
from rich.console import Console

from .base import ChatProvider

console = Console()


class LMStudioChatSession(ChatProvider):
    def __init__(self, base_url: str, model: str, history_manager):
        super().__init__(history_manager)
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
    
    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters or param in ["repeat_penalty", "frequency_penalty"]:
            if param in ["max_tokens"]:
                value = int(value)
            elif param in ["top_p", "temperature", "frequency_penalty", "presence_penalty", "repeat_penalty"]:
                value = float(value)
            elif param == "stop":
                value = value.split() if isinstance(value, str) else value
            elif param == "verbose":
                value = str(value).lower() == "true"
            
            if param in ["repeat_penalty", "frequency_penalty"]:
                self.parameters["frequency_penalty"] = value
            else:
                self.parameters[param] = value
            
            self.history_manager.save_parameters(self.parameters)
            
            if param != "verbose" or value:
                console.print(f"Parameter '{param}' set to {value}", style="cyan")
        else:
            console.print(f"Invalid parameter: {param}", style="bold red")

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        data = {
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
            data["stop"] = self.parameters["stop"]

        headers = {
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/v1/chat/completions"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    complete_message = ""
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: "):
                                if line == "data: [DONE]":
                                    break
                                json_str = line[6:]
                                try:
                                    response_json = json.loads(json_str)
                                    content = response_json['choices'][0]['delta'].get('content', '')
                                    if content:
                                        complete_message += content
                                        yield content  # Always yield the content
                                except json.JSONDecodeError:
                                    continue
                    
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