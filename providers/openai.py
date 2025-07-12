"""OpenAI chat provider implementation."""

import json
import aiohttp
from rich.console import Console

from .base import ChatProvider

console = Console()


class OpenAIChatSession(ChatProvider):
    def __init__(self, api_key: str, base_url: str, model: str, history_manager):
        super().__init__(history_manager)
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
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.parameters.get("temperature", 0.8),
            "max_tokens": self.parameters.get("max_tokens", 8192),
            "frequency_penalty": self.parameters.get("frequency_penalty", 0.0),
            "stream": True
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as response:
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
                                        yield content
                                except json.JSONDecodeError:
                                    continue
                    formatted_message = self.format_message(complete_message)
                    self.add_to_history("assistant", formatted_message)
                    if self.parameters.get("verbose", False):
                        tokens = self.calculate_tokens(formatted_message)
                        total_tokens = self.calculate_total_tokens()
                        console.print(f"Response tokens: {tokens}", style="cyan")
                        console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
                    yield None  # Signal end of streaming
                    yield formatted_message  # Yield the formatted message as the last item
                else:
                    error_message = f"Error: {response.status} - {await response.text()}"
                    console.print(error_message, style="bold red")
                    yield error_message
