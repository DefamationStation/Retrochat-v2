"""Anthropic chat provider implementation."""

import aiohttp
from rich.console import Console

from .base import ChatProvider

console = Console()


class AnthropicChatSession(ChatProvider):
    def __init__(self, api_key: str, model_url: str, history_manager, model: str):
        super().__init__(history_manager, "anthropic")
        self.api_key = api_key
        self.model_url = model_url
        self.model = model

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = self.prepare_messages()
        
        if not messages:
            error_message = "Error: No valid messages to send to the API."
            console.print(error_message, style="bold red")
            yield error_message
            return

        data = {
            "model": self.model,
            "max_tokens": self.parameters.get("max_tokens", 8192),
            "temperature": self.parameters.get("temperature", 0.8),
            "messages": messages,
            "stream": False
        }

        if self.system_message:
            data["system"] = self.system_message

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.model_url, json=data, headers=headers) as response:
                if response.status == 200:
                    response_json = await response.json()
                    assistant_message = response_json.get('content', [{}])[0].get('text', '')
                    if assistant_message:
                        formatted_message = self.format_message(assistant_message)
                        self.add_to_history("assistant", formatted_message)
                        if self.parameters.get("verbose", False):
                            tokens = self.calculate_tokens(formatted_message)
                            total_tokens = self.calculate_total_tokens()
                            console.print(f"Response tokens: {tokens}", style="cyan")
                            console.print(f"Total conversation tokens: {total_tokens}", style="cyan")
                        yield formatted_message
                    else:
                        error_message = "No response content received."
                        console.print(error_message, style="bold red")
                        yield error_message
                else:
                    error_message = f"Error: {response.status} - {await response.text()}"
                    console.print(error_message, style="bold red")
                    yield error_message

    def prepare_messages(self):
        messages = []
        last_role = None
        for msg in self.chat_history:
            if msg.role != "system" and msg.content.strip():
                if msg.role == last_role:
                    # If we have consecutive messages with the same role, combine them
                    messages[-1]["content"] += "\n\n" + msg.content.strip()
                else:
                    messages.append({"role": msg.role, "content": msg.content.strip()})
                    last_role = msg.role
        
        # Ensure the last message is from the user
        if messages and messages[-1]["role"] != "user":
            messages.pop()
        
        return messages
