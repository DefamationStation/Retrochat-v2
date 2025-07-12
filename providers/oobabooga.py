"""Oobabooga chat provider implementation."""

import json
from typing import Any
import aiohttp
from rich.console import Console

from .base import ChatProvider

console = Console()


class OobaboogaChatSession(ChatProvider):
    def __init__(self, base_url: str, character: str, history_manager):
        super().__init__(history_manager)
        self.base_url = base_url
        self.character = character
        self.headers = {"Content-Type": "application/json"}
        self.default_parameters.update({
            "max_new_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
            "typical_p": 1,
            "repetition_penalty": 1.15,
            "encoder_repetition_penalty": 1.0,
            "top_k": 40,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "early_stopping": False,
            "seed": -1,
            "add_bos_token": True,
            "truncation_length": 2048,
            "ban_eos_token": False,
            "skip_special_tokens": True,
            "stopping_strings": []
        })

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        data = {
            "mode": "chat",
            "character": self.character,
            "messages": messages,
            "stream": True,
            **{k: v for k, v in self.parameters.items() if v is not None}
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/v1/chat/completions", headers=self.headers, json=data, ssl=False) as response:
                if response.status == 200:
                    complete_message = ""
                    async for line in response.content:
                        if line:
                            try:
                                response_json = json.loads(line)
                                message_content = response_json['choices'][0]['delta'].get('content', '')
                                if message_content:
                                    complete_message += message_content
                                    yield message_content  # Yield each chunk for streaming
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

    def set_parameter(self, param: str, value: Any):
        if param in self.default_parameters or param == "verbose":
            if param in ["max_new_tokens", "top_k", "min_length", "no_repeat_ngram_size", "num_beams", "seed", "truncation_length"]:
                value = int(value)
            elif param in ["temperature", "top_p", "typical_p", "repetition_penalty", "encoder_repetition_penalty", "penalty_alpha", "length_penalty"]:
                value = float(value)
            elif param in ["add_bos_token", "ban_eos_token", "skip_special_tokens", "early_stopping", "verbose"]:
                value = str(value).lower() == "true"
            elif param == "stopping_strings":
                value = value.split(',') if isinstance(value, str) else value
            
            self.parameters[param] = value
            self.history_manager.save_parameters(self.parameters)
            
            if param != "verbose" or value:
                console.print(f"Parameter '{param}' set to {value}", style="cyan")
        else:
            console.print(f"Invalid parameter: {param}", style="bold red")
