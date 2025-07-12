"""OpenRouter chat provider implementation."""

import json
import aiohttp
from rich.console import Console

from .base import ChatProvider
from models.chat_message import ChatMessage
from utils.env_manager import EnvManager
from config import Config

console = Console()


class OpenRouterChatSession(ChatProvider):
    def __init__(self, api_key: str, model: str, history_manager):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model = model
        self.default_parameters.update({
            "top_p": 1,
            "temperature": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "top_k": 0,
        })

    @classmethod
    def get_available_models(cls):
        default_model = "meta-llama/llama-3.1-8b-instruct:free"
        env_models = EnvManager.get_env_variable(Config.OPENROUTER_MODELS_KEY, default_model)
        if not env_models:
            env_models = default_model
        return str(env_models).split(',')

    @classmethod
    def add_model(cls, model_name: str):
        models = cls.get_available_models()
        if model_name not in models:
            models.append(model_name)
            EnvManager.set_env_variable(Config.OPENROUTER_MODELS_KEY, ','.join(models))
            return True
        return False

    @classmethod
    def remove_model(cls, model_name: str):
        models = cls.get_available_models()
        if model_name in models:
            models.remove(model_name)
            EnvManager.set_env_variable(Config.OPENROUTER_MODELS_KEY, ','.join(models))
            return True
        return False

    async def send_message(self, message: str):
        self.add_to_history("user", message)
        messages = [{"role": msg.role, "content": msg.content} for msg in self.chat_history]
        
        if self.system_message:
            messages.insert(0, {"role": "system", "content": self.system_message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **{k: v for k, v in self.parameters.items() if v is not None}
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "Retrochat-v2",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data) as response:
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
