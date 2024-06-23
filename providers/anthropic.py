import requests
from .provider_base import ChatProvider
from utils import print_slow
from colorama import Fore

class AnthropicProvider(ChatProvider):
    def __init__(self, history_manager, api_key, model_url):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model_url = model_url

    def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 4096,
            "temperature": 0.0,
            "system": "Keep your answers short and to the point.",
            "messages": self.chat_history
        }
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        response = requests.post(self.model_url, json=data, headers=headers)
        if response.status_code == 200:
            try:
                response_json = response.json()
                assistant_message = response_json.get('content', [{}])[0].get('text', '')
                if assistant_message:
                    self.add_to_history("assistant", assistant_message)
                    self.save_history()
                    return assistant_message
                else:
                    return "No response content received."
            except Exception as e:
                return f"Error processing response: {str(e)}"
        else:
            return f"Error: {response.status_code} - {response.text}"

