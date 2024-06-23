import requests
import json  # Add this import to use the json module
from .provider_base import ChatProvider
from utils import print_streaming, print_slow
from colorama import Fore

class OllamaProvider(ChatProvider):
    def __init__(self, history_manager, model_url, model):
        super().__init__(history_manager)
        self.model_url = model_url
        self.model = model

    def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": self.model,
            "messages": self.chat_history,
            "stream": True  # Enable streaming
        }
        response = requests.post(self.model_url, json=data, stream=True)
        if response.status_code == 200:
            complete_message = ""
            try:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        response_json = json.loads(line)  # Use json to parse the response line
                        message_content = response_json.get('message', {}).get('content', '')
                        complete_message += message_content
                        print_streaming(message_content, color=Fore.YELLOW)
                        if response_json.get('done', False):
                            break
                self.add_to_history("assistant", complete_message)
                self.save_history()
                print()  # Print new line after the complete message
            except Exception as e:
                print_slow(f"Error processing response: {str(e)}", color=Fore.RED)
        else:
            print_slow(f"Error: {response.status_code} - {response.text}", color=Fore.RED)
