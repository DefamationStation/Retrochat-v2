from openai import OpenAI
from .provider_base import ChatProvider
from utils import print_streaming, print_slow
from colorama import Fore

class OpenaiGpt4oProvider(ChatProvider):
    def __init__(self, history_manager, api_key, model="gpt-4o"):
        super().__init__(history_manager)
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)  # Initialize the OpenAI client

    def send_message(self, message: str):
        self.add_to_history("user", message)
        data = {
            "model": self.model,
            "messages": self.chat_history,
            "stream": True
        }
        try:
            # Stream the response
            stream = self.client.chat.completions.create(**data)
            complete_message = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content  # Directly access content
                if content is not None:  # Ensure content is not None before concatenation
                    complete_message += content
                    print_streaming(content, color=Fore.YELLOW)
            self.add_to_history("assistant", complete_message)
            self.save_history()
            print()  # Print new line after the complete message
        except Exception as e:
            print_slow(f"Error processing response: {str(e)}", color=Fore.RED)
