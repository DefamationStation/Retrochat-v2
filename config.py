import os
from dotenv import load_dotenv, set_key

class ConfigManager:
    def __init__(self):
        self.retrochat_dir = os.path.join(os.path.expanduser('~'), '.retrochat')
        os.makedirs(self.retrochat_dir, exist_ok=True)
        self.env_file = os.path.join(self.retrochat_dir, '.env')
        self.db_file = os.path.join(self.retrochat_dir, 'chat_history.db')
        self.load_environment()

    def load_environment(self):
        if not os.path.exists(self.env_file):
            open(self.env_file, 'a').close()
        load_dotenv(self.env_file)

    def get_db_file(self):
        return self.db_file

    def get_api_key(self, key_name):
        return os.getenv(key_name)

    def set_api_key(self, key_name, key_value):
        set_key(self.env_file, key_name, key_value)
        load_dotenv(self.env_file)

    def get_model_url(self, model_type):
        model_urls = {
            'ollama_models': "http://192.168.1.82:11434/api/tags",
            'ollama_chat': "http://192.168.1.82:11434/api/chat",
            'anthropic_chat': "https://api.anthropic.com/v1/messages"
        }
        return model_urls.get(model_type)

    def get_last_chat_name(self):
        return os.getenv('LAST_CHAT_NAME', 'default')

    def set_last_chat_name(self, chat_name):
        set_key(self.env_file, 'LAST_CHAT_NAME', chat_name)
        load_dotenv(self.env_file)

