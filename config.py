"""
Configuration module for Retrochat-v2

This module contains all configuration constants and settings for the application.
"""

import os


class Config:
    """Configuration class containing all application settings and constants."""
    
    # Base directories
    USER_HOME = os.path.expanduser('~')
    RETROCHAT_DIR = os.path.join(USER_HOME, '.retrochat')
    
    # File paths
    ENV_FILE = os.path.join(RETROCHAT_DIR, '.env')
    DB_FILE = os.path.join(RETROCHAT_DIR, 'chat_history.db')
    SETTINGS_FILE = os.path.join(RETROCHAT_DIR, 'settings.json')
    RETROCHAT_SCRIPT = os.path.join(RETROCHAT_DIR, 'retrochat.py')
    CHROMA_PATH = os.path.join(RETROCHAT_DIR, "chroma")
    
    # API Key environment variable names
    ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY"
    OPENAI_API_KEY_NAME = "OPENAI_API_KEY"
    GOOGLE_API_KEY_NAME = "GOOGLE_API_KEY"
    OPENROUTER_API_KEY_NAME = "OPENROUTER_API_KEY"
    
    # Configuration keys
    OPENROUTER_MODELS_KEY = "OPENROUTER_MODELS"
    LAST_CHAT_NAME_KEY = "LAST_CHAT_NAME"
    OLLAMA_IP_KEY = "OLLAMA_IP"
    OLLAMA_PORT_KEY = "OLLAMA_PORT"
    LAST_PROVIDER_KEY = "LAST_PROVIDER"
    LAST_MODEL_KEY = "LAST_MODEL"
    LMSTUDIO_BASE_URL_KEY = "LMSTUDIO_BASE_URL"

    @classmethod
    def initialize(cls):
        """Initialize the configuration by creating necessary directories."""
        os.makedirs(cls.RETROCHAT_DIR, exist_ok=True)
