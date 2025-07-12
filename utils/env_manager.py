"""
Environment variable management utility for Retrochat-v2

This module provides utilities for loading and managing environment variables.
"""

import os
from dotenv import load_dotenv, set_key
from config import Config


class EnvManager:
    """Utility class for managing environment variables."""
    
    @staticmethod
    def load_env_variables():
        """Load environment variables from the .env file."""
        if os.path.exists(Config.ENV_FILE):
            load_dotenv(Config.ENV_FILE, override=True)

    @staticmethod
    def set_env_variable(key, value):
        """Set an environment variable and save it to the .env file."""
        set_key(Config.ENV_FILE, key, value)
        load_dotenv(Config.ENV_FILE, override=True)

    @staticmethod
    def get_env_variable(key, default=None):
        """Get an environment variable with an optional default value."""
        return os.getenv(key, default)
