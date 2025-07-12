"""
Tokenizer management utility for Retrochat-v2

This module provides utilities for calculating token counts in text.
"""

import tiktoken


class TokenizerManager:
    """Utility class for managing text tokenization."""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def calculate_tokens(self, text: str) -> int:
        """Calculate the number of tokens in the given text."""
        return len(self.tokenizer.encode(text))
