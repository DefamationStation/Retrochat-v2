"""
Console management utility for Retrochat-v2

This module provides a wrapper around the Rich console for consistent UI output.
"""

from rich.console import Console
from rich.prompt import Prompt


class ConsoleManager:
    """Wrapper around Rich console for consistent UI output."""
    
    def __init__(self):
        self.console = Console()

    def print(self, message, style="default", end="\n"):
        self.console.print(message, style=style, end=end)

    def clear(self):
        self.console.clear()

    def ask(self, prompt, choices=None):
        return Prompt.ask(prompt, choices=choices)


# Global console instance for application-wide use
console = ConsoleManager()
