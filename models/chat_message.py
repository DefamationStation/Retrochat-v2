"""
Chat message data model for Retrochat-v2

This module defines the ChatMessage dataclass used throughout the application.
"""

from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Data class representing a single chat message."""
    role: str
    content: str

    def __post_init__(self):
        self.content = str(self.content)

    def to_dict(self) -> dict:
        """Convert the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        """Create a ChatMessage from a dictionary."""
        return cls(
            role=data["role"],
            content=data["content"]
        )
