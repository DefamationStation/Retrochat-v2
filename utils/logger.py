"""
Logger utility for Retrochat-v2

This module provides a simple logging interface for the application.
"""

import logging


class Logger:
    """Simple logging utility class with static methods for different log levels."""
    
    @staticmethod
    def debug(message):
        logging.debug(message)

    @staticmethod
    def info(message):
        logging.info(message)

    @staticmethod
    def warning(message):
        logging.warning(message)

    @staticmethod
    def error(message):
        logging.error(message)
