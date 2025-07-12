"""
Logging suppression utility for Retrochat-v2

This module provides a context manager for temporarily suppressing logging output.
"""

import logging


class SuppressLogging:
    """Context manager for temporarily suppressing logging output."""
    
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)
