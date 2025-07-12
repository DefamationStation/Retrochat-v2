"""
Code block formatting for Retrochat-v2

This module provides functionality for formatting and displaying code blocks
in chat responses with syntax highlighting.
"""

import pyperclip
from rich.syntax import Syntax
from rich.panel import Panel

from utils.console import console


class CodeBlockFormatter:
    """Handles formatting and display of code blocks in chat responses."""
    
    def __init__(self):
        self.total_blocks = 0

    def format_code_blocks(self, text):
        """Format code blocks in text with syntax highlighting and panels."""
        lines = text.split('\n')
        formatted_lines = []
        code_blocks = []
        in_code_block = False
        code_block = []
        language = ''

        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    # Close the current code block
                    if code_block:
                        code = '\n'.join(code_block)
                        self.total_blocks += 1
                        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
                        panel = Panel(syntax, border_style="bold", expand=False)
                        formatted_lines.append(panel)
                        formatted_lines.append(f'Code Block {self.total_blocks}')
                        code_blocks.append(code)
                    in_code_block = False
                    code_block = []
                    language = ''
                else:
                    # Start a new code block
                    in_code_block = True
                    language = line[3:].strip() or 'text'
            elif in_code_block:
                code_block.append(line)
            else:
                formatted_lines.append(line)

        # Handle any open code block at the end
        if in_code_block and code_block:
            code = '\n'.join(code_block)
            self.total_blocks += 1
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            panel = Panel(syntax, border_style="bold", expand=False)
            formatted_lines.append(panel)
            formatted_lines.append(f'Code Block {self.total_blocks}')
            code_blocks.append(code)

        return formatted_lines, code_blocks

    def reset(self):
        """Reset the block counter."""
        self.total_blocks = 0


def copy_code_to_clipboard(code):
    """Copy code to the system clipboard."""
    try:
        if isinstance(code, list):
            code = '\n'.join(code)  # Join list elements into a single string
        pyperclip.copy(code)
    except Exception as e:
        console.print(f"Failed to copy code to clipboard: {e}", style="bold red")
