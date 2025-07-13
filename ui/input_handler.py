from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from typing import Optional, List


class InputHandler:
    def __init__(self, chat_app):
        self.chat_app = chat_app
        self._prompt_session = None

    def _get_prompt_session(self):
        """Get or create the prompt session for consistent input handling."""
        if self._prompt_session is None:
            kb = KeyBindings()

            @kb.add('c-j')  # Ctrl+J as a workaround for Ctrl+Enter
            def _(event):
                event.current_buffer.insert_text('\n')

            @kb.add('enter')  # Enter key
            def _(event):
                if event.current_buffer.document.is_cursor_at_the_end:
                    event.current_buffer.validate_and_handle()
                else:
                    event.current_buffer.insert_text('\n')

            self._prompt_session = PromptSession(multiline=True, key_bindings=kb)
        return self._prompt_session

    async def get_multiline_input(self) -> str:
        """Get multiline input for chat messages."""
        prompt_session = self._get_prompt_session()

        try:
            with patch_stdout():
                user_input = await prompt_session.prompt_async(
                    "",  # No prompt message
                    bottom_toolbar=None,  # Remove the code blocks counter
                )
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    async def get_single_input(self, prompt: str, choices: Optional[List[str]] = None) -> str:
        """Get single line input with optional choices validation."""
        try:
            with patch_stdout():
                simple_session = PromptSession()
                while True:
                    user_input = await simple_session.prompt_async(f"{prompt}: ")
                    user_input = user_input.strip()
                    
                    if choices and user_input not in choices:
                        from utils.console import console
                        console.print(f"Invalid choice. Please choose from: {', '.join(choices)}", style="bold red")
                        continue
                    
                    return user_input
        except (EOFError, KeyboardInterrupt):
            return ""