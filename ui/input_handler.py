from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout


class InputHandler:
    def __init__(self, chat_app):
        self.chat_app = chat_app

    async def get_multiline_input(self) -> str:
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

        prompt_session = PromptSession(multiline=True, key_bindings=kb)

        try:
            with patch_stdout():
                user_input = await prompt_session.prompt_async(
                    "",  # No prompt message
                    bottom_toolbar=None,  # Remove the code blocks counter
                )
            return user_input.strip()
        except (EOFError, KeyboardInterrupt):
            return ""