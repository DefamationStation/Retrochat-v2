from rich.panel import Panel
from rich.markdown import Markdown
from utils.console import console


class DisplayManager:
    def __init__(self, chat_app):
        self.chat_app = chat_app

    def display_chat_history(self):
        if self.chat_app.current_session and self.chat_app.current_session.chat_history:
            for message in self.chat_app.current_session.chat_history:
                if message.role == "user":
                    console.print(Markdown(message.content), style="green")
                else:
                    formatted_content, _ = self.chat_app.code_block_formatter.format_code_blocks(message.content)
                    for line in formatted_content:
                        if isinstance(line, Panel):
                            console.print(line)
                        elif isinstance(line, str):
                            console.print(Markdown(line), style="yellow")
                        else:
                            console.print(str(line), style="yellow")