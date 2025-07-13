import re
from rich.panel import Panel
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text
from utils.console import console


class DisplayManager:
    def __init__(self, chat_app):
        self.chat_app = chat_app

    def display_think_thought(self, thought: str):
        """Display the thought process in a panel."""
        if thought:
            panel_content = Text(thought.strip(), style="bright_black")
            panel = Panel(
                Padding(panel_content, (1, 2)),
                title="[dim]Model thoughts[/dim]",
                border_style="bright_black",
                expand=False
            )
            console.print(panel)

    def display_chat_history(self):
        if self.chat_app.current_session and self.chat_app.current_session.chat_history:
            for message in self.chat_app.current_session.chat_history:
                if message.role == "user":
                    console.print(Markdown(message.content), style="green")
                else:
                    content = message.content
                    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                    
                    if think_match:
                        thought = think_match.group(1)
                        self.display_think_thought(thought)
                        content = content.replace(think_match.group(0), "").strip()

                    if content:
                        formatted_content, _ = self.chat_app.code_block_formatter.format_code_blocks(content)
                        for line in formatted_content:
                            if isinstance(line, Panel):
                                console.print(line)
                            elif isinstance(line, str):
                                console.print(Markdown(line), style="yellow")
                            else:
                                console.print(str(line), style="yellow")