#!/usr/bin/env python3
"""
Test script to verify the Model thoughts panel functionality during streaming.
"""

import asyncio
import re
from rich.panel import Panel
from rich.text import Text
from rich.padding import Padding
from utils.console import console

def test_think_detection():
    """Test the think tag detection logic."""
    
    # Test content with think tags
    test_content = """
    <think>
    The user is asking about the model thoughts panel. I need to consider:
    1. How streaming works
    2. How to detect think tags during streaming
    3. How to show an animated panel
    </think>
    
    Here's the response to your question about the Model thoughts panel.
    """
    
    # Simulate streaming chunks
    chunks = [
        "<th", "ink>", "\nThe user is asking about the model thoughts panel. I need to consider:\n1. How streaming works\n2. How to detect think tags during streaming\n3. How to show an animated panel\n</th", "ink>\n\nHere's the response to your question about the Model thoughts panel."
    ]
    
    accumulated_content = ""
    in_think_block = False
    think_content = ""
    think_panel_shown = False
    
    for chunk in chunks:
        accumulated_content += chunk
        
        # Check for think tags during streaming
        if "<think>" in accumulated_content and not in_think_block:
            in_think_block = True
            if not think_panel_shown:
                console.print("🤔 Detected think tags - showing loading panel")
                display_streaming_thoughts()
                think_panel_shown = True
        
        if in_think_block and "</think>" in accumulated_content:
            in_think_block = False
            # Extract the complete think content
            think_match = re.search(r"<think>(.*?)</think>", accumulated_content, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
    
    # Show completed thoughts
    if think_panel_shown and think_content:
        console.print("✅ Streaming completed - showing final thoughts")
        display_streaming_thoughts_complete(think_content)
        
        # Show content without think tags
        final_content = re.sub(r"<think>.*?</think>", "", accumulated_content, flags=re.DOTALL).strip()
        console.print("\n📝 Final response content:")
        console.print(final_content, style="yellow")

def display_streaming_thoughts():
    """Display a loading panel for model thoughts during streaming."""
    panel_content = Text("Processing thoughts...", style="bright_black")
    panel = Panel(
        Padding(panel_content, (1, 2)),
        title="[dim]🤔 Model thoughts[/dim]",
        border_style="bright_black", 
        expand=False
    )
    console.print(panel)

def display_streaming_thoughts_complete(thought: str):
    """Display the completed thought after streaming."""
    panel_content = Text(thought.strip(), style="bright_black")
    panel = Panel(
        Padding(panel_content, (1, 2)),
        title="[dim]💭 Model thoughts[/dim]",
        border_style="bright_black",
        expand=False
    )
    console.print(panel)

if __name__ == "__main__":
    console.print("Testing Model Thoughts Panel Functionality", style="bold cyan")
    console.print("=" * 50, style="cyan")
    test_think_detection()
    console.print("\n✅ Test completed!", style="bold green")
