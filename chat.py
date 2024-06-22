import os
import sys
import time
import requests
from colorama import init, Fore, Style
from dotenv import load_dotenv

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("ANTHROPIC_API_KEY")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text, delay=0.01, color=Fore.WHITE):  # Reduced delay from 0.03 to 0.01
    for char in text:
        sys.stdout.write(color + char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def chat_with_claude(message):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 4096,
        "temperature": 0.0,
        "system":"Keep your answers short and to the point.",
        "messages": [{"role": "user", "content": message}]
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['content'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"

def main():
    if not API_KEY:
        print_slow("Error: ANTHROPIC_API_KEY not found in environment variables.", color=Fore.RED)
        print_slow("Please set your API key as an environment variable named ANTHROPIC_API_KEY.", color=Fore.YELLOW)
        return

    clear_screen()
    print_slow("Welcome to the Retro Claude 3.5 Sonnet Chat Interface", color=Fore.CYAN)
    print_slow("Type 'exit' to quit the chat.", color=Fore.YELLOW)
    
    while True:
        user_input = input(Fore.GREEN)
        if user_input.lower() == 'exit':
            print_slow("Thank you for chatting. Goodbye!", color=Fore.CYAN)
            break
        
        response = chat_with_claude(user_input)
        print_slow(response, color=Fore.YELLOW)
        print()  # Add a newline only after Claude's response

if __name__ == "__main__":
    main()