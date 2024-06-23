import os
import sys
import time
from colorama import Fore

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text, delay=0.001, color=Fore.WHITE):
    for char in text:
        sys.stdout.write(color + char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_streaming(text, color=Fore.WHITE):
    sys.stdout.write(color + text)
    sys.stdout.flush()
