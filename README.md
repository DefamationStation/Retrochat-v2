# RetroChat

RetroChat is a powerful command-line interface for interacting with various AI language models. It provides a seamless experience for engaging with different chat providers while offering robust features for managing and customizing your conversations.

## Features

- **Multi-Provider Support**: Choose between Ollama, Anthropic (Claude), and OpenAI (GPT) models.
- **Persistent Chat History**: Save and resume conversations across sessions.
- **Chat Management**: Create, rename, delete, and switch between multiple chat sessions.
- **Customizable Settings**: Fine-tune AI behavior with adjustable parameters for all providers.
- **Conversation Editing**: Edit entire conversation history using your preferred text editor.
- **System Message Support**: Set global system messages to guide AI behavior across all chats.
- **Multi-line Input**: Enter complex queries or code snippets with ease.
- **Command System**: Control various aspects of the chat and application with '/' prefixed commands.
- **Local Setup**: Easy installation in your home directory for system-wide access.
- **Auto-Update**: Check for and apply updates automatically.
- **Provider Switching**: Easily switch between different AI providers and models during a session.
- **Token Counting**: Display token usage for messages and entire conversations.
- **Environment Variable Management**: Securely store and manage API keys and other settings.
- **Cross-Platform Support**: Works on Windows, macOS, and Linux.

## Installation

To run RetroChat, you need Python 3.7 or higher installed on your system. Follow these steps to set up the environment:

1. Download `retrochat.py` and the `requirements.txt` files.
2. Navigate to the project directory or right-click in the directory and select 'Open in Terminal'.
   ```
   cd path/to/retrochat
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the script with the setup flag to configure RetroChat:
   ```
   python retrochat.py --setup
   ```

## Usage

After installation, you can start RetroChat by running the shortcut command:
```
rchat
```
This command can be used from any directory in your terminal.

## Commands

RetroChat supports various commands to manage your chat sessions and settings:

- `/chat rename <new_name>` - Rename the current chat
- `/chat delete` - Delete the current chat
- `/chat new <chat_name>` - Create a new chat
- `/chat reset` - Reset the current chat history
- `/chat list` - List all available chats
- `/chat open <chat_name>` - Open a specific chat
- `/set system <message>` - Set the system message
- `/set` - Show available parameters and their current values
- `/set <parameter> <value>` - Set a parameter
- `/edit` - Edit the entire conversation
- `/show length` - Display the total conversation tokens
- `/switch` - Switch to a different provider or model
- `/help` - Display the help message
- `/exit` - Exit the program

## Configuration

RetroChat uses a `.env` file to store configuration settings. This file is automatically created in the `.retrochat` directory in your home folder. You can manually edit this file to set API keys and other preferences.

## Updates

RetroChat checks for updates automatically when you start the application. If updates are available, you'll be prompted to install them.

[Watch the demo video](https://vimeo.com/981646011)

## Contributing

Contributions to RetroChat are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

MIT License.
