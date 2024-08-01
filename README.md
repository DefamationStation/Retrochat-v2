# RetroChat
![RetroChat Screenshot](https://i.imgur.com/5hh7cVb.png)
RetroChat is a powerful command-line interface for interacting with various AI language models. It provides a seamless experience for engaging with different chat providers while offering robust features for managing and customizing your conversations.

ATTENTION: RAG only works with nomic-embed-text currently, all you need to do is have it on Ollama.

1. Create a folder in user/.retrochat and put all your files in it.
2. In the chat type /load <folder name> and you'll get a message if it's executed successfully.
3. Then use @<folder name> to ask that specific folder's documents questions.

## Features

- **Multi-Provider Support**: Choose between Ollama, Anthropic (Claude), and OpenAI (GPT) models.
- **Customizable Settings**: Fine-tune AI behavior with adjustable parameters for all providers.
- **Conversation Editing**: Edit entire conversation history using your preferred text editor.
- **Multi-line Input**: Enter complex queries or code snippets with ease.
- **Command System**: Control various aspects of the chat and application with '/' prefixed commands.
- **Local Setup**: Easy installation in your home directory for system-wide access.
- **Auto-Update**: Check for and apply updates automatically.
- **Provider Switching**: Easily switch between different AI providers and models during a session.
- **Token Counting**: Display token usage for messages and entire conversations.
- **Document Loading and Querying**: Load documents from local folders and query them using AI models.

## Installation

To run RetroChat, you need Python 3.7 or higher installed on your system. [Link to Python 3.12 from the MS store.](https://apps.microsoft.com/detail/9ncvdn91xzqp?hl=en-US&gl=US) Follow these steps to set up the environment:

1. Download `retrochat.py` and the `requirements.txt` files and either run them from anywhere 
   or place them in the 'C:\Users\your username\\.retrochat' directory.
2. Navigate to the project directory or right-click in the directory and select 'Open in Terminal'.
   ```
   cd C:\Users\<you username\.retrochat
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the script with the setup flag to configure RetroChat:
   ```
   python retrochat.py --setup
   ```
NOTE: If chromadb doesn't properly install for you, all you need to do is download and install [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and after installing that navigate to the individual components and install MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest) and Windows 11 SDK (10.0.22621.0)
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
- `/show context` - Display the context of the last query
- `/switch` - Switch to a different provider or model
- `/help` - Display the help message
- `/exit` - Exit the program
- `/load <folder_name>` - Load documents from a specified folder
- `@<folder_name> <query>` - Query loaded documents from a specific folder

## Document Loading and Querying

RetroChat now supports loading and querying documents:

- Use `/load <folder_name>` to load documents from a specific folder in your `.retrochat` directory.
- Query loaded documents using `@<folder_name> <your question>`.
- Supported file types include .txt, .pdf, .doc, .docx, and .md.

## Configuration

RetroChat uses a `.env` file to store configuration settings. This file is automatically created in the `.retrochat` directory in your home folder. You can manually edit this file to set API keys and other preferences.

## Updates

RetroChat checks for updates automatically when you start the application. If updates are available, you'll be prompted to install them.

[Watch the demo video](https://vimeo.com/981646011)

## Contributing

Contributions to RetroChat are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

MIT License.
