
# RetroChat v2
![RetroChat Screenshot](https://i.imgur.com/5hh7cVb.png)
RetroChat is a powerful command-line interface for interacting with various AI language models. It provides a seamless experience for engaging with different chat providers while offering robust features for managing and customizing your conversations.

## 🚀 Installation

**One installer for all platforms (Windows/Linux/macOS):**

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | iex"

# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/DefamationStation/Retrochat-v2/main/install.ps1 | pwsh -
```

Then run `rchat` from anywhere! 🎉

**What the installer does:**
- Downloads to `~/.retrochat/`
- Sets up Python environment 
- Installs dependencies
- Creates global `rchat` command
- Enables auto-updates

**Interactive installation:**
```bash
# Download first, then run with options
./install.ps1 -Interactive
./install.ps1 -Force
./install.ps1 -CustomPath "C:\MyApps\RetroChat"
```

## ✨ Features

- **🤖 Multi-Provider Support**: Ollama, Anthropic, OpenAI, Google, OpenRouter
- **📝 Document RAG**: Load and query local documents (PDF, Word, Markdown, TXT)
- **🔄 Auto-Updates**: Always get the latest features automatically
- **⚙️ Customizable Settings**: Fine-tune AI behavior with adjustable parameters
- **✏️ Conversation Editing**: Edit entire conversation history
- **📋 Multi-line Input**: Enter complex queries with ease
- **🎯 Command System**: Control everything with '/' prefixed commands
- **🔢 Token Counting**: Track usage for messages and conversations
- **🔀 Provider Switching**: Switch between AI providers mid-conversation

## 📚 Document RAG System

**ATTENTION**: RAG only works with `nomic-embed-text` on Ollama currently.

1. Create a folder in `~/.retrochat/` and put your files in it
2. Type `/load <folder_name>` to process documents
3. Use `@<folder_name> <question>` to query documents

## 🛠️ Manual Installation

For development or custom setup:

1. Clone the repository: `git clone https://github.com/DefamationStation/Retrochat-v2.git`
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/macOS)
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python retrochat.py`
   ```
   cd C:\Users\<your username>\.retrochat
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
