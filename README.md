# RetroChat

## Installation

To run RetroChat, you need Python 3.7 or higher installed on your system. Follow these steps to set up the environment:

1. Clone the repository or download the source code.

2. Navigate to the project directory:
   ```
   cd path/to/retrochat
   ```

3. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the script:
   ```
   python retrochat.py
   ```

## Usage

After installation, you can start RetroChat by running:

```
python retrochat.py
```

For first-time setup, use:

```
python retrochat.py --setup
```

This will set up the necessary directories and files for RetroChat to function properly.

## Updating

To update the required packages to their latest versions, run:

```
pip install --upgrade -r requirements.txt
```

## Troubleshooting

If you encounter any issues with package installation, ensure that you have the latest version of pip:

```
pip install --upgrade pip
```

Then try installing the requirements again.

If you still face problems, please open an issue on the GitHub repository with details about the error and your system configuration.