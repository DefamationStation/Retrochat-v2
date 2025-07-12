"""
Chat history management for Retrochat-v2

This module provides functionality for managing chat sessions, messages, and related data in SQLite.
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any

from config import Config
from utils.logger import Logger
from utils.console import console
from models.chat_message import ChatMessage


class ChatHistoryManager:
    """Manages chat history, sessions, and parameters in SQLite database."""
    
    def __init__(self, db_file: str, chat_name: str = 'default'):
        Config.initialize()  # Ensure the directory exists
        self.db_file = db_file
        self.chat_name = chat_name
        self.conn = sqlite3.connect(self.db_file)
        self._create_tables()
        self._update_schema()

    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        with self.conn:
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_name TEXT NOT NULL UNIQUE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    system_message TEXT,
                    parameters TEXT
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                );
                CREATE TABLE IF NOT EXISTS code_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    block_content TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                );
            ''')

    def _update_schema(self):
        """Update database schema by adding missing columns."""
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA table_info(chat_sessions)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'system_message' not in columns:
                    self.conn.execute('ALTER TABLE chat_sessions ADD COLUMN system_message TEXT')
                if 'parameters' not in columns:
                    self.conn.execute('ALTER TABLE chat_sessions ADD COLUMN parameters TEXT')
                Logger.info("Database schema updated successfully.")
        except Exception as e:
            Logger.error(f"Error updating database schema: {e}")

    def _get_session_id(self, chat_name: str) -> int:
        """Get or create session ID for the given chat name."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM chat_sessions WHERE chat_name = ?', (chat_name,))
        session = cursor.fetchone()
        if session:
            return session[0]
        cursor.execute('INSERT INTO chat_sessions (chat_name) VALUES (?)', (chat_name,))
        session_id = cursor.lastrowid
        if session_id is None:
            # Fallback: fetch the id again
            cursor.execute('SELECT id FROM chat_sessions WHERE chat_name = ?', (chat_name,))
            session = cursor.fetchone()
            if session:
                return session[0]
            else:
                raise RuntimeError("Failed to create or retrieve chat session id.")
        return session_id

    def set_chat_name(self, chat_name: str):
        """Set the current chat name."""
        self.chat_name = chat_name

    def save_history(self, history: List[ChatMessage]):
        """Save chat history to the database."""
        session_id = self._get_session_id(self.chat_name)
        try:
            with self.conn:
                self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
                self.conn.executemany('''
                    INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)
                ''', [(session_id, msg.role, str(msg.content)) for msg in history])
        except Exception as e:
            Logger.error(f"Error saving chat history: {e}")

    def save_code_blocks(self, code_blocks: List[str]):
        """Save code blocks to the database."""
        session_id = self._get_session_id(self.chat_name)
        try:
            with self.conn:
                self.conn.execute('DELETE FROM code_blocks WHERE session_id = ?', (session_id,))
                self.conn.executemany('''
                    INSERT INTO code_blocks (session_id, block_content) VALUES (?, ?)
                ''', [(session_id, block) for block in code_blocks])
        except Exception as e:
            Logger.error(f"Error saving code blocks: {e}")

    def load_code_blocks(self) -> List[str]:
        """Load code blocks from the database."""
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT block_content FROM code_blocks WHERE session_id = ? ORDER BY id', (session_id,))
        try:
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            Logger.error(f"Error loading code blocks: {e}")
            return []
        
    def load_history(self) -> List[ChatMessage]:
        """Load chat history from the database."""
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
        try:
            return [ChatMessage(role=row[0], content=str(row[1])) for row in cursor.fetchall()]
        except Exception as e:
            Logger.error(f"Error loading chat history: {e}")
            return []

    def save_system_message(self, system_message: str):
        """Save system message to the database."""
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('UPDATE chat_sessions SET system_message = ? WHERE id = ?', (system_message, session_id))

    def load_system_message(self) -> Optional[str]:
        """Load system message from the database."""
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT system_message FROM chat_sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        return result[0] if result else None

    def save_parameters(self, parameters: Dict[str, Any]):
        """Save chat parameters to the database."""
        session_id = self._get_session_id(self.chat_name)
        parameters_json = json.dumps(parameters)
        with self.conn:
            self.conn.execute('UPDATE chat_sessions SET parameters = ? WHERE id = ?', (parameters_json, session_id))

    def load_parameters(self) -> Dict[str, Any]:
        """Load chat parameters from the database."""
        session_id = self._get_session_id(self.chat_name)
        cursor = self.conn.cursor()
        cursor.execute('SELECT parameters FROM chat_sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        return json.loads(result[0]) if result and result[0] else {}

    def clear_history(self):
        """Clear chat history for the current session."""
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))

    def rename_history(self, new_name: str):
        """Rename the current chat session."""
        session_id = self._get_session_id(self.chat_name)
        try:
            with self.conn:
                self.conn.execute('UPDATE chat_sessions SET chat_name = ? WHERE id = ?', (new_name, session_id))
            self.set_chat_name(new_name)
        except sqlite3.IntegrityError:
            console.print(f"Error: A chat with the name '{new_name}' already exists.", style="bold red")

    def delete_history(self):
        """Delete the current chat session and all its data."""
        session_id = self._get_session_id(self.chat_name)
        with self.conn:
            self.conn.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
            self.conn.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        self.set_chat_name('default')

    def list_chats(self) -> List[str]:
        """List all available chat sessions."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT chat_name FROM chat_sessions')
        return [row[0] for row in cursor.fetchall()]
