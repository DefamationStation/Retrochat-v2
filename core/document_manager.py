"""
Document management for Retrochat-v2

This module provides functionality for loading, processing, and querying documents
using vector embeddings and Chroma database.
"""

import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import Config
from utils.console import console
from utils.env_manager import EnvManager


def get_embedding_function():
    """Get the embedding function for document processing."""
    EnvManager.load_env_variables()
    ollama_ip = EnvManager.get_env_variable(Config.OLLAMA_IP_KEY, 'localhost')
    ollama_port = EnvManager.get_env_variable(Config.OLLAMA_PORT_KEY, '11434')
    return OllamaEmbeddings(
        base_url=f"http://{ollama_ip}:{ollama_port}",
        model="nomic-embed-text"
    )


class DocumentManager:
    """Manages document loading, processing, and querying for RAG functionality."""
    
    def __init__(self):
        self.chroma_path = Config.CHROMA_PATH
        self.embedding_function = get_embedding_function()

    def load_documents(self, folder_name: str) -> bool:
        """Load documents from a specified folder and add them to the vector database."""
        data_path = os.path.join(Config.RETROCHAT_DIR, folder_name)
        console.print(f"Attempting to load documents from: {data_path}", style="cyan")
        
        if not os.path.exists(data_path):
            console.print(f"Folder '{data_path}' does not exist.", style="bold red")
            return False
        
        try:
            documents = []
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if os.path.isfile(file_path):
                    loader = self.get_loader_for_file(file_path)
                    if loader:
                        # Load the content of the file and combine into a single document
                        loaded_docs = loader.load()
                        if loaded_docs:
                            combined_content = "\n".join([doc.page_content for doc in loaded_docs])
                            combined_document = Document(page_content=combined_content, metadata={"source": filename})
                            documents.append(combined_document)
                            console.print(f"Loaded content from {filename} as a single document", style="green")
                    else:
                        console.print(f"No loader found for file: {filename}", style="yellow")
                else:
                    console.print(f"Skipping non-file: {filename}", style="yellow")
            
            if not documents:
                console.print("No valid documents found to load.", style="bold red")
                return False

            console.print(f"Total documents loaded: {len(documents)}", style="green")
            
            chunks = self.split_documents(documents)
            console.print(f"Documents split into {len(chunks)} chunks", style="green")
            
            self.add_to_chroma(chunks, folder_name)
            console.print("Chunks added to Chroma database", style="green")
            
            return True
        except Exception as e:
            console.print(f"Error loading documents: {str(e)}", style="bold red")
            return False

    def get_loader_for_file(self, file_path: str) -> Optional[BaseLoader]:
        """Get the appropriate document loader for a given file type."""
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.pdf':
            return PyPDFDirectoryLoader(os.path.dirname(file_path))
        elif ext == '.txt':
            return TextLoader(file_path)
        elif ext in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            console.print(f"Unsupported file type: {file_path}", style="yellow")
            return None

    def split_documents(self, documents: List[Document]):
        """Split documents into smaller chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        console.print(f"Split {len(documents)} documents into {len(chunks)} chunks", style="green")
        return chunks

    def add_to_chroma(self, chunks: List[Document], folder_name: str):
        """Add document chunks to the Chroma vector database."""
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)

        chunks_with_ids = self.calculate_chunk_ids(chunks, folder_name)

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if new_chunks:
            console.print(f"-> Adding {len(new_chunks)} new chunks to the database", style="cyan")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            console.print("[OK] No new documents to add", style="green")

    def calculate_chunk_ids(self, chunks: List[Document], folder_name: str):
        """Calculate unique IDs for document chunks."""
        chunk_counter = {}
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "unknown")
            
            # Create a base ID
            base_id = f"{folder_name}/{source}:{page}"
            
            # If this base_id hasn't been seen before, initialize its counter
            if base_id not in chunk_counter:
                chunk_counter[base_id] = 0
            
            # Increment the counter for this base_id
            chunk_counter[base_id] += 1
            
            # Create a unique ID by appending the counter
            chunk_id = f"{base_id}:{chunk_counter[base_id] - 1}"
            
            # Store the unique ID in the chunk's metadata
            chunk.metadata["id"] = chunk_id

        return chunks

    def query_documents(self, folder_name: str, query: str) -> List[Document]:
        """Query documents in a specific folder using similarity search."""
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        all_docs = db.get()
        
        folder_docs_ids = [doc_id for doc_id in all_docs['ids'] if doc_id.startswith(f"{folder_name}/")]
        
        if not folder_docs_ids:
            console.print(f"No documents found for folder '{folder_name}'", style="yellow")
            return []
        
        results = db.similarity_search(
            query,
            k=min(5, len(folder_docs_ids)),
        )
        # Filter results to only those in folder_docs_ids
        filtered_results = [doc for doc in results if getattr(doc, 'metadata', {}).get('id', '').startswith(f"{folder_name}/")]
        return filtered_results
