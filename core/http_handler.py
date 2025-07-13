"""
Unified HTTP request/response handler for all chat providers.

This module eliminates duplicate HTTP handling code across providers by providing
a common interface for API communication, streaming, and error handling.
"""

import json
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils.console import console


@dataclass
class RequestConfig:
    """Configuration for HTTP requests."""
    url: str
    headers: Dict[str, str]
    json_data: Dict[str, Any]
    stream: bool = True
    timeout: int = 30


@dataclass 
class StreamChunk:
    """Represents a chunk of streaming data."""
    content: str
    is_complete: bool = False
    raw_data: Optional[Dict[str, Any]] = None


class ResponseProcessor(ABC):
    """Abstract base class for processing provider-specific responses."""
    
    @abstractmethod
    def process_stream_line(self, line: str) -> Optional[StreamChunk]:
        """Process a single line from a streaming response."""
        pass
    
    @abstractmethod
    def process_non_stream_response(self, response_data: Dict[str, Any]) -> StreamChunk:
        """Process a non-streaming response."""
        pass
    
    @abstractmethod
    def extract_error_message(self, status: int, response_text: str) -> str:
        """Extract error message from failed response."""
        pass


class OpenAIResponseProcessor(ResponseProcessor):
    """Response processor for OpenAI-compatible APIs (OpenAI, LM Studio)."""
    
    def process_stream_line(self, line: str) -> Optional[StreamChunk]:
        if not line.startswith("data: "):
            return None
            
        if line == "data: [DONE]":
            return StreamChunk("", is_complete=True)
            
        json_str = line[6:]
        try:
            response_json = json.loads(json_str)
            content = response_json['choices'][0]['delta'].get('content', '')
            return StreamChunk(content, raw_data=response_json) if content else None
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
    
    def process_non_stream_response(self, response_data: Dict[str, Any]) -> StreamChunk:
        try:
            content = response_data['choices'][0]['message']['content']
            return StreamChunk(content, is_complete=True, raw_data=response_data)
        except (KeyError, IndexError):
            return StreamChunk("", is_complete=True)
    
    def extract_error_message(self, status: int, response_text: str) -> str:
        return f"Error: {status} - {response_text}"


class AnthropicResponseProcessor(ResponseProcessor):
    """Response processor for Anthropic API."""
    
    def process_stream_line(self, line: str) -> Optional[StreamChunk]:
        # Anthropic typically doesn't use streaming in this implementation
        return None
    
    def process_non_stream_response(self, response_data: Dict[str, Any]) -> StreamChunk:
        try:
            content = response_data.get('content', [{}])[0].get('text', '')
            return StreamChunk(content, is_complete=True, raw_data=response_data)
        except (IndexError, KeyError):
            return StreamChunk("", is_complete=True)
    
    def extract_error_message(self, status: int, response_text: str) -> str:
        return f"Error: {status} - {response_text}"


class OllamaResponseProcessor(ResponseProcessor):
    """Response processor for Ollama API."""
    
    def process_stream_line(self, line: str) -> Optional[StreamChunk]:
        try:
            response_json = json.loads(line)
            content = response_json.get('message', {}).get('content', '')
            is_complete = response_json.get('done', False)
            return StreamChunk(content, is_complete=is_complete, raw_data=response_json)
        except json.JSONDecodeError:
            return None
    
    def process_non_stream_response(self, response_data: Dict[str, Any]) -> StreamChunk:
        content = response_data.get('message', {}).get('content', '')
        return StreamChunk(content, is_complete=True, raw_data=response_data)
    
    def extract_error_message(self, status: int, response_text: str) -> str:
        return f"Error: {status} - {response_text}"


class UnifiedHttpHandler:
    """Unified HTTP handler for all chat providers."""
    
    def __init__(self, response_processor: ResponseProcessor):
        self.response_processor = response_processor
    
    async def send_request(self, config: RequestConfig) -> AsyncGenerator[Union[str, None], None]:
        """
        Send HTTP request and yield response chunks.
        
        Args:
            config: Request configuration
            
        Yields:
            str: Content chunks from the response
            None: Signals end of streaming
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    config.url,
                    headers=config.headers,
                    json=config.json_data,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_msg = self.response_processor.extract_error_message(
                            response.status, await response.text()
                        )
                        console.print(error_msg, style="bold red")
                        yield error_msg
                        return
                    
                    if config.stream:
                        async for chunk in self._process_streaming_response(response):
                            yield chunk
                    else:
                        async for chunk in self._process_non_streaming_response(response):
                            yield chunk
                        
            except aiohttp.ClientError as e:
                error_msg = f"Connection error: {str(e)}"
                console.print(error_msg, style="bold red")
                yield error_msg
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                console.print(error_msg, style="bold red")
                yield error_msg
    
    async def _process_streaming_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Union[str, None], None]:
        """Process streaming response."""
        complete_message = ""
        
        async for line in response.content:
            if not line:
                continue
                
            line_str = line.decode('utf-8').strip()
            chunk = self.response_processor.process_stream_line(line_str)
            
            if chunk:
                if chunk.is_complete:
                    yield None  # Signal end of streaming
                    break
                elif chunk.content:
                    complete_message += chunk.content
                    yield chunk.content
        
        # If we collected a complete message, yield it at the end
        if complete_message:
            yield None  # Signal end of streaming
    
    async def _process_non_streaming_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Union[str, None], None]:
        """Process non-streaming response."""
        response_data = await response.json()
        chunk = self.response_processor.process_non_stream_response(response_data)
        
        if chunk.content:
            yield chunk.content
        yield None  # Signal end


class HttpHandlerFactory:
    """Factory for creating HTTP handlers with appropriate response processors."""
    
    _processors = {
        'openai': OpenAIResponseProcessor,
        'lmstudio': OpenAIResponseProcessor,  # LM Studio uses OpenAI-compatible API
        'anthropic': AnthropicResponseProcessor,
        'ollama': OllamaResponseProcessor,
    }
    
    @classmethod
    def create_handler(cls, provider_type: str) -> UnifiedHttpHandler:
        """Create HTTP handler for a specific provider type."""
        processor_class = cls._processors.get(provider_type.lower())
        if not processor_class:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        return UnifiedHttpHandler(processor_class())
    
    @classmethod
    def register_processor(cls, provider_type: str, processor_class: type):
        """Register a custom response processor for a provider."""
        cls._processors[provider_type.lower()] = processor_class
