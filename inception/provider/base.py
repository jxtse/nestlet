"""
Base LLM Provider abstraction.

Defines the interface for all LLM providers, supporting:
- Chat completions
- Tool/function calling
- Streaming (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable


class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ImageContent:
    """Image content for multimodal messages."""
    # Either url or base64 data
    url: Optional[str] = None
    base64_data: Optional[str] = None
    media_type: str = "image/png"  # image/png, image/jpeg, image/gif, image/webp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        if self.url:
            return {
                "type": "image_url",
                "image_url": {"url": self.url}
            }
        elif self.base64_data:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self.media_type};base64,{self.base64_data}"
                }
            }
        else:
            raise ValueError("ImageContent must have either url or base64_data")


@dataclass
class Message:
    """A message in the conversation."""
    role: MessageRole
    content: str
    # For multimodal content (images)
    images: Optional[List[ImageContent]] = None
    # For tool calls from assistant
    tool_calls: Optional[List[ToolCall]] = None
    # For tool responses
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # Tool name for tool messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result: Dict[str, Any] = {
            "role": self.role.value,
        }

        # Handle multimodal content
        if self.images:
            content_parts: List[Dict[str, Any]] = []
            if self.content:
                content_parts.append({"type": "text", "text": self.content})
            for img in self.images:
                content_parts.append(img.to_dict())
            result["content"] = content_parts
        else:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str, images: Optional[List[ImageContent]] = None) -> Message:
        """Create a user message, optionally with images."""
        return cls(role=MessageRole.USER, content=content, images=images)

    @classmethod
    def user_with_image(cls, content: str, image_path: str) -> Message:
        """Create a user message with an image from file path."""
        import base64
        from pathlib import Path

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Determine media type
        suffix = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")

        # Read and encode
        with open(path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")

        return cls(
            role=MessageRole.USER,
            content=content,
            images=[ImageContent(base64_data=base64_data, media_type=media_type)]
        )

    @classmethod
    def user_with_image_url(cls, content: str, image_url: str) -> Message:
        """Create a user message with an image URL."""
        return cls(
            role=MessageRole.USER,
            content=content,
            images=[ImageContent(url=image_url)]
        )

    @classmethod
    def assistant(
        cls,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> Message:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> Message:
        """Create a tool response message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name
        )


@dataclass
class ToolCall:
    """A tool call requested by the model."""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        import json
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            }
        }


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_call_id: str
    name: str
    result: Any
    success: bool = True
    error: Optional[str] = None

    def to_message(self) -> Message:
        """Convert to a tool message for the conversation."""
        if self.success:
            content = str(self.result) if self.result is not None else ""
        else:
            content = f"Error: {self.error}"
        return Message.tool(
            content=content,
            tool_call_id=self.tool_call_id,
            name=self.name
        )


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class CompletionResponse:
    """Response from a completion request."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    # Usage stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implementations should handle:
    - Authentication
    - Rate limiting
    - Retries
    - Error handling
    """

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: Conversation history
            tools: Available tools for the model to call
            tool_choice: "auto", "none", or specific tool name
            **kwargs: Provider-specific options

        Returns:
            CompletionResponse with content and/or tool calls
        """
        pass

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        tool_executor: Callable[[ToolCall], Awaitable[ToolResult]],
        max_iterations: int = 10,
        **kwargs: Any
    ) -> tuple[CompletionResponse, List[Message]]:
        """
        Complete with automatic tool execution loop.

        Continues until the model stops calling tools or max_iterations reached.

        Args:
            messages: Conversation history
            tools: Available tools
            tool_executor: Async function to execute tool calls
            max_iterations: Maximum tool call iterations
            **kwargs: Provider-specific options

        Returns:
            Tuple of (final response, full message history)
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
