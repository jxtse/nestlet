"""
Base LLM Provider abstraction.

Defines the interface for all LLM providers, supporting:
- Chat completions
- Tool/function calling
- Streaming (future)

The data models (``Message``, ``ToolCall``, ``ToolDefinition``, ...) are pydantic
``BaseModel`` instances so payloads coming from YAML, CLI, or LLM tool calls are
validated at construction time. The classmethod constructors and ``to_dict``
shapes are preserved verbatim so existing callers do not need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ImageContent(BaseModel):
    """Image content for multimodal messages."""

    model_config = ConfigDict(extra="ignore")

    # Either url or base64 data
    url: Optional[str] = None
    base64_data: Optional[str] = None
    media_type: str = "image/png"  # image/png, image/jpeg, image/gif, image/webp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        if self.url:
            return {"type": "image_url", "image_url": {"url": self.url}}
        elif self.base64_data:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{self.media_type};base64,{self.base64_data}"},
            }
        else:
            raise ValueError("ImageContent must have either url or base64_data")


class ToolCall(BaseModel):
    """A tool call requested by the model."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI Chat Completions tool-call dict."""
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class Message(BaseModel):
    """A message in the conversation."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    role: MessageRole
    content: str
    images: Optional[List[ImageContent]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # Tool name for tool messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for OpenAI Chat Completions API calls."""
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
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str, images: Optional[List[ImageContent]] = None) -> "Message":
        """Create a user message, optionally with images."""
        return cls(role=MessageRole.USER, content=content, images=images)

    @classmethod
    def user_with_image(cls, content: str, image_path: str) -> "Message":
        """Create a user message with an image from file path."""
        import base64
        from pathlib import Path

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")

        with open(path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")

        return cls(
            role=MessageRole.USER,
            content=content,
            images=[ImageContent(base64_data=base64_data, media_type=media_type)],
        )

    @classmethod
    def user_with_image_url(cls, content: str, image_url: str) -> "Message":
        """Create a user message with an image URL."""
        return cls(role=MessageRole.USER, content=content, images=[ImageContent(url=image_url)])

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> "Message":
        """Create a tool response message."""
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id, name=name)


class ToolResult(BaseModel):
    """Result from executing a tool (provider-level wire shape)."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    tool_call_id: str
    name: str
    result: Any = None
    success: bool = True
    error: Optional[str] = None

    def to_message(self) -> Message:
        """Convert to a tool message for the conversation."""
        if self.success:
            content = str(self.result) if self.result is not None else ""
        else:
            content = f"Error: {self.error}"
        return Message.tool(content=content, tool_call_id=self.tool_call_id, name=self.name)


class ToolDefinition(BaseModel):
    """Definition of a tool for the LLM."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)  # JSON Schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI Chat Completions function format (nested)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_responses_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI Responses API function format (flat).

        The Responses API expects ``{"type": "function", "name": ..., ...}``
        directly without nesting under a ``function`` key.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class CompletionResponse(BaseModel):
    """Response from a completion request."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    # Usage stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Raw provider output (Responses API only). Holds the original
    # ``response.output`` array so the tool-call loop can re-feed reasoning
    # items per OpenAI's requirement. ``None`` on the Chat path.
    raw_output: Optional[List[Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


# ─── Streaming events ──────────────────────────────────────────────────────
# Providers that opt into streaming (`complete(stream=True)`) return an
# AsyncIterator of these events. The shapes mirror the Substrate / OpenAI Chat
# Completions SSE deltas: content tokens, reasoning summary tokens, and
# index-keyed tool-call argument deltas. UsageEvent / DoneEvent close out the
# stream.


class ContentDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["content"] = "content"
    text: str


class ReasoningDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["reasoning"] = "reasoning"
    text: str


class ToolCallDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments_chunk: Optional[str] = None


class UsageEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["usage"] = "usage"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
    type: Literal["done"] = "done"
    response: CompletionResponse


StreamEvent = Union[ContentDelta, ReasoningDelta, ToolCallDelta, UsageEvent, DoneEvent]


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
        **kwargs: Any,
    ) -> Union[CompletionResponse, AsyncIterator[StreamEvent]]:
        """
        Generate a completion for the given messages.

        Args:
            messages: Conversation history
            tools: Available tools for the model to call
            tool_choice: "auto", "none", or specific tool name
            **kwargs: Provider-specific options. Pass ``stream=True`` to get
                back an ``AsyncIterator[StreamEvent]`` instead of a single
                ``CompletionResponse``.

        Returns:
            ``CompletionResponse`` when ``stream`` is false / unset.
            ``AsyncIterator[StreamEvent]`` when ``stream=True``.
        """
        pass

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        tool_executor: Callable[[ToolCall], Awaitable[ToolResult]],
        max_iterations: int = 10,
        **kwargs: Any,
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
