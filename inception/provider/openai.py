"""
OpenAI Provider implementation.

Supports:
- OpenAI API
- Azure OpenAI
- OpenAI-compatible APIs (via base_url)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from inception.config.settings import ProviderConfig, ProviderType
from inception.provider.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider.

    Also supports Azure OpenAI and OpenAI-compatible APIs.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize the OpenAI provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._client = self._create_client()

    def _create_client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        """Create the appropriate async client."""
        if self.config.type == ProviderType.AZURE:
            return AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.api_version or "2024-02-15-preview",
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
            )
        else:
            return AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
            )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        if self.config.type == ProviderType.AZURE:
            return self.config.azure_deployment or self.config.model
        return self.config.model

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.config.type.value

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        result = []
        for msg in messages:
            msg_dict = msg.to_dict()
            # Handle tool_calls format
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in msg.tool_calls
                ]
            result.append(msg_dict)
        return result

    def _parse_tool_calls(self, tool_calls: Any) -> List[ToolCall]:
        """Parse tool calls from OpenAI response."""
        if not tool_calls:
            return []

        result = []
        for tc in tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {tc.function.arguments}")
                arguments = {}

            result.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=arguments
            ))
        return result

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Generate a completion.

        Args:
            messages: Conversation history
            tools: Available tools
            tool_choice: Tool selection mode
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            CompletionResponse
        """
        # Build request parameters
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # Add tools if provided
        if tools:
            params["tools"] = [t.to_dict() for t in tools]
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = tool_choice
                else:
                    # Specific tool name
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice}
                    }

        # Make the API call
        logger.debug(f"Calling OpenAI API with model: {self.model_name}")
        response = await self._client.chat.completions.create(**params)

        # Parse response
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = self._parse_tool_calls(choice.message.tool_calls)

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

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

        Args:
            messages: Initial conversation history
            tools: Available tools
            tool_executor: Function to execute tool calls
            max_iterations: Maximum iterations
            **kwargs: Additional options

        Returns:
            Tuple of (final response, complete message history)
        """
        history = list(messages)
        iterations = 0

        while iterations < max_iterations:
            # Get completion
            response = await self.complete(
                messages=history,
                tools=tools,
                tool_choice="auto",
                **kwargs
            )

            # Add assistant message to history
            history.append(Message.assistant(
                content=response.content,
                tool_calls=response.tool_calls if response.has_tool_calls else None
            ))

            # Check if we should stop
            if not response.has_tool_calls:
                return response, history

            # Execute tool calls
            for tool_call in response.tool_calls:
                logger.debug(f"Executing tool: {tool_call.name}")
                result = await tool_executor(tool_call)
                history.append(result.to_message())

            iterations += 1

        # Max iterations reached
        logger.warning(f"Max tool iterations ({max_iterations}) reached")
        return response, history
