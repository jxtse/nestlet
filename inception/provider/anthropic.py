"""
Anthropic Provider implementation.

Supports Claude models via the Anthropic API.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

from inception.config.settings import ProviderConfig
from inception.provider.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """
    Anthropic API provider for Claude models.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize the Anthropic provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "anthropic"

    def _convert_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            elif msg.role == MessageRole.USER:
                converted.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                content: List[Dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                converted.append({"role": "assistant", "content": content or msg.content})
            elif msg.role == MessageRole.TOOL:
                # Find the last user message or create a new one with tool result
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }]
                })

        return system_prompt, converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tool definitions to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _parse_response(self, response: Any) -> CompletionResponse:
        """Parse Anthropic response."""
        content_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return CompletionResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "end_turn",
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            total_tokens=(
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage else 0
            ),
        )

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
            **kwargs: Additional options

        Returns:
            CompletionResponse
        """
        system_prompt, converted_messages = self._convert_messages(messages)

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice:
                if tool_choice == "auto":
                    params["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    # Don't pass tool_choice to disable tools
                    del params["tools"]
                elif tool_choice == "required":
                    params["tool_choice"] = {"type": "any"}
                else:
                    params["tool_choice"] = {"type": "tool", "name": tool_choice}

        logger.debug(f"Calling Anthropic API with model: {self.model_name}")
        response = await self._client.messages.create(**params)

        return self._parse_response(response)

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
            response = await self.complete(
                messages=history,
                tools=tools,
                tool_choice="auto",
                **kwargs
            )

            # Add assistant message
            history.append(Message.assistant(
                content=response.content,
                tool_calls=response.tool_calls if response.has_tool_calls else None
            ))

            if not response.has_tool_calls:
                return response, history

            # Execute tool calls
            for tool_call in response.tool_calls:
                logger.debug(f"Executing tool: {tool_call.name}")
                result = await tool_executor(tool_call)
                history.append(result.to_message())

            iterations += 1

        logger.warning(f"Max tool iterations ({max_iterations}) reached")
        return response, history
