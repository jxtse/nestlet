"""
OpenAI Provider implementation.

Supports:
- OpenAI Chat Completions API (legacy gpt-4*, etc.)
- OpenAI Responses API (gpt-5*, o1*, o3*, o4*) with reasoning controls
- Azure OpenAI (Chat Completions only)
- OpenAI-compatible APIs (via base_url)

The provider auto-selects the API surface based on
``ProviderConfig.resolved_api_mode()`` (which honors ``api_mode = auto | chat |
responses``). The Responses path is required for GPT-5.5 reasoning features.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from inception.config.settings import ProviderConfig, ProviderType
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

        # For OpenRouter, default to its API endpoint and key env var
        # if the user didn't explicitly set them.
        base_url = self.config.base_url
        api_key = self.config.api_key
        if self.config.type == ProviderType.OPENROUTER:
            base_url = base_url or "https://openrouter.ai/api/v1"
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
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

    def _use_responses_api(self) -> bool:
        """Whether to route requests through ``client.responses.create``.

        Azure OpenAI does not currently expose the Responses API through the
        deployment-based routing we use, so we always fall back to Chat
        Completions for Azure regardless of ``api_mode``.
        """
        if self.config.type == ProviderType.AZURE:
            return False
        return self.config.resolved_api_mode() == "responses"

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI Chat Completions format."""
        return [msg.to_dict() for msg in messages]

    def _max_tokens_param_name(self) -> str:
        """Choose the right token-budget param for the active model on the Chat path.

        OpenAI's gpt-5*, o1*, and o3* families reject ``max_tokens`` and require
        ``max_completion_tokens``. Older models only accept ``max_tokens``.
        """
        model = (self.model_name or "").lower()
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            return "max_completion_tokens"
        return "max_tokens"

    def _token_budget_kwargs(self, max_tokens: int) -> Dict[str, int]:
        """Pick the correct token-budget kwarg for the active API surface."""
        if self._use_responses_api():
            return {"max_output_tokens": max_tokens}
        return {self._max_tokens_param_name(): max_tokens}

    def _parse_tool_calls(self, tool_calls: Any) -> List[ToolCall]:
        """Parse tool calls from OpenAI Chat Completions response."""
        if not tool_calls:
            return []

        result = []
        for tc in tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {tc.function.arguments}")
                arguments = {}

            result.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
        return result

    # ------------------------------------------------------------------ Chat path

    async def _complete_chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> CompletionResponse:
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        params.update(self._token_budget_kwargs(kwargs.get("max_tokens", self.config.max_tokens)))

        if tools:
            params["tools"] = [t.to_dict() for t in tools]
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = tool_choice
                else:
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

        logger.debug(f"Calling OpenAI Chat Completions with model: {self.model_name}")
        response = await self._client.chat.completions.create(**params)

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

    # ------------------------------------------------------------ Responses path

    def _convert_messages_to_responses_input(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Translate our ``Message`` history to the Responses API ``input`` shape.

        System messages collapse into the top-level ``instructions`` parameter
        (the Responses API recommends this). Assistant messages with tool calls
        expand into one ``function_call`` item per call. Tool result messages
        become ``function_call_output`` items.
        """
        instructions_parts: List[str] = []
        items: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                if msg.content:
                    instructions_parts.append(msg.content)
                continue
            if msg.role == MessageRole.TOOL:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content,
                    }
                )
                continue
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                if msg.content:
                    items.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": msg.content}],
                        }
                    )
                for tc in msg.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    )
                continue
            # Plain user / assistant text. Images are not currently translated
            # on the Responses path — Chat Completions remains the multimodal
            # path for gpt-4o-style models.
            items.append({"role": msg.role.value, "content": msg.content})

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        return instructions, items

    def _build_reasoning_param(self) -> Optional[Dict[str, Any]]:
        """Build the ``reasoning`` kwarg if the config opts in."""
        if not self.config.reasoning_effort:
            return None
        reasoning: Dict[str, Any] = {"effort": self.config.reasoning_effort}
        if self.config.reasoning_summary:
            reasoning["summary"] = self.config.reasoning_summary
        return reasoning

    def _parse_responses_output(self, output: List[Any]) -> tuple[str, List[ToolCall]]:
        """Pull assistant text + tool calls out of ``response.output``."""
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for item in output:
            item_type = _attr(item, "type")
            if item_type == "message":
                for content in _attr(item, "content", []) or []:
                    if _attr(content, "type") == "output_text":
                        text_parts.append(_attr(content, "text", "") or "")
            elif item_type == "function_call":
                raw_args = _attr(item, "arguments", "") or ""
                try:
                    arguments = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments: {raw_args!r}")
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=_attr(item, "call_id") or _attr(item, "id") or "",
                        name=_attr(item, "name", "") or "",
                        arguments=arguments,
                    )
                )
        return "".join(text_parts), tool_calls

    async def _complete_responses(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> CompletionResponse:
        instructions, input_items = self._convert_messages_to_responses_input(messages)

        params: Dict[str, Any] = {
            "model": self.model_name,
            "input": input_items,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if instructions:
            params["instructions"] = instructions
        params.update(self._token_budget_kwargs(kwargs.get("max_tokens", self.config.max_tokens)))

        reasoning = self._build_reasoning_param()
        if reasoning:
            params["reasoning"] = reasoning

        if tools:
            params["tools"] = [t.to_responses_dict() for t in tools]
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = tool_choice
                else:
                    params["tool_choice"] = {"type": "function", "name": tool_choice}

        logger.debug(f"Calling OpenAI Responses API with model: {self.model_name}")
        response = await self._client.responses.create(**params)

        output = list(getattr(response, "output", []) or [])
        content, tool_calls = self._parse_responses_output(output)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        finish_reason = "tool_calls" if tool_calls else "stop"

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_output=output,
        )

    # ------------------------------------------------------------------ Public API

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using whichever OpenAI API surface fits the model."""
        if self._use_responses_api():
            return await self._complete_responses(messages, tools, tool_choice, **kwargs)
        return await self._complete_chat(messages, tools, tool_choice, **kwargs)

    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        tool_executor: Callable[[ToolCall], Awaitable[ToolResult]],
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> tuple[CompletionResponse, List[Message]]:
        """Drive the chat-style tool loop using whichever API surface is active.

        We keep the public ``Message``-based interface so callers don't need to
        know which API is in play. The Responses path stitches conversation
        state back into ``Message`` objects (assistant text + tool_calls, tool
        result), so subsequent calls re-derive the proper ``input`` shape via
        ``_convert_messages_to_responses_input``.
        """
        history = list(messages)
        iterations = 0
        response: Optional[CompletionResponse] = None

        while iterations < max_iterations:
            response = await self.complete(
                messages=history, tools=tools, tool_choice="auto", **kwargs
            )

            history.append(
                Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls if response.has_tool_calls else None,
                )
            )

            if not response.has_tool_calls:
                return response, history

            for tool_call in response.tool_calls:
                logger.debug(f"Executing tool: {tool_call.name}")
                result = await tool_executor(tool_call)
                history.append(result.to_message())

            iterations += 1

        logger.warning(f"Max tool iterations ({max_iterations}) reached")
        assert response is not None  # loop ran at least once
        return response, history


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or dict key — Responses items come back as pydantic
    models in the SDK but tests often hand us plain dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
