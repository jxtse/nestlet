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
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI

from inception.config.settings import ProviderConfig, ProviderType
from inception.provider.base import (
    BaseProvider,
    CompletionResponse,
    ContentDelta,
    DoneEvent,
    Message,
    MessageRole,
    ReasoningDelta,
    StreamEvent,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    ToolResult,
    UsageEvent,
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

    # ------------------------------------------------------------------ Chat path

    def _build_chat_params(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
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
        return params

    async def _complete_chat_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """SSE stream the Chat Completions response and yield StreamEvents.

        Substrate (and any OpenAI-compatible backend behind a custom
        ``base_url``) requires ``stream=True`` for long completions, otherwise
        the gateway kills the request on timeout. We always stream internally;
        the public ``complete()`` only collects events when the caller didn't
        opt in to streaming.
        """
        params = self._build_chat_params(messages, tools, tool_choice, **kwargs)
        params["stream"] = True
        # ``stream_options`` is rejected by older Azure Chat Completions API
        # versions (e.g. our default ``2024-02-15-preview``), so we only ask
        # for usage on backends known to accept it. We lose usage stats on
        # Azure stream paths as a result — preferable to a 400.
        if self.config.type != ProviderType.AZURE:
            params["stream_options"] = {"include_usage": True}

        logger.debug(f"Streaming OpenAI Chat Completions with model: {self.model_name}")

        tool_acc: Dict[int, Dict[str, Any]] = {}
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0

        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0 if details else 0
                yield UsageEvent(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                )

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            fr = getattr(choice, "finish_reason", None)
            if fr:
                finish_reason = fr
            if delta is None:
                continue

            text = getattr(delta, "content", None)
            if text:
                content_parts.append(text)
                yield ContentDelta(text=text)

            reasoning_text = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
                or getattr(delta, "cot_summary", None)
            )
            if reasoning_text:
                yield ReasoningDelta(text=reasoning_text)

            tc_deltas = getattr(delta, "tool_calls", None) or []
            for tc in tc_deltas:
                idx = getattr(tc, "index", 0) or 0
                entry = tool_acc.setdefault(idx, {"id": "", "name": "", "arg_parts": []})
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    entry["id"] = tc_id
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                if name:
                    entry["name"] = name
                args_chunk = getattr(fn, "arguments", None) if fn else None
                if args_chunk:
                    entry["arg_parts"].append(args_chunk)
                yield ToolCallDelta(index=idx, id=tc_id, name=name, arguments_chunk=args_chunk)

        tool_calls: List[ToolCall] = []
        for idx in sorted(tool_acc.keys()):
            entry = tool_acc[idx]
            raw_args = "".join(entry["arg_parts"])
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse streamed tool arguments: {raw_args!r}")
                arguments = {}
            tool_calls.append(ToolCall(id=entry["id"], name=entry["name"], arguments=arguments))

        response = CompletionResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        yield DoneEvent(response=response)

    async def _complete_chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Non-streaming public entry: drives the streaming helper and
        collects the final ``DoneEvent`` response. Internally still streams so
        Substrate-style gateways don't time out on long completions."""
        final: Optional[CompletionResponse] = None
        async for event in self._complete_chat_stream(messages, tools, tool_choice, **kwargs):
            if isinstance(event, DoneEvent):
                final = event.response
        assert final is not None, "stream ended without DoneEvent"
        return final

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
            if msg.role == MessageRole.USER and msg.images:
                content_blocks: List[Dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "input_text", "text": msg.content})
                for img in msg.images:
                    if img.url:
                        image_url = img.url
                    elif img.base64_data:
                        image_url = f"data:{img.media_type};base64,{img.base64_data}"
                    else:
                        continue
                    content_blocks.append({"type": "input_image", "image_url": image_url})
                items.append({"role": "user", "content": content_blocks})
                continue
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
        input_items: Optional[List[Dict[str, Any]]] = None,
        instructions_override: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Non-streaming public entry — drives the streaming helper and returns
        the final response. Always streams internally to avoid gateway timeouts
        on long reasoning outputs."""
        final: Optional[CompletionResponse] = None
        async for event in self._complete_responses_stream(
            messages,
            tools,
            tool_choice,
            input_items=input_items,
            instructions_override=instructions_override,
            **kwargs,
        ):
            if isinstance(event, DoneEvent):
                final = event.response
        assert final is not None, "Responses stream ended without DoneEvent"
        return final

    def _build_responses_params(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        tool_choice: Optional[str],
        input_items: Optional[List[Dict[str, Any]]],
        instructions_override: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if input_items is None:
            instructions, input_items = self._convert_messages_to_responses_input(messages)
        else:
            instructions = instructions_override

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
        return params

    async def _complete_responses_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        tool_choice: Optional[str],
        input_items: Optional[List[Dict[str, Any]]] = None,
        instructions_override: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the Responses API and yield StreamEvents.

        Uses the SDK's typed event stream so we don't reimplement SSE byte
        parsing. The final ``DoneEvent`` carries the full assembled
        ``CompletionResponse`` including ``raw_output``, which the tool loop
        re-feeds (reasoning items must accompany ``function_call_output`` on
        the next turn).
        """
        params = self._build_responses_params(
            messages, tools, tool_choice, input_items, instructions_override, **kwargs
        )

        logger.debug(f"Streaming OpenAI Responses API with model: {self.model_name}")

        # Track function-call output_index → accumulator. The SDK emits
        # function_call_arguments.delta events keyed by output_index; the
        # corresponding `id` / `call_id` / `name` come from the final response.
        arg_acc: Dict[int, List[str]] = {}

        async with self._client.responses.stream(**params) as stream:
            async for event in stream:
                etype = getattr(event, "type", "") or ""
                if etype == "response.output_text.delta":
                    text = getattr(event, "delta", "") or ""
                    if text:
                        yield ContentDelta(text=text)
                elif etype in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    text = getattr(event, "delta", "") or ""
                    if text:
                        yield ReasoningDelta(text=text)
                elif etype == "response.function_call_arguments.delta":
                    idx = getattr(event, "output_index", 0) or 0
                    chunk = getattr(event, "delta", "") or ""
                    arg_acc.setdefault(idx, []).append(chunk)
                    yield ToolCallDelta(index=idx, arguments_chunk=chunk)

            final_response = await stream.get_final_response()

        output = list(getattr(final_response, "output", []) or [])
        content, tool_calls = self._parse_responses_output(output)

        usage = getattr(final_response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
        if usage:
            reasoning_details = getattr(usage, "output_tokens_details", None)
            reasoning_toks = (
                getattr(reasoning_details, "reasoning_tokens", 0) or 0 if reasoning_details else 0
            )
            yield UsageEvent(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_toks,
            )

        response = CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_output=output,
        )
        yield DoneEvent(response=response)

    # ------------------------------------------------------------------ Public API

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[CompletionResponse, AsyncIterator[StreamEvent]]:
        """Generate a completion using whichever OpenAI API surface fits the model.

        Pass ``stream=True`` to get back an ``AsyncIterator[StreamEvent]``
        instead of a single ``CompletionResponse``. The non-streaming default
        still uses SSE internally so Substrate / gateway endpoints don't time
        out on long completions.
        """
        stream = bool(kwargs.pop("stream", False))
        if self._use_responses_api():
            if stream:
                return self._complete_responses_stream(messages, tools, tool_choice, **kwargs)
            return await self._complete_responses(messages, tools, tool_choice, **kwargs)
        if stream:
            return self._complete_chat_stream(messages, tools, tool_choice, **kwargs)
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

        On the Responses path we maintain a running ``input`` list and re-feed
        the raw output items (including ``reasoning`` items) per OpenAI's
        requirement: any reasoning item returned alongside a tool call must be
        passed back with the corresponding ``function_call_output``, otherwise
        the next request 400s.
        """
        if self._use_responses_api():
            return await self._complete_with_tools_responses(
                messages, tools, tool_executor, max_iterations, **kwargs
            )
        return await self._complete_with_tools_chat(
            messages, tools, tool_executor, max_iterations, **kwargs
        )

    async def _complete_with_tools_chat(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        tool_executor: Callable[[ToolCall], Awaitable[ToolResult]],
        max_iterations: int,
        **kwargs: Any,
    ) -> tuple[CompletionResponse, List[Message]]:
        history = list(messages)
        iterations = 0
        response: Optional[CompletionResponse] = None

        while iterations < max_iterations:
            response = await self._complete_chat(
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
        assert response is not None
        return response, history

    async def _complete_with_tools_responses(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        tool_executor: Callable[[ToolCall], Awaitable[ToolResult]],
        max_iterations: int,
        **kwargs: Any,
    ) -> tuple[CompletionResponse, List[Message]]:
        history = list(messages)
        instructions, input_items = self._convert_messages_to_responses_input(history)

        iterations = 0
        response: Optional[CompletionResponse] = None

        while iterations < max_iterations:
            response = await self._complete_responses(
                messages=history,
                tools=tools,
                tool_choice="auto",
                input_items=input_items,
                instructions_override=instructions,
                **kwargs,
            )

            history.append(
                Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls if response.has_tool_calls else None,
                )
            )

            if not response.has_tool_calls:
                return response, history

            # Re-feed the full raw output (reasoning items + function_call items)
            # before the function_call_output items. OpenAI requires reasoning
            # items to accompany the tool outputs they triggered.
            if response.raw_output:
                input_items.extend(response.raw_output)

            for tool_call in response.tool_calls:
                logger.debug(f"Executing tool: {tool_call.name}")
                result = await tool_executor(tool_call)
                history.append(result.to_message())
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.id,
                        "output": result.to_message().content,
                    }
                )

            iterations += 1

        logger.warning(f"Max tool iterations ({max_iterations}) reached")
        assert response is not None
        return response, history


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or dict key — Responses items come back as pydantic
    models in the SDK but tests often hand us plain dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
