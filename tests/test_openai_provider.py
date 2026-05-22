"""Smoke tests for OpenAIProvider helpers (no network)."""

from __future__ import annotations

import json

import pytest

from inception.config.settings import ProviderConfig, ProviderType
from inception.provider.base import (
    ContentDelta,
    DoneEvent,
    ImageContent,
    Message,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    ToolResult,
    UsageEvent,
)


@pytest.fixture
def make_provider(monkeypatch):
    """Build OpenAIProvider without touching the network / proxy env."""
    # Strip proxy env so httpx doesn't try to import socks during client init.
    for var in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        monkeypatch.delenv(var, raising=False)

    from inception.provider.openai import OpenAIProvider

    def _make(model: str, **overrides) -> OpenAIProvider:
        cfg = ProviderConfig(type=ProviderType.OPENAI, model=model, api_key="sk-test", **overrides)
        return OpenAIProvider(cfg)

    return _make


def test_max_tokens_param_for_legacy_models(make_provider):
    p = make_provider("gpt-4o-mini")
    assert p._max_tokens_param_name() == "max_tokens"


def test_max_tokens_param_for_gpt5(make_provider):
    p = make_provider("gpt-5")
    assert p._max_tokens_param_name() == "max_completion_tokens"


@pytest.mark.parametrize("model", ["o1-preview", "o3-mini", "o4-pro"])
def test_max_tokens_param_for_reasoning_models(make_provider, model):
    p = make_provider(model)
    assert p._max_tokens_param_name() == "max_completion_tokens"


def test_convert_messages_plain(make_provider):
    p = make_provider("gpt-4o-mini")
    out = p._convert_messages([Message.user("hello")])
    assert out == [{"role": "user", "content": "hello"}]


def test_convert_messages_assistant_with_tool_calls(make_provider):
    p = make_provider("gpt-4o-mini")
    tc = ToolCall(id="call_1", name="search", arguments={"q": "cats"})
    out = p._convert_messages([Message.assistant(content="ok", tool_calls=[tc])])
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"][0]["id"] == "call_1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "search"
    # arguments are serialized as JSON string per OpenAI spec
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"q": "cats"}


# ----------------------------------------------------------------- Responses API


def test_use_responses_api_for_gpt5_5(make_provider):
    p = make_provider("gpt-5.5")
    assert p._use_responses_api() is True


def test_use_responses_api_for_gpt4o_is_false(make_provider):
    p = make_provider("gpt-4o-mini")
    assert p._use_responses_api() is False


@pytest.mark.parametrize("model", ["o1-preview", "o3-mini", "o4-pro"])
def test_use_responses_api_for_reasoning_models(make_provider, model):
    p = make_provider(model)
    assert p._use_responses_api() is True


def test_explicit_chat_mode_disables_responses(make_provider):
    p = make_provider("gpt-5.5", api_mode="chat")
    assert p._use_responses_api() is False


def test_token_budget_kwargs_for_responses_path(make_provider):
    p = make_provider("gpt-5.5")
    assert p._token_budget_kwargs(2048) == {"max_output_tokens": 2048}


def test_token_budget_kwargs_for_chat_path(make_provider):
    p = make_provider("gpt-4o-mini")
    assert p._token_budget_kwargs(2048) == {"max_tokens": 2048}


def test_convert_messages_to_responses_input_pulls_system(make_provider):
    p = make_provider("gpt-5.5")
    instructions, items = p._convert_messages_to_responses_input(
        [Message.system("be terse"), Message.user("hi")]
    )
    assert instructions == "be terse"
    assert items == [{"role": "user", "content": "hi"}]


def test_convert_messages_to_responses_input_tool_round_trip(make_provider):
    p = make_provider("gpt-5.5")
    tc = ToolCall(id="call_42", name="lookup", arguments={"q": "x"})
    msgs = [
        Message.user("find x"),
        Message.assistant(content="", tool_calls=[tc]),
        Message.tool(content="result-x", tool_call_id="call_42", name="lookup"),
    ]
    instructions, items = p._convert_messages_to_responses_input(msgs)
    assert instructions is None
    # user, function_call, function_call_output  (assistant text was empty so skipped)
    assert items[0] == {"role": "user", "content": "find x"}
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_42"
    assert items[1]["name"] == "lookup"
    assert json.loads(items[1]["arguments"]) == {"q": "x"}
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_42",
        "output": "result-x",
    }


def test_build_reasoning_param_includes_summary(make_provider):
    p = make_provider("gpt-5.5", reasoning_effort="high", reasoning_summary="auto")
    assert p._build_reasoning_param() == {"effort": "high", "summary": "auto"}


def test_build_reasoning_param_none_when_unset(make_provider):
    p = make_provider("gpt-5.5")
    assert p._build_reasoning_param() is None


def test_parse_responses_output_text_and_tool_calls(make_provider):
    p = make_provider("gpt-5.5")
    output = [
        {"type": "reasoning", "id": "r_1"},  # ignored
        {
            "type": "message",
            "content": [
                {"type": "output_text", "text": "Hello, "},
                {"type": "output_text", "text": "world."},
            ],
        },
        {
            "type": "function_call",
            "call_id": "call_77",
            "name": "search",
            "arguments": json.dumps({"q": "cats"}),
        },
    ]
    text, tcs = p._parse_responses_output(output)
    assert text == "Hello, world."
    assert len(tcs) == 1
    assert tcs[0].id == "call_77"
    assert tcs[0].name == "search"
    assert tcs[0].arguments == {"q": "cats"}


def test_tool_definition_responses_shape_is_flat():
    td = ToolDefinition(
        name="search",
        description="search the web",
        parameters={"type": "object", "properties": {}},
    )
    flat = td.to_responses_dict()
    assert flat["type"] == "function"
    assert flat["name"] == "search"
    assert flat["description"] == "search the web"
    assert flat["parameters"] == {"type": "object", "properties": {}}
    # And the legacy nested form still works for the Chat path.
    nested = td.to_dict()
    assert nested["function"]["name"] == "search"


# ----------------------------------------------------------- image translation


def test_responses_input_translates_image_url(make_provider):
    p = make_provider("gpt-5.5")
    msg = Message.user(
        "what is in this image?",
        images=[ImageContent(url="https://example.com/cat.png")],
    )
    _, items = p._convert_messages_to_responses_input([msg])
    assert items == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in this image?"},
                {"type": "input_image", "image_url": "https://example.com/cat.png"},
            ],
        }
    ]


def test_responses_input_translates_base64_image(make_provider):
    p = make_provider("gpt-5.5")
    msg = Message.user(
        "describe",
        images=[ImageContent(base64_data="AAAA", media_type="image/jpeg")],
    )
    _, items = p._convert_messages_to_responses_input([msg])
    assert items[0]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,AAAA",
    }


# ----------------------------------------------------------- tool loop re-feed


@pytest.mark.asyncio
async def test_responses_tool_loop_refeeds_raw_output(make_provider, monkeypatch):
    """Verify reasoning items + function_call items from prior turn are re-fed."""
    p = make_provider("gpt-5.5", reasoning_effort="medium")

    reasoning_item = {"type": "reasoning", "id": "rs_1", "summary": []}
    function_call_item = {
        "type": "function_call",
        "call_id": "call_a",
        "id": "fc_1",
        "name": "lookup",
        "arguments": json.dumps({"q": "x"}),
    }
    final_message_item = {
        "type": "message",
        "content": [{"type": "output_text", "text": "done"}],
    }

    calls: list[dict] = []

    class _StubUsage:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

    class _StubFinalResponse:
        def __init__(self, output):
            self.output = output
            self.usage = _StubUsage()

    class _StubStream:
        def __init__(self, final_output):
            self._final = _StubFinalResponse(final_output)

        def __aiter__(self):
            async def _aiter():
                if False:
                    yield  # never yields events — final assembled via get_final_response

            return _aiter()

        async def get_final_response(self):
            return self._final

    class _StubStreamCtx:
        def __init__(self, final_output):
            self._stream = _StubStream(final_output)

        async def __aenter__(self):
            return self._stream

        async def __aexit__(self, exc_type, exc, tb):
            return False

    outputs_to_return = [
        [reasoning_item, function_call_item],
        [final_message_item],
    ]

    def fake_stream(**params):
        calls.append(params)
        return _StubStreamCtx(outputs_to_return.pop(0))

    monkeypatch.setattr(p._client.responses, "stream", fake_stream)

    async def tool_executor(tc: ToolCall) -> ToolResult:
        return ToolResult(tool_call_id=tc.id, name=tc.name, result="x-result")

    tools = [ToolDefinition(name="lookup", description="", parameters={})]
    response, history = await p.complete_with_tools(
        messages=[Message.user("find x")],
        tools=tools,
        tool_executor=tool_executor,
    )

    assert response.content == "done"
    assert len(calls) == 2

    second_input = calls[1]["input"]
    # Original user message
    assert second_input[0] == {"role": "user", "content": "find x"}
    # Reasoning item from first response must be present before the tool output
    assert reasoning_item in second_input
    assert function_call_item in second_input
    # function_call_output for the tool we executed
    assert {
        "type": "function_call_output",
        "call_id": "call_a",
        "output": "x-result",
    } in second_input
    # Reasoning must come before its function_call_output
    reasoning_idx = second_input.index(reasoning_item)
    output_idx = next(
        i
        for i, it in enumerate(second_input)
        if isinstance(it, dict) and it.get("type") == "function_call_output"
    )
    assert reasoning_idx < output_idx


# ----------------------------------------------------------------- Chat streaming


class _ChatDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content
        self.reasoning = None
        self.cot_summary = None


class _ChatChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _ChatChunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


class _ChatUsage:
    def __init__(self, p, c, r=0):
        self.prompt_tokens = p
        self.completion_tokens = c

        class _Det:
            reasoning_tokens = r

        self.completion_tokens_details = _Det()


class _ChatToolDelta:
    def __init__(self, index, id=None, name=None, args=None):
        self.index = index
        self.id = id

        class _Fn:
            pass

        fn = _Fn()
        fn.name = name
        fn.arguments = args
        self.function = fn


async def _async_iter(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_chat_stream_yields_content_and_tool_deltas(make_provider, monkeypatch):
    p = make_provider("gpt-4o-mini")

    chunks = [
        _ChatChunk(choices=[_ChatChoice(_ChatDelta(content="Hel"))]),
        _ChatChunk(choices=[_ChatChoice(_ChatDelta(content="lo"))]),
        _ChatChunk(
            choices=[
                _ChatChoice(
                    _ChatDelta(
                        tool_calls=[_ChatToolDelta(0, id="call_a", name="search", args='{"q":')]
                    )
                )
            ]
        ),
        _ChatChunk(
            choices=[
                _ChatChoice(
                    _ChatDelta(tool_calls=[_ChatToolDelta(0, args='"cats"}')]),
                    finish_reason="tool_calls",
                )
            ]
        ),
        _ChatChunk(usage=_ChatUsage(10, 5, 0)),
    ]

    async def fake_create(**params):
        assert params["stream"] is True
        return _async_iter(chunks)

    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)

    events = []
    iterator = await p.complete(
        messages=[Message.user("hi")],
        tools=[ToolDefinition(name="search", description="", parameters={})],
        stream=True,
    )
    async for ev in iterator:
        events.append(ev)

    content_events = [e for e in events if isinstance(e, ContentDelta)]
    tc_events = [e for e in events if isinstance(e, ToolCallDelta)]
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    done_events = [e for e in events if isinstance(e, DoneEvent)]

    assert [e.text for e in content_events] == ["Hel", "lo"]
    assert len(tc_events) == 2
    assert usage_events and usage_events[0].prompt_tokens == 10
    assert done_events
    final = done_events[0].response
    assert final.content == "Hello"
    assert final.has_tool_calls
    assert final.tool_calls[0].name == "search"
    assert final.tool_calls[0].arguments == {"q": "cats"}


@pytest.mark.asyncio
async def test_chat_non_stream_uses_stream_internally(make_provider, monkeypatch):
    """Confirm the default complete() still streams under the hood (Substrate-safe)."""
    p = make_provider("gpt-4o-mini")

    chunks = [
        _ChatChunk(choices=[_ChatChoice(_ChatDelta(content="ok"), finish_reason="stop")]),
        _ChatChunk(usage=_ChatUsage(1, 1)),
    ]
    saw_stream_true = []

    async def fake_create(**params):
        saw_stream_true.append(params.get("stream") is True)
        return _async_iter(chunks)

    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)

    response = await p.complete(messages=[Message.user("hi")])
    assert saw_stream_true == [True]
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_azure_chat_stream_omits_stream_options(monkeypatch):
    """Older Azure API versions reject stream_options — make sure we don't send it."""
    for var in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        monkeypatch.delenv(var, raising=False)

    from inception.provider.openai import OpenAIProvider

    cfg = ProviderConfig(
        type=ProviderType.AZURE,
        model="gpt-4o-mini",
        api_key="sk-test",
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="dep",
    )
    p = OpenAIProvider(cfg)

    chunks = [
        _ChatChunk(choices=[_ChatChoice(_ChatDelta(content="ok"), finish_reason="stop")]),
    ]
    captured = {}

    async def fake_create(**params):
        captured.update(params)
        return _async_iter(chunks)

    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)

    response = await p.complete(messages=[Message.user("hi")])
    assert captured["stream"] is True
    assert "stream_options" not in captured
    assert response.content == "ok"


# ----------------------------------------------------------- Responses streaming


class _RespEvent:
    def __init__(self, type, **fields):
        self.type = type
        for k, v in fields.items():
            setattr(self, k, v)


class _RespUsage:
    def __init__(self, p, c, t, reasoning=0):
        self.input_tokens = p
        self.output_tokens = c
        self.total_tokens = t

        class _Det:
            reasoning_tokens = reasoning

        self.output_tokens_details = _Det()


class _RespFinal:
    def __init__(self, output, usage):
        self.output = output
        self.usage = usage


class _RespStream:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    def __aiter__(self):
        events = self._events

        async def _aiter():
            for ev in events:
                yield ev

        return _aiter()

    async def get_final_response(self):
        return self._final


class _RespStreamCtx:
    def __init__(self, events, final):
        self._stream = _RespStream(events, final)

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_responses_stream_yields_text_reasoning_and_tool_deltas(make_provider, monkeypatch):
    p = make_provider("gpt-5.5", reasoning_effort="medium")

    events = [
        _RespEvent("response.reasoning_summary_text.delta", delta="thinking..."),
        _RespEvent("response.output_text.delta", delta="Hello "),
        _RespEvent("response.output_text.delta", delta="world"),
        _RespEvent("response.function_call_arguments.delta", output_index=1, delta='{"q":'),
        _RespEvent("response.function_call_arguments.delta", output_index=1, delta='"cats"}'),
    ]
    final_output = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Hello world"}],
        },
        {
            "type": "function_call",
            "call_id": "call_z",
            "name": "search",
            "arguments": json.dumps({"q": "cats"}),
        },
    ]
    final = _RespFinal(final_output, _RespUsage(10, 5, 15, reasoning=3))

    def fake_stream(**params):
        return _RespStreamCtx(events, final)

    monkeypatch.setattr(p._client.responses, "stream", fake_stream)

    iterator = await p.complete(
        messages=[Message.user("hi")],
        tools=[ToolDefinition(name="search", description="", parameters={})],
        stream=True,
    )
    collected = []
    async for ev in iterator:
        collected.append(ev)

    content_text = "".join(e.text for e in collected if isinstance(e, ContentDelta))
    reasoning_text = "".join(
        e.text for e in collected if hasattr(e, "type") and e.type == "reasoning"
    )
    tc_chunks = [e for e in collected if isinstance(e, ToolCallDelta) and e.arguments_chunk]
    usage_evs = [e for e in collected if isinstance(e, UsageEvent)]
    done = [e for e in collected if isinstance(e, DoneEvent)]

    assert content_text == "Hello world"
    assert reasoning_text == "thinking..."
    assert "".join(c.arguments_chunk for c in tc_chunks) == '{"q":"cats"}'
    assert usage_evs and usage_evs[0].reasoning_tokens == 3
    assert done
    final_resp = done[0].response
    assert final_resp.content == "Hello world"
    assert final_resp.tool_calls[0].arguments == {"q": "cats"}
    # raw_output must be preserved for the tool loop re-feed
    assert final_resp.raw_output == final_output


@pytest.mark.asyncio
async def test_responses_non_stream_uses_stream_internally(make_provider, monkeypatch):
    p = make_provider("gpt-5.5")

    final_output = [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ]
    final = _RespFinal(final_output, _RespUsage(1, 1, 2))
    saw_calls = []

    def fake_stream(**params):
        saw_calls.append(params)
        return _RespStreamCtx([], final)

    monkeypatch.setattr(p._client.responses, "stream", fake_stream)

    response = await p.complete(messages=[Message.user("hi")])
    assert len(saw_calls) == 1
    assert response.content == "ok"
    assert response.raw_output == final_output
