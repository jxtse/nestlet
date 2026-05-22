"""Smoke tests for AnthropicProvider helpers (no network)."""

from __future__ import annotations

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
    """Build AnthropicProvider without touching the network / proxy env."""
    for var in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        monkeypatch.delenv(var, raising=False)

    from inception.provider.anthropic import AnthropicProvider

    def _make(model: str = "claude-3-5-sonnet-latest", **overrides):
        cfg = ProviderConfig(
            type=ProviderType.ANTHROPIC, model=model, api_key="sk-ant-test", **overrides
        )
        return AnthropicProvider(cfg)

    return _make


# ---------------------------------------------------------- message conversion


def test_convert_messages_plain_text(make_provider):
    p = make_provider()
    system, msgs = p._convert_messages([Message.user("hello")])
    assert system is None
    assert msgs == [{"role": "user", "content": "hello"}]


def test_convert_messages_pulls_system(make_provider):
    p = make_provider()
    system, msgs = p._convert_messages([Message.system("be terse"), Message.user("hi")])
    assert system == "be terse"
    assert msgs == [{"role": "user", "content": "hi"}]


def test_convert_messages_user_with_url_image(make_provider):
    p = make_provider()
    msg = Message.user(
        "what is this?",
        images=[ImageContent(url="https://example.com/cat.png")],
    )
    _, msgs = p._convert_messages([msg])
    assert msgs == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/cat.png"},
                },
            ],
        }
    ]


def test_convert_messages_user_with_base64_image(make_provider):
    p = make_provider()
    msg = Message.user(
        "describe",
        images=[ImageContent(base64_data="AAAA", media_type="image/jpeg")],
    )
    _, msgs = p._convert_messages([msg])
    assert msgs[0]["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "AAAA",
        },
    }


def test_convert_messages_user_with_image_no_text(make_provider):
    p = make_provider()
    msg = Message.user("", images=[ImageContent(url="https://example.com/x.png")])
    _, msgs = p._convert_messages([msg])
    # no leading text block, just the image
    assert msgs[0]["content"] == [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
    ]


def test_convert_messages_assistant_tool_call(make_provider):
    p = make_provider()
    tc = ToolCall(id="t_1", name="search", arguments={"q": "x"})
    _, msgs = p._convert_messages([Message.assistant(content="ok", tool_calls=[tc])])
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"][0] == {"type": "text", "text": "ok"}
    assert msgs[0]["content"][1] == {
        "type": "tool_use",
        "id": "t_1",
        "name": "search",
        "input": {"q": "x"},
    }


def test_convert_messages_tool_result(make_provider):
    p = make_provider()
    tool_msg = Message.tool(content="found it", tool_call_id="t_1", name="search")
    _, msgs = p._convert_messages([tool_msg])
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t_1",
        "content": "found it",
    }


# ------------------------------------------------------------------ streaming


class _AnthEvent:
    def __init__(self, type, **fields):
        self.type = type
        for k, v in fields.items():
            setattr(self, k, v)


class _AnthBlock:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class _AnthUsage:
    def __init__(self, p, c):
        self.input_tokens = p
        self.output_tokens = c


class _AnthFinalMessage:
    def __init__(self, content, usage, stop_reason="end_turn"):
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason


class _AnthStream:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    def __aiter__(self):
        events = self._events

        async def _aiter():
            for ev in events:
                yield ev

        return _aiter()

    async def get_final_message(self):
        return self._final


class _AnthStreamCtx:
    def __init__(self, events, final):
        self._stream = _AnthStream(events, final)

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_anthropic_stream_yields_content_and_tool_deltas(make_provider, monkeypatch):
    p = make_provider()

    events = [
        _AnthEvent(
            "content_block_start",
            index=0,
            content_block=_AnthBlock(type="text"),
        ),
        _AnthEvent(
            "content_block_delta",
            index=0,
            delta=_AnthBlock(type="text_delta", text="Hel"),
        ),
        _AnthEvent(
            "content_block_delta",
            index=0,
            delta=_AnthBlock(type="text_delta", text="lo"),
        ),
        _AnthEvent(
            "content_block_start",
            index=1,
            content_block=_AnthBlock(type="tool_use", id="t_1", name="search"),
        ),
        _AnthEvent(
            "content_block_delta",
            index=1,
            delta=_AnthBlock(type="input_json_delta", partial_json='{"q":"cats"}'),
        ),
    ]
    final_blocks = [
        _AnthBlock(type="text", text="Hello"),
        _AnthBlock(type="tool_use", id="t_1", name="search", input={"q": "cats"}),
    ]
    final = _AnthFinalMessage(final_blocks, _AnthUsage(10, 5), stop_reason="tool_use")

    def fake_stream(**params):
        return _AnthStreamCtx(events, final)

    monkeypatch.setattr(p._client.messages, "stream", fake_stream)

    iterator = await p.complete(
        messages=[Message.user("hi")],
        tools=[ToolDefinition(name="search", description="", parameters={})],
        stream=True,
    )
    collected = []
    async for ev in iterator:
        collected.append(ev)

    content_text = "".join(e.text for e in collected if isinstance(e, ContentDelta))
    tc_starts = [e for e in collected if isinstance(e, ToolCallDelta) and e.id == "t_1"]
    tc_chunks = [e for e in collected if isinstance(e, ToolCallDelta) and e.arguments_chunk]
    usage_evs = [e for e in collected if isinstance(e, UsageEvent)]
    done = [e for e in collected if isinstance(e, DoneEvent)]

    assert content_text == "Hello"
    assert tc_starts and tc_starts[0].name == "search"
    assert "".join(c.arguments_chunk for c in tc_chunks) == '{"q":"cats"}'
    assert usage_evs and usage_evs[0].prompt_tokens == 10
    assert done
    final_resp = done[0].response
    assert final_resp.content == "Hello"
    assert final_resp.tool_calls[0].arguments == {"q": "cats"}


@pytest.mark.asyncio
async def test_anthropic_non_stream_uses_stream_internally(make_provider, monkeypatch):
    p = make_provider()

    final_blocks = [_AnthBlock(type="text", text="ok")]
    final = _AnthFinalMessage(final_blocks, _AnthUsage(1, 1))
    saw_calls = []

    def fake_stream(**params):
        saw_calls.append(params)
        return _AnthStreamCtx([], final)

    monkeypatch.setattr(p._client.messages, "stream", fake_stream)

    response = await p.complete(messages=[Message.user("hi")])
    assert len(saw_calls) == 1
    assert response.content == "ok"


# ------------------------------------------------------------- tool loop smoke


@pytest.mark.asyncio
async def test_anthropic_tool_loop_runs_to_completion(make_provider, monkeypatch):
    p = make_provider()

    # First turn: assistant requests a tool call. Second turn: assistant replies.
    turn_1_blocks = [
        _AnthBlock(type="tool_use", id="t_1", name="lookup", input={"q": "x"}),
    ]
    turn_1_final = _AnthFinalMessage(turn_1_blocks, _AnthUsage(5, 2), stop_reason="tool_use")

    turn_2_blocks = [_AnthBlock(type="text", text="done")]
    turn_2_final = _AnthFinalMessage(turn_2_blocks, _AnthUsage(7, 1), stop_reason="end_turn")

    finals = [turn_1_final, turn_2_final]
    call_count = {"n": 0}

    def fake_stream(**params):
        idx = call_count["n"]
        call_count["n"] += 1
        return _AnthStreamCtx([], finals[idx])

    monkeypatch.setattr(p._client.messages, "stream", fake_stream)

    async def tool_executor(tc: ToolCall) -> ToolResult:
        return ToolResult(tool_call_id=tc.id, name=tc.name, result="x-result")

    tools = [ToolDefinition(name="lookup", description="", parameters={})]
    response, history = await p.complete_with_tools(
        messages=[Message.user("find x")],
        tools=tools,
        tool_executor=tool_executor,
    )

    assert response.content == "done"
    assert call_count["n"] == 2
    # history: original user msg + assistant(tool_use) + tool result + assistant(text)
    assert len(history) == 4
