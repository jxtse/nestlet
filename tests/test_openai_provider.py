"""Smoke tests for OpenAIProvider helpers (no network)."""

from __future__ import annotations

import json

import pytest

from inception.config.settings import ProviderConfig, ProviderType
from inception.provider.base import ImageContent, Message, ToolCall, ToolDefinition, ToolResult


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

    class _StubResponse:
        def __init__(self, output):
            self.output = output
            self.usage = _StubUsage()

    responses_to_return = [
        _StubResponse([reasoning_item, function_call_item]),
        _StubResponse([final_message_item]),
    ]

    async def fake_create(**params):
        calls.append(params)
        return responses_to_return.pop(0)

    monkeypatch.setattr(p._client.responses, "create", fake_create)

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
