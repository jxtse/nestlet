"""Smoke tests for OpenAIProvider helpers (no network)."""

from __future__ import annotations

import json

import pytest

from inception.config.settings import ProviderConfig, ProviderType
from inception.provider.base import Message, ToolCall


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

    def _make(model: str) -> OpenAIProvider:
        cfg = ProviderConfig(type=ProviderType.OPENAI, model=model, api_key="sk-test")
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
