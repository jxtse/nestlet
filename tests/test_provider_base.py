"""Smoke tests for the pydantic provider data models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from inception.provider.base import (
    CompletionResponse,
    ImageContent,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def test_message_user_factory():
    m = Message.user("hello")
    assert m.role == MessageRole.USER
    assert m.content == "hello"
    assert m.to_dict() == {"role": "user", "content": "hello"}


def test_message_with_image_url_to_dict():
    m = Message.user_with_image_url("look", "https://example.com/x.png")
    out = m.to_dict()
    assert out["role"] == "user"
    assert isinstance(out["content"], list)
    assert out["content"][0] == {"type": "text", "text": "look"}
    assert out["content"][1]["type"] == "image_url"


def test_tool_call_to_dict_shape():
    tc = ToolCall(id="call_x", name="run", arguments={"a": 1})
    d = tc.to_dict()
    assert d["id"] == "call_x"
    assert d["type"] == "function"
    assert d["function"]["name"] == "run"
    assert json.loads(d["function"]["arguments"]) == {"a": 1}


def test_tool_result_to_message_success():
    r = ToolResult(tool_call_id="call_1", name="t", result=42)
    m = r.to_message()
    assert m.role == MessageRole.TOOL
    assert m.tool_call_id == "call_1"
    assert m.content == "42"


def test_tool_result_to_message_failure():
    r = ToolResult(tool_call_id="call_1", name="t", success=False, error="boom")
    m = r.to_message()
    assert "Error: boom" in m.content


def test_tool_definition_chat_and_responses_shapes():
    td = ToolDefinition(
        name="search",
        description="search the web",
        parameters={"type": "object", "properties": {}},
    )
    nested = td.to_dict()
    assert nested["type"] == "function"
    assert nested["function"]["name"] == "search"

    flat = td.to_responses_dict()
    assert flat["type"] == "function"
    assert flat["name"] == "search"
    assert "function" not in flat


def test_completion_response_defaults():
    r = CompletionResponse(content="hi")
    assert r.tool_calls == []
    assert r.has_tool_calls is False
    assert r.raw_output is None


def test_completion_response_with_raw_output_for_responses_path():
    raw = [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]
    r = CompletionResponse(content="hi", raw_output=raw)
    assert r.raw_output == raw


def test_image_content_requires_url_or_b64():
    img = ImageContent()  # constructible — to_dict is what enforces
    with pytest.raises(ValueError):
        img.to_dict()


def test_message_rejects_bad_role():
    # pydantic v2 raises ValidationError on bad enum values
    with pytest.raises(ValidationError):
        Message(role="banana", content="x")
