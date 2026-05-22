"""Smoke tests for Action data class round-trips."""

from __future__ import annotations

from inception.agent.base import Action, ActionResult, ActionType


def test_tool_call_action():
    action = Action.tool_call("search", {"q": "hello"})
    assert action.type == ActionType.TOOL_CALL
    assert action.tool_name == "search"
    assert action.tool_args == {"q": "hello"}


def test_code_exec_action():
    action = Action.code_exec("print(1)")
    assert action.type == ActionType.CODE_EXEC
    assert action.code == "print(1)"


def test_respond_action():
    action = Action.respond("hi there")
    assert action.type == ActionType.RESPOND
    assert action.response == "hi there"


def test_create_tool_action_round_trip():
    action = Action.create_tool(
        name="reverse",
        description="reverse a string",
        code="def reverse(s: str) -> str:\n    return s[::-1]",
    )
    assert action.type == ActionType.CREATE_TOOL
    assert action.metadata["tool_name"] == "reverse"
    assert action.metadata["tool_description"] == "reverse a string"
    assert "def reverse" in action.metadata["tool_code"]


def test_action_result_ok_and_fail():
    ok = ActionResult.ok(result=42)
    assert ok.success is True
    assert ok.result == 42

    bad = ActionResult.fail("boom")
    assert bad.success is False
    assert bad.error == "boom"
