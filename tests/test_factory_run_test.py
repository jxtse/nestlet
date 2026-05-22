"""Smoke tests for ToolFactory._run_test sync/async compatibility."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from inception.tool.base import ParameterSpec, ParameterType, ToolResult, ToolSpec
from inception.tool.factory import GeneratedTool, ToolFactory
from inception.tool.registry import ToolRegistry


def _make_add_tool() -> GeneratedTool:
    def add(a: int, b: int) -> int:
        return a + b

    spec = ToolSpec(
        name="add",
        description="add two ints",
        parameters={
            "a": ParameterSpec(name="a", type=ParameterType.INTEGER, description="", required=True),
            "b": ParameterSpec(name="b", type=ParameterType.INTEGER, description="", required=True),
        },
        is_generated=True,
    )
    return GeneratedTool(spec, add)


def _make_factory() -> ToolFactory:
    return ToolFactory(ToolRegistry())


def test_run_test_synchronous_context():
    factory = _make_factory()
    tool = _make_add_tool()
    result = factory._run_test(tool, {"inputs": {"a": 2, "b": 3}, "expected": 5})
    assert result.passed, result.error


def test_run_test_expected_mismatch():
    factory = _make_factory()
    tool = _make_add_tool()
    result = factory._run_test(tool, {"inputs": {"a": 2, "b": 3}, "expected": 99})
    assert not result.passed
    assert "Expected 99" in (result.error or "")


@pytest.mark.asyncio
async def test_run_test_inside_running_loop():
    """_run_test must work even when invoked from inside a running event loop."""
    factory = _make_factory()
    tool = _make_add_tool()
    # No await needed: _run_test is sync but must safely bridge to its async
    # tool from within an active loop.
    result = factory._run_test(tool, {"inputs": {"a": 10, "b": 20}, "expected": 30})
    assert result.passed, result.error


@pytest.mark.asyncio
async def test_run_test_reports_tool_failure_in_loop():
    factory = _make_factory()

    class FailingTool(GeneratedTool):
        async def execute(self, **kwargs: Any) -> ToolResult:
            return ToolResult.fail(error="boom")

    spec = ToolSpec(name="fail", description="x", is_generated=True)
    failing = FailingTool(spec, lambda: None)
    result = factory._run_test(failing, {"inputs": {}})
    assert not result.passed
    assert result.error == "boom"


def test_run_test_uses_asyncio_run_when_no_loop():
    """When no loop is running, the call should not crash with 'attached to a different loop'."""
    factory = _make_factory()
    tool = _make_add_tool()
    # Call twice to make sure asyncio.run doesn't leave dangling state.
    for _ in range(2):
        r = factory._run_test(tool, {"inputs": {"a": 1, "b": 1}, "expected": 2})
        assert r.passed, r.error
    # Sanity check: no event loop policy leak.
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()
