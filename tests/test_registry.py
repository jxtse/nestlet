"""Smoke tests for ToolRegistry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inception.tool.base import ParameterSpec, ParameterType, Tool, ToolResult, ToolSpec
from inception.tool.registry import ToolRegistry


class _DummyTool(Tool):
    def __init__(self, name: str = "dummy", category: str = "test"):
        self._spec = ToolSpec(
            name=name,
            description=f"dummy tool {name}",
            parameters={
                "x": ParameterSpec(
                    name="x",
                    type=ParameterType.INTEGER,
                    description="value",
                    required=True,
                )
            },
            category=category,
            tags=["smoke", category],
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.ok(result=kwargs.get("x"))


def test_register_and_lookup():
    reg = ToolRegistry()
    reg.register(_DummyTool("alpha"))
    assert reg.has("alpha")
    assert reg.get("alpha") is not None
    assert reg.get_spec("alpha").name == "alpha"
    assert "alpha" in reg.list_all()
    assert "alpha" in reg.list_by_category("test")
    assert "alpha" in reg.list_by_tag("smoke")


def test_duplicate_register_raises():
    reg = ToolRegistry()
    reg.register(_DummyTool("alpha"))
    with pytest.raises(ValueError):
        reg.register(_DummyTool("alpha"))


def test_unregister_removes_from_indexes():
    reg = ToolRegistry()
    reg.register(_DummyTool("alpha"))
    assert reg.unregister("alpha") is True
    assert not reg.has("alpha")
    assert "alpha" not in reg.list_by_category("test")
    assert "alpha" not in reg.list_by_tag("smoke")
    assert reg.unregister("missing") is False


def test_record_usage_stats():
    reg = ToolRegistry()
    reg.register(_DummyTool("alpha"))
    reg.record_usage("alpha")
    reg.record_usage("alpha")
    stats = reg.get_usage_stats()
    assert stats["alpha"] == 2


def test_save_skips_builtin_tools(tmp_path: Path):
    storage = tmp_path / "tools.json"
    reg = ToolRegistry(storage_path=storage)
    reg.register(_DummyTool("alpha"), is_builtin=True)
    reg.save()
    # Builtin tools without source_code are not persisted.
    data = json.loads(storage.read_text())
    assert data == {}


def test_load_returns_zero_when_no_file(tmp_path: Path):
    reg = ToolRegistry(storage_path=tmp_path / "absent.json")
    assert reg.load() == 0
