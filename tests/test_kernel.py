"""Smoke tests for the Python kernel."""

from __future__ import annotations

import pytest

from inception.executor.kernel import PythonKernel


@pytest.mark.asyncio
async def test_kernel_basic_execution():
    kernel = PythonKernel()
    await kernel.initialize()

    result = await kernel.execute("x = 1 + 2")
    assert result.success, result.error
    assert "x" in kernel.list_variables()
    assert kernel.get_variable("x") == 3


@pytest.mark.asyncio
async def test_kernel_state_persists():
    kernel = PythonKernel()
    await kernel.initialize()

    await kernel.execute("counter = 10")
    result = await kernel.execute("counter = counter + 5\ncounter")
    assert result.success, result.error
    assert result.result == 15
    assert kernel.get_variable("counter") == 15


@pytest.mark.asyncio
async def test_kernel_captures_stdout():
    kernel = PythonKernel()
    await kernel.initialize()

    result = await kernel.execute("print('hello')")
    assert result.success
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_kernel_blocks_network_modules():
    kernel = PythonKernel()
    await kernel.initialize()

    result = await kernel.execute("import socket")
    assert not result.success
    assert "not allowed" in str(result.error)


@pytest.mark.asyncio
async def test_kernel_allowlist_rejects_others():
    kernel = PythonKernel(allowed_modules={"math"})
    await kernel.initialize()

    result = await kernel.execute("import json")
    assert not result.success
    assert "not in the allowed list" in str(result.error)


@pytest.mark.asyncio
async def test_kernel_reports_error_type():
    kernel = PythonKernel()
    await kernel.initialize()

    result = await kernel.execute("1 / 0")
    assert not result.success
    assert result.error_type == "ZeroDivisionError"


@pytest.mark.asyncio
async def test_kernel_reset_clears_state():
    kernel = PythonKernel()
    await kernel.initialize()
    await kernel.execute("foo = 'bar'")
    assert kernel.get_variable("foo") == "bar"

    kernel.reset()
    assert kernel.get_variable("foo") is None
