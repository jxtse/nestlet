"""Smoke tests for CodeValidator."""

from __future__ import annotations

from inception.tool.factory import CodeValidator


def test_validator_accepts_pure_function():
    validator = CodeValidator()
    result = validator.validate("def add(a: int, b: int) -> int:\n    return a + b\n")
    assert result.is_valid, result.errors


def test_validator_rejects_blocked_import():
    validator = CodeValidator()
    result = validator.validate("import socket\n\ndef f():\n    return socket\n")
    assert not result.is_valid
    assert any("Blocked module import" in e for e in result.errors)


def test_validator_rejects_eval_builtin():
    validator = CodeValidator()
    result = validator.validate("def f(x):\n    return eval(x)\n")
    assert not result.is_valid
    assert any("Blocked builtin" in e for e in result.errors)


def test_validator_rejects_syntax_error():
    validator = CodeValidator()
    result = validator.validate("def f(:\n")
    assert not result.is_valid
    assert any("Syntax error" in e for e in result.errors)


def test_validator_requires_function_definition():
    validator = CodeValidator()
    result = validator.validate("x = 1\n")
    assert not result.is_valid
    assert any("No function definition" in e for e in result.errors)


def test_validator_warns_on_multiple_functions():
    validator = CodeValidator()
    result = validator.validate("def a():\n    return 1\n\ndef b():\n    return 2\n")
    assert result.is_valid
    assert any("Multiple function definitions" in w for w in result.warnings)
