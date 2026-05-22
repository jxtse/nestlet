"""Smoke tests for Settings configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from inception.config.settings import ProviderConfig, ProviderType, Settings


def test_defaults_use_gpt_5_5():
    settings = Settings()
    assert settings.provider.model == "gpt-5.5"
    assert settings.provider.type == ProviderType.OPENAI
    assert settings.provider.api_mode == "auto"
    # auto-mode resolves to responses for gpt-5*
    assert settings.provider.resolved_api_mode() == "responses"


def test_from_yaml_openrouter():
    config = {
        "provider": {
            "type": "openrouter",
            "model": "openai/gpt-4o-mini",
            "api_key": "sk-test",
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        path = Path(f.name)

    try:
        settings = Settings.from_yaml(path)
        assert settings.provider.type == ProviderType.OPENROUTER
        assert settings.provider.model == "openai/gpt-4o-mini"
        assert settings.provider.api_key == "sk-test"
        # OpenRouter routes via Chat Completions (no Responses API)
        assert settings.provider.resolved_api_mode() == "chat"
    finally:
        path.unlink()


def test_from_yaml_anthropic():
    config = {"provider": {"type": "anthropic", "model": "claude-3-5-sonnet-latest"}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        path = Path(f.name)

    try:
        settings = Settings.from_yaml(path)
        assert settings.provider.type == ProviderType.ANTHROPIC
        assert settings.provider.model == "claude-3-5-sonnet-latest"
    finally:
        path.unlink()


def test_provider_type_enum_has_openrouter():
    assert ProviderType.OPENROUTER.value == "openrouter"


def test_reasoning_effort_validates():
    # valid
    cfg = ProviderConfig(type=ProviderType.OPENAI, reasoning_effort="medium")
    assert cfg.reasoning_effort == "medium"

    # invalid value rejected by pydantic
    with pytest.raises(ValidationError):
        ProviderConfig(type=ProviderType.OPENAI, reasoning_effort="ludicrous")


def test_api_mode_explicit_overrides_auto():
    cfg = ProviderConfig(type=ProviderType.OPENAI, model="gpt-4o-mini", api_mode="responses")
    assert cfg.resolved_api_mode() == "responses"

    cfg = ProviderConfig(type=ProviderType.OPENAI, model="gpt-5.5", api_mode="chat")
    assert cfg.resolved_api_mode() == "chat"


def test_settings_round_trip_yaml():
    s = Settings()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        path = Path(f.name)
    try:
        s.save_yaml(path)
        s2 = Settings.from_yaml(path)
        assert s2.provider.model == "gpt-5.5"
        assert s2.provider.api_mode == "auto"
    finally:
        path.unlink()
