"""Smoke tests for Settings configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from inception.config.settings import ProviderType, Settings


def test_defaults_use_gpt_4o_mini():
    settings = Settings()
    assert settings.provider.model == "gpt-4o-mini"
    assert settings.provider.type == ProviderType.OPENAI


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
