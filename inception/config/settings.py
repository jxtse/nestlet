"""
Configuration settings for Inception.

Manages all configuration including LLM providers, execution settings,
and security policies.

The provider/execution/memory/web_search models use pydantic v2 BaseModel for
runtime validation. Hand-rolled ``from_dict`` / ``to_dict`` / ``from_yaml`` /
``save_yaml`` are preserved as thin wrappers over pydantic so call sites in the
agent / settings loaders do not change.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    OPENROUTER = "openrouter"


class ExecutionMode(str, Enum):
    """Code execution mode."""

    SANDBOX = "sandbox"  # Restricted execution
    TRUSTED = "trusted"  # Full access (for trusted environments)


# Token-budget kwarg name resolved at the provider level — the config keeps a
# single canonical ``max_tokens``. Reasoning effort levels mirror what OpenAI's
# Responses API accepts for ``reasoning.effort``.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
ReasoningSummary = Literal["auto", "concise", "detailed"]
ApiMode = Literal["auto", "chat", "responses"]

DEFAULT_MODEL = "gpt-5.5"


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    model_config = ConfigDict(
        # ProviderConfig is constructed both from typed kwargs (tests, env loader)
        # and from raw dicts (YAML). ``use_enum_values=False`` keeps ``type`` as
        # an enum on the instance; ``model_dump`` serializes via ``mode='json'``
        # when we need the wire form.
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    type: ProviderType
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    # Azure-specific
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    # Rate limiting
    max_retries: int = 3
    timeout: float = 60.0
    # Token limits
    max_tokens: int = 4096
    temperature: float = 0.7
    # Responses API dispatch + reasoning controls
    api_mode: ApiMode = "auto"
    reasoning_effort: Optional[ReasoningEffort] = None
    reasoning_summary: Optional[ReasoningSummary] = None

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            return ProviderType(value)
        return value

    @model_validator(mode="after")
    def _validate_provider_combo(self) -> "ProviderConfig":
        if self.type == ProviderType.AZURE and not self.azure_endpoint:
            raise ValueError("Azure provider requires azure_endpoint")
        return self

    def resolved_api_mode(self) -> Literal["chat", "responses"]:
        """Resolve ``api_mode='auto'`` against the active model name.

        gpt-5*, o1*, o3*, o4* are routed to the Responses API; everything else
        falls back to the Chat Completions API. Callers that need to know the
        active mode (e.g. the OpenAI provider) should use this helper instead
        of reading ``api_mode`` directly.
        """
        if self.api_mode == "responses":
            return "responses"
        if self.api_mode == "chat":
            return "chat"
        model = (self.model or "").lower()
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            return "responses"
        return "chat"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create from dictionary, applying env-var fallback for api_key."""
        data = dict(data or {})
        provider_type = data.get("type", "openai")
        if isinstance(provider_type, str):
            provider_type = ProviderType(provider_type)
        data["type"] = provider_type

        # Preserve the historical env-var fallback behavior — pydantic
        # validation can't pull from env vars on its own.
        if not data.get("api_key"):
            data["api_key"] = os.getenv(f"{provider_type.value.upper()}_API_KEY")
        data.setdefault("model", DEFAULT_MODEL)
        data.setdefault("api_version", "2024-02-15-preview")
        return cls.model_validate(data)


# Default module lists (defined outside class for reference in from_dict)
# Allow common modules needed for file operations and data analysis
DEFAULT_ALLOWED_MODULES: List[str] = [
    # Core Python modules
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "json",
    "re",
    "string",
    "textwrap",
    # File system modules (needed for reading files, checking paths)
    "os",
    "sys",
    "pathlib",
    "shutil",
    "glob",
    "fnmatch",
    # Data analysis libraries
    "numpy",
    "pandas",
    "scipy",
    "openpyxl",
    "xlrd",
    "olefile",
    # Other useful modules
    "importlib",
    "platform",
    "io",
    "csv",
    "pickle",
]
# Only block network-related and truly dangerous modules
DEFAULT_BLOCKED_MODULES: List[str] = [
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "telnetlib",
    "asyncio.subprocess",
]


class ExecutionConfig(BaseModel):
    """Code execution configuration."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    mode: ExecutionMode = ExecutionMode.SANDBOX
    timeout: float = 30.0  # seconds
    max_memory_mb: int = 512
    allowed_modules: List[str] = Field(default_factory=lambda: DEFAULT_ALLOWED_MODULES.copy())
    blocked_modules: List[str] = Field(default_factory=lambda: DEFAULT_BLOCKED_MODULES.copy())
    workspace_dir: Optional[Path] = None

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            return ExecutionMode(value)
        return value

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def _coerce_workspace(cls, value: Any) -> Any:
        if value is None or isinstance(value, Path):
            return value
        return Path(value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionConfig":
        return cls.model_validate(data or {})


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    max_conversation_turns: int = 50
    max_working_memory_items: int = 20
    persist_tools: bool = True
    tools_storage_path: Optional[Path] = None

    @field_validator("tools_storage_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Any:
        if value is None or isinstance(value, Path):
            return value
        return Path(value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        return cls.model_validate(data or {})


class WebSearchConfig(BaseModel):
    """Web search configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    backend: str = "tavily"  # "tavily" | "duckduckgo"
    tavily_api_key: Optional[str] = None
    default_max_results: int = 5
    deep_search_max_results: int = 10
    default_language: str = "en"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSearchConfig":
        data = dict(data or {})
        if not data.get("tavily_api_key"):
            data["tavily_api_key"] = os.getenv("TAVILY_API_KEY")
        return cls.model_validate(data)


class Settings(BaseModel):
    """Main settings container for Inception."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    provider: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(type=ProviderType.OPENAI)
    )
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    agent_name: str = "Inception"
    verbose: bool = False
    debug: bool = False
    plugins_dir: Optional[Path] = None

    @field_validator("plugins_dir", mode="before")
    @classmethod
    def _coerce_plugins_dir(cls, value: Any) -> Any:
        if value is None or isinstance(value, Path):
            return value
        return Path(value)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Settings":
        """Load settings from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create from dictionary."""
        data = dict(data or {})
        return cls(
            provider=ProviderConfig.from_dict(data.get("provider", {})),
            execution=ExecutionConfig.from_dict(data.get("execution", {})),
            memory=MemoryConfig.from_dict(data.get("memory", {})),
            web_search=WebSearchConfig.from_dict(data.get("web_search", {})),
            agent_name=data.get("agent_name", "Inception"),
            verbose=data.get("verbose", False),
            debug=data.get("debug", False),
            plugins_dir=data.get("plugins_dir"),
        )

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        provider_type = os.getenv("INCEPTION_PROVIDER", "openai")
        provider_config = ProviderConfig(
            type=ProviderType(provider_type),
            api_key=os.getenv(f"{provider_type.upper()}_API_KEY"),
            model=os.getenv("INCEPTION_MODEL", DEFAULT_MODEL),
            base_url=os.getenv("INCEPTION_BASE_URL"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            api_mode=os.getenv("INCEPTION_API_MODE", "auto"),  # type: ignore[arg-type]
            reasoning_effort=os.getenv("INCEPTION_REASONING_EFFORT") or None,  # type: ignore[arg-type]
            reasoning_summary=os.getenv("INCEPTION_REASONING_SUMMARY") or None,  # type: ignore[arg-type]
        )

        execution_config = ExecutionConfig(
            mode=ExecutionMode(os.getenv("INCEPTION_EXEC_MODE", "sandbox")),
            timeout=float(os.getenv("INCEPTION_EXEC_TIMEOUT", "30")),
        )

        web_search_config = WebSearchConfig(
            enabled=os.getenv("INCEPTION_WEB_SEARCH_ENABLED", "true").lower() == "true",
            backend=os.getenv("INCEPTION_WEB_SEARCH_BACKEND", "tavily"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            default_language=os.getenv("INCEPTION_WEB_SEARCH_LANGUAGE", "en"),
        )

        return cls(
            provider=provider_config,
            execution=execution_config,
            web_search=web_search_config,
            verbose=os.getenv("INCEPTION_VERBOSE", "").lower() == "true",
            debug=os.getenv("INCEPTION_DEBUG", "").lower() == "true",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for YAML serialization).

        The output is intentionally a curated subset of the model — secrets
        (api keys), workspace paths, and unset reasoning fields are excluded so
        the YAML stays clean and shareable.
        """
        provider = {
            "type": self.provider.type.value,
            "model": self.provider.model,
            "base_url": self.provider.base_url,
            "max_retries": self.provider.max_retries,
            "timeout": self.provider.timeout,
            "max_tokens": self.provider.max_tokens,
            "temperature": self.provider.temperature,
            "api_mode": self.provider.api_mode,
        }
        if self.provider.reasoning_effort is not None:
            provider["reasoning_effort"] = self.provider.reasoning_effort
        if self.provider.reasoning_summary is not None:
            provider["reasoning_summary"] = self.provider.reasoning_summary

        return {
            "provider": provider,
            "execution": {
                "mode": self.execution.mode.value,
                "timeout": self.execution.timeout,
                "max_memory_mb": self.execution.max_memory_mb,
                "allowed_modules": self.execution.allowed_modules,
                "blocked_modules": self.execution.blocked_modules,
            },
            "memory": {
                "max_conversation_turns": self.memory.max_conversation_turns,
                "max_working_memory_items": self.memory.max_working_memory_items,
                "persist_tools": self.memory.persist_tools,
            },
            "web_search": {
                "enabled": self.web_search.enabled,
                "backend": self.web_search.backend,
                "default_max_results": self.web_search.default_max_results,
                "deep_search_max_results": self.web_search.deep_search_max_results,
                "default_language": self.web_search.default_language,
            },
            "agent_name": self.agent_name,
            "verbose": self.verbose,
            "debug": self.debug,
        }

    def save_yaml(self, path: Path | str) -> None:
        """Save settings to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
